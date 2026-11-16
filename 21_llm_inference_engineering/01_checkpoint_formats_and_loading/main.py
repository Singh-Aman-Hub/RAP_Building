"""
GOAL:  Understand what is INSIDE a model checkpoint and how it gets into memory: why pickle can run code,
       how a .safetensors file is laid out (parsed by hand), lazy mmap loading vs eager loading (time + RSS),
       sharded checkpoints with an index.json, and the Hugging Face cache layout.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/01_checkpoint_formats_and_loading/main.py
EXPECTED OUTPUT (shortened; timings differ per machine):
    --- 1. pickle can execute code ---
    >>> CODE RAN DURING pickle.loads (this could have been anything) <<<
    torch.load(weights_only=True) refused: UnpicklingError ...
    --- 2. safetensors header, parsed by hand ---
    header length N = 3xx bytes ; tensors: ['embed.weight', 'layer0.w', 'layer0.b'] ; round-trip identical: True
    --- 3. eager vs lazy loading (synthetic 300 MB checkpoint, each mode in a fresh process) ---
    eager load_file          load 0.6 s | RSS 30 -> 291 MB (+261) | peak 628 MB
    safe_open, read 1 tensor load 0.0 s | RSS 29 -> 79 MB (+50)
    --- 4. sharded checkpoint --- ... loaded by shard: True
    --- 5. real models: distilgpt2 / gpt2 via from_pretrained ... meta device: full param count, no weight data
    --- 6. HF cache layout --- snapshots/<commit>/model.safetensors -> ../../blobs/<sha256>
NEEDS: numpy, torch, safetensors, transformers (gpt2 + distilgpt2 cached; no downloads).  ~30-60 s on CPU.
       Writes a ~300 MB temp folder inside this step and deletes it at the end (SIZE_MB env var to change).
"""

import io
import json
import mmap
import os
import pickle
import pickletools
import re
import resource
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIZE_MB = int(os.environ.get("SIZE_MB", 300))        # size of the synthetic checkpoint


def peak_rss_mb():
    """High-water mark of resident memory of THIS process. macOS reports bytes, Linux reports kilobytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3


def current_rss_mb():
    """RSS of THIS process right now (not the peak), via `ps`: works on macOS and Linux."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1e3                     # ps reports kilobytes


# ======================================================================================================
# CHILD MODE: measure one loading strategy in a FRESH process.  ru_maxrss never goes down inside a process,
# so to compare strategies fairly each one must run in its own process. The parent calls us with --child.
# ======================================================================================================
def child(mode, arg):
    if mode.startswith("hf_"):                        # import the heavy libraries BEFORE the clock starts:
        import torch                                  # we want to time the weight loading, not "import torch"
        from transformers import AutoConfig, AutoModelForCausalLM
    base = current_rss_mb()                           # memory used by the interpreter + imports (before loading)
    t0 = time.perf_counter()
    extra = ""
    if mode == "eager":                               # read EVERY tensor into RAM
        from safetensors.numpy import load_file
        sd = load_file(arg)
        extra = f"{len(sd)} tensors, {sum(v.nbytes for v in sd.values()) / 1e6:.0f} MB materialised"
    elif mode == "lazy1":                             # open (mmap) the file, read ONE tensor
        from safetensors import safe_open
        with safe_open(arg, framework="numpy") as f:
            names = list(f.keys())                    # header only: nothing is read yet
            first = f.get_tensor(names[0])            # only now are those bytes copied out
        extra = f"{len(names)} tensors visible, 1 read ({first.nbytes / 1e6:.0f} MB)"
    elif mode == "memmap":                            # raw numpy memmap of the whole file, view one tensor
        with open(arg, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        mm = np.memmap(arg, dtype=np.uint8, mode="r")
        name = [k for k in header if k != "__metadata__"][0]
        s, e = header[name]["data_offsets"]
        view = mm[8 + n + s: 8 + n + e].view(np.float32)   # zero-copy slice of the mapped file
        extra = f"view of {name}: {view.nbytes / 1e6:.0f} MB addressable, sum of first 1000 = {float(view[:1000].sum()):.3f}"
    elif mode.startswith("hf_"):                      # real model through transformers
        if mode == "hf_default":
            m = AutoModelForCausalLM.from_pretrained(arg)
        elif mode == "hf_bf16":
            m = AutoModelForCausalLM.from_pretrained(arg, dtype=torch.bfloat16)
        elif mode == "hf_devicemap":
            m = AutoModelForCausalLM.from_pretrained(arg, device_map="cpu")   # goes through the 'accelerate' loading path
        elif mode == "hf_meta":                       # build the architecture on the 'meta' device: shapes only, no storage
            cfg = AutoConfig.from_pretrained(arg)
            with torch.device("meta"):
                m = AutoModelForCausalLM.from_config(cfg)
        p = next(m.parameters())
        n_par = sum(x.numel() for x in m.parameters())
        extra = f"{n_par / 1e6:.0f}M params, dtype {p.dtype}, device {p.device}"
    dt = time.perf_counter() - t0
    print(json.dumps({"mode": mode, "seconds": dt, "base_mb": base, "after_mb": current_rss_mb(), "peak_mb": peak_rss_mb(), "extra": extra}))


def run_child(mode, arg):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    out = subprocess.run([sys.executable, __file__, "--child", mode, arg], capture_output=True, text=True, env=env)
    last = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not last:
        return {"mode": mode, "seconds": float("nan"), "base_mb": 0, "after_mb": 0, "peak_mb": 0, "extra": "FAILED: " + out.stderr[-200:]}
    return json.loads(last[-1])


def show(r, label):
    print(f"{label:<26} load {r['seconds']:6.2f} s | RSS {r['base_mb']:5.0f} -> {r['after_mb']:5.0f} MB (+{r['after_mb'] - r['base_mb']:4.0f}) | peak {r['peak_mb']:5.0f} MB | {r['extra']}")


# ======================================================================================================
# 1. PICKLE CAN EXECUTE CODE
# ======================================================================================================
class Harmless:
    """A pickle stores 'how to rebuild me'. __reduce__ can return ANY callable + args; loading CALLS it.
    Here the callable is print (harmless). An attacker would put os.system or subprocess.run instead."""
    def __reduce__(self):
        return (print, (">>> CODE RAN DURING pickle.loads (this could have been anything) <<<",))


def section_pickle():
    print("--- 1. pickle can execute code ---")
    blob = pickle.dumps(Harmless())                   # bytes; contains the instruction 'call builtins.print(...)'
    print(f"pickle bytes: {len(blob)}; disassembly of the opcodes that matter:")
    buf = io.StringIO()
    pickletools.dis(blob, out=buf)                    # human-readable opcode listing
    for line in buf.getvalue().splitlines():
        if any(k in line for k in ("GLOBAL", "REDUCE", "STACK_GLOBAL", "SHORT_BINUNICODE")):
            print("   ", line.strip()[:110])
    print("calling pickle.loads(blob) ...")
    pickle.loads(blob)                                # <-- prints the message: loading = running code
    # modern torch: torch.load defaults to weights_only=True (since torch 2.6) and refuses arbitrary callables
    import torch
    path = HERE / "_evil.pt"
    torch.save(Harmless(), path)
    try:
        torch.load(path)                              # default weights_only=True in torch >= 2.6
        print("torch.load: loaded?! (older torch defaulted to weights_only=False)")
    except Exception as e:                            # pickle.UnpicklingError in practice
        msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).splitlines()[0]      # strip terminal colour codes
        print(f"torch.load(default) refused: {type(e).__name__}: {msg[:60]}...")
    path.unlink()
    print("=> pickle format = code; safetensors format = numbers only.\n")


# ======================================================================================================
# 2. SAFETENSORS FILE LAYOUT, PARSED BY HAND
# ======================================================================================================
def section_safetensors(tmp):
    print("--- 2. safetensors header, parsed by hand ---")
    from safetensors.numpy import save_file
    rng = np.random.default_rng(0)
    tensors = {
        "embed.weight": rng.standard_normal((16, 8)).astype(np.float32),
        "layer0.w": rng.standard_normal((8, 8)).astype(np.float16),
        "layer0.b": np.arange(8, dtype=np.int64),
    }
    path = tmp / "tiny.safetensors"
    save_file(tensors, str(path), metadata={"format": "np", "note": "hand-parsed demo"})

    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])               # 8 bytes, little-endian unsigned 64-bit = header length
    header = json.loads(raw[8:8 + n])                 # next n bytes: JSON text
    data_start = 8 + n                                # everything after is raw tensor bytes
    print(f"file {len(raw)} bytes = 8 (length) + {n} (JSON header) + {len(raw) - data_start} (raw data)")
    print("__metadata__:", header.pop("__metadata__"))
    DT = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "BF16": None}   # BF16 has no numpy dtype
    ok = True
    for name, info in header.items():
        s, e = info["data_offsets"]                   # offsets are relative to data_start
        arr = np.frombuffer(raw, dtype=DT[info["dtype"]], count=(e - s) // np.dtype(DT[info["dtype"]]).itemsize, offset=data_start + s)
        arr = arr.reshape(info["shape"])
        same = np.array_equal(arr, tensors[name])
        ok &= same
        print(f"  {name:<14} dtype {info['dtype']:<4} shape {str(info['shape']):<9} bytes[{s}:{e}]  equal to original: {same}")
    print(f"round-trip identical: {ok}   (no pickle, no code: just json + np.frombuffer)\n")


# ======================================================================================================
# 3. EAGER vs LAZY (mmap) LOADING  --  each strategy in a fresh process
# ======================================================================================================
def write_synthetic(tmp):
    from safetensors.numpy import save_file
    n_tensors = 6
    per = SIZE_MB * 1_000_000 // n_tensors // 4       # float32 elements per tensor
    side = int(per ** 0.5)
    rng = np.random.default_rng(1)
    sd = {f"block{i}.weight": rng.standard_normal((side, side), dtype=np.float32) for i in range(n_tensors)}
    path = tmp / "synthetic.safetensors"
    save_file(sd, str(path))
    return path


def section_loading(tmp):
    print(f"--- 3. eager vs lazy loading (synthetic ~{SIZE_MB} MB checkpoint, each mode in a fresh process) ---")
    path = write_synthetic(tmp)
    print(f"wrote {path.name}: {path.stat().st_size / 1e6:.0f} MB (file is in the OS page cache now, so these are WARM-cache numbers)")
    for mode, label in (("eager", "eager load_file"), ("lazy1", "safe_open, read 1 tensor"), ("memmap", "np.memmap, view 1 tensor")):
        show(run_child(mode, str(path)), label)
    print("=> eager copies everything (peak RSS ~ file size + overhead). Lazy mmap touches only the pages you read.\n")


# ======================================================================================================
# 4. SHARDED CHECKPOINTS
# ======================================================================================================
def section_sharded(tmp):
    print("--- 4. sharded checkpoint: model-0000X-of-0000N.safetensors + model.safetensors.index.json ---")
    from safetensors import safe_open
    from safetensors.numpy import save_file
    rng = np.random.default_rng(2)
    sd = {f"model.layers.{i}.mlp.w": rng.standard_normal((64, 64)).astype(np.float32) for i in range(6)}
    sd["model.embed_tokens.weight"] = rng.standard_normal((100, 64)).astype(np.float32)
    names = sorted(sd)
    N = 3
    chunks = [names[i::N] for i in range(N)]          # round-robin split of tensor names into N shards
    shard_dir = tmp / "sharded"
    shard_dir.mkdir()
    weight_map, total = {}, 0
    for i, chunk in enumerate(chunks, 1):
        fname = f"model-{i:05d}-of-{N:05d}.safetensors"
        save_file({k: sd[k] for k in chunk}, str(shard_dir / fname), metadata={"format": "pt"})
        weight_map.update({k: fname for k in chunk})
        total += sum(sd[k].nbytes for k in chunk)
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (shard_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print("files:", sorted(p.name for p in shard_dir.iterdir()))
    print("index.json total_size:", total, "| first entries:", dict(list(weight_map.items())[:3]))

    # load ONE tensor without reading other shards: look up its shard in weight_map, open only that file
    idx = json.loads((shard_dir / "model.safetensors.index.json").read_text())
    want = "model.layers.4.mlp.w"
    with safe_open(str(shard_dir / idx["weight_map"][want]), framework="numpy") as f:
        t = f.get_tensor(want)
    print(f"'{want}' lives in {idx['weight_map'][want]}; loaded alone, equal to original: {np.array_equal(t, sd[want])}")
    # load everything shard by shard (this is what from_pretrained does: one shard in RAM at a time)
    merged = {}
    for fname in sorted(set(idx["weight_map"].values())):
        with safe_open(str(shard_dir / fname), framework="numpy") as f:
            for k in f.keys():
                merged[k] = f.get_tensor(k)
    print(f"loaded by shard: {all(np.array_equal(merged[k], sd[k]) for k in sd)} ({len(merged)} tensors from {N} shards)\n")


# ======================================================================================================
# 5. REAL MODELS THROUGH from_pretrained
# ======================================================================================================
def section_real_models():
    print("--- 5. real models: distilgpt2 and gpt2 via from_pretrained (fresh process each) ---")
    for name in ("distilgpt2", "gpt2"):
        for mode, label in (("hf_default", f"{name} default"), ("hf_bf16", f"{name} dtype=bf16"),
                            ("hf_devicemap", f"{name} device_map=cpu"), ("hf_meta", f"{name} meta device")):
            if name == "gpt2" and mode in ("hf_bf16", "hf_devicemap"):
                continue                               # keep the run short
            show(run_child(mode, name), label)
    print("=> 'meta' = shapes only, ~0 extra RAM, instant: used to plan memory and to build the skeleton before filling it.\n")


# ======================================================================================================
# 6. HF CACHE LAYOUT
# ======================================================================================================
def section_cache():
    print("--- 6. Hugging Face cache layout (models--gpt2) ---")
    print("HF_HOME =", os.environ.get("HF_HOME", "(unset -> ~/.cache/huggingface)"),
          "| HF_HUB_CACHE =", os.environ.get("HF_HUB_CACHE", "(unset -> $HF_HOME/hub)"),
          "| HF_HUB_OFFLINE =", os.environ.get("HF_HUB_OFFLINE", "(unset)"))
    root = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")) / "models--gpt2"
    if not root.exists():
        print("gpt2 is not in the local cache: skipping (this step never downloads)")
        return
    print("refs/main ->", (root / "refs" / "main").read_text().strip())
    blobs = sorted((root / "blobs").iterdir(), key=lambda p: -p.stat().st_size)
    print(f"blobs/ ({len(blobs)} files, content-addressed: the filename is the sha256 or git-sha of the content):")
    for b in blobs[:3]:
        print(f"   {b.name[:16]}...  {b.stat().st_size:>12,} bytes")
    for snap in (root / "snapshots").iterdir():
        print(f"snapshots/{snap.name[:8]}...  (one folder per commit; entries are symlinks):")
        for f in sorted(snap.iterdir()):
            print(f"   {f.name:<22} -> {os.readlink(f)[:34]}...   is_symlink={f.is_symlink()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
        sys.exit(0)
    tmp = Path(tempfile.mkdtemp(prefix="_scratch_", dir=HERE))     # temp folder under this step, removed at the end
    try:
        section_pickle()
        section_safetensors(tmp)
        section_loading(tmp)
        section_sharded(tmp)
        section_real_models()
        section_cache()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)         # always clean the big temp file
    print("\ncleanup done: temp folder removed")

# TRY IT: SIZE_MB=800 python main.py : eager peak RSS grows with the file; the lazy modes barely move.
# TRY IT: add a 4th tensor to the section 2 dict (dtype np.float64 -> "F64") and re-read the header by hand.
# TRY IT: HF_HOME=/tmp/empty_hf HF_HUB_OFFLINE=1 python main.py : section 5 children fail fast (offline) instead of downloading.
# BREAK IT: in section 2 read the offsets from the start of the file instead of data_start and compare the numbers.
# BREAK IT: change "<Q" to ">Q" (big-endian) and watch the header length become absurdly large.
