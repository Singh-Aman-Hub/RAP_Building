# Step 01 — Checkpoint formats and loading: pickle, safetensors, mmap, shards, the HF cache

## The story so far
This module starts where [10/03 Serialization](../../10_ml_system_architecture/03_serialization/WALKTHROUGH.md) stopped. There you learned the rule of thumb: joblib/pickle only for files you made, data-only formats such as safetensors for sharing weights. Here we go one level deeper, because every inference topic that follows begins with the same act: getting billions of numbers from a file into memory, quickly and safely.

## The pain point
A 7B model is 14 GB in fp16; a 70B model is 140 GB. Loading it the naive way (read the whole file into RAM, then copy it again into the model) needs twice the model size in memory and minutes of waiting. And if the file is a pickle, "loading a model" can also mean "running a stranger's code".

## The spark
Make the file format boring and honest: a JSON table of contents followed by raw bytes (safetensors). Then everything else falls out: you can validate it without running anything, map it lazily into memory (mmap), split it into shards with an index, and let the library place each tensor straight onto its target device.

## Goal
Know what is inside a checkpoint, why pickle is dangerous and safetensors is not, how lazy loading differs from eager loading in time and peak RAM (measured), how sharded checkpoints and the Hugging Face cache are laid out, and which `from_pretrained` options change memory.

## The theory (read this before the interview)

### Pickle and `torch.load`
Pickle is a tiny stack-machine program. The opcode `REDUCE` means "call this callable with these arguments", and an object's `__reduce__` method decides which callable is written. So **unpickling = executing what the file says**. `torch.save` writes a zip archive that contains a pickle (the structure) plus raw tensor storages. `torch.load(path, weights_only=True)` uses a restricted unpickler that only rebuilds tensors and plain containers and refuses everything else; this became the default in torch 2.6, so old tutorials that say "torch.load is always unsafe" or that rely on `weights_only=False` are out of date. The old `.bin`/`.pt` checkpoints remain a risk when `weights_only=False` or a library unpickles them for you.

### safetensors layout
```
bytes 0..7         N  = header length, little-endian uint64
bytes 8..8+N-1     JSON header:  {"name": {"dtype": "F32", "shape": [16, 8], "data_offsets": [begin, end]}, ...,
                                  "__metadata__": {"format": "pt"}}
bytes 8+N ...      one raw byte buffer; tensor = bytes[begin:end] of this buffer (offsets relative to its start)
```
- dtypes are strings (`F32`, `F16`, `BF16`, `I64`, `U8`, `BOOL`, ... newer versions also have fp8 types), shapes are lists, data is row-major little-endian.
- No code runs on load; the header can be validated (offsets must be contiguous, no overlaps, sizes match shape x dtype) before touching the data.
- Zero-copy and **lazy**: `safe_open` memory-maps the file; you can read one tensor or a slice without reading the rest.

### mmap, page cache, RSS (why lazy loading is cheap)
`mmap` maps a file into the process address space. Nothing is read until a page is touched; then the OS reads that page, keeps it in the **page cache** and counts it in the process's resident set (RSS). Consequences:
- Opening a huge file is almost free; **peak RSS follows what you touch**, not the file size.
- A second process that maps the same file shares the same cached pages.
- The first read of an uncached file is **cold** (disk speed); later reads are **warm** (RAM speed). Always say which one you measured.
- Honest caveat: a model that is going to run needs all of its weights eventually. Lazy loading mostly helps startup latency, memory sharing, and copying tensors straight to the GPU without an extra CPU copy; it does not make a 14 GB model fit in 4 GB of RAM.

### Sharded checkpoints
Big models ship as `model-00001-of-00004.safetensors` ... plus `model.safetensors.index.json`:
```json
{"metadata": {"total_size": 123904}, "weight_map": {"model.layers.4.mlp.w": "model-00003-of-00003.safetensors", ...}}
```
A loader reads the index, then loads shard by shard (peak RAM ~ one shard plus the model) or fetches only the shard it needs.

### Hugging Face cache layout
```
~/.cache/huggingface/hub/models--gpt2/
    refs/main                      text file: the commit hash of branch "main"
    blobs/<sha>                    actual files, named by content hash (stored once)
    snapshots/<commit>/model.safetensors -> ../../blobs/<sha>      symlinks; one folder per revision
```
`HF_HOME` moves everything, `HF_HUB_CACHE` moves just the hub cache, `HF_HUB_OFFLINE=1` forbids network use (so a missing file fails immediately instead of downloading). Two revisions that share a file share one blob.

### `from_pretrained` knobs that change memory
- **dtype** (`dtype=torch.bfloat16`; older releases call it `torch_dtype`): converts while loading. Always pass it explicitly or print `model.dtype`, because defaults have changed between versions.
- **meta device**: tensors with shape and dtype but no storage. `with torch.device("meta"): model = AutoModelForCausalLM.from_config(cfg)` builds the architecture at ~zero memory; this is how a skeleton is created before real weights are poured in.
- **`low_cpu_mem_usage` / `device_map`**: build on meta, then place each loaded tensor directly on its device instead of first allocating a randomly initialised full copy. `device_map="auto"` (needs the `accelerate` package) also spreads layers across available GPUs, CPU RAM and disk. Recent transformers versions already load this way most of the time, so the flag matters mostly in older code; the idea is what to remember.
- The same checkpoint can be re-saved with `save_pretrained(..., max_shard_size="2GB")` to control shard size.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Checkpoint | weights file, model file, state dict on disk | Mostly | The serialised parameters (and sometimes optimizer state). Inference needs only the weights. |
| state_dict | parameter dictionary | Yes | `{name: tensor}`; the thing that safetensors stores. |
| safetensors | `.safetensors`, "safe tensors" | Yes | JSON header + raw bytes; no code execution. |
| pickle / `.bin` / `.pt` / `.pth` | torch checkpoint | Related | Python objects serialised by pickle; can run code on load. |
| mmap | memory-mapped file, lazy load | Yes | The OS maps the file; pages are read when touched. |
| Eager loading | full load | Yes | Read all bytes into RAM up front. |
| Shard | split file, `model-0000X-of-0000N` | Yes | One piece of a checkpoint; the index maps tensor name to shard. |
| meta device | "empty weights", `init_empty_weights` | Related | Tensors without storage, to plan or build the skeleton. |
| `low_cpu_mem_usage` / `device_map` | accelerate loading, big-model inference | Related | Place each tensor directly on its device; avoid a second full copy. |
| `HF_HOME` / `HF_HUB_CACHE` / `TRANSFORMERS_CACHE` | cache folders | Related | HF_HOME is the root; HF_HUB_CACHE the hub cache; the older TRANSFORMERS_CACHE variable is legacy. |

## Run it
```bash
cd 21_llm_inference_engineering/01_checkpoint_formats_and_loading
HF_HUB_OFFLINE=1 python main.py                 # a few minutes on a busy machine, about one minute on a quiet one
SIZE_MB=800 HF_HUB_OFFLINE=1 python main.py     # bigger synthetic file
# from repo root (use the global heavy lock when other jobs run):
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/01_checkpoint_formats_and_loading/main.py
```
Needs gpt2 and distilgpt2 in the local cache (no downloads). It writes a temporary folder inside the step and deletes it at the end.

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–22)

```python
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
```

- **In simple words:** What the step shows and the shortened expected output.
- **Exactly:** Docstring only. It says that the step writes a temporary ~300 MB file inside its own folder and deletes it at the end.
- **Why it is here:** Tells you the shape of a correct result and warns about the temporary disk use.

### 2. Imports and constants  (`main.py` lines 24–43)

```python
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
```

- **In simple words:** Standard-library tools plus numpy; `SIZE_MB` sets how big the synthetic checkpoint is.
- **Exactly:** `resource` gives memory statistics, `subprocess` starts child processes, `pickletools` disassembles pickles, `struct` and `json` parse the safetensors header. `HERE` is this file's folder.
- **Why it is here:** Everything in sections 1-4 and 6 needs only the standard library, numpy and torch/safetensors.

### 3. Peak memory helper  (`main.py` lines 46–49)

```python
def peak_rss_mb():
    """High-water mark of resident memory of THIS process. macOS reports bytes, Linux reports kilobytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3
```

- **In simple words:** Return the most RAM this process has used, in MB.
- **Exactly:** `resource.getrusage(RUSAGE_SELF).ru_maxrss` is the high-water mark of resident memory. macOS reports bytes, Linux reports kilobytes, hence the division.
- **Why it is here:** It only ever goes up inside a process, which dictates the next design decision.

### 4. Current RSS helper  (`main.py` lines 52–55)

```python
def current_rss_mb():
    """RSS of THIS process right now (not the peak), via `ps`: works on macOS and Linux."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return int(out.strip()) / 1e3                     # ps reports kilobytes
```

- **In simple words:** Return the RAM this process holds right now.
- **Exactly:** Runs `ps -o rss= -p <pid>` (kilobytes) because the peak from `ru_maxrss` is dominated by imports; the difference between current RSS before and after a load shows what the load itself kept.
- **Why it is here:** Peak and current RSS answer different questions; the table prints both.

### 5. Child mode: setup  (`main.py` lines 58–68)

```python
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
```

- **In simple words:** Measure one loading strategy inside a fresh process.
- **Exactly:** Because `ru_maxrss` never decreases, comparing strategies in one process would be meaningless. The parent starts `python main.py --child <mode> <path>`; the child imports torch/transformers first (so import time is not counted as load time), records current `base` RSS, then starts the clock.
- **Why it is here:** Fair measurement: one strategy, one process, imports excluded.

### 6. Child mode: eager, lazy and memmap loading  (`main.py` lines 69–87)

```python
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
```

- **In simple words:** Three ways to get tensors out of a safetensors file.
- **Exactly:** `load_file` reads every tensor into RAM (copying). `safe_open` mmaps the file and lets you list keys (header only) and `get_tensor` one tensor. The raw `np.memmap` path parses the header by hand and takes a zero-copy view of one tensor inside the mapped bytes: nothing is read until the values are touched.
- **Why it is here:** Shows the cost difference between materialising everything and mapping lazily.

### 7. Child mode: real models through transformers  (`main.py` lines 88–103)

```python
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
```

- **In simple words:** Load distilgpt2 or gpt2 four different ways and print time and memory.
- **Exactly:** `from_pretrained(name)` loads fp32 by default; `dtype=torch.bfloat16` converts while loading; `device_map="cpu"` goes through the accelerate-based loader (builds on the meta device, then fills tensors); `torch.device("meta")` + `from_config` builds only the architecture: tensors have shapes and dtypes but no storage. The last lines print one JSON line for the parent.
- **Why it is here:** These are the knobs behind `low_cpu_mem_usage`, `device_map="auto"` and "load in half precision".

### 8. Run a child and show its result  (`main.py` lines 106–116)

```python
def run_child(mode, arg):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    out = subprocess.run([sys.executable, __file__, "--child", mode, arg], capture_output=True, text=True, env=env)
    last = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not last:
        return {"mode": mode, "seconds": float("nan"), "base_mb": 0, "after_mb": 0, "peak_mb": 0, "extra": "FAILED: " + out.stderr[-200:]}
    return json.loads(last[-1])


def show(r, label):
    print(f"{label:<26} load {r['seconds']:6.2f} s | RSS {r['base_mb']:5.0f} -> {r['after_mb']:5.0f} MB (+{r['after_mb'] - r['base_mb']:4.0f}) | peak {r['peak_mb']:5.0f} MB | {r['extra']}")
```

- **In simple words:** The parent starts the child, takes its JSON line and prints a formatted row.
- **Exactly:** `subprocess.run([...], capture_output=True, text=True, env=...)`; `HF_HUB_OFFLINE=1` is forced so nothing can download. If the child fails the row shows the tail of stderr.
- **Why it is here:** Isolates measurements from each other and from the parent's own memory.

### 9. The pickle trap: a harmless class  (`main.py` lines 119–126)

```python
# ======================================================================================================
# 1. PICKLE CAN EXECUTE CODE
# ======================================================================================================
class Harmless:
    """A pickle stores 'how to rebuild me'. __reduce__ can return ANY callable + args; loading CALLS it.
    Here the callable is print (harmless). An attacker would put os.system or subprocess.run instead."""
    def __reduce__(self):
        return (print, (">>> CODE RAN DURING pickle.loads (this could have been anything) <<<",))
```

- **In simple words:** A class whose pickle runs `print` when loaded.
- **Exactly:** `__reduce__` returns `(callable, args)`; pickle stores "call `builtins.print` with this argument". An attacker would return `os.system` with a shell command.
- **Why it is here:** This is the entire attack: unpickling calls whatever the file says.

### 10. Section 1: pickle and torch.load  (`main.py` lines 129–151)

```python
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
```

- **In simple words:** Dump the harmless object, look at the opcodes, load it, then see what modern torch does.
- **Exactly:** `pickletools.dis` lists the opcodes: `STACK_GLOBAL` (find `builtins.print`) and `REDUCE` (call it). `pickle.loads(blob)` prints the message: loading = running code. `torch.save(obj)` then `torch.load(path)` fails under the default `weights_only=True` (default since torch 2.6) with an `UnpicklingError`; the regex only strips terminal colour codes from the message.
- **Why it is here:** Shows both the danger and why modern torch is safer by default; safetensors removes the problem completely.

### 11. Section 2: safetensors header by hand  (`main.py` lines 154–184)

```python
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
```

- **In simple words:** Write a three-tensor file, then decode it with only `struct`, `json` and `np.frombuffer`.
- **Exactly:** `struct.unpack("<Q", raw[:8])` = header length N (little-endian uint64). `json.loads(raw[8:8+N])` = `{name: {dtype, shape, data_offsets}, "__metadata__": {...}}`. Data starts at `8+N`; `data_offsets` are relative to that. `np.frombuffer(raw, dtype, count, offset)` makes an array view over the bytes, then `reshape`. Tensor order in the file is decided by the writer (sorted by dtype then name), not by insertion order.
- **Why it is here:** The whole format fits in your head: length, JSON, bytes. That is why it is safe, fast and easy to implement in any language.

### 12. Section 3: eager versus lazy loading  (`main.py` lines 187–208)

```python
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
```

- **In simple words:** Write a ~300 MB synthetic checkpoint and compare three loaders, each in a fresh process.
- **Exactly:** `write_synthetic` makes six float32 matrices of about 50 MB each with `save_file`. `section_loading` then runs the `eager`, `lazy1` and `memmap` children. The file was just written, so it sits in the OS page cache: these are warm-cache numbers (a cold first read from disk is slower).
- **Why it is here:** This is the "mmap loading" idea, measured: eager peak RSS follows the file size, lazy follows what you touch.

### 13. Section 4: sharded checkpoints  (`main.py` lines 211–249)

```python
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
```

- **In simple words:** Split tensors across three files and an index, then load one tensor and then everything by shard.
- **Exactly:** Names are dealt round-robin into N shards `model-0000i-of-0000N.safetensors`. `model.safetensors.index.json` has `metadata.total_size` and `weight_map` (tensor name -> shard file). To fetch one tensor: look up its shard, `safe_open` only that file. To load all: iterate shards one at a time (what `from_pretrained` does, so peak RAM is about one shard).
- **Why it is here:** This is how 7B-405B models ship: no single file bigger than a few GB, and tools can fetch only what they need.

### 14. Section 5: real models through from_pretrained  (`main.py` lines 252–263)

```python
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
```

- **In simple words:** Load distilgpt2 (82M) and gpt2 (124M) with different options and compare.
- **Exactly:** Each combination runs in a fresh child. Not every combination is run for gpt2, to keep the step short. Note the `meta` row: the model has the right number of parameters but no data.
- **Why it is here:** Connects the file-format ideas to the actual calls people use.

### 15. Section 6: the Hugging Face cache layout  (`main.py` lines 266–286)

```python
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
```

- **In simple words:** Inspect `~/.cache/huggingface/hub/models--gpt2` with plain pathlib.
- **Exactly:** `refs/main` is a text file with the commit hash; `blobs/` holds content-addressed files; `snapshots/<commit>/` holds symlinks (`os.readlink`) into `blobs/`. `HF_HOME`, `HF_HUB_CACHE` and `HF_HUB_OFFLINE` change where it is and whether the network may be used. Nothing is downloaded if the model is missing: the section just skips.
- **Why it is here:** When a load is slow, "wrong" or fails offline, this is the folder to look at.

### 16. Entry point and cleanup  (`main.py` lines 289–303)

```python
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
```

- **In simple words:** Create a temp folder, run all sections, always delete the temp folder.
- **Exactly:** `tempfile.mkdtemp(dir=HERE)` creates the scratch folder inside the step; `try/finally` guarantees `shutil.rmtree` even if a section crashes. `--child` short-circuits to the measurement mode.
- **Why it is here:** The big temporary file never survives the run.

### 17. TRY IT / BREAK IT  (`main.py` lines 305–309)

```python
# TRY IT: SIZE_MB=800 python main.py : eager peak RSS grows with the file; the lazy modes barely move.
# TRY IT: add a 4th tensor to the section 2 dict (dtype np.float64 -> "F64") and re-read the header by hand.
# TRY IT: HF_HOME=/tmp/empty_hf HF_HUB_OFFLINE=1 python main.py : section 5 children fail fast (offline) instead of downloading.
# BREAK IT: in section 2 read the offsets from the start of the file instead of data_start and compare the numbers.
# BREAK IT: change "<Q" to ">Q" (big-endian) and watch the header length become absurdly large.
```

- **In simple words:** Experiments in comments.
- **Exactly:** Reading offsets from the start of the file instead of `data_start` gives wrong numbers; using big-endian `">Q"` makes the header length huge and decoding the header fails (`UnicodeDecodeError`).
- **Why it is here:** Practice reading a binary format.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `struct.unpack("<Q", b)` | `<` little-endian, `Q` unsigned 64-bit; returns a tuple | `(n,) = struct.unpack("<Q", raw[:8])` |
| `np.frombuffer(buf, dtype, count, offset)` | typed view over existing bytes, no copy | `np.frombuffer(raw, np.float32, count=128, offset=264)` |
| `np.memmap(path, dtype, mode="r")` | array backed by a mapped file | `mm[a:b].view(np.float32)` |
| `with safe_open(p, framework="numpy") as f:` | context manager that opens (maps) and closes the file | `f.keys()`, `f.get_tensor(k)` |
| `subprocess.run([...], capture_output=True, text=True)` | run a child process and capture its output | child mode |
| `try: ... finally:` | cleanup runs even on error | `shutil.rmtree(tmp)` |
| `with torch.device("meta"):` | tensors created inside have no storage | `from_config(cfg)` |
| `os.readlink(path)` | where a symlink points | cache snapshots |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Internally |
|---|---|---|---|
| `pickle.dumps/loads` | built-in | serialise / rebuild Python objects | writes/executes a small opcode program; `REDUCE` calls a callable |
| `pickletools.dis` | built-in | print opcodes | pure-Python disassembler |
| `torch.save/load` | torch | save tensors as zip + pickle | `load` uses a restricted unpickler when `weights_only=True` |
| `safetensors.numpy.save_file/load_file` | safetensors | write / read a whole file | builds the JSON header, writes raw buffers; `load_file` copies everything into RAM |
| `safetensors.safe_open` | safetensors | lazy reader | maps the file, parses only the header, copies a tensor on `get_tensor` |
| `resource.getrusage` | built-in | process resource usage | `ru_maxrss` = peak resident set size |
| `AutoModelForCausalLM.from_pretrained` | transformers | build the model and load weights | resolves files in the HF cache, reads the (sharded) safetensors, converts dtype, moves to device |
| `AutoModelForCausalLM.from_config` | transformers | build the architecture only | random init (or meta tensors under `torch.device("meta")`) |

## Execution trace (real run)
```
--- 1. pickle can execute code ---
pickle bytes: 108; disassembly of the opcodes that matter:
    11: \x8c SHORT_BINUNICODE 'builtins'
    22: \x8c SHORT_BINUNICODE 'print'
    30: \x93 STACK_GLOBAL
    32: \x8c SHORT_BINUNICODE '>>> CODE RAN DURING pickle.loads (this could have been anything) <<<'
    105: R    REDUCE
calling pickle.loads(blob) ...
>>> CODE RAN DURING pickle.loads (this could have been anything) <<<
torch.load(default) refused: UnpicklingError: Weights only load failed. This file can still be loaded, to ...
=> pickle format = code; safetensors format = numbers only.

--- 2. safetensors header, parsed by hand ---
file 968 bytes = 8 (length) + 256 (JSON header) + 704 (raw data)
__metadata__: {'format': 'np', 'note': 'hand-parsed demo'}
  layer0.b       dtype I64  shape [8]       bytes[0:64]  equal to original: True
  embed.weight   dtype F32  shape [16, 8]   bytes[64:576]  equal to original: True
  layer0.w       dtype F16  shape [8, 8]    bytes[576:704]  equal to original: True
round-trip identical: True   (no pickle, no code: just json + np.frombuffer)

--- 3. eager vs lazy loading (synthetic ~300 MB checkpoint, each mode in a fresh process) ---
wrote synthetic.safetensors: 300 MB (file is in the OS page cache now, so these are WARM-cache numbers)
eager load_file            load   0.56 s | RSS    30 ->   291 MB (+ 261) | peak   628 MB | 6 tensors, 300 MB materialised
safe_open, read 1 tensor   load   0.01 s | RSS    29 ->    79 MB (+  50) | peak   131 MB | 6 tensors visible, 1 read (50 MB)
np.memmap, view 1 tensor   load   0.01 s | RSS    29 ->    29 MB (+   0) | peak    30 MB | view of block0.weight: 50 MB addressable, sum of first 1000 = -18.163
=> eager copies everything (peak RSS ~ file size + overhead). Lazy mmap touches only the pages you read.

--- 4. sharded checkpoint: model-0000X-of-0000N.safetensors + model.safetensors.index.json ---
files: ['model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors', 'model-00003-of-00003.safetensors', 'model.safetensors.index.json']
index.json total_size: 123904 | first entries: {'model.embed_tokens.weight': 'model-00001-of-00003.safetensors', 'model.layers.2.mlp.w': 'model-00001-of-00003.safetensors', 'model.layers.5.mlp.w': 'model-00001-of-00003.safetensors'}
'model.layers.4.mlp.w' lives in model-00003-of-00003.safetensors; loaded alone, equal to original: True
loaded by shard: True (7 tensors from 3 shards)

--- 5. real models: distilgpt2 and gpt2 via from_pretrained (fresh process each) ---
distilgpt2 default         load   0.24 s | RSS   408 ->   429 MB (+  21) | peak   440 MB | 82M params, dtype torch.float32, device cpu
distilgpt2 dtype=bf16      load   0.84 s | RSS   449 ->   503 MB (+  54) | peak   847 MB | 82M params, dtype torch.bfloat16, device cpu
distilgpt2 device_map=cpu  load   0.29 s | RSS   449 ->   469 MB (+  20) | peak   481 MB | 82M params, dtype torch.float32, device cpu
distilgpt2 meta device     load   0.24 s | RSS   449 ->   466 MB (+  17) | peak   478 MB | 82M params, dtype torch.float32, device meta
gpt2 default               load   0.30 s | RSS   420 ->   443 MB (+  22) | peak   453 MB | 124M params, dtype torch.float32, device cpu
gpt2 meta device           load   0.16 s | RSS   425 ->   442 MB (+  17) | peak   453 MB | 124M params, dtype torch.float32, device meta
=> 'meta' = shapes only, ~0 extra RAM, instant: used to plan memory and to build the skeleton before filling it.

--- 6. Hugging Face cache layout (models--gpt2) ---
HF_HOME = (unset -> ~/.cache/huggingface) | HF_HUB_CACHE = (unset -> $HF_HOME/hub) | HF_HUB_OFFLINE = 1
refs/main -> 607a30d783dfa663caf39e06633721c8d4cfcd7e
blobs/ (7 files, content-addressed: the filename is the sha256 or git-sha of the content):
   248dfc3911869ec4...   548,105,171 bytes
   4b988bccc9dc5ada...     1,355,256 bytes
   1f1d9aaca301414e...     1,042,301 bytes
snapshots/607a30d7...  (one folder per commit; entries are symlinks):
   config.json            -> ../../blobs/10c66461e4c109db5a2196...   is_symlink=True
   generation_config.json -> ../../blobs/3dc481ecc3b2c47a06ab4e...   is_symlink=True
   merges.txt             -> ../../blobs/226b0752cac7789c48f0cb...   is_symlink=True
   model.safetensors      -> ../../blobs/248dfc3911869ec493c76e...   is_symlink=True
   tokenizer.json         -> ../../blobs/4b988bccc9dc5adacd403c...   is_symlink=True
   tokenizer_config.json  -> ../../blobs/be4d21d94f3b4687e5a54d...   is_symlink=True
   vocab.json             -> ../../blobs/1f1d9aaca301414e7f6c93...   is_symlink=True

cleanup done: temp folder removed
```
- Pickle: the opcodes `STACK_GLOBAL` (find `builtins.print`) and `REDUCE` (call it) are all it takes; `pickle.loads` printed the message. `torch.load` with the default `weights_only=True` refused the file with `UnpicklingError: Weights only load failed`.
- safetensors: 968 bytes = 8 + 256 (JSON) + 704 raw; data offsets 0:64, 64:576, 576:704 are contiguous; all three tensors rebuilt from `np.frombuffer` equal the originals.
- Loading (300 MB file, warm cache, fresh process each): eager `load_file` RSS grew by about 261 MB with a peak of 628 MB (the copy briefly coexists with the mapped file); `safe_open` reading one 50 MB tensor grew RSS by 50 MB; the raw `np.memmap` view grew it by 0 MB because no page was touched. Times vary by machine and load (an earlier run on a busier machine showed eager 2.8-5.4 s).
- RSS caveat: on macOS under memory pressure RSS can be compressed or undercounted (one busy-machine run showed eager peak below the data size), so read RSS as indicative, not exact.
- Shards: 7 tensors in 3 shards, one tensor fetched from its own shard via `weight_map`, all tensors recovered shard by shard.
- Real models: `from_pretrained` of distilgpt2 shows only +21 MB RSS growth for a 328 MB model because safetensors weights are mapped lazily and only touched pages count; bf16 conversion forces a real copy (+54 MB RSS, peak 847 MB). The `meta` rows report the full parameter count (82M, 124M) on device `meta` with no weight data.
- Cache: `snapshots/607a30d7.../model.safetensors` is a symlink to a 548,105,171-byte blob; every entry is a symlink.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `SIZE_MB=800` | eager peak RSS grows with the file; `safe_open`/`memmap` rows barely move | lazy mapping touches only the pages you read |
| read all six tensors in `lazy1` | peak RSS approaches the eager row | touching pages makes them resident |
| add an `np.float64` tensor to section 2 | dtype string `F64` appears in the header | dtype is a plain string in the JSON |
| `N = 5` shards in section 4 | five files; `weight_map` changes | the index is the only source of truth |
| drop the file out of the page cache (reboot / `purge` on macOS) before a run | the first load is slower | cold versus warm cache |

## Debugging practice
**BREAK IT 1:** change `"<Q"` to `">Q"` in section 2 (big-endian). The header length becomes an absurd number, `raw[8:8+n]` silently slices to the end of the file and `json.loads` raises an error (in a real run: `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 ...`, because the slice runs into raw tensor bytes; the header length read as 4035225266123964416). Lesson: the format is little-endian; a "garbage header size" error usually means a wrong file or wrong endianness.

**BREAK IT 2:** use `offset=s` instead of `offset=data_start + s`. The values are wrong, but there is no exception: the tensor is silently read from header bytes. Always compare against a known tensor (here `equal to original: True`).

**Real-world error:** `UnpicklingError: Weights only load failed` from `torch.load` means the file contains something other than plain tensors. Do not just add `weights_only=False`: first ask where the file came from; prefer converting it to safetensors in a trusted environment.

## Common mistakes
- Trusting a downloaded `.bin`/`.pt` with `weights_only=False`.
- Comparing load times without saying cold or warm cache.
- Reading peak RSS in the same process for several strategies (it only goes up).
- Believing mmap makes a model fit in less RAM than it needs to run.
- Forgetting that a checkpoint saved in bf16 is converted when you load it as fp32 (twice the RAM).
- Deleting `snapshots/` entries by hand: they are symlinks into `blobs/`.
- Mixing up `HF_HOME` and `HF_HUB_CACHE`, then wondering why a model "is not cached".

## Rebuild checklist
1. Write the 8-byte length + JSON header parser and rebuild a tensor with `np.frombuffer`.
2. Write a safetensors file with `save_file` and verify round-trip equality.
3. Write a harmless `__reduce__` class and show `pickle.loads` runs it.
4. Time `load_file` vs `safe_open` for one tensor in a fresh process.
5. Write shards plus `model.safetensors.index.json` and load one tensor via `weight_map`.
6. List `blobs/`, `snapshots/`, `refs/` of a cached model and follow a symlink.

## Say it aloud
- "Pickle is a program: unpickling calls whatever callable the file names, so loading an untrusted pickle is arbitrary code execution; safetensors is an 8-byte length, a JSON header and raw bytes, so it is data only."
- "Because safetensors is memory-mapped I can open a huge file instantly and touch only the tensors I need; peak RSS follows what I read, not the file size."
- "Big checkpoints are sharded with an index.json weight_map, and `from_pretrained` with `device_map` builds the model on the meta device and places each tensor straight onto its device to avoid a second copy."
- "The Hugging Face cache stores files once in `blobs/` by content hash and each revision is a folder of symlinks in `snapshots/`."

## The new problem (cliffhanger)
safetensors and the Hugging Face cache are the GPU-server world. The laptop world (llama.cpp, Ollama) uses a different single-file format with its own quantized block types: [step 02, GGUF and the llama.cpp formats](../02_gguf_and_llamacpp_formats/WALKTHROUGH.md).
