"""
GOAL:  Build a tiny valid GGUF v3 file with Python struct, read it back with a gguf-dump style printout,
       implement Q8_0 (and Q4_0) block quantization, and compute file sizes for 7B/8B models at each quant type.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 21_llm_inference_engineering/02_gguf_and_llamacpp_formats/main.py
EXPECTED OUTPUT (shortened):
    --- 1. wrote tiny.gguf: 15,584 bytes | tensor data starts at byte 736 (multiple of 32: True)
    first 24 bytes: 47 47 55 46 03 00 00 00 ... -> magic b'GGUF' | version 3 | tensors 4 | kv 9
    GGUF v3 | tensors: 4 | metadata kv: 9 | alignment 32 | data starts at byte 736
      general.architecture             STRING   'tinyllama-demo'
      blk.0.attn_q.weight      dims [64, 64]   type Q8_0  offset 2048   bytes 4352
    Q8_0: 8.50 bits/weight | RMSE 0.000107 ...   Q4_0: 4.50 bits/weight | RMSE 0.001720 ...
    Q4_K_M  4.85 bits/weight   4.1 GB (7B)   4.9 GB (8B)
NEEDS: numpy only (no downloads).  Optional: the `gguf` package for a cross-check (skipped if absent).  < 2 s.
"""

import mmap
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------------
# GGUF constants (from the ggml project's docs/gguf.md)
# ---------------------------------------------------------------------------------------------------
MAGIC = b"GGUF"
VERSION = 3
ALIGNMENT = 32                                   # default general.alignment

# metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = range(13)
TYPE_NAME = {0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32", 5: "INT32", 6: "FLOAT32", 7: "BOOL",
             8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64", 12: "FLOAT64"}
SCALAR_FMT = {T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h", T_UINT32: "<I", T_INT32: "<i",
              T_FLOAT32: "<f", T_BOOL: "<?", T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d"}

# tensor (ggml) types we use, plus the block layout of each: (values per block, bytes per block)
GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0 = 0, 1, 2, 8
GGML_NAME = {GGML_F32: "F32", GGML_F16: "F16", GGML_Q4_0: "Q4_0", GGML_Q8_0: "Q8_0"}
BLOCK = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_Q4_0: (32, 18), GGML_Q8_0: (32, 34)}


# ---------------------------------------------------------------------------------------------------
# 1. Block quantization: Q8_0 and Q4_0  (block of 32 weights = one fp16 scale + the integers)
# ---------------------------------------------------------------------------------------------------
def quantize_q8_0(x):
    """x: float32 array, size multiple of 32.  Returns bytes: per block [fp16 scale d][32 x int8]  = 34 bytes."""
    blocks = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(blocks).max(axis=1)                       # biggest magnitude in each block
    d = amax / 127.0                                         # scale so that the biggest value maps to +-127
    inv = np.where(d == 0, 0.0, 1.0 / np.where(d == 0, 1, d))
    q = np.rint(blocks * inv[:, None]).astype(np.int8)       # round to nearest integer in [-127, 127]
    out = np.empty(len(blocks), dtype=[("d", "<f2"), ("q", "i1", 32)])   # packed record = 2 + 32 bytes, no padding
    out["d"] = d.astype(np.float16)                          # the scale itself is stored in half precision
    out["q"] = q
    return out.tobytes()


def dequantize_q8_0(buf, n):
    rec = np.frombuffer(buf, dtype=[("d", "<f2"), ("q", "i1", 32)], count=n // 32)
    return (rec["d"].astype(np.float32)[:, None] * rec["q"].astype(np.float32)).reshape(-1)


def quantize_q4_0(x):
    """Q4_0: block of 32, fp16 scale + 32 four-bit values packed two per byte = 18 bytes (4.5 bits/weight).
    Following ggml's reference: d = (value with largest |.|, keeping its sign) / -8; q = clip(round(x/d) + 8, 0, 15)."""
    blocks = x.reshape(-1, 32).astype(np.float32)
    idx = np.abs(blocks).argmax(axis=1)
    mx = blocks[np.arange(len(blocks)), idx]                 # signed max-magnitude element
    d = mx / -8.0
    inv = np.where(d == 0, 0.0, 1.0 / np.where(d == 0, 1, d))
    q = np.clip(np.rint(blocks * inv[:, None]) + 8, 0, 15).astype(np.uint8)
    packed = q[:, :16] | (q[:, 16:] << 4)                    # byte j = (q[j], q[j+16]) low nibble, high nibble
    out = np.empty(len(blocks), dtype=[("d", "<f2"), ("qs", "u1", 16)])
    out["d"] = d.astype(np.float16)
    out["qs"] = packed
    return out.tobytes()


def dequantize_q4_0(buf, n):
    rec = np.frombuffer(buf, dtype=[("d", "<f2"), ("qs", "u1", 16)], count=n // 32)
    lo = (rec["qs"] & 0x0F).astype(np.int8) - 8
    hi = (rec["qs"] >> 4).astype(np.int8) - 8
    q = np.concatenate([lo, hi], axis=1).astype(np.float32)
    return (rec["d"].astype(np.float32)[:, None] * q).reshape(-1)


# ---------------------------------------------------------------------------------------------------
# 2. Writing a GGUF file
# ---------------------------------------------------------------------------------------------------
def w_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b                     # GGUF string = uint64 length + utf-8 bytes (no NUL)


def w_value(vtype, v):
    if vtype == T_STRING:
        return w_string(v)
    if vtype == T_ARRAY:                                     # v = (element_type, list)
        et, items = v
        return struct.pack("<IQ", et, len(items)) + b"".join(w_value(et, it) for it in items)
    return struct.pack(SCALAR_FMT[vtype], v)


def pad_to(n, align):
    return (-n) % align                                      # bytes of zero padding needed to reach a multiple


def write_gguf(path, metadata, tensors):
    """metadata: list of (key, type, value).  tensors: list of (name, numpy_shape, ggml_type, raw_bytes)."""
    head = MAGIC + struct.pack("<I", VERSION) + struct.pack("<QQ", len(tensors), len(metadata))
    kv = b"".join(w_string(k) + struct.pack("<I", t) + w_value(t, v) for k, t, v in metadata)

    infos, offset = b"", 0
    for name, shape, gtype, raw in tensors:
        dims = list(reversed(shape))                         # ggml lists the FASTEST-varying dimension first
        infos += w_string(name) + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        infos += struct.pack("<IQ", gtype, offset)           # offset is relative to the start of the data section
        offset += len(raw) + pad_to(len(raw), ALIGNMENT)     # every tensor starts on a 32-byte boundary

    header = head + kv + infos
    header += b"\x00" * pad_to(len(header), ALIGNMENT)       # data section itself is aligned too
    with open(path, "wb") as f:
        f.write(header)
        for _, _, _, raw in tensors:
            f.write(raw + b"\x00" * pad_to(len(raw), ALIGNMENT))
    return len(header)


# ---------------------------------------------------------------------------------------------------
# 3. Reading it back (a mini gguf-dump)
# ---------------------------------------------------------------------------------------------------
class Reader:
    def __init__(self, buf):
        self.b, self.p = buf, 0

    def take(self, fmt):
        size = struct.calcsize(fmt)
        v = struct.unpack_from(fmt, self.b, self.p)
        self.p += size
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("<Q")
        s = bytes(self.b[self.p:self.p + n]).decode("utf-8")
        self.p += n
        return s

    def value(self, vtype):
        if vtype == T_STRING:
            return self.string()
        if vtype == T_ARRAY:
            et, n = self.take("<IQ")
            return [self.value(et) for _ in range(n)]
        return self.take(SCALAR_FMT[vtype])


def read_gguf(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)  # the file is mapped, not read: what llama.cpp does
    r = Reader(mm)
    assert bytes(mm[:4]) == MAGIC, "not a GGUF file"
    r.p = 4
    version = r.take("<I")
    n_tensors, n_kv = r.take("<QQ")
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        vt = r.take("<I")
        meta[key] = (vt, r.value(vt))
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.take("<I")
        dims = [r.take("<Q") for _ in range(nd)]
        gtype, off = r.take("<IQ")
        infos.append((name, dims, gtype, off))
    align = meta.get("general.alignment", (None, ALIGNMENT))[1]
    data_start = r.p + pad_to(r.p, align)
    return mm, version, meta, infos, data_start, align


def tensor_nbytes(dims, gtype):
    n = int(np.prod(dims))
    per, nbytes = BLOCK[gtype]
    return n // per * nbytes


def dump(path):
    mm, version, meta, infos, data_start, align = read_gguf(path)
    print(f"GGUF v{version} | tensors: {len(infos)} | metadata kv: {len(meta)} | alignment {align} | data starts at byte {data_start}")
    for k, (vt, v) in meta.items():
        shown = repr(v) if not isinstance(v, list) else f"[{len(v)} items] {v[:4]}..."
        print(f"  {k:<32} {TYPE_NAME[vt]:<8} {shown[:70]}")
    for name, dims, gtype, off in infos:
        print(f"  {name:<24} dims {str(dims):<10} type {GGML_NAME[gtype]:<5} offset {off:<6} bytes {tensor_nbytes(dims, gtype)}")
    return mm, meta, infos, data_start


def load_tensor(mm, data_start, name, dims, gtype, off):
    n = int(np.prod(dims))
    start = data_start + off
    raw = mm[start:start + tensor_nbytes(dims, gtype)]
    shape = list(reversed(dims))                             # back to numpy order
    if gtype == GGML_F32:
        return np.frombuffer(raw, dtype="<f4").reshape(shape)
    if gtype == GGML_F16:
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
    if gtype == GGML_Q8_0:
        return dequantize_q8_0(raw, n).reshape(shape)
    if gtype == GGML_Q4_0:
        return dequantize_q4_0(raw, n).reshape(shape)


# ---------------------------------------------------------------------------------------------------
# 4. Bits-per-weight table and file-size estimates
# ---------------------------------------------------------------------------------------------------
# Approximate bits per weight. Pure types follow from the block layout (e.g. Q4_0: 18 bytes per 32 weights = 4.5).
# The *_K_M / *_K_S "mixes" keep a few sensitive tensors at a higher type, so their bpw is an average (approximate).
BPW = [
    ("F16",      16.0,   "unquantized half precision (reference)"),
    ("Q8_0",      8.5,   "block 32: fp16 scale + 32 int8 = 34 B"),
    ("Q6_K",   6.5625,   "k-quant, super-block 256 (16 sub-blocks of 16)"),
    ("Q5_K_M",    5.7,   "mix of Q5_K and Q6_K (approx)"),
    ("Q4_K_M",    4.85,  "mix of Q4_K and Q6_K (approx; the common default)"),
    ("Q4_0",      4.5,   "block 32: fp16 scale + 32 x 4 bit = 18 B (legacy)"),
    ("IQ4_XS",    4.25,  "i-quant: non-linear 4-bit codebook, super-block 256"),
    ("Q3_K_M",    3.9,   "mix of Q3_K and Q4_K/Q5_K (approx)"),
    ("Q2_K",      2.6,   "k-quant 2-bit mix (approx; quality drops a lot)"),
]


def size_table():
    print(f"{'type':<8}{'bits/weight':>12}{'7B (6.74B) GB':>16}{'8B (8.03B) GB':>16}   note")
    for name, bpw, note in BPW:
        gb7 = 6.74e9 * bpw / 8 / 1e9
        gb8 = 8.03e9 * bpw / 8 / 1e9
        print(f"{name:<8}{bpw:>12.2f}{gb7:>16.1f}{gb8:>16.1f}   {note}")


def main():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((8, 64)).astype(np.float32)
    wq = (rng.standard_normal((64, 64)) * 0.05).astype(np.float32)
    wk = (rng.standard_normal((64, 64)) * 0.05).astype(np.float32)
    norm = np.ones(64, dtype=np.float32)

    metadata = [
        ("general.architecture", T_STRING, "tinyllama-demo"),
        ("general.name", T_STRING, "tiny GGUF from scratch"),
        ("general.alignment", T_UINT32, ALIGNMENT),
        ("general.file_type", T_UINT32, 7),                  # 7 = "mostly Q8_0" in llama.cpp's file-type enum
        ("tinyllama-demo.block_count", T_UINT32, 1),
        ("tinyllama-demo.context_length", T_UINT32, 128),
        ("tinyllama-demo.rope.freq_base", T_FLOAT32, 10000.0),
        ("tokenizer.ggml.tokens", T_ARRAY, (T_STRING, ["<s>", "</s>", "hello", "world"])),
        ("tokenizer.ggml.scores", T_ARRAY, (T_FLOAT32, [0.0, 0.0, -1.0, -2.0])),
    ]
    tensors = [
        ("token_embd.weight", emb.shape, GGML_F32, emb.tobytes()),
        ("blk.0.attn_q.weight", wq.shape, GGML_Q8_0, quantize_q8_0(wq.reshape(-1))),
        ("blk.0.attn_k.weight", wk.shape, GGML_F16, wk.astype("<f2").tobytes()),
        ("blk.0.attn_norm.weight", norm.shape, GGML_F32, norm.tobytes()),
    ]
    tmp = Path(tempfile.mkdtemp(prefix="gguf_demo_"))
    path = tmp / "tiny.gguf"
    data_start = write_gguf(path, metadata, tensors)
    print(f"--- 1. wrote tiny.gguf: {path.stat().st_size:,} bytes | tensor data starts at byte {data_start} "
          f"(multiple of {ALIGNMENT}: {data_start % ALIGNMENT == 0})")
    with open(path, "rb") as f:
        head = f.read(24)
    print("first 24 bytes:", head.hex(" "), "-> magic", head[:4], "| version", struct.unpack("<I", head[4:8])[0],
          "| tensors", struct.unpack("<Q", head[8:16])[0], "| kv", struct.unpack("<Q", head[16:24])[0])

    print("\n--- 2. dump (like `gguf-dump`) ---")
    mm, meta, infos, ds = dump(path)

    print("\n--- 3. read tensors back from the memory-mapped file ---")
    for name, dims, gtype, off in infos:
        t = load_tensor(mm, ds, name, dims, gtype, off)
        ref = {"token_embd.weight": emb, "blk.0.attn_q.weight": wq, "blk.0.attn_k.weight": wk, "blk.0.attn_norm.weight": norm}[name]
        print(f"  {name:<24} {GGML_NAME[gtype]:<4} shape {t.shape} max abs error vs original: {np.abs(t - ref).max():.5f}")

    print("\n--- 4. Q8_0 vs Q4_0 on 1M gaussian weights (std 0.02, like a real layer) ---")
    w = (rng.standard_normal(1_000_000) * 0.02).astype(np.float32)
    w = w[: len(w) // 32 * 32]
    for label, q, dq, nb in (("Q8_0", quantize_q8_0, dequantize_q8_0, 34), ("Q4_0", quantize_q4_0, dequantize_q4_0, 18)):
        buf = q(w)
        back = dq(buf, len(w))
        rmse = float(np.sqrt(np.mean((back - w) ** 2)))
        print(f"  {label}: {len(buf) / len(w) * 8:.2f} bits/weight | RMSE {rmse:.6f} | relative to weight std {rmse / w.std():.4f} | max abs err {np.abs(back - w).max():.5f}")

    print("\n--- 5. file size estimates (params x bits per weight / 8) ---")
    size_table()

    print("\n--- 6. optional cross-check with the `gguf` package ---")
    try:
        import gguf                                          # pip install gguf (not installed in this course venv)
        r = gguf.GGUFReader(str(path))
        print("gguf package reads", len(r.tensors), "tensors:", [t.name for t in r.tensors])
    except ImportError:
        print("`gguf` package not installed -> skipped (our hand-written reader above is the check).")

    mm.close()
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()                                              # remove the temp file


if __name__ == "__main__":
    main()

# Reference commands (NEVER run in this course; they need a llama.cpp checkout and a downloaded HF model):
#   python convert_hf_to_gguf.py /path/to/hf_model --outfile model-f16.gguf --outtype f16
#   ./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M
#   ./llama-cli -m model-Q4_K_M.gguf -p "Hello" -n 64          # llama.cpp mmaps the file by default
# TRY IT: change the std of `wq` to 1.0 and 0.001: Q8_0 relative error stays about the same (the scale adapts per block).
# TRY IT: add one outlier (value 5.0) to a block of wq and see that block's error grow (one scale per 32 weights).
# BREAK IT: delete the `header += b"\x00" * pad_to(...)` line in write_gguf: the reader aligns data_start, the writer did not, so tensors decode as garbage.
# BREAK IT: change the Q8_0 tensor shape to (64, 30): the toy still runs, but real GGML needs row length % 32 == 0 (blocks must not straddle rows).
