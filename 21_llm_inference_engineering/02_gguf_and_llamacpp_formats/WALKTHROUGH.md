# Step 02 — GGUF and the llama.cpp formats

## The story so far
[Step 01](../01_checkpoint_formats_and_loading/WALKTHROUGH.md) opened a safetensors file by hand: header length, JSON, raw bytes. That is the format of the Hugging Face and GPU world. Now meet the other big format, the one you download for a laptop.

## The pain point
You want to run a 7B model on a MacBook or a modest PC. The fp16 safetensors file is 13-14 GB; the tokenizer, config and generation settings are separate files; and the weights are too big for the RAM you have. People keep talking about `Q4_K_M`, `Q8_0`, `IQ4_XS` as if everyone knew what the letters mean.

## The spark
GGUF is one self-describing file: header, typed metadata (architecture, tokenizer, RoPE settings), tensor descriptions, then the tensor data, aligned so the file can be memory-mapped and used in place. Its quantization types store small blocks of weights with their own scale. Once you can write a GGUF yourself, the names stop being magic: each one is "block size, bits, how the scale is stored".

## Goal
Write and read a valid GGUF v3 file with `struct`, implement Q8_0 (and Q4_0), and be able to estimate the file size of any model at any quantization.

## The theory (read this before the interview)

### GGUF layout
```
offset 0   : "GGUF"                       4 bytes magic
             version                      uint32   (3)
             tensor_count                 uint64
             metadata_kv_count            uint64
             metadata KV pairs ...        key(string) | value_type(uint32) | value
             tensor infos ...             name | n_dims(uint32) | dims[n_dims](uint64) | ggml_type(uint32) | offset(uint64)
             zero padding up to a multiple of general.alignment (default 32)
data start : tensor 0 bytes (padded to alignment) | tensor 1 bytes (padded) | ...
```
- A string is a `uint64` length followed by UTF-8 bytes (not NUL-terminated).
- An array is element type (`uint32`), count (`uint64`), then the items.
- Tensor `offset` is relative to the start of the data section, not the file.
- ggml lists dimensions fastest-varying first: a numpy shape `(rows, cols)` appears as `[cols, rows]`.
- Everything is little-endian in this course's version of the format (GGUF can also be big-endian, rare).

### Block quantization
```
Q8_0 block (32 weights):   [ d : fp16 ][ q0 q1 ... q31 : int8 ]   = 2 + 32 = 34 bytes -> 34*8/32 = 8.5 bits per weight
Q4_0 block (32 weights):   [ d : fp16 ][ 16 bytes = 32 x 4-bit ]  = 2 + 16 = 18 bytes -> 4.5 bits per weight
dequantize:   w ~ d * q            (Q4_0: w ~ d * (nibble - 8))
```
The scale costs bits too, so "8-bit" is 8.5 bits per weight. A smaller block follows the local range better but spends a larger fraction of bits on scales.

### K-quants and i-quants (names decoded)
- `Q4_K_M`: **Q** quantized, **4** about 4 bits, **K** the k-quant family (super-blocks of 256 weights split into sub-blocks, with the sub-block scales themselves quantized to a few bits), **M** "medium" mix (S/M/L = small/medium/large): some sensitive tensors get a higher-bit type (for example Q6_K) and the rest the base type. So the bits per weight is an average.
- Pure per-type costs you can derive: Q4_K = 144 bytes per 256 weights = 4.5 bpw; Q5_K = 176/256 = 5.5; Q6_K = 210/256 = 6.5625.
- `IQ4_XS`, `IQ3_XXS` ...: **i-quants** use non-linear, codebook-style quantization (often with an importance matrix) to reach very low bits per weight; IQ4_XS is about 4.25 bpw. They cost a little more compute per weight.
- The legacy types (`Q4_0`, `Q4_1`, `Q5_0`, `Q8_0`) have block size 32 and no super-block.
- **Importance matrix (imatrix):** activation statistics collected on sample text and given to `llama-quantize` so error is weighted toward weights that matter. It is a calibration signal like in GPTQ/AWQ (step 06).

### Size table (approximate; params x bits-per-weight / 8)
| Type | bits/weight | 7B (6.74B) | 8B (8.03B) |
|---|---|---|---|
| F16 | 16 | 13.5 GB | 16.1 GB |
| Q8_0 | 8.5 | 7.2 GB | 8.5 GB |
| Q6_K | 6.56 | 5.5 GB | 6.6 GB |
| Q5_K_M | about 5.7 | 4.8 GB | 5.7 GB |
| Q4_K_M | about 4.85 | 4.1 GB | 4.9 GB |
| Q4_0 | 4.5 | 3.8 GB | 4.5 GB |
| IQ4_XS | about 4.25 | 3.6 GB | 4.3 GB |
The `_M` rows are averages of mixes; real files differ by model architecture (for example the embedding and output tensors are often kept at higher precision), so treat sizes as +-5 percent. This is weights only: add the KV cache from [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md).

### How llama.cpp and Ollama use it
- llama.cpp (C/C++ on the ggml tensor library) **memory-maps** the GGUF file by default: tensor data is used in place, pages are loaded on demand and shared through the OS page cache, so a second start is fast. Backends: CPU (with SIMD), Metal, CUDA, Vulkan and others.
- Ollama is a wrapper: a model registry (pull by name), a local server and REST API (port 11434), and a Modelfile for templates and parameters; it runs GGUF models through llama.cpp/ggml (and its own engine for some models).
- Conversion pipeline (reference only, never run in this course):
```bash
python convert_hf_to_gguf.py /path/to/hf_model --outfile model-f16.gguf --outtype f16   # safetensors+config+tokenizer -> GGUF
./llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M                               # f16 GGUF -> quantized GGUF
./llama-cli -m model-Q4_K_M.gguf -p "Hello" -n 64                                      # run it
```
- Positioning: GGUF/llama.cpp = local, CPU or mixed CPU+GPU, low-bit weight-only formats. For GPU servers the usual formats are safetensors with AWQ/GPTQ/FP8 for vLLM-style engines (steps 04-06, 15).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| GGUF | ggml file, llama.cpp model file | Yes (successor of GGML/GGJT files) | Single self-describing file: metadata + tensors, aligned for mmap. |
| Q8_0 / Q4_0 | legacy quants | Family | Block of 32 + fp16 scale; 8.5 / 4.5 bits per weight. |
| Q4_K_M / Q5_K_M / Q6_K | k-quants | Family | 256-weight super-blocks; `_M` is a per-tensor mix, so bpw is an average. |
| IQ4_XS / IQ3_XXS | i-quants | Family | Codebook/non-linear, usually with an importance matrix. |
| bits per weight (bpw) | bits/param | Yes | Total file bits over parameter count, scales included. |
| imatrix | importance matrix | Yes | Calibration statistics for the quantizer, not part of the model. |
| mmap | memory-mapped file | Yes | OS maps the file; pages load lazily. |
| Ollama | llama.cpp wrapper | No | Registry + server around the engine; the engine is llama.cpp/ggml. |
| safetensors vs GGUF | HF format vs ggml format | No | safetensors = tensors + small header (no tokenizer); GGUF = tensors + full model metadata + quant block types. |

## Run it
```bash
cd 21_llm_inference_engineering/02_gguf_and_llamacpp_formats
python main.py                       # under 2 seconds, numpy only
# from repo root:
.venv/bin/python 21_llm_inference_engineering/02_gguf_and_llamacpp_formats/main.py
```

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–16)

```python
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
```

- **In simple words:** What the step builds and the shortened expected output.
- **Exactly:** Docstring only; the numbers in it come from a real run (15,584-byte file, data starting at byte 736).
- **Why it is here:** Lets you compare your output with the expected one.

### 2. Imports  (`main.py` lines 18–24)

```python
import mmap
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
```

- **In simple words:** Standard-library tools for bytes and files, plus numpy.
- **Exactly:** `struct` packs/unpacks fixed binary layouts, `mmap` maps a file into memory, `tempfile` makes a throw-away folder.
- **Why it is here:** GGUF is just bytes: no third-party package is needed to read or write it.

### 3. GGUF constants and type tables  (`main.py` lines 26–43)

```python
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
```

- **In simple words:** The magic text, the version, the default alignment, and numbered tables for metadata value types and tensor (ggml) types.
- **Exactly:** `MAGIC=b"GGUF"`, `VERSION=3`, `ALIGNMENT=32`. Value types 0..12 (UINT8 .. FLOAT64, STRING=8, ARRAY=9). `SCALAR_FMT` maps scalar types to `struct` codes (`<` = little-endian). `BLOCK[type] = (values per block, bytes per block)`: F32 (1,4), F16 (1,2), Q4_0 (32,18), Q8_0 (32,34).
- **Why it is here:** The file refers to types by number, so writer and reader must agree on the table; block sizes give tensor byte sizes.

### 4. Q8_0 block quantization  (`main.py` lines 46–64)

```python
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
```

- **In simple words:** Cut the weights into blocks of 32. For each block find the biggest absolute value, divide by 127 to get a scale, store each weight as a rounded integer in [-127,127] plus the scale.
- **Exactly:** `x.reshape(-1, 32)` gives `(n_blocks, 32)`; `d = amax/127` has shape `(n_blocks,)`; `q = rint(blocks/d)` is int8. The structured dtype `[("d","<f2"),("q","i1",32)]` is a packed 34-byte record. `dequantize_q8_0` reads the record back and returns `d * q`.
- **Why it is here:** The simplest real GGUF quantization; one scale per small block means an outlier only hurts its block.

### 5. Q4_0 block quantization  (`main.py` lines 67–88)

```python
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
```

- **In simple words:** The same idea at 4 bits: 16 levels per weight, two weights packed per byte.
- **Exactly:** Following the ggml reference: `d = (signed element of largest magnitude) / -8`, `q = clip(round(x/d)+8, 0, 15)`; byte j holds `q[j]` (low nibble) and `q[j+16]` (high nibble). 18 bytes per 32 weights = 4.5 bits per weight. Dequantize: `d * (nibble - 8)`.
- **Why it is here:** Shows what 4 bits costs in accuracy and why block size and scale storage matter for bits-per-weight.

### 6. Writing a GGUF file  (`main.py` lines 91–130)

```python
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
```

- **In simple words:** Pack the header, metadata, tensor descriptions and the aligned tensor data into one file.
- **Exactly:** `w_string` = uint64 length + UTF-8 bytes. `w_value` writes scalars with `struct.pack`, strings, or arrays (element type `uint32`, count `uint64`, items). `write_gguf` writes `b"GGUF"`, version, tensor count and KV count, the KV pairs, then per tensor: name, `n_dims`, dims (reversed numpy shape), ggml type, offset relative to the data section. Zero padding brings the header and every tensor to a multiple of 32.
- **Why it is here:** The entire GGUF v3 layout in about 40 lines.

### 7. Reading a GGUF file (mini gguf-dump)  (`main.py` lines 133–215)

```python
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
```

- **In simple words:** Walk the same layout in the same order, then look up each tensor by offset in a memory-mapped file.
- **Exactly:** `Reader.take(fmt)` wraps `struct.unpack_from` and advances a cursor. `read_gguf` memory-maps the file (`mmap.ACCESS_READ`, like llama.cpp), checks the magic, reads version/counts, KV pairs, tensor infos, then rounds the cursor up to the alignment to get `data_start`. `load_tensor` slices the map, wraps it with `np.frombuffer` and dequantizes if needed.
- **Why it is here:** If the reader rebuilds every tensor, the file is valid. GGUF loads fast because the data section can be used in place.

### 8. Bits-per-weight table and size estimates  (`main.py` lines 218–241)

```python
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
```

- **In simple words:** A table of quantization names with approximate bits per weight and the resulting file size for a 7B and 8B model.
- **Exactly:** `size = params x bpw / 8`. Pure types follow from block layout (Q8_0: 34 B/32 = 8.5; Q4_0: 18/32 = 4.5; Q6_K: 210 B per 256 = 6.5625). The `_K_M` entries are mixes, so their value is an approximate average, flagged in the note column.
- **Why it is here:** Back-of-envelope sizing is the standard interview move before downloading anything.

### 9. Build the tensors and write the file  (`main.py` lines 244–276)

```python
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
```

- **In simple words:** Make small tensors (float32, float16, Q8_0), list the metadata, write the file and look at the first 24 bytes.
- **Exactly:** Metadata includes `general.architecture`, `general.alignment`, `general.file_type=7` (llama.cpp's id for mostly-Q8_0), `tokenizer.ggml.tokens` as a STRING array. The first 24 bytes decode as `47 47 55 46` ("GGUF"), `03 00 00 00` (version 3), tensor count 4, KV count 9.
- **Why it is here:** A GGUF header is readable with a hex dump.

### 10. Dump and read the tensors back  (`main.py` lines 278–285)

```python
    print("\n--- 2. dump (like `gguf-dump`) ---")
    mm, meta, infos, ds = dump(path)

    print("\n--- 3. read tensors back from the memory-mapped file ---")
    for name, dims, gtype, off in infos:
        t = load_tensor(mm, ds, name, dims, gtype, off)
        ref = {"token_embd.weight": emb, "blk.0.attn_q.weight": wq, "blk.0.attn_k.weight": wk, "blk.0.attn_norm.weight": norm}[name]
        print(f"  {name:<24} {GGML_NAME[gtype]:<4} shape {t.shape} max abs error vs original: {np.abs(t - ref).max():.5f}")
```

- **In simple words:** Print the file like `gguf-dump` and check every tensor against the original.
- **Exactly:** `dump(path)` prints metadata and tensor infos; `load_tensor` returns numpy arrays in the original shape. Max errors: F32 0, F16 about 6e-5, Q8_0 about 6e-4.
- **Why it is here:** Proof the file is parseable and quantization error is small.

### 11. Q8_0 versus Q4_0 error on 1M weights  (`main.py` lines 287–294)

```python
    print("\n--- 4. Q8_0 vs Q4_0 on 1M gaussian weights (std 0.02, like a real layer) ---")
    w = (rng.standard_normal(1_000_000) * 0.02).astype(np.float32)
    w = w[: len(w) // 32 * 32]
    for label, q, dq, nb in (("Q8_0", quantize_q8_0, dequantize_q8_0, 34), ("Q4_0", quantize_q4_0, dequantize_q4_0, 18)):
        buf = q(w)
        back = dq(buf, len(w))
        rmse = float(np.sqrt(np.mean((back - w) ** 2)))
        print(f"  {label}: {len(buf) / len(w) * 8:.2f} bits/weight | RMSE {rmse:.6f} | relative to weight std {rmse / w.std():.4f} | max abs err {np.abs(back - w).max():.5f}")
```

- **In simple words:** Quantize a million Gaussian weights both ways and compare bits and error.
- **Exactly:** std 0.02 mimics a real layer. RMSE relative to the weight standard deviation: Q8_0 about 0.5 percent, Q4_0 about 8.6 percent.
- **Why it is here:** A feel for the quality/size trade-off behind all the naming.

### 12. File sizes, optional gguf package, cleanup  (`main.py` lines 296–310)

```python
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
```

- **In simple words:** Print the size table, try the optional `gguf` package, delete the temp file.
- **Exactly:** `import gguf` is inside `try/except ImportError`; the package is not installed here, so a skip message is printed. `mm.close()` unmaps the file before deletion.
- **Why it is here:** Optional dependencies must never break the mandatory path.

### 13. Entry point, reference commands, TRY IT / BREAK IT  (`main.py` lines 313–323)

```python
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
```

- **In simple words:** Run `main()`; comments list the real llama.cpp commands (never run here) and experiments.
- **Exactly:** `convert_hf_to_gguf.py` converts a Hugging Face model directory to an f16 GGUF; `llama-quantize` converts a GGUF to a chosen type; `llama-cli` loads a GGUF (mmap by default).
- **Why it is here:** Connects the toy file to the real workflow.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `struct.pack("<QQ", a, b)` | `<` little-endian; `Q` = uint64, `I` = uint32, `H` = uint16, `B` = uint8, `f`/`d` = float32/64 | `struct.pack("<I", 3)` is `03 00 00 00` |
| `struct.unpack_from(fmt, buf, pos)` | read from a buffer at a position without slicing | cursor walk in `Reader.take` |
| `(-n) % align` | bytes of padding to the next multiple | `(-100) % 32 == 28` |
| `np.dtype([("d","<f2"),("q","i1",32)])` | structured (record) dtype: one record = 2-byte half + 32 int8 | `np.frombuffer(buf, dtype=rec)` |
| `x >> 4`, `x & 0x0F`, `a | (b << 4)` | pack/unpack two 4-bit values in a byte | Q4_0 |
| `bytes.hex(" ")` | hex dump with spaces | `head.hex(" ")` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Internally |
|---|---|---|---|
| `struct` | stdlib | binary packing | C-style fixed layouts from a format string |
| `mmap.mmap(fileno, 0, access=ACCESS_READ)` | stdlib | map whole file read-only | OS virtual-memory mapping; bytes are paged in when touched |
| `np.frombuffer(buf, dtype, count)` | numpy | view bytes as an array | no copy, shares the buffer |
| `np.rint` | numpy | round to nearest (ties to even) | vectorised |
| `gguf.GGUFReader` | optional package `gguf` (llama.cpp project) | official reader | not installed here, skipped |

## Execution trace (real run)
```
--- 1. wrote tiny.gguf: 15,584 bytes | tensor data starts at byte 736 (multiple of 32: True)
first 24 bytes: 47 47 55 46 03 00 00 00 04 00 00 00 00 00 00 00 09 00 00 00 00 00 00 00 -> magic b'GGUF' | version 3 | tensors 4 | kv 9

--- 2. dump (like `gguf-dump`) ---
GGUF v3 | tensors: 4 | metadata kv: 9 | alignment 32 | data starts at byte 736
  general.architecture             STRING   'tinyllama-demo'
  general.name                     STRING   'tiny GGUF from scratch'
  general.alignment                UINT32   32
  general.file_type                UINT32   7
  tinyllama-demo.block_count       UINT32   1
  tinyllama-demo.context_length    UINT32   128
  tinyllama-demo.rope.freq_base    FLOAT32  10000.0
  tokenizer.ggml.tokens            ARRAY    [4 items] ['<s>', '</s>', 'hello', 'world']...
  tokenizer.ggml.scores            ARRAY    [4 items] [0.0, 0.0, -1.0, -2.0]...
  token_embd.weight        dims [64, 8]    type F32   offset 0      bytes 2048
  blk.0.attn_q.weight      dims [64, 64]   type Q8_0  offset 2048   bytes 4352
  blk.0.attn_k.weight      dims [64, 64]   type F16   offset 6400   bytes 8192
  blk.0.attn_norm.weight   dims [64]       type F32   offset 14592  bytes 256

--- 3. read tensors back from the memory-mapped file ---
  token_embd.weight        F32  shape (8, 64) max abs error vs original: 0.00000
  blk.0.attn_q.weight      Q8_0 shape (64, 64) max abs error vs original: 0.00063
  blk.0.attn_k.weight      F16  shape (64, 64) max abs error vs original: 0.00006
  blk.0.attn_norm.weight   F32  shape (64,) max abs error vs original: 0.00000

--- 4. Q8_0 vs Q4_0 on 1M gaussian weights (std 0.02, like a real layer) ---
  Q8_0: 8.50 bits/weight | RMSE 0.000107 | relative to weight std 0.0054 | max abs err 0.00037
  Q4_0: 4.50 bits/weight | RMSE 0.001720 | relative to weight std 0.0860 | max abs err 0.00807

--- 5. file size estimates (params x bits per weight / 8) ---
type     bits/weight   7B (6.74B) GB   8B (8.03B) GB   note
F16            16.00            13.5            16.1   unquantized half precision (reference)
Q8_0            8.50             7.2             8.5   block 32: fp16 scale + 32 int8 = 34 B
Q6_K            6.56             5.5             6.6   k-quant, super-block 256 (16 sub-blocks of 16)
Q5_K_M          5.70             4.8             5.7   mix of Q5_K and Q6_K (approx)
Q4_K_M          4.85             4.1             4.9   mix of Q4_K and Q6_K (approx; the common default)
Q4_0            4.50             3.8             4.5   block 32: fp16 scale + 32 x 4 bit = 18 B (legacy)
IQ4_XS          4.25             3.6             4.3   i-quant: non-linear 4-bit codebook, super-block 256
Q3_K_M          3.90             3.3             3.9   mix of Q3_K and Q4_K/Q5_K (approx)
Q2_K            2.60             2.2             2.6   k-quant 2-bit mix (approx; quality drops a lot)

--- 6. optional cross-check with the `gguf` package ---
`gguf` package not installed -> skipped (our hand-written reader above is the check).
```
- The file is 15,584 bytes; the data section starts at byte 736, a multiple of 32, and each tensor offset (0, 2048, 6400, 14592) is also a multiple of 32.
- Tensor dims are reversed: numpy `(8, 64)` is stored as `[64, 8]`.
- Q8_0 on the 64x64 tensor: 4,352 bytes against 16,384 for fp32 (and 8,192 for fp16) with a max error of about 6e-4.
- On 1M Gaussian weights Q8_0 has about 0.5 percent relative RMSE and Q4_0 about 8.6 percent: four fewer bits cost real accuracy, which is why k-quants and imatrix exist.
- Q4_K_M on a 7B model is about 4.1 GB versus 13.5 GB in fp16, roughly 3.3x smaller.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| Scale `wq` std to 1.0 or 0.001 | Max error scales with the std; relative error about the same | One scale per block adapts to the local range |
| Put one outlier (5.0) in a block | That block's error jumps, others unchanged | The outlier sets the whole block's scale |
| Compare Q8_0 and Q4_0 for std 0.02 | 8.5 vs 4.5 bits, RMSE about 16x larger at 4 bits | 16 levels versus 255 |
| Add a metadata key | `metadata kv` count and data start change, data start stays a multiple of 32 | Header padding recomputed |

## Debugging practice
**BREAK IT:** delete the line `header += b"\x00" * pad_to(len(header), ALIGNMENT)` in `write_gguf`. Real run of the broken file: `tensor data starts at byte 714 (multiple of 32: False)`, the dump still lists the right dims and offsets, then reading back gives `max abs error vs original: nan` plus a numpy `RuntimeWarning` (the fp16 scales are read from the wrong bytes). Shapes look right but values are garbage. Fix: pad the header exactly as the reader rounds up. In real life the symptom is "the GGUF loads but outputs nonsense" or an out-of-bounds tensor error, so check alignment first.

Second experiment: change `wq` to shape `(64, 30)`. My toy quantizer still runs (it flattens the tensor, so blocks straddle row boundaries and the file would not be a valid GGUF layout). Real GGML requires each row length to be a multiple of the block size (32 for legacy types, 256 for k-quants), which is one reason some tensors are left in F16/F32.

## Common mistakes
- Thinking "Q4" means exactly 4 bits per weight: scales add about 0.5 bit, and `_M` is a mix.
- Quoting file size for the weights as the memory needed: the KV cache and compute buffers come on top ([step 03](../03_gpu_memory_calculator/WALKTHROUGH.md)).
- Reversing the dimension order and thinking the file is transposed.
- Believing Ollama and llama.cpp are different inference math: Ollama is the wrapper.
- Re-quantizing an already-quantized GGUF: quality loss compounds; quantize from the f16/bf16 source.
- Forgetting alignment padding when writing or parsing.

## Rebuild checklist
1. Write `w_string`, `w_value` and `pad_to`.
2. Write `write_gguf`: magic, version, counts, KV pairs, tensor infos with reversed dims and offsets, padding, data.
3. Write a `Reader` with `take` and `string`, then `read_gguf` that mmaps and returns `data_start`.
4. Implement `quantize_q8_0` with a structured dtype and check the 34-byte block size.
5. Re-compute the size table from `params x bpw / 8`.

## Say it aloud
- "GGUF is a single self-describing file: header, typed metadata including the tokenizer, tensor infos, then aligned tensor data that llama.cpp memory-maps."
- "Q8_0 is blocks of 32 with an fp16 scale, so 8.5 bits per weight; Q4_K_M uses 256-weight super-blocks and a mix of types per tensor and lands around 4.8 bits per weight."
- "A 7B model is about 13.5 GB in fp16 and roughly 4 GB at Q4_K_M, before adding the KV cache."
- "Ollama wraps llama.cpp with a registry and a REST API; the conversion path is HF safetensors, then `convert_hf_to_gguf.py`, then `llama-quantize`."

## The new problem (cliffhanger)
We now know how weights are stored and how big they are. But "how many bytes does this model plus its conversation take on my GPU" needs a formula you can do on paper: [step 03, GPU memory calculator](../03_gpu_memory_calculator/WALKTHROUGH.md). After that, the quantization math itself: [step 04](../04_quantization_math/WALKTHROUGH.md).
