# Step 04 — Quantization math from scratch

## The story so far
[Step 03](../03_gpu_memory_calculator/WALKTHROUGH.md) gave you the arithmetic of memory: weights = parameters x bytes, KV cache = 2 x layers x kv_heads x head_dim x tokens x bytes. Its table had an `int4` row, and it ended with "quantise the cache separately". The earlier lesson [05_llm_parameters/05_quantization](../../05_llm_parameters/05_quantization/WALKTHROUGH.md) showed the simplest recipe (symmetric int8, one scale per tensor, group-wise rescue for outliers). This step starts where that one stops.

## The pain point
"Just use int4" hides a dozen decisions: symmetric or asymmetric? One scale for the whole tensor, per row, per group of 32/64/128? Round to the biggest value or clip it? Integers or tiny floats (FP8, NF4, FP4)? Each answer changes the error and the real storage cost (scales are not free). Without the math you cannot read a model card that says `W4A16 g128 asym`, and you cannot answer "why NF4?" in an interview.

## The spark
Every scheme is the same formula `q = round(x/s) + z`, `x_hat = (q - z) * s`; they differ only in (1) which numbers share an `(s, z)`, (2) how the range `[lo, hi]` is chosen, and (3) the grid of representable values (uniform integers vs a non-uniform codebook). Build the formula once, then vary one thing at a time and measure.

## Goal
From scratch in numpy: affine int8/int4, symmetric vs asymmetric, per-tensor vs per-row vs per-group error tables on a real distilgpt2 weight, clipping and MSE-optimal scale, real packing of two int4 per byte, FP8 E4M3/E5M2 simulation, NF4 with double quantization, and block-scaled FP4 (MXFP4/NVFP4, emerging). Report MSE, SQNR, max error, cosine similarity and bits per weight.

## The theory (read this before the interview)
**Affine quantization** (affine map, uniform quantization, linear quantization): for a b-bit unsigned code,

```
s = (hi - lo) / (2^b - 1)          scale = real width of one step
z = round(-lo / s)                 zero-point = integer code that means exactly 0.0
q = clip(round(x / s) + z, 0, 2^b - 1)
x_hat = (q - z) * s
error per element <= s/2 inside [lo, hi]   (rounding), more if x was clipped
```

**Symmetric (absmax)**: `z = 0`, signed codes in `[-(2^(b-1)-1), 2^(b-1)-1]`, `s = max|x| / (2^(b-1)-1)`. Cheaper kernels (no zero-point term), but wastes half the grid on one-sided data. **Asymmetric (min-max)**: uses `lo`, `hi`, so it fits skewed data (post-ReLU/GELU activations); costs a zero-point per group.

**Granularity** = how many numbers share `(s, z)`:

```
per-tensor   1 scale for the whole matrix          cheapest, worst with outliers
per-channel  1 scale per output row (out_features)   standard for int8 weights
per-group    1 scale per g consecutive weights       g = 32/64/128: standard for 4-bit
             bits per weight = bits + scale_bits/g     (g=128, fp16 scale: 4 + 16/128 = 4.125)
```

**Range choice**: absmax lets one outlier stretch the grid. Clipping to a fraction c of absmax saturates the rare giants but makes the step finer for everyone else. Rules: min-max, percentile (e.g. 99.9th), or search c minimising MSE. Heavy tails and few bits both favour small c.

**Metrics**: MSE = mean(err^2); SQNR = 10 log10(sum x^2 / sum err^2) dB (+6.02 dB per extra bit for uniform data); max|err|; cosine(x, x_hat).

**FP8**: floats have sign, exponent, mantissa. E4M3 (4 exponent, 3 mantissa bits, max 448) = more precision, less range; E5M2 (max 57344) = more range, less precision. Spacing between neighbours = 2^(exponent - mantissa_bits): relative error roughly constant, so large values tolerate outliers. Engines still keep a per-tensor (or per-head/per-channel) scale.

**4-bit codebooks**: INT4 = 16 evenly spaced levels. **NF4** (NormalFloat4, QLoRA) = 16 quantiles of N(0,1) normalised to [-1,1], applied blockwise (block 64, scale = block absmax) because pretrained weights are roughly Gaussian. **Double quantization** stores the per-block fp32 scales as 8-bit numbers in blocks of 256: 4 + 8/64 + 32/(64*256) = 4.127 bits instead of 4.5. **FP4 (E2M1)** values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} and their negatives. **MXFP4** (OCP microscaling): blocks of 32 sharing a power-of-two (E8M0) scale. **NVFP4** (NVIDIA Blackwell): blocks of 16 sharing an FP8-E4M3 scale plus one FP32 per-tensor scale. MXFP4/NVFP4 are emerging, hardware-accelerated on the newest GPUs; weight-only INT4 (GPTQ/AWQ) and FP8 are current mainstream practice.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Scale `s` | step size, delta, quantization step | Yes | Real value of one integer step. |
| Zero-point `z` | offset, bias (quantization), `zp` | Yes | Integer that decodes to 0.0; 0 in symmetric schemes. |
| Symmetric quantization | absmax, signed, zero-point-free | Yes | Range `[-m, m]`. |
| Asymmetric quantization | min-max, affine, zero-point quantization | Mostly | Range `[min, max]`; needs `z`. |
| Per-channel | per-row, per-output-channel, channel-wise | Yes for weights | One scale per output row (careful: for activations "channel" means a column). |
| Per-group | group-wise, block-wise, block quantization | Yes | Groups of g consecutive weights; "block" is the llama.cpp / bitsandbytes word. |
| Fake quantization | simulated quantization, quantize-dequantize (QDQ) | Yes | Round then convert back to float so error can be measured; no integer kernel used. |
| SQNR | signal-to-quantization-noise ratio, SNR | Yes | `10 log10(signal energy / error energy)` in dB. |
| FP8 E4M3 / E5M2 | float8 formats (OCP FP8) | Related | Same 8 bits, different split between range and precision. |
| NF4 | NormalFloat4, QLoRA 4-bit | Yes | 16 normal quantiles, not uniform. |
| MXFP4 / NVFP4 | microscaling FP4 / NVIDIA FP4 | Related | Both block-scaled FP4; block 32 + E8M0 scale vs block 16 + E4M3 scale. |
| bpw | bits per weight, effective bit width | Yes | Includes scale (and zero-point) overhead. |

## Run it
```bash
cd 21_llm_inference_engineering/04_quantization_math
python main.py
# or from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/04_quantization_math/main.py
```
Runtime about 10 s of CPU once the model file is read; about 0.6 GB RAM (measured peak 625 MB). Only the real-weight test loads distilgpt2 (cached); without it a synthetic matrix is used. Timings vary by machine (and wait time if other jobs hold the shared lock).

## Code, step by step
All reusable pieces (`qdq`, `metrics`, `pack_int4`, `fp8_qdq`, `nf4_qdq`, `fp4_block_qdq`, ...) live in [`../quantlib.py`](../quantlib.py), which steps 05-07 import too. `main.py` is the experiment script.

### 1. Header, imports, the shared helper module, and the report row  (`main.py` lines 1–33)

```python
"""
GOAL:  Build every common quantization scheme from scratch in numpy (affine int8/int4, per-tensor/row/group, clipping,
       packed int4, FP8, NF4, block-scaled FP4) and compare them with MSE / SQNR / max error / cosine and bits-per-weight.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/04_quantization_math/main.py
EXPECTED OUTPUT (shortened):
    asymmetric beats symmetric on post-ReLU data (int4 SQNR ~ +6 dB), about equal on a Gaussian
    granularity table (int4, real distilgpt2 mlp.c_fc): per-tensor worst ... g=32 best; bpw 4.0 -> 4.5
    int4 packing: 1,000,000 values -> 500,000 bytes, unpack == original: True
    FP8 E4M3 vs INT8: Gaussian int8 wins, heavy-tailed data E4M3/E5M2 win
    4-bit families on a real weight: NF4 / FP4-block / int4-g64 SQNR side by side
NEEDS: numpy, scipy (NF4 quantiles), torch+transformers only for the real-weight test (distilgpt2 cached; skipped if missing).  ~10 s on CPU.

Prerequisite: 05_llm_parameters/05_quantization (symmetric int8 + group-wise idea).  This step goes deeper: the general
affine map, WHICH granularity, HOW to pick the range, and the 4-bit/8-bit FLOATING-point alternatives.
All code lives in ../quantlib.py (shared with steps 05-07); this file is the experiment script.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quantlib as ql          # our helpers: qdq, metrics, pack_int4, fp8_qdq, nf4_qdq, fp4_block_qdq, bits_per_weight

rng = np.random.default_rng(0)


def row(name, x, xh, extra=""):
    """Print one line of the standard 4-metric report."""
    m = ql.metrics(x, xh)
    print(f"  {name:<26} MSE {m['mse']:.3e} | SQNR {m['sqnr_db']:6.2f} dB | max|err| {m['max_abs']:.4f} | cos {m['cos']:.5f} {extra}")
```

- **In simple words:** We import our small shared library `quantlib.py` (so 04-07 do not duplicate code) and define one function that prints the four standard error numbers.
- **Exactly:** `sys.path.insert(0, '..')` makes `import quantlib` work when the script is started from its own folder or from the repo root. `row()` calls `ql.metrics(x, xh)` which returns a dict: MSE = mean((x-xh)^2); SQNR (signal-to-quantization-noise ratio) = 10*log10(sum x^2 / sum err^2) in dB, so +6 dB is one extra bit; max|err| is the worst single element; cos is cosine similarity of the flattened tensors.
- **Why it is here:** Quantization quality is only meaningful when measured; four metrics catch different failures (a few huge errors show in max|err| but not in MSE, direction loss shows in cosine).

### 2. The affine map on six numbers  (`main.py` lines 36–44)

```python
# [[1: the affine map on 6 numbers]]
# q = round(x / s) + z   and   x_hat = (q - z) * s.   s = scale (real step between two integer codes), z = zero-point (the integer that means 0.0).
print("== 1. Affine quantization by hand (uint8, asymmetric) ==")
x = np.array([-0.4, 0.0, 0.3, 1.1, 2.6, 3.0], dtype=np.float32)
lo, hi = min(x.min(), 0.0), max(x.max(), 0.0)           # the range must contain 0 so that 0.0 survives exactly
s = (hi - lo) / 255                                      # 256 codes -> 255 steps
z = int(round(-lo / s))                                  # zero-point: code that decodes to 0.0
q = np.clip(np.round(x / s) + z, 0, 255).astype(np.uint8)
print(f"  s={s:.5f} z={z} q={q.tolist()}  x_hat={np.round((q.astype(np.int32) - z) * s, 4).tolist()}")
```

- **In simple words:** Quantizing means choosing a step size s and a zero-point z, then storing integers; we do it by hand on 6 numbers.
- **Exactly:** `s = (hi - lo) / 255` is the real distance between two adjacent uint8 codes. `z = round(-lo/s)` is the integer code that decodes to exactly 0.0. `q = clip(round(x/s) + z, 0, 255)` is stored as `uint8`. Decoding is `(q - z) * s`; the `.astype(np.int32)` avoids uint8 wrap-around on the subtraction.
- **Why it is here:** Every scheme in this step is a variation of this one formula: what is `s`, what is `z`, what is the integer range, and which values share the same `(s, z)`.

### 3. Symmetric vs asymmetric  (`main.py` lines 46–55)

```python
# [[2: symmetric (absmax) vs asymmetric (min-max)]]
# Symmetric wastes half of the integer range when the data is one-sided, e.g. activations AFTER ReLU/GELU.
print("\n== 2. Symmetric vs asymmetric, int4 and int8 ==")
gauss = rng.normal(0, 1, (256, 256)).astype(np.float32)             # centred data: both schemes fit
relu = np.maximum(rng.normal(0, 1, (256, 256)), 0).astype(np.float32)   # post-ReLU: all >= 0, about half exactly 0
for name, data in (("Gaussian", gauss), ("post-ReLU (skewed)", relu)):
    print(f" {name}:")
    for bits in (8, 4):
        row(f"int{bits} symmetric", data, ql.qdq(data, bits, True))
        row(f"int{bits} asymmetric", data, ql.qdq(data, bits, False))
```

- **In simple words:** Compare two ways of choosing the range on centred data and on post-ReLU data.
- **Exactly:** `ql.qdq(x, bits, symmetric, gran)` quantizes and de-quantizes a 2-D array. Symmetric uses `s = max|x|/(2^(b-1)-1)` and no zero-point (zero-point 0); asymmetric uses min and max and a zero-point. After ReLU all values are >= 0, so symmetric never uses its negative half: it effectively has one bit fewer.
- **Why it is here:** Explains why activation quantizers (post-GELU/ReLU, skewed) often use asymmetric ranges while weights (roughly centred) use symmetric: symmetric is cheaper in kernels because there is no zero-point term in the matmul.

### 4. Granularity: per-tensor, per-row, per-group on a real weight  (`main.py` lines 57–75)

```python
# [[3: granularity - one scale for what?]]
# A real weight matrix when available (distilgpt2 block 3 mlp.c_fc, 768 x 3072); else a synthetic one with an outlier channel.
print("\n== 3. Granularity at int4 (symmetric): per-tensor / per-row / per-group ==")
try:
    tok, model = ql.load_distilgpt2()
    layer = dict(ql.conv1d_layers(model))["transformer.h.3.mlp.c_fc"]
    W = ql.get_w_out_in(layer)                                       # (out=3072, in=768): nn.Linear layout; groups run along 'in'
    src = "real distilgpt2 h.3.mlp.c_fc"
except Exception as e:                                               # no model cached: synthetic fallback
    print("  (distilgpt2 unavailable, synthetic weight used:", type(e).__name__, ")")
    W = rng.normal(0, 0.05, (3072, 768)).astype(np.float32)
    W[:, 7] *= 12
    model = None
    src = "synthetic"
print(f"  weight: {src}, shape {W.shape}, absmax {np.abs(W).max():.3f}, std {W.std():.3f}")
for gran in ("tensor", "row", 128, 64, 32):
    bpw = ql.bits_per_weight(4, None if gran in ("tensor", "row") else gran)      # fp16 scale per group
    row(f"int4 {gran}", W, ql.qdq(W, 4, True, gran), f"| bpw {bpw:.3f}")
row("int4 g=64 asymmetric", W, ql.qdq(W, 4, False, 64), f"| bpw {ql.bits_per_weight(4, 64, 16, 4):.3f} (scale+4-bit zero)")
```

- **In simple words:** The smaller the set of numbers sharing one scale, the smaller the error, but the more scales you must store.
- **Exactly:** `ql.load_distilgpt2()` returns the cached tokenizer and model; `ql.conv1d_layers(model)` lists GPT-2's `Conv1D` layers (their weight is stored (in, out)); `ql.get_w_out_in` returns it as (out, in), the nn.Linear layout. `gran='row'` gives one scale per output channel, `gran=g` reshapes to (n/g, g) so every g consecutive weights along the input dimension share a scale. `bits_per_weight` adds `16/g` bits for an fp16 scale. If the model is not cached, a synthetic matrix with an outlier column is used.
- **Why it is here:** This is the central trade-off of every 4-bit format (GPTQ, AWQ, llama.cpp Q4_K, NF4 all use groups of 32-128); the table shows error and bpw side by side.

### 5. Clipping and the MSE-optimal scale  (`main.py` lines 77–85)

```python
# [[4: choosing the range - clipping and MSE-optimal scale]]
# absmax lets ONE outlier set the step for everybody.  Clipping saturates the outlier but shrinks the step for all others.
print("\n== 4. Clipping: search the clip factor that minimises MSE (int4, per-tensor) ==")
heavy = (rng.standard_t(3, (256, 256)) * 0.02).astype(np.float32)   # Student-t with 3 degrees of freedom: heavy tails
for name, data in (("Gaussian", gauss * 0.02), ("heavy-tailed t3", heavy), ("real weight", W)):
    c, best, at1 = ql.mse_optimal_clip(data, 4, True, "tensor")
    p99 = np.percentile(np.abs(data), 99.9) / np.abs(data).max()     # 'percentile' range: clip at the 99.9th percentile of |x|
    pm = ql.metrics(data, ql.qdq(data, 4, True, "tensor", p99))["mse"]
    print(f"  {name:<16} absmax MSE {at1:.3e} | percentile(99.9) clip={p99:.2f} MSE {pm:.3e} | MSE-search clip={c:.2f} MSE {best:.3e}  ({at1 / best:.1f}x better)")
```

- **In simple words:** Instead of stretching the grid to the biggest value, sometimes it is better to saturate the rare giants and use a finer step for everyone else.
- **Exactly:** `ql.mse_optimal_clip` tries 36 clip factors c from 0.3 to 1.0 (range = c*absmax) and returns the c with the smallest reconstruction MSE. The percentile variant clips at the 99.9th percentile of |x|. Heavy-tailed data (Student-t, df=3) has rare huge values so clipping helps a lot more than on Gaussian data.
- **Why it is here:** Range selection is the whole job of 'calibration' in step 05: min-max, percentile and MSE are the three standard rules.

### 6. Packing two 4-bit codes into one byte  (`main.py` lines 87–93)

```python
# [[5: packing two 4-bit values into one byte]]
print("\n== 5. int4 packing ==")
big = rng.integers(-7, 8, size=(1000, 1000)).astype(np.int8)        # a million 4-bit codes held in int8 (1 byte each)
packed = ql.pack_int4(big)                                           # shape (1000, 500) uint8: two codes per byte
back = ql.unpack_int4(packed)
print(f"  {big.size:,} values: int8 container {big.nbytes:,} B -> packed {packed.nbytes:,} B; unpack == original: {bool((back == big).all())}")
print(f"  example byte: codes {big[0,0]},{big[0,1]} -> byte 0x{packed[0,0]:02X} (low nibble {packed[0,0] & 15}, high nibble {packed[0,0] >> 4}; stored value = code + 8)")
```

- **In simple words:** There is no 4-bit dtype in numpy, so real engines store two codes per byte; we pack and unpack and prove nothing is lost.
- **Exactly:** Codes in [-7, 7] get +8 to become 1..15 (a nibble). `pack_int4` stores the even-index code in the low nibble (`& 0x0F`) and the odd-index code in the high nibble (`<< 4`). `unpack_int4` masks and shifts back and subtracts 8. `np.array_equal`-style comparison `(back == big).all()` proves a lossless round trip and `packed.nbytes` is exactly half.
- **Why it is here:** The memory saving of 4-bit is only real if the bits are really packed; kernels (Marlin, ExLlama, llama.cpp) unpack inside the matmul so the bandwidth saved is what makes decoding faster.

### 7. FP8 (E4M3, E5M2) vs INT8  (`main.py` lines 95–110)

```python
# [[6: FP8 (E4M3, E5M2) vs INT8]]
# Floating point has NON-uniform spacing: tiny values get fine steps, big values coarse steps (relative error ~ constant).
print("\n== 6. FP8 simulation vs int8 (per-tensor scaling) ==")
try:                                                                 # cross-check our rounding against torch's real fp8 dtype
    import torch
    t = np.clip(rng.normal(0, 100, 5000), -448, 448).astype(np.float32)
    ref = torch.from_numpy(t).to(torch.float8_e4m3fn).float().numpy()
    print(f"  our E4M3 rounding == torch.float8_e4m3fn on 5000 values: {bool(np.array_equal(ql.fp_round(t, 'e4m3'), ref))}")
except Exception as e:
    print("  (torch fp8 cross-check skipped:", type(e).__name__, ")")
print("  representable E4M3 values near 1.0:", np.unique(ql.fp_round(np.linspace(0.9, 1.3, 200), "e4m3")).tolist()[:6], "(3 mantissa bits -> 8 steps per octave)")
for name, data in (("Gaussian", gauss), ("heavy-tailed t3", heavy / 0.02)):
    print(f" {name}:")
    row("int8 per-tensor", data, ql.qdq(data, 8, True))
    row("FP8 E4M3 per-tensor", data, ql.fp8_qdq(data, "e4m3"))
    row("FP8 E5M2 per-tensor", data, ql.fp8_qdq(data, "e5m2"))
```

- **In simple words:** Floating-point numbers are dense near zero and sparse for large values; integers are evenly spaced.
- **Exactly:** `ql.fp_round(x, fmt)` rounds to the nearest value of the format: spacing = 2^(exponent - mantissa_bits), exponent fixed at the minimum for subnormals, saturating at 448 (E4M3) or 57344 (E5M2). We check it matches `torch.float8_e4m3fn` exactly. `fp8_qdq` first scales so absmax maps to the format maximum (the per-tensor scale an FP8 engine keeps).
- **Why it is here:** FP8 W8A8 is the mainstream fast path on Hopper/Ada/Blackwell GPUs; the comparison shows FP8 tolerates outliers better (more dynamic range) while int8 is better for well-behaved Gaussian data (more mantissa precision).

### 8. NF4, FP4 and block-scaled FP4 (MXFP4, NVFP4)  (`main.py` lines 112–124)

```python
# [[7: NF4, FP4 and block-scaled FP4 (MXFP4 / NVFP4)]]
# NF4: 16 levels = quantiles of a normal distribution (weights are ~Gaussian) + blockwise absmax.  MXFP4 / NVFP4: FP4 (E2M1)
# values {0,.5,1,1.5,2,3,4,6} times a per-block scale.  MXFP4 / NVFP4 are EMERGING (hardware-supported on newest GPUs).
print("\n== 7. 4-bit families on the same weight ==")
print("  NF4 levels:", np.round(ql.nf4_levels(), 3).tolist())
flat = W.reshape(-1, 64)[: (W.size // 64) // 256 * 256].reshape(-1)  # trim so shapes divide evenly for double quantization
Wt = flat.reshape(-1, 64)
row("int4 symmetric g=64", Wt, ql.qdq(Wt, 4, True, "row"), f"| bpw {ql.bits_per_weight(4, 64):.3f}")
row("int4 asymmetric g=64", Wt, ql.qdq(Wt, 4, False, "row"), f"| bpw {ql.bits_per_weight(4, 64, 16, 4):.3f}")
row("NF4 block 64", Wt, ql.nf4_qdq(Wt, 64), f"| bpw {ql.bits_per_weight(4, 64):.3f}")
row("NF4 + double quant", Wt, ql.nf4_qdq(Wt, 64, True), f"| bpw {4 + 8 / 64 + 32 / (64 * 256):.3f}")
row("MXFP4 (block 32, E8M0)", Wt, ql.fp4_block_qdq(Wt, 32, "e8m0"), f"| bpw {ql.bits_per_weight(4, 32, 8):.3f}")
row("NVFP4 (block 16, E4M3)", Wt, ql.fp4_block_qdq(Wt, 16, "e4m3"), f"| bpw {ql.bits_per_weight(4, 16, 8):.3f} (+tiny per-tensor scale)")
```

- **In simple words:** Different 16-value codebooks for 4 bits: uniform ints, normal-distribution quantiles (NF4), tiny floats (FP4).
- **Exactly:** `ql.nf4_levels()` takes 16 quantiles of N(0,1) (using `scipy.stats.norm.ppf`) so that each level covers the same probability mass, and rescales to [-1,1]; `ql.nf4_qdq` divides each block of 64 by its absmax and snaps to the nearest level (via midpoints and `np.searchsorted`). With `double_quant=True` the per-block scales are themselves stored as int8. `fp4_block_qdq` uses the E2M1 values {0, .5, 1, 1.5, 2, 3, 4, 6} with a block scale: MXFP4 = 32 values per power-of-two (E8M0) scale, NVFP4 = 16 values per FP8-E4M3 scale plus a tensor scale. These two are emerging hardware formats.
- **Why it is here:** Interview questions ask 'what is NF4' and 'what is MXFP4/NVFP4': both are 'a better 16-point grid plus a block scale'; the table lets you compare them numerically.

### 9. Bits per weight including scales  (`main.py` lines 126–130)

```python
# [[8: bits per weight including scales]]
print("\n== 8. Bits per weight = bits + scale bits / group ==")
for name, bits, g, sb in (("int8 per-row", 8, 4096, 16), ("int4 g=128 fp16 scale", 4, 128, 16), ("int4 g=64 fp16 scale", 4, 64, 16),
                          ("int4 g=32 fp16 scale", 4, 32, 16), ("int3 g=128", 3, 128, 16), ("MXFP4 (e8m0/32)", 4, 32, 8)):
    print(f"  {name:<24} {ql.bits_per_weight(bits, g, sb):.4f} bpw  -> 7B params ~ {7e9 * ql.bits_per_weight(bits, g, sb) / 8 / 1e9:.2f} GB")
```

- **In simple words:** The real cost of 4-bit is a bit more than 4 bits because every group stores a scale.
- **Exactly:** `bits + scale_bits/group`: g=128 with an fp16 scale gives 4.125 bpw; model size = params * bpw / 8 bytes.
- **Why it is here:** Gives the correct size estimates for 'will a 70B model fit' questions (links to ../03_gpu_memory_calculator/WALKTHROUGH.md).

### 10. TRY IT / BREAK IT  (`main.py` lines 132–136)

```python
# TRY IT: in section 3 change the layer to 'transformer.h.0.attn.c_attn' - does per-tensor still lose by as much?
# TRY IT: in section 4 use bits=8 - how much does clipping still matter at 256 levels?
# TRY IT: in section 2 scale `relu` by 100 - do asymmetric and symmetric SQNR change? (They should not: scale invariance.)
# BREAK IT: in qdq() drop the 'range must contain 0' min/max with 0.0 and quantize data that is all positive and far from 0 - zero is no longer exact.
# BREAK IT: in pack_int4 forget the +8 offset; negative codes wrap to 252..255 and unpack gives garbage. Print the round-trip check.
```

- **In simple words:** Experiments to run after reading.
- **Exactly:** Comments only; no code runs.
- **Why it is here:** Changing one thing at a time is how you build intuition for which knob matters.


## Syntax explained

| Syntax | Meaning |
|---|---|
| `sys.path.insert(0, ".../..")` | Put the parent folder first on the import path so `import quantlib` finds `21_llm_inference_engineering/quantlib.py`. |
| `x.reshape(-1, g)` | Re-view a matrix as rows of g numbers; `-1` means "infer this dimension"; each row becomes one scale group. |
| `np.clip(a, lo, hi)` | Force values into `[lo, hi]` (saturation). |
| `(p & 0x0F)`, `(u << 4)` | Bit mask / shift: take the low 4 bits; move 4 bits up. `\|` combines two nibbles into a byte. |
| `f"{v:.3e}"`, `f"{v:6.2f}"` | Scientific notation / fixed width floating format. |
| `try: ... except Exception` | Fall back to a synthetic matrix when the model is not available. |
| `np.searchsorted(mid, v)` | Index of the interval each value falls in; with midpoints between sorted levels it gives the nearest level (used in `quantlib`). |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `ql.qdq(x, bits, symmetric, gran, clip)` | Quantize then de-quantize a 2-D array | Reshape to groups, per-group scale (and zero), `round`, `clip`, multiply back |
| `ql.metrics(x, xh)` | MSE, SQNR dB, max error, cosine | A few numpy reductions in float64 |
| `ql.pack_int4 / unpack_int4` | Two 4-bit codes per byte | Offset by +8, shift/or, mask/shift back |
| `ql.fp_round(x, fmt)` | Round to FP8 E4M3/E5M2 values | `floor(log2(abs(x)))` gives the exponent, step = 2^(e - mantissa_bits), `round(x/step)*step`, saturate |
| `ql.nf4_levels()` | The 16 NF4 levels | `scipy.stats.norm.ppf` of evenly spaced probabilities (asymmetric so that 0 is a level), scaled to [-1, 1] |
| `ql.fp4_block_qdq` | MXFP4 / NVFP4 simulation | Per block scale (power of two or FP8), nearest of the 15 FP4 values |
| `torch.from_numpy(a).to(torch.float8_e4m3fn)` | PyTorch's real FP8 conversion | Used only to confirm our rounding is bit-identical |

## Execution trace (real run)
```
== 1. Affine quantization by hand (uint8, asymmetric) ==
  s=0.01333 z=30 q=[0, 30, 52, 112, 225, 255]  x_hat=[-0.4, 0.0, 0.2933, 1.0933, 2.6, 3.0]

== 2. Symmetric vs asymmetric, int4 and int8 ==
 Gaussian:
  int8 symmetric             MSE 1.164e-04 | SQNR  39.34 dB | max|err| 0.0186 | cos 0.99994 
  int8 asymmetric            MSE 1.086e-04 | SQNR  39.63 dB | max|err| 0.0181 | cos 0.99995 
  int4 symmetric             MSE 3.806e-02 | SQNR  14.19 dB | max|err| 0.3380 | cos 0.98147 
  int4 asymmetric            MSE 3.149e-02 | SQNR  15.01 dB | max|err| 0.3075 | cos 0.98459 
 post-ReLU (skewed):
  int8 symmetric             MSE 4.669e-05 | SQNR  40.29 dB | max|err| 0.0168 | cos 0.99995 
  int8 asymmetric            MSE 1.158e-05 | SQNR  46.35 dB | max|err| 0.0084 | cos 0.99999 
  int4 symmetric             MSE 1.540e-02 | SQNR  15.11 dB | max|err| 0.3048 | cos 0.98493 
  int4 asymmetric            MSE 3.371e-03 | SQNR  21.71 dB | max|err| 0.1422 | cos 0.99664 

== 3. Granularity at int4 (symmetric): per-tensor / per-row / per-group ==
  weight: real distilgpt2 h.3.mlp.c_fc, shape (3072, 768), absmax 1.207, std 0.127
  int4 tensor                MSE 2.482e-03 | SQNR   8.11 dB | max|err| 0.0862 | cos 0.93065 | bpw 4.000
  int4 row                   MSE 3.167e-04 | SQNR  17.06 dB | max|err| 0.0861 | cos 0.99030 | bpw 4.000
  int4 128                   MSE 2.230e-04 | SQNR  18.58 dB | max|err| 0.0854 | cos 0.99314 | bpw 4.125
  int4 64                    MSE 1.875e-04 | SQNR  19.33 dB | max|err| 0.0854 | cos 0.99422 | bpw 4.250
  int4 32                    MSE 1.521e-04 | SQNR  20.24 dB | max|err| 0.0854 | cos 0.99531 | bpw 4.500
  int4 g=64 asymmetric       MSE 1.334e-04 | SQNR  20.81 dB | max|err| 0.0507 | cos 0.99588 | bpw 4.312 (scale+4-bit zero)

== 4. Clipping: search the clip factor that minimises MSE (int4, per-tensor) ==
  Gaussian         absmax MSE 1.522e-05 | percentile(99.9) clip=0.69 MSE 7.374e-06 | MSE-search clip=0.52 MSE 5.214e-06  (2.9x better)
  heavy-tailed t3  absmax MSE 8.118e-04 | percentile(99.9) clip=0.17 MSE 1.799e-04 | MSE-search clip=0.30 MSE 3.226e-04  (2.5x better)
  real weight      absmax MSE 2.482e-03 | percentile(99.9) clip=0.38 MSE 3.545e-04 | MSE-search clip=0.30 MSE 2.525e-04  (9.8x better)

== 5. int4 packing ==
  1,000,000 values: int8 container 1,000,000 B -> packed 500,000 B; unpack == original: True
  example byte: codes -4,7 -> byte 0xF4 (low nibble 4, high nibble 15; stored value = code + 8)

== 6. FP8 simulation vs int8 (per-tensor scaling) ==
  our E4M3 rounding == torch.float8_e4m3fn on 5000 values: True
  representable E4M3 values near 1.0: [0.875, 0.9375, 1.0, 1.125, 1.25] (3 mantissa bits -> 8 steps per octave)
 Gaussian:
  int8 per-tensor            MSE 1.164e-04 | SQNR  39.34 dB | max|err| 0.0186 | cos 0.99994 
  FP8 E4M3 per-tensor        MSE 6.934e-04 | SQNR  31.59 dB | max|err| 0.1689 | cos 0.99965 
  FP8 E5M2 per-tensor        MSE 2.797e-03 | SQNR  25.53 dB | max|err| 0.3375 | cos 0.99860 
 heavy-tailed t3:
  int8 per-tensor            MSE 2.921e-02 | SQNR  20.11 dB | max|err| 0.2965 | cos 0.99516 
  FP8 E4M3 per-tensor        MSE 2.142e-03 | SQNR  31.46 dB | max|err| 2.0636 | cos 0.99964 
  FP8 E5M2 per-tensor        MSE 7.918e-03 | SQNR  25.78 dB | max|err| 3.6243 | cos 0.99868 

== 7. 4-bit families on the same weight ==
  NF4 levels: [-1.0, -0.6959999799728394, -0.5249999761581421, -0.39500001072883606, -0.2840000092983246, -0.1850000023841858, -0.09099999815225601, 0.0, 0.07999999821186066, 0.16099999845027924, 0.2460000067949295, 0.33799999952316284, 0.4410000145435333, 0.5630000233650208, 0.7229999899864197, 1.0]
  int4 symmetric g=64        MSE 1.875e-04 | SQNR  19.33 dB | max|err| 0.0854 | cos 0.99422 | bpw 4.250
  int4 asymmetric g=64       MSE 1.334e-04 | SQNR  20.81 dB | max|err| 0.0507 | cos 0.99588 | bpw 4.312
  NF4 block 64               MSE 1.372e-04 | SQNR  20.69 dB | max|err| 0.1073 | cos 0.99573 | bpw 4.250
  NF4 + double quant         MSE 1.372e-04 | SQNR  20.69 dB | max|err| 0.1073 | cos 0.99573 | bpw 4.127
  MXFP4 (block 32, E8M0)     MSE 2.132e-04 | SQNR  18.78 dB | max|err| 0.2071 | cos 0.99338 | bpw 4.250
  NVFP4 (block 16, E4M3)     MSE 1.453e-04 | SQNR  20.44 dB | max|err| 0.1212 | cos 0.99548 | bpw 4.500 (+tiny per-tensor scale)

== 8. Bits per weight = bits + scale bits / group ==
  int8 per-row             8.0039 bpw  -> 7B params ~ 7.00 GB
  int4 g=128 fp16 scale    4.1250 bpw  -> 7B params ~ 3.61 GB
  int4 g=64 fp16 scale     4.2500 bpw  -> 7B params ~ 3.72 GB
  int4 g=32 fp16 scale     4.5000 bpw  -> 7B params ~ 3.94 GB
  int3 g=128               3.1250 bpw  -> 7B params ~ 2.73 GB
  MXFP4 (e8m0/32)          4.2500 bpw  -> 7B params ~ 3.72 GB
```

How to read it:
- **Asymmetric wins on one-sided data**: post-ReLU int4 goes from 15.11 dB (symmetric) to 21.71 dB (asymmetric), about one extra bit; on centred Gaussian data it is only +0.8 dB. Symmetric wastes the negative half of its grid on data that has no negatives.
- **Granularity is the big lever on a real weight**: int4 per-tensor 8.11 dB, per-row 17.06, g=128 18.58, g=64 19.33, g=32 20.24 dB, costing 4.0 -> 4.125 -> 4.25 -> 4.5 bits per weight. The max|err| barely moves (0.085) because it is set by the one largest weight; groups reduce the AVERAGE error. Group size is a quality-versus-size dial, which is why 4.125 bpw (g=128) is the common default.
- **Clipping**: on the real weight the MSE-optimal clip is 0.30 of absmax (grid edge) and cuts MSE about 9.8x; even Gaussian data gains 2.9x at int4. Absmax is almost never MSE-optimal at 4 bits.
- **FP8 vs int8**: on Gaussian data int8 is 7.8 dB better (39.3 vs 31.6) because it has 7 mantissa-like bits; on the heavy-tailed data E4M3 beats int8 by 11 dB (31.5 vs 20.1) because int8's step is dictated by the rare giants. Note E4M3's max|err| is larger in absolute terms (it is big where the values are big) while its cosine is higher.
- **4-bit families** at about 4.25 bpw on the same weight: asymmetric int4 g64 20.81 dB, NF4 20.69, NVFP4 (block 16) 20.44, symmetric int4 g64 19.33, MXFP4 (block 32, power-of-two scale) 18.78. Double quantization changed the error in no visible digit but saves 0.12 bpw (4.25 -> 4.127). The coarse power-of-two scale of MXFP4 costs about 1.5 dB against NVFP4's finer scale.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| Section 3: use layer `transformer.h.0.attn.c_attn` | Per-tensor vs group gap differs by layer | Outlier structure varies per layer. |
| Section 4: `bits=8` | Clipping gain falls to about 1.2x | 256 levels already resolve the bulk; there is less to gain. |
| Section 2: multiply `relu` by 100 | SQNR unchanged | Quantization error scales with the data (scale invariance). |
| Section 4: widen the grid below 0.3 | Heavy-tailed int4 optimum moves below 0.3 | The search hit the grid edge. |
| Section 7: use `block=32` for NF4 | Slightly better, 0.5 bpw more scale overhead | Smaller blocks adapt better. |

## Debugging practice
**BREAK IT:** in `pack_int4` remove the `+ 8` offset. Negative codes become huge uint8 values, the high nibble overflows into garbage, and the check prints `unpack == original: False`. Fix: offset signed codes into 0..15 before packing, subtract after unpacking.
**BREAK IT:** in `qdq` (asymmetric branch) drop the `minimum(.., 0.0)` / `maximum(.., 0.0)` lines and quantize data that is all positive and far from zero. The range no longer contains 0, so `z` is out of `[0, 2^b-1]` after clipping and zero decodes to a non-zero number (padding and ReLU zeros become noise). Fix: always include 0 in the range.

## Common mistakes
- Reporting only MSE; a good MSE can hide large max errors on the important outliers.
- Comparing methods at different bits-per-weight without counting scales (g=32 is 4.5 bpw, not 4).
- Using symmetric int8 for one-sided activations and concluding "int8 is bad".
- Forgetting that `Conv1D` in GPT-2 stores weights as `(in, out)`; grouping along the wrong axis changes the result.
- Believing FP8 is "more accurate than int8": it is more robust to outliers, less precise on well-behaved data.
- Treating MXFP4/NVFP4 as widely deployed; they are emerging and hardware-specific.

## Rebuild checklist
1. Write `s, z`, quantize and dequantize with the affine formula on a toy vector.
2. Implement `qdq` for symmetric and asymmetric with groups via `reshape(-1, g)`.
3. Write `metrics` (MSE, SQNR dB, max, cosine).
4. Add the clip factor and a search over 0.3..1.0.
5. Write `pack_int4` / `unpack_int4` with the +8 offset; test on 1000x1000.
6. Write `fp_round` (exponent via `log2`, spacing `2^(e - m)`, saturate) and test against `torch.float8_e4m3fn`.
7. Build the NF4 levels from `norm.ppf`, snap with midpoints, blockwise absmax.
8. Compute bits per weight for g=128 fp16 scale (4.125).

## Say it aloud
- "Every quantizer is `q = round(x/s) + z`; what differs is what shares a scale, how the range is picked, and which grid of values is allowed."
- "On a real distilgpt2 matrix, int4 per-tensor gives 8 dB, per-row 17 dB, group 128 gives 18.6 dB for 4.125 bits per weight, so group size is the quality-versus-size dial."
- "Symmetric wastes half its levels on post-ReLU data; asymmetric fixes it with a zero-point at the cost of a slightly more complex kernel."
- "FP8 E4M3 beats int8 on heavy-tailed data and loses on Gaussian data; NF4 is a normal-quantile codebook with blockwise absmax; MXFP4 and NVFP4 are block-scaled FP4 and still emerging."

## The new problem (cliffhanger)
Everything here quantized weights, which are static and tame (per-input-channel absmax max/median of only 2.3 in the layer we used). The numbers flowing through the network, the activations, change with every input and contain outlier channels. Do they survive int8 too, and how do you choose their ranges before seeing user traffic? That is [step 05: activation quantization and calibration](../05_activation_quant_and_calibration/WALKTHROUGH.md).
