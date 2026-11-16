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


# [[1: the affine map on 6 numbers]]
# q = round(x / s) + z   and   x_hat = (q - z) * s.   s = scale (real step between two integer codes), z = zero-point (the integer that means 0.0).
print("== 1. Affine quantization by hand (uint8, asymmetric) ==")
x = np.array([-0.4, 0.0, 0.3, 1.1, 2.6, 3.0], dtype=np.float32)
lo, hi = min(x.min(), 0.0), max(x.max(), 0.0)           # the range must contain 0 so that 0.0 survives exactly
s = (hi - lo) / 255                                      # 256 codes -> 255 steps
z = int(round(-lo / s))                                  # zero-point: code that decodes to 0.0
q = np.clip(np.round(x / s) + z, 0, 255).astype(np.uint8)
print(f"  s={s:.5f} z={z} q={q.tolist()}  x_hat={np.round((q.astype(np.int32) - z) * s, 4).tolist()}")

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

# [[4: choosing the range - clipping and MSE-optimal scale]]
# absmax lets ONE outlier set the step for everybody.  Clipping saturates the outlier but shrinks the step for all others.
print("\n== 4. Clipping: search the clip factor that minimises MSE (int4, per-tensor) ==")
heavy = (rng.standard_t(3, (256, 256)) * 0.02).astype(np.float32)   # Student-t with 3 degrees of freedom: heavy tails
for name, data in (("Gaussian", gauss * 0.02), ("heavy-tailed t3", heavy), ("real weight", W)):
    c, best, at1 = ql.mse_optimal_clip(data, 4, True, "tensor")
    p99 = np.percentile(np.abs(data), 99.9) / np.abs(data).max()     # 'percentile' range: clip at the 99.9th percentile of |x|
    pm = ql.metrics(data, ql.qdq(data, 4, True, "tensor", p99))["mse"]
    print(f"  {name:<16} absmax MSE {at1:.3e} | percentile(99.9) clip={p99:.2f} MSE {pm:.3e} | MSE-search clip={c:.2f} MSE {best:.3e}  ({at1 / best:.1f}x better)")

# [[5: packing two 4-bit values into one byte]]
print("\n== 5. int4 packing ==")
big = rng.integers(-7, 8, size=(1000, 1000)).astype(np.int8)        # a million 4-bit codes held in int8 (1 byte each)
packed = ql.pack_int4(big)                                           # shape (1000, 500) uint8: two codes per byte
back = ql.unpack_int4(packed)
print(f"  {big.size:,} values: int8 container {big.nbytes:,} B -> packed {packed.nbytes:,} B; unpack == original: {bool((back == big).all())}")
print(f"  example byte: codes {big[0,0]},{big[0,1]} -> byte 0x{packed[0,0]:02X} (low nibble {packed[0,0] & 15}, high nibble {packed[0,0] >> 4}; stored value = code + 8)")

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

# [[8: bits per weight including scales]]
print("\n== 8. Bits per weight = bits + scale bits / group ==")
for name, bits, g, sb in (("int8 per-row", 8, 4096, 16), ("int4 g=128 fp16 scale", 4, 128, 16), ("int4 g=64 fp16 scale", 4, 64, 16),
                          ("int4 g=32 fp16 scale", 4, 32, 16), ("int3 g=128", 3, 128, 16), ("MXFP4 (e8m0/32)", 4, 32, 8)):
    print(f"  {name:<24} {ql.bits_per_weight(bits, g, sb):.4f} bpw  -> 7B params ~ {7e9 * ql.bits_per_weight(bits, g, sb) / 8 / 1e9:.2f} GB")

# TRY IT: in section 3 change the layer to 'transformer.h.0.attn.c_attn' - does per-tensor still lose by as much?
# TRY IT: in section 4 use bits=8 - how much does clipping still matter at 256 levels?
# TRY IT: in section 2 scale `relu` by 100 - do asymmetric and symmetric SQNR change? (They should not: scale invariance.)
# BREAK IT: in qdq() drop the 'range must contain 0' min/max with 0.0 and quantize data that is all positive and far from 0 - zero is no longer exact.
# BREAK IT: in pack_int4 forget the +8 offset; negative codes wrap to 252..255 and unpack gives garbage. Print the round-trip check.
