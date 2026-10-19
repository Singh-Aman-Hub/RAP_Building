"""
GOAL:   Understand quantization by doing it: shrink float32 weights to int8/int4, measure the
        memory saved and the accuracy lost, on a toy matrix and on real distilgpt2 weights.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/05_quantization/main.py
EXPECTED OUTPUT (short): int8 is 4x smaller with tiny error; int4 is 8x smaller with ~18x bigger
        error; distilgpt2 loss barely moves at int8 but degrades badly at int4 per-tensor (group-wise rescues it).
NEEDS:  numpy, torch, transformers (distilgpt2 cached). ~20 s.

Quantization = store each weight with FEWER BITS.  float32 uses 32 bits (4 bytes); int8 uses 8 bits.
Smaller weights -> less memory, less memory bandwidth (usually the bottleneck for LLM inference),
so faster and cheaper - at the price of some rounding error.

Symmetric quantization recipe (per tensor):
    scale  = max(|w|) / (2^(bits-1) - 1)          # e.g. max/127 for int8
    q      = round(w / scale)  (stored as integers)
    w_hat  = q * scale          (de-quantize when you need real numbers)
This file SIMULATES quantization (quantize -> de-quantize, stored as float) so you can measure the
error on any laptop. Real engines (bitsandbytes, GPTQ, AWQ, llama.cpp) store the integers and use
special kernels - see 06_llm_training_and_fast_inference for QLoRA.
"""

import copy
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def quantize_symmetric(w, bits=8):
    """Return (integer_tensor, scale).  `w` is any numpy array of floats."""
    qmax = 2 ** (bits - 1) - 1                    # 127 for int8, 7 for int4
    scale = np.abs(w).max() / qmax
    q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)   # int8 container even for 4-bit demo
    return q, scale


def dequantize(q, scale):
    return q.astype(np.float32) * scale


# ---------------------------------------------------------------------------
# 1. Toy example you can read
# ---------------------------------------------------------------------------
w = np.array([0.823, -0.317, 0.054, -1.274, 0.447], dtype=np.float32)
q, s = quantize_symmetric(w, 8)
print("weights  :", w)
print("int8     :", q, f"(scale={s:.5f})")
print("restored :", np.round(dequantize(q, s), 4))
print("abs error:", np.round(np.abs(w - dequantize(q, s)), 4))

# ---------------------------------------------------------------------------
# 2. A bigger random matrix: error and memory at 8 vs 4 bits
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(768, 768)).astype(np.float32)    # typical LLM weight scale
print("\nmatrix 768x768 (float32 =", W.nbytes // 1024, "KiB)")
for bits in (8, 4):
    q, s = quantize_symmetric(W, bits)
    err = np.abs(W - dequantize(q, s)).mean()
    mem = W.size * bits / 8 / 1024
    print(f"  int{bits}: ~{mem:6.0f} KiB ({32 // bits}x smaller), mean abs error = {err:.6f}")

# ---------------------------------------------------------------------------
# 3. Quantize ALL weight matrices of real distilgpt2 and measure what happens to the loss
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained("distilgpt2")
text = ("The quick brown fox jumps over the lazy dog. Machine learning models learn patterns from data "
        "and use them to make predictions on new examples.")
batch = tok(text, return_tensors="pt")
base = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()


def loss_of(m):
    with torch.no_grad():
        return m(**batch, labels=batch["input_ids"]).loss.item()


def fake_quantize_model(model, bits):
    """Return a copy of `model` whose 2-D weight matrices were quantized then de-quantized."""
    m = copy.deepcopy(model)
    for name, p in m.named_parameters():
        if p.ndim == 2 and "wte" not in name and "wpe" not in name:    # linear-layer matrices only
            q, s = quantize_symmetric(p.detach().numpy(), bits)
            p.data = torch.from_numpy(dequantize(q, s))
    return m


print(f"\nfloat32 baseline loss: {loss_of(base):.3f}")
n_matrix_params = sum(p.numel() for n, p in base.named_parameters() if p.ndim == 2 and "wte" not in n and "wpe" not in n)
for bits in (8, 6, 4, 3):
    print(f"int{bits} (simulated) loss: {loss_of(fake_quantize_model(base, bits)):.3f}   "
          f"-> matrices would shrink {32 // bits if bits in (8, 4) else 32 / bits:.1f}x "
          f"({n_matrix_params * 4 / 1e6:.0f} MB -> {n_matrix_params * bits / 8 / 1e6:.0f} MB)")

# ---------------------------------------------------------------------------
# 4. Outliers: ONE huge weight ruins per-tensor scaling -> why real methods quantize in small groups
# ---------------------------------------------------------------------------
W2 = W.copy()
W2[0, 0] = 5.0                                    # a single outlier, 250x typical size
q, s = quantize_symmetric(W2, 8)
q0, s0 = quantize_symmetric(W, 8)
print(f"\nint8 mean abs error, no outlier  : {np.abs(W - dequantize(q0, s0)).mean():.6f}")
print(f"int8 mean abs error, one outlier : {np.abs(W2 - dequantize(q, s))[1:].mean():.6f} -> the scale was stretched by the outlier")

# ---------------------------------------------------------------------------
# 5. Fix: GROUP-WISE quantization - one scale per small group of weights (what GPTQ/AWQ/llama.cpp do).
#    An outlier now only hurts its own group of 64 numbers.
# ---------------------------------------------------------------------------
def fake_quantize_grouped(model, bits, group=64):
    m = copy.deepcopy(model)
    for name, p in m.named_parameters():
        if p.ndim == 2 and "wte" not in name and "wpe" not in name:
            flat = p.detach().numpy().reshape(-1, group)          # every row = one group of `group` weights
            qmax = 2 ** (bits - 1) - 1
            scales = np.abs(flat).max(axis=1, keepdims=True) / qmax   # one scale PER GROUP
            q = np.clip(np.round(flat / scales), -qmax, qmax)
            p.data = torch.from_numpy((q * scales).reshape(p.shape).astype(np.float32))
    return m

print("\nper-tensor vs group-wise (group=64) loss on distilgpt2 (baseline %.3f):" % loss_of(base))
for bits in (8, 4):
    print(f"  int{bits}: per-tensor {loss_of(fake_quantize_model(base, bits)):.3f} | "
          f"group-wise {loss_of(fake_quantize_grouped(base, bits)):.3f}")
print("  (cost of groups: one extra scale number per 64 weights = a little more memory)")

# TRY IT: change `text` to something very different - does int4 hurt more or less?
# TRY IT: quantize in groups of 64 values (reshape to (-1, 64), one scale per row) and re-measure the outlier case.
# BREAK IT: remove np.clip - values can overflow the int8 range and wrap around. Print q.min()/q.max().
# BREAK IT: use dtype=np.uint8 for q: negative weights wrap to huge numbers. Why is signed int needed?
