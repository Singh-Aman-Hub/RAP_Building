"""Solution ex05: int4 quantization error vs group size on a matrix that contains outliers.
RUN: .venv/bin/python 05_llm_parameters/solutions/ex05_group_size_curve.py"""
import numpy as np

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(256, 256)).astype(np.float32)
W.flat[rng.choice(W.size, 40, replace=False)] *= 40       # sprinkle 40 outliers


def quant_error(W, bits, group):
    flat = W.reshape(-1, group)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(flat).max(axis=1, keepdims=True) / qmax
    restored = np.clip(np.round(flat / scale), -qmax, qmax) * scale
    return np.abs(flat - restored).mean()


print("group size | mean abs error (int4) | scale overhead")
for g in (W.size, 4096, 1024, 256, 64, 16):
    # overhead: one 16-bit scale per group, relative to the 4 bits per weight
    overhead = 16 / g / 4 * 100
    print(f"{g:10d} | {quant_error(W, 4, g):.6f}              | {overhead:.2f}%")
print("Smaller groups -> lower error, more scale numbers to store: a classic accuracy vs memory trade-off.")
