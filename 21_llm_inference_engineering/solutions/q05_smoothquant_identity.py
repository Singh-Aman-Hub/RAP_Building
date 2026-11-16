"""Exercise Q5 solution: SmoothQuant keeps X@W^T identical in float but makes per-tensor W8A8 much more accurate when ONE channel is an outlier."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quantlib as ql

rng = np.random.default_rng(0)
X = rng.normal(0, 1, (512, 256)).astype(np.float32)
X[:, 17] *= 60                                           # one outlier input channel (like LayerNorm outputs in real LLMs)
W = rng.normal(0, 0.05, (128, 256)).astype(np.float32)   # (out, in)
Y = X @ W.T

def w8a8(X, W):
    Xq = ql.qdq(X, 8, True, "tensor")                    # per-tensor activation int8
    Wq = ql.qdq(W, 8, True, "tensor")                    # per-tensor weight int8
    return Xq @ Wq.T

before = ql.metrics(Y, w8a8(X, W))["sqnr_db"]
alpha = 0.5
s = np.abs(X).max(0) ** alpha / np.abs(W).max(0) ** (1 - alpha)      # s_j = max|X_j|^a / max|W_j|^(1-a)
Xs, Ws = X / s, W * s                                    # divide activations, multiply the matching weight COLUMNS
assert np.allclose(Xs @ Ws.T, Y, rtol=1e-3, atol=1e-3)  # identical in float
after = ql.metrics(Y, w8a8(Xs, Ws))["sqnr_db"]
print(f"per-tensor W8A8 SQNR: {before:.1f} dB -> {after:.1f} dB with SmoothQuant alpha={alpha}")
assert after > before + 3
print("OK")
