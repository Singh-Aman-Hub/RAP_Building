"""Exercise Q6 solution: per-channel scales win for K (outlier channels), per-token scales win for V (no outlier channels), int4."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quantlib as ql

rng = np.random.default_rng(0)
T, D = 256, 64
K = rng.normal(0, 1, (T, D)).astype(np.float32) * (1 + 0.05 * rng.random(D)).astype(np.float32)
K[:, [3, 40]] *= 100                                      # K: a few huge channels, the same channels in every token
V = (rng.normal(0, 1, (T, D)) * rng.lognormal(0, 0.8, (T, 1))).astype(np.float32)   # V: token magnitudes vary, channels are alike

def per_token(a, bits=4):   return ql.qdq(a, bits, False, "row")          # rows = tokens, scale over the 64 channels
def per_channel(a, bits=4): return ql.qdq(a.T, bits, False, "row").T      # transpose: scale over the T tokens of each channel

keep = np.setdiff1d(np.arange(D), [3, 40])               # judge the NORMAL channels: whole-tensor SQNR is dominated by the outlier channels' energy
def sq(a, f, cols=slice(None)):
    return ql.metrics(a[:, cols], f(a)[:, cols])["sqnr_db"]

for name, a in (("K", K), ("V", V)):
    print(f"{name} (normal channels): per-token {sq(a, per_token, keep):6.2f} dB | per-channel {sq(a, per_channel, keep):6.2f} dB")
assert sq(K, per_channel, keep) > sq(K, per_token, keep) + 5      # outliers crush the other 62 channels' resolution under per-token scaling
assert sq(V, per_token, keep) > sq(V, per_channel, keep)
print("OK")
