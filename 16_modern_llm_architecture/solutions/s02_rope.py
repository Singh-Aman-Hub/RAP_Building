"""
GOAL:  Solutions for step 02: RoPE wavelengths for a huge base, relative-position test at large positions, the odd-D
       failure, and the equivalence of the interleaved and rotate-half layouts.

RUN (from this folder):  python s02_rope.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s02_rope.py
EXPECTED OUTPUT (shortened):
    base 10000 ... base 1000000 ...
NEEDS: numpy only. < 1 s.
"""
import numpy as np


def freqs(d, base):
    return base ** (-np.arange(0, d, 2) / d)               # theta_i = base^(-2i/d), one per pair


def rope_interleaved(x, pos, base=10000.0):
    """x: (d,) rotated at integer position pos. Pairs are (x0,x1), (x2,x3), ..."""
    d = x.shape[0]
    ang = pos * freqs(d, base)
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out


def rope_half(x, pos, base=10000.0):
    """Rotate-half layout (Llama/HF): pairs are (x_i, x_{i+d/2})."""
    d = x.shape[0]
    ang = pos * freqs(d, base)
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[: d // 2], x[d // 2:]
    return np.concatenate([a * c - b * s, a * s + b * c])


# 1. longest wavelength 2 pi / theta_min
D = 64
for base in (10000.0, 1_000_000.0):
    wl = 2 * np.pi / freqs(D, base)
    print(f"base {base:>9,.0f}: longest wavelength {wl.max():,.0f} tokens; pairs with wavelength > 4,096: {(wl > 4096).sum()} of {D // 2}")

# 2. relative property: score depends on (n - m) only; float error grows with absolute position
rng = np.random.default_rng(0)
q, k = rng.normal(size=D), rng.normal(size=D)
for dist in (0, 50):
    scores = [rope_interleaved(q, p) @ rope_interleaved(k, p + dist) for p in (10, 10_000)]
    print(f"distance {dist:>2}: score at pos 10 = {scores[0]:.12f}, at pos 10,000 = {scores[1]:.12f}, diff {abs(scores[0] - scores[1]):.2e}")
print("(the diff is float rounding of cos/sin of big angles; in bf16 it would be far larger: another reason long-context needs care)")

# 3. odd head dimension: pairs cannot be formed
try:
    rope_interleaved(np.ones(7), 3)
except ValueError as e:
    print("D=7 fails at the pair assignment:", e)

# 4. the two layouts give the same score once dims are permuted consistently
perm = np.concatenate([np.arange(0, D, 2), np.arange(1, D, 2)])   # interleaved index -> half layout index
q2, k2 = q[perm], k[perm]
for p, n in ((3, 9), (500, 777)):
    s1 = rope_interleaved(q, p) @ rope_interleaved(k, n)
    s2 = rope_half(q2, p) @ rope_half(k2, n)
    print(f"positions ({p},{n}): interleaved {s1:.10f} | rotate-half on permuted dims {s2:.10f}")
    assert abs(s1 - s2) < 1e-9
