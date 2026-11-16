"""Exercise Q1 solution: scale and zero-point for the range [-0.4, 3.0] at uint8; check the worst-case error is s/2."""
import numpy as np

lo, hi = -0.4, 3.0
s = (hi - lo) / 255                       # real width of one integer step
z = round(-lo / s)                        # integer code that decodes to exactly 0.0
x = np.linspace(lo, hi, 10001)
q = np.clip(np.round(x / s) + z, 0, 255)
xh = (q - z) * s
print(f"s = {s:.6f}, z = {z}, max abs error = {np.abs(x - xh).max():.6f}, s/2 = {s / 2:.6f}")
assert np.abs(x - xh).max() <= s / 2 + 1e-9    # rounding error never exceeds half a step inside the range
assert (np.clip(np.round(0.0 / s) + z, 0, 255) - z) * s == 0.0   # 0.0 is exactly representable
print("OK")
