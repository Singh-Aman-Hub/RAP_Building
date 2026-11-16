"""Exercise 8: how precise is a gold-set accuracy estimate? Wilson 95% interval for n pages, and the n needed for a given half-width.
RUN: python s08_how_many_pages.py"""
import math


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


for k, n in [(45, 50), (90, 100), (450, 500)]:
    lo, hi = wilson(k, n)
    print(f"{k}/{n} correct = {k / n:.2f}   95% interval [{lo:.3f}, {hi:.3f}]  (width {hi - lo:.3f})")
lo, hi = wilson(45, 50)
assert 0.78 < lo < 0.80 and 0.94 < hi < 0.96
need = next(n for n in range(10, 5000) if (lambda lh: (lh[1] - lh[0]) / 2)(wilson(round(0.9 * n), n)) <= 0.03)
print("pages needed for a +-3 point interval at 90% accuracy:", need)
print("ok")
