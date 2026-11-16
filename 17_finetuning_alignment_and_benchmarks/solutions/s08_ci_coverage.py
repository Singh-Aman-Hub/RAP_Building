"""Exercise 08 (Modify): how often does a 95% interval contain the true accuracy? Percentile bootstrap vs Wilson at small n.
RUN: .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s08_ci_coverage.py"""
import math, numpy as np
rng = np.random.default_rng(0)
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return c-h, c+h
def boot(x, B=500):
    m = rng.choice(x, size=(B, len(x))).mean(1); return np.quantile(m, [0.025, 0.975])
for n, p in ((20, 0.9), (30, 0.5), (200, 0.5)):
    hit_w = hit_b = 0
    T = 300
    for _ in range(T):
        x = (rng.random(n) < p).astype(float)
        lo, hi = wilson(int(x.sum()), n); hit_w += lo <= p <= hi
        lo, hi = boot(x); hit_b += lo <= p <= hi
    print(f"n={n:<4} p={p}: coverage Wilson {hit_w/T:.2f}  bootstrap {hit_b/T:.2f}  (target 0.95)")
