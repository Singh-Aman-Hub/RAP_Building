"""Exercise D2: how many micro-batches keep the pipeline bubble under 10% with p=8 stages? Verify by simulation."""
import numpy as np


def bubble_formula(p, m):
    return (p - 1) / (m + p - 1)


def bubble_sim(p, m):
    busy = np.zeros((p, m + p - 1), int)
    for s in range(p):
        busy[s, s:s + m] = 1                           # stage s works on micro-batches at ticks s..s+m-1
    return 1 - busy.sum() / busy.size


p, target = 8, 0.10
# (p-1)/(m+p-1) <= b  =>  m >= (p-1)(1-b)/b
m_min = int(np.ceil((p - 1) * (1 - target) / target))
assert bubble_formula(p, m_min) <= target < bubble_formula(p, m_min - 1)
assert abs(bubble_sim(p, m_min) - bubble_formula(p, m_min)) < 1e-12
print(f"D2 ok: p={p}, target {target:.0%} -> need m >= {m_min} micro-batches (bubble {bubble_formula(p, m_min):.3f}; m-1 gives {bubble_formula(p, m_min - 1):.3f})")
