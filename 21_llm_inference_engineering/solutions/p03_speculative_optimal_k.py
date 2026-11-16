"""Exercise P3 solution: best draft length k for given acceptance alpha and cost ratio c; check the formula by Monte Carlo."""
import numpy as np


def speedup(alpha, k, c):
    return (1 - alpha ** (k + 1)) / (1 - alpha) / (k * c + 1)


for alpha, c in ((0.6, 0.05), (0.8, 0.05), (0.8, 0.3), (0.9, 0.02)):
    best = max(range(1, 33), key=lambda k: speedup(alpha, k, c))
    print(f"alpha={alpha} c={c}: best k={best}, speedup {speedup(alpha, best, c):.2f}x")
assert max(range(1, 33), key=lambda k: speedup(0.8, k, 0.05)) > max(range(1, 33), key=lambda k: speedup(0.8, k, 0.3))   # costlier draft -> shorter k

# Monte Carlo of tokens per round: accept each draft token with prob alpha until the first reject; always +1 target token
rng = np.random.default_rng(0)
alpha, k, n = 0.8, 4, 200_000
tokens = np.empty(n)
for i in range(n):
    j = 0
    while j < k and rng.random() < alpha:
        j += 1
    tokens[i] = j + 1
formula = (1 - alpha ** (k + 1)) / (1 - alpha)
print(f"tokens per target pass: simulated {tokens.mean():.3f} vs formula {formula:.3f}")
assert abs(tokens.mean() - formula) < 0.02
print("OK")
