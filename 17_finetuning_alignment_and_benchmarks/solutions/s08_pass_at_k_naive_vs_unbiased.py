"""Exercise 08 (Explain): a single random size-k subset gives a 0-or-1 answer per problem (unbiased on average but very noisy); the formula averages over ALL subsets.
RUN: .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s08_pass_at_k_naive_vs_unbiased.py"""
import numpy as np
rng = np.random.default_rng(0)
def unbiased(n, c, k):
    return 1.0 if n - c < k else 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
n, c, k = 10, 3, 5
true = 1 - np.prod([(10 - 3 - i) / (10 - i) for i in range(k)])   # exact for this fixed sample
naive = []
for _ in range(2000):
    s = rng.permutation([1] * c + [0] * (n - c))
    naive.append(float(s[:k].any()))                              # naive: ONE random size-k subset per problem
print(f"unbiased formula {unbiased(n, c, k):.4f} | exact {true:.4f} | naive single-subset mean {np.mean(naive):.4f} std per problem {np.std(naive):.2f} (0 or 1 each time)")
