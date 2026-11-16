"""Exercise D1: tensor-parallel linear for any N, both modes, with an exactness check."""
import numpy as np


def tp_linear(X, W, n, mode):
    if mode == "column":                               # split output columns, concatenate (all-gather)
        return np.concatenate([X @ w for w in np.split(W, n, axis=1)], axis=1)
    if mode == "row":                                  # split input rows + X columns, sum partials (all-reduce)
        return np.sum([x @ w for x, w in zip(np.split(X, n, axis=1), np.split(W, n, axis=0))], axis=0)
    raise ValueError(mode)


rng = np.random.default_rng(1)
X, W = rng.standard_normal((7, 48)), rng.standard_normal((48, 96))
for n in (1, 2, 3, 4, 6, 8):                           # 48 and 96 are divisible by all of these
    for mode in ("column", "row"):
        assert np.allclose(tp_linear(X, W, n, mode), X @ W, atol=1e-10), (n, mode)
print("D1 ok: column and row parallel equal the unsplit layer for N in 1,2,3,4,6,8")
try:
    tp_linear(X, W, 5, "column")                       # 96 % 5 != 0
except ValueError as e:
    print("N=5 fails as expected:", str(e)[:50])
