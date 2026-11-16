"""
GOAL:  Exercise 3 solution. Numerically stable softmax / log-sum-exp / cross-entropy in numpy, the gradient
       (p - onehot)/N proven against torch, and a demonstration that the naive version overflows.

RUN (from this folder):  python ex03_softmax_ce.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex03_softmax_ce.py
EXPECTED OUTPUT (shortened):
    naive softmax on [1000, 1001, 1002]: [nan nan nan]
    ...
NEEDS: numpy, torch (CPU). ~1 s.
"""
import warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")


def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)    # shift: same result, exp never overflows
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(logits, y):
    """Mean CE using log-sum-exp (never forms probabilities): -z_y + log sum exp(z)."""
    m = logits.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True))).ravel()
    return np.mean(lse - logits[np.arange(len(y)), y])


big = np.array([1000.0, 1001.0, 1002.0])
print("naive softmax on [1000, 1001, 1002]:", softmax_naive(big))
print("stable softmax                      :", softmax(big).round(4))

rng = np.random.default_rng(0)
Z = rng.normal(size=(5, 4)) * 3
y = rng.integers(0, 4, size=5)
mine = cross_entropy(Z, y)
zt = torch.tensor(Z, requires_grad=True)
ref = F.cross_entropy(zt, torch.tensor(y))
ref.backward()
print(f"CE mine {mine:.10f}  torch {ref.item():.10f}")
assert abs(mine - ref.item()) < 1e-12

# gradient: (softmax - onehot) / N
onehot = np.eye(4)[y]
grad = (softmax(Z) - onehot) / len(y)
print("max |grad - torch grad| =", np.abs(grad - zt.grad.numpy()).max())
assert np.allclose(grad, zt.grad.numpy())
print("each gradient row sums to", grad.sum(axis=1).round(12), "(softmax gradients always sum to 0 over classes)")
print("initial loss for random logits ~ ln(K):", round(float(np.log(4)), 4))
