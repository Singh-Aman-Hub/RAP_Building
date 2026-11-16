"""
GOAL:  Exercise 2 solution. A 1-hidden-layer tanh MLP for regression with MSE loss: write the forward
       pass and the backward pass by hand (with shapes), then verify EVERY gradient with centered finite
       differences in float64.

RUN (from this folder):  python ex02_backprop_gradcheck.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex02_backprop_gradcheck.py
EXPECTED OUTPUT (shortened):
    W1 max relative error ...e-10
    ...
NEEDS: numpy only. < 1 s on CPU.
"""
import numpy as np

rng = np.random.default_rng(0)
N, D, H = 6, 3, 4
X = rng.normal(size=(N, D))
y = rng.normal(size=(N, 1))
P = {"W1": rng.normal(size=(D, H)) * 0.5, "b1": np.zeros(H),
     "W2": rng.normal(size=(H, 1)) * 0.5, "b2": np.zeros(1)}


def forward(X, P):
    z1 = X @ P["W1"] + P["b1"]       # (N, H)
    a1 = np.tanh(z1)                 # (N, H)
    out = a1 @ P["W2"] + P["b2"]     # (N, 1)
    return z1, a1, out


def loss(P):
    return 0.5 * np.mean((forward(X, P)[2] - y) ** 2)   # mean over N samples


def backward(P):
    z1, a1, out = forward(X, P)
    d_out = (out - y) / N                     # dL/dout, shape (N, 1)   (the 1/N comes from the mean)
    g = {}
    g["W2"] = a1.T @ d_out                    # (H, N) @ (N, 1) -> (H, 1)  same shape as W2
    g["b2"] = d_out.sum(axis=0)               # (1,)
    d_a1 = d_out @ P["W2"].T                  # (N, 1) @ (1, H) -> (N, H)
    d_z1 = d_a1 * (1 - a1 ** 2)               # tanh' = 1 - tanh^2
    g["W1"] = X.T @ d_z1                      # (D, N) @ (N, H) -> (D, H)
    g["b1"] = d_z1.sum(axis=0)                # (H,)
    return g


g = backward(P)
eps = 1e-6
for name, arr in P.items():
    num = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps; lp = loss(P)
        arr[i] = old - eps; lm = loss(P)
        arr[i] = old
        num[i] = (lp - lm) / (2 * eps)
    rel = np.max(np.abs(num - g[name]) / (np.abs(num) + np.abs(g[name]) + 1e-12))
    print(f"{name} max relative error {rel:.2e}")
    assert rel < 1e-6
print("all gradients match finite differences")
