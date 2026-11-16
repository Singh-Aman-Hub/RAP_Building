"""
GOAL:  Exercise 8 solution (PyTorch). Write inverted dropout as an nn.Module, show that it matches the 1/(1-p)
       scaling (mean preserved), that eval() makes it the identity, and that forgetting eval() makes the same
       input give different outputs.

RUN (from this folder):  python ex08_dropout_train_eval.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex08_dropout_train_eval.py
EXPECTED OUTPUT (shortened):
    train-mode mean ... (input mean 1.0)
    eval: identical -> True
NEEDS: torch (CPU). < 1 s.
"""
import torch
import torch.nn as nn


class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:      # nn.Module.training is flipped by .train() / .eval()
            return x                              # inference: identity
        keep = (torch.rand_like(x) >= self.p).to(x.dtype)   # 1 with prob 1-p
        return x * keep / (1 - self.p)            # scale survivors so E[output] == input


torch.manual_seed(0)
x = torch.ones(100000)
d = MyDropout(0.3)
y = d(x)
print(f"train-mode mean {y.mean():.4f} (input mean 1.0); fraction zeroed {(y == 0).float().mean():.3f}; survivor value {y.max():.4f} = 1/0.7")

d.eval()
print("eval: identical ->", torch.equal(d(x), x))

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), MyDropout(0.5), nn.Linear(8, 2))
inp = torch.randn(1, 4)
a, b = net(inp), net(inp)
print("forgot eval(): same input, two outputs differ ->", not torch.equal(a, b))
net.eval()
with torch.no_grad():
    a, b = net(inp), net(inp)
print("after eval(): outputs identical ->", torch.equal(a, b))
# Compare with the library layer: same behaviour
ref = nn.Dropout(0.3)
print(f"nn.Dropout train mean {ref(x).mean():.4f}")
