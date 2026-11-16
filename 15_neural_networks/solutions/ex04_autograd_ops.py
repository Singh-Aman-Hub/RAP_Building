"""
GOAL:  Exercise 4 solution. Extend a micrograd-style scalar autograd `Value` with sigmoid, log and division,
       then verify gradients against torch on a function that uses all of them, including a node that is
       reused (so gradients must accumulate with +=).

RUN (from this folder):  python ex04_autograd_ops.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex04_autograd_ops.py
EXPECTED OUTPUT (shortened):
    f = ...   torch f = ...
    df/dx mine ... torch ...
NEEDS: numpy-free pure Python + torch for the check. < 1 s.
"""
import math
import torch


class Value:
    def __init__(self, data, parents=(), backward_fn=lambda: None):
        self.data, self.grad = float(data), 0.0
        self._parents, self._backward = parents, backward_fn

    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def bw():
            self.grad += out.grad; o.grad += out.grad          # d(a+b)/da = 1
        out._backward = bw
        return out

    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def bw():
            self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = bw
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def bw():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = bw
        return out

    # ---- the three new operations ----
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,))
        def bw():
            self.grad += s * (1 - s) * out.grad                # sigmoid' = s (1 - s)
        out._backward = bw
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        def bw():
            self.grad += (1 / self.data) * out.grad            # log' = 1/x
        out._backward = bw
        return out

    def __truediv__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        return self * (o ** -1)                                # a / b = a * b^-1 (reuses mul and pow)

    def backward(self):
        order, seen = [], set()
        def visit(v):                                          # topological sort (post-order DFS)
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


def f(x, y):
    s = x.sigmoid()
    # x is used three times (sigmoid, product, ratio) -> its gradient is the SUM of three paths
    return (s * y + x * x).log() / (y * 1.0 + x)


x, y = Value(0.7), Value(1.3)
out = f(x, y)
out.backward()

xt, yt = torch.tensor(0.7, dtype=torch.float64, requires_grad=True), torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
ot = (torch.sigmoid(xt) * yt + xt * xt).log() / (yt + xt)
ot.backward()
print(f"f = {out.data:.10f}   torch f = {ot.item():.10f}")
print(f"df/dx mine {x.grad:.10f} torch {xt.grad.item():.10f}")
print(f"df/dy mine {y.grad:.10f} torch {yt.grad.item():.10f}")
assert abs(x.grad - xt.grad.item()) < 1e-12 and abs(y.grad - yt.grad.item()) < 1e-12
print("OK: matches torch")
