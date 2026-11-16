"""
GOAL:  Build a micrograd-style autograd engine in ~60 lines: a scalar `Value` that remembers how it was
       computed (a graph), and a `backward()` that walks the graph in reverse topological order applying
       the chain rule. Check it against torch, then train a small MLP built only from `Value`s on XOR.
       This is what `loss.backward()` does inside PyTorch, minus tensors and speed.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/03_tiny_autograd_engine/main.py
EXPECTED OUTPUT (shortened):
    f = 8.0 | df/da = -3.0 (=b), df/db = 2.0 (=a), df/dc = 6.0 (=2c)
    ours vs torch, max abs gradient difference: 0.0
    y = x*x + x at x=3: y = 12.0, dy/dx = 7.0 (expected 7.0)
    XOR predictions: [0.0, 1.0, 1.0, 0.0] target: [0.0, 1.0, 1.0, 0.0]
NEEDS: Python + torch (only for the cross-check). A few seconds.
"""

import math
import random

import torch


# [1] A Value stores a number, its gradient, and a recipe for how it was made
class Value:
    def __init__(self, data, children=(), op=""):
        self.data = float(data)            # the number itself
        self.grad = 0.0                    # dLoss/dThis, filled in by backward()
        self._prev = tuple(children)       # the Values this one was computed from (graph edges; a tuple keeps order deterministic)
        self._op = op                      # name of the operation, for printing
        self._backward = lambda: None      # how to push our grad to the children (set by each op)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# [2] Each operation records its local derivative inside a small closure
def _add(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")
    def _backward():                       # d(a+b)/da = 1 and d(a+b)/db = 1
        self.grad += 1.0 * out.grad        # += because a Value may be used in many places
        other.grad += 1.0 * out.grad
    out._backward = _backward
    return out


def _mul(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")
    def _backward():                       # d(a*b)/da = b and d(a*b)/db = a
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out


def _pow(self, k):                         # k is a plain number, not a Value
    out = Value(self.data ** k, (self,), f"**{k}")
    def _backward():                       # d(a^k)/da = k * a^(k-1)
        self.grad += k * self.data ** (k - 1) * out.grad
    out._backward = _backward
    return out


def _tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,), "tanh")
    def _backward():                       # d tanh(a)/da = 1 - tanh(a)^2
        self.grad += (1 - t ** 2) * out.grad
    out._backward = _backward
    return out


def _exp(self):
    out = Value(math.exp(self.data), (self,), "exp")
    def _backward():                       # d e^a/da = e^a (which is out.data)
        self.grad += out.data * out.grad
    out._backward = _backward
    return out


def _relu(self):
    out = Value(max(0.0, self.data), (self,), "relu")
    def _backward():                       # slope 1 where the input was positive, else 0
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward
    return out


# Attach the operations to Value, and derive the rest (-, /, reversed operands) from the primitives
Value.__add__ = _add
Value.__mul__ = _mul
Value.__pow__ = _pow
Value.tanh = _tanh
Value.exp = _exp
Value.relu = _relu
Value.__radd__ = lambda self, other: self + other
Value.__rmul__ = lambda self, other: self * other
Value.__neg__ = lambda self: self * -1
Value.__sub__ = lambda self, other: self + (-other)
Value.__rsub__ = lambda self, other: (-self) + other
Value.__truediv__ = lambda self, other: self * other ** -1
Value.__rtruediv__ = lambda self, other: Value(other) * self ** -1


# [3] backward(): order the graph so every node comes after its inputs, then walk it in reverse
def backward(root):
    topo, visited = [], set()
    def build(v):                          # depth-first search; append a node AFTER all its children
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)
    build(root)
    root.grad = 1.0                        # d root / d root = 1: the seed of the chain rule
    for v in reversed(topo):               # outputs first, inputs last
        v._backward()
    return topo


# [4] Test on a small expression and compare with torch
a, b, c = Value(2.0), Value(-3.0), Value(3.0)
f = a * b + c ** 2                         # = -6 + 9 = 3
f = f * 1.0 + 5.0                          # = 8
topo = backward(f)
print(f"f = {f.data} | df/da = {a.grad} (=b), df/db = {b.grad} (=a), df/dc = {c.grad} (=2c)")
print("graph has", len(topo), "nodes:", sorted({v._op for v in topo if v._op}), "plus the leaves")

ta, tb, tc = (torch.tensor(x, dtype=torch.float64, requires_grad=True) for x in (2.0, -3.0, 3.0))
tf = (ta * tb + tc ** 2) * 1.0 + 5.0
tf.backward()
print("ours vs torch, max abs gradient difference:",
      max(abs(a.grad - ta.grad.item()), abs(b.grad - tb.grad.item()), abs(c.grad - tc.grad.item())))


# [5] A Value used twice must ACCUMULATE gradient from both uses
x = Value(3.0)
y = x * x + x                              # dy/dx = 2x + 1 = 7
backward(y)
print(f"y = x*x + x at x=3: y = {y.data}, dy/dx = {x.grad} (expected 7.0)")


# [6] A neural network built only from Values: Neuron -> Layer -> MLP
class Neuron:
    def __init__(self, n_in, act=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]   # one weight per input
        self.b = Value(0.0)
        self.act = act

    def __call__(self, x):
        z = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)        # w.x + b, as a graph of + and *
        return z.tanh() if self.act else z

    def parameters(self):
        return self.w + [self.b]


class MLP:
    def __init__(self, sizes):             # e.g. [2, 4, 1] = 2 inputs, 4 hidden, 1 output
        self.layers = [[Neuron(sizes[i], act=(i < len(sizes) - 2)) for _ in range(sizes[i + 1])]
                       for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for layer in self.layers:
            x = [n(x) for n in layer]
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for n in layer for p in n.parameters()]


# [7] Train on XOR with squared error: forward, zero grads, backward, update (the universal loop)
random.seed(1)
model = MLP([2, 4, 1])
print("parameters:", len(model.parameters()))
X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]
for step in range(301):
    preds = [model(x) for x in X]                                  # forward: builds a fresh graph each step
    loss = sum(((p - y) ** 2 for p, y in zip(preds, Y)), Value(0.0))   # sum of squared errors
    for p in model.parameters():
        p.grad = 0.0                                               # gradients accumulate, so reset them
    backward(loss)
    for p in model.parameters():
        p.data -= 0.05 * p.grad                                    # gradient descent
    if step % 50 == 0:
        print(f"  step {step:>3} loss {loss.data:.4f}")
print("XOR predictions:", [round(model(x).data, 3) for x in X], "target:", Y)

# TRY IT: add a `sigmoid` op (1 / (1 + exp(-x))) built from existing ops and train with it instead of tanh.
# BREAK IT: remove the `p.grad = 0.0` reset loop and watch the loss behave differently (gradients pile up across steps).
