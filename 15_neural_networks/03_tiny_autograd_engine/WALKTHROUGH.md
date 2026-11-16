# Step 03 — A tiny autograd engine

## The story so far
[Step 02](../02_mlp_backprop_from_scratch/WALKTHROUGH.md) derived backpropagation by hand for one architecture and checked it against `loss.backward()`. [Module 03 step 03](../../03_ml_foundations/03_pytorch_tensors_autograd/WALKTHROUGH.md) showed `requires_grad` and `.backward()` as a black box.

## The pain point
Hand-derived backprop does not scale: change the activation, add a skip connection, add attention, and you must re-derive and re-debug every line. Nobody writes research code that way. Something has to differentiate *any* program.

## The spark
Record the computation as it happens. Every arithmetic result remembers its inputs and its own tiny local derivative. The output is then a node at the end of a graph; the chain rule is a walk over that graph in reverse. This is Karpathy's *micrograd* idea; PyTorch does the same on tensors, in C++, with many more operations.

## Goal
Implement a scalar `Value` with `+`, `*`, `**`, `tanh`, `exp`, `relu`, a topological-sort `backward`, check it against torch (difference 0.0), see gradient accumulation on a reused variable, then train a 17-parameter MLP built purely from `Value`s on XOR.

## The theory (read this before the interview)
**Computational graph.** Every operation creates a node holding (a) the result, (b) pointers to its input nodes, (c) a function that knows the local derivative. For `f = a*b + c**2`:
```
 a ---       (*) --- t1 --- b ---/               (+) --- f
 c --- (**2) --- t2 --/
```
**Reverse-mode automatic differentiation.** Start at the output with `df/df = 1`. A node whose output has gradient `g` hands each input `local_derivative * g`. When a node is used in several places, its gradient is the **sum** of what arrives from every use (that is why the code uses `+=`).

| Operation `out = ...` | Local derivative(s) | Backward closure adds |
|---|---|---|
| `a + b` | 1, 1 | `a.grad += out.grad`, `b.grad += out.grad` |
| `a * b` | b, a | `a.grad += b * out.grad`, `b.grad += a * out.grad` |
| `a ** k` | `k a^(k-1)` | `a.grad += k a^(k-1) out.grad` |
| `tanh(a)` | `1 - tanh(a)^2` | `a.grad += (1 - out^2) out.grad` |
| `exp(a)` | `exp(a)` | `a.grad += out * out.grad` |
| `relu(a)` | 1 if a > 0 else 0 | `a.grad += (out > 0) out.grad` |

Subtraction, division and negation are built from these (`a - b = a + (b * -1)`, `a / b = a * b**-1`): a small set of primitives is enough.

**Topological order.** Before a node can pass gradient to its inputs it must have received gradient from *all* of its uses. A topological order lists every node after all of its inputs; walking it in reverse guarantees that. Depth-first search that appends a node after visiting its children produces it.

**Forward vs reverse mode.** Reverse mode gives the gradient of one scalar output with respect to all inputs in one backward sweep, whose cost is a small constant times the forward pass. That fits deep learning (millions of parameters, one loss). Forward mode is better for few inputs, many outputs.

**Dynamic graphs.** The graph is rebuilt on every forward pass ('define-by-run'), as in PyTorch eager mode. That is why Python `if` and `for` work inside models, and why the training loop below constructs new `Value`s each step. `torch.compile` and JAX trace the program once and optimise it.

**What PyTorch adds.** Tensors instead of scalars (one node per matrix multiply, not per number), a large library of fused kernels with hand-written backward rules, `no_grad` / `detach` to cut the graph, hooks, higher-order derivatives (`create_graph=True`), and GPU support. `tensor.grad_fn` is the attribute that holds the node's backward function.

**Gradient accumulation (the PyTorch behaviour).** `.grad` is *added to*, never overwritten, which is why the loop must zero it each step (`optimizer.zero_grad()`), and why people deliberately skip zeroing to accumulate over micro-batches ([module 06](../../06_llm_training_and_lora/README.md)).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| autograd | automatic differentiation, autodiff, AD, reverse-mode AD, backprop engine | Autograd is PyTorch's name for its reverse-mode AD engine | Backprop = the idea; autograd = the implementation; JAX calls it `grad` |
| computational graph | autograd graph, DAG, tape, trace | Same idea | 'Tape' (older) = a flat recording; 'graph' = nodes and edges |
| gradient accumulation | summing gradients, `+=` into `.grad` | Two meanings | (1) the engine adds contributions from several uses of a node; (2) a training trick that sums grads over several mini-batches before one update |
| leaf node | input, parameter, `is_leaf=True` | Same | A leaf has no recorded `grad_fn`; parameters are leaves |
| `detach()` / `no_grad()` | cut the graph, stop gradient | Related, not identical | `detach()` is per tensor; `no_grad()` is a context for all operations inside it |
| eager mode | define-by-run, dynamic graph | Same | Contrast with static graphs (old TF1) and traced graphs (`torch.compile`, JAX `jit`) |
| topological sort | topological order, linearisation of a DAG | Same | Any order where each node follows its inputs |

## Run it
```bash
cd 15_neural_networks/03_tiny_autograd_engine
python main.py       # a few seconds, pure Python (torch only for the cross-check)
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-22)
```python
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


```
- **In simple words:** Import `math` for `tanh`/`exp`, `random` for starting weights, and torch just to check our answers.
- **Exactly:** `random` is the standard library RNG (not numpy); we seed it later with `random.seed(1)`. The engine itself uses no external library.
- **Why it is here:** The point of the file is that the engine is plain Python; torch is only the referee.

### 1. A Value stores a number, its gradient, and a recipe for how it was made  (`main.py` lines 23-33)
```python
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
```
- **In simple words:** A Value is a number with a diary. It remembers what it is, who it was made from, and has a spot to store how much the final answer depends on it.
- **Exactly:** `data` is a Python float; `grad` starts at 0.0 (`dLoss/dself`); `_prev` is the tuple of input Values (the graph edges; a tuple instead of a set keeps the visit order, and therefore floating-point summation order, deterministic); `_op` is a label; `_backward` is a function that pushes `self.grad` into the children, a no-op for leaves.
- **Why it is here:** These four fields are the entire graph. Using a `set` here made results differ in the 16th digit between runs (set order depends on object addresses), and a training run amplified that into visibly different loss curves.

### 2. Each operation records its local derivative inside a small closure  (`main.py` lines 36-103)
```python
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
```
- **In simple words:** Each operation makes a new Value and also writes down its own little derivative rule, ready to be used later.
- **Exactly:** Each `_x` function builds `out` with the inputs as `_prev` and defines a closure `_backward` that captures `self`, `other` and `out`. Multiplying: `self.grad += other.data * out.grad`. The primitives are `+`, `*`, `**k` (k a plain number), `tanh`, `exp`, `relu`; `-`, `/` and the reversed forms (`2 * v`, `1 - v`) are derived from them in the last lines, e.g. `a / b = a * b**-1`. Wrapping a plain number with `Value(other)` lets `v + 1.0` work.
- **Why it is here:** Closures capture the exact numbers of this forward pass, so backward needs no recomputation. Using `+=` rather than `=` is essential because one Value can feed several operations.

### 3. backward(): order the graph so every node comes after its inputs, then walk it in reverse  (`main.py` lines 106-119)
```python
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
```
- **In simple words:** To walk the graph backwards you first need a list where everything comes after the things it was built from. Then walk that list from the end, telling each node to pass its blame down.
- **Exactly:** `build` is a depth-first search: it visits each node once (`visited`), recurses into children, then appends the node, so `topo` lists inputs before outputs. `root.grad = 1.0` seeds the chain rule. `for v in reversed(topo): v._backward()` calls each node's closure when its own `grad` is complete. It returns `topo` so callers can count nodes.
- **Why it is here:** Without the correct order a node could pass on a half-finished gradient, giving silently wrong numbers whenever a value is reused.

### 4. Test on a small expression and compare with torch  (`main.py` lines 122-134)
```python
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
```
- **In simple words:** Try it on a small formula we can differentiate in our heads, and compare with PyTorch.
- **Exactly:** `a=2, b=-3, c=3`: `f = (a*b + c**2) * 1.0 + 5.0 = 8`. By hand: `df/da = b = -3`, `df/db = a = 2`, `df/dc = 2c = 6`, the printed numbers. The graph has 10 nodes: 3 inputs `a, b, c`, the constants `1.0` and `5`, and the 5 operation results. The torch version gives gradients identical to the last bit (difference `0.0`).
- **Why it is here:** It is the minimal test for any autodiff engine: a hand-checkable expression plus a reference implementation.

### 5. A Value used twice must ACCUMULATE gradient from both uses  (`main.py` lines 137-141)
```python
# [5] A Value used twice must ACCUMULATE gradient from both uses
x = Value(3.0)
y = x * x + x                              # dy/dx = 2x + 1 = 7
backward(y)
print(f"y = x*x + x at x=3: y = {y.data}, dy/dx = {x.grad} (expected 7.0)")
```
- **In simple words:** If you use the same number twice, both uses push blame back to it, and the blame adds up.
- **Exactly:** `y = x*x + x` at `x = 3` gives `y = 12`; calculus says `2x + 1 = 7`. In the graph `x` has three outgoing edges (twice in `x*x`, once in `+`): the multiply contributes `3 + 3` and the add contributes `1`. The `+=` makes it 7.0; with `=` we would get the wrong 3.0 or 1.0.
- **Why it is here:** Residual connections and shared weights ([step 05](../05_initialization_and_normalization/WALKTHROUGH.md), RNNs in step 09) all rely on exactly this accumulation.

### 6. A neural network built only from Values: Neuron -> Layer -> MLP  (`main.py` lines 144-170)
```python
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
```
- **In simple words:** Build a neuron, a layer and a network out of Values, just like `nn.Linear`, but one number at a time.
- **Exactly:** `Neuron(n_in)` has `n_in` weights uniform in (-1, 1) and a bias; `__call__` computes `sum(w_i * x_i, start=b)` which creates `n_in` multiplications and additions in the graph, then `tanh` unless it is the last layer. `MLP([2, 4, 1])` makes layers of 4 and 1 neurons: `2*4 + 4 + 4*1 + 1 = 17` parameters, as printed. `parameters()` flattens everything so the training loop can update them.
- **Why it is here:** It shows that a 'neural network' is only nested function calls; any `nn.Module` is the same thing in tensor form.

### 7. Train on XOR with squared error: forward, zero grads, backward, update (the universal loop)  (`main.py` lines 173-192)
```python
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
```
- **In simple words:** Train it: predict four points, add up the squared errors, clear old gradients, backpropagate, move every parameter a little against its gradient.
- **Exactly:** `preds` builds a fresh graph; `loss` is the sum of squared errors as one Value; `p.grad = 0.0` clears accumulated gradients; `backward(loss)` fills every parameter's grad; `p.data -= 0.05 * p.grad` is gradient descent. Loss: 2.4547 at step 0, 0.2538 at step 50, 0.0027 at step 100, 0.0000 afterwards. Predictions `[0.0, 1.0, 1.0, 0.0]`.
- **Why it is here:** This is the universal training loop of [module 03](../../03_ml_foundations/04_pytorch_training_loop/WALKTHROUGH.md) in 6 lines, running on an engine you wrote.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `lambda: None` | an anonymous function that does nothing | default `_backward` for leaves |
| nested `def _backward():` | a closure: an inner function that remembers variables of its enclosing function | captures `self`, `other`, `out` |
| `Value.__add__ = _add` | attach a function to a class after it was defined (monkey-patching) | makes `a + b` call `_add(a, b)` |
| `__radd__`, `__rmul__` | used when the left operand is not a Value | `2 * v` calls `v.__rmul__(2)` |
| `isinstance(other, Value)` | type test | wrap plain floats |
| `sum(iterable, start)` | add items starting from `start` | `sum(terms, self.b)` starts from the bias |
| `reversed(list)` | iterate backwards without copying | `reversed(topo)` |
| `self.grad += ...` | in-place accumulate | needed when a node is reused |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `math.tanh`, `math.exp` | built-in `math` | scalar tanh and exp | float | float |
| `random.uniform(a, b)`, `random.seed(n)` | built-in `random` | uniform sample; seed the generator | bounds / int | float / None |
| `torch.tensor(x, requires_grad=True)` | torch | leaf that records operations | `dtype`, `requires_grad` | `Tensor` |
| `tensor.grad` | torch | the accumulated gradient after `backward()` | - | `Tensor` |

## Execution trace (real run)
```
f = 8.0 | df/da = -3.0 (=b), df/db = 2.0 (=a), df/dc = 6.0 (=2c)
graph has 10 nodes: ['*', '**2', '+'] plus the leaves
ours vs torch, max abs gradient difference: 0.0
y = x*x + x at x=3: y = 12.0, dy/dx = 7.0 (expected 7.0)
parameters: 17
  step   0 loss 2.4547
  step  50 loss 0.2538
  step 100 loss 0.0027
  step 150 loss 0.0000
  step 200 loss 0.0000
  step 250 loss 0.0000
  step 300 loss 0.0000
XOR predictions: [0.0, 1.0, 1.0, 0.0] target: [0.0, 1.0, 1.0, 0.0]
```
- `f = 8.0` and the three gradients equal the hand derivatives `b`, `a`, `2c`.
- `graph has 10 nodes`: 5 leaves (`a`, `b`, `c`, the constant 1.0, the constant 5) and 5 operations (`*`, `**2`, `+`, `*`, `+`); the set printed lists the distinct operation names.
- `max abs gradient difference: 0.0`: identical to torch, to the last bit, on this example.
- `y = 12.0, dy/dx = 7.0`: accumulation works.
- `parameters: 17`: the count from the layer sizes.
- The loss falls from 2.45 to 0.0027 in 100 steps and the four predictions round to the XOR table.

## Experiments (change ONE thing)
1. Learning rate 0.1 instead of 0.05: loss 0.0000 by step 100, then a spike to 0.0562 at step 150 and recovery to 0.0000 by step 200. A step that was fine on a flat basin overshoots when a saturated `tanh` suddenly has a bigger slope. Loss curves are not always monotone.
2. Replace `tanh` by a sigmoid built from existing ops, `1 / (1 + (-z).exp())`: the loss stalls near 1.0 (0.9608 after 300 steps), predictions about 0.5 everywhere (`[0.49, 0.502, 0.516, 0.488]`). Sigmoid outputs are never centred on zero and its slope is at most 0.25, so learning is far slower. Step 04 measures this.
3. Remove the gradient reset (`p.grad = 0.0`): the loss wanders (1.85, 0.50, 1.19, 0.32, 1.07, 1.26 at steps 50 to 300) and the predictions are wrong (`[-0.025, 0.627, 0.599, -0.175]`). Old gradients pile up and the update becomes a mix of every previous step.

## Debugging practice
BREAK IT: raise a Value to a Value power: `x = Value(2.0); y = x ** Value(2.0)`.
```
  File ".../main.py", line 58, in _pow
    out = Value(self.data ** k, (self,), f"**{k}")
                ~~~~~~~~~~^^~~
TypeError: unsupported operand type(s) for ** or pow(): 'float' and 'Value'
```
Read it: the error is raised inside our own `_pow`, on `self.data ** k`. `self.data` is a float and `k` is a `Value`, which Python cannot raise a float to. Our engine supports only a *number* as exponent (documented in a comment); a general `a ** b` needs the extra derivative `d(a^b)/db = a^b ln a`. Errors inside your own library code usually mean you hit a feature you did not implement, not a Python bug.

## Rebuild checklist
- [ ] `Value` with data, grad, children, op, backward closure
- [ ] add, mul, pow, tanh with correct local derivatives
- [ ] derive sub/div from primitives
- [ ] DFS topological sort, seed `grad = 1`, call closures in reverse
- [ ] `+=` in every closure; zero grads each training step
- [ ] verify against torch on a small expression

## The new problem (cliffhanger)
Our engine differentiates anything, and the XOR net trained fine with `tanh`. But the sigmoid experiment stalled. Why do some activations train quickly and others barely move, especially in deep networks? We need to measure gradient magnitudes layer by layer. Next: [step 04, activations and vanishing gradients](../04_activations_and_vanishing_gradients/WALKTHROUGH.md).

> **Say it aloud (interview box):** Autograd records each operation as a node holding its inputs and a local derivative. Backward sets the output gradient to 1, orders the graph topologically, and walks it in reverse, multiplying the incoming gradient by each local derivative and adding the result into every input's gradient. Gradients add because a value can be used more than once, which is also why you must zero them between steps. PyTorch does the same on tensors with fused kernels; the graph is built dynamically on each forward pass.
