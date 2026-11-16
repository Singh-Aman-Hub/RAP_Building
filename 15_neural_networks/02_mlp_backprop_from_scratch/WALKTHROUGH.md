# Step 02 — An MLP and backpropagation from scratch

## The story so far
[Step 01](../01_neuron_and_perceptron/WALKTHROUGH.md) hand-built a 2-layer network that solves XOR and noted that the step function is flat, so the perceptron rule cannot train hidden layers. [Module 03 step 02](../../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md) trained a one-layer model with hand-derived gradients and [step 03](../../03_ml_foundations/03_pytorch_tensors_autograd/WALKTHROUGH.md) showed `loss.backward()`. Here both ideas meet: we derive the gradient of a two-layer network by hand.

## The pain point
With a hidden layer, a weight in the first layer affects the loss only through the hidden units and then the output units. Which direction should it move? You cannot read the answer off the loss. And if you trust `loss.backward()` without ever having derived it, a wrong gradient in your own custom layer (a missing `/N`, a transposed matrix) silently trains a bad model.

## The spark
The chain rule. The loss is a composition of simple functions: `loss(softmax(W2 . tanh(W1 . x + b1) + b2))`. Differentiate each piece, multiply the local derivatives from the loss backwards, and reuse the numbers from the forward pass (the cache). That is **backpropagation**: reverse-mode automatic differentiation applied to a network. Then prove it right two independent ways: finite differences and torch autograd.

## Goal
Implement forward, loss and backward for a 2-layer tanh MLP with explicit shapes; verify with a numerical gradient check (error ~5e-9) and torch autograd (difference ~3e-17); train on XOR and on a 3-class spiral (99.3 percent train accuracy) where a linear model reaches only 54 percent.

## The theory (read this before the interview)
**The network.** For a batch `X` of shape `(N, n_in)`:
```
z1     = X W1 + b1           (N, n_hidden)       pre-activation
h      = tanh(z1)            (N, n_hidden)       hidden activation
logits = h W2 + b2           (N, n_out)          class scores
p      = softmax(logits)     (N, n_out)          probabilities
L      = -(1/N) sum_i log p[i, y_i]              mean cross-entropy (one number)
```
**The backward pass, line by line.** Write `dX` for `dL/dX` (same shape as `X`). Work from the loss backwards:
```
dlogits = (p - onehot(y)) / N              (N, n_out)         softmax + cross-entropy collapse to this
dW2     = h^T  dlogits                     (n_hidden, n_out)  each weight: (its input h) x (the error at its output)
db2     = sum over batch of dlogits        (n_out,)
dh      = dlogits W2^T                     (N, n_hidden)      send the error back through W2
dz1     = dh * (1 - h^2)                   (N, n_hidden)      tanh'(z) = 1 - tanh(z)^2
dW1     = X^T dz1                          (n_in, n_hidden)
db1     = sum over batch of dz1            (n_hidden,)
```
**Shape rule.** `dW` always has the shape of `W`. For `Y = A B`, `dA = dY B^T` and `dB = A^T dY`. If your shapes do not match, you transposed the wrong thing. Biases get the sum over the batch because one bias is added to every row.

**Why softmax + cross-entropy gives `p - onehot`.** `dL/dlogit_j = p_j - 1[j = y]`: the probability you gave minus 1 for the right class. That is why the gradient is large when you are confidently wrong and tiny when you are right (the 'cross-entropy gives big gradients to mistakes' property, [module 03 step 06](../../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md)).

**Why cache the forward values.** `dW2` needs `h`; `dz1` needs `h` (for `1 - h^2`); `dW1` needs `X`. Training memory is dominated by these saved activations: this is why activation memory grows with batch size and depth, and why *activation checkpointing* (recompute instead of store) exists ([module 06](../../06_llm_training_and_lora/README.md)).

**Gradient checking.** Centered finite difference: `(L(w + eps) - L(w - eps)) / (2 eps)`. Error is `O(eps^2)`. Use float64, `eps ~ 1e-5`, and compare with a *relative* error `|a - n| / (|a| + |n|)`; expect below 1e-6 for a correct gradient. It costs two forward passes per parameter, so use it only on tiny nets as a unit test, never in training.

**Gradient descent update.** `W <- W - lr * dW` for every parameter, after each full pass (full batch here; mini-batch and SGD are in [module 03 step 04](../../03_ml_foundations/04_pytorch_training_loop/WALKTHROUGH.md), better optimizers in step 06).

**Backprop versus autograd.** Backprop is the algorithm (chain rule backwards, reuse of intermediate values). Autograd is the software that does it for you on any computation graph (step 03 builds one). Same maths, different amounts of typing.

**Why depth helps (and what changes).** A hidden layer of `tanh` units bends the space; the output layer then separates in the bent space. With 1 hidden unit the spiral accuracy stays at 57 percent, with 2 units 69 percent, with 8 units 99 percent (see Experiments). A linear model on the spiral gets 54 percent.

**Common bugs.** Forgetting `/N` (gradient too big by `N`), transposed matrices, forgetting the softmax-max subtraction (overflow), using probabilities in a loss that expects logits, using `loss.sum()` vs `loss.mean()` inconsistently, and zero-initialising all weights (every hidden unit gets the same gradient and stays identical, the 'symmetry problem', step 05).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| backprop | backpropagation, reverse-mode autodiff, reverse-mode automatic differentiation | Backprop is reverse-mode autodiff applied to a network | Backprop = the algorithm; autograd = the library that implements it for any graph |
| MLP | feed-forward network, dense network, fully connected network, multi-layer perceptron | Same | 'Feed-forward' also names the 2-layer block inside a transformer |
| dense layer | linear layer, fully connected layer, `nn.Linear`, affine layer, FC | Same: `x @ W + b` | Keras says Dense, PyTorch Linear (stores `W` transposed as `(out, in)`) |
| logits | pre-softmax scores, raw scores, pre-activations of the last layer | Same | 'Logit' in statistics means log-odds; in deep learning any raw score |
| loss | cost, objective, error, criterion | Mostly same | Cost = loss averaged over the dataset; PyTorch calls the object a criterion |
| step / iteration / epoch | update, batch, pass | Step = iteration = one parameter update; epoch = one pass over all data | Full batch: 1 epoch = 1 step. Mini-batch: 1 epoch = N/batch steps |
| gradient | partial derivatives, `dL/dW`, `.grad` | Same | 'Gradient' = vector of all partials; `.grad` stores it per tensor |
| numerical gradient | finite differences, gradient check, `gradcheck` | Same idea | Numerical = approximate and slow; analytic (backprop) = exact and fast |

## Run it
```bash
cd 15_neural_networks/02_mlp_backprop_from_scratch
python main.py       # a few seconds on CPU, numpy + torch
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-21)
```python
"""
GOAL:  Write a 2-layer MLP in numpy: forward pass, loss, and the backward pass derived by hand with shapes.
       Prove the gradients right with a numerical gradient check and with torch autograd, then train the
       net on XOR and on a 3-class spiral that no single neuron could ever separate.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/02_mlp_backprop_from_scratch/main.py
EXPECTED OUTPUT (shortened):
    gradient check (numeric vs analytic): max relative error ~5e-9
    torch autograd vs hand backprop: max abs difference ~3e-17
    XOR predictions: [0, 1, 1, 0] target: [0, 1, 1, 0]
    spiral train accuracy: 0.993
NEEDS: Python + numpy + torch. A few seconds on CPU.
"""

import numpy as np
import torch

rng = np.random.default_rng(0)


```
- **In simple words:** Import numpy for the maths and torch only to cross-check our answer. Seed a generator.
- **Exactly:** The module-level `rng = np.random.default_rng(0)` is used for the spiral data; other parts pass their own generators so results do not depend on call order.
- **Why it is here:** Reproducibility: the printed accuracy 0.993 must be the same every run.

### 1. Parameters: two weight matrices and two bias vectors  (`main.py` lines 22-29)
```python
# [1] Parameters: two weight matrices and two bias vectors
def init_params(n_in, n_hidden, n_out, rng):
    return {
        "W1": rng.normal(0, 0.5, size=(n_in, n_hidden)),   # (n_in, n_hidden)
        "b1": np.zeros(n_hidden),                           # (n_hidden,)
        "W2": rng.normal(0, 0.5, size=(n_hidden, n_out)),   # (n_hidden, n_out)
        "b2": np.zeros(n_out),                              # (n_out,)
    }
```
- **In simple words:** Make the weights: two matrices and two bias vectors, stored in a dictionary so we can loop over them.
- **Exactly:** `W1: (n_in, n_hidden)`, `b1: (n_hidden,)`, `W2: (n_hidden, n_out)`, `b2: (n_out,)`. Weights are Gaussian with std 0.5, biases zero. For the XOR check net: `(2,4), (4,), (4,2), (2,)`. For the spiral: `(2,64), (64,), (64,3), (3,)`, which is 2*64+64+64*3+3 = 387 parameters.
- **Why it is here:** Random (not zero) weights break symmetry: if all hidden units started equal they would receive identical gradients and stay identical. The dict lets `for k in p` update everything generically.

### 2. Forward pass: linear -> tanh -> linear, keeping what the backward pass will need  (`main.py` lines 32-37)
```python
# [2] Forward pass: linear -> tanh -> linear, keeping what the backward pass will need
def forward(X, p):
    z1 = X @ p["W1"] + p["b1"]       # (N, n_hidden)  pre-activation of the hidden layer
    h = np.tanh(z1)                  # (N, n_hidden)  hidden activations
    logits = h @ p["W2"] + p["b2"]   # (N, n_out)     raw class scores
    return logits, (X, h)            # cache: inputs and hidden values are reused by backward
```
- **In simple words:** Push the data through the network: multiply, add, bend with tanh, multiply and add again. Save the numbers the backward pass will need.
- **Exactly:** `X @ W1` is `(N, n_in) @ (n_in, n_hidden) = (N, n_hidden)`; the bias `(n_hidden,)` broadcasts over rows. `np.tanh` squashes each value to (-1, 1). The function returns `logits` and a cache tuple `(X, h)`.
- **Why it is here:** The cache avoids recomputation: backward needs `X` for `dW1` and `h` for `dW2` and the tanh derivative. In PyTorch the same job is done by the autograd graph.

### 3. Loss: softmax then cross-entropy, averaged over the batch (module 03 step 06)  (`main.py` lines 40-48)
```python
# [3] Loss: softmax then cross-entropy, averaged over the batch (module 03 step 06)
def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))   # subtract the row max: numerically stable
    return e / e.sum(axis=1, keepdims=True)                   # (N, n_out) probabilities per row


def loss_fn(logits, y):
    probs = softmax(logits)
    return -np.log(probs[np.arange(len(y)), y]).mean(), probs  # mean of -log p(true class)
```
- **In simple words:** Turn scores into probabilities, then measure how little probability the right answer got.
- **Exactly:** `logits.max(axis=1, keepdims=True)` has shape `(N, 1)`; subtracting it leaves the softmax result unchanged but keeps `exp` from overflowing. `probs[np.arange(len(y)), y]` is fancy indexing: row i, column `y[i]`, shape `(N,)`. The loss is the mean of `-log` of those. It returns `probs` too, because backward needs them.
- **Why it is here:** This is the loss every classifier in this repo uses ([module 03 step 06](../../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md)). Without the max-subtraction, `exp(1000)` is `inf` and the loss becomes `nan`.

### 4. Backward pass: the chain rule applied layer by layer, from the loss back to the inputs  (`main.py` lines 51-65)
```python
# [4] Backward pass: the chain rule applied layer by layer, from the loss back to the inputs
def backward(y, probs, cache, p):
    X, h = cache
    N = len(y)
    dlogits = probs.copy()                      # (N, n_out)
    dlogits[np.arange(N), y] -= 1               # softmax + cross-entropy gradient is (p - onehot)
    dlogits /= N                                # because the loss is a MEAN over N samples
    grads = {}
    grads["W2"] = h.T @ dlogits                 # (n_hidden, N) @ (N, n_out) -> (n_hidden, n_out)
    grads["b2"] = dlogits.sum(axis=0)           # (n_out,)  the bias touches every sample
    dh = dlogits @ p["W2"].T                    # (N, n_out) @ (n_out, n_hidden) -> (N, n_hidden)
    dz1 = dh * (1 - h ** 2)                     # tanh'(z) = 1 - tanh(z)^2, elementwise
    grads["W1"] = X.T @ dz1                     # (n_in, N) @ (N, n_hidden) -> (n_in, n_hidden)
    grads["b1"] = dz1.sum(axis=0)               # (n_hidden,)
    return grads
```
- **In simple words:** Walk backwards: first how the loss changes with the scores, then how the scores change with W2 and the hidden values, then how those change with W1.
- **Exactly:** `dlogits = probs - onehot` (the one-hot subtraction is done by decrementing the true-class entry), divided by N since the loss is a mean. `h.T @ dlogits` is `(n_hidden, N) @ (N, n_out)`. `dh = dlogits @ W2.T` is `(N, n_out) @ (n_out, n_hidden)`. `dh * (1 - h**2)` applies `tanh'` elementwise. Every gradient has the same shape as its parameter.
- **Why it is here:** This is the whole algorithm. A transposed matrix or a missing `/N` produces a model that still runs and trains, only worse, which is why the next block checks it.

### 5. Numerical gradient check: nudge each parameter by +-eps and compare with the backward pass  (`main.py` lines 68-92)
```python
# [5] Numerical gradient check: nudge each parameter by +-eps and compare with the backward pass
def total_loss(X, y, p):
    return loss_fn(forward(X, p)[0], y)[0]


X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
p_check = init_params(2, 4, 2, np.random.default_rng(1))
loss, probs = loss_fn(forward(X_xor, p_check)[0], y_xor)
analytic = backward(y_xor, probs, forward(X_xor, p_check)[1], p_check)

eps = 1e-5
worst = 0.0
for name, arr in p_check.items():
    numeric = np.zeros_like(arr)
    for idx in np.ndindex(arr.shape):                 # every single number in this parameter
        old = arr[idx]
        arr[idx] = old + eps; up = total_loss(X_xor, y_xor, p_check)
        arr[idx] = old - eps; down = total_loss(X_xor, y_xor, p_check)
        arr[idx] = old                                # restore
        numeric[idx] = (up - down) / (2 * eps)        # centered finite difference
    rel = np.abs(numeric - analytic[name]) / (np.abs(numeric) + np.abs(analytic[name]) + 1e-12)
    worst = max(worst, rel.max())
    print(f"  {name}: shape {arr.shape}, max relative error {rel.max():.2e}")
print(f"gradient check (numeric vs analytic): max relative error {worst:.2e}")
```
- **In simple words:** A gradient detective: for every single weight, wiggle it a tiny bit up and down, see how the loss changes, and compare with what backprop claims.
- **Exactly:** `np.ndindex(arr.shape)` yields every index tuple. For each: loss at `old + 1e-5`, loss at `old - 1e-5`, then `(up - down) / (2*eps)`. The arrays are modified in place and restored. Max relative errors: `W1 5.51e-09`, `b1 3.67e-09`, `W2 4.04e-10`, `b2 1.94e-11`. The XOR net has only 2*4+4+4*2+2 = 22 parameters, so 44 forward passes.
- **Why it is here:** It catches wrong gradients with no reference implementation. Rule of thumb: relative error below 1e-6 means correct; around 1e-2 or higher means a bug.

### 6. Same network in torch: autograd must give the same gradients  (`main.py` lines 95-102)
```python
# [6] Same network in torch: autograd must give the same gradients
tp = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in p_check.items()}
t_logits = torch.tanh(torch.tensor(X_xor) @ tp["W1"] + tp["b1"]) @ tp["W2"] + tp["b2"]
t_loss = torch.nn.functional.cross_entropy(t_logits, torch.tensor(y_xor))
t_loss.backward()                                     # reverse-mode autodiff: fills .grad on every leaf
diff = max(np.abs(tp[k].grad.numpy() - analytic[k]).max() for k in tp)
print(f"torch loss {t_loss.item():.6f} vs numpy loss {loss:.6f}")
print(f"torch autograd vs hand backprop: max abs difference {diff:.2e}")
```
- **In simple words:** Ask PyTorch the same question and see whether it agrees with our hand-written maths.
- **Exactly:** `torch.tensor(v, dtype=torch.float64, requires_grad=True)` makes leaf tensors; the forward pass builds a graph; `cross_entropy` takes logits and integer labels and applies softmax itself; `t_loss.backward()` fills `.grad`. Loss `0.689411` in both; the max absolute gradient difference is `2.78e-17`, which is float64 rounding noise.
- **Why it is here:** It confirms our derivation against an independent implementation, and shows that `backward()` is nothing mystical: it computes the same seven lines.

### 7. Training loop: full-batch gradient descent, update = parameter - lr * gradient  (`main.py` lines 105-121)
```python
# [7] Training loop: full-batch gradient descent, update = parameter - lr * gradient
def train(X, y, n_hidden, lr, steps, rng, log_every):
    p = init_params(X.shape[1], n_hidden, int(y.max()) + 1, rng)
    for step in range(steps + 1):
        logits, cache = forward(X, p)
        loss, probs = loss_fn(logits, y)
        if step % log_every == 0:
            print(f"  step {step:>5}  loss {loss:.4f}  acc {(logits.argmax(1) == y).mean():.2f}")
        grads = backward(y, probs, cache, p)
        for k in p:
            p[k] -= lr * grads[k]                      # in-place gradient-descent update
    return p


print("XOR with a 4-unit hidden layer:")
p_xor = train(X_xor, y_xor, n_hidden=4, lr=0.5, steps=1000, rng=np.random.default_rng(0), log_every=250)
print("XOR predictions:", forward(X_xor, p_xor)[0].argmax(1).tolist(), "target:", y_xor.tolist())
```
- **In simple words:** The learning loop: predict, measure, compute gradients, nudge every weight downhill. Repeat.
- **Exactly:** `range(steps + 1)` so the printed final step is the loss after `steps` updates. `p[k] -= lr * grads[k]` updates in place. For XOR with 4 hidden units and `lr=0.5`: loss 0.7307 -> 0.0313 (step 250) -> 0.0029 (step 1000), accuracy 0.75 -> 1.00. Predictions `[0, 1, 1, 0]`.
- **Why it is here:** This is [module 03's](../../03_ml_foundations/04_pytorch_training_loop/WALKTHROUGH.md) five-line loop with the backward pass written out by hand.

### 8. A harder dataset: three interleaved spiral arms (not separable by any line)  (`main.py` lines 124-143)
```python
# [8] A harder dataset: three interleaved spiral arms (not separable by any line)
def make_spiral(points_per_class=100, classes=3, noise=0.2, rng=rng):
    X = np.zeros((points_per_class * classes, 2))
    y = np.zeros(points_per_class * classes, dtype=int)
    for c in range(classes):
        ix = range(points_per_class * c, points_per_class * (c + 1))
        r = np.linspace(0.0, 1, points_per_class)                                  # radius grows along the arm
        t = np.linspace(c * 4, (c + 1) * 4, points_per_class) + rng.normal(0, noise, points_per_class)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y


Xs, ys = make_spiral()
print("spiral data:", Xs.shape, "labels:", np.bincount(ys))
p_sp = train(Xs, ys, n_hidden=64, lr=1.0, steps=3000, rng=np.random.default_rng(0), log_every=1000)
print(f"spiral train accuracy: {(forward(Xs, p_sp)[0].argmax(1) == ys).mean():.3f}")

# TRY IT: change n_hidden for the spiral to 2, 8, 64, 256 and note the final accuracy.
# BREAK IT: delete the `dlogits /= N` line. The gradient check then fails by a factor of N (a scale bug).
```
- **In simple words:** Make a really tangled dataset: three arms spiralling out from the centre, each a different class.
- **Exactly:** `make_spiral` returns `X: (300, 2)` and `y: (300,)` with 100 points per class. Radius `r` grows from 0 to 1; the angle `t` sweeps 4 radians per arm plus Gaussian noise 0.2. Training with 64 hidden units, `lr=1.0`, 3000 steps takes the loss from 1.4102 to 0.0268 and the accuracy from 0.29 to 0.993. A linear softmax model on the same data scores 0.54.
- **Why it is here:** XOR has 4 points; the spiral shows that the same code scales to a problem a straight line cannot solve. It also shows the point of the module: a nonlinear hidden layer bends the space.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `probs[np.arange(N), y]` | fancy indexing: pick element `y[i]` from row `i` | `a[[0,1],[2,0]]` -> `[a[0,2], a[1,0]]` |
| `dlogits[np.arange(N), y] -= 1` | in-place subtract 1 at the true-class positions | turns `p` into `p - onehot` |
| `keepdims=True` | keep the reduced axis as size 1 so broadcasting works | `(N,3).max(1, keepdims=True)` -> `(N,1)` |
| `np.ndindex(shape)` | iterate over all index tuples of a shape | `(2,2)` -> `(0,0),(0,1),(1,0),(1,1)` |
| `{k: f(v) for k, v in d.items()}` | dict comprehension | `tp` in block 6 |
| `np.c_[a, b]` | stack 1-D arrays as columns | `(n,), (n,)` -> `(n, 2)` |
| `a ** 2` | elementwise power | `h ** 2` |
| `f"{step:>5}"` | right-align in width 5 | `'  250'` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `np.tanh` | numpy | hyperbolic tangent, range (-1, 1) | array | `ndarray` |
| `np.exp`, `np.log` | numpy | elementwise exp / natural log | array | `ndarray` |
| `np.bincount(y)` | numpy | count occurrences of each integer label | int array | `ndarray` |
| `np.linspace(a, b, n)` | numpy | n evenly spaced numbers | start, stop, n | `ndarray` |
| `torch.tensor(..., requires_grad=True)` | torch | leaf tensor tracked by autograd | `dtype`, `requires_grad` | `Tensor` |
| `F.cross_entropy(logits, y)` | torch | softmax + negative log-likelihood, mean over batch | logits `(N,C)`, int labels `(N,)` | scalar tensor |
| `tensor.backward()` | torch | reverse-mode autodiff from a scalar | - | None; fills `.grad` |

## Execution trace (real run)
```
  W1: shape (2, 4), max relative error 5.51e-09
  b1: shape (4,), max relative error 3.67e-09
  W2: shape (4, 2), max relative error 4.04e-10
  b2: shape (2,), max relative error 1.94e-11
gradient check (numeric vs analytic): max relative error 5.51e-09
torch loss 0.689411 vs numpy loss 0.689411
torch autograd vs hand backprop: max abs difference 2.78e-17
XOR with a 4-unit hidden layer:
  step     0  loss 0.7307  acc 0.75
  step   250  loss 0.0313  acc 1.00
  step   500  loss 0.0080  acc 1.00
  step   750  loss 0.0043  acc 1.00
  step  1000  loss 0.0029  acc 1.00
XOR predictions: [0, 1, 1, 0] target: [0, 1, 1, 0]
spiral data: (300, 2) labels: [100 100 100]
  step     0  loss 1.4102  acc 0.29
  step  1000  loss 0.0521  acc 0.99
  step  2000  loss 0.0335  acc 0.99
  step  3000  loss 0.0268  acc 0.99
spiral train accuracy: 0.993
```
- The four lines `W1`, `b1`, `W2`, `b2` are the per-parameter relative errors from the finite-difference check: all between 2e-11 and 6e-9, so our derivation is right.
- `torch loss 0.689411 vs numpy loss 0.689411`: the forward passes agree to six digits; the gradients agree to 3e-17.
- XOR: starts at 75 percent (3 of 4 right, like the best perceptron line in step 01) and reaches 100 percent by step 250.
- Spiral: loss 1.41 is about `ln 3 = 1.10` plus the damage of random weights; at step 1000 the loss is 0.0521 with 99 percent accuracy; the last 2000 steps polish it to 0.0268. Accuracy 0.993 means 2 of 300 points are wrong.

## Experiments (change ONE thing)
1. Hidden width on the spiral (3000 steps, `lr=1`): 1 unit 0.573, 2 units 0.693, 8 units 0.990, 64 units 0.993, 256 units 0.993. The curve flattens: past a point, more width only costs compute. (Train accuracy only: this says nothing about generalisation, see step 07.)
2. Learning rate (64 units): `lr=0.01` loss 0.6907 (acc 0.563, still far from done), `lr=0.1` loss 0.1255 (0.970), `lr=1.0` loss 0.0268 (0.993), `lr=10` loss 0.0385 (0.983). Too small is slow; too big overshoots and is noisier, as in [module 03 step 02](../../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md).
3. A linear softmax model (no hidden layer) trained with the same loop reaches 0.54 on the spiral. The hidden layer is what gives 0.99.

## Debugging practice
BREAK IT: delete the line `dlogits /= N`. The program runs, but the check prints
```
  W1: shape (2, 4), max relative error 6.00e-01
  b1: shape (4,), max relative error 6.00e-01
  W2: shape (4, 2), max relative error 6.00e-01
  b2: shape (2,), max relative error 6.00e-01
```
Every parameter is off by the same factor: the analytic gradient is `N = 4` times the numerical one. With our relative-error formula a factor of 4 gives `|4g - g| / (|4g| + |g|) = 3/5 = 0.6`, exactly the printed number. Uniform error across all parameters points to something near the loss (a scale), an error confined to one parameter points to that layer's derivative. In training this bug would not crash; it would act like a learning rate 4 (or 300) times larger and may diverge.

## Rebuild checklist
- [ ] init: `W` Gaussian, `b` zero, shapes `(in, out)`
- [ ] forward with a cache
- [ ] stable softmax and mean cross-entropy
- [ ] backward: `p - onehot`, divide by N, `dW = input^T @ dout`, `dh = dout @ W^T`, multiply by the activation derivative
- [ ] numerical gradient check with relative error
- [ ] compare with `loss.backward()`
- [ ] the update `p -= lr * grad`

## The new problem (cliffhanger)
We derived backprop by hand for exactly one architecture. Change the activation or add a layer and you re-derive everything. PyTorch does that for any function by recording operations in a graph. What does that graph look like, and how does `backward()` walk it? Next: [step 03, a tiny autograd engine](../03_tiny_autograd_engine/WALKTHROUGH.md).

> **Say it aloud (interview box):** An MLP alternates affine maps and nonlinearities. Training minimises a loss by gradient descent, and backprop computes all the gradients in one backward sweep using the chain rule and the values cached in the forward pass. For softmax with cross-entropy the gradient at the logits is probabilities minus the one-hot target, divided by the batch size for a mean loss. Each weight gradient is the layer's input transposed times the incoming error, so shapes match the weights. I verify with a finite-difference gradient check, which catches bugs like a missing 1/N, and I compare with autograd.
