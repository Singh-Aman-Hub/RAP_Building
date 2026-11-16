# Step 01 — The neuron, the perceptron and logistic regression

## The story so far
**Prerequisites:** [03 step 02, linear regression by gradient descent](../../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md) (the loop and the `y_hat - y` gradient) and [03 step 06, softmax and cross-entropy](../../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md) (log-loss and `p - onehot`). Logistic regression is where those two meet.

[Module 03](../../03_ml_foundations/README.md) trained a linear model `y = w*x + b` by gradient descent and used PyTorch `nn.Linear`. That model has a very specific shape: a weighted sum plus a bias. This step opens the box and shows that this one unit, followed by a nonlinearity, is the **artificial neuron**, the atom of every network in this module and of every transformer in [module 04](../../04_llm_fundamentals/README.md).

## The pain point
One neuron draws one straight line (a hyperplane in higher dimensions). That is enough for AND and OR, but not for XOR, the smallest problem where the classes sit on opposite diagonals of a square. Minsky and Papert pointed this out in 1969 and it helped trigger the first 'AI winter'. If you cannot say *why* a single neuron fails, you cannot say why depth is needed.

## The spark
(Part 1) Keep the neuron, add a second layer. A hidden layer re-describes the inputs (here: 'is at least one on' and 'are both on') so that the output neuron only needs a straight line in the new description. Depth plus a nonlinearity turns lines into curves.

(Part 2) The perceptron's step function is flat, so it gives no gradient and no probability, and it never settles on overlapping classes. Replace the step with a smooth **sigmoid** and score the output with **binary cross-entropy**: now there is a gradient (`p - y` times the input), the loss is convex, and the output is a probability. That model, one neuron with a sigmoid trained with log-loss, is **logistic regression**, and it is the building block of every classification head in this repo.

## Goal
Part 1: implement a neuron and the perceptron learning rule in numpy, watch it converge on AND, OR and two blobs, watch it fail on XOR, prove by brute force that no line works, and hand-build a 2-layer network that solves XOR. Part 2: implement binary logistic regression from scratch (sigmoid, BCE, the gradient `(p - y) x` with a numerical check, gradient descent), read its decision boundary and probabilities, compare it with the perceptron on the same data, check its calibration, and match scikit-learn's coefficients.

## The theory (read this before the interview)
**The neuron.** Given an input vector `x` (n numbers), a weight vector `w` (n numbers) and a scalar bias `b`:
```
z = w1*x1 + w2*x2 + ... + wn*xn + b  =  x . w + b        (pre-activation, also called logit or net input)
a = activation(z)                                       (the neuron's output)
```
- The **weights** say how much each input matters and with which sign. The **bias** shifts the threshold: it is the value of `z` when all inputs are 0.
- Geometrically `x . w + b = 0` is a line (2 inputs), a plane (3), a hyperplane (n). `w` points perpendicular to it, `b` moves it away from the origin. The neuron answers 'which side of the hyperplane is x on?'.

**Perceptron (Rosenblatt, 1958).** A neuron with a step activation, `a = 1 if z >= 0 else 0`, and this learning rule, applied sample by sample:
```
pred   = step(x . w + b)
error  = y - pred                  # -1, 0 or +1
w     += lr * error * x            # only changes when wrong
b     += lr * error
```
If the data are **linearly separable**, the **perceptron convergence theorem** guarantees the rule finds a separating line after a finite number of mistakes. The number depends on the margin: roughly `(R / margin)^2`, where `R` is the largest input norm. If the data are not separable, the rule never settles and keeps cycling.

**Linear separability.** A dataset is linearly separable if some hyperplane puts all class-0 points on one side and all class-1 points on the other.

| Gate | Truth table (00, 01, 10, 11) | Separable? | One neuron? |
|---|---|---|---|
| AND | 0 0 0 1 | yes | yes, e.g. `w=[1,1], b=-1.5` |
| OR | 0 1 1 1 | yes | yes, e.g. `w=[1,1], b=-0.5` |
| XOR | 0 1 1 0 | **no** | **no** |

*Proof sketch for XOR.* Suppose `w1*0+w2*0+b < 0` (so `b<0`), `w2+b >= 0`, `w1+b >= 0`, and `w1+w2+b < 0`. Add the two middle inequalities: `w1+w2+2b >= 0`. But the last says `w1+w2 < -b`, so `w1+w2+2b < b < 0`. Contradiction.

**What fixes it.** Add a hidden layer with a nonlinearity: `out = step(W2 . step(W1 . x + b1) + b2)`. The hidden units are two lines; the output combines them. Without a nonlinearity (identity activation) stacking layers collapses to one linear map, `W2 (W1 x) = (W2 W1) x`, and you are back to one line. A second fix is a hand-made **feature cross** `x1*x2`: with that third input XOR becomes separable (the experiments show a perceptron then converges). Hidden layers learn such features instead of you designing them.

**Universal approximation (preview).** A network with one hidden layer and enough units can approximate any continuous function on a bounded domain (Cybenko 1989, Hornik 1991). It says nothing about how many units, or how to find the weights. Depth makes it efficient.

**Logistic regression (a neuron with a sigmoid and a probabilistic loss).** For a binary label `y in {0, 1}`:
```
z = w . x + b                      (the logit = log-odds:  z = ln( p / (1 - p) ))
p = sigmoid(z) = 1 / (1 + e^-z)    (probability that y = 1; always strictly between 0 and 1)
L = -[ y ln p + (1 - y) ln(1 - p) ]            (binary cross-entropy = log loss = negative log-likelihood, per sample)
```
- **Why this loss:** assume `y` is a coin flip with probability `p`. The likelihood of one observation is `p^y (1-p)^(1-y)`. Maximising the likelihood of the whole dataset = minimising the sum of `-ln` of it = the BCE. So BCE is not an arbitrary choice: it is **maximum likelihood** for a Bernoulli model.
- **The gradient, step by step.** `dL/dp = -y/p + (1-y)/(1-p)`. The sigmoid's derivative is `dp/dz = p (1-p)`. Multiply: `dL/dz = -y(1-p) + (1-y)p = p - y`. And `dz/dw = x`, `dz/db = 1`. Therefore
```
dL/dw = (p - y) * x          dL/db = (p - y)            (mean over the batch: X^T (p - y) / n)
```
  The `1/p` and `p(1-p)` cancel, which is why sigmoid + BCE is a natural pair (squared error with a sigmoid would keep a `p(1-p)` factor that vanishes when the unit saturates: slow learning exactly when the model is confidently wrong).
- **Update:** `w -= lr * X^T (p - y) / n`. Identical in shape to linear regression's `w -= lr * X^T (y_hat - y) / n`: the residual is `prediction - target` in both. Only `y_hat` became `sigmoid(z)`. The softmax version (`p - onehot`) for K classes is [03 step 06](../../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md).
- **Decision boundary:** `p >= 0.5` exactly when `z >= 0`, so the boundary is the same kind of hyperplane as the perceptron's. The weights are the log-odds change per unit of each feature: `e^w1` multiplies the odds when `x1` rises by 1. Different threshold than 0.5 (cost-sensitive) moves the boundary parallel to itself.
- **Convex and unique.** The BCE of a linear model is convex in `(w, b)`: no local minima, gradient descent with a small enough learning rate decreases it every step. If the classes are perfectly separable, the minimum is at infinity (`|w| -> inf`, probabilities -> 0/1), so use **L2 regularisation** (`C` in sklearn, [step 07](../07_regularization/WALKTHROUGH.md)) or early stopping.
- **Perceptron vs logistic regression.** Same hyperplane family. Perceptron: step output, update only on mistakes, converges only if separable, no probabilities. Logistic regression: smooth output, update on every sample proportional to how wrong the probability is (a confident miss moves the line a lot, a correct confident point almost not at all), works on overlapping classes, gives probabilities. Neither can solve XOR (a single hyperplane), but with hidden layers the logistic neuron becomes the sigmoid unit of an MLP ([step 02](../02_mlp_backprop_from_scratch/WALKTHROUGH.md)).
- **Calibration.** A model is calibrated if, among all points with `p ~ 0.8`, about 80% are class 1. Logistic regression fit by BCE on well-specified data is approximately calibrated; the *ranking* quality (AUC) and the *probability* quality (BCE, ECE, Brier score) are different things, and accuracy only sees the sign of `z`. Scaling the logits by a factor > 1 leaves every decision unchanged but makes the model overconfident. Post-hoc fix: **temperature scaling** (divide logits by `T`, choose `T` on validation data).
- **Numerical safety.** Compute `sigmoid` from `exp(-|z|)` (no overflow), clip probabilities before `ln`, and in real libraries prefer the fused `BCEWithLogitsLoss` / `log_softmax` which work on `z` directly (see [03 step 06](../../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md)).

**Where the perceptron still lives.** Every `nn.Linear` followed by a nonlinearity is a layer of such neurons; the learning rule is replaced by gradient descent on a smooth loss (steps 02-03), because the step function has zero gradient almost everywhere.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| weights | parameters, coefficients, `w`, `theta` | Weights are one kind of parameter | Parameters = weights + biases (+ norm scales, embeddings...) |
| bias | intercept, offset, `b` | Yes (same role) | Not the same as 'bias' in the fairness or bias-variance sense |
| pre-activation | logit, net input, `z`, weighted sum | Same number | 'Logit' is mostly used for the last layer's scores; `z` for any layer |
| activation | nonlinearity, transfer function, squashing function | Same thing | Also used for the output `a = f(z)` itself ('activations in memory') |
| perceptron | single-layer perceptron, threshold unit, McCulloch-Pitts-style neuron | Close | Perceptron = step neuron + the specific learning rule; a modern neuron uses a smooth activation and gradient descent |
| MLP | multi-layer perceptron, feed-forward network, dense net, fully connected net | Same | 'MLP' in a transformer paper means the per-token two-layer block |
| hyperplane | decision boundary, separating surface | Boundary of a linear model | Nonlinear models have curved decision boundaries |
| linearly separable | linearly classifiable | Same | Says nothing about real data being noise-free |
| logistic regression | logit model, logit regression, maximum-entropy classifier (binary case), sigmoid classifier, a single neuron with sigmoid + BCE | Same model, different communities | Statistics fits it by maximum likelihood (IRLS, L-BFGS); ML trains it by gradient descent; it is *classification*, despite the word 'regression' |
| sigmoid | logistic function, logistic sigmoid, expit, squashing function | Same function | `1/(1+e^-z)`; `scipy.special.expit`; not 'tanh' (which maps to (-1,1)) |
| binary cross-entropy (BCE) | log loss, logistic loss, negative log-likelihood (NLL) of a Bernoulli, `BCELoss` | Same number | Cross-entropy is the general name; 'log loss' is the sklearn/Kaggle name; with logits use `BCEWithLogitsLoss` |
| logit | log-odds, pre-activation of the output unit | Same thing | `logit(p) = ln(p/(1-p))` is the inverse of the sigmoid |
| odds | `p / (1-p)` | Not the same as probability | p = 0.75 means odds 3 to 1; `e^z` = odds |
| calibration | reliability, probability calibration | Same | Whether `p = 0.8` happens 80% of the time; ECE and the reliability diagram measure it |
| temperature scaling | logit scaling, Platt-like post-hoc calibration | Related to Platt scaling | One scalar `T` dividing the logits, fitted on validation data; changes confidence, not the predicted class |

## Run it
```bash
cd 15_neural_networks/01_neuron_and_perceptron
python main.py       # instant (< 1 s); numpy only, scikit-learn optional for block 12
STEPS=500 LR=0.1 python main.py   # knobs for the logistic regression part
```
From the repo root: `.venv/bin/python 15_neural_networks/01_neuron_and_perceptron/main.py`.

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets. Blocks 0-6 are Part 1 (the perceptron); blocks 7-12 are Part 2 (logistic regression).

### 0. Docstring and imports  (`main.py` lines 1-26)
```python
"""
GOAL:  Build one artificial neuron (weighted sum + bias + activation), train it with the perceptron
       learning rule, and SEE what it can and cannot do: it learns AND / OR / a separable blob, but it
       can never learn XOR, because XOR is not linearly separable. A tiny hidden layer fixes that.
       PART 2 (blocks 7-12): the smooth version of the same neuron, binary logistic regression from scratch
       (sigmoid + binary cross-entropy + gradient (p - y) x), compared with the perceptron and with sklearn.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/01_neuron_and_perceptron/main.py
EXPECTED OUTPUT (shortened):
    z = x.w + b = 0.1 | step(z) = 1
    AND : converged at epoch 6, w=[2. 1.], b=-3.0, accuracy 4/4
    XOR : never converged in 100 epochs, errors per epoch (last 6): [4, 4, 4, 4, 4, 4], ...
    best XOR accuracy over 20,000 random lines: 3 / 4
    hand-built 2-layer net on XOR: [0, 1, 1, 0] target: [0, 1, 1, 0]
    gradient check: max |analytic - numeric| ~ 1e-10
    logistic regression: loss 0.693 -> ~0.32, test accuracy ~0.87; perceptron on the same overlapping data: never converges
    calibration table (predicted vs observed) and the 3x overconfident copy; sklearn coefficients ~ ours
NEEDS: Python + numpy (instant). Optional scikit-learn for block 12 (skipped with a message if missing).
"""

import os

import numpy as np

rng = np.random.default_rng(0)  # seeded: same numbers on every run
```
- **In simple words:** Load numpy and create a seeded random generator, so every run prints the same numbers.
- **Exactly:** `np.random.default_rng(0)` returns a `Generator` object; later calls like `rng.normal(...)` draw from a reproducible stream. The docstring documents the goal and how to run.
- **Why it is here:** Without a seed, the blob data and the random-line search change on each run and you cannot compare results.

### 1. A neuron is a weighted sum plus a bias, passed through an activation  (`main.py` lines 29-44)
```python
# [1] A neuron is a weighted sum plus a bias, passed through an activation
def step(z):
    """Heaviside step activation: 1 if z >= 0 else 0 (the classic perceptron activation)."""
    return (z >= 0).astype(int)


def neuron(x, w, b, activation=step):
    """x: (n_features,) or (batch, n_features); w: (n_features,); b: scalar -> output per sample."""
    z = x @ w + b            # weighted sum + bias, the "pre-activation" (logit); shape () or (batch,)
    return activation(z)     # squash / threshold it


x = np.array([1.0, 2.0])
w = np.array([0.5, -0.25])
b = 0.1
print("z = x.w + b =", x @ w + b, "| step(z) =", neuron(x, w, b))
```
- **In simple words:** A neuron multiplies each input by a weight, adds them up, adds a bias and decides what to output. Think of a voting machine where each input has a different voting power and the bias is the entry fee.
- **Exactly:** `x @ w` is the dot product, a scalar for a single sample or shape `(batch,)` for a `(batch, n_features)` matrix. For `x=[1,2], w=[0.5,-0.25], b=0.1`: `0.5 - 0.5 + 0.1 = 0.1`, so `z = 0.1` and `step(z) = 1` (the printed line). `step` uses `(z >= 0).astype(int)`; the `>=` means exactly zero counts as 1.
- **Why it is here:** This is the one formula every later step reuses; `nn.Linear` is exactly `x @ W.T + b`.

### 2. The perceptron learning rule: nudge the weights only when the neuron is wrong  (`main.py` lines 47-64)
```python
# [2] The perceptron learning rule: nudge the weights only when the neuron is wrong
def train_perceptron(X, y, epochs=100, lr=1.0):
    """X: (n, d) inputs, y: (n,) labels in {0,1}. Returns weights, bias, history of errors per epoch."""
    w = np.zeros(X.shape[1])
    b = 0.0
    history = []
    for epoch in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            pred = int(xi @ w + b >= 0)       # forward pass for ONE sample
            update = lr * (yi - pred)         # 0 if right, +lr if we said 0 but truth is 1, -lr if the opposite
            w += update * xi                  # move the boundary toward the missed point
            b += update
            errors += int(update != 0)
        history.append(errors)
        if errors == 0:                       # a full clean pass: the data is separated, stop
            break
    return w, b, history
```
- **In simple words:** Go through the samples one at a time. If the neuron is right, do nothing. If it is wrong, push the line toward the point it got wrong.
- **Exactly:** `w` and `b` start at zero. `update = lr * (yi - pred)` is `0` when correct, `+lr` when truth is 1 and prediction 0 (increase `w . x`), `-lr` when truth is 0 and prediction 1. `history` holds the number of mistakes per epoch; zero mistakes in a full pass means converged. `w += update * xi` modifies the array in place.
- **Why it is here:** This is the oldest learning algorithm still taught: it shows 'learn from mistakes' in 4 lines and contrasts with the gradient-based update of step 02.

### 3. Logic gates as tiny datasets: AND and OR are separable, XOR is not  (`main.py` lines 67-77)
```python
# [3] Logic gates as tiny datasets: AND and OR are separable, XOR is not
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
gates = {"AND": np.array([0, 0, 0, 1]), "OR": np.array([0, 1, 1, 1]), "XOR": np.array([0, 1, 1, 0])}

for name, y in gates.items():
    w, b, hist = train_perceptron(X, y, epochs=100)
    acc = int((neuron(X, w, b) == y).sum())
    if hist[-1] == 0:
        print(f"{name:<4}: converged at epoch {len(hist)}, w={w}, b={b}, accuracy {acc}/4")
    else:
        print(f"{name:<4}: never converged in {len(hist)} epochs, errors per epoch (last 6): {hist[-6:]}, final accuracy {acc}/4")
```
- **In simple words:** Turn the logic gates into four-row tables of data and try to learn each one.
- **Exactly:** `X` is `(4, 2)`, the four input pairs; each `y` is `(4,)`. AND converges at epoch 6 with `w=[2,1], b=-3`; OR at epoch 4 with `w=[1,1], b=-1`; XOR runs all 100 epochs with 4 mistakes in every pass (final accuracy 2/4 because the last weights are just where the cycle stopped).
- **Why it is here:** It is the smallest experiment that shows the limit of one neuron. If AND works and XOR does not, with the same code, the limitation is the model, not a bug.

### 4. A separable blob dataset: the rule finds *a* separating line (not the best one)  (`main.py` lines 80-87)
```python
# [4] A separable blob dataset: the rule finds *a* separating line (not the best one)
n = 100
blob0 = rng.normal([-2, -2], 0.7, size=(n, 2))      # class 0 around (-2,-2)
blob1 = rng.normal([+2, +2], 0.7, size=(n, 2))      # class 1 around (+2,+2)
Xb = np.vstack([blob0, blob1])                      # (200, 2)
yb = np.array([0] * n + [1] * n)                    # (200,)
w, b, hist = train_perceptron(Xb, yb, epochs=50)
print(f"blobs: converged after {len(hist)} epochs, accuracy {(neuron(Xb, w, b) == yb).mean():.2f}, w={w.round(2)}, b={b}")
```
- **In simple words:** A realistic separable problem: two clouds of points far from each other. The perceptron finds some line between them.
- **Exactly:** `rng.normal(mean, 0.7, size=(100, 2))` draws 100 2-D points per cloud, stacked into `Xb` of shape `(200, 2)`. It converges after 2 epochs with accuracy 1.00. The boundary is just one valid separator, with no notion of margin; an SVM or logistic regression would pick a better one.
- **Why it is here:** It shows the theorem in action: separable data means convergence, and a wide margin means few epochs.

### 5. Why XOR fails: no line separates it. Brute-force a million lines to prove it  (`main.py` lines 90-96)
```python
# [5] Why XOR fails: no line separates it. Brute-force a million lines to prove it
best = 0
for _ in range(20000):
    w_try = rng.normal(size=2) * 3
    b_try = rng.normal() * 3
    best = max(best, int((neuron(X, w_try, b_try) == gates["XOR"]).sum()))
print("best XOR accuracy over 20,000 random lines:", best, "/ 4")
```
- **In simple words:** Try 20,000 random lines on XOR and keep the best score. If any line worked, we would see 4/4.
- **Exactly:** Each try samples `w_try` (2 values), `b_try` (1 value), computes the number of correct labels with `neuron(X, ...)`. The best is `3 / 4`, never 4. This is empirical evidence, not a proof (the proof is in the theory section).
- **Why it is here:** It converts 'XOR is not separable' from a claim into a number you can quote, and it shows that even the best linear model gets one point wrong.

### 6. The fix: a hidden layer. Hand-picked weights: h1 = OR(x1,x2), h2 = AND(x1,x2), out = h1 AND NOT h2  (`main.py` lines 99-108)
```python
# [6] The fix: a hidden layer. Hand-picked weights: h1 = OR(x1,x2), h2 = AND(x1,x2), out = h1 AND NOT h2
W1 = np.array([[1.0, 1.0],     # (2 inputs, 2 hidden): column 0 = OR unit, column 1 = AND unit
               [1.0, 1.0]])
b1 = np.array([-0.5, -1.5])    # OR fires if sum >= 0.5, AND fires if sum >= 1.5
W2 = np.array([1.0, -1.0])     # output = h_or - h_and
b2 = -0.5
h = step(X @ W1 + b1)          # (4, 2) hidden activations
out = step(h @ W2 + b2)        # (4,)
print("hidden layer activations:\n", h)
print("hand-built 2-layer net on XOR:", out.tolist(), "target:", gates["XOR"].tolist())
```
- **In simple words:** Give the network a middle layer. Neuron 1 says 'at least one input is on', neuron 2 says 'both are on'. The output says 'first yes, second no'.
- **Exactly:** `W1` is `(2, 2)`: both columns have weights `[1, 1]` so `X @ W1` has shape `(4, 2)` and each hidden unit sums the inputs; thresholds `b1 = [-0.5, -1.5]` turn column 0 into OR and column 1 into AND. The printed hidden matrix rows are `[0 0], [1 0], [1 0], [1 1]`. `W2 = [1, -1]`, `b2 = -0.5` computes `h_or - h_and - 0.5 >= 0`, which is true only for `[1, 0]`. The output `[0, 1, 1, 0]` equals the XOR target.
- **Why it is here:** It proves depth plus a nonlinearity solves what width alone cannot, and it is the picture to draw on a whiteboard. Step 02 learns such weights automatically.

### Part 2 banner and knobs  (`main.py` lines 110-115)
```python
# =====================================================================================================
# PART 2 -- LOGISTIC REGRESSION FROM SCRATCH (the smooth neuron: sigmoid + binary cross-entropy)
# Prerequisites: 03_ml_foundations/02 (gradient descent on linear regression), 03_ml_foundations/06 (softmax + CE)
# =====================================================================================================
STEPS = int(os.environ.get("STEPS", 2000))       # gradient-descent steps for the logistic regression
LR = float(os.environ.get("LR", 0.5))            # learning rate
```
- **In simple words:** A banner that separates the two halves of the file, and two knobs you can change from the shell without editing the code (`STEPS=500 LR=0.1 python main.py`).
- **Exactly:** `os.environ.get(name, default)` returns a *string* (or the default), so `int(...)` and `float(...)` convert it. `import os` was added at the top (line 22).
- **Why it is here:** Learning rate and step count are the first two things to vary when training misbehaves.

### 7. Sigmoid, binary cross-entropy and the gradient (p - y) x  (`main.py` lines 118-148)
```python
# [7] Sigmoid, binary cross-entropy (BCE) and the gradient (p - y) x
def sigmoid(z):
    """1 / (1 + e^-z), written so that exp() never overflows: use e^-z for z >= 0 and e^z / (1 + e^z) otherwise."""
    z = np.asarray(z, dtype=float)
    ez = np.exp(-np.abs(z))                       # always in (0, 1]: safe
    return np.where(z >= 0, 1 / (1 + ez), ez / (1 + ez))


def bce(p, y, eps=1e-12):
    """Mean binary cross-entropy (log loss, negative log-likelihood): -mean[ y log p + (1-y) log(1-p) ]."""
    p = np.clip(p, eps, 1 - eps)                  # log(0) = -inf: keep p strictly inside (0, 1)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def logreg_grad(X, y, w, b):
    """Gradient of the mean BCE. With p = sigmoid(Xw + b): dL/dw = X^T (p - y) / n, dL/db = mean(p - y).
    Derivation: dL/dp = -(y/p) + (1-y)/(1-p);  dp/dz = p(1-p);  product = p - y;  dz/dw = x  ->  (p - y) x."""
    r = sigmoid(X @ w + b) - y                    # (n,) residual: how far each probability is from its label
    return X.T @ r / len(y), float(r.mean())


# numerical gradient check: nudge each parameter by +-h and compare the slope with the formula
rng_lr = np.random.default_rng(42)
Xc, yc = rng_lr.normal(size=(8, 2)), rng_lr.integers(0, 2, size=8)
wc, bc, h = rng_lr.normal(size=2), 0.3, 1e-6
gw, gb = logreg_grad(Xc, yc, wc, bc)
num = np.array([(bce(sigmoid(Xc @ (wc + h * e) + bc), yc) - bce(sigmoid(Xc @ (wc - h * e) + bc), yc)) / (2 * h) for e in np.eye(2)])
num_b = (bce(sigmoid(Xc @ wc + bc + h), yc) - bce(sigmoid(Xc @ wc + bc - h), yc)) / (2 * h)
print(f"\n[logreg] sigmoid(0)={float(sigmoid(0)):.2f}  sigmoid(+-1000)={float(sigmoid(-1000)):.1f}/{float(sigmoid(1000)):.1f} (no overflow)  "
      f"BCE of a coin flip p=0.5: {bce(np.array([0.5]), np.array([1])):.3f} (= ln 2)")
print(f"gradient check: max |analytic - numeric| = {max(np.abs(gw - num).max(), abs(gb - num_b)):.1e}")
```
- **In simple words:** Three small functions: squash any number into a probability (`sigmoid`), score a probability against a 0/1 label with a penalty that explodes for confident mistakes (`bce`), and compute which way to move each weight (`logreg_grad`). Then check the formula against a slow, independent estimate.
- **Exactly:** `sigmoid(z) = 1/(1+e^-z)`, computed from `exp(-|z|)` so it never overflows (the naive form overflows for `z = -1000`). `bce(p, y) = -mean(y ln p + (1-y) ln(1-p))`; `np.clip` keeps `p` away from 0 and 1 because `ln 0 = -inf`. The gradient of the mean BCE is `X^T (p - y) / n` for `w` and `mean(p - y)` for `b`: the sigmoid derivative `p(1-p)` cancels the `1/p` and `1/(1-p)` of the loss, leaving the plain residual. The numerical check nudges each parameter by `+-1e-6`, takes the central difference of the loss, and the printed maximum difference is about `4e-11`. A coin-flip prediction scores `ln 2 = 0.693`.
- **Why it is here:** This is the whole algorithm. The residual `p - y` is the same form as linear regression's `y_hat - y` ([module 03 step 02](../../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md)) and as softmax cross-entropy's `p - onehot` ([step 06](../../03_ml_foundations/06_softmax_cross_entropy/WALKTHROUGH.md)): one formula, three models.

### 8. Train it: full-batch gradient descent on overlapping clouds  (`main.py` lines 151-172)
```python
# [8] Train it: full-batch gradient descent on two OVERLAPPING Gaussian clouds (so no line is perfect)
def train_logreg(X, y, steps=STEPS, lr=LR):
    w, b = np.zeros(X.shape[1]), 0.0
    losses = [bce(sigmoid(X @ w + b), y)]            # loss BEFORE any step: w=0 -> p=0.5 everywhere -> ln 2
    for _ in range(steps):
        gw, gb = logreg_grad(X, y, w, b)
        w -= lr * gw                              # same update as linear regression (module 03 step 02) ...
        b -= lr * gb                              # ... only the gradient formula and the loss changed
        losses.append(bce(sigmoid(X @ w + b), y))
    return w, b, losses


m = 200
Xl = np.vstack([rng_lr.normal([-1, -1], 1.2, size=(m, 2)), rng_lr.normal([+1, +1], 1.2, size=(m, 2))])   # (400, 2)
yl = np.array([0] * m + [1] * m)
perm = rng_lr.permutation(len(yl))                # shuffle, then hold out 25% to test on unseen points
tr, te = perm[:300], perm[300:]
w_lr, b_lr, losses = train_logreg(Xl[tr], yl[tr])
p_te = sigmoid(Xl[te] @ w_lr + b_lr)
print(f"\nlogreg GD: loss before training / after {STEPS // 10} / after {STEPS} steps: {losses[0]:.3f} / {losses[STEPS // 10]:.3f} / {losses[-1]:.3f} (starts at ln 2 = 0.693: w=0 predicts p=0.5)")
print(f"  w={w_lr.round(3)}, b={b_lr:.3f} | train acc {((sigmoid(Xl[tr] @ w_lr + b_lr) >= 0.5) == yl[tr]).mean():.3f} | test acc {((p_te >= 0.5) == yl[te]).mean():.3f}")
print(f"  loss never went up between steps: {all(a >= b - 1e-12 for a, b in zip(losses, losses[1:]))}")
```
- **In simple words:** Start with all weights at zero (so every probability is 0.5), repeatedly nudge the weights against the gradient, and record the loss. The two clouds overlap, so even the best line makes some mistakes. A quarter of the points is held out to test.
- **Exactly:** `Xl` is `(400, 2)`: 200 points around (-1,-1) and 200 around (+1,+1), standard deviation 1.2. `rng_lr.permutation` shuffles indices; `tr` (300) trains, `te` (100) tests. The printed loss falls from 0.693 (before any step) to 0.324 within 200 steps and stays (converged; the loss is convex, so there is one minimum). The accuracy is about 0.87 on train and test. `all(a >= b - 1e-12 ...)` confirms the loss never increased: with a small enough learning rate, gradient descent on a convex loss is monotone.
- **Why it is here:** It is the training loop of module 03 with a different loss, and it is the same loop used for every neural network later (forward, loss, gradient, update).

### 9. The decision boundary  (`main.py` lines 175-180)
```python
# [9] The decision boundary: where p = 0.5, i.e. z = w.x + b = 0 (a straight line, like the perceptron's)
slope, icpt = -w_lr[0] / w_lr[1], -b_lr / w_lr[1]
print(f"\ndecision boundary: x2 = {slope:.2f} * x1 {icpt:+.2f}  (the set where w.x + b = 0, i.e. p = 0.5)")
for pt in ([-2.0, -2.0], [0.0, 0.0], [2.0, 2.0], [1.0, -1.0]):
    z = np.array(pt) @ w_lr + b_lr
    print(f"  x={pt}: z={z:+.2f}  p(class 1)={float(sigmoid(z)):.3f}  odds={float(np.exp(z)):.2f}")
```
- **In simple words:** The model says class 1 when the probability is above 0.5, which happens exactly when the weighted sum `z` is above 0. That is a straight line. Further from the line means more confident.
- **Exactly:** `w.x + b = 0` solved for `x2` gives `x2 = -(w1/w2) x1 - b/w2`: printed as slope -0.92 and intercept -0.13. For four test points the code prints `z`, `p = sigmoid(z)` and the odds `p/(1-p) = e^z`. The point (0,0) sits just inside class 1 (`z = +0.17`, `p = 0.541`); the point (1,-1) lies almost on the line (`z = +0.06`, `p = 0.515`); (-2,-2) and (2,2) are far from it (p = 0.007 and 0.995).
- **Why it is here:** It connects logistic regression to the perceptron: the same hyperplane. What differs is the output: a probability that decays smoothly with distance from the line, not a hard 0/1. `z` is the **log-odds** (logit): `ln(p/(1-p)) = w.x + b`, which is the textbook definition of the logit model.

### 10. Same data, perceptron rule  (`main.py` lines 183-190)
```python
# [10] Same data, perceptron rule: no probabilities, and it cannot settle on overlapping classes
w_p, b_p, hist_p = train_perceptron(Xl[tr], yl[tr], epochs=50)
print(f"\nperceptron on the overlapping data: errors per epoch (last 6) {hist_p[-6:]} -> never converged, "
      f"test acc {(neuron(Xl[te], w_p, b_p) == yl[te]).mean():.3f} (just where the cycle stopped)")
w_s, b_s, losses_s = train_logreg(Xb, yb, steps=200)                 # the separable blobs from block 4
w_s2, b_s2, losses_s2 = train_logreg(Xb, yb, steps=2000)
print(f"separable blobs: both separate them (logreg acc {((sigmoid(Xb @ w_s + b_s) >= 0.5) == yb).mean():.2f}), but unregularised logreg keeps "
      f"growing |w| ({np.linalg.norm(w_s):.1f} after 200 steps, {np.linalg.norm(w_s2):.1f} after 2000) as loss -> 0 ({losses_s2[-1]:.4f}); the perceptron just stops (|w|={np.linalg.norm(w):.1f})")
```
- **In simple words:** Give the perceptron the same overlapping points: it never settles, because it keeps being 'wrong' on a few points and keeps moving. Then look at separable data, where logistic regression has its own quirk.
- **Exactly:** `train_perceptron` returns the errors per epoch: `[62, 60, 55, 61, 58, 55]`, no clean epoch ever. Its final weights are wherever the 50th epoch stopped (the test accuracy 0.870 equals logistic regression's here by coincidence, not by design; re-run with other seeds and it varies). On the separable blobs from block 4 both methods separate the classes, but the unregularised logistic loss keeps decreasing as `|w|` grows: 2.7 after 200 steps, 4.0 after 2000, loss 0.0005. It would grow without limit (the probabilities are pushed to exactly 0 and 1). The perceptron simply stops once nothing is misclassified.
- **Why it is here:** It states the practical differences in numbers: logistic regression has a smooth loss (gradient descent works on non-separable data, outputs probabilities), the perceptron has a mistake-driven rule. It also motivates **regularisation** (L2) for logistic regression on separable data, covered in [step 07](../07_regularization/WALKTHROUGH.md).

### 11. Calibration: does '0.8' mean 80%?  (`main.py` lines 193-212)
```python
# [11] Calibration: does "0.8" mean 80%? Bin the predictions and compare with the observed fraction of class 1
def reliability(p, y, bins=5):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            rows.append((lo, hi, int(mask.sum()), float(p[mask].mean()), float(y[mask].mean())))
    ece = sum(n * abs(pm - ym) for _, _, n, pm, ym in rows) / len(y)  # expected calibration error
    return rows, ece


p_all = sigmoid(Xl @ w_lr + b_lr)
rows, ece = reliability(p_all, yl)
print("\ncalibration of the logistic regression (bin | n | mean predicted | observed fraction of 1):")
for lo, hi, n_, pm, ym in rows:
    print(f"  [{lo:.1f},{hi:.1f}) {n_:4d}   {pm:.2f}   {ym:.2f}")
over = sigmoid(3 * (Xl @ w_lr + b_lr))                                # same boundary, logits x3: same labels, bolder probabilities
print(f"ECE {ece:.3f} vs overconfident copy {reliability(over, yl)[1]:.3f} | BCE {bce(p_all, yl):.3f} vs {bce(over, yl):.3f} | "
      f"accuracy {((p_all >= .5) == yl).mean():.3f} vs {((over >= .5) == yl).mean():.3f} (accuracy cannot see the difference)")
```
- **In simple words:** Group predictions by confidence and, in each group, compare the average predicted probability with how often class 1 really happens. If the model says 0.95 and is right 97% of the time, it is calibrated. Then make a copy with the same decisions but bolder probabilities and see the table get worse while accuracy stays identical.
- **Exactly:** `reliability` splits `[0, 1]` into 5 equal bins, and for each non-empty bin returns the count, the mean `p` and the mean `y`. ECE (expected calibration error) is the count-weighted average gap. The model is well calibrated (ECE 0.035; bin 0.95 vs 0.97). Multiplying the logits by 3 keeps the boundary and the accuracy (0.865 both) but doubles ECE (0.075) and raises BCE from 0.313 to 0.496: BCE punishes confident mistakes. The data were generated from two equal-variance Gaussians, which is exactly the model logistic regression assumes, so it is calibrated by construction. Real, misspecified data (and neural nets, which tend to be overconfident) need checking and sometimes **temperature scaling** (divide logits by `T`, fit `T` on validation data).
- **Why it is here:** It is the interview point that separates accuracy from probability quality: BCE, ECE and the Brier score see calibration; accuracy does not. It is why we train with BCE and not with the perceptron's 0/1 error.

### 12. Check against scikit-learn  (`main.py` lines 215-223)
```python
# [12] Check against scikit-learn (optional dependency)
try:
    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(C=1e6, max_iter=1000).fit(Xl[tr], yl[tr])    # huge C = almost no L2 penalty -> maximum likelihood, like ours
    print(f"\nsklearn: w={sk.coef_[0].round(3)}, b={sk.intercept_[0]:.3f} | ours: w={w_lr.round(3)}, b={b_lr:.3f} | "
          f"max abs difference {max(np.abs(sk.coef_[0] - w_lr).max(), abs(sk.intercept_[0] - b_lr)):.4f}")
    print(f"  predict_proba agreement on the test set: max |p_sklearn - p_ours| = {np.abs(sk.predict_proba(Xl[te])[:, 1] - p_te).max():.4f}")
except ImportError:
    print("\n(scikit-learn not installed: skipping the cross-check; everything above is pure numpy)")
```
- **In simple words:** Fit the same data with the standard library and compare numbers. They agree to three decimals.
- **Exactly:** `LogisticRegression(C=1e6)` solves the same problem with an L-BFGS optimiser (a second-order-ish method that uses the recent gradient history; internally it iterates until the gradient is tiny). `C` is the inverse of the L2 penalty strength (`C=1e6`: almost no penalty), so the solution is the maximum-likelihood one that gradient descent approaches. Coefficients `[1.209 1.318], b=0.166` vs ours `[1.209 1.317], b=0.166`; the predicted probabilities differ by at most 0.0003. The `try/except ImportError` keeps the file runnable without scikit-learn.
- **Why it is here:** An independent implementation agreeing to three decimals is the best evidence that the formula and the code are right. Note sklearn's default is `C=1.0`, i.e. regularised: its default coefficients would be smaller than the unregularised ones here.

### 13. TRY IT / BREAK IT comments  (`main.py` lines 225-230)
```python
# TRY IT: run with LR=0.01 and LR=5 (env var LR=...): the loss after 100 steps is 0.467 vs 0.324 -- small crawls, and here even 5 is stable (the loss is convex and smooth).
# TRY IT: change the cloud centres from +-1 to +-0.3 (more overlap): what happens to the test accuracy and to the calibration table?
# TRY IT: change the AND labels to [0, 0, 0, 1] -> [0, 1, 1, 1] (OR) and watch the epoch count change.
# BREAK IT: in logreg_grad return -X.T @ r / len(y), -r.mean(): the sign flips, gradient *ascent*; the loss climbs (18.9 after 30 steps) instead of falling. The gradient check does NOT catch this (it checks the formula's size, not that you subtract it).
# BREAK IT: train_logreg with labels in {-1, +1}: bce() still returns a number but it goes negative (-9.2 after 500 steps) and the weights explode (~120): not a log-likelihood any more.
# BREAK IT: pass y with values {-1, +1} instead of {0, 1} to train_perceptron; yi - pred is then wrong and training diverges.
```
- **In simple words:** Experiments to run yourself; the numbers in the first lines come from real runs (see Experiments and Debugging practice below).
- **Exactly:** Comments only; nothing executes.
- **Why it is here:** Learning by changing one thing is the point of the file.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `x @ w` | matrix/dot product (Python 3.5+ operator) | `np.array([1,2]) @ np.array([3,4])` -> `11` |
| `(z >= 0).astype(int)` | elementwise comparison to booleans, then to 0/1 | `np.array([-1., 0.]) >= 0` -> `[False, True]` |
| `w += update * xi` | in-place add; changes the array `w` itself | `w` stays the same object |
| `for xi, yi in zip(X, y)` | iterate over rows of `X` together with labels | `xi.shape == (2,)` |
| `np.vstack([a, b])` | stack arrays on top of each other (rows) | two `(100,2)` -> `(200,2)` |
| `np.array([0] * n + [1] * n)` | list repetition and concatenation | `[0]*3 + [1]*2` -> `[0,0,0,1,1]` |
| `f"{name:<4}"` | format spec: left-align in width 4 | `'OR  '` |
| `f"{x:+.2f}"` | format with an explicit sign, 2 decimals | `+0.17`, `-0.13` |
| `np.where(cond, a, b)` | elementwise choice between two arrays | sigmoid picks the safe formula per element |
| `X.T @ r / len(y)` | transpose then matrix-vector product: sums `r_i * x_i` over samples | `(2, n) @ (n,)` -> `(2,)` |
| `for e in np.eye(2)` | iterate over rows of the identity: unit vectors `[1,0]`, `[0,1]` | nudge one parameter at a time |
| `all(a >= b for a, b in zip(l, l[1:]))` | check a list is non-increasing by comparing neighbours | `zip(l, l[1:])` pairs item i with i+1 |
| `os.environ.get("STEPS", 2000)` | read a shell variable with a default (always a string: wrap in `int`) | `STEPS=500 python main.py` |
| `try: ... except ImportError:` | optional dependency guard | file still runs without scikit-learn |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `np.random.default_rng(seed)` | numpy | reproducible random generator | `seed` | `Generator` |
| `rng.normal(loc, scale, size)` | numpy | Gaussian samples | `loc` mean, `scale` std, `size` shape | `ndarray` |
| `np.vstack` | numpy | stack along rows | tuple/list of arrays | `ndarray` |
| `np.zeros(n)` | numpy | array of zeros (weights start here) | shape | `ndarray` |
| `np.exp`, `np.log` | numpy | elementwise `e^x` and natural log | array | `ndarray` |
| `np.clip(a, lo, hi)` | numpy | limit values to `[lo, hi]` | `a`, `lo`, `hi` | `ndarray` |
| `np.where(c, a, b)` | numpy | elementwise if/else | boolean array, two arrays | `ndarray` |
| `np.linalg.norm(w)` | numpy | length (L2 norm) of a vector | vector | float |
| `rng.permutation(n)` | numpy | random ordering of `0..n-1` | `n` | `ndarray` of ints |
| `LogisticRegression(C=1e6).fit(X, y)` | scikit-learn | fits logistic regression with L2 penalty `1/C`; internally L-BFGS (a quasi-Newton optimiser) minimises BCE + penalty | `C` inverse regularisation, `max_iter` | fitted model: `.coef_` `(1, d)`, `.intercept_` `(1,)` |
| `model.predict_proba(X)` | scikit-learn | probabilities per class | `X` | `(n, 2)`; column 1 is `P(y=1)` |

## Execution trace (real run)
```text
z = x.w + b = 0.1 | step(z) = 1
AND : converged at epoch 6, w=[2. 1.], b=-3.0, accuracy 4/4
OR  : converged at epoch 4, w=[1. 1.], b=-1.0, accuracy 4/4
XOR : never converged in 100 epochs, errors per epoch (last 6): [4, 4, 4, 4, 4, 4], final accuracy 2/4
blobs: converged after 2 epochs, accuracy 1.00, w=[1.91 2.09], b=-1.0
best XOR accuracy over 20,000 random lines: 3 / 4
hidden layer activations:
 [[0 0]
 [1 0]
 [1 0]
 [1 1]]
hand-built 2-layer net on XOR: [0, 1, 1, 0] target: [0, 1, 1, 0]

[logreg] sigmoid(0)=0.50  sigmoid(+-1000)=0.0/1.0 (no overflow)  BCE of a coin flip p=0.5: 0.693 (= ln 2)
gradient check: max |analytic - numeric| = 4.3e-11

logreg GD: loss before training / after 200 / after 2000 steps: 0.693 / 0.324 / 0.324 (starts at ln 2 = 0.693: w=0 predicts p=0.5)
  w=[1.209 1.317], b=0.166 | train acc 0.863 | test acc 0.870
  loss never went up between steps: True

decision boundary: x2 = -0.92 * x1 -0.13  (the set where w.x + b = 0, i.e. p = 0.5)
  x=[-2.0, -2.0]: z=-4.89  p(class 1)=0.007  odds=0.01
  x=[0.0, 0.0]: z=+0.17  p(class 1)=0.541  odds=1.18
  x=[2.0, 2.0]: z=+5.22  p(class 1)=0.995  odds=184.85
  x=[1.0, -1.0]: z=+0.06  p(class 1)=0.515  odds=1.06

perceptron on the overlapping data: errors per epoch (last 6) [62, 60, 55, 61, 58, 55] -> never converged, test acc 0.870 (just where the cycle stopped)
separable blobs: both separate them (logreg acc 1.00), but unregularised logreg keeps growing |w| (2.7 after 200 steps, 4.0 after 2000) as loss -> 0 (0.0005); the perceptron just stops (|w|=2.8)

calibration of the logistic regression (bin | n | mean predicted | observed fraction of 1):
  [0.0,0.2)  142   0.06   0.08
  [0.2,0.4)   42   0.31   0.21
  [0.4,0.6)   40   0.50   0.42
  [0.6,0.8)   35   0.70   0.74
  [0.8,1.0)  141   0.95   0.97
ECE 0.035 vs overconfident copy 0.075 | BCE 0.313 vs 0.496 | accuracy 0.865 vs 0.865 (accuracy cannot see the difference)

sklearn: w=[1.209 1.318], b=0.166 | ours: w=[1.209 1.317], b=0.166 | max abs difference 0.0005
  predict_proba agreement on the test set: max |p_sklearn - p_ours| = 0.0003
```
- `z = x.w + b = 0.1 | step(z) = 1`: the arithmetic from block 1 by hand.
- **AND** and **OR** converge (epoch 6 and 4) and the printed weights check out: for AND, `2*1+1*1-3 = 0 >= 0` fires only at `[1,1]`.
- **XOR** keeps `[4, 4, 4, 4, 4, 4]` errors forever: the algorithm cycles. Compare with AND: the only difference is the label vector.
- **blobs** converged in 2 epochs, accuracy 1.00.
- **3 / 4** is the best out of 20,000 random lines. Three of four is the ceiling for any line.
- The hidden activations and the final `[0, 1, 1, 0]` match the XOR target exactly.
- **Gradient check 4.3e-11**: the formula `(p - y) x` equals the numerical slope to 11 digits. Run it before trusting any hand-derived gradient.
- **Loss 0.693 -> 0.324**: it starts at `ln 2` because zero weights predict 0.5 for everyone, then falls and stops; test accuracy 0.870 on 100 unseen points. The two clouds overlap, so about 13% error is the best a line can do here.
- **Perceptron on the same data**: errors per epoch `[62, 60, 55, 61, 58, 55]`: it never settles. Logistic regression converged because its loss is smooth and convex.
- **Separable blobs**: `|w|` grows from 2.7 to 4.0 and the loss to 0.0005 while the perceptron stops at `|w| = 2.8`: unregularised logistic regression wants infinite confidence on separable data.
- **Calibration**: mean predicted vs observed agree bin by bin (0.95 vs 0.97 in the top bin; 0.31 vs 0.21 in a bin of only 42 points: small bins are noisy). The 3x-logit copy has the same accuracy (0.865) but ECE 0.075 and BCE 0.496 vs 0.313.
- **sklearn** agrees: coefficients within 0.0005, probabilities within 0.0003. Differences come from the stopping point of gradient descent and sklearn's tolerance.

## Experiments (change ONE thing)
1. Learning rate on AND (zero init): `lr=1.0` converges in 6 epochs with `w=[2,1], b=-3`; `lr=0.1` in 4 epochs with `w=[0.2,0.1], b=-0.2`; `lr=0.01` in 6 epochs with `w=[0.02,0.01], b=-0.03`. With a zero start the learning rate only scales the weights; the boundary is the same. (This is special to the perceptron rule with zero init; for gradient descent the learning rate matters a lot, see [module 03 step 02](../../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md).)
2. Reverse the row order of AND (`X[::-1], y[::-1]`): 8 epochs and `w=[1,2], b=-3` instead of 6 epochs and `w=[2,1]`. The perceptron finds a different valid line depending on data order: it is not the 'best' line.
3. Add the feature cross `x1*x2` as a third input to XOR: converges in 8 epochs with `w=[1,1,-3], b=-1`. XOR is linearly separable in the new 3-D space. A hidden layer learns such features on its own.
4. `LR=0.01 python main.py` (logistic regression): after 200 steps the loss is still 0.402 (converged 0.324 at lr 0.5), after 2000 steps 0.325 with `w=[1.107 1.19]`: a small learning rate just needs more steps. `LR=5` gives the same final numbers as 0.5 (the loss is smooth and convex; a rate this large would not be safe for a deep net). `STEPS=50` already reaches loss 0.324.
5. More overlap: change the cloud centres from +-1 to +-0.3 (edit the two `rng_lr.normal` lines). The loss barely leaves `ln 2` (0.656), test accuracy falls to 0.67, the weights shrink to about `[0.30 0.32]`, and the perceptron makes 141-150 errors per epoch. The model honestly reports low confidence (probabilities between 0.2 and 0.8), which is the correct behaviour: the classes really are hard to tell apart.
6. Compare `|w|` after `STEPS=200` and `STEPS=2000` on the separable blobs (block 10 prints both): 2.7 vs 4.0.

## Debugging practice
BREAK IT 1 (silent failure): use labels `{-1, +1}` instead of `{0, 1}` for AND: `train_perceptron(X, np.array([-1,-1,-1,1]), epochs=20)`. No error is raised; the history is `[4, 4, 4, ...]` for all 20 epochs. The `step` function predicts only 0 or 1, so `yi - pred` is `-1 - 0 = -1` for a correct class-0 answer. The rule 'corrects' correct answers. Lesson: a label convention is part of the algorithm, and bugs like this give no traceback, only a metric that never improves.

BREAK IT 2 (loud failure): give a weight vector of the wrong length, `neuron(np.ones((4, 2)), np.ones(3), 0.0)`:
```
  File ".../main.py", line 37, in neuron
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 2)
```
Read it: `matmul` is the `@`; operand 1 is `w`; its dimension 0 has size 3 but `x` has 2 columns. The rule: the last dimension of the left operand must equal the first of the right one. Print `.shape` of both before `@`.

BREAK IT 3 (silent, logistic regression): in `logreg_grad` return `-X.T @ r / len(y), -r.mean()` (sign flipped). No error: the update `w -= lr * g` now climbs the loss. Run 30 steps from zero: the loss is 18.9 instead of falling toward 0.32. The numerical gradient check from block 7 compares against the function you wrote, so also compare the *direction*: after one step the loss must go down. Lesson: log the loss; a metric that rises is the symptom.

BREAK IT 4 (silent, logistic regression): train with labels in `{-1, +1}` instead of `{0, 1}`. `bce` still returns numbers, but after 500 steps the loss is -9.2 (a negative log-loss is impossible for valid probabilities and labels) and `w` is about `[120, 115]`. The formula `y ln p + (1-y) ln(1-p)` assumes `y` is 0 or 1; with `y = -1` the second term becomes `+2 ln(1-p)`. Lesson: BCE needs {0, 1}; the `{-1, +1}` convention belongs to the logistic loss written as `ln(1 + e^(-y z))` (and to SVMs).

## Common mistakes (logistic regression)
- Calling it 'regression' in an interview and forgetting it is a classifier whose *output* is a probability.
- Using accuracy to judge probabilities (it cannot see calibration); use BCE, Brier score, ECE.
- Mixing label conventions: BCE needs `{0, 1}`; `ln(1 + e^(-yz))` needs `{-1, +1}`.
- Applying `sigmoid` and then a loss that also applies it (`BCEWithLogitsLoss` expects raw logits).
- Taking `ln(sigmoid(z))` naively: `sigmoid` underflows to 0 for very negative `z`; use `log_sigmoid` / the fused loss.
- Forgetting the bias (the boundary is forced through the origin).
- Unregularised fit on separable data: weights drift to infinity; add L2 or stop early.
- Not standardising features before gradient descent when scales differ (slow, zig-zag convergence; see [module 03 step 02](../../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md)).
- Reading the 0.5 threshold as sacred: choose it from the cost of false positives vs false negatives (see [module 03 step 07](../../03_ml_foundations/07_metrics/WALKTHROUGH.md)).

## Rebuild checklist
- [ ] `neuron(x, w, b)`: `x @ w + b`, then activation
- [ ] `train_perceptron`: zero init, per-sample update `lr * (y - pred)`, stop on a clean epoch
- [ ] explain why XOR cannot be learned in one inequality chain
- [ ] hand-set weights for AND, OR and the 2-layer XOR network
- [ ] say why a stack of identity layers collapses to one layer
- [ ] `sigmoid` (overflow-safe), `bce` (with clipping), `logreg_grad` = `X.T @ (p - y) / n` and `mean(p - y)`
- [ ] derive `p - y` on paper: `dL/dp`, `dp/dz = p(1-p)`, product
- [ ] numerical gradient check with central differences
- [ ] `train_logreg`: zero init, full-batch update, record the loss
- [ ] boundary `x2 = -(w1 x1 + b) / w2`; explain the log-odds reading of `w`
- [ ] reliability table (bins, mean p, mean y) and why x3 logits keep accuracy but hurt BCE

## The new problem (cliffhanger)
A hand-built 2-layer network solves XOR, but we picked its weights by thinking. How does the machine find them? The step function is flat, so the perceptron rule cannot pass blame backwards through a hidden layer. We need a smooth activation and the chain rule. Next: [step 02, an MLP with backpropagation from scratch](../02_mlp_backprop_from_scratch/WALKTHROUGH.md).

> **Say it aloud (logistic regression):** Logistic regression is a single neuron with a sigmoid output trained with binary cross-entropy: `p = sigmoid(w.x + b)`, loss `-[y ln p + (1-y) ln(1-p)]`, which is the negative log-likelihood of a Bernoulli. Its gradient simplifies to `(p - y) x`, the same residual form as linear regression and softmax. The loss is convex, the boundary is a hyperplane where `w.x + b = 0`, the weights are log-odds, and unlike the perceptron it works on overlapping classes and outputs probabilities that you should check for calibration. On separable data unregularised weights grow without bound, so regularise.

> **Say it aloud (interview box):** A neuron is a dot product of inputs and weights plus a bias, passed through an activation; geometrically it splits space with one hyperplane. The perceptron rule updates weights only on mistakes and is guaranteed to converge when the data are linearly separable. XOR is not separable, so any single neuron gets at most 3 of 4 right. Adding a hidden layer with a nonlinearity lets the output neuron work on re-described features, which is why depth is needed, and why the nonlinearity matters: stacked linear layers collapse into one.
