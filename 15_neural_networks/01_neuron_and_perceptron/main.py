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


# [4] A separable blob dataset: the rule finds *a* separating line (not the best one)
n = 100
blob0 = rng.normal([-2, -2], 0.7, size=(n, 2))      # class 0 around (-2,-2)
blob1 = rng.normal([+2, +2], 0.7, size=(n, 2))      # class 1 around (+2,+2)
Xb = np.vstack([blob0, blob1])                      # (200, 2)
yb = np.array([0] * n + [1] * n)                    # (200,)
w, b, hist = train_perceptron(Xb, yb, epochs=50)
print(f"blobs: converged after {len(hist)} epochs, accuracy {(neuron(Xb, w, b) == yb).mean():.2f}, w={w.round(2)}, b={b}")


# [5] Why XOR fails: no line separates it. Brute-force a million lines to prove it
best = 0
for _ in range(20000):
    w_try = rng.normal(size=2) * 3
    b_try = rng.normal() * 3
    best = max(best, int((neuron(X, w_try, b_try) == gates["XOR"]).sum()))
print("best XOR accuracy over 20,000 random lines:", best, "/ 4")


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

# =====================================================================================================
# PART 2 -- LOGISTIC REGRESSION FROM SCRATCH (the smooth neuron: sigmoid + binary cross-entropy)
# Prerequisites: 03_ml_foundations/02 (gradient descent on linear regression), 03_ml_foundations/06 (softmax + CE)
# =====================================================================================================
STEPS = int(os.environ.get("STEPS", 2000))       # gradient-descent steps for the logistic regression
LR = float(os.environ.get("LR", 0.5))            # learning rate


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


# [9] The decision boundary: where p = 0.5, i.e. z = w.x + b = 0 (a straight line, like the perceptron's)
slope, icpt = -w_lr[0] / w_lr[1], -b_lr / w_lr[1]
print(f"\ndecision boundary: x2 = {slope:.2f} * x1 {icpt:+.2f}  (the set where w.x + b = 0, i.e. p = 0.5)")
for pt in ([-2.0, -2.0], [0.0, 0.0], [2.0, 2.0], [1.0, -1.0]):
    z = np.array(pt) @ w_lr + b_lr
    print(f"  x={pt}: z={z:+.2f}  p(class 1)={float(sigmoid(z)):.3f}  odds={float(np.exp(z)):.2f}")


# [10] Same data, perceptron rule: no probabilities, and it cannot settle on overlapping classes
w_p, b_p, hist_p = train_perceptron(Xl[tr], yl[tr], epochs=50)
print(f"\nperceptron on the overlapping data: errors per epoch (last 6) {hist_p[-6:]} -> never converged, "
      f"test acc {(neuron(Xl[te], w_p, b_p) == yl[te]).mean():.3f} (just where the cycle stopped)")
w_s, b_s, losses_s = train_logreg(Xb, yb, steps=200)                 # the separable blobs from block 4
w_s2, b_s2, losses_s2 = train_logreg(Xb, yb, steps=2000)
print(f"separable blobs: both separate them (logreg acc {((sigmoid(Xb @ w_s + b_s) >= 0.5) == yb).mean():.2f}), but unregularised logreg keeps "
      f"growing |w| ({np.linalg.norm(w_s):.1f} after 200 steps, {np.linalg.norm(w_s2):.1f} after 2000) as loss -> 0 ({losses_s2[-1]:.4f}); the perceptron just stops (|w|={np.linalg.norm(w):.1f})")


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


# [12] Check against scikit-learn (optional dependency)
try:
    from sklearn.linear_model import LogisticRegression
    sk = LogisticRegression(C=1e6, max_iter=1000).fit(Xl[tr], yl[tr])    # huge C = almost no L2 penalty -> maximum likelihood, like ours
    print(f"\nsklearn: w={sk.coef_[0].round(3)}, b={sk.intercept_[0]:.3f} | ours: w={w_lr.round(3)}, b={b_lr:.3f} | "
          f"max abs difference {max(np.abs(sk.coef_[0] - w_lr).max(), abs(sk.intercept_[0] - b_lr)):.4f}")
    print(f"  predict_proba agreement on the test set: max |p_sklearn - p_ours| = {np.abs(sk.predict_proba(Xl[te])[:, 1] - p_te).max():.4f}")
except ImportError:
    print("\n(scikit-learn not installed: skipping the cross-check; everything above is pure numpy)")

# TRY IT: run with LR=0.01 and LR=5 (env var LR=...): the loss after 100 steps is 0.467 vs 0.324 -- small crawls, and here even 5 is stable (the loss is convex and smooth).
# TRY IT: change the cloud centres from +-1 to +-0.3 (more overlap): what happens to the test accuracy and to the calibration table?
# TRY IT: change the AND labels to [0, 0, 0, 1] -> [0, 1, 1, 1] (OR) and watch the epoch count change.
# BREAK IT: in logreg_grad return -X.T @ r / len(y), -r.mean(): the sign flips, gradient *ascent*; the loss climbs (18.9 after 30 steps) instead of falling. The gradient check does NOT catch this (it checks the formula's size, not that you subtract it).
# BREAK IT: train_logreg with labels in {-1, +1}: bce() still returns a number but it goes negative (-9.2 after 500 steps) and the weights explode (~120): not a log-likelihood any more.
# BREAK IT: pass y with values {-1, +1} instead of {0, 1} to train_perceptron; yi - pred is then wrong and training diverges.
