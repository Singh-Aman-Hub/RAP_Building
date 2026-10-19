"""
GOAL: Train a model by gradient descent in plain numpy so that the words
      loss / learning rate / epoch / batch size mean something concrete.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/02_linear_regression_numpy/main.py
EXPECTED OUTPUT: loss falling each epoch, learned w and b near 2 and 1, then a learning-rate and batch-size comparison.
NEEDS: numpy
"""

import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=200)                         # shape: (200,) | 200 random inputs between -1 and 1
y = 2.0 * X + 1.0 + rng.normal(0, 0.1, size=200)         # shape: (200,) | true rule: y = 2x + 1, plus a little noise


def train(X, y, learning_rate=0.1, epochs=50, batch_size=None, verbose=False):
    """Fit y ~ w*x + b.  Returns (w, b, loss_history).

    learning_rate : size of each step downhill.  Too small = slow, too big = diverges.
    epochs        : how many full passes over the dataset.
    batch_size    : samples used per update. None = all (full-batch). Smaller = noisier but more updates.
    """
    w, b = 0.0, 0.0                                      # start knowing nothing
    n = len(X)
    batch_size = batch_size or n
    history = []
    for epoch in range(epochs):
        order = rng.permutation(n)                       # shuffle every epoch
        for start in range(0, n, batch_size):            # range with step -> walk through mini-batches
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]                        # shape: (batch_size,) each
            y_hat = w * xb + b                           # shape: (batch_size,) | 1. forward pass: predict
            error = y_hat - yb
            # 2. gradients: derivative of mean((y_hat - y)^2) with respect to w and b
            grad_w = 2 * np.mean(error * xb)
            grad_b = 2 * np.mean(error)
            # 3. update: move AGAINST the gradient (downhill)
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
        loss = np.mean((w * X + b - y) ** 2)             # full-data loss after the epoch
        history.append(loss)
        if verbose and (epoch < 5 or epoch % 10 == 0):
            print(f"epoch {epoch:3d}  loss={loss:.4f}  w={w:.3f}  b={b:.3f}")
    return w, b, history


print("=== One training run (full batch) ===")
w, b, history = train(X, y, learning_rate=0.1, epochs=50, verbose=True)
print(f"learned w={w:.3f}, b={b:.3f}   (true values: 2 and 1)")

print("\n=== Experiment A: learning rate (epochs=50, full batch) ===")
for lr in (0.001, 0.1, 0.9, 1.1):
    w_, b_, h = train(X, y, learning_rate=lr, epochs=50)
    print(f"lr={lr:<6} final loss={h[-1]:.4g}   w={w_:.3f}")
# lr=0.001 barely moves; lr=1.1 is so big each step overshoots and the loss EXPLODES (diverges).

print("\n=== Experiment B: batch size (lr=0.1, epochs=5) ===")
for bs in (200, 32, 1):
    updates = 5 * -(-200 // bs)                          # ceil division: updates per run
    w_, b_, h = train(X, y, learning_rate=0.1, epochs=5, batch_size=bs)
    print(f"batch_size={bs:<4} updates={updates:<5} loss after 5 epochs={h[-1]:.4f}")
# Smaller batches => more updates per epoch (faster progress per epoch) but noisier steps.

print("\n=== Experiment C: epochs ===")
for ep in (1, 5, 50):
    _, _, h = train(X, y, learning_rate=0.1, epochs=ep)
    print(f"epochs={ep:<3} final loss={h[-1]:.4f}")
# TRY IT: set the noise to 1.0 in the data line. What is the lowest loss you can reach? (can't beat the noise)
# BREAK IT: delete the `- learning_rate * grad` sign (use +=). Loss climbs: you walked UPHILL.
