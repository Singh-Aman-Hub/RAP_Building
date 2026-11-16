# Step 02 — Gradient descent from scratch (numpy)

## The story so far
Chapter 1 ended with a guess typed by hand (`w=5, b=45`) and a loss that told us how wrong it was. Chapter 2 asks the obvious question: can the machine nudge those numbers itself?

## The pain point
Trying parameter values by hand does not scale. With two parameters you might get lucky; with a million you cannot even search. And with no rule for the step size, the first attempt either crawls or blows up.

## The spark
The slope of the loss tells you which way is uphill for each parameter. Step the opposite way, a little at a time, and repeat. That is gradient descent: forward, loss, gradient, update. The step size is the learning rate.

## Goal
See loss, learning rate, epochs and batch size act on a model you wrote yourself.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/02_linear_regression_numpy/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Learning rate | step size, alpha, eta, lr | Yes | The multiplier on the gradient in each update; the most-tuned hyperparameter |
| Epoch vs step vs iteration | pass, update | No | An epoch is one full pass over the data; a step/iteration is one parameter update. Steps per epoch = n / batch_size |
| Batch / mini-batch / SGD | full-batch GD, mini-batch GD, stochastic GD | Related | Full-batch uses all samples per update; mini-batch a slice; "SGD" strictly means batch size 1 but is used for any mini-batch variant |
| Gradient descent | steepest descent, backprop (partly) | No | Gradient descent is the update rule; backpropagation is the algorithm that computes the gradients |
| Loss function | cost function, objective | Mostly | The number being minimised |
| Parameters vs hyperparameters | weights vs settings | No | Parameters (w, b) are learned; hyperparameters (learning rate, epochs, batch size) are chosen by you |
| Diverge | blow up, explode | Yes | The loss grows each step instead of shrinking (seen at lr=1.1) |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–9)

```python
"""
GOAL: Train a model by gradient descent in plain numpy so that the words
      loss / learning rate / epoch / batch size mean something concrete.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/02_linear_regression_numpy/main.py
EXPECTED OUTPUT: loss falling each epoch, learned w and b near 2 and 1, then a learning-rate and batch-size comparison.
NEEDS: numpy
"""
```

- **In simple words:** The header says what the file demonstrates and what you should see.
- **Exactly:** Docstring only; no runtime effect. It promises: loss falls, w and b end near 2 and 1, then two comparisons.
- **Why it is here:** A reader knows the goal before reading 70 lines.

### 2. Make 200 noisy points on the line y = 2x + 1  (`main.py` lines 11–15)

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=200)                         # shape: (200,) | 200 random inputs between -1 and 1
y = 2.0 * X + 1.0 + rng.normal(0, 0.1, size=200)         # shape: (200,) | true rule: y = 2x + 1, plus a little noise
```

- **In simple words:** Invent data where we know the true answer, so we can check the model finds it.
- **Exactly:** `rng = np.random.default_rng(0)` is one global seeded generator (it is reused later inside `train`, so its state advances). `X` is `(200,)` uniform in [-1, 1). `y = 2.0*X + 1.0 + normal(0, 0.1, size=200)` is `(200,)`: the true line plus Gaussian noise with std 0.1. So the best possible MSE is about 0.1**2 = 0.01.
- **Why it is here:** Synthetic data with a known rule is a test: if training cannot recover w=2, b=1 the code is wrong. The noise floor (about 0.01) is why the final loss never reaches 0.

### 3. Define train() and initialise the parameters  (`main.py` lines 18–28)

```python
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
```

- **In simple words:** A function that learns a line. It starts with w=0 and b=0 ("knows nothing") and prepares to record the loss.
- **Exactly:** Signature `train(X, y, learning_rate=0.1, epochs=50, batch_size=None, verbose=False)` returns `(w, b, history)`. `w, b = 0.0, 0.0` are plain Python floats. `n = 200`. `batch_size or n` turns `None` into 200 (full batch) because `None` is falsy. `history` is a list that will hold one loss per epoch.
- **Why it is here:** Wrapping the loop in a function lets the experiments below re-run training with different settings.

### 4. Shuffle each epoch and cut out a mini-batch  (`main.py` lines 29–33)

```python
    for epoch in range(epochs):
        order = rng.permutation(n)                       # shuffle every epoch
        for start in range(0, n, batch_size):            # range with step -> walk through mini-batches
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]                        # shape: (batch_size,) each
```

- **In simple words:** Each epoch, deal the cards in a new order, then take them in handfuls of `batch_size`.
- **Exactly:** `rng.permutation(n)` returns shuffled indices `(200,)`. `range(0, n, batch_size)` yields start offsets 0, 32, 64, ... `order[start:start+batch_size]` is the batch's indices (the last batch may be shorter). `X[idx]` is fancy indexing and copies those rows, so `xb, yb` have shape `(batch_size,)`.
- **Why it is here:** Shuffling stops the model from seeing data in a fixed order that could bias the updates. Batches are the unit of one parameter update.

### 5. Forward pass and error  (`main.py` lines 34–35)

```python
            y_hat = w * xb + b                           # shape: (batch_size,) | 1. forward pass: predict
            error = y_hat - yb
```

- **In simple words:** Predict with the current line, then see how far off each prediction is.
- **Exactly:** `y_hat = w * xb + b` is `(batch,)`. `error = y_hat - yb` is also `(batch,)`; positive means the model guessed too high.
- **Why it is here:** The error vector is the raw material for both the loss and the gradients.

### 6. Compute the two gradients by hand  (`main.py` lines 36–38)

```python
            # 2. gradients: derivative of mean((y_hat - y)^2) with respect to w and b
            grad_w = 2 * np.mean(error * xb)
            grad_b = 2 * np.mean(error)
```

- **In simple words:** Work out which way to nudge w and b to reduce the error, using calculus done on paper.
- **Exactly:** Loss L = mean((w*x + b - y)**2). dL/dw = 2*mean(error * x) and dL/db = 2*mean(error). Both are scalars. These are exactly the formulas autograd reproduces in step 03 (`dL/dw = -35.0` matched the hand value there).
- **Why it is here:** Without gradients you would not know which direction reduces the loss. Writing them by hand makes "backprop" less magical.

### 7. Update: step against the gradient  (`main.py` lines 39–41)

```python
            # 3. update: move AGAINST the gradient (downhill)
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
```

- **In simple words:** Move each number a small step downhill.
- **Exactly:** `w -= learning_rate * grad_w` means w = w - lr * dL/dw. With lr=0.1 the first full-batch step moves w from 0 to about 0.163 (see epoch 0 line in the trace). The minus sign is the whole trick: the gradient points uphill.
- **Why it is here:** Flip the sign to `+=` and the loss climbs (the BREAK IT at the bottom). Too large an lr overshoots (lr=1.1 diverges to 7.5e8).

### 8. Record and print the full-data loss each epoch  (`main.py` lines 42–46)

```python
        loss = np.mean((w * X + b - y) ** 2)             # full-data loss after the epoch
        history.append(loss)
        if verbose and (epoch < 5 or epoch % 10 == 0):
            print(f"epoch {epoch:3d}  loss={loss:.4f}  w={w:.3f}  b={b:.3f}")
    return w, b, history
```

- **In simple words:** After a full pass, check how well the line fits all 200 points and optionally print progress.
- **Exactly:** `np.mean((w*X + b - y)**2)` uses ALL data (not only the last batch) so the history is comparable across batch sizes. It is appended to `history`. The `verbose` condition prints for epochs 0-4 and every 10th. Real output at epoch 0: `loss=2.0422 w=0.163 b=0.230`. `return w, b, history` hands back the learned values.
- **Why it is here:** Watching the loss curve is the first debugging tool in all ML. If it does not fall, something is wrong.

### 9. Run one full-batch training  (`main.py` lines 49–51)

```python
print("=== One training run (full batch) ===")
w, b, history = train(X, y, learning_rate=0.1, epochs=50, verbose=True)
print(f"learned w={w:.3f}, b={b:.3f}   (true values: 2 and 1)")
```

- **In simple words:** Train 50 epochs and compare to the truth.
- **Exactly:** `batch_size=None` means 1 update per epoch, 50 updates total. Real output: `learned w=1.943, b=0.997 (true values: 2 and 1)`. w is still creeping toward 2; more epochs would close the gap.
- **Why it is here:** This is the baseline the next experiments are compared with.

### 10. Experiment A: try four learning rates  (`main.py` lines 53–57)

```python
print("\n=== Experiment A: learning rate (epochs=50, full batch) ===")
for lr in (0.001, 0.1, 0.9, 1.1):
    w_, b_, h = train(X, y, learning_rate=lr, epochs=50)
    print(f"lr={lr:<6} final loss={h[-1]:.4g}   w={w_:.3f}")
# lr=0.001 barely moves; lr=1.1 is so big each step overshoots and the loss EXPLODES (diverges).
```

- **In simple words:** Same training, only the step size changes.
- **Exactly:** Real output: lr=0.001 gives final loss 2.402 and w=0.079 (barely moved); lr=0.1 gives 0.01113; lr=0.9 gives 0.01042 (fast); lr=1.1 gives 7.508e+08 and w=-3347.759 (diverged). Full-batch gradient descent on this MSE is stable only while lr < 2 / (largest curvature). The curvature of the loss in b is 2, so the limit is about lr = 1.0: 0.9 converges, 1.1 diverges.
- **Why it is here:** Shows the learning rate is a hyperparameter with a safe window: too small is slow, too large blows up.

### 11. Experiment B: try three batch sizes  (`main.py` lines 59–64)

```python
print("\n=== Experiment B: batch size (lr=0.1, epochs=5) ===")
for bs in (200, 32, 1):
    updates = 5 * -(-200 // bs)                          # ceil division: updates per run
    w_, b_, h = train(X, y, learning_rate=0.1, epochs=5, batch_size=bs)
    print(f"batch_size={bs:<4} updates={updates:<5} loss after 5 epochs={h[-1]:.4f}")
# Smaller batches => more updates per epoch (faster progress per epoch) but noisier steps.
```

- **In simple words:** Same learning rate and number of epochs; only how many samples feed each update changes.
- **Exactly:** `-(-200 // bs)` is ceiling division (negate, floor-divide, negate). Real output: batch 200 gives 5 updates and loss 0.7539; batch 32 gives 35 updates and 0.0168; batch 1 gives 1000 updates and 0.0126. More updates per epoch means faster progress but each update is noisier.
- **Why it is here:** This is the reason deep learning uses mini-batches: a good balance of progress per epoch and efficient use of vectorised hardware.

### 12. Experiment C: try three epoch counts, then TRY/BREAK comments  (`main.py` lines 66–71)

```python
print("\n=== Experiment C: epochs ===")
for ep in (1, 5, 50):
    _, _, h = train(X, y, learning_rate=0.1, epochs=ep)
    print(f"epochs={ep:<3} final loss={h[-1]:.4f}")
# TRY IT: set the noise to 1.0 in the data line. What is the lowest loss you can reach? (can't beat the noise)
# BREAK IT: delete the `- learning_rate * grad` sign (use +=). Loss climbs: you walked UPHILL.
```

- **In simple words:** Same settings, train for 1, 5 or 50 passes.
- **Exactly:** Real output: `epochs=1 final loss=2.0422`, `epochs=5 0.7539`, `epochs=50 0.0111`. The loss only drops while updates continue. The final comments suggest raising noise to 1.0 (loss floor becomes about 1.0) and flipping the update sign to see the loss climb.
- **Why it is here:** Training longer helps until the loss reaches the noise floor, after which extra epochs only risk overfitting on real data.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `def train(X, y, learning_rate=0.1, ...)` | default parameter values | `train(X, y)` |
| `return w, b, history` | return a tuple, unpack on the left | `w, b, h = train(...)` |
| `range(0, n, batch_size)` | counting with a step | `list(range(0,10,4))` → `[0,4,8]` |
| `X[idx]` | fancy indexing: pick rows by an index array | |
| `batch_size or n` | use `n` if `batch_size` is None | |
| `-(-a // b)` | ceiling division trick | `-(-200 // 32)` → `7` |
| `w -= lr * g` | update in place | |

## Libraries & functions
| Name | Origin | Does |
|---|---|---|
| `rng.uniform(lo, hi, size)` | numpy | random numbers in range |
| `rng.normal(mean, std, size)` | numpy | gaussian noise |
| `np.mean` | numpy | average |

## Execution trace (real run)
Each update: **forward** (`y_hat = w*x + b`) → **loss** → **gradients** (`2*mean(error*x)`, `2*mean(error)`) → **update**.
```
epoch   0  loss=2.0422  w=0.163  b=0.230
epoch   1  loss=1.5425  w=0.310  b=0.411
epoch  10  loss=0.2594  w=1.167  b=0.990
epoch  40  loss=0.0131  w=1.901  b=1.003
learned w=1.943, b=0.997   (true values: 2 and 1)
lr=0.001  final loss=2.402   w=0.079
lr=0.1    final loss=0.01113   w=1.943
lr=0.9    final loss=0.01042   w=1.987
lr=1.1    final loss=7.508e+08   w=-3347.759
batch_size=200  updates=5     loss after 5 epochs=0.7539
batch_size=32   updates=35    loss after 5 epochs=0.0168
batch_size=1    updates=1000  loss after 5 epochs=0.0126
```
Lessons: too-small lr = slow; too-large lr = divergence; smaller batches = more updates per epoch; noise sets a floor on loss (~0.01 here).

## Experiments
1. `lr=0.5`, `lr=1.0`; find where it starts to diverge (on this data around 1.0).
2. Epochs 1, 5, 50 (printed).
3. Increase noise `0.1 → 1.0`: the final loss floor rises to ~1.

## Debugging practice
Change `w -= ...` to `w += ...`: loss rises each epoch (uphill). Symptom "loss goes up" → check the update sign, learning rate and gradient formula, in that order.

## Rebuild checklist
- [ ] forward / loss / gradient / update loop.
- [ ] Define epoch, batch, step.
- [ ] What happens at lr too big/small.

## The new problem (cliffhanger)
This works, but we derived `2*mean(error*x)` on paper for one formula with two parameters. A deep network has millions of parameters across many layers, and nobody can derive that by hand. We need the computer to differentiate for us: [03_pytorch_tensors_autograd](../03_pytorch_tensors_autograd/).

## Say it aloud
> **Say it aloud (interview box):** Training repeats four things: predict, measure the loss, compute how each parameter changes the loss (the gradient), and nudge the parameters downhill by the learning rate. An epoch is one pass over the data and a batch is the chunk used per update. The trade-off is that big batches are stable but give few updates, while small batches are noisy but update more often and use less memory.
