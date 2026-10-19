# Step 02 — Gradient descent from scratch (numpy)

## Goal
See loss, learning rate, epochs and batch size act on a model you wrote yourself.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/02_linear_regression_numpy/main.py
```

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

## Say it aloud
"Training repeats four things: predict, measure the loss, compute how each parameter changes the loss (gradient), and nudge the parameters downhill by the learning rate. An epoch is one pass over the data; a batch is the chunk used per update. Trade-off: big batches are stable but few updates; small batches are noisy but faster per epoch and use less memory."
