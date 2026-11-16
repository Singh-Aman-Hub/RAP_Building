# Step 05 — Overfitting and validation

## The story so far
Step 04's loop drove the loss down on a simple linear rule, and validation loss followed. That is the happy case. Chapter 4 shows the unhappy one.

## The pain point
A flexible model can reach near-zero training error by memorising the noise in the examples. The training loss looks perfect, yet on new data the predictions are wild. If you only watch training loss, you ship a model that fails in production.

## The spark
Hold out data the model never trains on and compare the two errors. Training error falling while validation error rises is the signature of memorising. Fixes follow from the cause: more data, a simpler model, regularisation, dropout, early stopping.

## Goal
Read train vs validation error to detect underfitting, good fit and overfitting.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/05_overfitting_train_val/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Overfitting | high variance, memorising, over-parameterised fit | Close | Train error low, validation error high |
| Underfitting | high bias | Close | Both errors high; the model is too simple |
| Validation set | dev set, held-out set | Mostly | Data the model never trained on, used to tune and monitor |
| Test set | hold-out test, final exam | No | Touched once at the end; validation can be reused to pick settings |
| Data leakage | train/test contamination | Yes | Information from the evaluation data influenced training |
| Regularisation | weight decay, L2, dropout, early stopping | Umbrella term | Any technique that limits model flexibility to reduce overfitting |
| Generalisation gap | train-val gap | Yes | `val loss - train loss` |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–9)

```python
"""
GOAL: See overfitting: a flexible model memorises noisy training data (train loss tiny)
      but does badly on unseen validation data.  Learn to spot it from the two loss curves.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/05_overfitting_train_val/main.py
EXPECTED OUTPUT: table of polynomial degree vs train/validation error; train keeps falling, validation turns up.
NEEDS: numpy
"""
```

- **In simple words:** Says what overfitting looks like.
- **Exactly:** Docstring only. Promise: a degree-vs-error table where train falls and validation rises.
- **Why it is here:** Sets expectations.

### 2. Make 30 noisy points and split 15/15  (`main.py` lines 11–17)

```python
import numpy as np

rng = np.random.default_rng(1)
x = rng.uniform(-1, 1, 30)                               # shape: (30,)
y = np.sin(3 * x) + rng.normal(0, 0.2, 30)               # true curve + noise
x_train, y_train = x[:15], y[:15]                        # shape: (15,) each | model sees these
x_val, y_val = x[15:], y[15:]                            # shape: (15,) each | model never trains on these
```

- **In simple words:** Sample points from a sine wave with noise, train on half, keep the other half hidden.
- **Exactly:** `rng = default_rng(1)`; `x` is `(30,)` in [-1, 1); `y = sin(3x) + N(0, 0.2)` is `(30,)`, so the noise floor MSE is about 0.04. `x_train, y_train` are the first 15 and `x_val, y_val` the other 15 (all `(15,)`). The data is random, so first/second half is a fair split.
- **Why it is here:** With only 15 training points a flexible model can memorise them. The hidden half exposes it.

### 3. Fit polynomials of rising degree and compare errors  (`main.py` lines 19–24)

```python
print(f"{'degree':>6} {'train MSE':>10} {'val MSE':>10}")
for degree in (1, 3, 5, 9, 12):                          # higher degree = more flexible curve
    coefs = np.polyfit(x_train, y_train, degree)         # least-squares polynomial fit (a model with degree+1 parameters)
    train_mse = np.mean((np.polyval(coefs, x_train) - y_train) ** 2)
    val_mse = np.mean((np.polyval(coefs, x_val) - y_val) ** 2)
    print(f"{degree:>6} {train_mse:>10.4f} {val_mse:>10.4f}")
```

- **In simple words:** Fit curves from very simple to very wiggly and print how well each does on seen and unseen points.
- **Exactly:** `np.polyfit(x_train, y_train, degree)` returns `degree + 1` coefficients minimising squared error. `np.polyval` evaluates them. Real table: degree 1 `0.1661 / 0.2041`; degree 3 `0.0308 / 0.0342`; degree 5 `0.0282 / 0.0335`; degree 9 `0.0229 / 5.9452`; degree 12 `0.0069 / 430.7111`. Train error only ever falls; validation error explodes once parameters (13) approach the 15 data points. The `:>6`, `:>10.4f` format specs right-align and print 4 decimals.
- **Why it is here:** This table is the classic bias-variance picture: choose the degree where validation error is lowest (3 to 5 here), not train error.

### 4. Remedies and experiments (comments)  (`main.py` lines 26–33)

```python
# Reading the table:
#  degree 1  -> underfitting: both errors high (model too simple)
#  degree 3  -> good fit: both low and close
#  degree 12 -> overfitting: train error ~0, validation error large (memorised noise)
# Remedies: more data, simpler model, regularisation, early stopping (stop when validation loss starts rising),
# dropout, data augmentation.
# TRY IT: split at 25 instead of 15 (x[:25] train, x[25:] val) and re-run: overfitting shrinks. More data helps.
# BREAK IT: evaluate "val" on x_train by mistake -> the gap disappears and you fool yourself. That's data leakage.
```

- **In simple words:** A reminder of how to fix overfitting and two suggested experiments.
- **Exactly:** Remedies: more data, simpler model, regularisation, early stopping, dropout, augmentation. TRY IT: with 25 train points degree 12 hurts less. BREAK IT: scoring "val" on `x_train` removes the gap and gives a false sense of success (data leakage).
- **Why it is here:** In interviews, naming remedies and the leakage pitfall is the expected answer.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `x[:15]` / `x[15:]` | first 15 / the rest | train/val slicing |
| `f"{degree:>6}"` | right-align in width 6 | table columns |
| `for degree in (1, 3, ...)` | loop over a tuple | |

## Libraries & functions
| Name | Origin | Does | Returns |
|---|---|---|---|
| `np.polyfit(x, y, deg)` | numpy | least-squares polynomial | coefficients |
| `np.polyval(coefs, x)` | numpy | evaluate polynomial | array |
| `rng.uniform`, `rng.normal` | numpy | synthetic data | arrays |

## Execution trace (real run)
```
degree  train MSE    val MSE
     1     0.1661     0.2041
     3     0.0308     0.0342
     5     0.0282     0.0335
     9     0.0229     5.9452
    12     0.0069   430.7111
```
Underfit (deg 1) → good (3–5) → overfit (9+): training error keeps falling while validation explodes.

## Experiments
1. Use 25 training points: the degree-9 validation error shrinks.
2. Fit degree 3 repeatedly with different seeds: small variation.
3. Plot with matplotlib if you like (`pip install matplotlib`).

## Debugging practice
Evaluating "validation" on `x_train` gives a tiny gap and false confidence: **data leakage**. Always check which array goes into the metric.

## Rebuild checklist
- [ ] Define underfit/overfit with both curves.
- [ ] List 4 remedies.
- [ ] Explain early stopping.

## The new problem (cliffhanger)
Now we can tell memorising from learning, but so far every task output a single number. Real tasks need a decision: "which class, and how sure are you?" Raw scores are not probabilities. Next: [06_softmax_cross_entropy](../06_softmax_cross_entropy/).

## Say it aloud
> **Say it aloud (interview box):** Overfitting is when a model memorises training noise, so training loss keeps falling while validation loss rises. I detect it by comparing the two, and I fix it with more data, a simpler model, regularisation, dropout or early stopping. It is why fine-tuning an LLM on a tiny dataset for many epochs makes it worse, not better.
