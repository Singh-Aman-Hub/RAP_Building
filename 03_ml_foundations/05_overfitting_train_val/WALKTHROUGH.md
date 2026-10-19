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
