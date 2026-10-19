# Step 05 — Overfitting and validation

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

## Say it aloud
"Overfitting is when the model memorises training noise: training loss keeps dropping but validation loss rises. I detect it by comparing the two. Fixes: more data, a simpler model, regularisation, dropout, early stopping. This is why fine-tuning an LLM on a tiny dataset for many epochs hurts."
