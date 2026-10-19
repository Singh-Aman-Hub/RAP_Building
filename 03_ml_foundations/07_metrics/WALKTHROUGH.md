# Step 07 — Metrics: accuracy, precision, recall, F1

## Goal
Compute metrics by hand, verify with scikit-learn, and know when accuracy misleads.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/07_metrics/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `zip(y_true, y_pred)` | iterate two lists in parallel | |
| `sum(1 for ... if ...)` | count matches | |
| `[0] * 99` | repeat | |
| `[1] + [0]*99` | list concatenation | |

## Libraries & functions
| Name | Origin | Does |
|---|---|---|
| `accuracy_score(y_true, y_pred)` | scikit-learn | fraction correct |
| `precision_score`, `recall_score`, `f1_score` | scikit-learn | per positive class; arg `zero_division` |
| `confusion_matrix` | scikit-learn | `[[TN FP] [FN TP]]` |

## Execution trace (real run)
```
TP=2 FP=1 FN=2 TN=5
by hand : acc=0.70 precision=0.67 recall=0.50 f1=0.57
sklearn : acc=0.70 precision=0.67 recall=0.50 f1=0.57
confusion matrix [[TN FP] [FN TP]]: [[5 1] [2 2]]
always-no model: accuracy = 0.99 recall = 0.0
```
precision = 2/(2+1); recall = 2/(2+2); F1 = harmonic mean = 0.571.

## Experiments
1. Flip one prediction; predict which metric moves.
2. Make the data balanced (50/50) and re-evaluate the always-no model.
3. Compute macro-F1 for both classes.

## Debugging practice
Precision with no predicted positives divides by zero; sklearn warns and returns 0 (set `zero_division=0`). Always know what your metric does at the edges.

## Rebuild checklist
- [ ] Draw the 2×2 confusion matrix.
- [ ] Precision vs recall in one sentence each.
- [ ] Say when you'd prefer recall.

## Say it aloud
"Accuracy is the fraction correct, but on imbalanced data a model that always says 'no' looks great. Precision asks 'of what I flagged, how much was right', recall asks 'of the real positives, how many did I find', and F1 balances them. Which matters depends on the cost of each kind of mistake."
