# Step 07 — Metrics: accuracy, precision, recall, F1

## The story so far
Step 06 gave us a probability and a loss. A loss is good for training, but humans want to know "how often is it right, and what does it miss?"

## The pain point
Accuracy can lie. With 1 fraud in 100 transactions, a model that always says "not fraud" is 99% accurate and catches nothing. One headline number hides which kind of mistake you are making.

## The spark
Count four outcomes (true/false positives and negatives) and build metrics from them. Precision asks how much of what I flagged was right. Recall asks how much of the real positives I found. F1 balances the two.

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
Precision computed by hand with no predicted positives gives `ZeroDivisionError: division by zero` (`tp / (tp + fp)` with both 0); sklearn warns and returns 0 (set `zero_division=0`). Always know what your metric does at the edges.

## Rebuild checklist
- [ ] Draw the 2×2 confusion matrix.
- [ ] Precision vs recall in one sentence each.
- [ ] Say when you'd prefer recall.

## The new problem (cliffhanger)
We can now classify numeric features and judge the result honestly. But the most valuable data is *text*, and text is not numeric. Can we put all of it together into one working pipeline? Next: [08_sklearn_text_classifier](../08_sklearn_text_classifier/).

## Say it aloud
> **Say it aloud (interview box):** Accuracy is the fraction correct, but on imbalanced data a model that always says no looks great while finding nothing. Precision asks how much of what I flagged was right, recall asks how many real positives I found, and F1 balances them. Which one matters depends on the cost of each kind of mistake.
