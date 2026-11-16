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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Recall | sensitivity, true positive rate (TPR), hit rate | Yes | `TP / (TP + FN)`: of all real positives, how many were caught |
| Precision | positive predictive value (PPV) | Yes | `TP / (TP + FP)`: of everything flagged, how many were right |
| Specificity | true negative rate (TNR) | Not the same as precision | `TN / (TN + FP)`: of all real negatives, how many were cleared |
| F1 | F-score, F-measure, Dice coefficient (for sets) | Yes (F-beta generalises) | Harmonic mean of precision and recall |
| Accuracy | overall correctness | No | Fraction correct; misleading on imbalanced data |
| False positive vs false negative | Type I vs Type II error | Yes | FP is a false alarm (Type I); FN is a miss (Type II) |
| Confusion matrix | error matrix, contingency table | Roughly | Grid of TP, FP, FN, TN; sklearn layout is `[[TN FP] [FN TP]]` |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–9)

```python
"""
GOAL: Compute accuracy, precision, recall and F1 by hand, then check them with scikit-learn.
      Learn why accuracy alone can lie on imbalanced data.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/07_metrics/main.py
EXPECTED OUTPUT: confusion counts, the four metrics, a "99% accurate but useless" demo.
NEEDS: scikit-learn
"""
```

- **In simple words:** Says we compute four metrics by hand and with sklearn.
- **Exactly:** Docstring only.
- **Why it is here:** Tells you the goal: understand what accuracy hides.

### 2. Import scikit-learn metric functions  (`main.py` line 11)

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
```

- **In simple words:** Bring in ready-made metric calculators for checking our own.
- **Exactly:** `sklearn.metrics` functions take `(y_true, y_pred)` and return a float (or a matrix for `confusion_matrix`).
- **Why it is here:** Trust but verify: matching sklearn proves the by-hand formulas are right.

### 3. Ten labelled emails and a model's guesses  (`main.py` lines 13–15)

```python
# 1 = spam (the "positive" class we care about), 0 = not spam
y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
```

- **In simple words:** Four real spam and six real non-spam; the model got some wrong.
- **Exactly:** Plain Python lists of ints. 1 = spam (the positive class). `y_true` has 4 ones and 6 zeros; `y_pred` flags one non-spam as spam (index 4) and misses two spam (indices 2 and 3).
- **Why it is here:** Deciding which class is "positive" is essential since precision and recall are about that class.

### 4. Count the four outcome types  (`main.py` lines 17–22)

```python
# Count the four outcomes
tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)   # true positive: said spam, was spam
fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)   # false positive: said spam, was NOT spam
fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)   # false negative: missed a spam
tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)   # true negative
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
```

- **In simple words:** Walk both lists in parallel and tally what happened for each email.
- **Exactly:** `zip(y_true, y_pred)` pairs elements; `sum(1 for ... if ...)` counts matches. Real output `TP=2 FP=1 FN=2 TN=5`; they add up to 10.
- **Why it is here:** All four metrics are different ratios of these four counts.

### 5. Compute the four metrics by hand  (`main.py` lines 24–28)

```python
accuracy = (tp + tn) / len(y_true)                   # fraction correct overall
precision = tp / (tp + fp)                           # of everything I flagged, how much was right?
recall = tp / (tp + fn)                              # of all real spam, how much did I catch?
f1 = 2 * precision * recall / (precision + recall)   # harmonic mean: high only if BOTH are high
print(f"by hand : acc={accuracy:.2f} precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")
```

- **In simple words:** Accuracy is the fraction right; precision and recall look at the spam class; F1 combines them.
- **Exactly:** accuracy = (2+5)/10 = 0.70; precision = 2/(2+1) = 0.67; recall = 2/(2+2) = 0.50; F1 = 2*0.667*0.5/(1.167) = 0.57. The harmonic mean is pulled toward the smaller value, so it is only high if both are high. If tp+fp were 0 this would raise `ZeroDivisionError`.
- **Why it is here:** Knowing the formulas lets you reason about trade-offs in an interview.

### 6. Check with scikit-learn and show the confusion matrix  (`main.py` lines 29–31)

```python
print(f"sklearn : acc={accuracy_score(y_true, y_pred):.2f} precision={precision_score(y_true, y_pred):.2f} "
      f"recall={recall_score(y_true, y_pred):.2f} f1={f1_score(y_true, y_pred):.2f}")
print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))
```

- **In simple words:** The library gives the same numbers.
- **Exactly:** Real output matches: acc 0.70, precision 0.67, recall 0.50, f1 0.57. `confusion_matrix` returns `[[5 1] [2 2]]` with rows = true class (0, 1) and columns = predicted (0, 1), so `[[TN FP] [FN TP]]`. sklearn's `precision_score` returns 0.0 with a warning instead of crashing when no positives are predicted.
- **Why it is here:** Used for cross-checking your own code.

### 7. Show why accuracy lies on imbalanced data  (`main.py` lines 33–37)

```python
# Why accuracy lies: 1 fraud in 100 transactions. A "model" that ALWAYS says "not fraud" is 99% accurate yet has 0 recall.
y_true2 = [1] + [0] * 99
y_pred2 = [0] * 100
print("\nalways-no model: accuracy =", accuracy_score(y_true2, y_pred2),
      "recall =", recall_score(y_true2, y_pred2), "(zero_division guard not needed: recall=0.0)")
```

- **In simple words:** One fraud in 100; a lazy model that always says "not fraud" is almost always right but useless.
- **Exactly:** `[1] + [0]*99` is 100 labels with a single positive; `[0]*100` is the always-no model. Real output: `accuracy = 0.99 recall = 0.0`. It never catches any positive.
- **Why it is here:** On imbalanced problems (fraud, disease, rare intents) use precision, recall, F1 or PR-AUC rather than accuracy.

### 8. Trade-offs and experiments (comments)  (`main.py` lines 38–40)

```python
# Which to prioritise?  Cancer screening -> recall.  Spam folder (don't lose real mail) -> precision.
# TRY IT: change one prediction in y_pred; predict which of precision/recall moves before running.
# BREAK IT: compute precision when the model predicts no positives -> ZeroDivisionError (tp / 0). sklearn warns instead.
```

- **In simple words:** A hint on which metric to prioritise in which situation.
- **Exactly:** Cancer screening wants high recall (do not miss cases); a spam filter wants high precision (do not delete real mail). BREAK IT: manual `tp / (tp + fp)` with zero predicted positives raises `ZeroDivisionError`.
- **Why it is here:** Choosing the metric is a business decision, a common interview question.

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
