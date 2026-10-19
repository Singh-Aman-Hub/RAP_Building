"""
GOAL: Compute accuracy, precision, recall and F1 by hand, then check them with scikit-learn.
      Learn why accuracy alone can lie on imbalanced data.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/07_metrics/main.py
EXPECTED OUTPUT: confusion counts, the four metrics, a "99% accurate but useless" demo.
NEEDS: scikit-learn
"""

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1 = spam (the "positive" class we care about), 0 = not spam
y_true = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
y_pred = [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]

# Count the four outcomes
tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)   # true positive: said spam, was spam
fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)   # false positive: said spam, was NOT spam
fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)   # false negative: missed a spam
tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)   # true negative
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

accuracy = (tp + tn) / len(y_true)                   # fraction correct overall
precision = tp / (tp + fp)                           # of everything I flagged, how much was right?
recall = tp / (tp + fn)                              # of all real spam, how much did I catch?
f1 = 2 * precision * recall / (precision + recall)   # harmonic mean: high only if BOTH are high
print(f"by hand : acc={accuracy:.2f} precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")
print(f"sklearn : acc={accuracy_score(y_true, y_pred):.2f} precision={precision_score(y_true, y_pred):.2f} "
      f"recall={recall_score(y_true, y_pred):.2f} f1={f1_score(y_true, y_pred):.2f}")
print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))

# Why accuracy lies: 1 fraud in 100 transactions. A "model" that ALWAYS says "not fraud" is 99% accurate yet has 0 recall.
y_true2 = [1] + [0] * 99
y_pred2 = [0] * 100
print("\nalways-no model: accuracy =", accuracy_score(y_true2, y_pred2),
      "recall =", recall_score(y_true2, y_pred2), "(zero_division guard not needed: recall=0.0)")
# Which to prioritise?  Cancer screening -> recall.  Spam folder (don't lose real mail) -> precision.
# TRY IT: change one prediction in y_pred; predict which of precision/recall moves before running.
# BREAK IT: compute precision when the model predicts no positives -> ZeroDivisionError (tp / 0). sklearn warns instead.
