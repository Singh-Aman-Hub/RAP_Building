"""Exercise 07: macro-F1 by hand for 3 classes, checked against scikit-learn."""
from sklearn.metrics import f1_score

y_true = ["a", "a", "b", "b", "c", "c", "c"]
y_pred = ["a", "b", "b", "b", "c", "a", "c"]


def f1_for(label):
    tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
    fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
    fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


macro = sum(f1_for(c) for c in "abc") / 3          # unweighted mean over classes
print("macro F1 by hand:", round(macro, 4), "| sklearn:", round(f1_score(y_true, y_pred, average="macro"), 4))
assert abs(macro - f1_score(y_true, y_pred, average="macro")) < 1e-9
