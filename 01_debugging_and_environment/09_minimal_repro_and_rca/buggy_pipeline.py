"""
A SILENT bug: no crash, but the numbers are wrong.  Practise the full workflow from DEBUG_JOURNAL_TEMPLATE.md:
  observe -> reproduce -> shrink to a minimal example -> hypothesise -> test -> find root cause -> fix -> verify.

RUN: python buggy_pipeline.py        (runs fine, prints a suspicious result)
Task: "accuracy per class" for a tiny sentiment classifier. Class 'neg' should NOT be 100%.
"""

# (text, true_label)
DATA = [
    ("great film", "pos"), ("loved it", "pos"), ("awful plot", "neg"),
    ("boring and slow", "neg"), ("fantastic acting", "pos"), ("terrible", "neg"),
]


def predict(text):
    # deliberately weak rule: only knows two negative words
    return "neg" if ("awful" in text or "boring" in text) else "pos"


def preprocess(rows):
    cleaned = []
    for text, label in rows:
        cleaned.append((text.lower().strip(), label))
    return cleaned


def accuracy_per_class(rows, labels=["pos", "neg"], results={}):     # <- look closely at this signature
    for label in labels:
        results.setdefault(label, [])
    for text, true_label in rows:
        results[true_label].append(predict(text) == true_label)
    return {k: sum(v) / len(v) for k, v in results.items()}


train_acc = accuracy_per_class(preprocess(DATA))
print("accuracy (first call) :", train_acc)

# Second call with different data: only 'neg' examples that the model gets right
extra = [("awful", "neg"), ("boring", "neg")]
print("accuracy (second call):", accuracy_per_class(preprocess(extra)))
print("expected second call  : only 'neg': 1.0 (the new data has no 'pos' rows)")
print("actual shows 'pos' and a changed 'neg' -> results from the FIRST call leaked into the second. Why?")
