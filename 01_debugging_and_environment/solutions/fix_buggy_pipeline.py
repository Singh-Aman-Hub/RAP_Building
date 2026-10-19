"""Solution for 09 buggy_pipeline.py: remove the mutable default `results={}` and `labels=[...]`."""

DATA = [("great film", "pos"), ("awful plot", "neg")]


def predict(text):
    return "neg" if ("awful" in text or "boring" in text) else "pos"


def accuracy_per_class(rows):
    results = {}                                   # fresh dict per call (the fix)
    for text, true_label in rows:
        results.setdefault(true_label, []).append(predict(text) == true_label)
    return {k: sum(v) / len(v) for k, v in results.items()}


print(accuracy_per_class(DATA))
second = accuracy_per_class([("awful", "neg"), ("boring", "neg")])
print(second)
assert second == {"neg": 1.0}, "state leaked between calls"
print("verified")
