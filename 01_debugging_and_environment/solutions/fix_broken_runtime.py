"""Solution for broken_runtime.py. Root cause: average() is called with an empty list.
Design decision: an empty batch has no average -> return None and let the caller decide (not a fake 0.0)."""


def average(xs):
    if not xs:                   # empty list is falsy
        return None
    return sum(xs) / len(xs)


for i, b in enumerate([[1, 2], [3], []]):
    print(f"batch {i} average:", average(b))
assert average([]) is None and average([2, 4]) == 3
print("verified")
