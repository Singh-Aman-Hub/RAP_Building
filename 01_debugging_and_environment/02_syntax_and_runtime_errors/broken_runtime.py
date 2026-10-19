"""EXPECTED TO CRASH at runtime (after printing two lines). Use the traceback to find the root cause."""


def average(xs):
    return sum(xs) / len(xs)


print("average of [2, 4]:", average([2, 4]))
batches = [[1, 2], [3], []]          # note the last batch is empty
for i, b in enumerate(batches):
    print(f"batch {i} average:", average(b))
