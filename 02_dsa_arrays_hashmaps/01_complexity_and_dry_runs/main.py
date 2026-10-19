"""
GOAL: See time complexity with your own eyes, and learn the "dry run" habit
      (trace values by hand) before writing any LeetCode solution.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/01_complexity_and_dry_runs/main.py
EXPECTED OUTPUT (numbers vary by machine):
    list lookup  (O(n))   : ~0.2 s
    set lookup   (O(1))   : ~0.0005 s
    ...
NEEDS: standard library only.
"""

import time  # standard library: time.perf_counter() gives a high-resolution clock


# ---------------------------------------------------------------------------
# Part 1: O(n) vs O(1) lookups  (why dictionaries and sets matter)
# ---------------------------------------------------------------------------
def time_it(label, func):
    """Run func() once and print how long it took. (A function that takes another function.)"""
    start = time.perf_counter()
    func()
    elapsed = time.perf_counter() - start
    print(f"{label:<28}: {elapsed:.4f} s")


N = 20_000                     # underscores in numbers are just for readability
numbers_list = list(range(N))  # a list: checking `x in list` scans items one by one -> O(n)
numbers_set = set(numbers_list)  # a set: hashing jumps straight to the item -> O(1) average
queries = range(N - 2000, N)   # look up 2000 numbers that sit near the END of the list (worst case)


def list_lookups():
    # TRY IT: change N to 40_000. The list time roughly quadruples (n * n), the set time only doubles.
    return sum(1 for q in queries if q in numbers_list)  # generator expression + sum


def set_lookups():
    return sum(1 for q in queries if q in numbers_set)


time_it("list lookup  (O(n) each)", list_lookups)
time_it("set lookup   (O(1) each)", set_lookups)


# ---------------------------------------------------------------------------
# Part 2: counting steps instead of seconds (machine independent!)
# ---------------------------------------------------------------------------
def count_pairs_brute(nums):
    """Nested loops -> about n*n/2 comparisons -> O(n^2)."""
    steps = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            steps += 1
    return steps


def count_pass_once(nums):
    """One loop -> n steps -> O(n)."""
    steps = 0
    for _ in nums:  # `_` means "I don't need the value"
        steps += 1
    return steps


for n in (10, 100, 1000):
    data = list(range(n))
    print(f"n={n:<5} brute steps={count_pairs_brute(data):<8} single-pass steps={count_pass_once(data)}")
# BREAK IT: replace range(i + 1, ...) with range(i, ...) and notice steps go up by n. Why?


# ---------------------------------------------------------------------------
# Part 3: a dry run, printed.  Do this on paper for EVERY problem before coding.
# ---------------------------------------------------------------------------
def sum_with_trace(nums):
    total = 0
    for index, value in enumerate(nums):  # enumerate gives (position, item) pairs
        total += value
        print(f"  step {index}: value={value} -> total={total}")
    return total


print("Dry run of sum([3, 1, 4]):")
result = sum_with_trace([3, 1, 4])
assert result == 8  # assert crashes loudly if the claim is false -> our tiny test

# Edge cases to ALWAYS check: empty input, one element, duplicates, negatives, all same.
assert sum_with_trace([]) == 0
print("All checks passed.")
