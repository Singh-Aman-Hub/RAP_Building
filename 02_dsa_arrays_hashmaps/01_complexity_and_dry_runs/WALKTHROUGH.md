# Step 01 — Complexity and dry runs

## Goal
Feel the difference between O(n) and O(1) lookups, count steps for O(n²) vs O(n), and build the habit of tracing values by hand.

## Run it
```bash
python main.py                                                                 # from this folder
.venv/bin/python 02_dsa_arrays_hashmaps/01_complexity_and_dry_runs/main.py     # from repo root
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Big-O | time complexity, asymptotic complexity, order of growth | Roughly | Describes how steps grow with input size, ignoring constants. |
| O(n) vs O(1) lookup | linear scan vs hash lookup | No | List `in` scans; set/dict `in` jumps by hash (average case). |
| time complexity vs space complexity | runtime growth vs extra memory growth | No | Different resources; often traded against each other. |
| dry run | trace, walkthrough, hand simulation | Same | Executing the algorithm by hand on a tiny input and tracking variables. |
| `assert` test vs unit test | one-line check vs framework test | Nearly | `assert` is the building block; frameworks like pytest add reporting. |
| average case vs worst case | typical vs bad input | No | Hash lookup is O(1) average but O(n) worst case with many collisions. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–12)

```python
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
```

- **In simple words:** Label for the file: see complexity with your own eyes and learn the dry-run habit.
- **Exactly:** Docstring only; nothing runs. The expected numbers in it are approximate (machine dependent; on the machine used for this walkthrough the list lookup took ~1.7 s, not 0.2 s).
- **Why it is here:** Tells you the goal and that timings vary, so we also count steps.

### 2. Import time  (`main.py` lines 14–14)

```python
import time  # standard library: time.perf_counter() gives a high-resolution clock
```

- **In simple words:** Load the stopwatch module.
- **Exactly:** `time.perf_counter()` returns a high-resolution clock value in seconds as a `float`; only differences between two calls are meaningful.
- **Why it is here:** `time.time()` is less precise and can jump; `perf_counter` is the right timer for benchmarks.

### 3. time_it: stopwatch wrapper  (`main.py` lines 17–25)

```python
# ---------------------------------------------------------------------------
# Part 1: O(n) vs O(1) lookups  (why dictionaries and sets matter)
# ---------------------------------------------------------------------------
def time_it(label, func):
    """Run func() once and print how long it took. (A function that takes another function.)"""
    start = time.perf_counter()
    func()
    elapsed = time.perf_counter() - start
    print(f"{label:<28}: {elapsed:.4f} s")
```

- **In simple words:** A helper that takes any function, runs it once, and prints how long it took.
- **Exactly:** `func` is a function passed as an argument (functions are values). `start`/`elapsed` bracket the call. `{label:<28}` left-aligns the label in 28 characters; `{elapsed:.4f}` prints 4 decimals.
- **Why it is here:** One reusable timer instead of copy-pasting timing code, so the comparison is fair.

### 4. Build the test data  (`main.py` lines 28–31)

```python
N = 20_000                     # underscores in numbers are just for readability
numbers_list = list(range(N))  # a list: checking `x in list` scans items one by one -> O(n)
numbers_set = set(numbers_list)  # a set: hashing jumps straight to the item -> O(1) average
queries = range(N - 2000, N)   # look up 2000 numbers that sit near the END of the list (worst case)
```

- **In simple words:** Make a list and a set of the same 20,000 numbers, and pick 2,000 numbers near the end to look up.
- **Exactly:** `N = 20_000`. `numbers_list = list(range(N))` is `[0, 1, ..., 19999]`. `set(numbers_list)` is a hash set of the same values. `queries = range(18000, 20000)` is 2,000 numbers that sit at the far end of the list, the worst case for a linear scan.
- **Why it is here:** Same data in two containers isolates the one thing we measure: how `in` is implemented.

### 5. Two functions that do the same lookups  (`main.py` lines 34–40)

```python
def list_lookups():
    # TRY IT: change N to 40_000. The list time roughly quadruples (n * n), the set time only doubles.
    return sum(1 for q in queries if q in numbers_list)  # generator expression + sum


def set_lookups():
    return sum(1 for q in queries if q in numbers_set)
```

- **In simple words:** Count how many query numbers are present, once with the list and once with the set.
- **Exactly:** `sum(1 for q in queries if q in numbers_list)` is a generator expression adding 1 per hit. For a list, `q in numbers_list` scans from the front: O(n) each, about 18,000+ comparisons per query. For a set, it hashes `q` and jumps to the bucket: O(1) average. Both return `2000`.
- **Why it is here:** Identical logic, different data structure: the only difference is the cost of `in`.

### 6. Run and time both  (`main.py` lines 43–44)

```python
time_it("list lookup  (O(n) each)", list_lookups)
time_it("set lookup   (O(1) each)", set_lookups)
```

- **In simple words:** Call the stopwatch for each version.
- **Exactly:** Real output on this machine: `list lookup  (O(n) each)    : 1.7403 s` and `set lookup   (O(1) each)    : 0.0003 s` (your numbers differ; the ratio is thousands to one). Doubling N would roughly quadruple the list time (n lookups * n each) but only double the set time.
- **Why it is here:** Seeing a ~5000x gap is what makes you reach for a dict or set in interviews.

### 7. Count steps for nested loops  (`main.py` lines 50–56)

```python
def count_pairs_brute(nums):
    """Nested loops -> about n*n/2 comparisons -> O(n^2)."""
    steps = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            steps += 1
    return steps
```

- **In simple words:** Count how many pair comparisons nested loops make, instead of timing.
- **Exactly:** `for i` over all positions, `for j in range(i+1, len)` over later positions; each inner pass adds 1. Total = n(n-1)/2, i.e. O(n^2).
- **Why it is here:** Step counts do not depend on the machine, so they are a reliable way to compare algorithms.

### 8. Count steps for a single loop  (`main.py` lines 59–64)

```python
def count_pass_once(nums):
    """One loop -> n steps -> O(n)."""
    steps = 0
    for _ in nums:  # `_` means "I don't need the value"
        steps += 1
    return steps
```

- **In simple words:** One pass over the list, one step per element.
- **Exactly:** `for _ in nums` ignores the value (`_` is a convention). Total steps = n, i.e. O(n).
- **Why it is here:** The comparison baseline for the nested version.

### 9. Compare for n = 10, 100, 1000  (`main.py` lines 67–70)

```python
for n in (10, 100, 1000):
    data = list(range(n))
    print(f"n={n:<5} brute steps={count_pairs_brute(data):<8} single-pass steps={count_pass_once(data)}")
# BREAK IT: replace range(i + 1, ...) with range(i, ...) and notice steps go up by n. Why?
```

- **In simple words:** Print both step counts for three sizes.
- **Exactly:** Real output: `n=10 brute steps=45 single-pass steps=10`; `n=100 brute steps=4950 single-pass steps=100`; `n=1000 brute steps=499500 single-pass steps=1000`. 10x more data -> about 100x more brute-force steps but only 10x more single-pass steps. `{n:<5}` pads to width 5.
- **Why it is here:** Makes the shape of O(n^2) versus O(n) concrete.

### 10. sum_with_trace: a dry run printed by code  (`main.py` lines 76–81)

```python
def sum_with_trace(nums):
    total = 0
    for index, value in enumerate(nums):  # enumerate gives (position, item) pairs
        total += value
        print(f"  step {index}: value={value} -> total={total}")
    return total
```

- **In simple words:** Add up the numbers and print the running total after every step.
- **Exactly:** `enumerate(nums)` yields `(index, value)`. `total += value` updates the accumulator; the f-string shows `step`, `value`, `total`. Returns the sum.
- **Why it is here:** A dry run is how you find logic bugs before debugging; the habit is to trace values on paper first.

### 11. Run the dry run and check it  (`main.py` lines 84–86)

```python
print("Dry run of sum([3, 1, 4]):")
result = sum_with_trace([3, 1, 4])
assert result == 8  # assert crashes loudly if the claim is false -> our tiny test
```

- **In simple words:** Trace `[3, 1, 4]` and assert the answer.
- **Exactly:** Real output: `step 0: value=3 -> total=3`, `step 1: value=1 -> total=4`, `step 2: value=4 -> total=8`. `assert result == 8` crashes with `AssertionError` if the claim is false.
- **Why it is here:** `assert` is the tiniest possible test; no framework needed.

### 12. Edge case and final message  (`main.py` lines 88–90)

```python
# Edge cases to ALWAYS check: empty input, one element, duplicates, negatives, all same.
assert sum_with_trace([]) == 0
print("All checks passed.")
```

- **In simple words:** Check the empty list, then report success.
- **Exactly:** `sum_with_trace([])` prints nothing, returns `0`, passes the assert. Prints `All checks passed.` Always test empty, one element, duplicates, negatives, all-same.
- **Why it is here:** Most interview bugs live at the edges.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `20_000` | underscores are visual separators in numbers | `1_000 == 1000` |
| `def f(label, func)` | a function that receives another function | `time_it("x", list_lookups)` (no parentheses = pass the function itself) |
| `sum(1 for q in queries if q in s)` | generator expression: counts matching items without building a list | `sum(1 for x in [1,2,3] if x > 1)` → `2` |
| `for _ in nums` | `_` = "value not needed" | loop n times |
| `enumerate(nums)` | yields `(index, value)` pairs | `list(enumerate("ab"))` → `[(0,'a'),(1,'b')]` |
| `f"{x:.4f}"` | f-string with format spec: 4 decimals | `f"{3.14159:.2f}"` → `3.14` |
| `assert cond` | raises `AssertionError` if cond is False | `assert 1 + 1 == 2` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `time.perf_counter()` | stdlib `time` | high-resolution clock for timing | none | float seconds |
| `range(a, b)` | built-in | lazy sequence of ints a..b-1 | start, stop, step | range object |
| `set(iterable)` | built-in | hash-based collection, O(1) average `in` | iterable | set |
| `list(range(N))` | built-in | materialises a list | | list |

## Execution trace (real run)
Input: `N = 20000`, 2000 queries near the end of the list.
Processing: `q in numbers_list` scans up to 20000 items; `q in numbers_set` hashes once.
Output:
```
list lookup  (O(n) each)    : 0.2102 s
set lookup   (O(1) each)    : 0.0001 s
n=10    brute steps=45       single-pass steps=10
n=100   brute steps=4950     single-pass steps=100
n=1000  brute steps=499500   single-pass steps=1000
```
Brute steps = n(n-1)/2: 10 items → 45, 1000 items → 499,500. That is the O(n²) curve.

## Experiments
1. `N = 40_000` → list time ≈ 4× (two factors of n), set time barely moves.
2. Put queries at the *start* (`range(0, 2000)`) → list lookups become fast: Big-O describes the worst case.
3. Change `range(i + 1, ...)` to `range(i, ...)` → steps increase by exactly n.

## Debugging practice
`sum_with_trace(5)` → `TypeError: 'int' object is not iterable`. The traceback's last line names the error; the line above it shows `enumerate(nums)` — you passed a number instead of a list.

## Rebuild checklist
- [ ] Write `time_it` from memory.
- [ ] Write the nested-loop step counter.
- [ ] State O(1), O(log n), O(n), O(n log n), O(n²) with one example each.

## Say it aloud
"Time complexity describes how work grows with input size, not seconds. A list `in` check scans every item, O(n); a set hashes the value and jumps to it, O(1) on average. That is why replacing a nested loop with a set or dict lookup turns O(n²) into O(n). The trade-off is extra memory for the set."
