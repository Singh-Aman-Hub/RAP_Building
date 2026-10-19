# Step 01 — Complexity and dry runs

## Goal
Feel the difference between O(n) and O(1) lookups, count steps for O(n²) vs O(n), and build the habit of tracing values by hand.

## Run it
```bash
python main.py                                                                 # from this folder
.venv/bin/python 02_dsa_arrays_hashmaps/01_complexity_and_dry_runs/main.py     # from repo root
```

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
