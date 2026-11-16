# Step 03 — Array scans: traversal, min/max, search, duplicates

## Goal
One-pass scans with running state; linear vs binary search; the "initialise from first element" habit.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/03_array_scans/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| array | list (Python), vector (C++), slice (Go) | Roughly | Python lists are dynamic arrays holding references. |
| linear search vs binary search | O(n) scan vs O(log n) halving | No | Binary search needs sorted data. |
| running minimum / maximum | prefix min, tracking variable, Kadane-style state | Roughly | State carried through a single scan so you never re-scan. |
| in-place | O(1) extra space, mutating the input | Roughly | In-place may change the input; "O(1) space" is about extra memory. |
| `float('inf')` vs `None` sentinel | sentinel values | Nearly | `inf` compares correctly with numbers; `None` cannot be compared with `<`. |
| traversal | iteration, scan, pass | Same | Visiting every element once. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–8)

```python
"""
GOAL: Master traversal, searching, min/max tracking and duplicate detection on arrays (Python lists).

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/03_array_scans/main.py
EXPECTED OUTPUT: a few result lines then "All array-scan tests passed."
NEEDS: standard library only.
"""
```

- **In simple words:** Label for the file: loops, min/max, searching and duplicate detection on lists.
- **Exactly:** Docstring only; no output until the test block at the bottom.
- **Why it is here:** States the five patterns practised.

### 2. traversal_demo: three ways to loop  (`main.py` lines 11–19)

```python
# 1. Traversal: three ways to loop.  Know all three.
def traversal_demo(nums):
    for x in nums:                        # value only
        pass
    for i in range(len(nums)):            # index only
        pass
    for i, x in enumerate(nums):          # both (most Pythonic when you need both)
        pass
    return len(nums)
```

- **In simple words:** Visit a list by value, by position, or by both.
- **Exactly:** `for x in nums` gives values; `for i in range(len(nums))` gives indices 0..n-1; `for i, x in enumerate(nums)` gives `(index, value)` pairs. Bodies are `pass` (do nothing). Returns `len(nums)`.
- **Why it is here:** Know all three: use `enumerate` when you need both, never `range(len(...))` just to read values.

### 3. find_max by hand  (`main.py` lines 22–31)

```python
# 2. Minimum and maximum -- by hand (interviewers ask you NOT to use max()).
def find_max(nums):
    """O(n) time, O(1) space. Returns None for empty input (decide & document edge cases!)."""
    if not nums:                          # empty list is "falsy"
        return None
    best = nums[0]                        # start from the first element, not 0 (negatives!)
    for x in nums[1:]:                    # slicing copies the list; fine here, or use range(1, len(nums))
        if x > best:
            best = x
    return best
```

- **In simple words:** Walk the list keeping the biggest value seen so far.
- **Exactly:** Empty list returns `None` (decision documented in the docstring). `best = nums[0]` starts from the first element, not 0, so all-negative lists work: `find_max([-5, -2, -9]) == -2`. `nums[1:]` copies the list (O(n) extra space; `range(1, len(nums))` avoids that). O(n) time.
- **Why it is here:** Interviewers ask you to code it without `max()`. Starting at 0 is the classic bug.

### 4. find_min_max in one pass  (`main.py` lines 34–44)

```python
def find_min_max(nums):
    """One pass returns both.  Returns a tuple (min, max)."""
    if not nums:
        return None, None
    lo = hi = nums[0]                     # chained assignment
    for x in nums:
        if x < lo:
            lo = x
        elif x > hi:
            hi = x
    return lo, hi
```

- **In simple words:** Track the smallest and the largest at the same time.
- **Exactly:** `lo = hi = nums[0]` is a chained assignment (both names bind the same first value). Each element is compared: smaller than `lo` -> new `lo`; else bigger than `hi` -> new `hi`. `find_min_max([4, 2, 8, 2]) == (2, 8)`. Returns a tuple; `(None, None)` for empty.
- **Why it is here:** One pass is half the work of two separate scans (still O(n)).

### 5. second_largest in one pass  (`main.py` lines 47–55)

```python
def second_largest(nums):
    """Largest value strictly smaller than the max.  One pass, O(n)/O(1)."""
    first = second = float("-inf")        # -infinity is smaller than every number
    for x in nums:
        if x > first:
            first, second = x, first      # tuple swap: old max becomes second
        elif first > x > second:          # chained comparison: first > x and x > second
            second = x
    return None if second == float("-inf") else second
```

- **In simple words:** Track the top two distinct values.
- **Exactly:** `first = second = float("-inf")`: negative infinity is below every real number. A new maximum pushes the old `first` down into `second` with `first, second = x, first`. `first > x > second` is a chained comparison for a value strictly between them (skips duplicates of the max). Trace `[5, 9, 9, 3]`: first=5; first=9,second=5; 9 ignored; 3 ignored -> `5`. `[4]` and `[2, 2]` leave `second` at -inf -> `None`.
- **Why it is here:** Using `-inf` instead of 0 or `nums[0]` avoids false answers with negatives; "strictly smaller" is the interview clarification to ask.

### 6. linear_search  (`main.py` lines 58–64)

```python
# 3. Searching
def linear_search(nums, target):
    """O(n). Returns index or -1 (same convention as str.find)."""
    for i, x in enumerate(nums):
        if x == target:
            return i
    return -1
```

- **In simple words:** Look at each item until you find the target.
- **Exactly:** `enumerate` gives the position; returns the first matching index or `-1` (same convention as `str.find`). O(n) time, O(1) space.
- **Why it is here:** Works on any list, sorted or not; the baseline for binary search.

### 7. binary_search on sorted data  (`main.py` lines 67–78)

```python
def binary_search(sorted_nums, target):
    """O(log n) -- ONLY works on SORTED input. Halve the search range each step."""
    lo, hi = 0, len(sorted_nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2              # // is integer (floor) division
        if sorted_nums[mid] == target:
            return mid
        if sorted_nums[mid] < target:
            lo = mid + 1                  # target must be in the right half
        else:
            hi = mid - 1                  # target must be in the left half
    return -1
```

- **In simple words:** Guess the middle; each wrong guess throws away half the remaining list.
- **Exactly:** `lo, hi` are inclusive bounds. `mid = (lo + hi) // 2` uses floor division. Found -> return `mid`; middle too small -> `lo = mid + 1`; too big -> `hi = mid - 1`. Loop ends when `lo > hi` (range empty) -> `-1`. Trace `([1,3,5,7,9], 7)`: lo=0,hi=4 -> mid=2 (5<7) -> lo=3; mid=3 -> `7` found -> returns `3`. O(log n).
- **Why it is here:** Only valid on SORTED input: on an unsorted list it can silently return -1 for a value that is present. The `lo <= hi` condition and the `+1/-1` updates are where most bugs live.

### 8. max_profit_brute  (`main.py` lines 81–88)

```python
# 4. Track a running minimum: Best Time to Buy and Sell Stock (one pass).
def max_profit_brute(prices):
    """Try every buy/sell pair. O(n^2)."""
    best = 0
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            best = max(best, prices[j] - prices[i])
    return best
```

- **In simple words:** Try every buy day and every later sell day and keep the best profit.
- **Exactly:** Double loop with `j > i`; `max(best, prices[j] - prices[i])`. `best` starts at 0 so a falling market yields 0 (no trade). O(n^2).
- **Why it is here:** The correctness baseline used in the tests.

### 9. max_profit in one pass  (`main.py` lines 91–97)

```python
def max_profit(prices):
    """Remember the cheapest price so far; profit if we sold today. O(n)/O(1)."""
    cheapest, best = float("inf"), 0
    for p in prices:
        cheapest = min(cheapest, p)
        best = max(best, p - cheapest)
    return best
```

- **In simple words:** Remember the cheapest price so far and ask what profit selling today would give.
- **Exactly:** `cheapest = float("inf")` and `best = 0` start. Each price: `cheapest = min(cheapest, p)` then `best = max(best, p - cheapest)`. Trace `[7,1,5,3,6,4]`: cheapest 7,1,1,1,1,1; profit today 0,0,4,2,5,3; best 0,0,4,4,5,5 -> `5`. O(n) time, O(1) space.
- **Why it is here:** Buy must come before sell; tracking the running minimum guarantees that order without a second loop.

### 10. find_duplicate_marking: the array as its own marker  (`main.py` lines 100–111)

```python
# 5. Duplicate detection when values are 1..n: use the array as its own marker (O(1) space).
def find_duplicate_marking(nums):
    """nums has n+1 numbers in range 1..n (exactly one repeated). Mark seen by negating index slots."""
    for x in nums:
        idx = abs(x) - 1
        if nums[idx] < 0:                 # already marked -> x is the duplicate
            result = abs(x)
            break
        nums[idx] = -nums[idx]
    else:
        result = None
    return result
```

- **In simple words:** With values 1..n in n+1 slots, use the sign of each slot as a "visited" flag, so no extra set is needed.
- **Exactly:** `idx = abs(x) - 1` maps value to slot. If `nums[idx] < 0` the slot was already flagged, so `abs(x)` is the duplicate; `break` leaves the loop. Otherwise flip the slot negative. `for ... else` sets `result = None` when the loop ends without `break`. Trace `[1,3,4,2,2]`: slots flip to `[-1,3,-4,-2,2]`, then `[-1,-3,-4,-2,2]`, then the last `2` finds slot 1 already negative -> returns `2`. O(n) time, O(1) extra space, but it MUTATES the input.
- **Why it is here:** Shows the trade-off: constant extra space at the price of changing (and needing to restore) the input.

### 11. Tests: max, min/max, second largest, search  (`main.py` lines 114–119)

```python
if __name__ == "__main__":
    assert find_max([3, -1, 9, 2]) == 9 and find_max([-5, -2, -9]) == -2 and find_max([]) is None
    assert find_min_max([4, 2, 8, 2]) == (2, 8) and find_min_max([7]) == (7, 7)
    assert second_largest([5, 9, 9, 3]) == 5 and second_largest([4]) is None and second_largest([2, 2]) is None
    assert linear_search([4, 5, 6], 6) == 2 and linear_search([], 1) == -1
    assert binary_search([1, 3, 5, 7, 9], 7) == 3 and binary_search([1, 3, 5], 4) == -1 and binary_search([], 1) == -1
```

- **In simple words:** Assert the answers on normal and edge inputs.
- **Exactly:** Covers negatives (`[-5, -2, -9]`), empty (`None`/`-1`), single element, duplicates of the max (`[2, 2]` -> `None`), found/not found/empty for binary search. `a and b and c` inside one `assert` requires all to be true.
- **Why it is here:** Each assertion is a documented edge case.

### 12. Tests: profit and duplicate, final message  (`main.py` lines 120–124)

```python
    for prices in ([7, 1, 5, 3, 6, 4], [7, 6, 4, 3, 1], [], [5]):
        assert max_profit(prices) == max_profit_brute(prices)
    print("max_profit([7,1,5,3,6,4]) =", max_profit([7, 1, 5, 3, 6, 4]))
    assert find_duplicate_marking([1, 3, 4, 2, 2]) == 2
    print("All array-scan tests passed.")
```

- **In simple words:** Check the fast version against brute force on four inputs and run the duplicate finder.
- **Exactly:** `for prices in ([7,1,5,3,6,4], [7,6,4,3,1], [], [5])` includes a falling market, empty and one price. Prints `max_profit([7,1,5,3,6,4]) = 5` then `All array-scan tests passed.`
- **Why it is here:** Comparing against the brute force is the standard way to trust an optimised version.

### 13. TRY IT and BREAK IT comments  (`main.py` lines 125–126)

```python
    # TRY IT: call binary_search on an UNSORTED list like [5, 1, 4]; see it fail silently. Why is that dangerous?
    # BREAK IT: in find_max start with best = 0 and test [-5, -2]; the answer is wrong. Which edge case did we miss?
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** `binary_search([5, 1, 4], 5)` returns `-1` although 5 is present (mid=1 holds 1 < 5, so it discards the left half where the 5 sits); starting `find_max` with `best = 0` on `[-5, -2]` returns `0`, which is not even in the list.
- **Why it is here:** Practice finding the edge case that breaks a plausible solution.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `if not nums:` | empty list is falsy | `not []` → `True` |
| `nums[1:]` | slice (copy) from index 1 | `[1,2,3][1:]` → `[2,3]` |
| `lo = hi = nums[0]` | chained assignment | |
| `a, b = b, a` | tuple swap | |
| `first > x > second` | chained comparison | `1 < 2 < 3` |
| `float("-inf")` | smaller than any number | |
| `//` | floor division | `7 // 2` → `3` |
| `for…else` | `else` runs if loop did not `break` | used in `find_duplicate_marking` |

## Libraries & functions
| Name | Origin | Does | Returns |
|---|---|---|---|
| `max(a, b)`, `min(a, b)` | built-in | larger/smaller of values | value |
| `abs(x)` | built-in | absolute value | number |
| `len(x)` | built-in | item count | int |

## Execution trace (real run)
`max_profit([7,1,5,3,6,4])`:
| price | cheapest | profit today | best |
|---|---|---|---|
| 7 | 7 | 0 | 0 |
| 1 | 1 | 0 | 0 |
| 5 | 1 | 4 | 4 |
| 3 | 1 | 2 | 4 |
| 6 | 1 | 5 | 5 |
| 4 | 1 | 3 | 5 |

Output: `max_profit([7,1,5,3,6,4]) = 5` then `All array-scan tests passed.`

`binary_search([1,3,5,7,9], 7)`: lo=0,hi=4 → mid=2 (5<7) → lo=3 → mid=3 (7) → return 3. Three steps instead of up to five; for 1,000,000 items ~20 steps.

## Experiments
1. `binary_search([5, 1, 4], 1)` → wrong (-1): needs sorted input.
2. `second_largest([5, 9, 9, 3])` → 5 (duplicates of the max don't count).
3. `max_profit` on strictly decreasing prices → 0 (never buy).

## Debugging practice
`find_max([-5, -2])` with `best = 0` returns 0 — a value not in the list. Dry-run it: 0 never gets beaten. Fix: initialise from `nums[0]`.

## Rebuild checklist
- [ ] `find_max` handling empty and all-negative.
- [ ] `binary_search` with `lo <= hi` and `mid ± 1`.
- [ ] `max_profit` in one pass.

## Say it aloud
"For min/max I keep one running variable updated as I scan, O(n) time and O(1) space; I start from the first element so negative numbers work. Binary search halves a sorted range each step, O(log n), but requires sorted input."
