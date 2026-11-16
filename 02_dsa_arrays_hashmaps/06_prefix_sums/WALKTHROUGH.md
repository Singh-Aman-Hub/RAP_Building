# Step 06 — Prefix sums (+ dict)

## Goal
Precompute running totals so any range sum is O(1); combine with a dict to count subarrays.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/06_prefix_sums/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| prefix sum | cumulative sum, running total, `itertools.accumulate` | Same | `prefix[i]` = sum of the first i items. |
| range sum query | RSQ, subarray sum query | Same | Answer sums over `[l, r]` using two prefix values. |
| subarray | contiguous slice | Same | Consecutive elements only (not a subsequence). |
| suffix product / prefix product | left-product / right-product arrays | Mirror | Same idea in the other direction. |
| prefix sum + hash map | running sum with `seen` dict | Same technique | Counts subarrays with sum k in O(n), including negatives. |
| inclusive vs exclusive end | `[l..r]` vs `[l:r)` | No | `query` is inclusive; Python slices are exclusive of the end. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–9)

```python
"""
GOAL: Use prefix sums to answer "sum of a range" in O(1), and combine them with a dict
      to count subarrays (the bridge between arrays and hash maps).

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/06_prefix_sums/main.py
EXPECTED OUTPUT: prefix list, range sums, "All prefix-sum tests passed."
NEEDS: standard library only.
"""
```

- **In simple words:** Label for the file: prefix sums for fast range sums, plus a dict to count subarrays.
- **Exactly:** Docstring only; output is at the bottom.
- **Why it is here:** States the bridge between arrays and hash maps.

### 2. build_prefix: running totals  (`main.py` lines 12–18)

```python
def build_prefix(nums):
    """prefix[i] = sum of nums[0:i]  (so prefix has len(nums)+1 items and prefix[0] == 0).
    Then sum of nums[l..r] inclusive = prefix[r+1] - prefix[l]."""
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)           # previous total + new element
    return prefix
```

- **In simple words:** Make a list where each entry is the sum of everything before that position.
- **Exactly:** `prefix = [0]`; for each `x`, append `prefix[-1] + x`. `prefix[i]` = sum of `nums[0:i]`, so it has `len(nums)+1` items. For `[3,1,4,1,5]` the real output is `prefix: [0, 3, 4, 8, 9, 14]`. Range sum of `nums[l..r]` inclusive = `prefix[r+1] - prefix[l]`.
- **Why it is here:** The leading 0 means a range starting at index 0 needs no special case.

### 3. RangeSum class: build once, query many times  (`main.py` lines 21–28)

```python
class RangeSum:
    """Build once in O(n), answer each query in O(1). (A class bundles data + methods.)"""

    def __init__(self, nums):                   # __init__ runs when you write RangeSum([...])
        self.prefix = build_prefix(nums)

    def query(self, left, right):               # inclusive both ends
        return self.prefix[right + 1] - self.prefix[left]
```

- **In simple words:** Pay O(n) once to build the prefix list, then answer each range-sum question in O(1).
- **Exactly:** `__init__` stores `self.prefix`. `query(left, right)` (both inclusive) returns `prefix[right+1] - prefix[left]`. `rs.query(1, 3)` = `9 - 3` = `6` (that is 1 + 4 + 1).
- **Why it is here:** Without it every query re-adds up to n numbers: O(n) each instead of O(1).

### 4. subarray_sum_brute  (`main.py` lines 31–40)

```python
def subarray_sum_brute(nums, k):
    """Count contiguous subarrays summing to k. O(n^2)."""
    count = 0
    for i in range(len(nums)):
        total = 0
        for j in range(i, len(nums)):
            total += nums[j]
            if total == k:
                count += 1
    return count
```

- **In simple words:** Count contiguous subarrays that add up to k by trying every start and growing the end.
- **Exactly:** `total` restarts at each `i`; `total += nums[j]`; each time `total == k`, `count += 1` (no `break`, because zeros/negatives could make later totals equal k again). O(n^2).
- **Why it is here:** The oracle for the faster version.

### 5. subarray_sum with prefix sums and a dict  (`main.py` lines 43–52)

```python
def subarray_sum(nums, k):
    """O(n)/O(n). If prefix[j] - prefix[i] == k then a subarray ends at j.
    So for each running total, ask the dict: how many earlier prefixes equal total - k?"""
    seen = {0: 1}                                # the empty prefix (total 0) has occurred once
    total = count = 0
    for x in nums:
        total += x
        count += seen.get(total - k, 0)          # complement lookup again!
        seen[total] = seen.get(total, 0) + 1
    return count
```

- **In simple words:** Keep a running total; ask how many earlier running totals are exactly `k` smaller.
- **Exactly:** If `prefix[j] - prefix[i] == k` then the subarray after `i` up to `j` sums to `k`; so at each step add `seen.get(total - k, 0)`. `seen = {0: 1}` records that the empty prefix (total 0) happened once. After counting, record `total` in `seen`. Trace `([1,1,1], 2)`: total 1 -> need -1: 0 found; total 2 -> need 0: 1 found (count 1); total 3 -> need 1: 1 found (count 2). Result `2`. O(n) time, O(n) space.
- **Why it is here:** Works with negatives and zeros (unlike a sliding window). Same complement lookup as Two Sum.

### 6. product_except_self without division  (`main.py` lines 55–67)

```python
def product_except_self(nums):
    """Product of all others without division. Prefix product pass + suffix product pass. O(n)."""
    n = len(nums)
    result = [1] * n                             # list repetition: [1, 1, ..., 1]
    running = 1
    for i in range(n):                           # result[i] = product of everything LEFT of i
        result[i] = running
        running *= nums[i]
    running = 1
    for i in range(n - 1, -1, -1):               # multiply by everything RIGHT of i (range counts down)
        result[i] *= running
        running *= nums[i]
    return result
```

- **In simple words:** For each position, multiply everything on its left by everything on its right.
- **Exactly:** `result = [1] * n`. First pass left to right: `result[i] = running` (product of items left of i) then `running *= nums[i]`. Second pass `range(n-1, -1, -1)` counts down: `result[i] *= running` (product of items right of i). For `[1,2,3,4]` the first pass gives `[1,1,2,6]`, the second pass walks from the right with `running` = 1, 4, 12, 24 and multiplies it in: `[1*24, 1*12, 2*4, 6*1]` = `[24, 12, 8, 6]`. O(n) time, O(1) extra apart from the output.
- **Why it is here:** Division fails with zeros; two passes work for any input.

### 7. Tests: prefix and range sums  (`main.py` lines 70–75)

```python
if __name__ == "__main__":
    nums = [3, 1, 4, 1, 5]
    print("prefix:", build_prefix(nums))
    rs = RangeSum(nums)
    print("sum of nums[1..3]:", rs.query(1, 3))   # 1 + 4 + 1 = 6
    assert rs.query(0, 4) == 14 and rs.query(2, 2) == 4 and build_prefix([]) == [0]
```

- **In simple words:** Print the prefix list and one range sum, then assert more.
- **Exactly:** Real output: `prefix: [0, 3, 4, 8, 9, 14]`, `sum of nums[1..3]: 6`. Asserts `query(0, 4) == 14`, `query(2, 2) == 4`, `build_prefix([]) == [0]`.
- **Why it is here:** A one-element range and the empty list are the edge cases.

### 8. Tests: subarray_sum vs brute force  (`main.py` lines 76–78)

```python
    for arr, k in (([1, 1, 1], 2), ([1, 2, 3], 3), ([], 0), ([1, -1, 0], 0), ([5], 5), ([5], 6)):
        assert subarray_sum(arr, k) == subarray_sum_brute(arr, k)
    print("subarray_sum([1,1,1], 2) =", subarray_sum([1, 1, 1], 2))
```

- **In simple words:** Compare on six cases, including empty, negatives/zero and no match.
- **Exactly:** Cases: `([1,1,1],2)`, `([1,2,3],3)`, `([],0)`, `([1,-1,0],0)`, `([5],5)`, `([5],6)`. Prints `subarray_sum([1,1,1], 2) = 2`.
- **Why it is here:** `[1,-1,0]` with k=0 has several overlapping answers; a good stress test.

### 9. Tests: product_except_self, final message and comments  (`main.py` lines 79–83)

```python
    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([0, 2, 3]) == [6, 0, 0] and product_except_self([5]) == [1]
    print("All prefix-sum tests passed.")
    # TRY IT: remove `{0: 1}` initialisation and run subarray_sum([1,2,3], 3). The answer drops. Which subarray gets lost?
    # BREAK IT: RangeSum([1,2,3]).query(0, 3) -> IndexError. Read the traceback: which line, which index?
```

- **In simple words:** Assert known answers, including a zero.
- **Exactly:** `[1,2,3,4]` -> `[24,12,8,6]`; `[0,2,3]` -> `[6,0,0]`; `[5]` -> `[1]`. Prints `All prefix-sum tests passed.` TRY IT: without `{0: 1}`, `subarray_sum([1,2,3], 3)` returns `1` instead of `2` (the subarray `[1,2]` that starts at index 0 is lost). BREAK IT: `RangeSum([1,2,3]).query(0, 3)` reads `prefix[4]`, but `prefix` has 4 items (index 0-3) -> `IndexError: list index out of range` on line 28.
- **Why it is here:** Shows why the empty prefix and the `+1` offset exist.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `prefix[-1]` | last element | `[1,2,3][-1]` → `3` |
| `[1] * n` | list of n ones | `[1]*3` → `[1,1,1]` |
| `range(n-1, -1, -1)` | count down to 0 | `list(range(2,-1,-1))` → `[2,1,0]` |
| `class X: def __init__(self, …)` | class with constructor | `RangeSum([1,2])` |
| `self.prefix` | attribute stored on the object | |
| `d.get(k, 0)` | default count | |

## Libraries & functions
Only built-ins (`list.append`, `dict.get`, `range`).

## Execution trace (real run)
`nums = [3, 1, 4, 1, 5]` →
```
prefix: [0, 3, 4, 8, 9, 14]
sum of nums[1..3]: 6
subarray_sum([1,1,1], 2) = 2
All prefix-sum tests passed.
```
`query(1, 3) = prefix[4] − prefix[1] = 9 − 3 = 6`.

`subarray_sum([1,1,1], 2)`: running totals 1,2,3; seen starts `{0:1}`. total=1: look for −1 → 0. total=2: look for 0 → 1 (subarray [1,1] at index 0-1). total=3: look for 1 → 1 (indices 1-2). Count = 2.

## Experiments
1. Remove `{0: 1}` → subarrays starting at index 0 are missed.
2. Call `rs.query(0, 0)` and `rs.query(4, 4)` to see single-element ranges.
3. Time 100,000 queries on a 100,000-item list: prefix vs `sum(nums[l:r+1])`.

## Debugging practice
`RangeSum([1,2,3]).query(0, 3)` → `IndexError: list index out of range` on the `self.prefix[right + 1]` line: the valid `right` is `len(nums)-1`. Read the bottom line first, then the line pointer.

## Rebuild checklist
- [ ] `build_prefix` with leading 0.
- [ ] Range query formula.
- [ ] Subarray Sum Equals K with dict.

## Say it aloud
"A prefix-sum array stores running totals, so the sum of any range is prefix[r+1] − prefix[l] in O(1) after O(n) setup. For counting subarrays summing to k, I keep a dict of how often each prefix total appeared and look up total − k, giving O(n) time and O(n) space."
