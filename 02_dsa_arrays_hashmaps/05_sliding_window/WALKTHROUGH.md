# Step 05 — Sliding window

## Goal
Maintain a window over a contiguous range and update its value incrementally.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/05_sliding_window/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| sliding window | moving window, window technique | Same | A contiguous range whose edges move forward, updating the answer incrementally. |
| fixed vs variable window | fixed-size vs grow/shrink window | No | Fixed: size k constant. Variable: size changes to satisfy a rule. |
| subarray vs subsequence vs substring | contiguous vs not necessarily contiguous | No | Subarray and substring are contiguous; a subsequence may skip elements. |
| window sum | running sum, rolling sum | Roughly | Sum maintained by adding the new item and subtracting the old one. |
| two pointers vs sliding window | pattern relatives | No | Two pointers often go inward; sliding window keeps both edges moving the same direction. |
| `last_seen` map | last index dict, jump table | Roughly | Lets `left` jump instead of crawling. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–9)

```python
"""
GOAL: Learn the sliding-window pattern: keep a moving range [left, right] and update
      its answer incrementally instead of recomputing from scratch.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/05_sliding_window/main.py
EXPECTED OUTPUT: "All sliding-window tests passed."
NEEDS: standard library only.
"""
```

- **In simple words:** Label for the file: keep a moving range and update its answer incrementally.
- **Exactly:** Docstring only; output appears at the bottom.
- **Why it is here:** States the pattern: do not recompute the window from scratch.

### 2. max_sum_k_brute (fixed window, slow)  (`main.py` lines 12–17)

```python
# --- Fixed-size window ------------------------------------------------------
def max_sum_k_brute(nums, k):
    """Re-add k numbers for every start index: O(n*k)."""
    if k <= 0 or k > len(nums):
        return None
    return max(sum(nums[i:i + k]) for i in range(len(nums) - k + 1))
```

- **In simple words:** Re-add k numbers for every possible start.
- **Exactly:** Guard returns `None` if `k <= 0` or `k > len(nums)`. `range(len(nums) - k + 1)` is every start index; `sum(nums[i:i+k])` slices and adds k numbers; `max(...)` of a generator. O(n*k).
- **Why it is here:** The reference answer for the tests.

### 3. max_sum_k: slide the window  (`main.py` lines 20–29)

```python
def max_sum_k(nums, k):
    """Slide: add the new right element, subtract the element that left. O(n)/O(1)."""
    if k <= 0 or k > len(nums):
        return None
    window = sum(nums[:k])                     # first window
    best = window
    for right in range(k, len(nums)):
        window += nums[right] - nums[right - k]  # one in, one out
        best = max(best, window)
    return best
```

- **In simple words:** Add the new element on the right, subtract the one that fell off the left.
- **Exactly:** `window = sum(nums[:k])` is the first window, computed once. For each `right` from `k` to the end, `window += nums[right] - nums[right - k]` (one in, one out). Trace `[2,1,5,1,3,2], k=3`: 8, then 7, 9, 6 -> best `9`. O(n) time, O(1) space.
- **Why it is here:** Turns O(n*k) into O(n): each element enters and leaves once.

### 4. longest_unique_brute  (`main.py` lines 33–43)

```python
def longest_unique_brute(s):
    """Check every substring: O(n^3) naive / O(n^2) with set. Fine for tiny inputs only."""
    best = 0
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.add(s[j])
            best = max(best, j - i + 1)
    return best
```

- **In simple words:** Check every start and extend until a repeat appears.
- **Exactly:** For each `i`, a fresh `seen` set; extend `j` until `s[j]` repeats (`break`); update `best = max(best, j - i + 1)`. About O(n^2) with a set.
- **Why it is here:** Oracle for the variable-window version.

### 5. longest_unique with a variable window  (`main.py` lines 46–56)

```python
def longest_unique(s):
    """Longest substring without repeating characters. O(n) time, O(k) space.
    last_seen maps char -> its most recent index so `left` can JUMP past a repeat."""
    last_seen = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:  # repeat inside current window
            left = last_seen[ch] + 1                   # shrink window just past old copy
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best
```

- **In simple words:** Grow the window to the right; when a character repeats, jump `left` just past its previous copy.
- **Exactly:** `last_seen` maps character -> last index. `left = best = 0` chained assignment. If `ch` was seen at an index `>= left` (inside the current window), set `left = last_seen[ch] + 1`. Always record `last_seen[ch] = right`, then `best = max(best, right - left + 1)`. Trace `"abba"`: r0 a; r1 b; r2 b repeats -> left=2; r3 a last seen at 0 < left=2, so it is ignored -> window `"ba"` -> best `2`. `"abcabcbb"` -> `3`.
- **Why it is here:** The `>= left` test stops old, already-discarded repeats from dragging `left` backwards.

### 6. min_subarray_len_brute  (`main.py` lines 59–68)

```python
def min_subarray_len_brute(target, nums):
    best = float("inf")
    for i in range(len(nums)):
        total = 0
        for j in range(i, len(nums)):
            total += nums[j]
            if total >= target:
                best = min(best, j - i + 1)
                break
    return 0 if best == float("inf") else best
```

- **In simple words:** Try every start, add numbers until the sum reaches the target, remember the shortest length.
- **Exactly:** `best = float("inf")`; inner loop adds `nums[j]`; when `total >= target` record `j - i + 1` and `break` (longer is worse). Returns `0` if nothing was found. O(n^2).
- **Why it is here:** Reference answer for the sliding version.

### 7. min_subarray_len with a variable window  (`main.py` lines 71–81)

```python
def min_subarray_len(target, nums):
    """Smallest subarray with sum >= target (positive numbers only!). O(n)."""
    left = total = 0
    best = float("inf")
    for right, x in enumerate(nums):
        total += x                              # grow
        while total >= target:                  # shrink while still valid
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best == float("inf") else best
```

- **In simple words:** Grow until the sum is big enough, then shrink from the left while it stays big enough.
- **Exactly:** `total += x` grows the window. `while total >= target:` record `right - left + 1`, subtract `nums[left]`, move `left` up. Trace `(7, [2,3,1,2,4,3])`: the sum first reaches 8 at `right=3` (window `[2,3,1,2]`, length 4); shrinking records lengths 4, then 3 (`[1,2,4]` sums 7); later at `right=5` the window shrinks to `[4,3]` (sum 7) and records length `2`, the answer. Needs non-negative numbers: shrinking assumes removing an element can only lower the sum.
- **Why it is here:** The shrink loop is what makes it O(n): `left` and `right` each move at most n times.

### 8. Tests: fixed window  (`main.py` lines 84–87)

```python
if __name__ == "__main__":
    for nums, k in ([[2, 1, 5, 1, 3, 2], 3], [[1, 2], 2], [[1], 2], [[], 1], [[-1, -2, -3], 2]):
        assert max_sum_k(nums, k) == max_sum_k_brute(nums, k)
    print("max_sum_k([2,1,5,1,3,2], 3) =", max_sum_k([2, 1, 5, 1, 3, 2], 3))
```

- **In simple words:** Compare fast and brute versions on five inputs and print one result.
- **Exactly:** Inputs include `k` bigger than the list (both `None`), empty list, and all negatives `[-1,-2,-3]`. Prints `max_sum_k([2,1,5,1,3,2], 3) = 9`.
- **Why it is here:** Negatives are fine for a fixed window because its size never changes.

### 9. Tests: longest_unique  (`main.py` lines 88–90)

```python
    for s in ("abcabcbb", "bbbbb", "pwwkew", "", "a", "abba"):
        assert longest_unique(s) == longest_unique_brute(s)
    print("longest_unique('abcabcbb') =", longest_unique("abcabcbb"))
```

- **In simple words:** Compare on six strings and print one result.
- **Exactly:** Includes `"bbbbb"`, `"pwwkew"`, `""`, `"a"`, `"abba"`. Prints `longest_unique('abcabcbb') = 3`.
- **Why it is here:** `"abba"` is the classic trap for the `>= left` check.

### 10. Tests: min_subarray_len, final message  (`main.py` lines 91–94)

```python
    for t, nums in ((7, [2, 3, 1, 2, 4, 3]), (4, [1, 4, 4]), (11, [1, 1, 1]), (5, [])):
        assert min_subarray_len(t, nums) == min_subarray_len_brute(t, nums)
    print("min_subarray_len(7, [2,3,1,2,4,3]) =", min_subarray_len(7, [2, 3, 1, 2, 4, 3]))
    print("All sliding-window tests passed.")
```

- **In simple words:** Compare on four cases and print one result.
- **Exactly:** Cases include no solution `(11, [1,1,1])` -> `0` and empty input. Prints `min_subarray_len(7, [2,3,1,2,4,3]) = 2` and `All sliding-window tests passed.`
- **Why it is here:** Covers the "no valid window" return value.

### 11. TRY IT and BREAK IT comments  (`main.py` lines 95–96)

```python
    # TRY IT: in longest_unique remove the `>= left` check and test "abba". Expected 2, you'll get 3. Why?
    # BREAK IT: call min_subarray_len with negatives like (1, [-2, 1]): it returns 0 but the answer is 1. The pattern assumes non-negatives.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** Removing `>= left` makes `longest_unique("abba")` return `3` instead of `2`. With negatives, `min_subarray_len(1, [-2, 1])` returns `0` but the true answer is `1` (the brute force gets it right).
- **Why it is here:** Shows the exact assumption the technique relies on (non-negative numbers for shrinking windows).

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `nums[i:i+k]` | slice of k items | `[1,2,3,4][1:3]` → `[2,3]` |
| `max(gen)` | max over generator | `max(x*x for x in [1,2])` |
| `float("inf")` | bigger than any number | |
| `x += a - b` | augmented assignment with expression | |
| `while cond:` inside `for` | shrink loop inside grow loop | |
| `for i, ch in enumerate(s)` | strings are iterable by character | |

## Libraries & functions
| Name | Origin | Does |
|---|---|---|
| `sum`, `max`, `min`, `len`, `enumerate` | built-ins | aggregates and iteration |
| `dict` as `last_seen` | built-in | char → last index |

## Execution trace (real run)
`max_sum_k([2,1,5,1,3,2], 3)`: first window [2,1,5] = 8. Slide: add `nums[3]=1`, drop `nums[0]=2` → 7; add 3, drop 1 → 9; add 2, drop 5 → 6. Best = `9`.

`longest_unique('abcabcbb')`: window grows to "abc" (3), at second 'a' left jumps to 1, and so on. Prints `3`. `min_subarray_len(7,[2,3,1,2,4,3])` prints `2` (subarray [4,3]). Final: `All sliding-window tests passed.`

Complexity: each pointer only moves forward → at most 2n moves → O(n). Brute force is O(n·k) or O(n²).

## Experiments
1. `max_sum_k(nums, 1)` equals `max(nums)`.
2. `longest_unique("abba")` = 2; remove `>= left` and it breaks.
3. Give `min_subarray_len` a huge target → returns 0 (document that "not found" convention).

## Debugging practice
Negative numbers break the shrink-while logic of `min_subarray_len` (`min_subarray_len(1, [-2, 1])` returns 0 instead of 1): shrinking assumes removing items lowers the sum. Lesson: check *preconditions* of a pattern.

## Rebuild checklist
- [ ] Fixed window max sum.
- [ ] Longest substring without repeats using dict of last index.
- [ ] Minimum subarray length using grow/shrink.

## Say it aloud
"A sliding window keeps a running answer for a contiguous range. For a fixed size I add the new element and drop the old one; for variable size I grow right until a rule breaks, then shrink left. Each index enters and leaves once so it's O(n). It requires the rule to be monotonic, e.g. non-negative numbers."
