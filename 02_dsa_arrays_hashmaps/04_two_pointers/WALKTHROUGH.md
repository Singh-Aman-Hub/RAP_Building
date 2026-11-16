# Step 04 — Two pointers

## Goal
Use two indexes to replace a nested loop: opposite ends moving inward (sorted/palindromic structure).

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/04_two_pointers/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| two pointers | two-pointer technique, left/right indexes, converging pointers | Same | Two indexes moving over the same array, usually inward from the ends. |
| fast/slow pointers | tortoise and hare, runner technique | Related | Two pointers moving at different speeds in the same direction (cycle detection, in-place filtering). |
| two pointers vs sliding window | moving ends vs moving range | No | Both use left/right; sliding window keeps running state for the range between them. |
| palindrome | reads same forwards and backwards | Same | Two-pointer check compares ends moving inward. |
| anchor | fixed element, pivot (not quicksort pivot) | Roughly | In 3Sum the number fixed by the outer loop. |
| in-place | O(1) extra space | Roughly | `reverse_in_place` mutates; `three_sum` makes a sorted copy so it is not in place. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–8)

```python
"""
GOAL: Learn the two-pointer pattern: two indexes walking the array to avoid a nested loop.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/04_two_pointers/main.py
EXPECTED OUTPUT: "All two-pointer tests passed."
NEEDS: standard library only.
"""
```

- **In simple words:** Label for the file: two indexes walking an array to avoid a nested loop.
- **Exactly:** Docstring only. The only printed lines are the two at the end of the test block.
- **Why it is here:** States the pattern: replace an O(n^2) double loop with two moving indexes.

### 2. reverse_in_place: swap from both ends  (`main.py` lines 11–18)

```python
def reverse_in_place(nums):
    """Swap ends moving inward. O(n) time, O(1) extra space. Modifies the list itself."""
    left, right = 0, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]  # tuple swap, no temp variable needed
        left += 1
        right -= 1
    return nums
```

- **In simple words:** Put one finger on each end, swap the two items, move both fingers inward until they meet.
- **Exactly:** `left, right = 0, len(nums) - 1`. `nums[left], nums[right] = nums[right], nums[left]` is a tuple swap (right side is evaluated first, so no temporary is needed). Trace `[1,2,3,4]`: swap(0,3) -> `[4,2,3,1]`; swap(1,2) -> `[4,3,2,1]`; then `left=2, right=1` so the loop stops. O(n) time, O(1) extra space, mutates the list.
- **Why it is here:** Shows the simplest two-pointer shape: pointers move toward each other and stop when they cross.

### 3. is_palindrome ignoring punctuation and case  (`main.py` lines 21–33)

```python
def is_palindrome(s):
    """Ignore non-alphanumerics and case. Pointers start at both ends and meet in the middle."""
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():    # skip punctuation/spaces
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True
```

- **In simple words:** Compare letters from both ends, skipping anything that is not a letter or digit.
- **Exactly:** Two inner `while` loops move a pointer past non-alphanumerics (`str.isalnum()`), guarded by `left < right` so they cannot run past each other. Then `.lower()` on both characters before comparing. Any mismatch returns `False`; reaching the middle returns `True`. `"A man, a plan, a canal: Panama"` -> `True`; `"race a car"` -> `False`; `""` -> `True` (`right = -1`, loop never starts).
- **Why it is here:** Handles real text without building a cleaned copy (O(1) extra space).

### 4. two_sum_sorted_brute  (`main.py` lines 36–41)

```python
def two_sum_sorted_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```

- **In simple words:** The slow reference: try all pairs.
- **Exactly:** Double loop with `j > i`; returns `[i, j]` or `[]`. O(n^2).
- **Why it is here:** Used in the tests to prove the fast version is right.

### 5. two_sum_sorted with two pointers  (`main.py` lines 44–56)

```python
def two_sum_sorted(nums, target):
    """Input MUST be sorted. Sum too small -> move left up; too big -> move right down. O(n)/O(1).
    Compare with the dict version in 02_hashmap_basics: same answer, but O(1) space (needs sorted input)."""
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left, right]
        if total < target:
            left += 1
        else:
            right -= 1
    return []
```

- **In simple words:** On a sorted list, if the sum is too small move the left pointer up; if too big move the right pointer down.
- **Exactly:** `total = nums[left] + nums[right]`. Equal -> return. `total < target` -> `left += 1` (only a bigger left value can help). Else `right -= 1`. Trace `([2,7,11,15], 9)`: (0,3) total 17 > 9 -> right=2; total 13 > 9 -> right=1; total 9 -> `[0, 1]`. O(n) time, O(1) space; REQUIRES sorted input.
- **Why it is here:** Same answer as the dict version in step 02 but without the extra memory, if the input is sorted.

### 6. max_area_brute (Container With Most Water)  (`main.py` lines 59–65)

```python
def max_area_brute(heights):
    """Container With Most Water, brute force O(n^2)."""
    best = 0
    for i in range(len(heights)):
        for j in range(i + 1, len(heights)):
            best = max(best, min(heights[i], heights[j]) * (j - i))
    return best
```

- **In simple words:** Try every pair of walls and compute the water between them.
- **Exactly:** Area = `min(heights[i], heights[j]) * (j - i)` (shorter wall times width). Keeps the maximum. O(n^2).
- **Why it is here:** Baseline for the optimised version.

### 7. max_area with two pointers  (`main.py` lines 68–77)

```python
def max_area(heights):
    """Move the SHORTER wall inward -- moving the taller one can never help. O(n)/O(1)."""
    left, right, best = 0, len(heights) - 1, 0
    while left < right:
        best = max(best, min(heights[left], heights[right]) * (right - left))
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return best
```

- **In simple words:** Start with the widest container, then always move the shorter wall inward.
- **Exactly:** `best = max(best, min(h[left], h[right]) * (right - left))`. If `h[left] < h[right]` move `left`, else `right`. Reason: the area is limited by the shorter wall; moving the taller one only reduces width and cannot raise the limit. For `[1,8,6,2,5,4,8,3,7]` the answer is `49` (walls of height 8 at index 1 and 7 at index 8: min 7 * width 7).
- **Why it is here:** A greedy argument that discards n^2 candidate pairs safely; be able to explain it aloud.

### 8. three_sum: sort and set up  (`main.py` lines 80–87)

```python
def three_sum(nums):
    """All unique triplets summing to 0. Sort, fix one number, two-pointer the rest. O(n^2)."""
    nums = sorted(nums)
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:             # skip duplicate anchors
            continue
        left, right = i + 1, len(nums) - 1
```

- **In simple words:** Sort the numbers so that two pointers can work, and prepare a result list.
- **Exactly:** `nums = sorted(nums)` makes a sorted copy (the input is not modified). `result = []`. The `for i in range(len(nums) - 2)` loop fixes one number (the anchor) and leaves room for two more. `if i > 0 and nums[i] == nums[i - 1]: continue` skips repeated anchors, so duplicate triplets are not produced. `left, right = i + 1, len(nums) - 1` start the inner two-pointer search.
- **Why it is here:** Reduces 3Sum from O(n^3) to O(n^2): one loop times one two-pointer pass.

### 9. three_sum: the inner two-pointer search  (`main.py` lines 88–99)

```python
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                while left < right and nums[left] == nums[left - 1]:  # skip duplicate values
                    left += 1
    return result
```

- **In simple words:** For the fixed anchor, find pairs on its right that bring the total to 0.
- **Exactly:** `total < 0` -> `left += 1`; `total > 0` -> `right -= 1`; equal -> record `[nums[i], nums[left], nums[right]]`, `left += 1`, then `while ... nums[left] == nums[left - 1]` skips equal values so the same triplet is not recorded twice. Trace `[-1,0,1,2,-1,-4]` sorted `[-4,-1,-1,0,1,2]`: anchor -1 (i=1) finds `[-1,-1,2]` then `[-1,0,1]`; the second -1 (i=2) is skipped as a duplicate anchor. Returns `[[-1, -1, 2], [-1, 0, 1]]`.
- **Why it is here:** The duplicate-skipping is the hard part in interviews; without it you return repeated triplets.

### 10. Tests: reverse and palindrome  (`main.py` lines 102–104)

```python
if __name__ == "__main__":
    assert reverse_in_place([1, 2, 3, 4]) == [4, 3, 2, 1] and reverse_in_place([]) == [] and reverse_in_place([9]) == [9]
    assert is_palindrome("A man, a plan, a canal: Panama") and not is_palindrome("race a car") and is_palindrome("")
```

- **In simple words:** Assert normal and edge inputs.
- **Exactly:** `reverse_in_place([1,2,3,4]) == [4,3,2,1]`, empty list and single element unchanged; palindrome examples above.
- **Why it is here:** Empty and single element are the edge cases for pointer loops.

### 11. Tests: two-sum and max_area against brute force, print result  (`main.py` lines 105–109)

```python
    for nums, t in ([[2, 7, 11, 15], 9], [[1, 2, 3, 4, 4], 8], [[1, 2], 10], [[], 1]):
        assert two_sum_sorted(nums, t) == two_sum_sorted_brute(nums, t)
    for h in ([1, 8, 6, 2, 5, 4, 8, 3, 7], [1, 1], [], [5]):
        assert max_area(h) == max_area_brute(h)
    print("max_area([1,8,6,2,5,4,8,3,7]) =", max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))
```

- **In simple words:** Compare each fast function with its brute-force twin on small inputs.
- **Exactly:** Two-sum cases include a duplicate pair `[1,2,3,4,4]` target 8, no solution `[1,2]`, and empty. Prints `max_area([1,8,6,2,5,4,8,3,7]) = 49`.
- **Why it is here:** Brute force is the oracle that makes you trust the clever version.

### 12. Tests: three_sum, final message and comments  (`main.py` lines 110–114)

```python
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]] and three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([]) == [] and three_sum([1, 2]) == []
    print("All two-pointer tests passed.")
    # TRY IT: print left/right every loop of two_sum_sorted for [2,7,11,15], 9 -- that's your dry run.
    # BREAK IT: change `while left < right` to `<=` in reverse_in_place; nothing breaks. In two_sum_sorted it does. Why?
```

- **In simple words:** Check a known result, all zeros (one unique triplet), empty and too-short inputs.
- **Exactly:** `three_sum([0,0,0,0]) == [[0,0,0]]` shows the duplicate-skipping works. Prints `All two-pointer tests passed.` BREAK IT: with `left <= right` in `two_sum_sorted`, `two_sum_sorted([3], 6)` would return `[0, 0]` (same element twice), while reversing in place is unaffected because swapping an element with itself does nothing.
- **Why it is here:** Shows why the loop condition `<` vs `<=` depends on whether pointing at the same element is legal.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `a[i], a[j] = a[j], a[i]` | in-place swap | |
| `while left < right:` | loop until pointers meet | |
| `s.isalnum()` | str is letters/digits only | `"a1".isalnum()` → `True` |
| `s.lower()` | lowercase copy | |
| `sorted(nums)` | new sorted list (original untouched) | |
| `continue` | skip to next loop iteration | |

## Libraries & functions
| Name | Origin | Does | Returns |
|---|---|---|---|
| `str.isalnum`, `str.lower` | built-in methods | character tests / case folding | bool / str |
| `sorted` | built-in | sort copy, Timsort O(n log n) | list |
| `min`, `max` | built-in | pick smaller/larger | value |

## Execution trace (real run)
`two_sum_sorted([2,7,11,15], 9)`:
| left | right | sum | action |
|---|---|---|---|
| 0 (2) | 3 (15) | 17 | too big → right-- |
| 0 (2) | 2 (11) | 13 | too big → right-- |
| 0 (2) | 1 (7) | 9 | found `[0, 1]` |

`max_area([1,8,6,2,5,4,8,3,7])` prints `49` (heights 8 and 7, distance 7). Final line: `All two-pointer tests passed.`

Why it's correct: if sum is too small, the *only* way to grow it is a bigger left value; moving right down could only shrink it.

## Experiments
1. Print `left, right` each loop of `two_sum_sorted`.
2. Run `three_sum([0, 0, 0, 0])` → `[[0, 0, 0]]` (duplicates skipped); remove the skip lines and see duplicates.
3. Compare to dict version (step 02): same answer, O(1) space but needs sorted data.

## Debugging practice
In `two_sum_sorted` change `<` to `<=`: `[3]` with target 6 returns `[0, 0]` (element used twice). Always ask "can the pointers cross/equal?".

## Rebuild checklist
- [ ] Reverse list in place.
- [ ] Two Sum II on sorted array.
- [ ] Palindrome with skipping non-alphanumerics.

## Say it aloud
"On a sorted array I put one pointer at each end. If the sum is too small I move the left pointer up, if too large I move the right pointer down. Each step discards at least one element so it's O(n) time, O(1) space — but it needs sorted input, and sorting costs O(n log n)."
