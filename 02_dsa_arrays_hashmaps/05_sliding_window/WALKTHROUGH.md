# Step 05 — Sliding window

## Goal
Maintain a window over a contiguous range and update its value incrementally.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/05_sliding_window/main.py
```

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
Negative numbers break the shrink-while logic of `min_subarray_len` (`(3, [2, -1, 2])`): shrinking assumes removing items lowers the sum. Lesson: check *preconditions* of a pattern.

## Rebuild checklist
- [ ] Fixed window max sum.
- [ ] Longest substring without repeats using dict of last index.
- [ ] Minimum subarray length using grow/shrink.

## Say it aloud
"A sliding window keeps a running answer for a contiguous range. For a fixed size I add the new element and drop the old one; for variable size I grow right until a rule breaks, then shrink left. Each index enters and leaves once so it's O(n). It requires the rule to be monotonic, e.g. non-negative numbers."
