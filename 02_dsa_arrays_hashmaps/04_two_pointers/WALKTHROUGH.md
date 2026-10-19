# Step 04 — Two pointers

## Goal
Use two indexes to replace a nested loop: opposite ends moving inward (sorted/palindromic structure).

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/04_two_pointers/main.py
```

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
