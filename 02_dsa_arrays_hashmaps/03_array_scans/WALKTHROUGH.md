# Step 03 — Array scans: traversal, min/max, search, duplicates

## Goal
One-pass scans with running state; linear vs binary search; the "initialise from first element" habit.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/03_array_scans/main.py
```

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
