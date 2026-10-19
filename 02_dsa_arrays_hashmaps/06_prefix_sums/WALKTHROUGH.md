# Step 06 — Prefix sums (+ dict)

## Goal
Precompute running totals so any range sum is O(1); combine with a dict to count subarrays.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/06_prefix_sums/main.py
```

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
