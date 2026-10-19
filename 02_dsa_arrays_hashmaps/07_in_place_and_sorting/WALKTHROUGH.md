# Step 07 — In-place edits and sorting basics

## Goal
Mutate lists with O(1) extra space (read/write pointers) and understand how sorting works and is used.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/07_in_place_and_sorting/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| inner `def` | function inside a function, can use outer variables | `reverse` inside `rotate_right` |
| `k %= n` | remainder assignment | `7 % 5` → `2` |
| `lambda p: p[1]` | anonymous one-expression function | `sorted(x, key=lambda p: p[1])` |
| `a; b` on one line | two statements | avoid in real code |
| `list(data)` | copy a list | so tests don't mutate inputs |
| `out.extend(x)` | append all items of x | |

## Libraries & functions
| Name | Origin | Does | Args | Returns |
|---|---|---|---|---|
| `sorted(it)` | built-in | new sorted list | `key=`, `reverse=` | list |
| `list.sort()` | built-in method | sorts in place | `key=`, `reverse=` | **None** |
| `str.lower` | built-in | key function for case-insensitive sort | | str |

## Execution trace (real run)
`move_zeroes([0,1,0,3,12])`: read=1 (1) swaps into write=0 → `[1,0,0,3,12]`; read=3 (3) swaps with write=1 → `[1,3,0,0,12]`; read=4 (12) → `[1,3,12,0,0]`.

Output of `sort_demo()`:
```
sorted default      : ['Apple', 'banana', 'cherry', 'fig']
sorted by length    : ['fig', 'Apple', 'banana', 'cherry']
sorted case-insens. : ['Apple', 'banana', 'cherry', 'fig']
sorted descending   : [3, 2, 1]
sorted by 2nd item  : [('b', 1), ('a', 2), ('c', 2)]
All in-place/sorting tests passed.
```
`('a', 2)` stays before `('c', 2)`: sorting is *stable*.

| Sort | Time | Space | Note |
|---|---|---|---|
| insertion | O(n²), O(n) if nearly sorted | O(1) | simple, stable |
| merge | O(n log n) | O(n) | stable |
| Python `sorted` (Timsort) | O(n log n) | O(n) | stable, use this |

## Experiments
1. `rotate_right(list(range(1,8)), 3)`; predict first.
2. `x = [3,1,2]; y = x.sort(); print(y)` → `None`. Why?
3. Sort `["b10", "b2"]` by default vs `key=lambda s: int(s[1:])`.

## Debugging practice
`rotate_right(5, 2)` → `TypeError: object of type 'int' has no len()`. The traceback line shows `n = len(nums)`.

## Rebuild checklist
- [ ] `move_zeroes` with read/write pointers.
- [ ] Insertion sort.
- [ ] Merge two sorted lists.

## Say it aloud
"An in-place algorithm reuses the input list, using a read pointer that scans and a write pointer that marks where the next kept element goes, so extra space is O(1). In practice I use Python's built-in sorted, which is O(n log n) and stable; writing insertion sort by hand shows me how sorting works."
