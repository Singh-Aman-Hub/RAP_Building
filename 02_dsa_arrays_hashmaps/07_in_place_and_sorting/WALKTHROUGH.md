# Step 07 — In-place edits and sorting basics

## Goal
Mutate lists with O(1) extra space (read/write pointers) and understand how sorting works and is used.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/07_in_place_and_sorting/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| in-place | O(1) extra space, destructive, mutating | Roughly | The input list itself is modified; "in-place" says nothing about speed. |
| stable sort | order-preserving sort | Same | Equal keys keep their original relative order. |
| Timsort | Python's built-in sort, `sorted()`/`list.sort()` | Same | Hybrid of merge sort and insertion sort; O(n log n) worst, O(n) on sorted data. |
| `sorted()` vs `list.sort()` | new list vs in place | No | `sort()` returns `None`. |
| read/write pointers | slow/fast pointers, reader/writer index | Same | `read` scans, `write` marks where the next kept item goes. |
| merge vs merge sort | combine two sorted lists vs full algorithm | No | Merge sort = split in halves, sort each, then merge. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–8)

```python
"""
GOAL: Modify arrays in place (O(1) extra space) and understand sorting basics.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/07_in_place_and_sorting/main.py
EXPECTED OUTPUT: sorted examples then "All in-place/sorting tests passed."
NEEDS: standard library only.
"""
```

- **In simple words:** Label for the file: change arrays in place and understand sorting basics.
- **Exactly:** Docstring only.
- **Why it is here:** States the two skills: O(1) extra space and sorting.

### 2. move_zeroes with read and write pointers  (`main.py` lines 11–18)

```python
def move_zeroes(nums):
    """Move all 0s to the end, keep the order of the others. Slow/fast pointers. O(n)/O(1)."""
    write = 0                                   # next slot for a non-zero
    for read in range(len(nums)):               # `read` scans, `write` lags behind
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1
    return nums
```

- **In simple words:** `read` scans every item; `write` marks the next spot for a non-zero. Non-zeros are swapped forward.
- **Exactly:** Trace `[0,1,0,3,12]`: read=1 (1) swaps with write=0 -> `[1,0,0,3,12]`; read=3 (3) swaps with write=1 -> `[1,3,0,0,12]`; read=4 (12) swaps with write=2 -> `[1,3,12,0,0]`. Order of non-zeros is kept. O(n) time, O(1) space.
- **Why it is here:** The slow/fast pointer pattern for in-place filtering.

### 3. remove_duplicates_sorted  (`main.py` lines 21–30)

```python
def remove_duplicates_sorted(nums):
    """Sorted input: keep one copy of each value at the FRONT; return new length. O(n)/O(1)."""
    if not nums:
        return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:        # new value -> keep it
            nums[write] = nums[read]
            write += 1
    return write                                # nums[:write] is the answer; the tail is junk
```

- **In simple words:** Keep one copy of each value at the front of a sorted list and return how many are kept.
- **Exactly:** `write = 1` (the first item is always kept). Each `read` item that differs from its left neighbour is copied to `nums[write]`. For `[1,1,2,2,2,3]` the list ends `[1,2,3,2,2,3]` and the function returns `3`; only `nums[:3]` is meaningful, the tail is junk.
- **Why it is here:** Matches the LeetCode contract: return the new length, do not allocate a new list.

### 4. rotate_right: setup  (`main.py` lines 33–38)

```python
def rotate_right(nums, k):
    """Rotate right by k using three reversals. O(n)/O(1).  [1..5], k=2 -> [4,5,1,2,3]"""
    n = len(nums)
    if n == 0:
        return nums
    k %= n                                      # rotating n times is a no-op; % keeps k in range
```

- **In simple words:** Handle the empty list and reduce k to a sensible value.
- **Exactly:** `n = len(nums)`. Empty -> return as is. `k %= n` because rotating n places returns the original (`rotate_right([1,2], 5)` acts like k=1 -> `[2,1]`).
- **Why it is here:** Without `%`, `k >= n` would go out of range or waste work.

### 5. rotate_right: inner reverse and three reversals  (`main.py` lines 40–48)

```python
    def reverse(lo, hi):                        # a function defined inside a function (inner function)
        while lo < hi:
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo, hi = lo + 1, hi - 1

    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
    return nums
```

- **In simple words:** Rotate by reversing the whole list, then each of the two parts.
- **Exactly:** `reverse(lo, hi)` is an inner function that swaps from both ends (it uses `nums` from the enclosing scope). For `[1,2,3,4,5]`, k=2: reverse all -> `[5,4,3,2,1]`; reverse `0..1` -> `[4,5,3,2,1]`; reverse `2..4` -> `[4,5,1,2,3]`. O(n) time, O(1) space.
- **Why it is here:** Three reversals rotate with no extra array; a neat trick interviewers like.

### 6. insertion_sort  (`main.py` lines 51–60)

```python
def insertion_sort(nums):
    """Grow a sorted prefix; insert each new item into place. O(n^2) worst, O(n) if already sorted. Stable."""
    for i in range(1, len(nums)):
        current = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > current:     # shift bigger items one step right
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = current
    return nums
```

- **In simple words:** Keep the left part sorted and slide each new item left into its spot, like sorting playing cards in your hand.
- **Exactly:** `current = nums[i]`; `j` walks left while `nums[j] > current`, shifting bigger items one step right; then `nums[j+1] = current`. It is stable (equal items are not passed). O(n^2) worst case, O(n) when already sorted.
- **Why it is here:** The simplest in-place sort and the building block of Timsort for small runs.

### 7. merge_sorted: the merge half of merge sort  (`main.py` lines 63–74)

```python
def merge_sorted(a, b):
    """Merge two sorted lists into one (the 'merge' half of merge sort). O(n+m)."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:                        # <= keeps it stable (ties favour the left list)
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])                           # one of these is empty; the other holds leftovers
    out.extend(b[j:])
    return out
```

- **In simple words:** Combine two sorted lists by repeatedly taking the smaller front item.
- **Exactly:** `i`, `j` are read positions. `a[i] <= b[j]` takes from `a` on ties (keeps stability). After the loop one list may have leftovers; `out.extend(a[i:])` and `out.extend(b[j:])` add them (one is empty). `merge_sorted([1,3,5],[2,4])` -> `[1,2,3,4,5]`. O(n+m) time, O(n+m) space.
- **Why it is here:** Merging is what makes merge sort O(n log n) and is also a standalone interview question.

### 8. sort_demo: sorted() with key and reverse  (`main.py` lines 77–85)

```python
def sort_demo():
    """Python's built-in sort is Timsort: O(n log n), stable. Learn sorted() vs list.sort()."""
    words = ["banana", "Apple", "cherry", "fig"]
    print("sorted default      :", sorted(words))                       # uppercase sorts before lowercase!
    print("sorted by length    :", sorted(words, key=len))              # key= takes a function
    print("sorted case-insens. :", sorted(words, key=str.lower))
    print("sorted descending   :", sorted([3, 1, 2], reverse=True))
    pairs = [("a", 2), ("b", 1), ("c", 2)]
    print("sorted by 2nd item  :", sorted(pairs, key=lambda p: p[1]))   # lambda = tiny anonymous function; ties keep order (stable)
```

- **In simple words:** Show the common ways to sort using Python's built-in Timsort (O(n log n), stable).
- **Exactly:** Real output: default `['Apple', 'banana', 'cherry', 'fig']` (uppercase letters sort before lowercase by code point); `key=len` -> `['fig', 'Apple', 'banana', 'cherry']`; `key=str.lower` -> case-insensitive; `reverse=True` on `[3,1,2]` -> `[3, 2, 1]`; sorting `[("a",2),("b",1),("c",2)]` by `lambda p: p[1]` -> `[('b', 1), ('a', 2), ('c', 2)]` (the two 2s keep their original order: stable).
- **Why it is here:** `key=` is how you sort objects by a field; `sorted()` returns a new list, `list.sort()` changes it in place.

### 9. Tests: move_zeroes, remove_duplicates, rotate_right  (`main.py` lines 88–93)

```python
if __name__ == "__main__":
    assert move_zeroes([0, 1, 0, 3, 12]) == [1, 3, 12, 0, 0] and move_zeroes([]) == [] and move_zeroes([0, 0]) == [0, 0]
    arr = [1, 1, 2, 2, 2, 3]
    n = remove_duplicates_sorted(arr)
    assert n == 3 and arr[:n] == [1, 2, 3] and remove_duplicates_sorted([]) == 0
    assert rotate_right([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3] and rotate_right([1, 2], 5) == [2, 1] and rotate_right([], 3) == []
```

- **In simple words:** Assert the answers and the edge cases.
- **Exactly:** Empty and all-zero lists; `arr[:n] == [1,2,3]` for the duplicates; rotation with `k > len` and an empty list.
- **Why it is here:** Edge cases (empty, all same, k larger than n) are where in-place code breaks.

### 10. Tests: insertion_sort, merge_sorted, sort_demo, final message  (`main.py` lines 94–98)

```python
    for data in ([5, 2, 4, 6, 1, 3], [], [1], [2, 2, 1], [1, 2, 3]):
        assert insertion_sort(list(data)) == sorted(data)   # list(data) copies so we don't mutate the test input
    assert merge_sorted([1, 3, 5], [2, 4]) == [1, 2, 3, 4, 5] and merge_sorted([], [1]) == [1] and merge_sorted([], []) == []
    sort_demo()
    print("All in-place/sorting tests passed.")
```

- **In simple words:** Compare with `sorted` and run the printing demo.
- **Exactly:** `insertion_sort(list(data))` copies first so the test data is not mutated. Prints the five `sorted` lines then `All in-place/sorting tests passed.`
- **Why it is here:** Using the built-in `sorted` as oracle is the standard way to test a hand-written sort.

### 11. TRY IT and BREAK IT comments  (`main.py` lines 99–100)

```python
    # TRY IT: change `if nums[read] != 0` in move_zeroes to `== 0`; predict the output before running.
    # BREAK IT: call rotate_right(5, 2) with an int instead of a list. Read the TypeError.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** Changing `!= 0` to `== 0` in `move_zeroes` moves zeros to the FRONT: `[0,1,0,3,12]` -> `[0,0,1,3,12]`. `rotate_right(5, 2)` raises `TypeError: object of type 'int' has no len()`.
- **Why it is here:** Predicting before running trains your dry-run skill.

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
