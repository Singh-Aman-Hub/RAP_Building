# Terms — In-place edits and sorting

Folder: `02_dsa_arrays_hashmaps/07_in_place_and_sorting` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [DSA README](../README.md)

**Terms in this folder:** [In-place algorithm](#in-place-algorithm--concept) · [Read/write (slow-fast) pointers](#readwrite-slow-fast-pointers--concept) · [Tuple swap](#tuple-swap--syntax) · [Move Zeroes](#move-zeroes--concept) · [Remove Duplicates from Sorted Array](#remove-duplicates-from-sorted-array--concept) · [Rotate array by k (three reversals)](#rotate-array-by-k-three-reversals--concept) · [Remainder assignment (`k %= n`)](#remainder-assignment-k--n--syntax) · [Inner (nested) function](#inner-nested-function--syntax) · [Insertion sort](#insertion-sort--concept) · [Merge of two sorted lists](#merge-of-two-sorted-lists--concept) · [list.extend()](#listextend--python-built-in) · [Timsort](#timsort--concept) · [Stable sort](#stable-sort--concept) · [sorted() vs list.sort()](#sorted-vs-listsort--python-built-in) · [key= parameter](#key-parameter--python-built-in) · [lambda](#lambda--syntax) · [reverse=True](#reversetrue--python-built-in) · [Copying a list (`list(data)`)](#copying-a-list-listdata--python-built-in) · [Statement separator (`;`)](#statement-separator---syntax)

---

## In-place algorithm  `(concept)`
**1. Definition:** An in-place algorithm transforms its input data structure directly, using only O(1) extra memory beyond the input itself.
**2. In plain words:** Instead of building a new list, you rearrange the existing one. It saves memory, but the caller's list changes, so you must say so. Interviewers often ask "can you do it in place?" to test whether you can reason about indices.
**3. Example to quote:**
> "`move_zeroes` rearranges the same list with two indexes and no second list, so extra space is O(1)."

**Seen in:** `main.py`, `move_zeroes`, `remove_duplicates_sorted`, `rotate_right`

## Read/write (slow-fast) pointers  `(concept)`
**1. Definition:** A technique with two indexes moving the same direction: `read` scans every element and `write` marks where the next kept element should be placed.
**2. In plain words:** `read` is the scout and `write` is the builder lagging behind. Whenever the scout finds something worth keeping, it goes to the `write` slot and `write` advances. This compacts or partitions an array in one pass.
**3. Example to quote:**
> "In `move_zeroes([0, 1, 0, 3, 12])`, `read` finds 1 and swaps it into slot `write`=0, giving `[1, 0, 0, 3, 12]`, and so on to `[1, 3, 12, 0, 0]`."

**Seen in:** `main.py`, `move_zeroes`, `remove_duplicates_sorted`

## Tuple swap  `(syntax)`
**1. Definition:** `a[i], a[j] = a[j], a[i]` swaps two list items by building a tuple of the right side before assigning.
**2. In plain words:** No temp variable is needed. When `read == write` the swap does nothing, which is harmless.
**3. Example to quote:**
> "I swap the non-zero into the `write` slot with a tuple assignment."

**Seen in:** `main.py`, `move_zeroes`, `rotate_right`, `reverse`

## Move Zeroes  `(concept)`
**1. Definition:** A problem to push all zeros to the end of a list while keeping the relative order of the non-zero items, in place.
**2. In plain words:** One pass with read/write pointers solves it in O(n) time and O(1) space.
**3. Example to quote:**
> "`[0, 1, 0, 3, 12]` becomes `[1, 3, 12, 0, 0]` with the 1, 3, 12 still in their original order."

**Seen in:** `main.py`, `move_zeroes`

## Remove Duplicates from Sorted Array  `(concept)`
**1. Definition:** A problem to keep one copy of each value at the front of a sorted array in place and return the new length.
**2. In plain words:** Because the array is sorted, duplicates are adjacent: compare `nums[read]` with `nums[read - 1]`. Only the first `write` items are meaningful afterwards; the tail is leftover.
**3. Example to quote:**
> "For `[1, 1, 2, 2, 2, 3]` it returns 3 and the first three items are `[1, 2, 3]`."

**Seen in:** `main.py`, `remove_duplicates_sorted`

## Rotate array by k (three reversals)  `(concept)`
**1. Definition:** Rotating an array right by k moves each element k positions to the right with wrap-around; it can be done in place by reversing the whole array, then the first k items, then the rest.
**2. In plain words:** `[1, 2, 3, 4, 5]` rotated by 2 becomes `[4, 5, 1, 2, 3]`. The three-reversal trick uses O(1) extra space and O(n) time.
**3. Example to quote:**
> "To rotate `[1..5]` by 2: reverse all to `[5,4,3,2,1]`, reverse the first 2 to `[4,5,3,2,1]`, reverse the rest to `[4,5,1,2,3]`."

**Seen in:** `main.py`, `rotate_right`

## Remainder assignment (`k %= n`)  `(syntax)`
**1. Definition:** `k %= n` sets `k` to the remainder of `k / n`.
**2. In plain words:** Rotating by the length of the list returns the original list, so `k` can be reduced to the range 0 to n-1. It also stops `k` larger than n from breaking indices.
**3. Example to quote:**
> "`rotate_right([1, 2], 5)` first turns k into `5 % 2 = 1`, then rotates once to `[2, 1]`."

**Seen in:** `main.py`, `rotate_right`

## Inner (nested) function  `(syntax)`
**1. Definition:** A function defined inside another function; it can read the outer function's variables.
**2. In plain words:** `reverse(lo, hi)` is used three times inside `rotate_right` and reuses the outer `nums` without it being passed in. It keeps the helper private to the function that needs it.
**3. Example to quote:**
> "I define `reverse` inside `rotate_right` so it can use `nums` directly."

**Seen in:** `main.py`, `rotate_right`

## Insertion sort  `(concept)`
**1. Definition:** Insertion sort builds a sorted prefix by taking each next item and shifting larger items right until its correct position is found; O(n²) worst case, O(n) on nearly sorted data, stable and in place.
**2. In plain words:** It is how many people sort playing cards in hand. Simple and good for tiny or almost-sorted lists, but too slow for big ones. In practice you call `sorted()`.
**3. Example to quote:**
> "Insertion sort is O(n²) in the worst case but O(n) if the list is already sorted, since nothing needs shifting."

**Seen in:** `main.py`, `insertion_sort`

## Merge of two sorted lists  `(concept)`
**1. Definition:** Merging walks two pointers through two sorted lists, repeatedly taking the smaller front item, producing one sorted list in O(n + m) time; it is the "merge" half of merge sort.
**2. In plain words:** Compare the two fronts, take the smaller, advance that pointer. When one list runs out, add the leftovers of the other. Using `<=` favours the left list on ties, keeping the sort stable.
**3. Example to quote:**
> "`merge_sorted([1, 3, 5], [2, 4])` gives `[1, 2, 3, 4, 5]` in a single pass."

**Seen in:** `main.py`, `merge_sorted`

## list.extend()  `(Python built-in)`
**1. Definition:** `list.extend(iterable)` appends every item of the iterable to the end of the list in place.
**2. In plain words:** After the merge loop, one list is empty and the other holds leftovers; `out.extend(a[i:])` adds all of them at once. Unlike `append`, which adds a single object.
**3. Example to quote:**
> "`[1].extend([2, 3])` makes `[1, 2, 3]`, while `append([2, 3])` would make `[1, [2, 3]]`."

**Seen in:** `main.py`, `merge_sorted`

## Timsort  `(concept)`
**1. Definition:** Timsort is the hybrid merge/insertion sorting algorithm behind Python's `sorted()` and `list.sort()`, running in O(n log n) time (O(n) on already-sorted data) and being stable.
**2. In plain words:** You almost never write your own sort in an interview unless asked; you use `sorted()` and know its cost. It makes a copy, so extra space is O(n).
**3. Example to quote:**
> "Python's built-in sort is Timsort, O(n log n) and stable, so I use `sorted` unless the interviewer asks me to implement one."

**Seen in:** `main.py`, `sort_demo`

## Stable sort  `(concept)`
**1. Definition:** A sort is stable if items that compare equal keep their original relative order.
**2. In plain words:** Sorting `[("a", 2), ("b", 1), ("c", 2)]` by the second value gives `("b", 1)`, then `("a", 2)`, then `("c", 2)`; `a` stays before `c`. It matters when you sort by one field after another.
**3. Example to quote:**
> "Because Python's sort is stable, ties keep their input order: 'a' stays ahead of 'c'."

**Seen in:** `main.py`, `sort_demo`

## sorted() vs list.sort()  `(Python built-in)`
**1. Definition:** `sorted(iterable)` returns a new sorted list and leaves the original; `list.sort()` sorts the list in place and returns `None`.
**2. In plain words:** A classic bug is writing `nums = nums.sort()`, which sets `nums` to None. Use `sorted` when you need the original later.
**3. Example to quote:**
> "`sorted(words)` gives me a new list; `words.sort()` changes the original and returns None."

**Seen in:** `main.py`, `sort_demo`, test lines

## key= parameter  `(Python built-in)`
**1. Definition:** The `key` argument of `sorted`/`sort` is a function applied to each item to produce the value used for comparison.
**2. In plain words:** `key=len` sorts by length, `key=str.lower` sorts ignoring case. By default uppercase letters sort before lowercase because of character codes.
**3. Example to quote:**
> "`sorted(words, key=len)` puts 'fig' first because it is the shortest word."

**Seen in:** `main.py`, `sort_demo`

## lambda  `(syntax)`
**1. Definition:** `lambda args: expression` creates a small anonymous function consisting of one expression.
**2. In plain words:** Handy for one-off key functions. `lambda p: p[1]` means "take the second item of each pair".
**3. Example to quote:**
> "`sorted(pairs, key=lambda p: p[1])` sorts the pairs by their second element."

**Seen in:** `main.py`, `sort_demo`

## reverse=True  `(Python built-in)`
**1. Definition:** The `reverse` argument of `sorted`/`sort` reverses the sort order (largest first).
**2. In plain words:** Use it for descending order instead of reversing afterwards.
**3. Example to quote:**
> "`sorted([3, 1, 2], reverse=True)` gives `[3, 2, 1]`."

**Seen in:** `main.py`, `sort_demo`

## Copying a list (`list(data)`)  `(Python built-in)`
**1. Definition:** `list(data)` creates a new list with the same items (a shallow copy).
**2. In plain words:** The tests sort a copy so the original test data is not changed by the in-place function.
**3. Example to quote:**
> "`insertion_sort(list(data))` sorts a copy, so I can compare with `sorted(data)` afterwards."

**Seen in:** `main.py`, tests

## Statement separator (`;`)  `(syntax)`
**1. Definition:** A semicolon lets you put two statements on one line.
**2. In plain words:** The merge code uses `out.append(a[i]); i += 1` to save space. It is legal but not recommended style in real projects.
**3. Example to quote:**
> "`out.append(a[i]); i += 1` takes the item and advances the pointer on one line, which I would split into two lines in production code."

**Seen in:** `main.py`, `merge_sorted`
