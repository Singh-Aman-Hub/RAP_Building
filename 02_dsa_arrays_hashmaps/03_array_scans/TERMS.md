# Terms — Array scans

Folder: `02_dsa_arrays_hashmaps/03_array_scans` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [DSA README](../README.md)

**Terms in this folder:** [Array (Python list)](#array-python-list--python-built-in) · [Traversal](#traversal--concept) · [Running variable (running min / max)](#running-variable-running-min--max--concept) · [Truthiness of an empty list (`if not nums`)](#truthiness-of-an-empty-list-if-not-nums--syntax) · [Slicing (`nums[1:]`)](#slicing-nums1--syntax) · [Chained assignment (`lo = hi = nums[0]`)](#chained-assignment-lo--hi--nums0--syntax) · [Tuple swap](#tuple-swap--syntax) · [Chained comparison](#chained-comparison--syntax) · [float("-inf") / float("inf")](#float-inf--floatinf--python-built-in) · [Linear search](#linear-search--concept) · [Binary search](#binary-search--concept) · [Floor division (`//`)](#floor-division---syntax) · [While loop with two bounds (`lo <= hi`)](#while-loop-with-two-bounds-lo--hi--syntax) · [max() and min()](#max-and-min--python-built-in) · [Best Time to Buy and Sell Stock](#best-time-to-buy-and-sell-stock--concept) · [abs()](#abs--python-built-in) · [In-place marking (array as its own hash set)](#in-place-marking-array-as-its-own-hash-set--concept) · [for...else](#forelse--syntax) · [Return value convention (None / -1)](#return-value-convention-none---1--concept)

---

## Array (Python list)  `(Python built-in)`
**1. Definition:** An array is an ordered, index-addressable sequence of items; in Python the usual array is the built-in `list`, which holds any objects and can grow.
**2. In plain words:** A row of numbered boxes starting at 0. Reading `nums[i]` is O(1); searching for a value is O(n). Most interview "array" problems are really Python list problems.
**3. Example to quote:**
> "I treat the Python list as an array: `nums[0]` is the first element and `nums[-1]` the last, both O(1)."

**Seen in:** `main.py`, all functions

## Traversal  `(concept)`
**1. Definition:** Traversal is visiting each element of a collection in order.
**2. In plain words:** The file shows three loop styles: by value (`for x in nums`), by index (`for i in range(len(nums))`), and both (`for i, x in enumerate(nums)`). Choose the one that gives you what you need.
**3. Example to quote:**
> "If I need both the position and the value I use `enumerate`; if only the value, a plain `for x in nums`."

**Seen in:** `main.py`, `traversal_demo`

## Running variable (running min / max)  `(concept)`
**1. Definition:** A running variable is a value updated as you scan, so that after each element it holds the answer for the part of the array seen so far.
**2. In plain words:** `best` holds the largest value seen so far; `cheapest` holds the lowest price seen so far. This gives O(n) time and O(1) space with a single pass. The key is initialising it correctly (from the first element or ±infinity), not with 0.
**3. Example to quote:**
> "I keep `best` as the biggest number seen so far and update it whenever I find a larger one, so one pass is enough."

**Seen in:** `main.py`, `find_max`, `max_profit`

## Truthiness of an empty list (`if not nums`)  `(syntax)`
**1. Definition:** In Python, empty containers (`[]`, `{}`, `""`) evaluate as False in a boolean context, so `if not nums` is True for an empty list.
**2. In plain words:** A short way to guard the empty-input edge case before accessing `nums[0]`, which would raise `IndexError`.
**3. Example to quote:**
> "`not []` is True, so `if not nums: return None` stops me from touching `nums[0]` on an empty list."

**Seen in:** `main.py`, `find_max`, `find_min_max`

## Slicing (`nums[1:]`)  `(syntax)`
**1. Definition:** Slicing `seq[start:stop]` returns a new list containing the items from start up to but excluding stop.
**2. In plain words:** `nums[1:]` means "everything except the first element". It creates a copy, which costs O(n) extra memory; the code notes you could use `range(1, len(nums))` instead.
**3. Example to quote:**
> "`[1, 2, 3][1:]` is `[2, 3]`. It is a copy, so on huge lists I would use an index range."

**Seen in:** `main.py`, `find_max`

## Chained assignment (`lo = hi = nums[0]`)  `(syntax)`
**1. Definition:** Chained assignment assigns the same value to several names in one statement.
**2. In plain words:** Both `lo` and `hi` start as the first element, because a one-item array has the same min and max. With numbers it is safe; with lists both names would point to the same object.
**3. Example to quote:**
> "`lo = hi = nums[0]` starts both trackers at the first element, then one pass improves them."

**Seen in:** `main.py`, `find_min_max`

## Tuple swap  `(syntax)`
**1. Definition:** `a, b = b, a` evaluates the right side first, then assigns, swapping two values without a temporary variable; `first, second = x, first` shifts values the same way.
**2. In plain words:** In `second_largest`, when a new maximum arrives the old maximum becomes second, and both updates happen in one line so neither overwrites the other too early.
**3. Example to quote:**
> "`first, second = x, first` makes the new value the maximum and the old maximum the runner-up in one step."

**Seen in:** `main.py`, `second_largest`

## Chained comparison  `(syntax)`
**1. Definition:** `a > b > c` is shorthand for `a > b and b > c`, evaluating `b` once.
**2. In plain words:** It reads like maths. `first > x > second` means x lies strictly between the two current leaders.
**3. Example to quote:**
> "`1 < 2 < 3` is True; I use `first > x > second` to check x is between the top two."

**Seen in:** `main.py`, `second_largest`

## float("-inf") / float("inf")  `(Python built-in)`
**1. Definition:** `float("-inf")` and `float("inf")` create floating-point values that are smaller or larger than every other number.
**2. In plain words:** They are safe starting values for "smallest so far" or "largest so far" trackers when you cannot assume the data is positive. They avoid the bug of starting from 0 with all-negative data.
**3. Example to quote:**
> "I start `cheapest` at infinity so the first price always replaces it."

**Seen in:** `main.py`, `second_largest`, `max_profit`

## Linear search  `(concept)`
**1. Definition:** Linear search checks each element in order until it finds the target or reaches the end, O(n).
**2. In plain words:** It works on any list, sorted or not. The convention here is to return the index, or -1 when absent (like `str.find`).
**3. Example to quote:**
> "`linear_search([4, 5, 6], 6)` returns 2 after three comparisons; if absent it returns -1."

**Seen in:** `main.py`, `linear_search`

## Binary search  `(concept)`
**1. Definition:** Binary search finds a target in a sorted array by repeatedly comparing with the middle element and discarding the half that cannot contain it, O(log n).
**2. In plain words:** Like guessing a number between 1 and 100: always guess the middle and ask "higher or lower?". For a million items it takes about 20 steps. It silently gives wrong answers on unsorted data.
**3. Example to quote:**
> "On `[1, 3, 5, 7, 9]` looking for 7: mid is 5, too small, so go right; mid is 7, found at index 3. Binary search needs sorted input."

```python
mid = (lo + hi) // 2
```
**Seen in:** `main.py`, `binary_search`

## Floor division (`//`)  `(syntax)`
**1. Definition:** `a // b` divides and rounds down to the nearest integer.
**2. In plain words:** An index must be a whole number, so the midpoint uses `//`: `7 // 2` is 3, whereas `7 / 2` is 3.5 and would raise `TypeError` when used as an index.
**3. Example to quote:**
> "`mid = (lo + hi) // 2` gives a valid integer index."

**Seen in:** `main.py`, `binary_search`

## While loop with two bounds (`lo <= hi`)  `(syntax)`
**1. Definition:** `while lo <= hi:` repeats while the search range still contains at least one candidate position.
**2. In plain words:** Using `<=` keeps a one-element range valid; each iteration moves `lo` or `hi` past `mid`, so the range shrinks and the loop ends. Getting `<`/`<=` or `mid ± 1` wrong is the classic off-by-one bug.
**3. Example to quote:**
> "I use `lo <= hi` so the last remaining element is still checked, and I move to `mid + 1` or `mid - 1` so I never revisit mid."

**Seen in:** `main.py`, `binary_search`

## max() and min()  `(Python built-in)`
**1. Definition:** `max(a, b, ...)` and `min(a, b, ...)` return the largest or smallest of their arguments (or of one iterable).
**2. In plain words:** Used inside the stock-profit loop to keep the best profit and the cheapest price. Interviewers sometimes want you to write the scan by hand, which is what `find_max` demonstrates.
**3. Example to quote:**
> "`best = max(best, p - cheapest)` keeps the larger of the old best and today's profit."

**Seen in:** `main.py`, `max_profit`

## Best Time to Buy and Sell Stock  `(concept)`
**1. Definition:** A problem asking for the maximum profit from buying on one day and selling on a later day, solved in one pass by tracking the cheapest price so far.
**2. In plain words:** For each day, assume you sell today and bought at the cheapest earlier price. Brute force tries every pair, O(n²); the running-minimum solution is O(n) time and O(1) space. If prices only fall, the answer is 0.
**3. Example to quote:**
> "For `[7, 1, 5, 3, 6, 4]` the cheapest is 1 and the best sell is 6, so max profit is 5."

**Seen in:** `main.py`, `max_profit`, `max_profit_brute`

## abs()  `(Python built-in)`
**1. Definition:** `abs(x)` returns the absolute (non-negative) value of a number.
**2. In plain words:** Needed because the duplicate finder marks slots by turning numbers negative, so later it must read the original magnitude.
**3. Example to quote:**
> "`abs(-3)` is 3, which lets me recover the original value after I flipped its sign as a marker."

**Seen in:** `main.py`, `find_duplicate_marking`

## In-place marking (array as its own hash set)  `(concept)`
**1. Definition:** In-place marking uses the input array itself to record "seen" state (for example by negating the number at index x-1), giving O(1) extra space.
**2. In plain words:** When values are known to be in 1..n, the value x points to slot x-1. If that slot is already negative, x was seen before, so it is the duplicate. The catch: it modifies the input and only works under that range assumption.
**3. Example to quote:**
> "For `[1, 3, 4, 2, 2]`, the second 2 finds slot 1 already negative, so 2 is the duplicate, with no extra set."

**Seen in:** `main.py`, `find_duplicate_marking`

## for...else  `(syntax)`
**1. Definition:** A `for` loop's `else` block runs only if the loop finished without hitting `break`.
**2. In plain words:** In `find_duplicate_marking`, `break` means "found the duplicate"; if the loop completes with no break, the `else` sets the result to None. It is rare, so be ready to explain it.
**3. Example to quote:**
> "The `else` on my `for` loop runs only if I never `break`, meaning no duplicate was found."

**Seen in:** `main.py`, `find_duplicate_marking`

## Return value convention (None / -1)  `(concept)`
**1. Definition:** A return convention is an agreed value signalling "not found" or "no answer", such as `None` for an empty input or `-1` for a missing index.
**2. In plain words:** Decide and document what the function returns for edge cases. This file returns `None` for empty `find_max` and `-1` from searches, mirroring Python's `str.find`.
**3. Example to quote:**
> "My `linear_search` returns -1 when the target is missing, the same convention as `str.find`."

**Seen in:** `main.py`, docstrings
