# Terms — Two pointers

Folder: `02_dsa_arrays_hashmaps/04_two_pointers` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [DSA README](../README.md)

**Terms in this folder:** [Two-pointer technique](#two-pointer-technique--concept) · [In-place modification](#in-place-modification--concept) · [Tuple swap](#tuple-swap--syntax) · [while left < right](#while-left--right--syntax) · [Compound assignment (`+=`, `-=`)](#compound-assignment-----syntax) · [Palindrome](#palindrome--concept) · [str.isalnum()](#strisalnum--python-built-in) · [str.lower()](#strlower--python-built-in) · [Sorted input requirement](#sorted-input-requirement--concept) · [sorted()](#sorted--python-built-in) · [Container With Most Water](#container-with-most-water--concept) · [min() and max()](#min-and-max--python-built-in) · [3Sum](#3sum--concept) · [Skipping duplicates](#skipping-duplicates--concept) · [continue](#continue--syntax) · [List append](#list-append--python-built-in) · [Nested while loops (inner skip loops)](#nested-while-loops-inner-skip-loops--syntax)

---

## Two-pointer technique  `(concept)`
**1. Definition:** The two-pointer technique uses two indexes that move through an array (often from opposite ends toward each other) so a problem that needs a nested loop can be solved in a single O(n) pass.
**2. In plain words:** Instead of trying every pair, you place a finger at each end and decide, from the values you see, which finger to move. Each move permanently rules out some candidates, which is why it is linear. It usually needs structure such as a sorted array.
**3. Example to quote:**
> "On a sorted array I put one pointer at each end; if the sum is too small I move the left one up, otherwise the right one down. That is O(n) instead of O(n²)."

**Seen in:** `main.py`, all functions

## In-place modification  `(concept)`
**1. Definition:** An in-place algorithm changes the input data structure directly using O(1) extra space instead of building a new one.
**2. In plain words:** `reverse_in_place` swaps items inside the original list and returns the same list. Saves memory but destroys the original order, so the caller must expect it.
**3. Example to quote:**
> "`reverse_in_place([1, 2, 3, 4])` gives `[4, 3, 2, 1]` by swapping inside the list; it uses O(1) extra space."

**Seen in:** `main.py`, `reverse_in_place`

## Tuple swap  `(syntax)`
**1. Definition:** `nums[left], nums[right] = nums[right], nums[left]` swaps two values by evaluating the right side as a tuple first and then assigning.
**2. In plain words:** No temporary variable is needed, and nothing is overwritten before it has been read.
**3. Example to quote:**
> "To reverse in place I swap the two ends with a tuple assignment and move both pointers inward."

**Seen in:** `main.py`, `reverse_in_place`

## while left < right  `(syntax)`
**1. Definition:** The loop condition `left < right` keeps looping while the two pointers have not met or crossed.
**2. In plain words:** It is the stopping rule of two-pointer problems. Using `<` rather than `<=` prevents pairing an element with itself, which matters in Two Sum on a sorted array: with `<=`, `[3]` and target 6 would wrongly return `[0, 0]`.
**3. Example to quote:**
> "I use `while left < right` so the pointers stop when they meet, and an element can never be paired with itself."

**Seen in:** `main.py`, `two_sum_sorted`

## Compound assignment (`+=`, `-=`)  `(syntax)`
**1. Definition:** `x += 1` is shorthand for `x = x + 1`; `-=` subtracts.
**2. In plain words:** The standard way to move a pointer one step: `left += 1` moves right, `right -= 1` moves left.
**3. Example to quote:**
> "`left += 1` advances the left pointer one position."

**Seen in:** `main.py`, every function

## Palindrome  `(concept)`
**1. Definition:** A palindrome is a sequence that reads the same forwards and backwards.
**2. In plain words:** Compare the first and last characters, then move inward. The version here ignores punctuation, spaces and case, so "A man, a plan, a canal: Panama" counts.
**3. Example to quote:**
> "'race a car' is not a palindrome: after ignoring spaces, 'raceacar' reversed is 'racaecar'."

**Seen in:** `main.py`, `is_palindrome`

## str.isalnum()  `(Python built-in)`
**1. Definition:** `str.isalnum()` returns True if the string is non-empty and every character is a letter or digit.
**2. In plain words:** Used to skip spaces and punctuation while scanning. On one character it answers "is this a real letter or digit?".
**3. Example to quote:**
> "`','.isalnum()` is False, so my left pointer skips the comma."

**Seen in:** `main.py`, `is_palindrome`

## str.lower()  `(Python built-in)`
**1. Definition:** `str.lower()` returns a copy of the string with all letters in lowercase.
**2. In plain words:** Makes the comparison case-insensitive so 'A' and 'a' match. The original string is unchanged (strings are immutable).
**3. Example to quote:**
> "I compare `s[left].lower()` with `s[right].lower()` so capitalisation does not matter."

**Seen in:** `main.py`, `is_palindrome`

## Sorted input requirement  `(concept)`
**1. Definition:** Some algorithms, such as the two-pointer Two Sum and binary search, are only correct when the input is already sorted.
**2. In plain words:** The pointer rule "sum too small, move left up" relies on the left side holding the smaller values. On unsorted data it would give wrong answers. If you must sort first, add O(n log n) to the cost.
**3. Example to quote:**
> "Two-pointer Two Sum is O(n) and O(1) space, but only on sorted data. For unsorted data I would use the dictionary version."

**Seen in:** `main.py`, `two_sum_sorted`

## sorted()  `(Python built-in)`
**1. Definition:** `sorted(iterable)` returns a new sorted list without modifying the original; it uses Timsort, O(n log n).
**2. In plain words:** `three_sum` sorts a copy first so it can use two pointers and skip duplicates easily.
**3. Example to quote:**
> "`nums = sorted(nums)` gives me a sorted copy, so I can sweep with two pointers and skip repeated values."

**Seen in:** `main.py`, `three_sum`

## Container With Most Water  `(concept)`
**1. Definition:** A problem that picks two vertical lines from a list of heights to maximise `min(height_i, height_j) * (j - i)`, solved by moving the shorter wall inward.
**2. In plain words:** Width shrinks as pointers move, so the only way to do better is to find a taller wall; moving the taller wall can never help because the shorter wall still limits the height. This gives O(n).
**3. Example to quote:**
> "For `[1, 8, 6, 2, 5, 4, 8, 3, 7]` the best area is 49: heights 8 and 7 are 7 apart."

**Seen in:** `main.py`, `max_area`, `max_area_brute`

## min() and max()  `(Python built-in)`
**1. Definition:** `min` and `max` return the smaller or larger of their arguments.
**2. In plain words:** Area uses `min` of the two heights (water level is the shorter wall) and `max` to keep the best area so far.
**3. Example to quote:**
> "`min(heights[left], heights[right]) * (right - left)` is the area, because the shorter wall sets the water level."

**Seen in:** `main.py`, `max_area`

## 3Sum  `(concept)`
**1. Definition:** 3Sum asks for all unique triplets in an array that sum to zero; the standard solution sorts, fixes one number and uses two pointers for the other two, O(n²).
**2. In plain words:** Reduce a three-number problem to many two-number problems. Sorting lets you skip duplicates so you never report the same triplet twice.
**3. Example to quote:**
> "For `[-1, 0, 1, 2, -1, -4]` the unique triplets are `[-1, -1, 2]` and `[-1, 0, 1]`."

**Seen in:** `main.py`, `three_sum`

## Skipping duplicates  `(concept)`
**1. Definition:** Skipping duplicates means advancing past repeated values so that each distinct combination is processed or reported once.
**2. In plain words:** After sorting, equal numbers are adjacent. The code skips an anchor equal to the previous one and skips repeated `left` values after a match. Forgetting this returns duplicate triplets.
**3. Example to quote:**
> "In `three_sum([0, 0, 0, 0])` the result is just `[[0, 0, 0]]` because I skip repeated values."

**Seen in:** `main.py`, `three_sum`

## continue  `(syntax)`
**1. Definition:** `continue` ends the current loop iteration immediately and jumps to the next one.
**2. In plain words:** Used to skip a duplicate anchor number without nesting the rest of the loop body in an `if`.
**3. Example to quote:**
> "If this anchor equals the previous one I `continue`, so I do not repeat the same triplets."

**Seen in:** `main.py`, `three_sum`

## List append  `(Python built-in)`
**1. Definition:** `list.append(item)` adds one item to the end of a list in amortised O(1) time, modifying the list in place.
**2. In plain words:** `result.append([a, b, c])` collects each valid triplet. It returns None, so do not write `result = result.append(...)`.
**3. Example to quote:**
> "Each time the sum hits zero I `append` the triplet to `result`."

**Seen in:** `main.py`, `three_sum`

## Nested while loops (inner skip loops)  `(syntax)`
**1. Definition:** A while loop placed inside another loop to advance a pointer over a run of unwanted items.
**2. In plain words:** In `is_palindrome`, inner loops move `left` and `right` past punctuation before the main comparison. The extra `left < right` guard prevents running off the ends.
**3. Example to quote:**
> "The inner `while left < right and not s[left].isalnum()` skips spaces and commas without crossing the other pointer."

**Seen in:** `main.py`, `is_palindrome`
