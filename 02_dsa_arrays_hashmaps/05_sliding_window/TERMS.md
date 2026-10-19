# Terms — Sliding window

Folder: `02_dsa_arrays_hashmaps/05_sliding_window` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [DSA README](../README.md)

**Terms in this folder:** [Sliding window](#sliding-window--concept) · [Fixed-size window](#fixed-size-window--concept) · [Variable-size window](#variable-size-window--concept) · [Slicing a window (`nums[i:i + k]`)](#slicing-a-window-numsii--k--syntax) · [max() over a generator](#max-over-a-generator--python-built-in) · [Augmented assignment with an expression (`window += a - b`)](#augmented-assignment-with-an-expression-window--a---b--syntax) · [float("inf")](#floatinf--python-built-in) · [Last-seen index map](#last-seen-index-map--concept) · [enumerate() over a string](#enumerate-over-a-string--python-built-in) · [Substring](#substring--concept) · [Shrink loop (`while` inside `for`)](#shrink-loop-while-inside-for--syntax) · [Minimum Size Subarray Sum](#minimum-size-subarray-sum--concept) · [Early exit with `break`](#early-exit-with-break--syntax) · [set (seen characters)](#set-seen-characters--python-built-in) · [sum(), min(), len()](#sum-min-len--python-built-in) · [Window validity guard (`k <= 0 or k > len(nums)`)](#window-validity-guard-k--0-or-k--lennums--concept)

---

## Sliding window  `(concept)`
**1. Definition:** The sliding-window pattern keeps a contiguous range `[left, right]` of an array or string and updates its answer incrementally as the range moves, instead of recomputing from scratch.
**2. In plain words:** Imagine a frame sliding along a row of numbers. When it moves one step you add the new item entering on the right and remove the item leaving on the left, rather than re-adding everything inside. Each pointer only moves forward, so the total work is O(n). It fits "contiguous subarray/substring" questions.
**3. Example to quote:**
> "For the max sum of 3 consecutive numbers in `[2, 1, 5, 1, 3, 2]`, I slide: add the new number, subtract the one that left. That is O(n) instead of re-adding 3 numbers at each position."

**Seen in:** `main.py`, all functions

## Fixed-size window  `(concept)`
**1. Definition:** A fixed-size window always contains exactly k elements and slides one position at a time.
**2. In plain words:** First sum the first k items, then for every next position do `window += nums[right] - nums[right - k]`: one in, one out. Brute force re-sums k items every time, O(n·k).
**3. Example to quote:**
> "The first window `[2, 1, 5]` sums to 8. Sliding adds 1 and drops 2 to get 7, then 9, then 6. The best is 9."

**Seen in:** `main.py`, `max_sum_k`

## Variable-size window  `(concept)`
**1. Definition:** A variable-size window grows by moving the right pointer and shrinks by moving the left pointer whenever a rule is broken (or a goal is reached).
**2. In plain words:** For "longest substring without repeats" you keep extending right; when a repeat appears you move left past it. For "smallest subarray with sum at least target" you shrink while the window is still valid. The left pointer never moves backwards.
**3. Example to quote:**
> "I grow the window with `right`, and whenever the condition breaks I advance `left` until it holds again."

**Seen in:** `main.py`, `longest_unique`, `min_subarray_len`

## Slicing a window (`nums[i:i + k]`)  `(syntax)`
**1. Definition:** `nums[i:i + k]` returns a new list with k items starting at index i.
**2. In plain words:** Used by the brute-force version to re-create each window. The slice copies k items and `sum` adds them, so every window costs O(k).
**3. Example to quote:**
> "`[1, 2, 3, 4][1:3]` is `[2, 3]`, which is the window starting at index 1 of size 2."

**Seen in:** `main.py`, `max_sum_k_brute`, `max_sum_k`

## max() over a generator  `(Python built-in)`
**1. Definition:** `max(iterable)` returns the largest item; passing a generator expression makes it compute values lazily.
**2. In plain words:** `max(sum(nums[i:i + k]) for i in range(...))` computes each window sum and keeps the biggest, without storing all sums.
**3. Example to quote:**
> "`max(x * x for x in [1, 2])` is 4, with no intermediate list."

**Seen in:** `main.py`, `max_sum_k_brute`

## Augmented assignment with an expression (`window += a - b`)  `(syntax)`
**1. Definition:** `x += expr` evaluates `expr` and adds it to `x` in place.
**2. In plain words:** `window += nums[right] - nums[right - k]` adds the new element and subtracts the old one in a single update, the heart of the fixed window.
**3. Example to quote:**
> "One line, `window += nums[right] - nums[right - k]`, is the entire slide: one in, one out."

**Seen in:** `main.py`, `max_sum_k`

## float("inf")  `(Python built-in)`
**1. Definition:** `float("inf")` is a floating-point value greater than every finite number.
**2. In plain words:** It is the starting value for "smallest length so far", so the first real length replaces it; if it is still infinity at the end, no valid window exists and the function returns 0.
**3. Example to quote:**
> "`best = float('inf')` means nothing found yet; at the end I return 0 if it is still infinity."

**Seen in:** `main.py`, `min_subarray_len`

## Last-seen index map  `(concept)`
**1. Definition:** A last-seen map is a dict from each item to the most recent index where it appeared.
**2. In plain words:** For "longest substring without repeating characters", the map tells `left` exactly where to jump when a repeat appears inside the window, instead of shrinking one step at a time. The check `last_seen[ch] >= left` ignores repeats that are already outside the window.
**3. Example to quote:**
> "For 'abba', when the second 'a' arrives its old index is 0, which is before `left`=2, so I must not move left backwards. That is what the `>= left` check protects."

**Seen in:** `main.py`, `longest_unique`

## enumerate() over a string  `(Python built-in)`
**1. Definition:** `enumerate(s)` on a string yields `(index, character)` pairs, because strings are iterable by character.
**2. In plain words:** Gives me `right` (the position) and `ch` (the letter) in one loop.
**3. Example to quote:**
> "`for right, ch in enumerate('abc')` gives `(0, 'a')`, `(1, 'b')`, `(2, 'c')`."

**Seen in:** `main.py`, `longest_unique`

## Substring  `(concept)`
**1. Definition:** A substring is a contiguous run of characters within a string.
**2. In plain words:** 'abc' is a substring of 'abcabcbb'; 'ac' is not (that would be a subsequence). The longest unique substring of 'abcabcbb' is 'abc', length 3.
**3. Example to quote:**
> "In 'pwwkew' the longest substring with no repeats is 'wke', length 3."

**Seen in:** `main.py`, `longest_unique`

## Shrink loop (`while` inside `for`)  `(syntax)`
**1. Definition:** A `while` loop nested in the main `for` loop that advances `left` as long as the window is valid (or invalid).
**2. In plain words:** It looks like nested loops but each element is added once and removed at most once, so the total is still O(n), not O(n²).
**3. Example to quote:**
> "The `while total >= target` shrink loop moves `left` forward; since `left` only moves forward the whole function is O(n)."

**Seen in:** `main.py`, `min_subarray_len`

## Minimum Size Subarray Sum  `(concept)`
**1. Definition:** A problem asking for the length of the shortest contiguous subarray whose sum is at least a target, solved with a variable window for positive numbers.
**2. In plain words:** Grow until the sum reaches the target, then shrink from the left while it still does, recording the shortest length. It relies on all numbers being non-negative: with negatives, shrinking can make the sum larger, breaking the logic.
**3. Example to quote:**
> "For target 7 and `[2, 3, 1, 2, 4, 3]` the shortest subarray is `[4, 3]`, length 2."

**Seen in:** `main.py`, `min_subarray_len`

## Early exit with `break`  `(syntax)`
**1. Definition:** `break` stops the innermost loop immediately.
**2. In plain words:** In the brute-force searches, once a repeat or a valid sum is found for a start index, there is no need to extend further from that start.
**3. Example to quote:**
> "In the brute-force unique-substring check I `break` as soon as a character repeats."

**Seen in:** `main.py`, `longest_unique_brute`, `min_subarray_len_brute`

## set (seen characters)  `(Python built-in)`
**1. Definition:** A `set` stores unique items with O(1) average membership tests.
**2. In plain words:** The brute-force version uses a set to detect a repeated character in the current substring.
**3. Example to quote:**
> "`if s[j] in seen: break` tells me instantly whether this letter is already in my substring."

**Seen in:** `main.py`, `longest_unique_brute`

## sum(), min(), len()  `(Python built-in)`
**1. Definition:** `sum` totals an iterable, `min` returns the smallest value, `len` returns the item count.
**2. In plain words:** The basic aggregates used to compute window sizes (`right - left + 1`) and sums.
**3. Example to quote:**
> "`right - left + 1` is the window length, because both ends are inclusive."

**Seen in:** `main.py`, throughout

## Window validity guard (`k <= 0 or k > len(nums)`)  `(concept)`
**1. Definition:** An input guard is a check at the top of a function that rejects impossible inputs before the main logic runs.
**2. In plain words:** A window of size 0 or larger than the list does not exist, so the function returns `None` instead of crashing.
**3. Example to quote:**
> "`max_sum_k([1], 2)` returns None because a window of 2 cannot fit in one element."

**Seen in:** `main.py`, `max_sum_k`
