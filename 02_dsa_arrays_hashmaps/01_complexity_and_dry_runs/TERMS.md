# Terms — Complexity and dry runs

Folder: `02_dsa_arrays_hashmaps/01_complexity_and_dry_runs` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [DSA README](../README.md)

**Terms in this folder:** [Time complexity](#time-complexity--concept) · [Big-O notation](#big-o-notation--concept) · [O(1) constant time](#o1-constant-time--concept) · [O(n) linear time](#on-linear-time--concept) · [O(n²) quadratic time](#on²-quadratic-time--concept) · [Hash lookup (set / dict)](#hash-lookup-set--dict--concept) · [Worst case](#worst-case--concept) · [time.perf_counter()](#timeperf_counter--standard-library-time) · [Function as an argument](#function-as-an-argument--syntax) · [Generator expression](#generator-expression--syntax) · [sum()](#sum--python-built-in) · [range()](#range--python-built-in) · [enumerate()](#enumerate--python-built-in) · [f-string format spec](#f-string-format-spec--syntax) · [Numeric underscore separator](#numeric-underscore-separator--syntax) · [Throwaway variable `_`](#throwaway-variable-_--syntax) · [assert](#assert--syntax) · [Dry run](#dry-run--concept) · [Edge case](#edge-case--concept) · [Space complexity](#space-complexity--concept)

---

## Time complexity  `(concept)`
**1. Definition:** Time complexity describes how the number of basic steps an algorithm performs grows as the input size `n` grows, ignoring machine speed and constant factors.
**2. In plain words:** Seconds depend on your laptop, but "how much more work when the input doubles" does not. If doubling the input doubles the work, that is linear; if it quadruples the work, that is quadratic. Interviewers care about this because a solution that works on 10 items may be unusably slow on a million. This folder counts steps (45, 4950, 499500) so you can see the growth without a stopwatch.
**3. Example to quote:**
> "My brute-force Two Sum compares every pair, so for 1,000 numbers it does about 500,000 comparisons. The dictionary version does 1,000 lookups. The complexity tells me how that gap grows as the input grows."

**Seen in:** `main.py`, Part 2 (`count_pairs_brute` vs `count_pass_once`)

## Big-O notation  `(concept)`
**1. Definition:** Big-O notation is a way to write the upper bound of an algorithm's growth rate using the dominant term only, such as O(n) or O(n²), dropping constants and smaller terms.
**2. In plain words:** It is shorthand for "roughly how bad does it get". n(n-1)/2 steps is written O(n²) because for large n the n² part dominates. It usually describes the worst case, so a list search that happens to find the item first is still called O(n). The limitation: two O(n) solutions can differ a lot in real speed, so Big-O is a guide, not a stopwatch.
**3. Example to quote:**
> "When I say my solution is O(n), I mean the work grows in proportion to the input. 10 items need about 10 steps, 1,000 items about 1,000."

```python
steps = n * (n - 1) // 2   # 45 for n=10  ->  O(n^2)
```
**Seen in:** `main.py` docstrings, `WALKTHROUGH.md` "Say it aloud"

## O(1) constant time  `(concept)`
**1. Definition:** An operation is O(1) if its cost does not depend on the input size.
**2. In plain words:** Looking something up by key in a set or dict takes about the same time whether it holds 10 or 10 million items, because hashing jumps straight to the slot. "Average" matters here: rare hash collisions can make a single lookup slower, but on average it stays flat. This is why swapping a list for a set is such a common optimisation.
**3. Example to quote:**
> "In the demo, 2,000 lookups in a 20,000-item set took 0.0001 s, versus 0.21 s in a list, because a set lookup is O(1) on average."

```python
numbers_set = set(range(20_000))
19_999 in numbers_set   # one hash, no scan
```
**Seen in:** `main.py`, `set_lookups`

## O(n) linear time  `(concept)`
**1. Definition:** An algorithm is O(n) if its number of steps grows proportionally with the input size n.
**2. In plain words:** One loop over the data is the typical O(n) shape. Checking `x in some_list` is also O(n), because Python may have to compare against every item. Doubling the data doubles the time. It is usually the target for array problems because you cannot answer most questions without looking at each element at least once.
**3. Example to quote:**
> "`count_pass_once` has one loop, so for n=1000 it does exactly 1000 steps. That is O(n)."

```python
for _ in nums:
    steps += 1
```
**Seen in:** `main.py`, `count_pass_once`, `list_lookups`

## O(n²) quadratic time  `(concept)`
**1. Definition:** An algorithm is O(n²) if its steps grow with the square of the input size, typically from a loop nested inside another loop over the same data.
**2. In plain words:** Comparing every pair of items is quadratic: 10 items give 45 comparisons, 100 give 4,950, 1,000 give 499,500. Ten times more data means a hundred times more work. It is fine for small inputs but is the classic "brute force" that interviewers expect you to improve.
**3. Example to quote:**
> "Brute-force pair checking printed 499,500 steps for 1,000 items. That is n(n-1)/2, so it is O(n²), and I would replace it with a dictionary to get O(n)."

```python
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        steps += 1
```
**Seen in:** `main.py`, `count_pairs_brute`

## Hash lookup (set / dict)  `(concept)`
**1. Definition:** A hash lookup computes a hash value from the key and uses it to jump directly to the storage slot, giving O(1) average-time membership tests and key access in `set` and `dict`.
**2. In plain words:** A list is like a stack of unlabelled papers you must flip through; a set or dict is like a filing cabinet where the label tells you exactly which drawer to open. The price is extra memory, and set/dict keys must be hashable (so lists cannot be keys).
**3. Example to quote:**
> "I build a set of the numbers once, then each `q in numbers_set` check is a direct hash jump instead of scanning 20,000 items."

```python
numbers_set = set(numbers_list)
```
**Seen in:** `main.py`, `numbers_set = set(numbers_list)`

## Worst case  `(concept)`
**1. Definition:** The worst case is the input arrangement that makes an algorithm do the most work for a given size n; Big-O usually quotes this bound.
**2. In plain words:** Searching a list is fast if the item is first and slow if it is last. The demo deliberately queries numbers near the end of the list to show the slow case. Quoting the worst case gives a guarantee: it will never be slower than this.
**3. Example to quote:**
> "I tested numbers near the end of the list on purpose, because that forces the scan to walk almost all 20,000 items, which is the worst case."

**Seen in:** `main.py`, `queries = range(N - 2000, N)`

## time.perf_counter()  `(standard library: time)`
**1. Definition:** `time.perf_counter()` returns the value, in fractional seconds, of a high-resolution monotonic clock; subtracting two readings gives elapsed time.
**2. In plain words:** It is a stopwatch. You read it before and after the code and subtract. Only the difference is meaningful, not the absolute number. It is better than `time.time()` for timing because it is not affected by system clock changes.
**3. Example to quote:**
> "I take `start = time.perf_counter()`, run the function, and subtract to get the elapsed time, 0.21 s for the list and 0.0001 s for the set."

```python
start = time.perf_counter()
func()
elapsed = time.perf_counter() - start
```
**Seen in:** `main.py`, `time_it`

## Function as an argument  `(syntax)`
**1. Definition:** In Python, functions are objects, so a function can be passed to another function by name (without parentheses) and called inside it.
**2. In plain words:** `time_it("label", list_lookups)` hands over the function itself rather than its result. Writing `list_lookups()` with parentheses would run it immediately and pass its return value instead. This lets one timing helper measure any function.
**3. Example to quote:**
> "I pass `list_lookups` without parentheses so `time_it` can start the timer, call it, and stop the timer."

```python
def time_it(label, func):
    func()
```
**Seen in:** `main.py`, `time_it(label, func)`

## Generator expression  `(syntax)`
**1. Definition:** A generator expression such as `(expr for x in iterable if cond)` produces items lazily, one at a time, without building a whole list in memory.
**2. In plain words:** It looks like a list comprehension without square brackets. Here `sum(1 for q in queries if q in numbers_list)` counts how many queries match by adding 1 for each one, without storing a list of matches. It saves memory when you only need to consume the values once.
**3. Example to quote:**
> "`sum(1 for x in [1, 2, 3] if x > 1)` gives 2, because it adds 1 for each item that passes the condition, with no intermediate list."

**Seen in:** `main.py`, `list_lookups`, `set_lookups`

## sum()  `(Python built-in)`
**1. Definition:** `sum(iterable, start=0)` returns the total of the items in an iterable plus an optional start value.
**2. In plain words:** It adds everything up. Combined with a generator of 1s it becomes a counter. It works on numbers; it will raise a `TypeError` on strings.
**3. Example to quote:**
> "`sum([3, 1, 4])` is 8, the same total my traced loop reaches step by step."

**Seen in:** `main.py`, lookup counters

## range()  `(Python built-in)`
**1. Definition:** `range(start, stop, step)` returns an immutable lazy sequence of integers from start up to but excluding stop.
**2. In plain words:** It generates numbers on demand instead of storing them, so `range(1_000_000)` uses almost no memory. The stop value is never included, which is why `range(len(nums))` covers indexes 0 to n-1. `range(i + 1, n)` starts after i, which is how the code avoids pairing an element with itself.
**3. Example to quote:**
> "`range(i + 1, len(nums))` starts one past i, so each pair is visited once and no element pairs with itself."

**Seen in:** `main.py`, `queries`, `count_pairs_brute`

## enumerate()  `(Python built-in)`
**1. Definition:** `enumerate(iterable, start=0)` yields `(index, item)` pairs while iterating.
**2. In plain words:** When you need both the position and the value, `enumerate` is cleaner than looping over `range(len(nums))` and indexing. In the dry-run helper it gives each step number alongside the value being added.
**3. Example to quote:**
> "`list(enumerate("ab"))` is `[(0, 'a'), (1, 'b')]`, so I get the index and the value together without manual indexing."

```python
for index, value in enumerate(nums):
    total += value
```
**Seen in:** `main.py`, `sum_with_trace`

## f-string format spec  `(syntax)`
**1. Definition:** Inside an f-string, `{value:spec}` formats the value using a format specification such as `.4f` (4 decimal places) or `<28` (left-align in 28 characters).
**2. In plain words:** It lets you control how a value prints. `{elapsed:.4f}` shows 0.2102 instead of 0.21023456789, and `{label:<28}` pads the label so the columns line up. It changes only the displayed text, not the number.
**3. Example to quote:**
> "`f"{3.14159:.2f}"` gives `3.14`, and I use `:<28` to align the labels in the timing table."

**Seen in:** `main.py`, `time_it`, loop at the end of Part 2

## Numeric underscore separator  `(syntax)`
**1. Definition:** Python allows underscores inside numeric literals, such as `20_000`, purely as visual separators; the value is unchanged.
**2. In plain words:** It makes big numbers readable, like writing 20,000 with a comma. `1_000 == 1000` is True.
**3. Example to quote:**
> "I wrote `N = 20_000` so it is easy to see it is twenty thousand, and Python reads it as 20000."

**Seen in:** `main.py`, `N = 20_000`

## Throwaway variable `_`  `(syntax)`
**1. Definition:** By convention, the name `_` is used for a variable whose value is intentionally not used.
**2. In plain words:** `for _ in nums` says "repeat once for each item, but I do not need the item". It signals intent to readers. Python does not treat `_` specially in this case; it is just a normal variable name.
**3. Example to quote:**
> "In `count_pass_once` I only care how many times the loop runs, so I use `_` for the unused item."

**Seen in:** `main.py`, `count_pass_once`

## assert  `(syntax)`
**1. Definition:** `assert condition` raises an `AssertionError` if the condition is False and does nothing if it is True.
**2. In plain words:** It is a one-line test: "I claim this is true; crash loudly if not". The file uses it to check the dry-run result is 8 and that an empty list sums to 0. Asserts can be disabled with `python -O`, so use them for tests, not for validating user input.
**3. Example to quote:**
> "`assert sum_with_trace([]) == 0` checks the empty-input edge case; if it were wrong the script would stop with an AssertionError."

**Seen in:** `main.py`, Part 3

## Dry run  `(concept)`
**1. Definition:** A dry run is manually tracing an algorithm on a small example, writing down the value of each variable after each step.
**2. In plain words:** Before you code, or when your code is wrong, you act as the computer on paper: step 0 value=3 total=3, step 1 value=1 total=4, and so on. It catches off-by-one and logic errors that never raise a traceback. Interviewers like to see it because it shows you can reason about code without running it.
**3. Example to quote:**
> "Let me dry-run `[3, 1, 4]`: after the first item the total is 3, then 4, then 8, and 8 is what I expect."

**Seen in:** `main.py`, `sum_with_trace`

## Edge case  `(concept)`
**1. Definition:** An edge case is an unusual or boundary input, such as an empty list, a single element, duplicates, negatives, or all-equal values, that can break an otherwise correct-looking solution.
**2. In plain words:** Most bugs hide at the edges, not in the typical input. The file lists them in a comment: empty, one element, duplicates, negatives, all same. Saying them out loud before coding also shows interviewers you think carefully.
**3. Example to quote:**
> "Before I submit, I try an empty list, one element, duplicates and negatives. `sum_with_trace([])` returns 0, so the empty case passes."

**Seen in:** `main.py`, final comment and asserts

## Space complexity  `(concept)`
**1. Definition:** Space complexity describes how much extra memory an algorithm needs as input size n grows.
**2. In plain words:** Making a set to speed up lookups uses O(n) extra memory; looping with a counter uses O(1). Most optimisations trade memory for time, which is why the walkthrough ends with "the trade-off is extra memory for the set".
**3. Example to quote:**
> "Replacing the nested loop with a set made it O(n) time, but the set itself costs O(n) extra memory, so I traded space for speed."

**Seen in:** `WALKTHROUGH.md`, "Say it aloud"
