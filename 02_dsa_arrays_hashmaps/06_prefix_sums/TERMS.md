# Terms — Prefix sums

Folder: `02_dsa_arrays_hashmaps/06_prefix_sums` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [DSA README](../README.md)

**Terms in this folder:** [Prefix sum](#prefix-sum--concept) · [Range-sum query](#range-sum-query--concept) · [list.append and negative indexing (`prefix[-1]`)](#listappend-and-negative-indexing-prefix-1--python-built-in) · [Class and `__init__`](#class-and-__init__--syntax) · [self and instance attribute](#self-and-instance-attribute--syntax) · [Inclusive vs exclusive bounds](#inclusive-vs-exclusive-bounds--concept) · [Subarray](#subarray--concept) · [Prefix sum + hash map (Subarray Sum Equals K)](#prefix-sum--hash-map-subarray-sum-equals-k--concept) · [Seeding the dict with `{0: 1}`](#seeding-the-dict-with-0-1--concept) · [dict.get(key, 0)](#dictgetkey-0--python-built-in) · [Product of Array Except Self](#product-of-array-except-self--concept) · [List repetition (`[1] * n`)](#list-repetition-1--n--syntax) · [Counting down with range](#counting-down-with-range--python-built-in) · [Running total variable](#running-total-variable--concept) · [IndexError](#indexerror--concept)

---

## Prefix sum  `(concept)`
**1. Definition:** A prefix-sum array stores in position i the sum of all elements before index i, so `prefix[i] = nums[0] + ... + nums[i-1]` and `prefix[0] = 0`.
**2. In plain words:** It is a running total written down once. After that, the sum of any range is just a subtraction of two stored totals instead of a fresh loop. Building it costs O(n), and each later query is O(1). The extra leading 0 avoids special cases at the start.
**3. Example to quote:**
> "For `[3, 1, 4, 1, 5]` the prefix array is `[0, 3, 4, 8, 9, 14]`. The sum of indices 1 to 3 is `prefix[4] - prefix[1] = 9 - 3 = 6`."

**Seen in:** `main.py`, `build_prefix`

## Range-sum query  `(concept)`
**1. Definition:** A range-sum query asks for the sum of elements between two indices, inclusive.
**2. In plain words:** If you must answer many such queries, re-adding each time is slow. With prefix sums you pay once and answer each in constant time, trading O(n) memory for speed.
**3. Example to quote:**
> "`rs.query(1, 3)` returns 6 in O(1) by computing `prefix[right + 1] - prefix[left]`."

**Seen in:** `main.py`, `RangeSum.query`

## list.append and negative indexing (`prefix[-1]`)  `(Python built-in)`
**1. Definition:** `append` adds an item to the end of a list; index `-1` refers to the last item.
**2. In plain words:** Each new prefix value is the previous total plus the new number: `prefix.append(prefix[-1] + x)`. `-1` saves writing `len(prefix) - 1`.
**3. Example to quote:**
> "`[1, 2, 3][-1]` is 3, so `prefix[-1]` is always my latest running total."

**Seen in:** `main.py`, `build_prefix`

## Class and `__init__`  `(syntax)`
**1. Definition:** A `class` bundles data and functions; `__init__` is the constructor method that runs automatically when you create an instance, e.g. `RangeSum([1, 2])`.
**2. In plain words:** `RangeSum` builds the prefix array once when created and stores it, so `query` can reuse it every time. The class is a neat way to say "preprocess once, answer many".
**3. Example to quote:**
> "I create `rs = RangeSum(nums)` once, and it precomputes the prefix. Then every `rs.query(l, r)` is O(1)."

**Seen in:** `main.py`, `RangeSum`

## self and instance attribute  `(syntax)`
**1. Definition:** `self` is the first parameter of an instance method and refers to the specific object; `self.prefix` is an attribute stored on that object.
**2. In plain words:** It is how a method reaches the data that `__init__` saved. Forgetting `self.` makes Python look for a local variable and raise `NameError`.
**3. Example to quote:**
> "`self.prefix = build_prefix(nums)` stores the array on the object so `query` can read it later."

**Seen in:** `main.py`, `RangeSum`

## Inclusive vs exclusive bounds  `(concept)`
**1. Definition:** An inclusive bound includes its endpoint; an exclusive bound does not. Python slices and `range` are start-inclusive, stop-exclusive.
**2. In plain words:** The query is inclusive on both ends, but the prefix array is one longer than the data, so the formula uses `right + 1`. Mixing these up causes off-by-one errors, and `query(0, 3)` on a 3-item list raises `IndexError`.
**3. Example to quote:**
> "For `[1, 2, 3]`, `query(0, 3)` fails with IndexError because index 3 does not exist; the last valid right index is 2."

**Seen in:** `main.py`, `RangeSum.query`

## Subarray  `(concept)`
**1. Definition:** A subarray is a contiguous, non-empty slice of an array.
**2. In plain words:** `[1, 2]` is a subarray of `[1, 2, 3]`; `[1, 3]` is not (it skips an element). Problems that say "contiguous" are usually prefix-sum or sliding-window problems.
**3. Example to quote:**
> "`[1, 1, 1]` has two subarrays that sum to 2: indices 0-1 and indices 1-2."

**Seen in:** `main.py`, `subarray_sum`

## Prefix sum + hash map (Subarray Sum Equals K)  `(concept)`
**1. Definition:** If `prefix[j] - prefix[i] == k`, a subarray ending at j sums to k. So for each running total you count how many earlier prefixes equal `total - k`, using a dict of counts.
**2. In plain words:** It is complement lookup (as in Two Sum) applied to running totals instead of values, and it works even with negative numbers where a sliding window cannot. It runs in O(n) time and O(n) space.
**3. Example to quote:**
> "For `[1, 1, 1]` and k=2, at total 2 I look up 0 (seen once) and at total 3 I look up 1 (seen once). The count is 2."

**Seen in:** `main.py`, `subarray_sum`

## Seeding the dict with `{0: 1}`  `(concept)`
**1. Definition:** Initialising the count dict with `{0: 1}` records that the empty prefix (total 0) has occurred once.
**2. In plain words:** Without it, subarrays that start at index 0 are missed, because their matching "earlier prefix" is the empty one. Removing it makes `subarray_sum([1, 2, 3], 3)` lose the subarray `[1, 2]`.
**3. Example to quote:**
> "The `{0: 1}` seed represents 'nothing added yet'. Without it I would not count a subarray that starts at the very beginning."

**Seen in:** `main.py`, `subarray_sum`

## dict.get(key, 0)  `(Python built-in)`
**1. Definition:** Returns the value for the key, or the default (here 0) if the key is missing, without raising `KeyError`.
**2. In plain words:** Used twice: to count how many earlier prefixes matched and to bump the count for the current total.
**3. Example to quote:**
> "`seen.get(total - k, 0)` gives 0 if that prefix was never seen, so I can add it to the count safely."

**Seen in:** `main.py`, `subarray_sum`

## Product of Array Except Self  `(concept)`
**1. Definition:** A problem that returns an array where each element is the product of all other elements, without division, using one pass for products to the left and one for products to the right.
**2. In plain words:** It is a "prefix product" followed by a "suffix product". Pass one stores in `result[i]` the product of everything left of i; pass two multiplies in everything right of i. It runs in O(n) and handles zeros correctly, which division cannot.
**3. Example to quote:**
> "For `[1, 2, 3, 4]` the answer is `[24, 12, 8, 6]`. For `[0, 2, 3]` it is `[6, 0, 0]`, which is why I avoid division."

**Seen in:** `main.py`, `product_except_self`

## List repetition (`[1] * n`)  `(syntax)`
**1. Definition:** Multiplying a list by an integer repeats its items, so `[1] * 3` is `[1, 1, 1]`.
**2. In plain words:** A quick way to create a list of n ones. It is safe for numbers but dangerous for nested lists, because `[[]] * 3` repeats the same inner list object.
**3. Example to quote:**
> "`result = [1] * n` creates n slots, each starting at 1."

**Seen in:** `main.py`, `product_except_self`

## Counting down with range  `(Python built-in)`
**1. Definition:** `range(start, stop, step)` with a negative step counts downwards; `range(n - 1, -1, -1)` yields n-1 down to 0.
**2. In plain words:** The second pass walks right to left. The stop value -1 is excluded, so the last index visited is 0.
**3. Example to quote:**
> "`list(range(2, -1, -1))` is `[2, 1, 0]`, which is the right-to-left sweep."

**Seen in:** `main.py`, `product_except_self`

## Running total variable  `(concept)`
**1. Definition:** A running variable that accumulates a value as the loop proceeds, such as `total += x` or `running *= nums[i]`.
**2. In plain words:** Prefix sums, prefix products and the subarray counter all rely on a running accumulator updated one element at a time.
**3. Example to quote:**
> "I keep `running` as the product so far, writing it into `result[i]` before multiplying in `nums[i]`."

**Seen in:** `main.py`, `subarray_sum`, `product_except_self`

## IndexError  `(concept)`
**1. Definition:** `IndexError` is raised when you access a list position that does not exist.
**2. In plain words:** The break-it drill triggers it with `query(0, 3)` on a 3-item list: `prefix[right + 1]` is `prefix[4]`, but the list has indexes 0-3. Read the traceback's last line and the line number above it.
**3. Example to quote:**
> "`IndexError: list index out of range` told me I asked for an index one past the end."

**Seen in:** `main.py`, BREAK IT comment
