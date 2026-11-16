# Step 02 — Hash-map basics: Two Sum, Contains Duplicate, Valid Anagram

## Goal
Replace nested loops with dict/set lookups; learn *complement lookup* and *frequency counting*.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/02_hashmap_basics/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| hash map | hash table, dictionary, dict, associative array, map | Same idea | Python's `dict` is a hash table; JavaScript `Map`/`Object` and Java `HashMap` are the same concept. |
| hash set | `set`, HashSet | Same idea | A hash map with keys only, no values. |
| complement lookup | two-sum trick, target minus x | Same | Store what you have seen, look up `target - x`. |
| frequency count | histogram, counter, tally, bag, multiset | Roughly | A dict from item to number of occurrences; `Counter` is the built-in. |
| brute force | naive, exhaustive search | Same | Try everything; the correctness baseline. |
| anagram check by sort vs by count | O(n log n) vs O(n) | No | Same answer, different cost. |

## Code, step by step

### 1. Header docstring and import Counter  (`main.py` lines 1–11)

```python
"""
GOAL: Solve the 3 classic hash-map problems (Two Sum, Contains Duplicate, Valid Anagram)
      first by brute force, then with a dict/set, and prove both agree.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/02_hashmap_basics/main.py
EXPECTED OUTPUT: lines for each problem and "All hash-map tests passed."
NEEDS: standard library only (collections.Counter).
"""

from collections import Counter  # Counter(iterable) -> dict-like {item: how_many_times}
```

- **In simple words:** Label for the file, then load the counting helper.
- **Exactly:** `Counter(iterable)` is a dict subclass mapping each item to how many times it occurs, e.g. `Counter("aab") == {'a': 2, 'b': 1}`.
- **Why it is here:** Three classic problems are solved first by brute force and then with a dict/set; Counter is the one-liner version.

### 2. two_sum_brute: try every pair  (`main.py` lines 17–23)

```python
def two_sum_brute(nums, target):
    """Try every pair.  Time O(n^2), Space O(1)."""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):      # j > i so we never reuse the same element
            if nums[i] + nums[j] == target:
                return [i, j]
    return []                                   # no pair found
```

- **In simple words:** Check every pair of numbers until two add up to the target.
- **Exactly:** Outer `i`, inner `j` starting at `i+1` so no element pairs with itself and no pair repeats. Returns the two indices as a list, or `[]`. Time O(n^2), space O(1).
- **Why it is here:** The correct-but-slow baseline you compare the faster version against.

### 3. two_sum_dict: remember what you have seen  (`main.py` lines 26–38)

```python
def two_sum_dict(nums, target):
    """One pass with a dict {value_seen: index}.  Time O(n), Space O(n).

    Idea ("complement lookup"): for each x, the partner we need is target - x.
    If we've already seen that partner, we're done; otherwise remember x.
    """
    seen = {}                                   # value -> index where we saw it
    for i, x in enumerate(nums):
        complement = target - x
        if complement in seen:                  # O(1) average dict lookup
            return [seen[complement], i]
        seen[x] = i                             # store AFTER checking so x can't pair with itself
    return []
```

- **In simple words:** For each number, ask "have I already seen the partner I need?".
- **Exactly:** `seen` maps value -> index. `complement = target - x`. `complement in seen` is an average O(1) hash lookup; if yes return `[seen[complement], i]`. Otherwise store `seen[x] = i` AFTER the check so a number cannot pair with itself. Time O(n), space O(n). For `([3, 2, 4], 6)`: i=0 x=3 need 3 -> not seen, store {3:0}; i=1 x=2 need 4 -> store {3:0, 2:1}; i=2 x=4 need 2 -> seen at 1 -> `[1, 2]`.
- **Why it is here:** Trades memory for speed: the dict removes the inner loop.

### 4. contains_duplicate_brute  (`main.py` lines 44–50)

```python
def contains_duplicate_brute(nums):
    """Compare every pair.  O(n^2) time, O(1) space."""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False
```

- **In simple words:** Compare every pair; True if any two are equal.
- **Exactly:** Same double loop as Two Sum. O(n^2) time, O(1) space.
- **Why it is here:** The baseline for the duplicate problem.

### 5. contains_duplicate_sort  (`main.py` lines 53–56)

```python
def contains_duplicate_sort(nums):
    """Sort then compare neighbours.  O(n log n) time, O(n) space here (sorted() copies)."""
    ordered = sorted(nums)
    return any(a == b for a, b in zip(ordered, ordered[1:]))  # zip pairs neighbours
```

- **In simple words:** Sort first, so equal numbers sit side by side, then compare neighbours.
- **Exactly:** `sorted(nums)` returns a new list (so O(n) extra space). `zip(ordered, ordered[1:])` pairs each element with the next one. `any(a == b ...)` is `True` at the first equal pair. O(n log n) time.
- **Why it is here:** A middle path when you cannot afford the extra set memory, or when the data is already sorted.

### 6. contains_duplicate_set  (`main.py` lines 59–66)

```python
def contains_duplicate_set(nums):
    """Set remembers what we've seen.  O(n) time, O(n) space.  Shortcut: len(set(nums)) != len(nums)."""
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False
```

- **In simple words:** Remember what you have seen; stop when a number appears again.
- **Exactly:** `x in seen` is O(1) on average; `seen.add(x)` inserts. Returns `True` on first repeat. O(n) time and space. One-liner equivalent: `len(set(nums)) != len(nums)` (but it always builds the full set).
- **Why it is here:** The standard interview answer; early exit when a duplicate appears.

### 7. is_anagram_sort  (`main.py` lines 72–74)

```python
def is_anagram_sort(s, t):
    """Sort both and compare.  O(n log n) time."""
    return sorted(s) == sorted(t)
```

- **In simple words:** Two words are anagrams if their sorted letters are the same.
- **Exactly:** `sorted("rat")` is `['a', 'r', 't']` (a list of characters); lists compare element by element. O(n log n).
- **Why it is here:** Shortest correct answer; good first answer in an interview.

### 8. is_anagram_count: hand-written frequency count  (`main.py` lines 77–88)

```python
def is_anagram_count(s, t):
    """Frequency counting with a hand-written dict.  O(n) time, O(k) space (k = alphabet size)."""
    if len(s) != len(t):                        # different length can never match -> early exit
        return False
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1      # dict.get(key, default) avoids KeyError
    for ch in t:
        if counts.get(ch, 0) == 0:              # t has a letter s doesn't have (or too many of it)
            return False
        counts[ch] -= 1
    return True
```

- **In simple words:** Count letters in the first word, then use them up with the second; any shortage means not an anagram.
- **Exactly:** `len(s) != len(t)` returns early. `counts.get(ch, 0) + 1` increments with a default 0 (no `KeyError`). In `t`, a letter whose count is 0 (or absent) means `t` has too many of it -> `False`; else subtract 1. O(n) time, O(k) space where k = distinct letters.
- **Why it is here:** The pattern behind many string/array problems; the length check makes the final `True` safe (all counts reach 0).

### 9. is_anagram_counter  (`main.py` lines 91–93)

```python
def is_anagram_counter(s, t):
    """Same idea in one line using collections.Counter."""
    return Counter(s) == Counter(t)
```

- **In simple words:** The same idea in one line.
- **Exactly:** `Counter(s) == Counter(t)` compares two frequency dicts for equality (order does not matter).
- **Why it is here:** Know both the manual and the library version; interviewers often ask you to code the manual one.

### 10. Tests for Two Sum  (`main.py` lines 99–104)

```python
if __name__ == "__main__":                       # runs only when executed directly, not on import
    cases = [([2, 7, 11, 15], 9), ([3, 2, 4], 6), ([3, 3], 6), ([1, 2], 10), ([], 5), ([5], 5)]
    for nums, target in cases:
        a, b = two_sum_brute(nums, target), two_sum_dict(nums, target)
        print(f"two_sum({nums}, {target}) brute={a} dict={b}")
        assert a == b
```

- **In simple words:** Run both versions on six inputs and require identical answers.
- **Exactly:** `if __name__ == "__main__":` runs only when executed directly. Real output: `([2,7,11,15], 9) -> [0, 1]`, `([3,2,4], 6) -> [1, 2]`, `([3,3], 6) -> [0, 1]`, and `[]` for `([1,2], 10)`, `([], 5)`, `([5], 5)`. `a, b = ..., ...` is a tuple assignment; `assert a == b` proves both agree.
- **Why it is here:** Edge cases first: duplicates (`[3,3]`), empty input, a single element that equals the target.

### 11. Tests for Contains Duplicate  (`main.py` lines 106–109)

```python
    for nums in ([1, 2, 3, 1], [1, 2, 3, 4], [], [7], [5, 5]):
        results = {contains_duplicate_brute(nums), contains_duplicate_sort(nums), contains_duplicate_set(nums)}
        print(f"contains_duplicate({nums}) -> {results}")
        assert len(results) == 1                 # all three methods agree
```

- **In simple words:** Run all three versions and expect exactly one distinct answer.
- **Exactly:** The `{...}` is a set of three booleans; if all agree it has length 1. Real output e.g. `contains_duplicate([1, 2, 3, 1]) -> {True}` and `[] -> {False}`.
- **Why it is here:** Cross-checking independent implementations is a cheap way to catch bugs.

### 12. Tests for Valid Anagram, final message  (`main.py` lines 111–116)

```python
    for s, t, expected in [("anagram", "nagaram", True), ("rat", "car", False), ("", "", True),
                           ("a", "ab", False), ("aab", "abb", False)]:
        assert is_anagram_sort(s, t) == is_anagram_count(s, t) == is_anagram_counter(s, t) == expected
        print(f"is_anagram({s!r}, {t!r}) -> {expected}")

    print("All hash-map tests passed.")
```

- **In simple words:** Check five pairs against the expected answer for all three implementations.
- **Exactly:** `a == b == c == expected` is a chained comparison. Pairs include the empty strings (True) and different lengths (False). Real output ends with `All hash-map tests passed.`
- **Why it is here:** Empty strings and unequal lengths are the edge cases people forget.

### 13. TRY IT and BREAK IT comments  (`main.py` lines 117–118)

```python
    # TRY IT: add a case with negative numbers to two_sum. Does the dict version still work? Why?
    # BREAK IT: move `seen[x] = i` ABOVE the `if` in two_sum_dict, then run two_sum([3, 2, 4], 6)... what goes wrong?
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** Negative numbers still work in `two_sum_dict` since the dict stores any hashable value. Moving `seen[x] = i` above the `if` makes `two_sum([3, 2, 4], 6)` return `[0, 0]`: at i=0, x=3 is stored first, then its complement 3 is found, so the element pairs with itself.
- **Why it is here:** Shows why the order (check, then store) matters.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `d = {}` / `d[k] = v` | dict: key → value | `d = {"a": 1}` |
| `k in d` | key membership, O(1) avg | `"a" in d` → `True` |
| `d.get(k, 0)` | value or default, no KeyError | `{}.get("x", 0)` → `0` |
| `zip(a, a[1:])` | pairs neighbours | `list(zip([1,2,3],[2,3]))` → `[(1,2),(2,3)]` |
| `any(gen)` | True if any item True | `any(x > 2 for x in [1,3])` |
| `if __name__ == "__main__":` | run block only when file executed directly | |
| `f"{s!r}"` | `!r` shows the repr (quotes) | `f"{'a'!r}"` → `'a'` |

## Libraries & functions
| Name | Origin | Does | Args | Returns |
|---|---|---|---|---|
| `collections.Counter` | stdlib | counts items | iterable | dict-like `{item: count}`; `==` compares counts |
| `sorted(x)` | built-in | new sorted list | `key`, `reverse` | list |
| `enumerate` | built-in | index+value | iterable, `start` | iterator |
| `set.add`, `in` | built-in | add / test | | None / bool |

## Execution trace (real run)
`two_sum_dict([2, 7, 11, 15], 9)`:
| i | x | complement | seen before | action |
|---|---|---|---|---|
| 0 | 2 | 7 | `{}` | store `{2: 0}` |
| 1 | 7 | 2 | `{2: 0}` → found | return `[0, 1]` |

Output excerpt:
```
two_sum([3, 2, 4], 6) brute=[1, 2] dict=[1, 2]
two_sum([3, 3], 6) brute=[0, 1] dict=[0, 1]
contains_duplicate([1, 2, 3, 1]) -> {True}
is_anagram('aab', 'abb') -> False
All hash-map tests passed.
```

| Problem | Brute | Better | Best |
|---|---|---|---|
| Two Sum | O(n²)/O(1) | — | dict O(n)/O(n) |
| Contains Duplicate | O(n²)/O(1) | sort O(n log n) | set O(n)/O(n) |
| Valid Anagram | — | sort O(n log n) | counts O(n)/O(k) |

## Experiments
1. Add negatives, e.g. `([-3, 4, 3, 90], 0)` → still `[0, 2]`; dict works for any ints.
2. Time brute vs dict on `list(range(5000))` with an impossible target.
3. `is_anagram_count("Listen", "Silent")` → False; case matters. How would you fix it? (`.lower()`)

## Debugging practice
Move `seen[x] = i` above the `if`. `two_sum([3], 6)` now returns `[0, 0]` — one element paired with itself. No traceback: a *logic* bug found only by an edge-case test.

## Rebuild checklist
- [ ] Two Sum with dict, no peeking.
- [ ] Contains Duplicate with a set (early return).
- [ ] Anagram with a dict, including the length check.

## Say it aloud
"Brute-force Two Sum checks all pairs, O(n²). Instead, for each number I look up target minus the number in a dict of numbers seen so far — O(1) per lookup, so O(n) total, at the cost of O(n) memory. I insert after checking so a number can't pair with itself."
