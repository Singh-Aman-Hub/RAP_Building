# Terms — Hash map basics

Folder: `02_dsa_arrays_hashmaps/02_hashmap_basics` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [DSA README](../README.md)

**Terms in this folder:** [Hash map (dict)](#hash-map-dict--python-built-in) · [Complement lookup](#complement-lookup--concept) · [Two Sum](#two-sum--concept) · [Brute force](#brute-force--concept) · [Frequency counting](#frequency-counting--concept) · [collections.Counter](#collectionscounter--standard-library-collections) · [dict.get(key, default)](#dictgetkey-default--python-built-in) · [Membership test (`in`)](#membership-test-in--syntax) · [set (add / membership)](#set-add--membership--python-built-in) · [sorted()](#sorted--python-built-in) · [zip()](#zip--python-built-in) · [any()](#any--python-built-in) · [enumerate()](#enumerate--python-built-in) · [len()](#len--python-built-in) · [Early return](#early-return--concept) · [if __name__ == "__main__"](#if-__name__--__main__--syntax) · [f-string `!r` conversion](#f-string-r-conversion--syntax) · [Time and space trade-off](#time-and-space-trade-off--concept) · [KeyError](#keyerror--concept)

---

## Hash map (dict)  `(Python built-in)`
**1. Definition:** A hash map is a key-to-value mapping that uses hashing for O(1) average-time insert, lookup and delete; Python's built-in version is `dict`.
**2. In plain words:** Think of a phone book where you look up a name and get a number instantly, instead of reading every line. In this folder a dict remembers "which number I saw and at which index", which is what makes Two Sum fast. Keys must be hashable (numbers, strings, tuples are fine; lists are not), and the dict costs extra memory.
**3. Example to quote:**
> "I keep `seen = {}` mapping each value to its index, so checking whether I have seen the number I need is a single O(1) lookup."

```python
seen = {2: 0}
7 in seen   # False
```
**Seen in:** `main.py`, `two_sum_dict`

## Complement lookup  `(concept)`
**1. Definition:** Complement lookup means that for each element x, you compute the partner value you need (for example `target - x`) and check whether it has already been stored.
**2. In plain words:** Instead of searching the whole array for a partner, you work out exactly which partner would complete the pair and ask the dictionary if it has seen it. That turns an inner loop into one lookup. It only works once you store seen values, and the order of "check then store" matters.
**3. Example to quote:**
> "For Two Sum, at x=2 with target 9 the complement is 7. It is not in `seen` yet, so I store 2. At x=7 the complement is 2, which is in `seen`, so I return `[0, 1]`."

**Seen in:** `main.py`, `two_sum_dict`

## Two Sum  `(concept)`
**1. Definition:** Two Sum is the problem of finding the indices of two numbers in a list whose sum equals a given target.
**2. In plain words:** Brute force checks every pair, O(n²). The dict version walks the list once, O(n). It is the standard "hello world" of hash-map interview problems.
**3. Example to quote:**
> "For `[2, 7, 11, 15]` and target 9, the answer is indices 0 and 1 because 2 + 7 = 9."

**Seen in:** `main.py`, `two_sum_brute`, `two_sum_dict`

## Brute force  `(concept)`
**1. Definition:** A brute-force solution tries every possibility without exploiting structure in the problem.
**2. In plain words:** It is the simplest correct approach and is the right first step in an interview: state it, give its complexity, then improve it. Here brute-force Two Sum uses nested loops. It is easy to get right and slow on large inputs, which is why the file also keeps the brute version to cross-check the fast one in tests.
**3. Example to quote:**
> "My first answer is brute force, O(n²) time and O(1) space. Then I improve it with a dictionary to O(n)."

**Seen in:** `main.py`, `two_sum_brute`, `contains_duplicate_brute`

## Frequency counting  `(concept)`
**1. Definition:** Frequency counting is building a mapping from each distinct item to the number of times it occurs.
**2. In plain words:** Two words are anagrams if each letter appears the same number of times. Counting letters in a dict makes that check O(n). The same technique answers "most common item" and "any duplicates" questions.
**3. Example to quote:**
> "For 'aab' vs 'abb' the counts are `{a:2, b:1}` and `{a:1, b:2}`. They differ, so it is not an anagram."

**Seen in:** `main.py`, `is_anagram_count`

## collections.Counter  `(standard library: collections)`
**1. Definition:** `Counter(iterable)` is a dict subclass that counts hashable items, returning a mapping of item to count; two Counters compare equal if they have the same counts.
**2. In plain words:** It is the one-line version of the hand-written counting loop. `Counter("anagram") == Counter("nagaram")` is True. In interviews, write the dict version first, then mention Counter as the shortcut.
**3. Example to quote:**
> "`Counter('rat') == Counter('car')` is False, because 'r', 'a', 't' have different counts than 'c', 'a', 'r'."

```python
from collections import Counter
Counter("aab")   # Counter({'a': 2, 'b': 1})
```
**Seen in:** `main.py`, `is_anagram_counter`

## dict.get(key, default)  `(Python built-in)`
**1. Definition:** `dict.get(key, default)` returns the value for `key` if present, otherwise `default` (None if omitted), without raising `KeyError`.
**2. In plain words:** It is the safe way to read a key that may not exist. `counts.get(ch, 0) + 1` means "the current count, or 0 if this is the first time". Using `counts[ch]` on a missing key would crash with `KeyError`.
**3. Example to quote:**
> "`{}.get('x', 0)` returns 0 instead of crashing, which is why I use it when counting letters."

**Seen in:** `main.py`, `is_anagram_count`

## Membership test (`in`)  `(syntax)`
**1. Definition:** The `in` operator tests whether an item is contained in a collection and returns True or False.
**2. In plain words:** On a dict it checks keys and on a set it checks items, both O(1) average; on a list it scans, O(n). So `complement in seen` is cheap because `seen` is a dict.
**3. Example to quote:**
> "`'a' in {'a': 1}` is True and is O(1) on average; the same test on a list would scan it."

**Seen in:** `main.py`, `two_sum_dict`, `contains_duplicate_set`

## set (add / membership)  `(Python built-in)`
**1. Definition:** A `set` is an unordered collection of unique hashable items with O(1) average `add` and `in`.
**2. In plain words:** It automatically refuses duplicates and answers "have I seen this?" instantly. The duplicate checker adds each number to a set and returns True as soon as one is already there.
**3. Example to quote:**
> "For `[1, 2, 3, 1]`, I add 1, 2, 3 and when I reach the second 1 it is already in the set, so there is a duplicate."

```python
seen = set(); seen.add(4); 4 in seen   # True
```
**Seen in:** `main.py`, `contains_duplicate_set`

## sorted()  `(Python built-in)`
**1. Definition:** `sorted(iterable, key=None, reverse=False)` returns a new sorted list and leaves the original unchanged.
**2. In plain words:** After sorting, equal values sit next to each other, so you can find duplicates by comparing neighbours; sorting both strings and comparing them checks anagrams. Sorting costs O(n log n) time and O(n) extra space because it makes a copy.
**3. Example to quote:**
> "`sorted('rat')` and `sorted('tar')` are both `['a','r','t']`, so they are anagrams."

**Seen in:** `main.py`, `contains_duplicate_sort`, `is_anagram_sort`

## zip()  `(Python built-in)`
**1. Definition:** `zip(a, b)` pairs up items from two iterables position by position, stopping at the shortest.
**2. In plain words:** `zip(ordered, ordered[1:])` pairs each item with the next one, so you can compare neighbours without index arithmetic.
**3. Example to quote:**
> "`list(zip([1, 2, 3], [2, 3]))` is `[(1, 2), (2, 3)]`, the neighbouring pairs."

**Seen in:** `main.py`, `contains_duplicate_sort`

## any()  `(Python built-in)`
**1. Definition:** `any(iterable)` returns True if at least one item in the iterable is truthy, stopping early at the first one.
**2. In plain words:** Combined with a generator it asks "is there at least one neighbouring pair that is equal?" without building a list.
**3. Example to quote:**
> "`any(x > 2 for x in [1, 3])` is True as soon as it sees 3."

**Seen in:** `main.py`, `contains_duplicate_sort`

## enumerate()  `(Python built-in)`
**1. Definition:** `enumerate(iterable)` yields `(index, item)` pairs.
**2. In plain words:** Two Sum needs the index of each number to return it, and `enumerate` gives it with the value in one loop.
**3. Example to quote:**
> "`for i, x in enumerate(nums)` gives me both the position i and the value x."

**Seen in:** `main.py`, `two_sum_dict`

## len()  `(Python built-in)`
**1. Definition:** `len(obj)` returns the number of items in a sequence or collection.
**2. In plain words:** Valid Anagram uses it as an early exit: strings of different length can never be anagrams, so return False without counting.
**3. Example to quote:**
> "`len('a') != len('ab')`, so I return False immediately."

**Seen in:** `main.py`, `is_anagram_count`

## Early return  `(concept)`
**1. Definition:** An early return exits a function as soon as the answer is known, instead of finishing the remaining work.
**2. In plain words:** Once the set contains a repeat, there is no need to scan the rest. It makes code faster and flatter.
**3. Example to quote:**
> "In `contains_duplicate_set` I return True the moment I see a repeated value; I do not scan the rest."

**Seen in:** `main.py`, `contains_duplicate_set`

## if __name__ == "__main__"  `(syntax)`
**1. Definition:** This condition is True only when the file is run directly as a script, and False when the file is imported as a module.
**2. In plain words:** The tests sit under it so running `python main.py` executes them, while another file can `import main` and reuse the functions without triggering the tests.
**3. Example to quote:**
> "I wrap the test code in `if __name__ == "__main__":` so importing my functions does not run the tests."

**Seen in:** `main.py`, test section

## f-string `!r` conversion  `(syntax)`
**1. Definition:** In an f-string, `{value!r}` inserts `repr(value)`, the developer-friendly representation, which shows quotes around strings.
**2. In plain words:** It makes empty strings and spaces visible in output: `''` instead of nothing. Handy in test prints.
**3. Example to quote:**
> "`f"{'a'!r}"` prints `'a'` with quotes, so I can tell an empty string from a missing value."

**Seen in:** `main.py`, anagram test print

## Time and space trade-off  `(concept)`
**1. Definition:** A time/space trade-off is spending more memory to reduce running time, or vice versa.
**2. In plain words:** The dict and set versions are O(n) time but O(n) memory; the brute-force versions use O(1) memory but take O(n²) time. Know both costs when you propose a solution.
**3. Example to quote:**
> "The dictionary solution is O(n) time but needs O(n) extra space, whereas brute force needs O(1) space but O(n²) time."

**Seen in:** `WALKTHROUGH.md`, complexity table

## KeyError  `(concept)`
**1. Definition:** `KeyError` is raised when you access a dict key that does not exist with `d[key]`.
**2. In plain words:** It is what `d.get(key, default)` helps you avoid when counting items you have not seen yet.
**3. Example to quote:**
> "`{}['x']` raises `KeyError: 'x'`; `{}.get('x', 0)` returns 0."

**Seen in:** `main.py`, comment on `dict.get`
