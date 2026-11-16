# Step 08 — Grouping, counting and strings

## Goal
Use dicts to group items by a *key function*, count frequencies, and solve common string problems.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/08_grouping_and_strings/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `defaultdict` vs `dict.setdefault` vs `dict.get` | three ways to handle missing keys | No | `defaultdict` auto-creates on access; `setdefault` creates on call; `get` just returns a default without storing. |
| `Counter` vs `dict` | counting dict vs plain dict | Subclass | `Counter` is a dict subclass with `most_common`, arithmetic and 0 for missing keys. |
| anagram key: sorted string vs count tuple | canonical form | Both work | Sorted is simpler (O(k log k)); count tuple is O(k). |
| bucket sort vs `most_common` | O(n) vs O(n log n) | Same answer | Bucket sort exploits that counts are at most n. |
| hashable vs mutable | list vs tuple as dict key | No | Lists are mutable and not hashable; tuples of hashables are. |
| subsequence vs substring | gaps allowed vs contiguous | No | "ace" is a subsequence of "abcde" but not a substring. |

## Code, step by step

### 1. Header docstring and imports  (`main.py` lines 1–10)

```python
"""
GOAL: Grouping and counting with dicts, plus essential string problems.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/08_grouping_and_strings/main.py
EXPECTED OUTPUT: grouped anagrams, top-k, then "All grouping/string tests passed."
NEEDS: standard library only (collections).
"""

from collections import Counter, defaultdict  # defaultdict(list): missing keys start as []
```

- **In simple words:** Label for the file, then two dict helpers from the standard library.
- **Exactly:** `Counter(iterable)` counts occurrences; `defaultdict(list)` is a dict that creates an empty list the first time you touch a missing key.
- **Why it is here:** Both remove the `if key not in d` boilerplate from grouping and counting code.

### 2. group_anagrams by sorted letters  (`main.py` lines 13–19)

```python
def group_anagrams(words):
    """Words with the same letters share a key: the sorted letters. O(n * k log k)."""
    groups = defaultdict(list)                  # no need to check "if key not in groups"
    for w in words:
        key = "".join(sorted(w))                # sorted("eat") -> ['a','e','t'] -> "aet"
        groups[key].append(w)
    return list(groups.values())
```

- **In simple words:** Words that are anagrams share the same key: their letters sorted.
- **Exactly:** `groups = defaultdict(list)`. `"".join(sorted(w))` turns `"eat"` into `"aet"` (sorted gives a list of characters; `join` glues them back). `groups[key].append(w)` files the word under its key. For `["eat","tea","tan","ate","nat","bat"]` the keys are `aet, aet, ant, aet, ant, abt`. Real output: `groups: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]`. O(n * k log k) for n words of length k.
- **Why it is here:** The "canonical form as dict key" trick is the core of all grouping problems.

### 3. group_anagrams_by_count with a 26-slot tuple key  (`main.py` lines 22–30)

```python
def group_anagrams_by_count(words):
    """Alternative key: a tuple of 26 letter counts. O(n * k) -- no sorting. Lists aren't hashable, tuples are."""
    groups = defaultdict(list)
    for w in words:
        counts = [0] * 26
        for ch in w:
            counts[ord(ch) - ord("a")] += 1     # ord('b') - ord('a') == 1
        groups[tuple(counts)].append(w)
    return list(groups.values())
```

- **In simple words:** Instead of sorting, count the letters and use the counts as the key.
- **Exactly:** `counts = [0] * 26` one slot per letter a-z. `ord(ch) - ord("a")` maps `"a"->0 ... "z"->25`. `tuple(counts)` is needed because a list is not hashable and cannot be a dict key but a tuple can. O(n * k). Only valid for lowercase a-z.
- **Why it is here:** Faster for long words and a good answer when asked for something better than sorting.

### 4. top_k_frequent with Counter  (`main.py` lines 33–35)

```python
def top_k_frequent(nums, k):
    """k most common values. Counter does the counting; most_common sorts by count. O(n log n)."""
    return [value for value, _ in Counter(nums).most_common(k)]
```

- **In simple words:** Count every value, then take the k most common.
- **Exactly:** `Counter(nums).most_common(k)` returns a list of `(value, count)` sorted by count, e.g. `[(1, 3), (2, 2)]` for `[1,1,1,2,2,3]`. The list comprehension keeps only the values, using `_` for the ignored count. Output: `top 2 frequent: [1, 2]`. O(n log n).
- **Why it is here:** The one-line answer; you must still explain what `most_common` does.

### 5. top_k_frequent_buckets: bucket sort in O(n)  (`main.py` lines 38–50)

```python
def top_k_frequent_buckets(nums, k):
    """Bucket sort by frequency: O(n). buckets[c] = values that appear exactly c times."""
    counts = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]        # list comprehension (NOT [[]] * n, which shares one list!)
    for value, c in counts.items():
        buckets[c].append(value)
    result = []
    for c in range(len(buckets) - 1, 0, -1):            # highest frequency first
        for value in buckets[c]:
            result.append(value)
            if len(result) == k:
                return result
    return result
```

- **In simple words:** Put each value into a bucket numbered by how often it appears, then read buckets from the highest count down.
- **Exactly:** A value can appear at most n times, so `buckets` has n+1 lists. `[[] for _ in range(...)]` builds separate lists (`[[]] * n` would repeat one shared list). For `[1,1,1,2,2,3]`: `buckets[3]=[1]`, `buckets[2]=[2]`, `buckets[1]=[3]`. Scanning `c` from `len(buckets)-1` down to 1, it appends 1 then 2 and returns when `len(result) == k`.
- **Why it is here:** Avoids sorting entirely: counting is O(n) and the bucket walk is O(n).

### 6. first_unique_char: count then scan  (`main.py` lines 53–59)

```python
def first_unique_char(s):
    """Index of the first character that appears once, else -1. Two passes. O(n)."""
    counts = Counter(s)
    for i, ch in enumerate(s):
        if counts[ch] == 1:
            return i
    return -1
```

- **In simple words:** Count each letter first, then find the first letter whose count is 1.
- **Exactly:** Pass 1: `Counter(s)`. Pass 2: `enumerate(s)` in original order; `counts[ch] == 1` returns the index. `"leetcode"` -> `0` (`l`), `"aabb"` -> `-1`. O(n).
- **Why it is here:** Two passes are needed because you only know uniqueness after seeing the whole string.

### 7. reverse_words  (`main.py` lines 62–64)

```python
def reverse_words(s):
    """'  the sky  is blue ' -> 'blue is sky the'.  split() with no args also trims extra spaces."""
    return " ".join(reversed(s.split()))
```

- **In simple words:** Split on whitespace, reverse the word list, join with single spaces.
- **Exactly:** `s.split()` with no argument splits on any run of whitespace and drops empty pieces, so `"  the sky  is blue "` -> `['the','sky','is','blue']`. `reversed(...)` is an iterator; `" ".join(...)` -> `"blue is sky the"`.
- **Why it is here:** `split()` without arguments quietly handles leading, trailing and repeated spaces.

### 8. longest_common_prefix by shrinking  (`main.py` lines 67–77)

```python
def longest_common_prefix(strs):
    """Shrink the prefix until every word starts with it. O(total characters)."""
    if not strs:
        return ""
    prefix = strs[0]
    for word in strs[1:]:
        while not word.startswith(prefix):
            prefix = prefix[:-1]                         # drop last char
            if not prefix:
                return ""
    return prefix
```

- **In simple words:** Start with the first word as the prefix and chop letters off until every word starts with it.
- **Exactly:** `word.startswith(prefix)` checks the beginning. `prefix[:-1]` drops the last character. If the prefix becomes empty return `""`. For `["flower","flow","flight"]`: `flower` -> `flow` fits `flow`; for `flight` shrinks `flow -> flo -> fl` -> `"fl"`. Empty input returns `""`.
- **Why it is here:** The empty-list guard and the empty-prefix early return are the edge cases.

### 9. is_subsequence with a pointer into s  (`main.py` lines 80–86)

```python
def is_subsequence(s, t):
    """Are the letters of s inside t in order (gaps allowed)? Two pointers. O(len(t))."""
    i = 0
    for ch in t:
        if i < len(s) and ch == s[i]:
            i += 1
    return i == len(s)
```

- **In simple words:** Walk through `t`; every time its letter matches the next needed letter of `s`, advance in `s`.
- **Exactly:** `i` counts matched letters of `s`. Gaps in `t` are allowed, order must be kept. `"ace"` in `"abcde"` -> `True`; `"aec"` -> `False`; empty `s` -> `True` (`i == 0 == len(s)`). O(len(t)).
- **Why it is here:** A two-pointer pattern on strings; note `i < len(s)` stops the index from running past the end.

### 10. majority_element by counting  (`main.py` lines 89–92)

```python
def majority_element(nums):
    """Value appearing more than n/2 times, via counting. (Boyer-Moore voting gives O(1) space; look it up later.)"""
    counts = Counter(nums)
    return max(counts, key=counts.get)                   # max over keys, comparing by their count
```

- **In simple words:** Count everything and return the value with the biggest count.
- **Exactly:** `max(counts, key=counts.get)` iterates the dict's keys and compares them by their counts. `[2,2,1,1,2]` -> `2`. O(n) time, O(n) space; Boyer-Moore voting does it in O(1) space.
- **Why it is here:** Simple and correct; the optimisation is a follow-up question.

### 11. Tests: grouping anagrams  (`main.py` lines 95–101)

```python
if __name__ == "__main__":
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    g1, g2 = group_anagrams(words), group_anagrams_by_count(words)
    print("groups:", g1)
    norm = lambda groups: sorted(sorted(g) for g in groups)   # order inside/between groups doesn't matter
    assert norm(g1) == norm(g2) == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams([]) == [] and group_anagrams([""]) == [[""]]
```

- **In simple words:** Run both grouping functions, print the first, and check they agree.
- **Exactly:** Real print: `groups: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]`. `norm` sorts words inside each group and the groups themselves, since dict order and group order are not part of the contract. Edge cases `[]` and `[""]`.
- **Why it is here:** Normalising before comparing avoids failing tests on harmless ordering differences.

### 12. Tests: top-k frequent  (`main.py` lines 103–106)

```python
    nums = [1, 1, 1, 2, 2, 3]
    print("top 2 frequent:", top_k_frequent(nums, 2))
    assert set(top_k_frequent(nums, 2)) == set(top_k_frequent_buckets(nums, 2)) == {1, 2}
    assert top_k_frequent([5], 1) == [5]
```

- **In simple words:** Compare the Counter and bucket versions as sets and print one.
- **Exactly:** Prints `top 2 frequent: [1, 2]`. Compared as sets because tied counts have no required order. Single-element case `[5]`.
- **Why it is here:** Comparing as sets matches what the problem statement guarantees.

### 13. Tests: the string functions and final message  (`main.py` lines 108–114)

```python
    assert first_unique_char("leetcode") == 0 and first_unique_char("aabb") == -1 and first_unique_char("") == -1
    assert reverse_words("  the sky  is blue ") == "blue is sky the" and reverse_words("") == ""
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl" and longest_common_prefix(["a", "b"]) == ""
    assert longest_common_prefix([]) == "" and longest_common_prefix(["solo"]) == "solo"
    assert is_subsequence("ace", "abcde") and not is_subsequence("aec", "abcde") and is_subsequence("", "x")
    assert majority_element([2, 2, 1, 1, 2]) == 2
    print("All grouping/string tests passed.")
```

- **In simple words:** Assert answers for the remaining functions and print success.
- **Exactly:** Includes empty string, empty list and single word. Final real output: `All grouping/string tests passed.`
- **Why it is here:** Each assert is a documented edge case you should state in an interview.

### 14. TRY IT and BREAK IT comments  (`main.py` lines 115–116)

```python
    # TRY IT: replace defaultdict(list) with a plain {} and run -- read the KeyError traceback.
    # BREAK IT: use `buckets = [[]] * (len(nums) + 1)` in top_k_frequent_buckets. Every bucket is the SAME list object!
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** A plain `{}` raises `KeyError` on `groups[key].append(w)` when the key is new. `[[]] * (len(nums)+1)` creates n+1 references to ONE list, so every value lands in the same bucket and the results are wrong.
- **Why it is here:** The two classic grouping bugs.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `defaultdict(list)` | dict whose missing keys start as `[]` | `d["x"].append(1)` |
| `"".join(list_of_str)` | glue strings | `"".join(["a","b"])` → `"ab"` |
| `tuple(counts)` | immutable, hashable → valid dict key | lists are NOT hashable |
| `[[] for _ in range(n)]` | list comprehension making independent lists | vs `[[]] * n` (shared!) |
| `ord("b")` | character → integer code | `ord("b") - ord("a")` → `1` |
| `max(d, key=d.get)` | key with the largest value | |
| `lambda` | small inline function | |

## Libraries & functions
| Name | Origin | Does | Returns |
|---|---|---|---|
| `collections.defaultdict` | stdlib | auto-initialised dict | dict subclass |
| `collections.Counter` | stdlib | frequency counts; `.most_common(k)` | list of `(item, count)` |
| `str.split()`, `str.startswith()` | built-in | tokenise / prefix test | list / bool |
| `reversed(seq)` | built-in | reverse iterator | iterator |

## Execution trace (real run)
`group_anagrams(["eat","tea","tan","ate","nat","bat"])`: keys `aet`→[eat,tea,ate], `ant`→[tan,nat], `abt`→[bat]. Output:
```
groups: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
top 2 frequent: [1, 2]
All grouping/string tests passed.
```
Complexity: sorting-key grouping O(n·k log k); count-tuple key O(n·k); top-k with buckets O(n).

## Experiments
1. Use a plain `{}` instead of `defaultdict` → `KeyError`.
2. Group words by *length* by changing the key to `len(w)`. Grouping = pick the right key.
3. `Counter("mississippi").most_common(2)`.

## Debugging practice
`buckets = [[]] * 5; buckets[1].append(9)` makes *every* bucket `[9]` because all slots are one object. Print `id(buckets[0]) == id(buckets[1])` to prove it.

## Rebuild checklist
- [ ] Group Anagrams with sorted-string key.
- [ ] Top K Frequent with `Counter`.
- [ ] First Unique Character in two passes.

## Say it aloud
"To group items I find a key that is identical for items that belong together — for anagrams, the sorted letters — and append each item to a dict list under that key. It's O(n) dictionary operations plus the cost of making each key. A defaultdict removes the 'key missing' check."
