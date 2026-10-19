# Terms — Grouping and strings

Folder: `02_dsa_arrays_hashmaps/08_grouping_and_strings` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [DSA README](../README.md)

**Terms in this folder:** [Grouping with a dict](#grouping-with-a-dict--concept) · [collections.defaultdict](#collectionsdefaultdict--standard-library-collections) · [Anagram](#anagram--concept) · [str.join()](#strjoin--python-built-in) · [Hashable keys: tuple vs list](#hashable-keys-tuple-vs-list--concept) · [ord()](#ord--python-built-in) · [Fixed-size count array](#fixed-size-count-array--concept) · [collections.Counter and most_common()](#collectionscounter-and-most_common--standard-library-collections) · [List comprehension](#list-comprehension--syntax) · [Bucket sort by frequency](#bucket-sort-by-frequency--concept) · [Shared-list pitfall (`[[]] * n`)](#shared-list-pitfall---n--concept) · [dict.items()](#dictitems--python-built-in) · [First Unique Character](#first-unique-character--concept) · [str.split() and reversed()](#strsplit-and-reversed--python-built-in) · [str.startswith()](#strstartswith--python-built-in) · [String slicing (`prefix[:-1]`)](#string-slicing-prefix-1--syntax) · [Subsequence](#subsequence--concept) · [max() with key=](#max-with-key--python-built-in) · [Majority element](#majority-element--concept) · [Normalising before comparing (`lambda` helper)](#normalising-before-comparing-lambda-helper--syntax)

---

## Grouping with a dict  `(concept)`
**1. Definition:** Grouping means collecting items into buckets keyed by a shared property, using a dict whose values are lists.
**2. In plain words:** Compute a key for each item and append the item to the list stored under that key. At the end the dict's values are your groups. It turns "which items belong together?" into a single pass.
**3. Example to quote:**
> "To group anagrams I use the sorted letters as the key, so 'eat', 'tea' and 'ate' all land under 'aet'."

**Seen in:** `main.py`, `group_anagrams`

## collections.defaultdict  `(standard library: collections)`
**1. Definition:** `defaultdict(factory)` is a dict subclass that creates a default value by calling `factory()` whenever a missing key is accessed.
**2. In plain words:** With `defaultdict(list)` you can write `groups[key].append(w)` without first checking whether the key exists. With a plain `{}` the same line raises `KeyError`. Accessing a missing key inserts it, which can surprise you.
**3. Example to quote:**
> "`d = defaultdict(list); d['x'].append(1)` just works, because 'x' starts as an empty list."

```python
from collections import defaultdict
groups = defaultdict(list)
```
**Seen in:** `main.py`, `group_anagrams`

## Anagram  `(concept)`
**1. Definition:** Two words are anagrams if they contain exactly the same letters with the same counts.
**2. In plain words:** 'eat', 'tea' and 'ate' are anagrams. Two ways to check: sort the letters, or count them. Both produce a canonical "signature" to use as a dict key.
**3. Example to quote:**
> "'eat' and 'tea' both sort to 'aet', so they share a signature and belong in one group."

**Seen in:** `main.py`, `group_anagrams`

## str.join()  `(Python built-in)`
**1. Definition:** `"sep".join(iterable_of_str)` concatenates the strings in an iterable, putting `sep` between them.
**2. In plain words:** `sorted("eat")` returns a list of letters; `"".join(...)` glues them back into the string 'aet'. The empty string means no separator.
**3. Example to quote:**
> "`''.join(['a', 'e', 't'])` is `'aet'`, which I use as the group key."

**Seen in:** `main.py`, `group_anagrams`, `reverse_words`

## Hashable keys: tuple vs list  `(concept)`
**1. Definition:** Dict keys must be hashable, meaning immutable with a stable hash; tuples, strings and numbers are hashable, lists are not.
**2. In plain words:** A list of letter counts cannot be a key because it could change, so the code converts it to a tuple. Using a list gives `TypeError: unhashable type: 'list'`.
**3. Example to quote:**
> "I convert the 26 counts to `tuple(counts)` so it can be a dict key; a list would raise `TypeError: unhashable type`."

**Seen in:** `main.py`, `group_anagrams_by_count`

## ord()  `(Python built-in)`
**1. Definition:** `ord(ch)` returns the integer Unicode code point of a single character.
**2. In plain words:** `ord('b') - ord('a')` equals 1, so each lowercase letter maps to an index 0-25 in a 26-slot count array, with no dict needed.
**3. Example to quote:**
> "`ord('c') - ord('a')` is 2, so 'c' increments `counts[2]`."

**Seen in:** `main.py`, `group_anagrams_by_count`

## Fixed-size count array  `(concept)`
**1. Definition:** A list of 26 integers where position i counts how often the i-th letter appears.
**2. In plain words:** An alternative to sorting letters: building the count tuple is O(k) per word instead of O(k log k), at the cost of assuming only lowercase a-z.
**3. Example to quote:**
> "For 'bat' the counts have 1 at positions for a, b and t. Equal count tuples mean anagrams, without sorting."

**Seen in:** `main.py`, `group_anagrams_by_count`

## collections.Counter and most_common()  `(standard library: collections)`
**1. Definition:** `Counter(iterable)` counts items; `.most_common(k)` returns the k highest-count `(item, count)` pairs sorted by count.
**2. In plain words:** `Counter([1,1,1,2,2,3]).most_common(2)` is `[(1, 3), (2, 2)]`. Sorting by count costs O(n log n) overall.
**3. Example to quote:**
> "For top-k frequent elements I call `Counter(nums).most_common(k)` and keep just the values."

**Seen in:** `main.py`, `top_k_frequent`, `first_unique_char`, `majority_element`

## List comprehension  `(syntax)`
**1. Definition:** A list comprehension `[expr for x in iterable if cond]` builds a new list in one expression.
**2. In plain words:** `[value for value, _ in Counter(nums).most_common(k)]` extracts just the values from the pairs. `[[] for _ in range(n)]` creates n separate empty lists.
**3. Example to quote:**
> "`[x * 2 for x in [1, 2, 3]]` gives `[2, 4, 6]` in one line."

**Seen in:** `main.py`, `top_k_frequent`, `top_k_frequent_buckets`

## Bucket sort by frequency  `(concept)`
**1. Definition:** Bucket sorting by frequency puts each value into `buckets[count]`, so values can be read from highest count to lowest in O(n) time without a comparison sort.
**2. In plain words:** The frequency can never exceed the list length, so you can use the count as an array index. Walk from the top bucket down and stop once you have k values.
**3. Example to quote:**
> "For `[1, 1, 1, 2, 2, 3]`, bucket 3 holds 1, bucket 2 holds 2, bucket 1 holds 3. Reading from the top gives the top-2: 1 and 2."

**Seen in:** `main.py`, `top_k_frequent_buckets`

## Shared-list pitfall (`[[]] * n`)  `(concept)`
**1. Definition:** `[[]] * n` creates n references to the same inner list rather than n separate lists.
**2. In plain words:** Appending to one bucket appears in all of them. `[[] for _ in range(n)]` runs the expression n times and gives independent lists. This is a classic bug the BREAK IT hint triggers.
**3. Example to quote:**
> "`[[]] * 3` gives three names for one list, so I use `[[] for _ in range(3)]` to get three separate lists."

**Seen in:** `main.py`, `top_k_frequent_buckets`

## dict.items()  `(Python built-in)`
**1. Definition:** `dict.items()` returns a view of `(key, value)` pairs.
**2. In plain words:** Used to loop over each distinct value together with its count: `for value, c in counts.items()`.
**3. Example to quote:**
> "`for value, c in counts.items()` lets me place each value in `buckets[c]`."

**Seen in:** `main.py`, `top_k_frequent_buckets`

## First Unique Character  `(concept)`
**1. Definition:** A problem asking for the index of the first character that appears exactly once in a string.
**2. In plain words:** Two passes: count all letters, then scan again in original order and return the first with count 1. O(n) time. Returns -1 if none.
**3. Example to quote:**
> "In 'leetcode' the first unique letter is 'l' at index 0; in 'aabb' there is none, so I return -1."

**Seen in:** `main.py`, `first_unique_char`

## str.split() and reversed()  `(Python built-in)`
**1. Definition:** `str.split()` with no arguments splits on any run of whitespace and drops empty pieces; `reversed(seq)` returns an iterator over the sequence backwards.
**2. In plain words:** `" ".join(reversed(s.split()))` reverses the word order and normalises messy spacing in one line.
**3. Example to quote:**
> "`'  the sky  is blue '` becomes `'blue is sky the'` because `split()` ignores the extra spaces."

**Seen in:** `main.py`, `reverse_words`

## str.startswith()  `(Python built-in)`
**1. Definition:** `str.startswith(prefix)` returns True if the string begins with the given prefix.
**2. In plain words:** Longest common prefix: start with the first word as the prefix and chop its last character until every word starts with it.
**3. Example to quote:**
> "`'flower'.startswith('flow')` is True, so 'flow' can still be the common prefix."

**Seen in:** `main.py`, `longest_common_prefix`

## String slicing (`prefix[:-1]`)  `(syntax)`
**1. Definition:** `s[:-1]` returns the string without its last character.
**2. In plain words:** Each time the prefix fails, drop its last letter. If it becomes empty, the answer is the empty string.
**3. Example to quote:**
> "`'flower'[:-1]` is `'flowe'`; I keep trimming until all words match."

**Seen in:** `main.py`, `longest_common_prefix`

## Subsequence  `(concept)`
**1. Definition:** A subsequence of a string keeps the original order of characters but may skip some.
**2. In plain words:** 'ace' is a subsequence of 'abcde' (a, c, e appear in order), while 'aec' is not. Two pointers solve it: advance through t, and advance the pointer in s only on a match.
**3. Example to quote:**
> "'ace' is a subsequence of 'abcde', but 'aec' is not because 'e' comes after 'c' there."

**Seen in:** `main.py`, `is_subsequence`

## max() with key=  `(Python built-in)`
**1. Definition:** `max(iterable, key=func)` returns the item whose `func(item)` is largest.
**2. In plain words:** `max(counts, key=counts.get)` iterates the dict's keys and picks the one with the highest count: the majority element.
**3. Example to quote:**
> "`max(counts, key=counts.get)` returns 2 for `[2, 2, 1, 1, 2]`, because 2 has the highest count."

**Seen in:** `main.py`, `majority_element`

## Majority element  `(concept)`
**1. Definition:** The majority element is the value appearing more than n/2 times in an array.
**2. In plain words:** Counting with a dict is O(n) time and O(n) space. A voting trick (Boyer-Moore) does it in O(1) space; the file mentions it as something to look up later.
**3. Example to quote:**
> "In `[2, 2, 1, 1, 2]`, 2 appears 3 times out of 5, more than half, so it is the majority."

**Seen in:** `main.py`, `majority_element`

## Normalising before comparing (`lambda` helper)  `(syntax)`
**1. Definition:** Sorting the groups and their members into a canonical order so two results can be compared regardless of ordering.
**2. In plain words:** Groups from a dict may come out in different orders. The test uses `lambda groups: sorted(sorted(g) for g in groups)` to compare them fairly.
**3. Example to quote:**
> "I sort inside each group and then sort the groups, so my two solutions compare equal even if they list groups in a different order."

**Seen in:** `main.py`, test section
