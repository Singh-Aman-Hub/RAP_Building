# Terms — Collections and Comprehensions

Folder: `00_python_foundations/04_collections_and_comprehensions` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [00 Python foundations README](../README.md)

**Terms in this folder:** [list](#list) · [list.append / insert / pop](#listappend--insert--pop) · [Slice](#slice) · [Membership test (in)](#membership-test-in) · [tuple](#tuple) · [dict](#dict) · [dict.get()](#dictget) · [dict.items / keys / values](#dictitems--keys--values) · [del](#del) · [KeyError](#keyerror) · [set](#set) · [Set operations](#set-operations--and-) · [List comprehension](#list-comprehension) · [Comprehension with filter](#comprehension-with-filter) · [Dict comprehension](#dict-comprehension) · [Set comprehension](#set-comprehension) · [Generator expression](#generator-expression) · [sorted() vs list.sort()](#sorted-vs-listsort) · [Aliasing vs copy](#aliasing-vs-copy) · [Frequency counting](#frequency-counting) · [Hash map and hashability](#hash-map-and-hashability)

---

## list  `(Python built-in)`
**1. Definition:** A `list` is a built-in ordered, mutable sequence written `[a, b, c]` that can hold items of any type.
**2. In plain words:** A list is a row of boxes you can add to, remove from and reorder. Order is kept, and duplicates are allowed. It's the default container for things like tokens. Searching it with `in` is slow for large lists because Python checks items one by one.
**3. Example to quote:**
> "I keep my tokens in a list, `['the', 'cat', 'sat']`, because order matters for a sentence."
```python
tokens = ["the", "cat", "sat"]
print(len(tokens))   # 3
```
**Seen in:** `main.py`, list section

## list.append / insert / pop  `(Python built-in)`
**1. Definition:** `append(x)` adds `x` at the end, `insert(i, x)` adds it at index `i`, and `pop()` removes and returns the last item (or the one at a given index).
**2. In plain words:** These are the basic edits. `append` and `pop` at the end are fast; `insert(0, ...)` shifts everything and is slower for big lists. `append` and `insert` return `None`, so don't write `tokens = tokens.append(...)`.
**3. Example to quote:**
> "I inserted `'<s>'` at index 0 to mark the start of the sentence, and `pop()` removed the last token `'down'`."
```python
t = ["a"]; t.append("b"); t.insert(0, "<s>"); t.pop()
```
**Seen in:** `main.py`, list section

## Slice  `(syntax)`
**1. Definition:** `xs[start:stop]` creates a new list containing items from index `start` up to but excluding `stop`.
**2. In plain words:** It lets you cut out a piece of a sequence. `tokens[1:3]` gives the items at positions 1 and 2. Leaving a side empty means "from the beginning" or "to the end".
**3. Example to quote:**
> "`tokens[1:3]` returned `['the', 'cat']`, positions 1 and 2, with the stop index excluded."
```python
print(["<s>", "the", "cat", "sat"][1:3])   # ['the', 'cat']
```
**Seen in:** `main.py`, `tokens[1:3]`

## Membership test (in)  `(syntax)`
**1. Definition:** `x in container` evaluates to `True` if `x` is an item of the container (a key for a dict).
**2. In plain words:** It's the readable way to ask "is this here?". On a list Python scans item by item (O(n)); on a set or dict it uses hashing (about O(1)). That difference is why Two Sum uses a dict.
**3. Example to quote:**
> "`'the' in tokens` returned `True`, but on a million-item list I'd use a set because list membership is O(n)."
```python
print(3 in {1, 2, 3})   # True
```
**Seen in:** `main.py`, list and set sections

## tuple  `(Python built-in)`
**1. Definition:** A `tuple` is a built-in ordered, immutable sequence written `(a, b)`.
**2. In plain words:** A tuple is like a list that can't be changed, so it's good for fixed records such as `shape = (2, 3)`. You can unpack it: `rows, cols = shape`. Because it can't change, it can be used as a dict key.
**3. Example to quote:**
> "`shape = (2, 3)` is a tuple, and `rows, cols = shape` gave me `rows = 2` and `cols = 3`."
```python
rows, cols = (2, 3)
```
**Seen in:** `main.py`, tuple section

## dict  `(Python built-in)`
**1. Definition:** A `dict` is a built-in mapping from unique, hashable keys to values, written `{key: value}`, with average O(1) lookup, insert and delete.
**2. In plain words:** A dict is a phone book: look up a name (key) and get the number (value) instantly. In NLP, a vocabulary maps each word to an id. JSON objects and request bodies in APIs are dicts too. Assigning to a new key adds it; assigning to an existing key overwrites it.
**3. Example to quote:**
> "My `vocab = {'the': 0, 'cat': 1}` maps words to ids, and `vocab['sat'] = 2` added a new entry."
```python
vocab = {"the": 0}
vocab["cat"] = 1
```
**Seen in:** `main.py`, dict section

## dict.get()  `(Python built-in)`
**1. Definition:** `d.get(key, default)` returns `d[key]` if the key exists, otherwise `default` (or `None`), without raising an error.
**2. In plain words:** Plain `d["dog"]` crashes with a `KeyError` if "dog" is missing. `.get("dog", -1)` quietly returns -1 instead. It's the key ingredient of the counting pattern.
**3. Example to quote:**
> "`vocab.get('dog', -1)` returned -1 for an unknown word instead of crashing."
```python
print({"a": 1}.get("b", 0))   # 0
```
**Seen in:** `main.py`, dict section and counting section

## dict.items / keys / values  `(Python built-in)`
**1. Definition:** `items()`, `keys()` and `values()` return live views over a dict's (key, value) pairs, keys, or values.
**2. In plain words:** Use `for word, idx in vocab.items():` to walk through both at once. Wrap a view in `list(...)` if you need an actual list. The views reflect later changes to the dict.
**3. Example to quote:**
> "I looped with `for word, idx in vocab.items()` to print each word and its id."
```python
for k, v in {"a": 1}.items():
    print(k, v)
```
**Seen in:** `main.py`, dict section

## del  `(syntax)`
**1. Definition:** `del d[key]` removes a key and its value from a dict (and `del xs[i]` removes a list item).
**2. In plain words:** It's a statement for deleting. Deleting a missing key raises `KeyError`, so check first or use `d.pop(key, None)`.
**3. Example to quote:**
> "`del vocab['the']` removed the entry for 'the'."
```python
d = {"a": 1}; del d["a"]
```
**Seen in:** `main.py`, dict section

## KeyError  `(Python built-in)`
**1. Definition:** `KeyError` is the exception raised when a dict key is not found.
**2. In plain words:** `print(vocab["dog"])` fails with `KeyError: 'dog'`. The message shows the missing key. Fix it by using `.get()` with a default or by checking `if "dog" in vocab` first.
**3. Example to quote:**
> "I got `KeyError: 'dog'` from `vocab['dog']` and fixed it with `vocab.get('dog', -1)`."
```python
vocab = {}
print(vocab.get("dog", -1))
```
**Seen in:** `main.py` BREAK IT; `WALKTHROUGH.md` debugging practice

## set  `(Python built-in)`
**1. Definition:** A `set` is a built-in unordered collection of unique, hashable items with average O(1) membership tests.
**2. In plain words:** Think of a guest list where nobody appears twice. `{1, 2, 2, 3}` automatically becomes `{1, 2, 3}`. Sets are ideal for removing duplicates and quick "have I seen this?" checks. They don't keep order.
**3. Example to quote:**
> "`seen = {1, 2, 2, 3}` collapsed to three items, and `3 in seen` is a fast check."
```python
seen = {1, 2, 2, 3}
seen.add(4)
```
**Seen in:** `main.py`, set section

## Set operations (& and |)  `(syntax)`
**1. Definition:** `a & b` is the intersection (items in both) and `a | b` is the union (items in either) of two sets.
**2. In plain words:** They answer "what do these two groups share?" and "what's in either?". `{1, 2, 3} & {2, 3, 4}` is `{2, 3}`. Useful for comparing token sets or finding common labels.
**3. Example to quote:**
> "`{1, 2, 3} & {2, 3, 4}` gave `{2, 3}`, the overlap of the two sets."
```python
print({1, 2} | {2, 5})   # {1, 2, 5}
```
**Seen in:** `main.py`, set section

## List comprehension  `(syntax)`
**1. Definition:** `[expression for item in iterable]` builds a new list by evaluating `expression` for every item.
**2. In plain words:** It's a compact replacement for a loop that appends to a list. `[n * n for n in range(4)]` gives `[0, 1, 4, 9]`. It's readable when short; if it needs several lines, use a regular loop.
**3. Example to quote:**
> "I built the squares with `[n * n for n in range(4)]` instead of a four-line loop."
```python
squares = [n * n for n in range(4)]
```
**Seen in:** `main.py`, comprehensions section

## Comprehension with filter  `(syntax)`
**1. Definition:** Adding `if condition` at the end, `[e for x in xs if condition]`, keeps only the items for which the condition is true.
**2. In plain words:** It combines "loop" and "keep only some". `[n for n in range(10) if n % 2 == 0]` yields the even numbers `[0, 2, 4, 6, 8]`.
**3. Example to quote:**
> "To keep only even numbers I wrote `[n for n in range(10) if n % 2 == 0]`."
```python
evens = [n for n in range(10) if n % 2 == 0]
```
**Seen in:** `main.py`, `evens`

## Dict comprehension  `(syntax)`
**1. Definition:** `{k: v for item in iterable}` builds a dict by evaluating a key and a value for each item.
**2. In plain words:** It makes a lookup table in one line. `{w: len(w) for w in ["the", "cat", "sat"]}` maps each word to its length, `{'the': 3, 'cat': 3, 'sat': 3}`.
**3. Example to quote:**
> "I built a word-to-length table with a dict comprehension in one line."
```python
lengths = {w: len(w) for w in ["the", "cat"]}
```
**Seen in:** `main.py`, `lengths`

## Set comprehension  `(syntax)`
**1. Definition:** `{expression for item in iterable}` builds a set, automatically dropping duplicates.
**2. In plain words:** It looks like a dict comprehension but with no colon. `{len(w) for w in ["a", "bb", "cc"]}` gives `{1, 2}` because the two 2s collapse into one.
**3. Example to quote:**
> "The set comprehension of word lengths returned `{1, 2}`, showing duplicates are removed."
```python
print({len(w) for w in ["a", "bb", "cc"]})
```
**Seen in:** `main.py`, `unique_lengths`

## Generator expression  `(syntax)`
**1. Definition:** `(expression for item in iterable)` creates a lazy iterator that produces one value at a time instead of building a whole list in memory.
**2. In plain words:** `sum(n * n for n in range(1000))` adds up squares without ever making a 1000-item list. That saves memory for huge sequences. You can only loop over a generator once.
**3. Example to quote:**
> "I summed a thousand squares with a generator expression so no big list was created."
```python
total = sum(n * n for n in range(1000))
```
**Seen in:** `main.py`, `total`

## sorted() vs list.sort()  `(Python built-in)`
**1. Definition:** `sorted(x)` returns a new sorted list and leaves `x` unchanged; `x.sort()` sorts the list in place and returns `None`; both accept `reverse=` and `key=`.
**2. In plain words:** A common bug is `nums = nums.sort()`, which sets `nums` to `None`. Use `sorted` when you want to keep the original, `.sort()` when you want to modify it.
**3. Example to quote:**
> "`sorted(nums)` printed `[1, 2, 3]` while `nums` was still `[3, 1, 2]`; `nums.sort(reverse=True)` changed it in place to `[3, 2, 1]`."
```python
nums = [3, 1, 2]; nums.sort(reverse=True)
```
**Seen in:** `main.py`, sorting and copying

## Aliasing vs copy  `(concept)`
**1. Definition:** `b = a` makes both names refer to the same list object, whereas `a.copy()` creates a new, independent (shallow) copy.
**2. In plain words:** Assigning a list to a new name doesn't duplicate it. After `b = a; b.append(3)`, `a` also shows `[1, 2, 3]`. Use `.copy()` when you want to change one without affecting the other. "Shallow" means nested lists inside are still shared.
**3. Example to quote:**
> "After `b = a` and `b.append(3)`, `a` became `[1, 2, 3]` because both names pointed to one list; `a.copy()` avoided that."
```python
a = [1, 2]; c = a.copy(); c.append(99)
```
**Seen in:** `main.py`, copying section

## Frequency counting  `(concept)`
**1. Definition:** Frequency counting tallies how often each item occurs, typically with a dict where `counts[item] = counts.get(item, 0) + 1`.
**2. In plain words:** Loop through the items; for each, read its current count (defaulting to 0) and add one. For "banana" the result is `{'b': 1, 'a': 3, 'n': 2}`. This pattern solves many interview problems like Valid Anagram.
**3. Example to quote:**
> "I counted letters in 'banana' with `counts.get(ch, 0) + 1` and got `{'b': 1, 'a': 3, 'n': 2}`."
```python
counts = {}
for ch in "banana":
    counts[ch] = counts.get(ch, 0) + 1
```
**Seen in:** `main.py`, counting section

## Hash map and hashability  `(concept)`
**1. Definition:** A hash map stores key-value pairs by computing a hash of each key to find its slot, giving average O(1) lookup; keys (and set items) must be hashable, meaning immutable.
**2. In plain words:** Python's `dict` and `set` are hash-based. That is why they are fast but also why a list can't be a dict key (it can change), whereas a tuple can. Trade-off: speed in exchange for extra memory.
**3. Example to quote:**
> "I use a dict for Two Sum because lookups are O(1) on average, but the keys must be hashable, so a list can't be a key."
```python
d = {(1, 2): "ok"}   # tuple key works
```
**Seen in:** `WALKTHROUGH.md`, "Say it aloud" and trade-off
