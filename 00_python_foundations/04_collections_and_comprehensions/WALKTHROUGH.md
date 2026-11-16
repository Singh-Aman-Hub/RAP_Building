# Walkthrough 04 — Lists, tuples, dicts, sets, comprehensions

## Goal
Choose the right container and build one in a single expression. Dicts and sets power module [02 LeetCode](../../02_dsa_arrays_hashmaps/) and every JSON payload.

## Run it
```bash
cd 00_python_foundations/04_collections_and_comprehensions
python main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| dict | hash map, hash table, associative array, mapping | Same idea | Python's `dict` is a hash table with insertion order kept. |
| list vs tuple | mutable vs immutable sequence | No | Lists can change; tuples cannot (and can be dict keys). |
| set vs frozenset | mutable vs immutable set | No | `frozenset` is hashable; `set` is not. |
| comprehension vs generator expression | `[...]` vs `(...)` | No | The list builds everything now; the generator yields items on demand. |
| `sorted()` vs `.sort()` | new list vs in place | No | `.sort()` returns `None`. |
| `b = a` vs `a.copy()` | alias vs shallow copy | No | Alias shares the object; copy makes a new top-level list (nested items still shared). |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–11)

```python
"""
GOAL:  Use lists, tuples, dicts and sets, and build them with comprehensions.

RUN:   cd 00_python_foundations/04_collections_and_comprehensions && python main.py

EXPECTED OUTPUT (first lines):
    [0, 1, 4, 9]
    ...

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** The file's label.
- **Exactly:** Module docstring; no runtime effect.
- **Why it is here:** Orients the reader.

### 2. List: append, insert, slice, pop, membership  (`main.py` lines 13–20)

```python
# ---- list: ordered, changeable --------------------------------------------
tokens = ["the", "cat", "sat"]
tokens.append("down")            # add to end
tokens.insert(0, "<s>")          # insert at index 0
print(tokens, len(tokens))
print(tokens[1:3])               # slice [start:stop) -> ['the', 'cat']
tokens.pop()                     # remove & return last item
print("the" in tokens)           # membership test: True (O(n) for lists!)
```

- **In simple words:** A list is an ordered row of boxes you can add to, remove from and cut pieces out of.
- **Exactly:** Start `['the','cat','sat']`; `append('down')` adds at the end; `insert(0,'<s>')` puts at index 0 -> `['<s>', 'the', 'cat', 'sat', 'down']`, length 5. `tokens[1:3]` -> `['the', 'cat']`. `pop()` removes and returns the last item (`'down'`, ignored here). `"the" in tokens` -> `True`, a linear O(n) scan.
- **Why it is here:** Lists are the default sequence type; the O(n) membership cost motivates sets and dicts later.

### 3. Tuple and unpacking  (`main.py` lines 22–25)

```python
# ---- tuple: ordered, NOT changeable (like a fixed record) ------------------
shape = (2, 3)
rows, cols = shape               # unpack
print(rows, cols)
```

- **In simple words:** A tuple is a fixed record: you can read it but not change it.
- **Exactly:** `shape = (2, 3)`; `rows, cols = shape` -> `rows=2, cols=3`. Output `2 3`. Trying `shape[0] = 5` would raise `TypeError`.
- **Why it is here:** Fixed shapes, coordinates and multiple return values are tuples; immutability also makes them usable as dict keys.

### 4. Dict: add, lookup, get, iterate, delete  (`main.py` lines 27–35)

```python
# ---- dict: key -> value lookup (a "hash map") ----------------------------
vocab = {"the": 0, "cat": 1}
vocab["sat"] = 2                         # add / overwrite
print(vocab["cat"])                      # lookup by key; KeyError if missing
print(vocab.get("dog", -1))              # .get(key, default) never raises -> -1
for word, idx in vocab.items():          # iterate key/value pairs
    print(word, "->", idx)
print(list(vocab.keys()), list(vocab.values()))
del vocab["the"]                         # delete a key
```

- **In simple words:** A dict is a phone book: look up a value by its key, instantly.
- **Exactly:** `vocab = {'the':0,'cat':1}`; `vocab['sat']=2` adds. `vocab['cat']` -> `1` (missing key -> `KeyError`). `.get('dog', -1)` -> `-1` without error. `.items()` yields `(key, value)` pairs: `the -> 0`, `cat -> 1`, `sat -> 2`. `.keys()`/`.values()` -> `['the','cat','sat'] [0,1,2]`. `del vocab['the']` removes a key.
- **Why it is here:** Token-to-id vocabularies, config and counting are all dicts. Average lookup is O(1).

### 5. Set: unique items and set math  (`main.py` lines 37–42)

```python
# ---- set: unique items, super-fast membership ---------------------------
seen = {1, 2, 2, 3}                      # duplicates vanish -> {1, 2, 3}
seen.add(4)
print(seen, 3 in seen)                   # `in` on a set is O(1) on average
print({1, 2, 3} & {2, 3, 4})             # intersection -> {2, 3}
print({1, 2} | {2, 5})                   # union
```

- **In simple words:** A set is a bag that keeps only one of each item and answers "is it in here?" very fast.
- **Exactly:** `{1,2,2,3}` -> `{1,2,3}`; `add(4)` -> `{1,2,3,4}`; `3 in seen` -> `True` (average O(1)). `&` intersection -> `{2,3}`; `|` union -> `{1,2,5}`. Sets are unordered.
- **Why it is here:** Deduplicating and fast membership checks are everywhere in text processing and interviews.

### 6. List, dict and set comprehensions  (`main.py` lines 44–55)

```python
# ---- comprehensions: build a collection in one expression ------------------
squares = [n * n for n in range(4)]                 # [expression for item in iterable]
print(squares)

evens = [n for n in range(10) if n % 2 == 0]        # ... with a filter
print(evens)

lengths = {w: len(w) for w in ["the", "cat", "sat"]}  # dict comprehension
print(lengths)

unique_lengths = {len(w) for w in ["a", "bb", "cc"]}  # set comprehension
print(unique_lengths)
```

- **In simple words:** A one-line recipe: for each item, build something, optionally only if a condition holds.
- **Exactly:** `[n*n for n in range(4)]` -> `[0,1,4,9]`; with `if n % 2 == 0` over `range(10)` -> `[0,2,4,6,8]`. Dict comprehension -> `{'the':3,'cat':3,'sat':3}`; set comprehension of `len` over `['a','bb','cc']` -> `{1,2}` (duplicate 2 collapses).
- **Why it is here:** Shorter and usually faster than append-in-a-loop; the dominant Python idiom.

### 7. Generator expression  (`main.py` lines 57–59)

```python
# Generator expression: like a list comprehension but lazy (no list built in memory)
total = sum(n * n for n in range(1000))
print("sum of squares:", total)
```

- **In simple words:** Like a list comprehension in round brackets, but it hands out items one at a time instead of storing them all.
- **Exactly:** `sum(n*n for n in range(1000))` -> `332833500`. No list of 1000 numbers is built; `sum` pulls values lazily.
- **Why it is here:** Saves memory on large data; same idea as streaming datasets.

### 8. sorted vs sort  (`main.py` lines 61–65)

```python
# ---- sorting and copying -------------------------------------------------
nums = [3, 1, 2]
print(sorted(nums), nums)       # sorted() returns a NEW list; original unchanged
nums.sort(reverse=True)         # .sort() changes the list IN PLACE and returns None
print(nums)
```

- **In simple words:** `sorted()` gives you a sorted copy; `.sort()` rearranges the list itself.
- **Exactly:** `sorted(nums)` -> `[1, 2, 3]` while `nums` stays `[3, 1, 2]`. `nums.sort(reverse=True)` changes `nums` to `[3, 2, 1]` and returns `None`.
- **Why it is here:** Writing `nums = nums.sort()` is a bug: it sets `nums` to `None`.

### 9. Aliasing versus copying  (`main.py` lines 67–73)

```python
a = [1, 2]
b = a                           # NOT a copy: both names point to the same list
b.append(3)
print("a after b.append:", a)   # [1, 2, 3]
c = a.copy()                    # real (shallow) copy
c.append(99)
print("a still:", a, "| c:", c)
```

- **In simple words:** `b = a` gives the same list a second name; `a.copy()` makes a separate list.
- **Exactly:** After `b.append(3)`, `a` is `[1, 2, 3]` because `a` and `b` are the same object. `c = a.copy()` is a shallow copy; `c.append(99)` leaves `a` as `[1, 2, 3]` and `c` becomes `[1, 2, 3, 99]`.
- **Why it is here:** Unexpected shared mutation is a top source of bugs, especially when passing lists to functions.

### 10. Counting with a dict  (`main.py` lines 75–79)

```python
# ---- counting: the pattern behind LeetCode "frequency counting" ----------
counts = {}
for ch in "banana":
    counts[ch] = counts.get(ch, 0) + 1   # default 0 then add one
print(counts)
```

- **In simple words:** Walk through the letters and keep a tally for each in a dict.
- **Exactly:** `counts.get(ch, 0) + 1` reads the current count (default 0) and adds one. For `"banana"` -> `{'b': 1, 'a': 3, 'n': 2}` (insertion order).
- **Why it is here:** The frequency-counting pattern from LeetCode (anagrams, top-k); `collections.Counter` does the same.

### 11. TRY IT and BREAK IT comments  (`main.py` lines 81–82)

```python
# TRY IT: turn `evens` into a comprehension of evens squared.
# BREAK IT: print(vocab["dog"]) -> KeyError. Then fix with .get().
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** `print(vocab['dog'])` -> `KeyError: 'dog'`; fix with `.get('dog')`.
- **Why it is here:** Practice the most common dict error.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `[a, b]` | list (ordered, changeable) | `tokens = ["the", "cat"]` |
| `(a, b)` | tuple (ordered, fixed) | `shape = (2, 3)` |
| `{k: v}` | dict (key -> value) | `{"the": 0}` |
| `{a, b}` | set (unique, fast `in`) | `{1, 2, 3}` |
| `xs[1:3]` | slice | `["the","cat"]` |
| `x in container` | membership test | `3 in seen` |
| `[e for x in xs if cond]` | list comprehension | `[n*n for n in range(4)]` |
| `{k: v for ...}` / `{e for ...}` | dict / set comprehension | `{w: len(w) for w in ws}` |
| `(e for x in xs)` | generator expression (lazy) | `sum(n*n for n in range(1000))` |
| `a & b`, `a \| b` | set intersection / union | `{1,2,3} & {2,3,4}` |

## Libraries & functions
| Name | Built-in | What it does | Returns |
|---|---|---|---|
| `list.append/insert/pop` | yes | add at end / at index / remove last | `None`, `None`, the item |
| `dict.get(k, default)` | yes | safe lookup | value or default |
| `dict.items()/keys()/values()` | yes | views for iteration | view objects |
| `sorted(x)` vs `x.sort()` | yes | new sorted list vs in-place sort | list vs `None` |
| `list.copy()` | yes | shallow copy | new list |

## Execution trace (real output)
Counting letters (`counts[ch] = counts.get(ch, 0) + 1`) in `"banana"`: b->1, a->1,2,3, n->1,2 ...
```
{'b': 1, 'a': 3, 'n': 2}
```
Aliasing: `b = a; b.append(3)` changes `a` too -> `a after b.append: [1, 2, 3]`; `c = a.copy()` does not.

## Experiments
1. Make `evens` into `[n * n for n in range(10) if n % 2 == 0]` -> `[0, 4, 16, 36, 64]`.
2. Use a list instead of a set for `seen` and time `3 in seen` on 1,000,000 items vs a set (`time.perf_counter`).
3. Replace `"banana"` with your name and read the count dictionary.

## Debugging practice
`print(vocab["dog"])` -> `KeyError: 'dog'`. Fix with `vocab.get("dog", -1)` (returns `-1`) or check `if "dog" in vocab`.

## Rebuild checklist
- [ ] Count characters of a string into a dict without `Counter`
- [ ] Dedupe a list with a set while keeping first-seen order (hint: loop + `seen` set)
- [ ] Write one list, one dict and one set comprehension

## Say it aloud
"A list is an ordered collection, a tuple is the immutable version, a dict maps keys to values with fast lookup, and a set stores unique items with fast membership checks. Comprehensions build these in one readable expression. Dict and set lookups are O(1) on average, while `in` on a list is O(n) — that is the core trick behind Two Sum."
Trade-off: dicts/sets use extra memory in exchange for speed, and set elements/dict keys must be hashable (no lists).
