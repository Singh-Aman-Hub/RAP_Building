"""
GOAL:  Use lists, tuples, dicts and sets, and build them with comprehensions.

RUN:   cd 00_python_foundations/04_collections_and_comprehensions && python main.py

EXPECTED OUTPUT (first lines):
    [0, 1, 4, 9]
    ...

NEEDS: nothing (standard library only), no internet.
"""

# ---- list: ordered, changeable --------------------------------------------
tokens = ["the", "cat", "sat"]
tokens.append("down")            # add to end
tokens.insert(0, "<s>")          # insert at index 0
print(tokens, len(tokens))
print(tokens[1:3])               # slice [start:stop) -> ['the', 'cat']
tokens.pop()                     # remove & return last item
print("the" in tokens)           # membership test: True (O(n) for lists!)

# ---- tuple: ordered, NOT changeable (like a fixed record) ------------------
shape = (2, 3)
rows, cols = shape               # unpack
print(rows, cols)

# ---- dict: key -> value lookup (a "hash map") ----------------------------
vocab = {"the": 0, "cat": 1}
vocab["sat"] = 2                         # add / overwrite
print(vocab["cat"])                      # lookup by key; KeyError if missing
print(vocab.get("dog", -1))              # .get(key, default) never raises -> -1
for word, idx in vocab.items():          # iterate key/value pairs
    print(word, "->", idx)
print(list(vocab.keys()), list(vocab.values()))
del vocab["the"]                         # delete a key

# ---- set: unique items, super-fast membership ---------------------------
seen = {1, 2, 2, 3}                      # duplicates vanish -> {1, 2, 3}
seen.add(4)
print(seen, 3 in seen)                   # `in` on a set is O(1) on average
print({1, 2, 3} & {2, 3, 4})             # intersection -> {2, 3}
print({1, 2} | {2, 5})                   # union

# ---- comprehensions: build a collection in one expression ------------------
squares = [n * n for n in range(4)]                 # [expression for item in iterable]
print(squares)

evens = [n for n in range(10) if n % 2 == 0]        # ... with a filter
print(evens)

lengths = {w: len(w) for w in ["the", "cat", "sat"]}  # dict comprehension
print(lengths)

unique_lengths = {len(w) for w in ["a", "bb", "cc"]}  # set comprehension
print(unique_lengths)

# Generator expression: like a list comprehension but lazy (no list built in memory)
total = sum(n * n for n in range(1000))
print("sum of squares:", total)

# ---- sorting and copying -------------------------------------------------
nums = [3, 1, 2]
print(sorted(nums), nums)       # sorted() returns a NEW list; original unchanged
nums.sort(reverse=True)         # .sort() changes the list IN PLACE and returns None
print(nums)

a = [1, 2]
b = a                           # NOT a copy: both names point to the same list
b.append(3)
print("a after b.append:", a)   # [1, 2, 3]
c = a.copy()                    # real (shallow) copy
c.append(99)
print("a still:", a, "| c:", c)

# ---- counting: the pattern behind LeetCode "frequency counting" ----------
counts = {}
for ch in "banana":
    counts[ch] = counts.get(ch, 0) + 1   # default 0 then add one
print(counts)

# TRY IT: turn `evens` into a comprehension of evens squared.
# BREAK IT: print(vocab["dog"]) -> KeyError. Then fix with .get().
