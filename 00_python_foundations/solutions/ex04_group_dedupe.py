"""Ex 04: group words by length; dedupe keeping first-seen order."""
words = ["the", "cat", "sat", "on", "the", "mat"]

groups = {}
for w in words:
    groups.setdefault(len(w), []).append(w)       # setdefault: create the list if missing
print(groups)                                     # {3: [...], 2: ['on']}

seen, unique = set(), []
for w in words:
    if w not in seen:                             # O(1) membership check
        seen.add(w)
        unique.append(w)
print(unique)
assert unique == ["the", "cat", "sat", "on", "mat"]
