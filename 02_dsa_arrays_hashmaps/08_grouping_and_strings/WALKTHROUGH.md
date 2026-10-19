# Step 08 — Grouping, counting and strings

## Goal
Use dicts to group items by a *key function*, count frequencies, and solve common string problems.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/08_grouping_and_strings/main.py
```

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
