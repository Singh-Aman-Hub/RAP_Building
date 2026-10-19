# Walkthrough 04 — Lists, tuples, dicts, sets, comprehensions

## Goal
Choose the right container and build one in a single expression. Dicts and sets power module [02 LeetCode](../../02_dsa_arrays_hashmaps/) and every JSON payload.

## Run it
```bash
cd 00_python_foundations/04_collections_and_comprehensions
python main.py
```

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
