# Step 02 — Hash-map basics: Two Sum, Contains Duplicate, Valid Anagram

## Goal
Replace nested loops with dict/set lookups; learn *complement lookup* and *frequency counting*.

## Run it
```bash
python main.py
.venv/bin/python 02_dsa_arrays_hashmaps/02_hashmap_basics/main.py
```

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
