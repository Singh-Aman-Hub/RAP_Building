# 02 — Arrays, Dictionaries & Strings (LeetCode track)

## Why this matters
The interview explicitly lists LeetCode on arrays and dictionaries. Interviewers watch *how* you think: clarify the problem, try brute force, spot the pattern, code it cleanly, test edge cases, state complexity. This module is a **daily 30–45 minute track** that runs in parallel with all other modules.

## Where this fits
**Builds on:** [../00_python_foundations/](../00_python_foundations/) — lists, dicts, sets, loops, functions, comprehensions; [../01_debugging_and_environment/](../01_debugging_and_environment/) — reading tracebacks when your solution crashes, `assert`-based tests.

**Leads to:**
- [../03_ml_foundations/](../03_ml_foundations/) — numpy arrays are "arrays" with vectorised operations; Counter/dict thinking returns in tokenization (word → id vocabularies, [../04_llm_fundamentals/](../04_llm_fundamentals/)).
- [../08_agent_harness/](../08_agent_harness/) — tool registries are dicts (`{name: function}`), conversation history is a list, and trimming history is a sliding window.
- [../09_api_development/](../09_api_development/) — JSON bodies are dicts/lists; validation = edge-case thinking.
- Time & space complexity vocabulary reappears in [../06_llm_training_and_fast_inference/](../06_llm_training_and_fast_inference/) (latency/throughput/memory) and [../10_ml_system_architecture/](../10_ml_system_architecture/) (scalability).

## Concept map
```
                 Time & space complexity (01)
                            |
        ----------------------------------------
        |                                      |
  Arrays (03)  ---- traversal, min/max, search      Hash maps / sets (02)
        |                                      |  key-value, counting, lookup
        |                                      |
  Patterns on arrays                     Patterns using maps
  - two pointers (04)                    - complement lookup (Two Sum)
  - sliding window (05)  <---- window + dict ----> longest unique substring
  - prefix sums (06)     <---- prefix + dict ----> subarray sum = k
  - in-place / sorting (07)              - grouping by key (08)
                                         - frequency counting (08)
                            |
                    Strings (08) = arrays of characters
```

## Study path
| # | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 01 | [01_complexity_and_dry_runs](01_complexity_and_dry_runs/) | Big-O by measuring, dry runs | `python main.py` | 45 min |
| 02 | [02_hashmap_basics](02_hashmap_basics/) | Two Sum, Contains Duplicate, Valid Anagram | `python main.py` | 90 min |
| 03 | [03_array_scans](03_array_scans/) | min/max, search, binary search, buy/sell stock | `python main.py` | 75 min |
| 04 | [04_two_pointers](04_two_pointers/) | reverse, palindrome, sorted Two Sum, max area, 3Sum | `python main.py` | 90 min |
| 05 | [05_sliding_window](05_sliding_window/) | fixed + variable windows | `python main.py` | 90 min |
| 06 | [06_prefix_sums](06_prefix_sums/) | range sums, subarray sum = k, product except self | `python main.py` | 90 min |
| 07 | [07_in_place_and_sorting](07_in_place_and_sorting/) | move zeroes, rotate, sorting | `python main.py` | 75 min |
| 08 | [08_grouping_and_strings](08_grouping_and_strings/) | group anagrams, top-k, string basics | `python main.py` | 90 min |

Follow the day-by-day schedule in [DSA_PRACTICE_PLAN.md](DSA_PRACTICE_PLAN.md) and use [DRY_RUN_TEMPLATE.md](DRY_RUN_TEMPLATE.md) for every problem.

## The problem-solving routine (use it every time)
1. Restate the problem; ask about empty input, duplicates, negatives, sorted-or-not.
2. Write a tiny example and solve it by hand (dry run).
3. Brute force first — say its complexity.
4. Ask "what am I re-computing or searching for?" → set/dict, pointer, window or prefix.
5. Code it; 6. test edge cases; 7. state time/space.

## Pattern chooser
| Clue in the problem | Try |
|---|---|
| "find two numbers/pair that…", "seen before?" | dict/set (complement lookup) |
| sorted array, pair/triplet | two pointers |
| "contiguous subarray/substring", size k or longest/shortest | sliding window |
| "sum of range", "subarray sum equals k" | prefix sums (+ dict) |
| "in place", O(1) extra space | read/write pointers |
| "group", "count", "most frequent" | dict / Counter / defaultdict |
| sorted + "find" | binary search |

## Explanations per syllabus bullet
- **Traversal/iteration:** `for x in nums`, `for i in range(len(nums))`, `for i, x in enumerate(nums)` — see 03.
- **Searching:** linear O(n) vs binary O(log n) (sorted only) — 03.
- **Min/max:** running variable seeded with first element — 03.
- **Two pointers / sliding window / prefix sums / in-place / sorting basics / duplicate detection:** steps 04–07, 02.
- **Key-value ops, frequency counting, lookup optimisation, complement lookup, grouping, counting occurrences, dict & set usage:** steps 02, 06, 08.
- **Strings, loops, list operations:** 08 and [../00_python_foundations/](../00_python_foundations/).
- **Complexity, brute vs optimised, edge cases & dry runs:** every step; origin in 01.

## Common mistakes & gotchas
- Using `max()`/`sorted()` when the interviewer wants you to implement it — ask.
- `[[]] * n` creates n references to one list.
- Modifying a list while looping over it.
- Off-by-one in `range`, slices (`a[i:j]` excludes j) and binary search bounds.
- Forgetting empty input; returning `None` vs `[]` inconsistently.
- Binary search on unsorted data; sliding window with negative numbers.
- Forgetting that `list.sort()` returns `None`.

## Self-check
- [ ] Traversal, searching, min/max written by hand
- [ ] Two pointers, sliding window, prefix sums, in-place edits
- [ ] Sorting basics (insertion/merge idea, `sorted` key=)
- [ ] Duplicate detection (set), frequency counting (dict/Counter), complement lookup, grouping
- [ ] Two Sum brute + dict, Contains Duplicate set, Valid Anagram counting — complexity explained
- [ ] Tested empty/duplicate/boundary cases for each

## Interview questions
1. **Why is a dict lookup O(1)?** It hashes the key to a bucket position; on average one step, regardless of size. *Trade-off:* worst case O(n) with many collisions; uses extra memory; keys must be hashable.
2. **Two Sum — brute vs optimal?** Brute O(n²)/O(1); dict O(n)/O(n). Time traded for space.
3. **Set vs list for membership?** Set O(1) avg, list O(n); set loses order/duplicates and needs hashable items.
4. **When does binary search apply?** Sorted (or monotonic) data; O(log n). Sorting first costs O(n log n), so it only pays off for many queries.
5. **Sliding window vs prefix sum?** Window: contiguous with a monotonic rule (non-negative). Prefix + dict: handles negatives for "sum equals k".
6. **What's stable sorting?** Equal items keep their original relative order. Python's Timsort is stable.
7. **Why start min/max from the first element?** Avoids wrong answers on all-negative/all-large inputs.
8. **What edge cases do you test?** Empty, single, duplicates, negatives, all equal, already sorted, max size.
9. **O(n) time but O(n) space — is that okay?** Depends: ask constraints; often yes. Mention the O(1)-space alternative (e.g. two pointers on sorted input).
10. **Valid Anagram O(n) space?** O(k) where k is alphabet size (26 lowercase letters) — constant for a fixed alphabet.
11. **Why can't a list be a dict key?** Lists are mutable and unhashable; use a tuple.
12. **How do you dry-run?** Table of variable values per loop iteration; check on 1-element and empty inputs.
