# Exercises — module 02

Do one task at a time, run `python main.py` after each change. Worked answers live in [solutions/](solutions/).

| Step | Modify it yourself | Break it & read the traceback | Rebuild from memory | Explain aloud |
|---|---|---|---|---|
| 01 | Add a third timing: `dict` lookup. Rank list/set/dict. | Pass an int to `sum_with_trace`. | `time_it` + nested-loop counter | "Why is a set lookup O(1)?" |
| 02 | Return *all* index pairs for Two Sum (see solutions/ex02). Make `is_anagram` case-insensitive. | Delete the `len(s) != len(t)` check; find an input that gives a wrong answer. | Two Sum dict (5 min, no peeking) | Brute vs optimised, with complexity |
| 03 | Write `find_min` and `third_largest`. | `find_max(None)` → read the `TypeError`. | `binary_search` | Why sorted input? |
| 04 | Valid Palindrome II (may delete one char). | Swap `<` for `<=` in two_sum_sorted. | `reverse_in_place` | Why move the shorter wall? |
| 05 | Longest substring with at most 2 distinct chars (solutions/ex05). | `max_sum_k([1,2,3], 0)` | fixed window sum | Window vs nested loops |
| 06 | Count subarrays with sum divisible by k (hard stretch). 2D? skip. | `query(0, len(nums))` → IndexError | Subarray Sum = K | Why `{0: 1}`? |
| 07 | Remove Element (in place), Sort Colors (Dutch flag) (solutions/ex07). | `rotate_right(None, 1)` | `move_zeroes` | read/write pointers |
| 08 | Valid Anagram variant: group by *length*; Ransom Note. | Use `{}` not `defaultdict` | Group Anagrams | Why a tuple key not a list? |

## Cumulative checks
1. Solve Two Sum, Contains Duplicate, Valid Anagram with a timer: 10 minutes total.
2. For each, write its complexity next to the code.
3. Run edge cases: empty, one element, duplicates, negatives.
