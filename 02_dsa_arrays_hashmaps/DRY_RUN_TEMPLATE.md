# Dry-run template (copy for every problem)

**Problem:** (one sentence)
**Input / output example:**
**Edge cases to test:** empty • single • duplicates • negatives • all equal • already sorted • very large

## 1. Brute force
Idea:
Time: O( ) Space: O( )

## 2. Pattern spotted
(dict/set • two pointers • sliding window • prefix sums • in-place • sort)
Why:

## 3. Dry run table (use the example input)
| step | i / left / right | value(s) | state variables (seen / total / best …) | decision |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |

## 4. Code (write without looking at solutions)

## 5. Tests (assert)
```python
assert func(...) == ...
```

## 6. Final complexity
Time: O( ) Space: O( ) — why?

## Worked example: Two Sum, nums=[2,7,11,15], target=9
| step | i | x | complement | seen | decision |
|---|---|---|---|---|---|
| 0 | 0 | 2 | 7 | {} | store 2→0 |
| 1 | 1 | 7 | 2 | {2:0} | 2 in seen → return [0,1] |
