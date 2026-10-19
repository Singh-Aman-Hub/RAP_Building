"""
GOAL: Solve the 3 classic hash-map problems (Two Sum, Contains Duplicate, Valid Anagram)
      first by brute force, then with a dict/set, and prove both agree.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/02_hashmap_basics/main.py
EXPECTED OUTPUT: lines for each problem and "All hash-map tests passed."
NEEDS: standard library only (collections.Counter).
"""

from collections import Counter  # Counter(iterable) -> dict-like {item: how_many_times}


# ===========================================================================
# 1. Two Sum: return indices of two numbers that add up to target.
# ===========================================================================
def two_sum_brute(nums, target):
    """Try every pair.  Time O(n^2), Space O(1)."""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):      # j > i so we never reuse the same element
            if nums[i] + nums[j] == target:
                return [i, j]
    return []                                   # no pair found


def two_sum_dict(nums, target):
    """One pass with a dict {value_seen: index}.  Time O(n), Space O(n).

    Idea ("complement lookup"): for each x, the partner we need is target - x.
    If we've already seen that partner, we're done; otherwise remember x.
    """
    seen = {}                                   # value -> index where we saw it
    for i, x in enumerate(nums):
        complement = target - x
        if complement in seen:                  # O(1) average dict lookup
            return [seen[complement], i]
        seen[x] = i                             # store AFTER checking so x can't pair with itself
    return []


# ===========================================================================
# 2. Contains Duplicate: does any value appear at least twice?
# ===========================================================================
def contains_duplicate_brute(nums):
    """Compare every pair.  O(n^2) time, O(1) space."""
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j]:
                return True
    return False


def contains_duplicate_sort(nums):
    """Sort then compare neighbours.  O(n log n) time, O(n) space here (sorted() copies)."""
    ordered = sorted(nums)
    return any(a == b for a, b in zip(ordered, ordered[1:]))  # zip pairs neighbours


def contains_duplicate_set(nums):
    """Set remembers what we've seen.  O(n) time, O(n) space.  Shortcut: len(set(nums)) != len(nums)."""
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False


# ===========================================================================
# 3. Valid Anagram: same letters, same counts?
# ===========================================================================
def is_anagram_sort(s, t):
    """Sort both and compare.  O(n log n) time."""
    return sorted(s) == sorted(t)


def is_anagram_count(s, t):
    """Frequency counting with a hand-written dict.  O(n) time, O(k) space (k = alphabet size)."""
    if len(s) != len(t):                        # different length can never match -> early exit
        return False
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1      # dict.get(key, default) avoids KeyError
    for ch in t:
        if counts.get(ch, 0) == 0:              # t has a letter s doesn't have (or too many of it)
            return False
        counts[ch] -= 1
    return True


def is_anagram_counter(s, t):
    """Same idea in one line using collections.Counter."""
    return Counter(s) == Counter(t)


# ===========================================================================
# Tests: edge cases first!
# ===========================================================================
if __name__ == "__main__":                       # runs only when executed directly, not on import
    cases = [([2, 7, 11, 15], 9), ([3, 2, 4], 6), ([3, 3], 6), ([1, 2], 10), ([], 5), ([5], 5)]
    for nums, target in cases:
        a, b = two_sum_brute(nums, target), two_sum_dict(nums, target)
        print(f"two_sum({nums}, {target}) brute={a} dict={b}")
        assert a == b

    for nums in ([1, 2, 3, 1], [1, 2, 3, 4], [], [7], [5, 5]):
        results = {contains_duplicate_brute(nums), contains_duplicate_sort(nums), contains_duplicate_set(nums)}
        print(f"contains_duplicate({nums}) -> {results}")
        assert len(results) == 1                 # all three methods agree

    for s, t, expected in [("anagram", "nagaram", True), ("rat", "car", False), ("", "", True),
                           ("a", "ab", False), ("aab", "abb", False)]:
        assert is_anagram_sort(s, t) == is_anagram_count(s, t) == is_anagram_counter(s, t) == expected
        print(f"is_anagram({s!r}, {t!r}) -> {expected}")

    print("All hash-map tests passed.")
    # TRY IT: add a case with negative numbers to two_sum. Does the dict version still work? Why?
    # BREAK IT: move `seen[x] = i` ABOVE the `if` in two_sum_dict, then run two_sum([3, 2, 4], 6)... what goes wrong?
