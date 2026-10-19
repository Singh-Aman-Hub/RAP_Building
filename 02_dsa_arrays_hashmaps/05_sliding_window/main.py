"""
GOAL: Learn the sliding-window pattern: keep a moving range [left, right] and update
      its answer incrementally instead of recomputing from scratch.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/05_sliding_window/main.py
EXPECTED OUTPUT: "All sliding-window tests passed."
NEEDS: standard library only.
"""


# --- Fixed-size window ------------------------------------------------------
def max_sum_k_brute(nums, k):
    """Re-add k numbers for every start index: O(n*k)."""
    if k <= 0 or k > len(nums):
        return None
    return max(sum(nums[i:i + k]) for i in range(len(nums) - k + 1))


def max_sum_k(nums, k):
    """Slide: add the new right element, subtract the element that left. O(n)/O(1)."""
    if k <= 0 or k > len(nums):
        return None
    window = sum(nums[:k])                     # first window
    best = window
    for right in range(k, len(nums)):
        window += nums[right] - nums[right - k]  # one in, one out
        best = max(best, window)
    return best


# --- Variable-size window (grow right, shrink left when the rule breaks) ----
def longest_unique_brute(s):
    """Check every substring: O(n^3) naive / O(n^2) with set. Fine for tiny inputs only."""
    best = 0
    for i in range(len(s)):
        seen = set()
        for j in range(i, len(s)):
            if s[j] in seen:
                break
            seen.add(s[j])
            best = max(best, j - i + 1)
    return best


def longest_unique(s):
    """Longest substring without repeating characters. O(n) time, O(k) space.
    last_seen maps char -> its most recent index so `left` can JUMP past a repeat."""
    last_seen = {}
    left = best = 0
    for right, ch in enumerate(s):
        if ch in last_seen and last_seen[ch] >= left:  # repeat inside current window
            left = last_seen[ch] + 1                   # shrink window just past old copy
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best


def min_subarray_len_brute(target, nums):
    best = float("inf")
    for i in range(len(nums)):
        total = 0
        for j in range(i, len(nums)):
            total += nums[j]
            if total >= target:
                best = min(best, j - i + 1)
                break
    return 0 if best == float("inf") else best


def min_subarray_len(target, nums):
    """Smallest subarray with sum >= target (positive numbers only!). O(n)."""
    left = total = 0
    best = float("inf")
    for right, x in enumerate(nums):
        total += x                              # grow
        while total >= target:                  # shrink while still valid
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best == float("inf") else best


if __name__ == "__main__":
    for nums, k in ([[2, 1, 5, 1, 3, 2], 3], [[1, 2], 2], [[1], 2], [[], 1], [[-1, -2, -3], 2]):
        assert max_sum_k(nums, k) == max_sum_k_brute(nums, k)
    print("max_sum_k([2,1,5,1,3,2], 3) =", max_sum_k([2, 1, 5, 1, 3, 2], 3))
    for s in ("abcabcbb", "bbbbb", "pwwkew", "", "a", "abba"):
        assert longest_unique(s) == longest_unique_brute(s)
    print("longest_unique('abcabcbb') =", longest_unique("abcabcbb"))
    for t, nums in ((7, [2, 3, 1, 2, 4, 3]), (4, [1, 4, 4]), (11, [1, 1, 1]), (5, [])):
        assert min_subarray_len(t, nums) == min_subarray_len_brute(t, nums)
    print("min_subarray_len(7, [2,3,1,2,4,3]) =", min_subarray_len(7, [2, 3, 1, 2, 4, 3]))
    print("All sliding-window tests passed.")
    # TRY IT: in longest_unique remove the `>= left` check and test "abba". Expected 2, you'll get 3. Why?
    # BREAK IT: call min_subarray_len with negatives like (3, [2, -1, 2]). The pattern assumes non-negatives.
