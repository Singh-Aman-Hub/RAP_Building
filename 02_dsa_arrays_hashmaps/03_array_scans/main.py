"""
GOAL: Master traversal, searching, min/max tracking and duplicate detection on arrays (Python lists).

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/03_array_scans/main.py
EXPECTED OUTPUT: a few result lines then "All array-scan tests passed."
NEEDS: standard library only.
"""


# 1. Traversal: three ways to loop.  Know all three.
def traversal_demo(nums):
    for x in nums:                        # value only
        pass
    for i in range(len(nums)):            # index only
        pass
    for i, x in enumerate(nums):          # both (most Pythonic when you need both)
        pass
    return len(nums)


# 2. Minimum and maximum -- by hand (interviewers ask you NOT to use max()).
def find_max(nums):
    """O(n) time, O(1) space. Returns None for empty input (decide & document edge cases!)."""
    if not nums:                          # empty list is "falsy"
        return None
    best = nums[0]                        # start from the first element, not 0 (negatives!)
    for x in nums[1:]:                    # slicing copies the list; fine here, or use range(1, len(nums))
        if x > best:
            best = x
    return best


def find_min_max(nums):
    """One pass returns both.  Returns a tuple (min, max)."""
    if not nums:
        return None, None
    lo = hi = nums[0]                     # chained assignment
    for x in nums:
        if x < lo:
            lo = x
        elif x > hi:
            hi = x
    return lo, hi


def second_largest(nums):
    """Largest value strictly smaller than the max.  One pass, O(n)/O(1)."""
    first = second = float("-inf")        # -infinity is smaller than every number
    for x in nums:
        if x > first:
            first, second = x, first      # tuple swap: old max becomes second
        elif first > x > second:          # chained comparison: first > x and x > second
            second = x
    return None if second == float("-inf") else second


# 3. Searching
def linear_search(nums, target):
    """O(n). Returns index or -1 (same convention as str.find)."""
    for i, x in enumerate(nums):
        if x == target:
            return i
    return -1


def binary_search(sorted_nums, target):
    """O(log n) -- ONLY works on SORTED input. Halve the search range each step."""
    lo, hi = 0, len(sorted_nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2              # // is integer (floor) division
        if sorted_nums[mid] == target:
            return mid
        if sorted_nums[mid] < target:
            lo = mid + 1                  # target must be in the right half
        else:
            hi = mid - 1                  # target must be in the left half
    return -1


# 4. Track a running minimum: Best Time to Buy and Sell Stock (one pass).
def max_profit_brute(prices):
    """Try every buy/sell pair. O(n^2)."""
    best = 0
    for i in range(len(prices)):
        for j in range(i + 1, len(prices)):
            best = max(best, prices[j] - prices[i])
    return best


def max_profit(prices):
    """Remember the cheapest price so far; profit if we sold today. O(n)/O(1)."""
    cheapest, best = float("inf"), 0
    for p in prices:
        cheapest = min(cheapest, p)
        best = max(best, p - cheapest)
    return best


# 5. Duplicate detection when values are 1..n: use the array as its own marker (O(1) space).
def find_duplicate_marking(nums):
    """nums has n+1 numbers in range 1..n (exactly one repeated). Mark seen by negating index slots."""
    for x in nums:
        idx = abs(x) - 1
        if nums[idx] < 0:                 # already marked -> x is the duplicate
            result = abs(x)
            break
        nums[idx] = -nums[idx]
    else:
        result = None
    return result


if __name__ == "__main__":
    assert find_max([3, -1, 9, 2]) == 9 and find_max([-5, -2, -9]) == -2 and find_max([]) is None
    assert find_min_max([4, 2, 8, 2]) == (2, 8) and find_min_max([7]) == (7, 7)
    assert second_largest([5, 9, 9, 3]) == 5 and second_largest([4]) is None and second_largest([2, 2]) is None
    assert linear_search([4, 5, 6], 6) == 2 and linear_search([], 1) == -1
    assert binary_search([1, 3, 5, 7, 9], 7) == 3 and binary_search([1, 3, 5], 4) == -1 and binary_search([], 1) == -1
    for prices in ([7, 1, 5, 3, 6, 4], [7, 6, 4, 3, 1], [], [5]):
        assert max_profit(prices) == max_profit_brute(prices)
    print("max_profit([7,1,5,3,6,4]) =", max_profit([7, 1, 5, 3, 6, 4]))
    assert find_duplicate_marking([1, 3, 4, 2, 2]) == 2
    print("All array-scan tests passed.")
    # TRY IT: call binary_search on an UNSORTED list like [5, 1, 4]; see it fail silently. Why is that dangerous?
    # BREAK IT: in find_max start with best = 0 and test [-5, -2]; the answer is wrong. Which edge case did we miss?
