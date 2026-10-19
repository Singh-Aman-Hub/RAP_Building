"""
GOAL: Use prefix sums to answer "sum of a range" in O(1), and combine them with a dict
      to count subarrays (the bridge between arrays and hash maps).

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/06_prefix_sums/main.py
EXPECTED OUTPUT: prefix list, range sums, "All prefix-sum tests passed."
NEEDS: standard library only.
"""


def build_prefix(nums):
    """prefix[i] = sum of nums[0:i]  (so prefix has len(nums)+1 items and prefix[0] == 0).
    Then sum of nums[l..r] inclusive = prefix[r+1] - prefix[l]."""
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)           # previous total + new element
    return prefix


class RangeSum:
    """Build once in O(n), answer each query in O(1). (A class bundles data + methods.)"""

    def __init__(self, nums):                   # __init__ runs when you write RangeSum([...])
        self.prefix = build_prefix(nums)

    def query(self, left, right):               # inclusive both ends
        return self.prefix[right + 1] - self.prefix[left]


def subarray_sum_brute(nums, k):
    """Count contiguous subarrays summing to k. O(n^2)."""
    count = 0
    for i in range(len(nums)):
        total = 0
        for j in range(i, len(nums)):
            total += nums[j]
            if total == k:
                count += 1
    return count


def subarray_sum(nums, k):
    """O(n)/O(n). If prefix[j] - prefix[i] == k then a subarray ends at j.
    So for each running total, ask the dict: how many earlier prefixes equal total - k?"""
    seen = {0: 1}                                # the empty prefix (total 0) has occurred once
    total = count = 0
    for x in nums:
        total += x
        count += seen.get(total - k, 0)          # complement lookup again!
        seen[total] = seen.get(total, 0) + 1
    return count


def product_except_self(nums):
    """Product of all others without division. Prefix product pass + suffix product pass. O(n)."""
    n = len(nums)
    result = [1] * n                             # list repetition: [1, 1, ..., 1]
    running = 1
    for i in range(n):                           # result[i] = product of everything LEFT of i
        result[i] = running
        running *= nums[i]
    running = 1
    for i in range(n - 1, -1, -1):               # multiply by everything RIGHT of i (range counts down)
        result[i] *= running
        running *= nums[i]
    return result


if __name__ == "__main__":
    nums = [3, 1, 4, 1, 5]
    print("prefix:", build_prefix(nums))
    rs = RangeSum(nums)
    print("sum of nums[1..3]:", rs.query(1, 3))   # 1 + 4 + 1 = 6
    assert rs.query(0, 4) == 14 and rs.query(2, 2) == 4 and build_prefix([]) == [0]
    for arr, k in (([1, 1, 1], 2), ([1, 2, 3], 3), ([], 0), ([1, -1, 0], 0), ([5], 5), ([5], 6)):
        assert subarray_sum(arr, k) == subarray_sum_brute(arr, k)
    print("subarray_sum([1,1,1], 2) =", subarray_sum([1, 1, 1], 2))
    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([0, 2, 3]) == [6, 0, 0] and product_except_self([5]) == [1]
    print("All prefix-sum tests passed.")
    # TRY IT: remove `{0: 1}` initialisation and run subarray_sum([1,2,3], 3). The answer drops. Which subarray gets lost?
    # BREAK IT: RangeSum([1,2,3]).query(0, 3) -> IndexError. Read the traceback: which line, which index?
