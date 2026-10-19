"""Exercise 02: return ALL index pairs (i < j) summing to target. O(n) average with a dict of value -> list of indices."""
from collections import defaultdict


def two_sum_all(nums, target):
    seen = defaultdict(list)          # value -> every index where it occurred
    pairs = []
    for j, x in enumerate(nums):
        for i in seen[target - x]:    # all earlier partners
            pairs.append([i, j])
        seen[x].append(j)
    return pairs


if __name__ == "__main__":
    assert two_sum_all([1, 2, 3, 2], 4) == [[0, 2], [1, 3]]
    assert two_sum_all([], 1) == [] and two_sum_all([2, 2, 2], 4) == [[0, 1], [0, 2], [1, 2]]
    print("ex02 ok")
