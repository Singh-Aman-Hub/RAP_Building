"""
GOAL: Learn the two-pointer pattern: two indexes walking the array to avoid a nested loop.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/04_two_pointers/main.py
EXPECTED OUTPUT: "All two-pointer tests passed."
NEEDS: standard library only.
"""


def reverse_in_place(nums):
    """Swap ends moving inward. O(n) time, O(1) extra space. Modifies the list itself."""
    left, right = 0, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]  # tuple swap, no temp variable needed
        left += 1
        right -= 1
    return nums


def is_palindrome(s):
    """Ignore non-alphanumerics and case. Pointers start at both ends and meet in the middle."""
    left, right = 0, len(s) - 1
    while left < right:
        while left < right and not s[left].isalnum():    # skip punctuation/spaces
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True


def two_sum_sorted_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []


def two_sum_sorted(nums, target):
    """Input MUST be sorted. Sum too small -> move left up; too big -> move right down. O(n)/O(1).
    Compare with the dict version in 02_hashmap_basics: same answer, but O(1) space (needs sorted input)."""
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left, right]
        if total < target:
            left += 1
        else:
            right -= 1
    return []


def max_area_brute(heights):
    """Container With Most Water, brute force O(n^2)."""
    best = 0
    for i in range(len(heights)):
        for j in range(i + 1, len(heights)):
            best = max(best, min(heights[i], heights[j]) * (j - i))
    return best


def max_area(heights):
    """Move the SHORTER wall inward -- moving the taller one can never help. O(n)/O(1)."""
    left, right, best = 0, len(heights) - 1, 0
    while left < right:
        best = max(best, min(heights[left], heights[right]) * (right - left))
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    return best


def three_sum(nums):
    """All unique triplets summing to 0. Sort, fix one number, two-pointer the rest. O(n^2)."""
    nums = sorted(nums)
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:             # skip duplicate anchors
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                while left < right and nums[left] == nums[left - 1]:  # skip duplicate values
                    left += 1
    return result


if __name__ == "__main__":
    assert reverse_in_place([1, 2, 3, 4]) == [4, 3, 2, 1] and reverse_in_place([]) == [] and reverse_in_place([9]) == [9]
    assert is_palindrome("A man, a plan, a canal: Panama") and not is_palindrome("race a car") and is_palindrome("")
    for nums, t in ([[2, 7, 11, 15], 9], [[1, 2, 3, 4, 4], 8], [[1, 2], 10], [[], 1]):
        assert two_sum_sorted(nums, t) == two_sum_sorted_brute(nums, t)
    for h in ([1, 8, 6, 2, 5, 4, 8, 3, 7], [1, 1], [], [5]):
        assert max_area(h) == max_area_brute(h)
    print("max_area([1,8,6,2,5,4,8,3,7]) =", max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]] and three_sum([0, 0, 0, 0]) == [[0, 0, 0]]
    assert three_sum([]) == [] and three_sum([1, 2]) == []
    print("All two-pointer tests passed.")
    # TRY IT: print left/right every loop of two_sum_sorted for [2,7,11,15], 9 -- that's your dry run.
    # BREAK IT: change `while left < right` to `<=` in reverse_in_place; nothing breaks. In two_sum_sorted it does. Why?
