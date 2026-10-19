"""Exercise 07: Sort Colors (0,1,2) in place in one pass: Dutch national flag with three pointers. O(n)/O(1)."""


def sort_colors(nums):
    low, mid, high = 0, 0, len(nums) - 1
    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1; mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:                                     # 2: send to the back; don't advance mid (new item unseen)
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
    return nums


if __name__ == "__main__":
    assert sort_colors([2, 0, 2, 1, 1, 0]) == [0, 0, 1, 1, 2, 2] and sort_colors([]) == [] and sort_colors([1]) == [1]
    print("ex07 ok")
