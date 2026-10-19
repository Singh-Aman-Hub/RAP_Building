"""
GOAL: Modify arrays in place (O(1) extra space) and understand sorting basics.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/07_in_place_and_sorting/main.py
EXPECTED OUTPUT: sorted examples then "All in-place/sorting tests passed."
NEEDS: standard library only.
"""


def move_zeroes(nums):
    """Move all 0s to the end, keep the order of the others. Slow/fast pointers. O(n)/O(1)."""
    write = 0                                   # next slot for a non-zero
    for read in range(len(nums)):               # `read` scans, `write` lags behind
        if nums[read] != 0:
            nums[write], nums[read] = nums[read], nums[write]
            write += 1
    return nums


def remove_duplicates_sorted(nums):
    """Sorted input: keep one copy of each value at the FRONT; return new length. O(n)/O(1)."""
    if not nums:
        return 0
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:        # new value -> keep it
            nums[write] = nums[read]
            write += 1
    return write                                # nums[:write] is the answer; the tail is junk


def rotate_right(nums, k):
    """Rotate right by k using three reversals. O(n)/O(1).  [1..5], k=2 -> [4,5,1,2,3]"""
    n = len(nums)
    if n == 0:
        return nums
    k %= n                                      # rotating n times is a no-op; % keeps k in range

    def reverse(lo, hi):                        # a function defined inside a function (inner function)
        while lo < hi:
            nums[lo], nums[hi] = nums[hi], nums[lo]
            lo, hi = lo + 1, hi - 1

    reverse(0, n - 1)
    reverse(0, k - 1)
    reverse(k, n - 1)
    return nums


def insertion_sort(nums):
    """Grow a sorted prefix; insert each new item into place. O(n^2) worst, O(n) if already sorted. Stable."""
    for i in range(1, len(nums)):
        current = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > current:     # shift bigger items one step right
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = current
    return nums


def merge_sorted(a, b):
    """Merge two sorted lists into one (the 'merge' half of merge sort). O(n+m)."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:                        # <= keeps it stable (ties favour the left list)
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:])                           # one of these is empty; the other holds leftovers
    out.extend(b[j:])
    return out


def sort_demo():
    """Python's built-in sort is Timsort: O(n log n), stable. Learn sorted() vs list.sort()."""
    words = ["banana", "Apple", "cherry", "fig"]
    print("sorted default      :", sorted(words))                       # uppercase sorts before lowercase!
    print("sorted by length    :", sorted(words, key=len))              # key= takes a function
    print("sorted case-insens. :", sorted(words, key=str.lower))
    print("sorted descending   :", sorted([3, 1, 2], reverse=True))
    pairs = [("a", 2), ("b", 1), ("c", 2)]
    print("sorted by 2nd item  :", sorted(pairs, key=lambda p: p[1]))   # lambda = tiny anonymous function; ties keep order (stable)


if __name__ == "__main__":
    assert move_zeroes([0, 1, 0, 3, 12]) == [1, 3, 12, 0, 0] and move_zeroes([]) == [] and move_zeroes([0, 0]) == [0, 0]
    arr = [1, 1, 2, 2, 2, 3]
    n = remove_duplicates_sorted(arr)
    assert n == 3 and arr[:n] == [1, 2, 3] and remove_duplicates_sorted([]) == 0
    assert rotate_right([1, 2, 3, 4, 5], 2) == [4, 5, 1, 2, 3] and rotate_right([1, 2], 5) == [2, 1] and rotate_right([], 3) == []
    for data in ([5, 2, 4, 6, 1, 3], [], [1], [2, 2, 1], [1, 2, 3]):
        assert insertion_sort(list(data)) == sorted(data)   # list(data) copies so we don't mutate the test input
    assert merge_sorted([1, 3, 5], [2, 4]) == [1, 2, 3, 4, 5] and merge_sorted([], [1]) == [1] and merge_sorted([], []) == []
    sort_demo()
    print("All in-place/sorting tests passed.")
    # TRY IT: change `if nums[read] != 0` in move_zeroes to `== 0`; predict the output before running.
    # BREAK IT: call rotate_right(5, 2) with an int instead of a list. Read the TypeError.
