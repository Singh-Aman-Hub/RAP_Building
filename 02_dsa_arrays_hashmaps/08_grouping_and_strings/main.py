"""
GOAL: Grouping and counting with dicts, plus essential string problems.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 02_dsa_arrays_hashmaps/08_grouping_and_strings/main.py
EXPECTED OUTPUT: grouped anagrams, top-k, then "All grouping/string tests passed."
NEEDS: standard library only (collections).
"""

from collections import Counter, defaultdict  # defaultdict(list): missing keys start as []


def group_anagrams(words):
    """Words with the same letters share a key: the sorted letters. O(n * k log k)."""
    groups = defaultdict(list)                  # no need to check "if key not in groups"
    for w in words:
        key = "".join(sorted(w))                # sorted("eat") -> ['a','e','t'] -> "aet"
        groups[key].append(w)
    return list(groups.values())


def group_anagrams_by_count(words):
    """Alternative key: a tuple of 26 letter counts. O(n * k) -- no sorting. Lists aren't hashable, tuples are."""
    groups = defaultdict(list)
    for w in words:
        counts = [0] * 26
        for ch in w:
            counts[ord(ch) - ord("a")] += 1     # ord('b') - ord('a') == 1
        groups[tuple(counts)].append(w)
    return list(groups.values())


def top_k_frequent(nums, k):
    """k most common values. Counter does the counting; most_common sorts by count. O(n log n)."""
    return [value for value, _ in Counter(nums).most_common(k)]


def top_k_frequent_buckets(nums, k):
    """Bucket sort by frequency: O(n). buckets[c] = values that appear exactly c times."""
    counts = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]        # list comprehension (NOT [[]] * n, which shares one list!)
    for value, c in counts.items():
        buckets[c].append(value)
    result = []
    for c in range(len(buckets) - 1, 0, -1):            # highest frequency first
        for value in buckets[c]:
            result.append(value)
            if len(result) == k:
                return result
    return result


def first_unique_char(s):
    """Index of the first character that appears once, else -1. Two passes. O(n)."""
    counts = Counter(s)
    for i, ch in enumerate(s):
        if counts[ch] == 1:
            return i
    return -1


def reverse_words(s):
    """'  the sky  is blue ' -> 'blue is sky the'.  split() with no args also trims extra spaces."""
    return " ".join(reversed(s.split()))


def longest_common_prefix(strs):
    """Shrink the prefix until every word starts with it. O(total characters)."""
    if not strs:
        return ""
    prefix = strs[0]
    for word in strs[1:]:
        while not word.startswith(prefix):
            prefix = prefix[:-1]                         # drop last char
            if not prefix:
                return ""
    return prefix


def is_subsequence(s, t):
    """Are the letters of s inside t in order (gaps allowed)? Two pointers. O(len(t))."""
    i = 0
    for ch in t:
        if i < len(s) and ch == s[i]:
            i += 1
    return i == len(s)


def majority_element(nums):
    """Value appearing more than n/2 times, via counting. (Boyer-Moore voting gives O(1) space; look it up later.)"""
    counts = Counter(nums)
    return max(counts, key=counts.get)                   # max over keys, comparing by their count


if __name__ == "__main__":
    words = ["eat", "tea", "tan", "ate", "nat", "bat"]
    g1, g2 = group_anagrams(words), group_anagrams_by_count(words)
    print("groups:", g1)
    norm = lambda groups: sorted(sorted(g) for g in groups)   # order inside/between groups doesn't matter
    assert norm(g1) == norm(g2) == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams([]) == [] and group_anagrams([""]) == [[""]]

    nums = [1, 1, 1, 2, 2, 3]
    print("top 2 frequent:", top_k_frequent(nums, 2))
    assert set(top_k_frequent(nums, 2)) == set(top_k_frequent_buckets(nums, 2)) == {1, 2}
    assert top_k_frequent([5], 1) == [5]

    assert first_unique_char("leetcode") == 0 and first_unique_char("aabb") == -1 and first_unique_char("") == -1
    assert reverse_words("  the sky  is blue ") == "blue is sky the" and reverse_words("") == ""
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl" and longest_common_prefix(["a", "b"]) == ""
    assert longest_common_prefix([]) == "" and longest_common_prefix(["solo"]) == "solo"
    assert is_subsequence("ace", "abcde") and not is_subsequence("aec", "abcde") and is_subsequence("", "x")
    assert majority_element([2, 2, 1, 1, 2]) == 2
    print("All grouping/string tests passed.")
    # TRY IT: replace defaultdict(list) with a plain {} and run -- read the KeyError traceback.
    # BREAK IT: use `buckets = [[]] * (len(nums) + 1)` in top_k_frequent_buckets. Every bucket is the SAME list object!
