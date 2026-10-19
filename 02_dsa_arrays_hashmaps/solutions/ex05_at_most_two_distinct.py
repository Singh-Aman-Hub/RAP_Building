"""Exercise 05: longest substring with at most 2 distinct characters. Variable window + dict of counts. O(n)."""


def longest_two_distinct(s):
    counts, left, best = {}, 0, 0
    for right, ch in enumerate(s):
        counts[ch] = counts.get(ch, 0) + 1
        while len(counts) > 2:                    # rule broken -> shrink from the left
            counts[s[left]] -= 1
            if counts[s[left]] == 0:
                del counts[s[left]]               # remove key so len(counts) = distinct count
            left += 1
        best = max(best, right - left + 1)
    return best


if __name__ == "__main__":
    assert longest_two_distinct("eceba") == 3 and longest_two_distinct("ccaabbb") == 5
    assert longest_two_distinct("") == 0 and longest_two_distinct("a") == 1
    print("ex05 ok")
