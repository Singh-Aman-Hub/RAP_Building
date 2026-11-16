"""Exercise 4: CTC collapse + count the alignments of 'ab' in 4 frames by brute force (step 05 says 15).
RUN: python s04_ctc_alignments.py"""
import itertools

BLANK = "-"


def collapse(path):
    out, prev = [], None
    for ch in path:                      # 1) merge repeats  2) drop blanks
        if ch != prev and ch != BLANK:
            out.append(ch)
        prev = ch
    return "".join(out)


assert collapse("aa-bb") == "ab" and collapse("a-a") == "aa" and collapse("aaa") == "a" and collapse("-a-a-") == "aa"
alphabet = ["a", "b", BLANK]
n = sum(collapse("".join(p)) == "ab" for p in itertools.product(alphabet, repeat=4))
print("alignments of 'ab' in 4 frames over {a, b, blank}:", n)
assert n == 15
# 'aa' needs a blank between the two a's: minimum frames 3
print("minimum frames for 'aa':", min(T for T in range(1, 6) if any(collapse("".join(p)) == "aa" for p in itertools.product(alphabet, repeat=T))))
print("ok")
