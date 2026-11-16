"""Exercise 2: CER and WER by hand and in code. RUN: python s02_cer_wer_by_hand.py"""


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


ref, hyp = "the quick brown fox", "the quik brown fox"
cer = lev(ref, hyp) / len(ref)
wer = lev(ref.split(), hyp.split()) / len(ref.split())
print(f"characters in ref: {len(ref)}  edit distance (chars): {lev(ref, hyp)}  CER = {cer:.4f}")
print(f"words in ref: {len(ref.split())}  edit distance (words): {lev(ref.split(), hyp.split())}  WER = {wer:.4f}")
assert lev(ref, hyp) == 1 and abs(cer - 1 / 19) < 1e-12 and wer == 0.25
# one missing letter = CER 5.3% but WER 25%: one word of four is wrong
ref2, hyp2 = "pay 1250 rupees", "pay 125 rupees"
print("amount example: CER", round(lev(ref2, hyp2) / len(ref2), 3), " WER", round(lev(ref2.split(), hyp2.split()) / 3, 3), "-> tiny CER, wrong amount")
print("ok")
