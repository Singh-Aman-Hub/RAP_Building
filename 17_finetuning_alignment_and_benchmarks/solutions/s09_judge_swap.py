"""Exercise 09 (Modify): measure position bias of ANY judge function by swapping answer order.
RUN: .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s09_judge_swap.py"""
import random
def make_judge(position_bias):
    def judge(a_text, b_text):
        qa, qb = len(set(a_text.split())), len(set(b_text.split()))      # quality = number of distinct words (a stand-in)
        return "A" if qa + position_bias > qb else "B"
    return judge
pairs = [("a b c d e f g", "a b c d e f"), ("x y z", "x y z w v"), ("one two three four", "one two three")]
for bias in (0, 1, 3):
    j = make_judge(bias); first_wins = consistent = 0
    for a, b in pairs:
        r1, r2 = j(a, b), j(b, a)                                         # original order, swapped order
        first_wins += (r1 == "A") + (r2 == "A")
        consistent += (r1 == "A") == (r2 == "B")                          # same underlying answer won both times
    print(f"bias={bias}: judge says 'A' {first_wins}/{2*len(pairs)} times; consistent after swap {consistent}/{len(pairs)}")
