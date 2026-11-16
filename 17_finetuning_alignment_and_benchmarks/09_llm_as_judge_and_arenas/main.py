"""
GOAL:  (A) LLM-as-judge mechanics with shared/llm_client.py: a rubric (pointwise) prompt, a pairwise prompt, and a deliberately
       BIASED mock judge so you can SEE position bias and verbosity bias, then fix position bias by swapping order.
       (B) Arena ranking: simulate pairwise human votes between 5 models with known strengths, then rank them with online Elo and with
       a Bradley-Terry maximum-likelihood fit (what the Arena leaderboard uses), with bootstrap uncertainty and Elo's order-dependence.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 17_finetuning_alignment_and_benchmarks/09_llm_as_judge_and_arenas/main.py
EXPECTED OUTPUT (shortened):
    rubric prompt, JSON score parsed from the judge, correlation with "human" scores
    position-bias table: raw A-rate above 50%, consistent-after-swap rate below 100%, naive accuracy above the both-orders-agree rate (the rest are ties)
    Elo vs Bradley-Terry rankings and their spread under reshuffled vote order
NEEDS: numpy, scipy; no model, no network (mock judge).  ~5 s.  LLM_BACKEND=anthropic would call a real judge (see shared/llm_client.py).
"""
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from llm_client import MockLLM, extract_json, generate   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
POSITION_BIAS = float(os.environ.get("POSITION_BIAS", 0.3))      # how many "quality points" the mock judge adds to answer A
random.seed(SEED)
rng = np.random.default_rng(SEED)

# ## 1. A tiny evaluation set: question, a good answer and a bad one, with a known truth
FACT = "refunds are issued within 9 days to the original payment method"
CASES = []
for i in range(40):
    good = f"Per policy, {FACT}." + (" Please keep your receipt." if i % 2 else "")
    bad = "I think refunds might take 9 days, not sure." + (" It depends on many things, really, and it varies a lot." * (i % 3))   # right number, hedged, sometimes padded
    CASES.append({"q": "How long do refunds take?", "good": good, "bad": bad})

# ## 2. The rubric (pointwise) prompt
RUBRIC = """You are a strict grader. Score the ANSWER from 1 to 5 using this rubric:
- +2 states the correct number of days (9)
- +1 mentions the refund goes to the original payment method
- +1 sounds confident (no hedging words like 'might' or 'not sure')
- +1 is concise (under 40 words)
Return ONLY JSON: {"score": <1-5>, "reason": "<one sentence>"}
QUESTION: {q}
ANSWER: {a}"""

# ## 3. A mock judge with deliberate flaws: it is a rule, so it is deterministic and offline
def mock_judge(messages, system):
    text = messages[-1]["content"]
    if "Return ONLY JSON" in text:                                       # pointwise rubric request
        ans = text.split("ANSWER:")[-1]
        s = 1 + 2 * ("9 days" in ans) + ("original payment method" in ans) + (not re.search(r"might|not sure", ans)) + (len(ans.split()) < 40)
        return f'Here is my grade: {{"score": {min(5, s)}, "reason": "rubric applied"}} Hope that helps.'
    a = re.search(r"Response A: (.*?)\nResponse B:", text, re.S).group(1)
    b = re.search(r"Response B: (.*?)\nWhich", text, re.S).group(1)
    quality = lambda t: 2.0 * ("9 days" in t) + 0.3 * (not re.search(r"might|not sure", t)) + 0.01 * len(t.split())   # note the length bonus
    return "A" if quality(a) + POSITION_BIAS > quality(b) else "B"       # +POSITION_BIAS for whichever answer is shown first
judge = MockLLM(rules=[(r".", mock_judge)])

# ## 4. Pointwise judging and parsing
def judge_score(q, a):
    reply = generate([{"role": "user", "content": RUBRIC.replace("{q}", q).replace("{a}", a)}], backend="mock", mock=judge)
    return extract_json(reply)["score"]
print("rubric reply for one good answer:", repr(generate([{"role": "user", "content": RUBRIC.replace('{q}', CASES[0]['q']).replace('{a}', CASES[0]['good'])}], backend="mock", mock=judge)))
good_scores = [judge_score(c["q"], c["good"]) for c in CASES]
bad_scores = [judge_score(c["q"], c["bad"]) for c in CASES]
print(f"pointwise mean score good={np.mean(good_scores):.2f} bad={np.mean(bad_scores):.2f}")
human = [5] * 40 + [1] * 40                                              # pretend human labels: all good = 5, all bad = 1
rho = spearmanr(good_scores + bad_scores, human).statistic
print(f"Spearman correlation judge vs 'human': {rho:.2f}")

# ## 5. Pairwise judging, once and with the order swapped
PAIR = "Compare two answers to: {q}\nResponse A: {a}\nResponse B: {b}\nWhich response is better? Answer with a single letter, A or B."
def pairwise(q, first, second):
    return generate([{"role": "user", "content": PAIR.format(q=q, a=first, b=second)}], backend="mock", mock=judge).strip()

# Put the GOOD answer first half the time so that "always say A" is not accidentally right.
raw_a, consistent, debiased_right, naive_right = 0, 0, 0, 0
for i, c in enumerate(CASES):
    good_first = i % 2 == 0
    r1 = pairwise(c["q"], c["good"], c["bad"]) if good_first else pairwise(c["q"], c["bad"], c["good"])
    r2 = pairwise(c["q"], c["bad"], c["good"]) if good_first else pairwise(c["q"], c["good"], c["bad"])   # swapped order
    win1 = (r1 == "A") == good_first                                     # did the GOOD answer win in order 1?
    win2 = (r2 == "B") == good_first                                     # ...and in order 2 (positions swapped)?
    raw_a += (r1 == "A") + (r2 == "A")
    consistent += win1 == win2
    naive_right += win1
    debiased_right += win1 and win2                                      # count it as a win only if it wins in BOTH orders
print(f"\nposition bias = {POSITION_BIAS}")
print(f"judge said 'A' in {raw_a / 80:.0%} of 80 calls (a fair judge says A about 50%)")
print(f"naive single-order accuracy        : {naive_right / 40:.2f}")
print(f"verdict unchanged after swapping   : {consistent / 40:.2f}")
print(f"both-orders-agree wins (debiased)  : {debiased_right / 40:.2f}   (the rest become ties, not errors)")

# ## 6. Simulate an arena: 5 models with true Bradley-Terry strengths, random pairings, noisy human votes
NAMES = ["model-A", "model-B", "model-C", "model-D", "model-E"]
TRUE = np.array([1.2, 0.6, 0.3, 0.0, -0.8])                              # latent logits; P(i beats j) = sigmoid(TRUE_i - TRUE_j)
def simulate_votes(n=1500):
    votes = []
    for _ in range(n):
        i, j = rng.choice(5, size=2, replace=False)
        p = 1 / (1 + math.exp(-(TRUE[i] - TRUE[j])))
        votes.append((i, j, 1 if rng.random() < p else 0))               # (model i, model j, 1 if i won)
    return votes
votes = simulate_votes()

# ## 7. Online Elo: update after every vote, in order
def elo(votes, k=16, base=1000.0):
    r = np.full(5, base)
    for i, j, w in votes:
        e = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))                        # expected score of i
        r[i] += k * (w - e)
        r[j] -= k * (w - e)
    return r

# ## 8. Bradley-Terry maximum likelihood (gradient ascent on log-likelihood), converted to the Elo scale
def bradley_terry(votes, iters=400, lr=0.5):
    theta = np.zeros(5)
    I = np.array([v[0] for v in votes]); J = np.array([v[1] for v in votes]); W = np.array([v[2] for v in votes], dtype=float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(theta[I] - theta[J])))
        g = np.zeros(5)
        np.add.at(g, I, W - p)                                           # d loglik / d theta_i
        np.add.at(g, J, -(W - p))
        theta += lr * g / len(votes) * 5
        theta -= theta.mean()                                            # only differences matter: pin the mean to 0
    return theta * 400 / math.log(10) + 1000.0                           # natural-log logits -> Elo-like points

bt = bradley_terry(votes)
e = elo(votes)
true_elo = TRUE * 400 / math.log(10) + 1000.0 - (TRUE.mean() * 400 / math.log(10))
print(f"\n{'model':<10}{'true':>8}{'Elo (1 pass)':>14}{'BT MLE':>10}")
for n_, t_, a_, b_ in zip(NAMES, true_elo, e, bt):
    print(f"{n_:<10}{t_:>8.0f}{a_ - e.mean() + 1000:>14.0f}{b_:>10.0f}")

# ## 9. Order sensitivity of Elo vs a bootstrap interval for Bradley-Terry
elos = []
for s in range(30):
    sh = votes[:]; random.Random(s).shuffle(sh)                          # same votes, different order
    elos.append(elo(sh))
print("Elo spread across 30 vote orderings (std, points):", np.round(np.std(elos, axis=0), 1))
boot = []
for s in range(60):
    idx = np.random.default_rng(s).integers(0, len(votes), len(votes))
    boot.append(bradley_terry([votes[i] for i in idx]))
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
for n_, a_, b_ in zip(NAMES, lo, hi):
    print(f"  BT 95% bootstrap interval {n_}: [{a_:.0f}, {b_:.0f}]")

# TRY IT: POSITION_BIAS=0 python main.py (an unbiased judge); change n in simulate_votes to 200 and look at the intervals.
# BREAK IT: POSITION_BIAS=5 python main.py -> the judge always says A; the naive single-order accuracy collapses to the share of good-first cases.
