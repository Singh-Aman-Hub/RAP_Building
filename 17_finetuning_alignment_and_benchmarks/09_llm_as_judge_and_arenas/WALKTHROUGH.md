# Step 09 — LLM-as-judge and arenas

## The story so far
[Step 08](../08_evaluation_from_scratch/WALKTHROUGH.md) scored models with code: log-likelihoods, exact match, pass@k. That works when the answer can be checked. A chat answer ("explain refunds politely") cannot be checked by a regex. You need a judge: either humans (slow, costly) or another model (fast, biased).

## The pain point
Judges are not neutral. A model asked "which answer is better, A or B?" tends to prefer the answer in a given slot, or the longer answer, or its own style. If you do not test for this, your "win rate" measures the judge's habits. And when you do collect votes, how do you turn thousands of "A beat B" votes into a ranking you can trust?

## The spark
Treat the judge as a measuring instrument and calibrate it. Run every pairwise comparison twice with the order swapped and count a win only if it survives the swap. For ranking, fit a Bradley-Terry model by maximum likelihood (what the Arena leaderboard is built on) instead of updating Elo vote by vote, and put a bootstrap interval around every rating.

## Goal
Build a rubric judge and a pairwise judge on top of `shared/llm_client.py` (mock backend, offline), measure position bias, fix it with swaps, then simulate an arena with known strengths and compare online Elo with Bradley-Terry.

## The theory (read this before the interview)
**Pointwise judging:** the judge sees one answer and a rubric and returns a score (ask for JSON, parse it with `extract_json`). Good for absolute tracking over time and for filtering. Weaker for fine distinctions; scores bunch up.

**Pairwise judging:** the judge sees two answers and says A or B. Better at fine distinctions, needs more calls to rank many systems.

**Known judge biases:**
| Bias | What happens | Test | Mitigation |
|---|---|---|---|
| Position bias | prefers a slot (often the first) | swap order, compare verdicts | run both orders, count only consistent wins, report ties |
| Verbosity bias | prefers longer answers | add padding to an answer, see if the verdict flips | length-controlled scoring, rubric that penalises padding |
| Self-preference | prefers its own family's style | compare judges from different vendors | use a judge from another family, ensemble judges |
| Rubric drift | scores shift with wording | re-run with paraphrased rubric | fixed, versioned rubric; calibrate against human labels |

**Calibration:** label 50-200 items by hand and compute the correlation (Spearman) or agreement between the judge and the humans. A judge you have not calibrated is an opinion.

**Arena ranking:** each vote is (model i, model j, winner). Bradley-Terry assumes `P(i beats j) = sigmoid(theta_i - theta_j)`. Maximum likelihood finds all thetas at once, so the result does not depend on vote order, and a bootstrap over votes gives intervals. Online Elo (`r_i += K*(w - expected)`) updates after each vote: simple, but order-dependent and noisier. To put theta on an Elo-like scale multiply by `400/ln(10)`.

**Modern (2026) practice:** judge ensembles, reference-guided or rubric-guided grading, length control, and publishing the judge model and prompt with every number. Arena-style human votes remain the reference for general chat quality; the public site rebranded from LMArena to "Arena" in January 2026 (check the site for the current method notes).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| LLM-as-judge | auto-eval, model-graded eval, AI feedback | yes | A model produces the score instead of a human or a regex |
| Pointwise vs pairwise | absolute vs comparative grading | no | One answer on a scale vs two answers head to head |
| Elo vs Bradley-Terry | online rating vs MLE rating | no | Elo updates per game and depends on order; BT fits all games at once |
| Win rate vs rating | head-to-head % vs Elo-like score | no | Win rate is for one pair; a rating summarises the whole field |
| Position bias vs verbosity bias | order effect vs length effect | no | Different causes, both fixed by controlled comparisons |
| RLAIF vs LLM-as-judge | training signal vs evaluation | same tool, different use | RLAIF uses AI labels to train, judge uses them to measure |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/09_llm_as_judge_and_arenas
python main.py
# optional knobs: POSITION_BIAS=0 python main.py   POSITION_BIAS=5 python main.py
# a real judge: LLM_BACKEND=anthropic with ANTHROPIC_API_KEY (see shared/llm_client.py); this script forces backend="mock" so it stays offline
```

## Code, step by step

Lines 1–14 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 1. Imports, settings and seeds  (`main.py` lines 15–31)
```python
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
```
- **In simple words:** This block loads the tools and the settings. `POSITION_BIAS` is a dial for how unfair the fake judge is. Like setting up a lab bench before an experiment.
- **Exactly:** `MockLLM`, `extract_json`, `generate` come from `shared/llm_client.py` (a `sys.path` insert two folders up reaches `shared/`). `POSITION_BIAS` defaults to 0.3 quality points added to whichever answer is shown first. Seeds: Python `random` and a NumPy `default_rng(0)` generator.
- **Why it is here:** Without a seed the vote simulation and the numbers in this walkthrough would change on every run.

### 2. A tiny evaluation set: question, a good answer and a bad one, with a known truth  (`main.py` lines 33–39)
```python
# ## 1. A tiny evaluation set: question, a good answer and a bad one, with a known truth
FACT = "refunds are issued within 9 days to the original payment method"
CASES = []
for i in range(40):
    good = f"Per policy, {FACT}." + (" Please keep your receipt." if i % 2 else "")
    bad = "I think refunds might take 9 days, not sure." + (" It depends on many things, really, and it varies a lot." * (i % 3))   # right number, hedged, sometimes padded
    CASES.append({"q": "How long do refunds take?", "good": good, "bad": bad})
```
- **In simple words:** We write 40 test cases, each with a good answer and a hedged "bad" one, so we know the truth.
- **Exactly:** `CASES` is a list of 40 dicts `{q, good, bad}`. Good answers state "9 days" and the original payment method; every second one adds a sentence. Bad answers say "might take 9 days, not sure" and every third one gets 0, 1 or 2 padding sentences (verbosity). The truth label is therefore always "good beats bad".
- **Why it is here:** A judge can only be tested against known truth. Padding is included to give the judge a length temptation.

### 3. The rubric (pointwise) prompt  (`main.py` lines 41–49)
```python
# ## 2. The rubric (pointwise) prompt
RUBRIC = """You are a strict grader. Score the ANSWER from 1 to 5 using this rubric:
- +2 states the correct number of days (9)
- +1 mentions the refund goes to the original payment method
- +1 sounds confident (no hedging words like 'might' or 'not sure')
- +1 is concise (under 40 words)
Return ONLY JSON: {"score": <1-5>, "reason": "<one sentence>"}
QUESTION: {q}
ANSWER: {a}"""
```
- **In simple words:** The rubric is a prompt that tells the judge exactly how to score and what format to answer in.
- **Exactly:** A template string with two placeholders, `{q}` and `{a}`, a four-line point scheme (+2 correct number, +1 original payment method, +1 confident, +1 concise) and the instruction `Return ONLY JSON`. Using `.replace` instead of `.format` avoids clashes with the braces of the JSON example.
- **Why it is here:** Explicit points make the grading repeatable; asking for JSON makes the reply machine-readable.

### 4. A mock judge with deliberate flaws: it is a rule, so it is deterministic and offline  (`main.py` lines 51–62)
```python
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
```
- **In simple words:** This is the fake judge: a plain function that behaves like a slightly unfair grader. It lets us run offline and measure bias exactly.
- **Exactly:** `mock_judge(messages, system)` reads the last user message. For the rubric prompt it applies the point scheme with string tests and returns chatter around a JSON object. For the pairwise prompt it extracts A and B with regexes and computes `quality = 2*("9 days" in t) + 0.3*(no hedging) + 0.01*len(words)`, then adds `POSITION_BIAS` to A's quality. `MockLLM(rules=[(r".", mock_judge)])` routes every message to it.
- **Why it is here:** With a real judge you cannot see its internal bias. A mock with a dial proves the test would catch a bias of a known size.

### 5. Pointwise judging and parsing  (`main.py` lines 64–74)
```python
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
```
- **In simple words:** We score every answer with the rubric, parse the JSON, and check the scores against the "human" labels.
- **Exactly:** `judge_score` formats the rubric, calls `generate(..., backend="mock", mock=judge)` and returns `extract_json(reply)["score"]`. The first print shows the raw reply with chatter. Means: good 5.00, bad 4.00. `spearmanr(scores, human).statistic` compares judge ranks with the labels (5 for good, 1 for bad); it is 1.00.
- **Why it is here:** Calibration against labels is how you decide whether to trust a judge at all.

### 6. Pairwise judging, once and with the order swapped  (`main.py` lines 76–97)
```python
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
```
- **In simple words:** We ask the judge to pick the better answer twice, once in each order, and count how often it is stable.
- **Exactly:** `PAIR` is the pairwise prompt. For each of the 40 cases the good answer goes first in the even cases, so "always A" cannot look good. `r1` is the verdict in the original order, `r2` with positions swapped. `win1` / `win2` are booleans for "the good answer won". Counters: how often the judge said "A" in all 80 calls, how often `win1 == win2` (consistent), naive accuracy `win1`, and `win1 and win2` (debiased: the win must hold in both orders).
- **Why it is here:** This is the whole position-bias test. Without the swap a biased judge looks accurate.

### 7. Simulate an arena: 5 models with true Bradley-Terry strengths, random pairings, noisy human votes  (`main.py` lines 99–109)
```python
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
```
- **In simple words:** Now the arena. We invent five models with known true strengths and simulate 1,500 votes between random pairs.
- **Exactly:** `TRUE` holds latent logits `[1.2, 0.6, 0.3, 0.0, -0.8]`. For a random pair `(i, j)`, `P(i wins) = sigmoid(TRUE[i] - TRUE[j])` and the vote is drawn with `rng.random() < p`. `votes` is a list of `(i, j, 1 if i won else 0)`.
- **Why it is here:** With the truth known we can check which ranking method recovers it; real leaderboards never know the truth.

### 8. Online Elo: update after every vote, in order  (`main.py` lines 111–118)
```python
# ## 7. Online Elo: update after every vote, in order
def elo(votes, k=16, base=1000.0):
    r = np.full(5, base)
    for i, j, w in votes:
        e = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))                        # expected score of i
        r[i] += k * (w - e)
        r[j] -= k * (w - e)
    return r
```
- **In simple words:** Elo updates two ratings after each vote, in the order the votes arrive.
- **Exactly:** `e = 1/(1+10**((r[j]-r[i])/400))` is the expected score of `i`. The update `r[i] += k*(w-e)` and `r[j] -= k*(w-e)` moves each rating by `k=16` times the surprise. All start at 1000.
- **Why it is here:** It is the classic, easy baseline; seeing its order dependence later is the point of the next steps.

### 9. Bradley-Terry maximum likelihood (gradient ascent on log-likelihood), converted to the Elo scale  (`main.py` lines 120–138)
```python
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
```
- **In simple words:** Bradley-Terry fits all votes at once. We climb the likelihood with simple gradient ascent and convert the result to Elo-like points.
- **Exactly:** `theta` has one logit per model. For each vote `p = sigmoid(theta[I]-theta[J])`; the gradient of the log-likelihood for model `i` is `sum(W - p)` over its wins/losses, scattered with `np.add.at`. After each step `theta -= theta.mean()` pins the arbitrary origin. `theta * 400/ln(10) + 1000` converts natural-log odds to the Elo scale. Output table compares true, Elo, BT.
- **Why it is here:** This is the order-independent estimator used by Arena-style leaderboards; its objective is concave, so gradient ascent reaches the unique optimum.

### 10. Order sensitivity of Elo vs a bootstrap interval for Bradley-Terry  (`main.py` lines 140–155)
```python
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
```
- **In simple words:** We measure how much Elo wobbles if the same votes arrive in a different order, and put bootstrap intervals around the Bradley-Terry ratings.
- **Exactly:** 30 shuffles of `votes` with `random.Random(s)`; the per-model standard deviation of the final Elo ratings is printed. Then 60 bootstrap resamples (indices drawn with replacement) refit BT and the 2.5th and 97.5th percentiles are printed per model. The last two lines of the file are the TRY IT and BREAK IT comments.
- **Why it is here:** A ranking without an interval invites over-reading a 20-point gap that is inside the noise.


## Syntax explained
| Syntax | Meaning |
|---|---|
| `MockLLM(rules=[(regex, callable)])` | a fake model: the first regex matching the last user message wins; a callable gets `(messages, system)` |
| `generate(messages, backend="mock", mock=judge)` | the one text-in/text-out function used across the course |
| `extract_json(text)` | takes the first `{...}` span of a chatty reply and `json.loads` it |
| `np.add.at(g, I, values)` | unbuffered scatter-add: repeated indices accumulate (a plain `g[I] += v` would not) |
| `rng.choice(5, size=2, replace=False)` | two distinct models for one match |
| `(r1 == "A") == good_first` | boolean that is True when the judge's pick equals the known better answer |

## Libraries & functions
| Library / function | What it does here |
|---|---|
| `shared/llm_client.py` | backend switch (mock, local, anthropic) behind one `generate` |
| `scipy.stats.spearmanr` | rank correlation between judge scores and "human" scores |
| `numpy` | vote simulation, gradient of the Bradley-Terry log-likelihood, percentile intervals |
| `random.Random(s).shuffle` | reorder the same votes to expose Elo's order dependence |

## Execution trace (real run)
```

```
- The rubric reply contains chatter around the JSON; `extract_json` pulls `{"score": 5, ...}` out.
- Pointwise: good answers score 5 and the hedged "bad" ones score 4. They still say "9 days", so the mock rubric rewards them for the number, which is why the gap is only 1 point. Spearman 1.00 means the *ranking* matches the human labels.
- Position bias 0.3: the judge says "A" in 82% of the 80 calls although a fair judge says it about half the time. Single-order accuracy is 0.68. Only 0.35 of the verdicts survive the swap, so a naive win rate would have been mostly the judge's slot preference.
- The arena table: Elo and Bradley-Terry both order the five models correctly, but Elo has the top model 54 points above truth after one pass (1217 vs 1163) while BT is closer (1185). The spread of Elo across 30 vote orderings is 20-31 points per model; the BT bootstrap intervals are 40-65 points wide: honest uncertainty.

## Experiments (change ONE thing)
| Change | Real result | Lesson |
|---|---|---|
| `POSITION_BIAS=0` | says "A" 50% of 80 calls; naive accuracy 1.00; unchanged after swap 1.00 | an unbiased judge needs no swap fix |
| default `POSITION_BIAS=0.3` | says "A" 82%; naive 0.68; consistent 0.35 | a small bias flips every close call |
| `POSITION_BIAS=5` | says "A" 100%; naive accuracy 0.50; consistent 0.00 | naive accuracy collapses to the share of good-first cases (half of them); swap detects it fully |
| change nothing else | Elo/BT table identical in all three runs | the arena part uses its own simulated votes, not the judge |

## Debugging practice
**BREAK IT:** make the judge reply `Score: 5` (no JSON) and run again. Real traceback (the file name is the temporary broken copy of `main.py`):
```
rubric reply for one good answer: 'Score: 5'
Traceback (most recent call last):
  File "/Users/aman/Desktop/Project_resume/RAP_interview/17_finetuning_alignment_and_benchmarks/09_llm_as_judge_and_arenas/_brk.py", line 69, in <module>
    good_scores = [judge_score(c["q"], c["good"]) for c in CASES]
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/aman/Desktop/Project_resume/RAP_interview/17_finetuning_alignment_and_benchmarks/09_llm_as_judge_and_arenas/_brk.py", line 67, in judge_score
    return extract_json(reply)["score"]
           ^^^^^^^^^^^^^^^^^^^
  File "/Users/aman/Desktop/Project_resume/RAP_interview/shared/llm_client.py", line 147, in extract_json
    raise ValueError(f"No JSON object found in model reply: {text!r}")
ValueError: No JSON object found in model reply: 'Score: 5'
```
How to read it: the last line is the cause. `extract_json` found no `{...}` span, so it raised `ValueError`. The two frames above show who called it: `judge_score`, inside the list comprehension at the pointwise step. Fix options: ask for JSON in the prompt (done), retry on `ValueError`, or use the provider's structured-output mode. Never let an unparsed judge reply silently become a score of 0.

## Rebuild checklist
- [ ] A rubric prompt that asks for JSON only, plus a parser that survives chatter
- [ ] A pairwise prompt that you can call with both orders
- [ ] The consistency count: win only if it wins in both orders
- [ ] Elo update and Bradley-Terry gradient `sum(w - p)` written from memory
- [ ] A bootstrap over votes for the rating intervals

## The new problem (cliffhanger)
You now know how to score and rank models. But which *named benchmarks* should you trust in Nov 2026, which are saturated or contaminated, and which tool runs them? On to [step 10](../10_benchmark_landscape_2026_and_tools/WALKTHROUGH.md).

> **Say it aloud (interview box):** "LLM judges have position, length and self-preference bias, so I swap the order and count only consistent wins, calibrate against human labels, and rank with Bradley-Terry plus bootstrap intervals instead of order-dependent online Elo."

