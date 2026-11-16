# Terms — LLM-as-judge and arenas

Folder: `17_finetuning_alignment_and_benchmarks/09_llm_as_judge_and_arenas` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [17 README](../README.md)

**Terms in this folder:** [LLM-as-judge](#llm-as-judge) · [Pointwise vs pairwise judging](#pointwise-vs-pairwise-judging) · [Rubric prompt](#rubric-prompt) · [extract_json](#extract_json) · [Position bias](#position-bias) · [Verbosity bias](#verbosity-bias) · [Self-preference bias](#self-preference-bias) · [Debiased verdict (both-orders agreement)](#debiased-verdict-both-orders-agreement) · [Spearman correlation](#spearman-correlation) · [Calibration against human labels](#calibration-against-human-labels) · [Elo rating](#elo-rating) · [Bradley-Terry model](#bradley-terry-model) · [Bootstrap interval for ratings](#bootstrap-interval-for-ratings) · [Arena (formerly LMArena, Chatbot Arena)](#arena-formerly-lmarena-chatbot-arena) · [Win rate vs rating](#win-rate-vs-rating) · [np.add.at](#npaddat) · [MockLLM](#mockllm) · [ValueError: No JSON object found in model reply](#valueerror-no-json-object-found-in-model-reply)

---

## LLM-as-judge  `(concept)`
**1. Definition:** Using a language model to score answers (pointwise against a rubric) or choose between two (pairwise).
**2. In plain words:** A model grading homework.
**3. Example to quote:** > "A judge is an instrument: calibrate it against human labels before trusting it."
```python
reply = generate(messages, backend='mock', mock=judge)
```
**Seen in:** `main.py`, sections 3-5

## Pointwise vs pairwise judging  `(concept)`
**1. Definition:** Pointwise: one answer in, a score out. Pairwise: two answers in, a winner out.
**2. In plain words:** A grade vs a head-to-head.
**3. Example to quote:** > "Pairwise is better at fine differences; pointwise is better for absolute tracking."
**Seen in:** `WALKTHROUGH.md` theory

## Rubric prompt  `(concept)`
**1. Definition:** A judge prompt with explicit scoring criteria and an output format.
**2. In plain words:** A marking scheme.
**3. Example to quote:** > "The rubric awarded +2 for the correct number of days and asked for JSON only."
**Seen in:** `main.py`, `RUBRIC`

## extract_json  `(syntax)`
**1. Definition:** A helper in shared/llm_client.py that pulls the first {...} object from a chatty reply and parses it.
**2. In plain words:** Find the JSON hiding in the chatter.
**3. Example to quote:** > "It raised ValueError on the reply 'Score: 5'."
```python
extract_json(reply)['score']
```
**Seen in:** `main.py`, `judge_score`

## Position bias  `(concept)`
**1. Definition:** A judge's tendency to prefer an answer because of its slot; revealed by swapping order.
**2. In plain words:** Always picking the left-hand one.
**3. Example to quote:** > "With bias 0.3 the judge said A in 82% of 80 calls and only 35% of verdicts survived the swap."
```python
r2 = pairwise(q, second, first)
```
**Seen in:** `main.py`, section 5

## Verbosity bias  `(concept)`
**1. Definition:** A judge's tendency to prefer longer answers.
**2. In plain words:** Longer looks smarter.
**3. Example to quote:** > "The mock judge adds 0.01 per word, so padded answers get a boost."
**Seen in:** `main.py`, `mock_judge`

## Self-preference bias  `(concept)`
**1. Definition:** A judge favouring outputs in its own style or from its own model family.
**2. In plain words:** Marking your own homework.
**3. Example to quote:** > "Mitigate by using a judge from another family or an ensemble."
**Seen in:** `WALKTHROUGH.md` theory

## Debiased verdict (both-orders agreement)  `(concept)`
**1. Definition:** Count a win only if the same answer wins in both presentation orders; otherwise record a tie.
**2. In plain words:** Two witnesses must agree.
**3. Example to quote:** > "Only 0.35 of verdicts survived the swap in the default run."
```python
debiased_right += win1 and win2
```
**Seen in:** `main.py`, section 5

## Spearman correlation  `(concept)`
**1. Definition:** Pearson correlation of ranks; measures monotonic agreement.
**2. In plain words:** Do the two lists order things the same way?
**3. Example to quote:** > "The judge's scores correlated 1.00 with the labels in rank terms."
```python
spearmanr(a, b).statistic
```
**Seen in:** `main.py`, section 4

## Calibration against human labels  `(concept)`
**1. Definition:** Comparing judge outputs to a hand-labelled sample to estimate agreement or correlation.
**2. In plain words:** Check the instrument against a known standard.
**3. Example to quote:** > "50-200 hand labels are typically enough to find gross problems."
**Seen in:** `WALKTHROUGH.md` theory

## Elo rating  `(concept)`
**1. Definition:** An online rating: r_i += K*(w - e), e = 1/(1+10^((r_j-r_i)/400)).
**2. In plain words:** Chess-style points after each game.
**3. Example to quote:** > "Across 30 shuffles of the same votes the standard deviation was 20-31 points per model."
```python
e = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))
```
**Seen in:** `main.py`, `elo`

## Bradley-Terry model  `(concept)`
**1. Definition:** P(i beats j) = sigmoid(theta_i - theta_j), fitted by maximum likelihood over all votes.
**2. In plain words:** Strengths that explain who beats whom.
**3. Example to quote:** > "BT recovered the true ordering and, unlike Elo, does not depend on vote order."
```python
p = 1 / (1 + np.exp(-(theta[I] - theta[J])))
```
**Seen in:** `main.py`, `bradley_terry`

## Bootstrap interval for ratings  `(concept)`
**1. Definition:** Resample votes with replacement, refit, take percentiles of each rating.
**2. In plain words:** How much would the ranking move with different votes?
**3. Example to quote:** > "BT intervals were roughly 40-65 points wide for 1,500 votes."
```python
np.percentile(boot, [2.5, 97.5], axis=0)
```
**Seen in:** `main.py`, section 9

## Arena (formerly LMArena, Chatbot Arena)  `(concept)`
**1. Definition:** A public platform where users vote between anonymous model answers; rankings are derived from pairwise votes. It rebranded to Arena in January 2026 (Wikipedia mirror; check the site).
**2. In plain words:** The crowd-sourced head-to-head leaderboard.
**3. Example to quote:** > "Its known weakness is style and length bias."
**Seen in:** `WALKTHROUGH.md` theory

## Win rate vs rating  `(concept)`
**1. Definition:** Win rate is for one pair of models; a rating summarises a whole field.
**2. In plain words:** One matchup vs the league table.
**3. Example to quote:** > "A win rate is a statement about one pair of models; a rating compresses all the pairs into one number per model."
**Seen in:** `WALKTHROUGH.md`

## np.add.at  `(library)`
**1. Definition:** Unbuffered in-place add at repeated indices.
**2. In plain words:** Add up contributions that share the same slot.
**3. Example to quote:** > "g[I] += v would drop repeated indices; np.add.at does not."
```python
np.add.at(g, I, W - p)
```
**Seen in:** `main.py`, `bradley_terry`

## MockLLM  `(library)`
**1. Definition:** A fake model from shared/llm_client.py driven by scripted replies or regex rules.
**2. In plain words:** A puppet that lets the pipeline run offline.
**3. Example to quote:** > "The judge is a rule-based callable behind a single regex."
```python
MockLLM(rules=[(r'.', mock_judge)])
```
**Seen in:** `main.py`, section 3

## ValueError: No JSON object found in model reply  `(error)`
**1. Definition:** extract_json found no braces in the reply.
**2. In plain words:** The judge answered in prose.
**3. Example to quote:** > "Retry, use structured output, or reject the sample; never default to zero."
```python
ValueError: No JSON object found in model reply: 'Score: 5'
```
**Seen in:** `WALKTHROUGH.md`, debugging practice
