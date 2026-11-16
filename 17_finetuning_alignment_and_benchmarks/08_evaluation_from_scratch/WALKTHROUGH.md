# Step 08 — Evaluation from scratch

## The story so far
[Step 07](../07_distillation_and_merging/WALKTHROUGH.md) produced, shrank and merged models, and each step ended with a number: an accuracy, a loss, a KL. Nobody asked how trustworthy those numbers were. Every earlier step quietly used one or two metrics; this step builds the whole toolbox that sits behind leaderboards and then shows how much *uncertainty* each number carries.

## The pain point
"gpt2 scores 60% on our quiz" sounds like a fact. It is not, until you say: scored how (sum of log-probabilities, per-token, per-byte, answer letter, or generated text and a regex)? On how many items? With what confidence interval? Was any of the quiz in the training data? Different answers to those questions change the score by tens of points, and with 30 questions a 10-point gap can be pure luck.

## The spark
Write every metric yourself: it is a few lines each, and once you have, the vocabulary of evaluation stops being magic. Then attach a bootstrap or Wilson interval to every accuracy, compare systems with a *paired* bootstrap, and check for contamination with n-gram overlap.

## Goal
On `gpt2` and a 30-question hand-written multiple-choice set, compute: perplexity (briefly; the theory is in module 06), exact match and token-F1 with SQuAD-style normalisation, multiple-choice accuracy by log-likelihood in four scorings (raw `acc`, per-token, per-byte `acc_norm_bytes`, and MMLU-style answer letters), 95% confidence intervals (bootstrap and Wilson), a paired bootstrap between two scorings, generative evaluation with answer extraction, the unbiased `pass@k` estimator checked against a Monte-Carlo simulation, and an n-gram contamination check.

## The theory (read this before the interview)
**Perplexity.** `ppl = exp( mean token cross-entropy )`. Derivation, meaning and the checkpoint workflow are in [06 step 03](../../06_llm_training_and_lora/03_eval_perplexity_checkpoints/WALKTHROUGH.md); here it is computed once for completeness and again as a regression probe in [step 11](../11_end_to_end_report_card/WALKTHROUGH.md). Perplexity is only comparable between models that share a tokenizer and a test text.

**Exact match (EM) and token-F1** (extractive-QA style). After *normalisation* (lowercase, strip punctuation, drop the articles a/an/the, collapse spaces): `EM = 1` iff prediction equals gold. `F1` treats both as bags of words: `precision = overlap / len(pred)`, `recall = overlap / len(gold)`, `F1 = 2PR / (P + R)`.

**Multiple choice by log-likelihood** (the `lm-evaluation-harness` way). Given a question, score each candidate continuation `c` by `log P(c | question)` (sum over its tokens) and pick the highest:
- `acc` = argmax of the raw sum. Longer choices have more tokens and so a lower total; this *favours short answers*.
- `acc_per_token` = argmax of `sum / n_tokens`.
- `acc_norm`-style = divide by the *length of the choice text* (in `lm-evaluation-harness`, `acc_norm` normalises by the length of the answer text; this file divides by the **byte** length and calls it `acc_norm_bytes`).
- **Letters (MMLU style)**: show all options as `A. ... B. ...`, then compare `log P(" A")`, `log P(" B")`, ... Needs a model that can handle the format (small base models are near chance).
Which scoring you use changes the score, so a leaderboard number is only reproducible with its scoring named.

**Generative evaluation:** let the model generate (greedy), *extract* the answer (here: first line), normalise and compare. The extraction rule is part of the benchmark: a rule that is too strict punishes correct answers, one too lenient rewards rambling.

**pass@k (Chen et al. 2021).** Generate `n` samples for a problem, `c` of them correct. The probability that at least one of `k` randomly chosen samples is correct is
```
pass@k = 1 - C(n - c, k) / C(n, k)   =   1 - prod_{i = n-c+1}^{n} ( 1 - k / i )
```
The product form is numerically stable. Computing "fraction of problems solved in the first k tries" instead is a *biased* estimator for k < n. If `n - c < k`, every size-k subset contains a correct sample and `pass@k = 1`.

**Uncertainty.** With `n` items and accuracy `p`, the standard error is about `sqrt(p (1 - p) / n)`: for `n = 30`, `p = 0.7` that is about 8.4 points, so a 95% interval is roughly +-16 points.
- **Percentile bootstrap:** resample the `n` items *with replacement* many times, compute the accuracy each time, take the 2.5% and 97.5% quantiles.
- **Wilson interval:** a closed-form interval for a proportion that behaves better than the textbook normal interval for small `n` or `p` near 0 or 1.
- **Paired bootstrap:** to compare systems A and B on the *same* items, resample item indices once and compute `mean(A) - mean(B)` on the same resample; items that both systems get right or wrong cancel, so the interval is tighter than comparing two separate intervals.

**Contamination:** if benchmark items were in the training data the score measures memory, not ability. A cheap check is the n-gram overlap between each test item and the training text. It catches verbatim leakage; it misses paraphrases and translations.

**Harness settings that change scores:** few-shot count, prompt wording, normalisation, answer extraction, scoring rule, chat template on or off, sampling versus greedy, number of samples. Report them. [Step 10](../10_benchmark_landscape_2026_and_tools/WALKTHROUGH.md) names the standard benchmarks and the `lm-evaluation-harness` tool that implements these scorings.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Perplexity | ppl, `exp(cross-entropy)` | yes | Exponentiated mean token loss, covered in [06 step 03](../../06_llm_training_and_lora/03_eval_perplexity_checkpoints/WALKTHROUGH.md) |
| Exact match | EM | yes | 1 if normalised strings are equal |
| Token-F1 | F1, word-overlap F1 | near | Harmonic mean of word precision and recall |
| `acc` / `acc_norm` | raw / length-normalised log-likelihood accuracy | no | `acc_norm` divides by choice length so long answers are not penalised |
| Log-likelihood MCQ | ranked choice, cloze-style MCQ | near | Score each choice by probability; no generation |
| Letter-scored MCQ | MMLU style, multiple-choice prompt format | no | Model must map options to letters |
| Generative eval | free-form eval, generation-based | no | Model writes; a rule extracts and compares |
| pass@k | coding success at k samples | yes | Probability at least one of k samples is correct |
| Bootstrap CI | percentile bootstrap, resampling interval | yes | Quantiles of the statistic over resamples |
| Wilson interval | Wilson score interval | yes | Closed-form CI for a proportion |
| Paired bootstrap | paired resampling test | near | Resamples the same items for both systems |
| Contamination | data leakage, benchmark leakage, test-set leakage | near | Test items present in training data |
| Held-out set | test set, validation set | near | Data not used for training or tuning |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/08_evaluation_from_scratch
python main.py
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/08_evaluation_from_scratch/main.py
# knobs: SEED, MODEL (default gpt2)
```
About a minute of compute on an idle laptop CPU (it runs `gpt2` on 30 questions x 4 choices x 2 formats, plus 8 generations); varies by machine and load.

## Code, step by step

Lines 1–13 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports, settings and the model  (`main.py` lines 14–31)
```python
import math
import os
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = int(os.environ.get("SEED", 0))
MODEL = os.environ.get("MODEL", "gpt2")
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL).eval()
```
- **In simple words:** Standard imports, a seed for torch and numpy, and the model under test.
- **Exactly:** `re` (regular expressions) and `string.punctuation` are for normalisation; `Counter` is a multiset used for token overlap; `np.random.default_rng(SEED)` creates a seeded NumPy generator (used by the pass@k check). `.eval()` turns dropout off, so scores are deterministic. `MODEL` can be changed to `distilgpt2` (see Experiments).
- **Why it is here:** The model under test is a parameter, so the same harness can score any causal LM.

### 1. A hand-written benchmark: (question, 4 choices, index of the right one)  (`main.py` lines 33–65)
```python
# ## 1. A hand-written benchmark: (question, 4 choices, index of the right one)
MCQ = [
    ("The capital of France is", ["Paris", "Rome", "Madrid", "Berlin"], 0),
    ("The capital of Italy is", ["Rome", "Paris", "Vienna", "Athens"], 0),
    ("The capital of Japan is", ["Seoul", "Tokyo", "Beijing", "Bangkok"], 1),
    ("The capital of Egypt is", ["Cairo", "Lima", "Oslo", "Delhi"], 0),
    ("Water freezes at", ["0 degrees Celsius", "50 degrees Celsius", "100 degrees Celsius", "200 degrees Celsius"], 0),
    ("The largest planet in our solar system is", ["Mars", "Venus", "Jupiter", "Mercury"], 2),
    ("A dog is a kind of", ["fish", "mammal", "insect", "plant"], 1),
    ("The sun rises in the", ["west", "north", "south", "east"], 3),
    ("Two plus two equals", ["three", "four", "five", "six"], 1),
    ("The chemical symbol for water is", ["H2O", "CO2", "NaCl", "O2"], 0),
    ("A triangle has how many sides?", ["two", "three", "four", "five"], 1),
    ("The opposite of hot is", ["cold", "warm", "red", "loud"], 0),
    ("Birds can usually", ["swim underground", "fly", "breathe fire", "read"], 1),
    ("The first month of the year is", ["March", "June", "January", "December"], 2),
    ("Humans breathe in", ["oxygen", "helium", "gold", "sand"], 0),
    ("The moon orbits the", ["Earth", "Sun", "Jupiter", "Pluto"], 0),
    ("A week has how many days?", ["five", "six", "seven", "ten"], 2),
    ("Bees make", ["honey", "milk", "silk", "oil"], 0),
    ("The color of grass is usually", ["blue", "green", "purple", "black"], 1),
    ("Ice is the solid form of", ["water", "iron", "salt", "wood"], 0),
    ("The capital of Germany is", ["Berlin", "Warsaw", "Prague", "Moscow"], 0),
    ("The capital of Spain is", ["Lisbon", "Madrid", "Dublin", "Athens"], 1),
    ("A year has how many months?", ["ten", "eleven", "twelve", "thirteen"], 2),
    ("Fish live in", ["water", "trees", "clouds", "deserts"], 0),
    ("The author of Hamlet is", ["Dickens", "Austen", "Shakespeare", "Orwell"], 2),
    ("Plants need sunlight to make", ["food", "metal", "plastic", "noise"], 0),
    ("The capital of Canada is", ["Ottawa", "Sydney", "Boston", "Lima"], 0),
    ("Spiders have how many legs?", ["four", "six", "eight", "ten"], 2),
    ("The currency of the USA is the", ["euro", "dollar", "yen", "peso"], 1),
    ("The Pacific is an", ["ocean", "desert", "mountain", "island"], 0),
]
```
- **In simple words:** Thirty easy questions with four options each; the last number says which option is right.
- **Exactly:** `MCQ` is a list of 30 tuples `(question, [choice0..choice3], gold_index)`. The correct option sits at different indices in different items (counted from the file: index 0 for 15 items, 1 for 8, 2 for 6, 3 for 1). Chance is 25%, but note the skew: a scorer that always answered "A" would get 15 of 30 = 50% without knowing anything, which matters for the letter-scored variant below.
- **Why it is here:** A small, readable benchmark lets you inspect every item, which is the point of building it from scratch. It is *deliberately* small: at `n = 30` the confidence intervals will be wide, which is itself the lesson. (The same 30 items are reused as `GENERAL_MCQ` in `evalkit.py`.)

### 2. Log-likelihood helper: log P(continuation | context)  (`main.py` lines 67–75)
```python
# ## 2. Log-likelihood helper: log P(continuation | context), summed over continuation tokens
@torch.no_grad()
def cont_logprob(context, continuation):
    c = tok(context, return_tensors="pt").input_ids                   # (1, Lc)
    k = tok(continuation, return_tensors="pt").input_ids              # (1, Lk)
    ids = torch.cat([c, k], 1)                                         # (1, Lc+Lk)
    logp = F.log_softmax(lm(ids).logits[:, :-1].float(), -1)           # (1, L-1, vocab)
    tok_lp = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)             # (1, L-1)
    return tok_lp[0, c.shape[1] - 1:].sum().item(), k.shape[1]         # (sum log-prob, number of continuation tokens)
```
- **In simple words:** How likely is it that the model, having read the context, continues with exactly these words? Add up the log-probability of each continuation token.
- **Exactly:** Tokenise context and continuation *separately* and concatenate the ids: `ids` is `(1, Lc+Lk)`. `lm(ids).logits` is `(1, L, vocab)`; `[:, :-1]` keeps the predictions for positions 1..L-1; `log_softmax` turns scores into log-probabilities; `.gather(-1, ids[:, 1:, None])` picks, at each position, the log-probability of the token that actually came next, giving `(1, L-1)`. Slicing from `c.shape[1] - 1` keeps only the continuation tokens (the first continuation token is predicted at index `Lc - 1`), and `.sum()` adds them. Returns `(float, int)`. This is the same computation as `seq_logprob` in `common.py`, written for floats instead of tensors with gradient. Tokenising separately is intentional: the continuation begins with a space (`" Paris"`) so that it tokenises the way it would inside a sentence.
- **Why it is here:** Every multiple-choice scoring below is a function of these two numbers.

### 3. Perplexity = exp(mean token cross-entropy)  (`main.py` lines 77–87)
```python
# ## 3. Perplexity = exp(mean token cross-entropy)
corpus = Path(__file__).resolve().parents[2] / "06_llm_training_and_lora" / "data" / "tiny_corpus.txt"
texts = [l.strip() for l in corpus.read_text().splitlines() if l.strip()][:20]
nll, ntok = 0.0, 0
for t in texts:
    ids = tok(t, return_tensors="pt").input_ids
    with torch.no_grad():
        loss = lm(ids, labels=ids).loss                                # mean CE over L-1 predicted tokens
    nll += loss.item() * (ids.shape[1] - 1)
    ntok += ids.shape[1] - 1
print(f"perplexity on 20 corpus lines: {math.exp(nll / ntok):.1f}   (cross-entropy {nll / ntok:.3f} nats/token)")
```
- **In simple words:** How "surprised" is gpt2 by 20 plain sentences? Average the surprise per token, then exponentiate.
- **Exactly:** Passing `labels=ids` makes the model compute its shifted cross-entropy and return the *mean over the L-1 predicted tokens*. Multiplying by `L-1` turns each line's mean back into a total, so the final average is *per token over the whole corpus* (not per line, which would over-weight short lines). `exp(mean CE)` is perplexity: the effective number of equally likely choices per token. The theory and a worked example are in [06 step 03](../../06_llm_training_and_lora/03_eval_perplexity_checkpoints/WALKTHROUGH.md); nothing new is taught here.
- **Why it is here:** Completeness, and it defines `texts`, which block 10 reuses as a stand-in training corpus for the contamination check.

### 4. Exact match and token-F1 (SQuAD-style normalisation)  (`main.py` lines 89–110)
```python
# ## 4. Exact match and token-F1 (SQuAD-style normalisation)
def normalise(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))     # drop punctuation
    s = re.sub(r"\b(a|an|the)\b", " ", s)                              # drop articles
    return " ".join(s.split())                                          # collapse whitespace

def exact_match(pred, gold):
    return float(normalise(pred) == normalise(gold))

def token_f1(pred, gold):
    p, g = normalise(pred).split(), normalise(gold).split()
    common = Counter(p) & Counter(g)                                    # multiset overlap
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)

print(f"\n{'prediction':<26}{'gold':<16}{'EM':>4}{'F1':>7}")
for pred, gold in [("Paris", "paris"), ("The Paris.", "Paris"), ("Paris, France", "Paris"), ("in Paris city", "Paris"), ("Rome", "Paris")]:
    print(f"{pred!r:<26}{gold!r:<16}{exact_match(pred, gold):>4.0f}{token_f1(pred, gold):>7.2f}")
```
- **In simple words:** Two ways to grade a typed answer: all-or-nothing (EM) and partial credit by shared words (F1), after tidying case, punctuation and articles.
- **Exactly:** `normalise` lowercases, removes punctuation characters, replaces whole-word `a / an / the` (regex `\b` = word boundary) by a space, and collapses whitespace with `" ".join(s.split())`. `Counter(p) & Counter(g)` is the multiset intersection (minimum count per word); its total is the overlap. Precision is overlap over predicted words, recall overlap over gold words; F1 is their harmonic mean. If there is no overlap, F1 is 0 (avoids division by zero). The five test cases show the behaviour: case and article differences vanish, extra words keep EM at 0 but give partial F1, a wrong answer scores 0 on both.
- **Why it is here:** EM and F1 are the metrics of most QA benchmarks; the normalisation function is where many "scores differ between papers" stories come from.

### 5. Multiple choice by log-likelihood (lm-evaluation-harness style)  (`main.py` lines 112–126)
```python
# ## 5. Multiple choice by log-likelihood (lm-evaluation-harness style)
def mcq_predictions():
    out = {"acc": [], "acc_per_token": [], "acc_norm_bytes": [], "letters": []}
    for q, choices, gold in MCQ:
        ctx = f"Question: {q}\nAnswer:"
        scored = [cont_logprob(ctx, " " + c) for c in choices]          # [(logprob, n_tokens), ...]
        out["acc"].append(int(np.argmax([s for s, _ in scored]) == gold))                                   # raw sum
        out["acc_per_token"].append(int(np.argmax([s / n for s, n in scored]) == gold))                      # per-token mean
        out["acc_norm_bytes"].append(int(np.argmax([s / len(c.encode()) for (s, _), c in zip(scored, choices)]) == gold))  # per byte
        # MMLU style: show all options, then compare log P(" A"), " B", ...
        menu = f"{q}\n" + "\n".join(f"{L}. {c}" for L, c in zip("ABCD", choices)) + "\nAnswer:"
        letter = [cont_logprob(menu, f" {L}")[0] for L in "ABCD"]
        out["letters"].append(int(np.argmax(letter) == gold))
    return {k: np.array(v) for k, v in out.items()}
res = mcq_predictions()
```
- **In simple words:** For each question, score each of the four answers in four different ways and record 1 (right) or 0 (wrong) per question for each way.
- **Exactly:** `ctx` is the prompt `Question: ...\nAnswer:`; each choice is scored as a continuation with a leading space. `scored` is a list of four `(sum_logprob, n_tokens)` pairs. The first three scorings differ only in the denominator: 1 (raw), the token count, the UTF-8 byte length of the choice (`len(c.encode())`). `np.argmax(...) == gold` is the prediction check. The *letters* scoring builds a menu (`"A. Paris\nB. Rome..."`) and compares the log-probability of the four letters; `[0]` takes only the log-prob from the `(logprob, n)` tuple. The function returns one 0/1 NumPy array per scoring (length 30).
- **Why it is here:** This is exactly how `lm-evaluation-harness` scores `acc` and `acc_norm` for many multiple-choice tasks, and the letters variant is how MMLU-style prompts are scored. The same model gets different accuracies from the same questions depending on the denominator.

### 6. Confidence intervals: bootstrap (percentile) and Wilson  (`main.py` lines 128–145)
```python
# ## 6. Confidence intervals: bootstrap (percentile) and Wilson
def bootstrap_ci(correct, n_boot=4000, alpha=0.05, seed=0):
    r = np.random.default_rng(seed)
    means = r.choice(correct, size=(n_boot, len(correct)), replace=True).mean(1)    # resample questions with replacement
    return np.quantile(means, [alpha / 2, 1 - alpha / 2])

def wilson_ci(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half

print(f"\nMCQ on n={len(MCQ)} questions (chance = 0.25), model {MODEL}")
print(f"{'scoring':<18}{'acc':>6}   {'bootstrap 95% CI':<20}{'Wilson 95% CI'}")
for name, c in res.items():
    lo, hi = bootstrap_ci(c)
    wl, wh = wilson_ci(int(c.sum()), len(c))
    print(f"{name:<18}{c.mean():>6.3f}   [{lo:.2f}, {hi:.2f}]{'':<8}[{wl:.2f}, {wh:.2f}]")
```
- **In simple words:** Two ways to say "the true accuracy is probably between X and Y".
- **Exactly:** *Bootstrap:* `r.choice(correct, size=(4000, 30), replace=True)` draws 4,000 resamples of the 30 outcomes (a `(4000, 30)` array); `.mean(1)` gives 4,000 accuracies; `np.quantile(..., [0.025, 0.975])` reads off the middle 95%. *Wilson:* closed-form with `z = 1.96` for 95%: the centre is pulled toward 0.5 and the width accounts for small `n`, so it never leaves `[0, 1]` and is not degenerate at `p = 0` or `1` (the plain bootstrap collapses to a point there). The loop prints each scoring's accuracy and both intervals.
- **Why it is here:** The point of the table is how *wide* the intervals are at `n = 30`. A gap smaller than the interval is not evidence.

### 7. Paired bootstrap: is scoring A really better than scoring B?  (`main.py` lines 147–154)
```python
# ## 7. Paired bootstrap: is scoring A really better than scoring B?
def paired_bootstrap(a, b, n_boot=4000, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), size=(n_boot, len(a)))                  # SAME resampled questions for both systems
    diff = a[idx].mean(1) - b[idx].mean(1)
    return diff.mean(), np.quantile(diff, [0.025, 0.975]), (diff <= 0).mean()
d, ci, p = paired_bootstrap(res["acc_norm_bytes"], res["acc"])
print(f"acc_norm_bytes - acc = {d:+.3f}, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], P(diff<=0) = {p:.2f}")
```
- **In simple words:** Compare two scorings on the *same* questions, resampling questions once for both, so shared easy and hard questions cancel out.
- **Exactly:** `idx` is a `(4000, 30)` array of random question indices; `a[idx]` and `b[idx]` index both 0/1 vectors with the *same* indices; the difference of the two means is computed per resample. It returns the mean difference, its 95% interval and the share of resamples where `a` was not better (`diff <= 0`), a bootstrap analogue of a one-sided p-value (not a formal test). If the interval contains 0, the difference is within noise.
- **Why it is here:** "Method B is better" is the most common unsupported claim in ML. This is the check. [Step 11](../11_end_to_end_report_card/WALKTHROUGH.md) uses the same function for before-versus-after.

### 8. Generative evaluation with answer extraction (3-shot, greedy)  (`main.py` lines 156–170)
```python
# ## 8. Generative evaluation with answer extraction (3-shot, greedy)
FEWSHOT = "Q: The capital of Italy is?\nA: Rome\n\nQ: A week has how many days?\nA: seven\n\nQ: The color of grass is usually?\nA: green\n\n"
GEN_QA = [("The capital of France is?", "Paris"), ("The capital of Japan is?", "Tokyo"), ("The capital of Egypt is?", "Cairo"),
          ("The capital of Germany is?", "Berlin"), ("A triangle has how many sides?", "three"), ("The opposite of hot is?", "cold"),
          ("The first month of the year is?", "January"), ("Bees make?", "honey")]
em, f1 = [], []
for q, gold in GEN_QA:
    enc = tok(FEWSHOT + f"Q: {q}\nA:", return_tensors="pt")
    with torch.no_grad():
        out = lm.generate(**enc, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0, enc.input_ids.shape[1]:])
    ans = raw.split("\n")[0].strip()                                    # extraction rule: first line only
    em.append(exact_match(ans, gold)); f1.append(token_f1(ans, gold))
    print(f"  {q:<36} raw={raw!r:<22} extracted={ans!r:<10} gold={gold!r}")
print(f"generative EM {np.mean(em):.3f}  F1 {np.mean(f1):.3f}  (n={len(GEN_QA)})")
```
- **In simple words:** Show the model three solved examples, ask a new question, let it write up to six tokens, cut the answer at the first newline and grade it with EM and F1.
- **Exactly:** `FEWSHOT` is the in-context demonstration (3-shot) in the format `Q: ...\nA: ...`; each test question is appended as `Q: ...\nA:`. `generate(..., do_sample=False, max_new_tokens=6)` is greedy decoding. `raw` is only the new tokens; `.split("\n")[0].strip()` is the **extraction rule** (first line, trimmed). The printed line shows raw output, extracted answer and gold so you can *see* extraction failures. n = 8 questions, so one question is 12.5 points.
- **Why it is here:** This is generative evaluation as practised by free-form benchmarks. Compare its score with the ranked-choice scores above: the same knowledge, different protocol, different number.

### 9. pass@k: unbiased estimator from n samples with c correct  (`main.py` lines 172–183)
```python
# ## 9. pass@k: unbiased estimator from n samples with c correct
def pass_at_k(n, c, k):
    """1 - C(n-c, k) / C(n, k), computed as a stable product (Chen et al. 2021)."""
    if n - c < k:
        return 1.0                                                      # every size-k subset contains a correct sample (the product below would also give this, via a zero factor, but k > n would not)
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
print(f"\npass@k from n=10 samples:  {'c':>3}  " + "  ".join(f"k={k:<2}" for k in (1, 2, 5, 10)))
for c in (0, 1, 3, 5, 10):
    print(f"{'':<26}{c:>3}  " + "  ".join(f"{pass_at_k(10, c, k):.3f}" for k in (1, 2, 5, 10)))
# Monte-Carlo check of the formula for n=10, c=3, k=5: draw k of the n samples at random, see if any is correct
mc = np.mean([rng.permutation([1] * 3 + [0] * 7)[:5].any() for _ in range(20000)])
print(f"check n=10 c=3 k=5: formula {pass_at_k(10, 3, 5):.4f} vs Monte-Carlo {mc:.4f}")
```
- **In simple words:** If you tried a problem 10 times and succeeded 3 times, what is the chance that 5 randomly chosen attempts include a success? A formula answers it exactly; a simulation checks the formula.
- **Exactly:** `1 - C(n-c, k) / C(n, k)` equals `1 - prod_{i=n-c+1}^{n} (1 - k/i)`, computed with `np.arange(n - c + 1, n + 1)` (no huge factorials). The `n - c < k` guard returns 1.0 when fewer than `k` failures exist. The table prints `pass@k` for `c` correct out of 10 and `k` in 1, 2, 5, 10. The Monte-Carlo line draws a random permutation of a list with 3 ones and 7 zeros 20,000 times, takes the first 5 and checks whether any is a one (`.any()`); its mean estimates the same probability.
- **Why it is here:** The formula is a classic interview question, and the simulation proves you derived it right.

### 10. Contamination: n-gram overlap between test items and a training corpus  (`main.py` lines 185–200)
```python
# ## 10. Contamination: n-gram overlap between test items and a training corpus
def ngrams(text, n=8):
    w = normalise(text).split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}
train_corpus = " ".join(texts) + " Quiz notes: Question: The capital of Japan is Answer: Tokyo. Question: Spiders have how many legs? Answer: eight."
corpus_grams = ngrams(train_corpus, 5)
print("\ncontamination check (5-gram word overlap of 'Question: <q> Answer: <correct choice>' with the training text):")
flagged = 0
for q, choices, gold in MCQ:
    item = f"Question: {q} Answer: {choices[gold]}"
    g = ngrams(item, 5)
    frac = len(g & corpus_grams) / max(1, len(g))
    if frac > 0:
        flagged += 1
        print(f"  overlap {frac:.2f}  {item!r}")
print(f"flagged {flagged} of {len(MCQ)} items")
```
- **In simple words:** Pretend the 20 corpus lines plus two quiz items are the training data, then check which test items share a run of five words with it.
- **Exactly:** `ngrams(text, n)` normalises, splits into words and returns the *set* of all `n`-word windows. `train_corpus` deliberately has two test items planted ("Japan ... Tokyo" and "Spiders ... eight"). For each test item the code builds `"Question: <q> Answer: <gold>"`, computes its 5-grams and the fraction found in the corpus's 5-grams (`g & corpus_grams` is set intersection). Any item with a positive overlap is flagged. A planted leak should be caught; unrelated items should not.
- **Why it is here:** Contamination is the main reason published scores go stale; verbatim n-gram overlap is the simplest check (it misses paraphrases). The numbers print only the flagged items.

### 11. TRY IT / BREAK IT comments  (`main.py` lines 202–203)
```python
# TRY IT: MODEL=distilgpt2 python main.py ; change the extraction rule to take only the first word.
# BREAK IT: change `np.arange(n - c + 1, n + 1)` to `np.arange(n - c, n + 1)` (an off-by-one) and compare the formula with the Monte-Carlo line.
```
- **In simple words:** Experiments; see below.
- **Exactly:** Comments only.
- **Why it is here:** Module convention.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `ids[:, 1:, None]` | next-token ids with a trailing axis, shape `(1, L-1, 1)`, for `gather` |
| `logp.gather(-1, idx)` | pick `logp[..., idx]` along the last axis |
| `Counter(a) & Counter(b)` | multiset intersection (minimum counts) |
| `{" ".join(...) for i in range(...)}` | a set comprehension (unique n-grams) |
| `a[idx]` with an integer array | fancy indexing: gather items by index (with repetition) |
| `r.choice(x, size=(B, n), replace=True)` | sample with replacement into a `(B, n)` array |
| `np.quantile(x, [0.025, 0.975])` | the 2.5% and 97.5% quantiles |
| `\b(a\|an\|the)\b` | regex: the whole words a, an, the |
| `f"{pred!r:<26}"` | `repr` of a value, left-aligned in 26 characters |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `model(ids, labels=ids).loss` | mean next-token cross-entropy | shifts logits and labels by one and averages over predicted positions |
| `np.random.default_rng(seed)` | reproducible random generator | independent of the global NumPy state |
| `np.argmax` | index of the largest value | first index on ties |
| `np.quantile` | empirical quantile | linear interpolation between order statistics by default |
| `np.prod` | product of an array | used for the stable pass@k product |
| `model.generate` | autoregressive decoding | repeated forward passes with a KV cache; greedy when `do_sample=False` |

## Execution trace (real run)
```
[transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.
perplexity on 20 corpus lines: 62.7   (cross-entropy 4.138 nats/token)

prediction                gold              EM     F1
'Paris'                   'paris'            1   1.00
'The Paris.'              'Paris'            1   1.00
'Paris, France'           'Paris'            0   0.67
'in Paris city'           'Paris'            0   0.50
'Rome'                    'Paris'            0   0.00

MCQ on n=30 questions (chance = 0.25), model gpt2
scoring              acc   bootstrap 95% CI    Wilson 95% CI
acc                0.667   [0.50, 0.83]        [0.49, 0.81]
acc_per_token      0.600   [0.43, 0.77]        [0.42, 0.75]
acc_norm_bytes     0.700   [0.53, 0.87]        [0.52, 0.83]
letters            0.500   [0.33, 0.67]        [0.33, 0.67]
acc_norm_bytes - acc = +0.031, 95% CI [-0.13, +0.20], P(diff<=0) = 0.43
  The capital of France is?            raw=' Paris\n\nQ: The'     extracted='Paris'    gold='Paris'
  The capital of Japan is?             raw=' Tokyo\n\nQ: The'     extracted='Tokyo'    gold='Tokyo'
  The capital of Egypt is?             raw=' Cairo\n\nQ: The'     extracted='Cairo'    gold='Cairo'
  The capital of Germany is?           raw=' Berlin\n\nQ: The'    extracted='Berlin'   gold='Berlin'
  A triangle has how many sides?       raw=' two\n\nQ: The'       extracted='two'      gold='three'
  The opposite of hot is?              raw=' red\n\nQ: The'       extracted='red'      gold='cold'
  The first month of the year is?      raw=' January\n\nQ: The'   extracted='January'  gold='January'
  Bees make?                           raw=' Yes\n\nQ: What'      extracted='Yes'      gold='honey'
generative EM 0.625  F1 0.625  (n=8)

pass@k from n=10 samples:    c  k=1   k=2   k=5   k=10
                            0  0.000  0.000  0.000  0.000
                            1  0.100  0.200  0.500  1.000
                            3  0.300  0.533  0.917  1.000
                            5  0.500  0.778  0.996  1.000
                           10  1.000  1.000  1.000  1.000
check n=10 c=3 k=5: formula 0.9167 vs Monte-Carlo 0.9137

contamination check (5-gram word overlap of 'Question: <q> Answer: <correct choice>' with the training text):
  overlap 1.00  'Question: The capital of Japan is Answer: Tokyo'
  overlap 1.00  'Question: Spiders have how many legs? Answer: eight'
flagged 2 of 30 items
```
(Progress bars omitted; the `transformers` `loss_type` note is a real stderr line.)
- **Perplexity** 62.7 (4.138 nats per token) on 20 plain lines: gpt2 is "choosing between about 63 tokens" per position on this text. (Meaning and caveats: [06 step 03](../../06_llm_training_and_lora/03_eval_perplexity_checkpoints/WALKTHROUGH.md).)
- **EM/F1 table**: `'The Paris.'` matches `'Paris'` exactly after normalisation (EM 1); `'Paris, France'` has EM 0 but F1 0.67 (precision 0.5, recall 1); `'in Paris city'` F1 0.50 (1 shared word of 3 predicted, recall 1); `'Rome'` scores 0 and 0.
- **MCQ accuracy by scoring**: `acc` 0.667, `acc_per_token` 0.600, `acc_norm_bytes` 0.700, `letters` 0.500 (chance is 0.25). The same model and questions score between 0.50 and 0.70 depending on the protocol. Every bootstrap interval is about 0.33 wide ([0.50, 0.83] for `acc`), and the Wilson intervals are very close to them. The letters scoring is lowest, consistent with a small base model handling the "A. B. C. D." format poorly (not tested further here).
- **Paired bootstrap**: `acc_norm_bytes - acc = +0.031`, interval [-0.13, +0.20]; `P(diff<=0) = 0.43`. The normalised scoring looks 3 points better but the interval contains 0 widely: within noise, as expected for 30 items (one question is 3.3 points).
- **Generative eval**: 5 of 8 correct (EM = F1 = 0.625). The raw text shows why extraction matters: the model keeps writing `\n\nQ: The` after its answer, so cutting at the first newline is what makes the grade possible. The three misses are real errors (`two` for triangle sides, `red` for the opposite of hot, `Yes` for "Bees make?"). n = 8, so one question is 12.5 points.
- **pass@k**: with `c=3` of `n=10`: pass@1 0.300 (equals c/n), pass@5 0.917, pass@10 1.0. The formula 0.9167 versus the Monte-Carlo estimate 0.9137: they agree to within simulation noise (20,000 draws).
- **Contamination**: the two planted items are flagged with overlap 1.00, and 28 of 30 are clean. An overlap of 1.00 means every 5-gram of the item is in the "training" text.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| (measured) four MCQ scorings | 0.667 / 0.600 / 0.700 / 0.500 | the protocol changes the score |
| (measured) paired `acc_norm_bytes - acc` | +0.031, CI [-0.13, +0.20] | within noise at n=30 |
| (not run) `MODEL=distilgpt2` | another model through the same harness; expect a different set of numbers with similarly wide intervals | the harness is model-agnostic |
| (not run) extraction = first word only | answers like `swim underground` would be cut to one word; for this file's short gold answers little would change | extraction rules interact with answer length |

## Debugging practice
**BREAK IT:** change `np.arange(n - c + 1, n + 1)` to `np.arange(n - c, n + 1)` in `pass_at_k`. The product then has one extra factor `1 - k/(n-c)`. By hand for `n=10, c=3, k=5`: the correct product runs over `i = 8, 9, 10` giving `(1-5/8)(1-5/9)(1-5/10) = 0.0833` so `pass@5 = 0.9167` (matches the real run's 0.9167). The off-by-one version adds the factor `i = 7`: `(1-5/7) = 0.2857`, the product becomes `0.0238` and `pass@5` becomes `0.976`, which would be printed next to a Monte-Carlo value near 0.914. (Hand arithmetic, not a separate execution.) No exception is raised; the *only* detector is the Monte-Carlo line, which is why the file prints both. The bug is also visible at the edge: for `c = 0` the correct range is empty (product 1, `pass@5 = 0`), while the broken range contains `i = 10`, giving `1 - (1 - 5/10) = 0.5` for a problem the model never solved (hand arithmetic). Rule: when you implement a formula from a paper, check it against a simulation or a hand case.

## Common mistakes
- Reporting an accuracy without `n` and an interval: at `n = 30`, +-15 points is normal.
- Comparing two models' separate intervals instead of a paired difference on the same items.
- Calling a number "benchmark X accuracy" without naming the scoring (raw, normalised, letters), the few-shot count and the extraction rule.
- Using raw `acc` on choices of very different lengths.
- Computing pass@k as "solved within the first k samples" instead of the unbiased estimator.
- Perplexity across different tokenizers: it is not comparable.
- Checking contamination only by exact string match: paraphrases and translations slip through.
- Testing on the data you tuned prompts or thresholds on.

## Rebuild checklist
1. `cont_logprob(context, continuation)` returning `(sum log-prob, n tokens)`.
2. `normalise`, `exact_match`, `token_f1` with `Counter` overlap.
3. Four MCQ scorings (raw, per-token, per-byte, letters) as 0/1 arrays.
4. `bootstrap_ci`, `wilson_ci`, `paired_bootstrap`.
5. A few-shot prompt, greedy `generate`, and an extraction rule.
6. `pass_at_k` as a product plus a Monte-Carlo check.
7. `ngrams` and an overlap-fraction contamination flag.

## Say it aloud
"Every benchmark number is a protocol plus a sample. I score multiple choice by log-likelihood and say which normalisation I used, because raw sums favour short answers; I attach a bootstrap or Wilson interval, which at 30 items is about plus or minus fifteen points; and I compare two systems with a paired bootstrap on the same items. For code I use the unbiased pass-at-k estimator, one minus C of n minus c choose k over C of n choose k. And I check contamination by n-gram overlap, which catches verbatim leaks but not paraphrases."

## The new problem (cliffhanger)
Everything here has a checkable answer: a label, a string, a unit test. A chat reply ("explain our refund policy politely") has no regex. You need human judges, which are slow, or a model as the judge, which is fast but biased. On to [step 09](../09_llm_as_judge_and_arenas/WALKTHROUGH.md).
