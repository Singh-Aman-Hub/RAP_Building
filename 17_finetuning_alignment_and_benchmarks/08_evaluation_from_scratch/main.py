"""
GOAL:  Score a model the way benchmarks really do it, from scratch, on gpt2 and a hand-written 30-question MCQ set:
       perplexity; exact match and token-F1; multiple-choice by log-likelihood (raw, per-token, per-byte as in lm-evaluation-harness
       "acc" / "acc_norm") and by answer letter (MMLU style); generative eval with answer extraction; unbiased pass@k;
       bootstrap and Wilson confidence intervals, paired bootstrap; and an n-gram contamination check.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/08_evaluation_from_scratch/main.py
EXPECTED OUTPUT (shortened):
    perplexity ~ tens on the tiny corpus; EM/F1 table for 5 hand cases; MCQ accuracy (acc, acc_per_token, acc_norm_bytes, letters)
    each with a 95% CI that is WIDE (n=30); pass@k table matching the formula; contamination flags
NEEDS: torch, transformers, numpy; cached gpt2.  ~1 minute on CPU.
"""
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

# ## 2. Log-likelihood helper: log P(continuation | context), summed over continuation tokens
@torch.no_grad()
def cont_logprob(context, continuation):
    c = tok(context, return_tensors="pt").input_ids                   # (1, Lc)
    k = tok(continuation, return_tensors="pt").input_ids              # (1, Lk)
    ids = torch.cat([c, k], 1)                                         # (1, Lc+Lk)
    logp = F.log_softmax(lm(ids).logits[:, :-1].float(), -1)           # (1, L-1, vocab)
    tok_lp = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)             # (1, L-1)
    return tok_lp[0, c.shape[1] - 1:].sum().item(), k.shape[1]         # (sum log-prob, number of continuation tokens)

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

# ## 7. Paired bootstrap: is scoring A really better than scoring B?
def paired_bootstrap(a, b, n_boot=4000, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(a), size=(n_boot, len(a)))                  # SAME resampled questions for both systems
    diff = a[idx].mean(1) - b[idx].mean(1)
    return diff.mean(), np.quantile(diff, [0.025, 0.975]), (diff <= 0).mean()
d, ci, p = paired_bootstrap(res["acc_norm_bytes"], res["acc"])
print(f"acc_norm_bytes - acc = {d:+.3f}, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], P(diff<=0) = {p:.2f}")

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

# TRY IT: MODEL=distilgpt2 python main.py ; change the extraction rule to take only the first word.
# BREAK IT: change `np.arange(n - c + 1, n + 1)` to `np.arange(n - c, n + 1)` (an off-by-one) and compare the formula with the Monte-Carlo line.
