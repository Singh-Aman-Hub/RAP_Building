"""
evalkit.py: the finished evaluation helpers from step 08 (copied, same logic), reused by step 11 and the solutions.
  normalise / exact_match / token_f1     SQuAD-style string metrics
  cont_logprob, mcq_accuracy             log-likelihood multiple choice (acc, acc_norm_bytes)
  perplexity                             exp(mean token cross-entropy)
  bootstrap_ci, paired_bootstrap         uncertainty
  pass_at_k                              unbiased estimator
GENERAL_MCQ: 30 hand-written common-knowledge questions (also used in step 08) as a "did we break general ability?" probe.
"""
import math
import re
import string
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

GENERAL_MCQ = [
    ("The capital of France is", ["Paris", "Rome", "Madrid", "Berlin"], 0), ("The capital of Italy is", ["Rome", "Paris", "Vienna", "Athens"], 0),
    ("The capital of Japan is", ["Seoul", "Tokyo", "Beijing", "Bangkok"], 1), ("The capital of Egypt is", ["Cairo", "Lima", "Oslo", "Delhi"], 0),
    ("Water freezes at", ["0 degrees Celsius", "50 degrees Celsius", "100 degrees Celsius", "200 degrees Celsius"], 0),
    ("The largest planet in our solar system is", ["Mars", "Venus", "Jupiter", "Mercury"], 2), ("A dog is a kind of", ["fish", "mammal", "insect", "plant"], 1),
    ("The sun rises in the", ["west", "north", "south", "east"], 3), ("Two plus two equals", ["three", "four", "five", "six"], 1),
    ("The chemical symbol for water is", ["H2O", "CO2", "NaCl", "O2"], 0), ("A triangle has how many sides?", ["two", "three", "four", "five"], 1),
    ("The opposite of hot is", ["cold", "warm", "red", "loud"], 0), ("Birds can usually", ["swim underground", "fly", "breathe fire", "read"], 1),
    ("The first month of the year is", ["March", "June", "January", "December"], 2), ("Humans breathe in", ["oxygen", "helium", "gold", "sand"], 0),
    ("The moon orbits the", ["Earth", "Sun", "Jupiter", "Pluto"], 0), ("A week has how many days?", ["five", "six", "seven", "ten"], 2),
    ("Bees make", ["honey", "milk", "silk", "oil"], 0), ("The color of grass is usually", ["blue", "green", "purple", "black"], 1),
    ("Ice is the solid form of", ["water", "iron", "salt", "wood"], 0), ("The capital of Germany is", ["Berlin", "Warsaw", "Prague", "Moscow"], 0),
    ("The capital of Spain is", ["Lisbon", "Madrid", "Dublin", "Athens"], 1), ("A year has how many months?", ["ten", "eleven", "twelve", "thirteen"], 2),
    ("Fish live in", ["water", "trees", "clouds", "deserts"], 0), ("The author of Hamlet is", ["Dickens", "Austen", "Shakespeare", "Orwell"], 2),
    ("Plants need sunlight to make", ["food", "metal", "plastic", "noise"], 0), ("The capital of Canada is", ["Ottawa", "Sydney", "Boston", "Lima"], 0),
    ("Spiders have how many legs?", ["four", "six", "eight", "ten"], 2), ("The currency of the USA is the", ["euro", "dollar", "yen", "peso"], 1),
    ("The Pacific is an", ["ocean", "desert", "mountain", "island"], 0),
]


def normalise(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def exact_match(pred, gold):
    return float(normalise(pred) == normalise(gold))


def token_f1(pred, gold):
    p, g = normalise(pred).split(), normalise(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def cont_logprob(model, tok, context, continuation):
    c = tok(context, return_tensors="pt").input_ids
    k = tok(continuation, return_tensors="pt").input_ids
    ids = torch.cat([c, k], 1)
    logp = F.log_softmax(model(ids).logits[:, :-1].float(), -1)
    tok_lp = logp.gather(-1, ids[:, 1:, None]).squeeze(-1)
    return tok_lp[0, c.shape[1] - 1:].sum().item(), k.shape[1]


def mcq_accuracy(model, tok, items, norm_bytes=True):
    """items: (question, choices, gold_index). Returns a 0/1 numpy array (one entry per item)."""
    model.eval()
    out = []
    for q, choices, gold in items:
        scored = [cont_logprob(model, tok, f"Question: {q}\nAnswer:", " " + c) for c in choices]
        s = [lp / (len(c.encode()) if norm_bytes else 1) for (lp, _), c in zip(scored, choices)]
        out.append(int(np.argmax(s) == gold))
    return np.array(out)


@torch.no_grad()
def perplexity(model, tok, texts):
    model.eval()
    nll, n = 0.0, 0
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids
        nll += model(ids, labels=ids).loss.item() * (ids.shape[1] - 1)
        n += ids.shape[1] - 1
    return math.exp(nll / n)


def bootstrap_ci(values, n_boot=4000, alpha=0.05, seed=0):
    r = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = r.choice(values, size=(n_boot, len(values)), replace=True).mean(1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def paired_bootstrap(after, before, n_boot=4000, seed=0):
    """Mean difference after-before on the SAME items, with a 95% CI."""
    r = np.random.default_rng(seed)
    after, before = np.asarray(after, float), np.asarray(before, float)
    idx = r.integers(0, len(after), size=(n_boot, len(after)))
    d = after[idx].mean(1) - before[idx].mean(1)
    return d.mean(), tuple(np.quantile(d, [0.025, 0.975]))


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))
