"""
sampling.py -- the decoding tricks (temperature, top-k, top-p, repetition penalty) written
from scratch on a 1-D tensor of logits.  Imported by main.py here and by step 03 / 07.

Pipeline used by every chat model, in this order:
    logits -> repetition penalty -> / temperature -> top-k filter -> top-p filter -> softmax -> sample
Filters work by setting removed tokens' logits to -inf, so softmax gives them probability exactly 0.
"""

import torch


def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """Make tokens that were ALREADY generated less likely (penalty > 1).

    Same rule as Hugging Face: a positive logit is divided by penalty, a negative one multiplied
    (both make the token less attractive).  penalty == 1.0 changes nothing.
    """
    logits = logits.clone()                                  # never mutate the caller's tensor
    for token_id in set(generated_ids):                      # set(): penalise each distinct token once
        score = logits[token_id]
        logits[token_id] = score / penalty if score > 0 else score * penalty
    return logits


def apply_temperature(logits, temperature=1.0):
    """Divide logits by T.  T<1 widens the gaps (sharper); T>1 shrinks them (flatter)."""
    return logits / temperature


def top_k_filter(logits, k):
    """Keep only the k highest logits; everything else becomes -inf.  k == 0 means 'disabled'."""
    if k <= 0 or k >= logits.numel():
        return logits
    kth_value = torch.topk(logits, k).values[-1]             # the k-th largest logit = the cut-off
    return torch.where(logits < kth_value, torch.tensor(float("-inf")), logits)


def top_p_filter(logits, p):
    """Nucleus sampling: keep the smallest set of top tokens whose probabilities sum to >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)          # running total of probability
    # Remove a token if the probability mass BEFORE it already reached p (so the first token always stays)
    remove_sorted = (cumulative - sorted_probs) >= p
    remove = torch.zeros_like(remove_sorted)
    remove[sorted_idx] = remove_sorted                       # map back to the original token order
    return logits.masked_fill(remove, float("-inf"))


def sample_next_token(logits, generated_ids=(), temperature=1.0, top_k=0, top_p=1.0,
                      repetition_penalty=1.0, greedy=False, generator=None):
    """Full recipe: returns ONE token id (int)."""
    if greedy:                                               # do_sample=False
        return int(torch.argmax(logits))
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    # torch.multinomial draws random indices according to the probabilities
    return int(torch.multinomial(probs, num_samples=1, generator=generator))
