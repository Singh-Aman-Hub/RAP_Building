"""
GOAL:  Speculative decoding with a small DRAFT model (distilgpt2) and a big TARGET model (gpt2, same tokenizer).
       Implement the greedy version (output provably identical to the target alone), measure acceptance rate and
       the number of target passes saved, and verify the general rejection-sampling rule on a toy distribution.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/13_speculative_decoding/main.py
EXPECTED OUTPUT (shortened):
    identical to target-only greedy: True
    gamma  acceptance  tokens/target pass  ...
    empirical distribution of speculative sampling == target: ...
NEEDS: torch, numpy, transformers; cached gpt2 + distilgpt2.  ~1 min on CPU.
"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
torch.set_num_threads(2)
tok = AutoTokenizer.from_pretrained("gpt2")                                       # distilgpt2 shares this tokenizer
target = AutoModelForCausalLM.from_pretrained("gpt2").eval()
draft = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
n_t, n_d = (sum(p.numel() for p in m.parameters()) / 1e6 for m in (target, draft))
print(f"target gpt2 {n_t:.0f}M params | draft distilgpt2 {n_d:.0f}M params | same vocab: {target.config.vocab_size == draft.config.vocab_size}")


# ---- 1. Plain greedy decoding with the target (the reference answer) --------------------------------------------------------
@torch.no_grad()
def greedy(model, ids, n_new):
    for _ in range(n_new):
        nxt = model(ids).logits[0, -1].argmax()                                   # (vocab,) -> scalar
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
    return ids


# ---- 2. Speculative decoding, greedy version -----------------------------------------------------------------------------------
@torch.no_grad()
def speculative_greedy(ids, n_new, gamma):
    stats = dict(target_passes=0, proposed=0, accepted=0)
    goal = ids.shape[1] + n_new
    while ids.shape[1] < goal:
        # (a) the cheap draft model proposes `gamma` tokens, one by one
        prop = ids
        for _ in range(gamma):
            nxt = draft(prop).logits[0, -1].argmax()
            prop = torch.cat([prop, nxt.view(1, 1)], dim=1)
        # (b) ONE target pass scores the prefix plus all gamma proposals at once (a normal forward pass is parallel over positions)
        logits = target(prop).logits[0]                                           # (len(prop), vocab)
        stats["target_passes"] += 1
        n0 = ids.shape[1]
        target_choice = logits[n0 - 1:].argmax(-1)                                # (gamma + 1,): target's own greedy token after each prefix
        proposed = prop[0, n0:]                                                   # (gamma,)
        # (c) accept the longest prefix of proposals that equals the target's own choice
        agree = (proposed == target_choice[:gamma]).int()
        k = int(agree.cumprod(0).sum())                                           # number of leading agreements
        stats["proposed"] += gamma
        stats["accepted"] += k
        # (d) append the k accepted tokens + ONE token from the target (the correction, or a bonus if all were accepted)
        ids = torch.cat([ids, proposed[:k].view(1, -1), target_choice[k].view(1, 1)], dim=1)
    return ids[:, :goal], stats


prompts = ["The capital of France is", "In machine learning, a neural network is", "def fibonacci(n):\n    if n <", "Once upon a time, there was a little"]
N_NEW = 32
print(f"\n{'gamma':>5}{'acceptance':>12}{'tokens/target pass':>20}{'target passes':>15}{'vs plain':>10}  identical")
results = {}
all_same = True
refs = {p: greedy(target, tok(p, return_tensors="pt").input_ids, N_NEW) for p in prompts}   # reference outputs computed ONCE
for gamma in (2, 4, 8):
    acc = prop = passes = 0
    same = True
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        ref = refs[p]
        out, st = speculative_greedy(ids, N_NEW, gamma)
        same &= torch.equal(out, ref)                                            # the whole point: identical output
        acc, prop, passes = acc + st["accepted"], prop + st["proposed"], passes + st["target_passes"]
    results[gamma] = (acc / prop, passes)
    all_same &= same
    print(f"{gamma:>5}{acc / prop:>12.2f}{len(prompts) * N_NEW / passes:>20.2f}{passes:>15}{passes / (len(prompts) * N_NEW):>10.2f}  {same}")
print("identical to target-only greedy for every gamma and prompt:", all_same)

# ---- 3. Wall-clock honesty check: speedup needs the draft to be MUCH cheaper than the target -------------------------------------
ids = tok(prompts[0], return_tensors="pt").input_ids
t0 = time.time(); greedy(target, ids, N_NEW); t_plain = time.time() - t0
t0 = time.time(); greedy(draft, ids, N_NEW); t_draft = time.time() - t0
c = t_draft / t_plain                                                             # cost of one draft step relative to one target step
t0 = time.time(); speculative_greedy(ids, N_NEW, 4); t_spec = time.time() - t0
print(f"\nmeasured: target-only {t_plain:.2f}s | draft-only {t_draft:.2f}s (c = {c:.2f}) | speculative gamma=4 {t_spec:.2f}s -> speedup {t_plain / t_spec:.2f}x")
print("(no KV cache here, and distilgpt2 is only 2x cheaper than gpt2: the wall-clock gain is small or negative)")


def expected_speedup(alpha, gamma, c):
    """Leviathan et al.: tokens per round (1 - a^(g+1)) / (1 - a), cost per round g*c + 1 target-pass units."""
    return (1 - alpha ** (gamma + 1)) / (1 - alpha) / (gamma * c + 1)


alpha = results[4][0]
print(f"formula with measured acceptance {alpha:.2f}: c=0.5 -> {expected_speedup(alpha, 4, 0.5):.2f}x | "
      f"c=0.05 (e.g. 70B target, 3B draft... ) -> {expected_speedup(alpha, 4, 0.05):.2f}x | c=0.01 -> {expected_speedup(alpha, 4, 0.01):.2f}x")

# ---- 4. The general (sampling) rule: accept draft token x with prob min(1, p(x)/q(x)); on rejection resample from norm(max(0, p - q)) ------
rng = np.random.default_rng(0)
p = np.array([0.5, 0.3, 0.15, 0.05])                                              # target distribution over 4 tokens
q = np.array([0.25, 0.25, 0.25, 0.25])                                            # draft distribution (quite wrong)
samples = np.zeros(4)
accepted = 0
TRIALS = 50_000
for _ in range(TRIALS):
    x = rng.choice(4, p=q)                                                        # draft samples a token
    if rng.random() < min(1.0, p[x] / q[x]):                                      # accept with prob min(1, p/q)
        samples[x] += 1
        accepted += 1
    else:
        resid = np.maximum(p - q, 0)                                              # residual distribution
        samples[rng.choice(4, p=resid / resid.sum())] += 1                        # resample from it
print("\ntarget p                :", p)
print("empirical speculative    :", (samples / TRIALS).round(3), f"(acceptance {accepted / TRIALS:.3f}; theory = sum min(p,q) = {np.minimum(p, q).sum():.3f})")

# TRY IT: change prompts to something unpredictable (random words) and watch acceptance fall; change gamma to 16.
# BREAK IT: replace `target_choice[k]` in the append with `proposed[k]` (index error when k == gamma) or remove cumprod:
#           the output stops matching target-only greedy ("identical" turns False), i.e. you accepted tokens after a mismatch.
