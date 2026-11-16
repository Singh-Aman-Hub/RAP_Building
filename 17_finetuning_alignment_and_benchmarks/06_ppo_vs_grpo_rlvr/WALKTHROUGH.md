# Step 06 — PPO vs GRPO and RLVR

## The story so far
[Step 05](../05_dpo_from_scratch/WALKTHROUGH.md) tuned a policy on a fixed set of preference pairs: offline, no new samples. That cannot teach a skill the data never demonstrated. For maths or code you do not need pairs: you can *check* the answer.

## The pain point
Classic RLHF with PPO needs four models in memory (policy, frozen reference, reward model, value network), a learned reward that can be exploited, and many fragile hyper-parameters. For tasks with a checkable answer (the sum is right, the unit tests pass) most of that machinery is waste.

## The spark
Replace the reward model with code (a **verifiable reward**) and replace the value network with a statistic you get for free: sample a *group* of answers to the same prompt and use the group mean as the baseline. That is **GRPO**, applied to **RLVR** (reinforcement learning with verifiable rewards).

## Goal
Implement GRPO from scratch on a toy verifiable task and watch the reward rise. A tiny GRU policy reads `a + b =` (digits 0-4) and writes digits then EOS; reward is 1.0 only if the number equals `a + b`. You also see GAE (what PPO's value network is for) on four numbers, the clipped ratio, and a KL penalty to a frozen reference.

## The theory (read this before the interview)
**RLHF with PPO (4 models):**
```
policy pi_theta      the model being trained
reference pi_ref     frozen copy; keeps the policy near the SFT model via a KL penalty
reward model r(x,y)  learned scorer (step 04)
value network V(s)   learned estimate of expected return; baseline for advantages (GAE)

per token:   A_t = GAE(r, V)                            advantage
             ratio_t = pi_new(a_t|s_t) / pi_old(a_t|s_t)
             L = - min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)  + beta * KL(pi || pi_ref)
```
**GAE:** `delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`; `A_t = sum_l (gamma*lambda)^l * delta_{t+l}`. Lambda trades bias and variance.

**GRPO (3 parts, no value net):**
```
for each prompt: sample G completions, get rewards r_1..r_G (verifier)
A_i = (r_i - mean(r)) / (std(r) + small)                 same advantage for every token of completion i
loss = - min(ratio*A, clip(ratio,1-eps,1+eps)*A) + beta * KL(pi || pi_ref)
```
If all G rewards are equal the advantage is 0 for the whole group: that prompt teaches nothing this step. Groups need both successes and failures.

**RLVR vs RLHF vs RLAIF:** same policy-gradient machinery; the reward comes from code (verifier), humans via a reward model, or an AI judge. A verifier cannot be "talked into" a high score the way a learned reward model can, but a weak verifier (checks only the last number, ignores units) can still be gamed.

**KL estimators:** the k3 estimator used here is `exp(d) - d - 1` with `d = log pi_ref - log pi` evaluated on sampled tokens. It is non-negative and unbiased for KL(pi || pi_ref) under samples from pi.

**Variants (verified in the TRL v1.17.1 GRPO docs, Nov 2026):** `scale_rewards="group"` (default; divide by group std), `"batch"`, or `False` (no std scaling); `loss_type` options listed in the docs are `"dapo"` (the default), `"dr_grpo"` (divides by a constant, the maximum completion length, instead of the sequence length) and `"sapo"` (soft gating instead of hard clipping); `beta` defaults to 0.0 (no reference model is loaded); `num_generations` defaults to 8. This step normalises by the number of valid tokens in the whole batch (token-level), which is the idea behind the token-level option; check the TRL docs for the precise definitions of each `loss_type`.

**Typical 2026 practice:** SFT or cold-start on reasoning traces, then GRPO-style RL with verifiers (math answers, code tests, format checks), large groups (8-16), small learning rates (about 1e-6) and often `beta` near 0 plus clipping. Reasoning models are produced this way.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| RLHF / RLAIF / RLVR | RL from human / AI / verifiable rewards | no | Only the source of the reward changes |
| PPO vs GRPO | actor-critic vs critic-free | no | GRPO replaces the value net with the group mean |
| Group baseline | group mean, leave-one-out baseline | near | Mean-reward baseline per prompt; variants exclude the sample itself |
| Verifier / reward function | rule-based reward, programmatic reward | yes | Code that returns a score |
| Reward model vs verifier | learned vs programmatic | no | One can be exploited; the other is only as good as its checker |
| KL penalty vs clipping | two different leashes | no | KL keeps the policy near the reference; clipping limits each update |
| Rollout / sampling / generation | on-policy samples | yes | Completions drawn from the current policy |
| Advantage | normalised reward, A | near | Reward relative to a baseline |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/06_ppo_vs_grpo_rlvr
python main.py
# knobs: STEPS, G (group size), BETA (KL), EPS (clip), SCALE_STD (1 or 0), SEED
```

## Code, step by step

Lines 1–13 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports, settings and seeds  (`main.py` lines 14–30)
```python
import copy
import os
import random

import torch
import torch.nn as nn

SEED = int(os.environ.get("SEED", 0))
STEPS = int(os.environ.get("STEPS", 200))
G = int(os.environ.get("G", 16))                    # group size: completions sampled per prompt
BETA = float(os.environ.get("BETA", 0.02))          # KL coefficient
EPS = float(os.environ.get("EPS", 0.2))             # PPO-style clip range
SCALE_STD = int(os.environ.get("SCALE_STD", 1))     # 1: divide advantage by the group std (original GRPO); 0: mean only (like TRL scale_rewards=False)
MU = 2                                              # optimisation epochs per batch of samples (needs >1 for clipping to matter)
LR = 3e-3
torch.manual_seed(SEED)
rng = random.Random(SEED)
```
- **In simple words:** Load tools and settings. The knobs are read from environment variables so you can change one thing per experiment.
- **Exactly:** `G=16` completions per prompt, `BETA=0.02` KL weight, `EPS=0.2` clip range, `SCALE_STD=1`, `MU=2` optimisation passes per batch of samples, `LR=3e-3`. `torch.manual_seed` seeds weights and sampling; a separate `random.Random(SEED)` picks prompts. `copy` and `torch.nn` are imported for the reference model and the GRU.
- **Why it is here:** Reproducibility, and `MU>1` is what makes clipping meaningful: with one pass the ratio is exactly 1.

### 1. Token vocabulary: digits 0-9, EOS, '+', '='  (`main.py` lines 32–34)
```python
# ## 1. Token vocabulary: digits 0-9, EOS, '+', '='
EOS, PLUS, EQ, V = 10, 11, 12, 13
T = 3                                                # the policy may write at most 3 tokens
```
- **In simple words:** Define the little alphabet the policy uses: ten digits, an end-of-answer symbol, plus and equals signs.
- **Exactly:** `EOS, PLUS, EQ, V = 10, 11, 12, 13`: token ids 0-9 are digits, 10 is EOS, 11 is `+`, 12 is `=`, so the vocabulary size `V` is 13. `T = 3` is the maximum answer length.
- **Why it is here:** A tiny vocabulary keeps the demonstration CPU-fast while keeping the real structure: sequences of tokens with a terminating symbol.

### 2. Prompts: pairs (a, b) with a,b in 0..4; hold out 5 pairs to test generalisation  (`main.py` lines 36–40)
```python
# ## 2. Prompts: pairs (a, b) with a,b in 0..4; hold out 5 pairs to test generalisation
pairs = [(a, b) for a in range(5) for b in range(5)]
rng.shuffle(pairs)
held_out, train_pairs = pairs[:5], pairs[5:]
print("train prompts:", len(train_pairs), " held-out prompts:", held_out)
```
- **In simple words:** Build the prompts and hold five of them back to see whether the policy learned a rule or just memorised.
- **Exactly:** `pairs` is all 25 `(a, b)` with `a, b` in 0..4, shuffled with the seeded `rng`; `held_out = pairs[:5]`, `train_pairs = pairs[5:]` (20 prompts). Output shows the five held-out pairs: `(2,4), (0,0), (4,3), (3,3), (1,1)`.
- **Why it is here:** Reward on the practice prompts alone can hide pure memorisation.

### 3. The verifier: pure code, returns 1.0 or 0.0  (`main.py` lines 42–51)
```python
# ## 3. The verifier: pure code, returns 1.0 or 0.0
def reward_fn(a, b, toks):
    """toks: list of ints the policy produced. Correct = digits then EOS spelling a+b with no leading zeros."""
    if EOS not in toks:
        return 0.0
    body = toks[:toks.index(EOS)]                    # everything before the first EOS
    if not body or any(t > 9 for t in body):         # empty answer or '+'/'=' inside it
        return 0.0
    s = "".join(map(str, body))
    return 1.0 if s == str(a + b) else 0.0
```
- **In simple words:** The verifier is a plain function that checks the answer. No learned model.
- **Exactly:** `reward_fn(a, b, toks)` returns 1.0 only if an EOS exists, the tokens before the first EOS are non-empty digits (no `+` or `=`), and their string equals `str(a + b)`. Everything else is 0.0, including a right number with no EOS.
- **Why it is here:** This is the "V" in RLVR. A weak verifier would be gamed, so it checks termination and format too.

### 4. A tiny autoregressive policy: embedding -> GRU -> logits over V tokens  (`main.py` lines 53–72)
```python
# ## 4. A tiny autoregressive policy: embedding -> GRU -> logits over V tokens
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V + 1, 32)           # +1 for a BOS index
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.out = nn.Linear(64, V)

    def forward(self, ids):                          # ids: (B, L) -> logits (B, L, V)
        h, _ = self.gru(self.emb(ids))
        return self.out(h)

BOS = V
policy = Policy()
reference = copy.deepcopy(policy).eval()             # frozen copy for the KL term
for p in reference.parameters():
    p.requires_grad_(False)

def prompt_ids(a, b):
    return [BOS, a, PLUS, b, EQ]                     # the model reads this, then writes the answer
```
- **In simple words:** The policy is a very small language model: embed tokens, run a GRU, predict the next token.
- **Exactly:** `Policy.forward(ids)` takes `(B, L)` ids and returns `(B, L, 13)` logits. The embedding table has `V+1 = 14` rows (13 tokens + BOS). `nn.GRU(32, 64, batch_first=True)` returns `(B, L, 64)`. `reference = copy.deepcopy(policy).eval()` is the frozen copy; `p.requires_grad_(False)` makes sure it never trains. `prompt_ids(a, b)` returns `[BOS, a, +, b, =]` (5 tokens).
- **Why it is here:** The reference gives the KL term something to be measured against; freezing it is essential.

### 5. Sample G completions for each prompt  (`main.py` lines 74–84)
```python
# ## 5. Sample G completions for each prompt
@torch.no_grad()
def sample(prompts, greedy=False):
    ids = torch.tensor([prompt_ids(a, b) for a, b in prompts])          # (N, 5)
    out = []
    for _ in range(T):
        logits = policy(ids)[:, -1]                                      # (N, V) next-token scores
        nxt = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()   # (N,)
        out.append(nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
    return torch.stack(out, dim=1)                                       # (N, T)
```
- **In simple words:** Sampling: ask the policy for the next token three times in a row to write an answer.
- **Exactly:** `sample(prompts)` builds `ids` of shape `(N, 5)`, then loops `T=3` times: take the logits of the last position `(N, 13)`, draw with `Categorical(logits=...).sample()` (or `argmax` when `greedy`), append to `ids`. Returns `(N, 3)`. Decorated with `@torch.no_grad()` because sampling is not part of the gradient.
- **Why it is here:** GRPO is on-policy: the samples must come from the current policy so the ratio math is valid.

### 6. Per-token log-probs of given completions under a model  (`main.py` lines 86–96)
```python
# ## 6. Per-token log-probs of given completions under a model
def token_logps(model, prompts, comps):
    ids = torch.cat([torch.tensor([prompt_ids(a, b) for a, b in prompts]), comps], dim=1)   # (N, 5+T)
    logits = model(ids)[:, 4:-1]                                         # positions that predict the T answer tokens: (N, T, V)
    return torch.log_softmax(logits, -1).gather(-1, comps.unsqueeze(-1)).squeeze(-1)       # (N, T)

def valid_mask(comps):
    """1 for tokens up to and including the first EOS (later tokens are never 'written')."""
    is_eos = (comps == EOS).long()
    before = torch.cumsum(is_eos, dim=1) - is_eos                        # number of EOS strictly before this position
    return (before == 0).float()                                          # (N, T)
```
- **In simple words:** Score the sampled tokens: how likely did the (current or reference) model find each of them?
- **Exactly:** `token_logps(model, prompts, comps)` concatenates prompt and completion to `(N, 5+T)`, runs the model, and keeps logits at positions `4:-1` (the positions that predict the three answer tokens), shape `(N, 3, 13)`. `log_softmax` then `gather` picks the log-prob of the sampled token: `(N, 3)`. `valid_mask` is 1 up to and including the first EOS and 0 after it, via `cumsum(is_eos) - is_eos`.
- **Why it is here:** Tokens after EOS are never "written", so they must not receive gradient.

### 7. GAE for contrast: what PPO computes with a learned value network  (`main.py` lines 98–108)
```python
# ## 7. GAE for contrast: what PPO computes with a learned value network
def gae(rewards, values, gamma=1.0, lam=0.95):
    adv, last = [], 0.0
    for t in reversed(range(len(rewards))):
        nxt_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nxt_v - values[t]                   # TD error
        last = delta + gamma * lam * last
        adv.insert(0, last)
    return adv
print("GAE demo: rewards [0,0,0,1], values [0.2,0.3,0.5,0.7] ->", [round(x, 3) for x in gae([0, 0, 0, 1.0], [0.2, 0.3, 0.5, 0.7])],
      "(PPO needs a value net V(s) for this; GRPO replaces it with the group mean)")
```
- **In simple words:** A small side demo of GAE, the thing PPO needs a value network for, using four invented numbers.
- **Exactly:** `gae(rewards, values, gamma, lam)` walks backwards: `delta = r + gamma*V_next - V`, `last = delta + gamma*lam*last`. With rewards `[0,0,0,1]` and values `[0.2,0.3,0.5,0.7]` it prints `[0.728, 0.661, 0.485, 0.3]`.
- **Why it is here:** Seeing GAE next to GRPO makes the difference concrete: PPO needs `values` from a trained network; GRPO needs only the rewards of the group.

### 8. The GRPO update on one batch of prompts  (`main.py` lines 110–143)
```python
# ## 8. The GRPO update on one batch of prompts
opt = torch.optim.Adam(policy.parameters(), lr=LR)
def grpo_step(prompts):
    rep = [p for p in prompts for _ in range(G)]                         # each prompt repeated G times -> N = len(prompts)*G
    comps = sample(rep)                                                  # (N, T)
    mask = valid_mask(comps)                                             # (N, T)
    rew = torch.tensor([reward_fn(a, b, c.tolist()) for (a, b), c in zip(rep, comps)])   # (N,)
    r = rew.view(len(prompts), G)                                        # (P, G): one row per group
    adv = r - r.mean(1, keepdim=True)                                    # advantage = reward minus the group mean (the baseline)
    if SCALE_STD:
        adv = adv / (r.std(1, keepdim=True) + 1e-4)                      # ...divided by the group std
    adv = adv.view(-1, 1)                                                # (N, 1) broadcast over tokens
    with torch.no_grad():
        old_lp = token_logps(policy, rep, comps)                         # behaviour policy log-probs, fixed for this batch
        ref_lp = token_logps(reference, rep, comps)
    clipped, kl_val = 0.0, 0.0
    for _ in range(MU):
        lp = token_logps(policy, rep, comps)                             # (N, T) with grad
        ratio = torch.exp(lp - old_lp)                                   # pi_new / pi_old per token
        unclipped = ratio * adv
        clip = torch.clamp(ratio, 1 - EPS, 1 + EPS) * adv
        pg = -torch.min(unclipped, clip)                                 # PPO clipped objective (negated: we minimise)
        d = ref_lp - lp                                                  # k3 KL estimator: exp(d) - d - 1  >= 0
        kl = torch.exp(d) - d - 1
        per_tok = (pg + BETA * kl) * mask
        loss = per_tok.sum() / mask.sum()                                # token-level average over all valid tokens in the batch
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        clipped = ((ratio - 1).abs() > EPS).float().mul(mask).sum().item() / mask.sum().item()
        kl_val = (kl * mask).sum().item() / mask.sum().item()
    nonzero = (r.std(1) > 0).float().mean().item()                       # groups with any signal at all
    return rew.mean().item(), kl_val, clipped, nonzero
```
- **In simple words:** One GRPO update: sample groups, turn rewards into advantages, then take a few clipped policy-gradient steps with a KL leash.
- **Exactly:** `rep` repeats each prompt `G` times (`N = 8*16 = 128`). `comps (N,3)`, `mask (N,3)`, `rew (N,)`, `r = rew.view(8,16)`. Advantage `adv = r - r.mean(1, keepdim=True)`, divided by `r.std(1, keepdim=True)+1e-4` if `SCALE_STD`, then `.view(-1,1)` to `(N,1)`. `old_lp` and `ref_lp` are computed without gradient. Inside the loop (`MU=2` times): `lp` with gradient, `ratio = exp(lp - old_lp)` `(N,3)`, clipped objective `-min(ratio*adv, clamp(ratio)*adv)`, k3 KL `exp(d) - d - 1` with `d = ref_lp - lp`, loss `((pg + BETA*kl) * mask).sum() / mask.sum()` (token-level average), gradient clipping at norm 1.0. It returns mean reward, KL, clip fraction and the fraction of groups whose rewards are not all equal.
- **Why it is here:** This is the algorithm. Take any line out: no baseline (high variance), no clip (unstable), no KL (drifts), no mask (trains on padding).

### 9. Greedy evaluation  (`main.py` lines 145–156)
```python
# ## 9. Greedy evaluation
def accuracy(prompts):
    comps = sample(prompts, greedy=True)
    return sum(reward_fn(a, b, c.tolist()) for (a, b), c in zip(prompts, comps)) / len(prompts)

print(f"\nbefore training: greedy accuracy train {accuracy(train_pairs):.2f} held-out {accuracy(held_out):.2f}")
print(f"{'step':>5} {'mean reward':>12} {'KL to ref':>10} {'clip frac':>10} {'groups w/ signal':>17}")
for step in range(STEPS + 1):
    batch = rng.sample(train_pairs, 8)                                   # 8 prompts x G completions per step
    rw, kl, cf, nz = grpo_step(batch)
    if step % 20 == 0:
        print(f"{step:>5} {rw:>12.3f} {kl:>10.4f} {cf:>10.3f} {nz:>17.2f}")
```
- **In simple words:** A helper that measures accuracy by decoding greedily, then the training loop that calls the GRPO step 200 times.
- **Exactly:** `accuracy(prompts)` calls `sample(prompts, greedy=True)` (one argmax sample per prompt) and averages `reward_fn`. The loop runs 201 steps; each picks 8 train prompts with `rng.sample`, calls `grpo_step`, and every 20 steps prints reward, KL, clip fraction and groups-with-signal.
- **Why it is here:** Training reward comes from random samples; greedy accuracy is what you would deploy, so both are reported.

### 10. After training  (`main.py` lines 158–165)
```python
# ## 10. After training
print(f"\nafter training: greedy accuracy train {accuracy(train_pairs):.2f} held-out {accuracy(held_out):.2f}")
for a, b in held_out[:3]:
    c = sample([(a, b)], greedy=True)[0].tolist()
    print(f"  {a}+{b}= -> tokens {c}  reward {reward_fn(a, b, c)}")

# TRY IT: SCALE_STD=0 python main.py (no std division) ; G=4 python main.py (small groups -> many zero-signal groups)
# BREAK IT: BETA=0 EPS=100 LR=3e-2 python main.py  (no KL leash, no clipping, huge LR) and compare KL and stability.
```
- **In simple words:** After training we measure again, on practice prompts and on held-out ones, and look at three decoded answers.
- **Exactly:** Prints greedy accuracy for `train_pairs` and `held_out` (0.90 and 0.00 in the default run), then decodes three held-out prompts, e.g. `2+4= -> tokens [4, 10, 4]` (a digit, EOS, then a junk token after EOS that the verifier ignores). The last two lines of the file are the TRY IT and BREAK IT comments.
- **Why it is here:** The held-out line is what separates "learned the rule" from "memorised the prompts"; the groups-with-signal column explains why learning slowed.


## Syntax explained
| Syntax | Meaning |
|---|---|
| `torch.distributions.Categorical(logits=l).sample()` | draw one token per row from softmax(l) |
| `torch.min(a, b)` | element-wise minimum: PPO's pessimistic bound |
| `torch.clamp(x, lo, hi)` | limit values to `[lo, hi]` |
| `.view(len(prompts), G)` | reshape a flat list of rewards to one row per prompt |
| `r.mean(1, keepdim=True)` | per-group mean, kept as `(P, 1)` for broadcasting |
| `torch.cumsum(is_eos, 1) - is_eos` | number of EOS tokens strictly before each position |
| `@torch.no_grad()` | skip the autograd graph (sampling does not need gradients) |

## Libraries & functions
| Library / function | What it does here |
|---|---|
| `torch.nn.GRU(32, 64, batch_first=True)` | the sequence model; input `(B, L, 32)`, output `(B, L, 64)` |
| `nn.Embedding(V + 1, 32)` | token id to vector; the extra row is the BOS token |
| `torch.log_softmax(...).gather(-1, comps.unsqueeze(-1))` | log-probability of the token that was actually sampled |
| `torch.nn.utils.clip_grad_norm_` | rescales gradients whose norm exceeds 1.0 |
| `copy.deepcopy(policy)` | the frozen reference |

## Execution trace (real run)
```
train prompts: 20  held-out prompts: [(2, 4), (0, 0), (4, 3), (3, 3), (1, 1)]
GAE demo: rewards [0,0,0,1], values [0.2,0.3,0.5,0.7] -> [0.728, 0.661, 0.485, 0.3] (PPO needs a value net V(s) for this; GRPO replaces it with the group mean)

before training: greedy accuracy train 0.00 held-out 0.00
 step  mean reward  KL to ref  clip frac  groups w/ signal
    0        0.008     0.0026      0.048              0.12
   20        0.172     1.5138      0.035              0.62
   40        0.289     1.4016      0.042              0.88
   60        0.625     2.4482      0.019              0.75
   80        0.969     1.7726      0.004              0.38
  100        0.875     2.1853      0.008              0.00
  120        0.750     1.5492      0.000              0.00
  140        0.844     2.3642      0.000              0.38
  160        0.859     2.9709      0.000              0.25
  180        0.969     6.8355      0.008              0.38
  200        0.969     2.1249      0.000              0.38

after training: greedy accuracy train 0.90 held-out 0.00
  2+4= -> tokens [4, 10, 4]  reward 0.0
  0+0= -> tokens [1, 10, 1]  reward 0.0
  4+3= -> tokens [5, 10, 5]  reward 0.0
```
- `train prompts: 20  held-out prompts: [...]`: 25 pairs of digits shuffled with the seed; 20 train, 5 held out.
- `GAE demo`: with rewards `[0,0,0,1]` and value guesses `[0.2,0.3,0.5,0.7]` the advantages are `[0.728, 0.661, 0.485, 0.3]`: the single reward at the end is spread backward over earlier steps. PPO needs a trained value net to produce those guesses.
- `before training: greedy accuracy 0.00`: a random GRU writes junk.
- Step 0: mean reward 0.008 (about 1 sample in 128 is right by luck); only 12% of groups contain any success, so most prompts have zero advantage.
- Steps 20-80: reward 0.172, 0.289, 0.625, 0.969. "groups w/ signal" first rises (0.62, 0.88) as successes appear, then falls (0.38 at step 80, 0.00 at step 100) as groups become all-correct: advantage 0 means nothing left to learn there. This is the known "vanishing signal" of GRPO on easy prompts.
- KL to the reference grows from 0.003 to roughly 1.4-7 (it spikes to 6.84 at step 180): with beta 0.02 the leash is weak, the policy moves far, and it matters little on a toy vocabulary.
- `clip frac` stays below 5%: two inner epochs over one batch rarely push the ratio outside [0.8, 1.2] at this learning rate.
- After training: greedy accuracy 0.90 on the 20 training prompts and 0.00 on the 5 held-out prompts. The tiny GRU memorised a lookup for the prompts it practised; the samples show it writes a plausible shape (`digit, EOS, digit`) but the wrong digit. RL optimises exactly the reward on exactly the prompts you give it.

## Experiments (change ONE thing)
| Change | Real result | Lesson |
|---|---|---|
| default (`SCALE_STD=1`, `G=16`) | reward 0.969 at step 200; greedy train 0.90, held-out 0.00 | the baseline run |
| `SCALE_STD=0` (mean only, no std division) | reward 0.805 at step 200 (0.938 at step 180); greedy train 0.80, held-out 0.00; slower early (0.078 at step 20 vs 0.172) | dividing by the std amplifies the few-success groups; without it learning starts more slowly |
| `G=4` | reward only 0.469 at step 200; groups with signal mostly 0.00-0.38; greedy train 0.80 | small groups often contain no success, so there is less learning signal per sample |
| `BETA=0 EPS=100 LR=3e-2` (no KL, no effective clipping, big LR) | reward 0.867 at step 200 but KL 8.63 and clip frac 0.000 everywhere (clipping disabled); greedy train 0.85, held-out 0.20 (1 of 5) | without the leashes the policy wanders far; the lucky 1/5 held-out result is one prompt, not evidence of generalisation |

## Debugging practice
**BREAK IT:** drop the `.view(-1, 1)` on the advantages so they stay shape `(N,)` while token ratios are `(N, T)`. Real traceback (the file name is the temporary broken copy of `main.py`):
```
GAE demo: rewards [0,0,0,1], values [0.2,0.3,0.5,0.7] -> [0.728, 0.661, 0.485, 0.3] (PPO needs a value net V(s) for this; GRPO replaces it with the group mean)

before training: greedy accuracy train 0.00 held-out 0.00
 step  mean reward  KL to ref  clip frac  groups w/ signal
Traceback (most recent call last):
  File "/Users/aman/Desktop/Project_resume/RAP_interview/17_finetuning_alignment_and_benchmarks/06_ppo_vs_grpo_rlvr/_brk.py", line 155, in <module>
    rw, kl, cf, nz = grpo_step(batch)
                     ^^^^^^^^^^^^^^^^
  File "/Users/aman/Desktop/Project_resume/RAP_interview/17_finetuning_alignment_and_benchmarks/06_ppo_vs_grpo_rlvr/_brk.py", line 130, in grpo_step
    unclipped = ratio * adv
                ~~~~~~^~~~~
RuntimeError: The size of tensor a (3) must match the size of tensor b (128) at non-singleton dimension 1
```
How to read it: the last line names the failing operation, `ratio * adv`. PyTorch broadcasts from the right: `(128, 3)` against `(128,)` tries to match 3 with 128. The sizes in the message are the diagnosis: `N=128` (8 prompts x 16 samples) and `T=3` tokens. The fix is the `(N, 1)` shape: one advantage per sequence, shared by all its tokens.

## Rebuild checklist
- [ ] Verifier function returning 0.0 / 1.0
- [ ] Sampler that returns `(N, T)` token ids and a mask up to the first EOS
- [ ] Per-token log-probs with the right shift (prediction positions `4:-1`)
- [ ] Group advantages `(r - mean) / (std + 1e-4)` shaped `(N, 1)`
- [ ] Clipped objective with `torch.min`, k3 KL, token-level average over valid tokens
- [ ] A frozen reference and a "groups with signal" metric

## The new problem (cliffhanger)
RL made a policy better at one skill; the next question is how to make a *small* model inherit a big one's behaviour, and how to combine several fine-tunes without retraining. On to [step 07](../07_distillation_and_merging/WALKTHROUGH.md).

> **Say it aloud (interview box):** "PPO needs a policy, a reference, a reward model and a value network. GRPO drops the value network: sample a group per prompt, use reward minus the group mean (divided by the std) as the advantage, keep the clipped ratio and the KL to a reference. With a verifiable reward that is RLVR, and groups where every sample gets the same reward teach nothing."

