"""
GOAL:  Implement GRPO (Group Relative Policy Optimisation) from scratch on a toy VERIFIABLE-reward task and watch the reward rise.
       Task: the prompt is "a + b =" (digits 0-4); the policy (a tiny GRU) writes digit tokens then EOS; reward 1.0 only if the
       written number equals a+b (checked by code, no reward model). Contrast with PPO's machinery (value network, GAE) on tiny numbers.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/06_ppo_vs_grpo_rlvr/main.py
EXPECTED OUTPUT (shortened):
    GAE demo on 4 steps (what PPO needs a value net for)
    step   0  reward 0.0x ... step 200  reward ~0.9+, clip fraction, KL to reference
    final greedy accuracy on train prompts and on 5 held-out prompts
NEEDS: torch only (no downloads).  ~30-60 s on CPU.
"""
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

# ## 1. Token vocabulary: digits 0-9, EOS, '+', '='
EOS, PLUS, EQ, V = 10, 11, 12, 13
T = 3                                                # the policy may write at most 3 tokens

# ## 2. Prompts: pairs (a, b) with a,b in 0..4; hold out 5 pairs to test generalisation
pairs = [(a, b) for a in range(5) for b in range(5)]
rng.shuffle(pairs)
held_out, train_pairs = pairs[:5], pairs[5:]
print("train prompts:", len(train_pairs), " held-out prompts:", held_out)

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

# ## 10. After training
print(f"\nafter training: greedy accuracy train {accuracy(train_pairs):.2f} held-out {accuracy(held_out):.2f}")
for a, b in held_out[:3]:
    c = sample([(a, b)], greedy=True)[0].tolist()
    print(f"  {a}+{b}= -> tokens {c}  reward {reward_fn(a, b, c)}")

# TRY IT: SCALE_STD=0 python main.py (no std division) ; G=4 python main.py (small groups -> many zero-signal groups)
# BREAK IT: BETA=0 EPS=100 LR=3e-2 python main.py  (no KL leash, no clipping, huge LR) and compare KL and stability.
