"""
GOAL:  Implement the DPO loss in torch with a frozen reference model on distilgpt2, train on 60 preference pairs, and log what
       practitioners watch: implicit rewards (chosen / rejected), margin, accuracy, and the log-prob of the chosen reply.
       Compare with two variants (IPO, SimPO-style) in the same code and print a variants table.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch/main.py
EXPECTED OUTPUT (shortened):
    step 0: loss 0.693 (= ln 2) because policy == reference, margin 0, accuracy ~0.5
    ... loss falls, reward margin rises, accuracy -> ~1.0 on held-out pairs; sampled reply becomes polite
    variants table (DPO, IPO, SimPO, KTO, ORPO)
NEEDS: torch, transformers; cached distilgpt2.  ~1-2 minutes on CPU.
"""
import copy
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_preference_data, seq_logprob   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
BETA = float(os.environ.get("BETA", 0.1))
LR = float(os.environ.get("LR", 5e-6))
EPOCHS = int(os.environ.get("EPOCHS", 3))
LOSS = os.environ.get("LOSS", "dpo")           # dpo | ipo | simpo
torch.manual_seed(SEED)

# ## 1. Policy (trainable) and reference (frozen copy of the SAME starting weights)
tok = AutoTokenizer.from_pretrained("distilgpt2")
policy = AutoModelForCausalLM.from_pretrained("distilgpt2")
reference = copy.deepcopy(policy).eval()
for p in reference.parameters():
    p.requires_grad_(False)                    # the reference never changes: it anchors the KL penalty
train, test = make_preference_data()

# ## 2. Log-probs of both replies under both models
def pair_logps(pair, grad):
    pi_c = seq_logprob(policy, tok, pair["prompt"], pair["chosen"], grad=grad)       # log pi(chosen | prompt), scalar
    pi_r = seq_logprob(policy, tok, pair["prompt"], pair["rejected"], grad=grad)
    with torch.no_grad():                                                            # reference: no graph needed
        ref_c = seq_logprob(reference, tok, pair["prompt"], pair["chosen"])
        ref_r = seq_logprob(reference, tok, pair["prompt"], pair["rejected"])
    return pi_c, pi_r, ref_c, ref_r

# ## 3. The loss family
def preference_loss(pi_c, pi_r, ref_c, ref_r, n_c, n_r, kind=LOSS, beta=BETA):
    logits = (pi_c - ref_c) - (pi_r - ref_r)        # difference of log-ratios (the "implicit reward margin" / beta)
    if kind == "dpo":
        return -F.logsigmoid(beta * logits)         # Bradley-Terry on implicit rewards r = beta * log(pi/ref)
    if kind == "ipo":
        return (logits - 1.0 / (2.0 * beta)) ** 2   # squared loss: regress the margin to 1/(2 beta) instead of pushing it to infinity
    if kind == "simpo":                             # no reference at all; length-normalised log-prob; fixed target margin gamma
        gamma = 1.0
        return -F.logsigmoid(2.0 * (pi_c / n_c - pi_r / n_r) - gamma)   # beta=2.0 in SimPO's usual range
    raise ValueError(kind)

# ## 4. Metrics on a list of pairs (no grad)
@torch.no_grad()
def metrics(pairs):
    rc, rr, lp = [], [], []
    for p in pairs:
        pi_c, pi_r, ref_c, ref_r = pair_logps(p, grad=False)
        rc.append(BETA * (pi_c - ref_c).item())          # implicit reward of chosen
        rr.append(BETA * (pi_r - ref_r).item())          # implicit reward of rejected
        lp.append(pi_c.item())
    n = len(pairs)
    return {"reward_chosen": sum(rc) / n, "reward_rejected": sum(rr) / n,
            "margin": sum(a - b for a, b in zip(rc, rr)) / n,
            "accuracy": sum(a > b for a, b in zip(rc, rr)) / n, "logp_chosen": sum(lp) / n}

def show(tag, m):
    print(f"{tag:<18}r_chosen {m['reward_chosen']:+.3f}  r_rejected {m['reward_rejected']:+.3f}  margin {m['margin']:+.3f}  "
          f"acc {m['accuracy']:.2f}  logp(chosen) {m['logp_chosen']:.2f}")

# ## 5. Before training
print(f"loss={LOSS} beta={BETA} lr={LR}  ln2={math.log(2):.3f}")
show("test, step 0", metrics(test))
first = train[0]
pi_c, pi_r, ref_c, ref_r = pair_logps(first, grad=False)
print(f"first pair loss at step 0 = {preference_loss(pi_c, pi_r, ref_c, ref_r, 1, 1).item():.3f}  (policy == reference so the margin is 0)")

# ## 6. Train: one pair per step, accumulate 4 pairs per optimiser step
opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.0)
rng = random.Random(SEED)
for ep in range(EPOCHS):
    policy.train()
    order = list(range(len(train)))
    rng.shuffle(order)
    run = []
    for s in range(0, len(order), 4):
        opt.zero_grad()
        batch_loss = 0.0
        for i in order[s:s + 4]:
            p = train[i]
            pi_c, pi_r, ref_c, ref_r = pair_logps(p, grad=True)
            n_c, n_r = len(tok(p["chosen"]).input_ids), len(tok(p["rejected"]).input_ids)
            loss = preference_loss(pi_c, pi_r, ref_c, ref_r, n_c, n_r) / 4
            loss.backward()
            batch_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        run.append(batch_loss)
    policy.eval()
    show(f"test, epoch {ep + 1}", metrics(test))
    print(f"{'':<18}mean train loss {sum(run) / len(run):.3f}")

# ## 7. What did the policy learn? sample greedily before vs after
def reply(model, prompt):
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:])
for q in ("User: Where is my order?\nAssistant:", "User: Can I get a refund?\nAssistant:"):
    print(f"\n{q!r}\n  reference: {reply(reference, q)!r}\n  policy   : {reply(policy, q)!r}")

# ## 8. Variants at a glance
print("""
variant  reference model?  loss in one line
DPO      yes               -log sigmoid(beta * [(log pi_c - log ref_c) - (log pi_r - log ref_r)])
IPO      yes               (margin - 1/(2 beta))^2 : squared loss, bounded target margin, resists over-fitting
KTO      yes               needs only thumbs-up / thumbs-down per example (no pairs); prospect-theory-shaped value function
ORPO     NO                SFT loss on chosen + odds-ratio penalty on rejected; one stage, no reference
SimPO    NO                length-normalised avg log-prob as reward + target margin gamma; no reference""")

# TRY IT: BETA=0.5 python main.py (stays closer to the reference)  |  LOSS=ipo python main.py  |  LOSS=simpo python main.py
# BREAK IT: LR=5e-4 python main.py  -> look at logp(chosen): the policy wrecks BOTH replies (likelihood displacement).
