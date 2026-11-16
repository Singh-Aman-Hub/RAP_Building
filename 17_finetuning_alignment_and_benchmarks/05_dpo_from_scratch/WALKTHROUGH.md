# Step 05 — DPO from scratch

## The story so far
[Step 04](../04_reward_model_bradley_terry/WALKTHROUGH.md) turned preference pairs into a scalar reward with the Bradley-Terry loss. The classic RLHF recipe would now run PPO against that reward model, which means a second trained network, sampling from the policy, and a value network ([step 06](../06_ppo_vs_grpo_rlvr/WALKTHROUGH.md) shows that machinery).

## The pain point
RLHF with a separate reward model is heavy and fragile: four models in memory, many hyper-parameters, and a learned reward the policy can exploit. If all you have is a fixed set of preference pairs, is there a way to optimise the policy on them directly?

## The spark
Direct Preference Optimization (DPO) observes that the optimum of the KL-regularised RLHF objective gives a closed form relation between the policy and the reward: `r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) ) + const(x)`. The policy *is* a reward model; its "implicit reward" is `beta` times the log-ratio against a frozen reference. Plug that into the Bradley-Terry loss of step 04 and the reward model disappears: you train the policy on pairs with a plain supervised-style loss.

## Goal
Implement the DPO loss in torch on `distilgpt2` with a frozen reference copy, train on 60 preference pairs (polite = chosen, rude = rejected), and log what practitioners watch: implicit rewards of chosen and rejected, their margin, accuracy, and the log-probability of the chosen reply. Switch the same code to IPO and SimPO-style losses, sample replies before and after, and read a table of the variants (DPO, IPO, KTO, ORPO, SimPO).

## The theory (read this before the interview)
**KL-regularised RLHF objective:** maximise `E[ r(x, y) ] - beta * KL( pi || pi_ref )`. Its optimal policy is `pi*(y|x) = pi_ref(y|x) * exp( r(x,y) / beta ) / Z(x)`. Rearranging gives the implicit reward above (the partition function `Z(x)` depends only on `x`).

**DPO loss.** Plug the implicit reward into Bradley-Terry (`P(y_w > y_l) = sigmoid(r_w - r_l)`); `Z(x)` cancels in the difference:
```
L_DPO = - log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x))  -  (log pi(y_l|x) - log ref(y_l|x)) ] )
                         \_______ implicit reward margin / beta (called `logits` in the code) _______/
```
- `pi` = the policy being trained; `ref` = a frozen copy of the starting policy (usually the SFT model); `y_w`/`y_l` = chosen/rejected; `log pi(y|x)` = sum of token log-probabilities of the reply given the prompt.
- **At step 0 policy equals reference**, so both log-ratios are 0, the margin is 0 and the loss is `-log sigmoid(0) = ln 2 = 0.693`.
- **`beta`** controls the leash: large `beta` keeps the policy near the reference, small `beta` lets it move far. Typical values are around 0.1.
- **Offline:** DPO never samples from the current policy; it learns from a fixed dataset. Online or on-policy variants (and RL as in step 06) sample new replies as the policy changes.
- **The reference** matters twice: it defines the implicit reward and anchors the KL.

**What to monitor:** implicit rewards of chosen and rejected (`beta * log-ratio`), the margin, accuracy (`reward_chosen > reward_rejected`), and `log pi(chosen)`. A healthy run has the margin growing. A known failure mode is **likelihood displacement**: the margin grows because the policy pushes *both* replies' probabilities down, rejected faster, so `log pi(chosen)` also falls. Watch it.

**Variants (in the code or in the table):**
| Variant | Reference model? | Idea |
|---|---|---|
| DPO | yes | Bradley-Terry on the implicit reward margin |
| IPO | yes | squared loss `(logits - 1/(2 beta))^2`: regress the margin to a finite target instead of pushing it to infinity |
| SimPO | no | reward is the *length-normalised* average log-probability, plus a target margin `gamma` |
| KTO | yes | needs only thumbs up / thumbs down per example (no pairs); loss shaped by prospect theory |
| ORPO | no | SFT loss on chosen plus an odds-ratio penalty on rejected; one stage |
Only DPO, IPO and SimPO are implemented in `main.py`; KTO and ORPO appear in the printed table.

**Practice (stated generally, Nov 2026):** DPO and its relatives are the most common way to apply preference data without an RL stack; reasoning-focused post-training typically adds an RL stage with verifiable rewards ([step 06](../06_ppo_vs_grpo_rlvr/WALKTHROUGH.md)). The TRL reference for DPO is in [step 10](../10_benchmark_landscape_2026_and_tools/WALKTHROUGH.md).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| DPO | direct preference optimization, direct alignment algorithm (the family) | DPO is one member | IPO, KTO, ORPO, SimPO are siblings |
| Reference model | `ref`, `pi_ref`, frozen policy, SFT model | yes | Frozen copy that anchors the KL and defines the implicit reward |
| Implicit reward | `beta * log(pi/ref)` | yes | The reward the policy "secretly" represents; no network predicts it |
| Reward margin | `r_chosen - r_rejected`, `logits * beta` | near | Often logged as `rewards/margins` in TRL |
| Reward accuracy | preference accuracy | yes | Fraction of pairs where chosen has the higher implicit reward |
| `beta` | KL coefficient, temperature of the implicit reward | near | Small `beta`: policy moves far; large: stays close to the reference |
| Chosen / rejected | preferred / dispreferred, `y_w` / `y_l` | yes | Naming only |
| Offline vs online | static pairs vs freshly sampled | no | DPO is offline; online DPO and RL sample from the current policy |
| Likelihood displacement | probability of chosen decreasing | no | Failure mode: both replies become less likely while the margin grows |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch
python main.py
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/05_dpo_from_scratch/main.py
# knobs: SEED, BETA (0.1), LR (5e-6), EPOCHS (3), LOSS = dpo | ipo | simpo
```
About 1–2 minutes of compute on an idle laptop CPU (two copies of distilgpt2 and 180 per-pair forward passes per epoch); varies by machine and load.

## Code, step by step

Lines 1–13 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports and settings  (`main.py` lines 14–33)
```python
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
```
- **In simple words:** Imports and the four knobs: `BETA` (how tight the leash is), `LR` (a very small learning rate), `EPOCHS`, and which loss to use.
- **Exactly:** `copy.deepcopy` will clone the model for the reference. `make_preference_data` and `seq_logprob` come from `common.py`. `LOSS` is a string chosen from the environment; unknown values raise `ValueError` in block 3. The learning rate (5e-6) is small because the policy is *supposed* to stay close to the reference.
- **Why it is here:** `BETA` and `LOSS` are the two things you vary in practice.

### 1. Policy (trainable) and reference (frozen copy of the SAME starting weights)  (`main.py` lines 35–41)
```python
# ## 1. Policy (trainable) and reference (frozen copy of the SAME starting weights)
tok = AutoTokenizer.from_pretrained("distilgpt2")
policy = AutoModelForCausalLM.from_pretrained("distilgpt2")
reference = copy.deepcopy(policy).eval()
for p in reference.parameters():
    p.requires_grad_(False)                    # the reference never changes: it anchors the KL penalty
train, test = make_preference_data()
```
- **In simple words:** Two identical models; one will be trained, the other frozen forever.
- **Exactly:** `copy.deepcopy(policy)` duplicates every tensor, so later updates to `policy` do not change `reference`. `.eval()` disables dropout (a stochastic reference would add noise to the log-ratio). `requires_grad_(False)` stops autograd from tracking it (saves memory and prevents accidental updates). Memory cost: two models (here 2 x 82M parameters; at scale this is why DPO needs a reference copy, or precomputed reference log-probabilities, or a LoRA adapter that can be switched off).
- **Why it is here:** The reference is half of the DPO loss. Everything the policy "learns" is measured relative to it.

### 2. Log-probs of both replies under both models  (`main.py` lines 43–50)
```python
# ## 2. Log-probs of both replies under both models
def pair_logps(pair, grad):
    pi_c = seq_logprob(policy, tok, pair["prompt"], pair["chosen"], grad=grad)       # log pi(chosen | prompt), scalar
    pi_r = seq_logprob(policy, tok, pair["prompt"], pair["rejected"], grad=grad)
    with torch.no_grad():                                                            # reference: no graph needed
        ref_c = seq_logprob(reference, tok, pair["prompt"], pair["chosen"])
        ref_r = seq_logprob(reference, tok, pair["prompt"], pair["rejected"])
    return pi_c, pi_r, ref_c, ref_r
```
- **In simple words:** For one pair compute four numbers: how likely each reply is under the policy and under the reference.
- **Exactly:** `seq_logprob` (in `common.py`) tokenises prompt and reply, runs one forward pass, applies log-softmax, gathers the log-probability of each true next token, and sums over the *reply* positions only. It returns a 0-d tensor. With `grad=True` the policy's numbers keep their autograd graph (needed to backpropagate); the reference numbers are computed under `torch.no_grad()`. Four forward passes per pair.
- **Why it is here:** These four scalars are all the DPO loss needs. (Real trainers batch these and often precompute the reference ones once.)

### 3. The loss family  (`main.py` lines 52–62)
```python
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
```
- **In simple words:** One line computes how much more the policy has moved toward the chosen reply than toward the rejected one; the three branches turn that number into a loss in three different ways.
- **Exactly:** `logits` is `(pi_c - ref_c) - (pi_r - ref_r)`: the difference of the two log-ratios (a 0-d tensor). **DPO**: `-logsigmoid(beta * logits)`: exactly the step-04 Bradley-Terry loss with the implicit reward. **IPO**: `(logits - 1/(2 beta))^2` has a finite target, so the margin is not pushed forever. **SimPO**: ignores the reference; `pi_c / n_c` is the average log-probability per token (`n_c`, `n_r` = token counts of the replies), multiplied by 2.0 (SimPO's `beta`) and shifted by a target margin `gamma = 1.0`. `kind` and `beta` default to the module-level knobs; `n_c, n_r` are used only by SimPO.
- **Why it is here:** One function, three losses: that makes the variants comparable and shows how little code separates them.

### 4. Metrics on a list of pairs (no grad)  (`main.py` lines 64–80)
```python
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
```
- **In simple words:** Compute the dashboard you would watch during a real DPO run, averaged over a list of pairs.
- **Exactly:** For each pair the implicit rewards are `BETA * (log pi - log ref)` for chosen and rejected. The function returns means of both rewards, their margin, accuracy (fraction of pairs with `reward_chosen > reward_rejected`) and the mean `log pi(chosen)`. `show` prints them on one line. These are the numbers TRL logs as `rewards/chosen`, `rewards/rejected`, `rewards/margins`, `rewards/accuracies`. Note: they always use the DPO-style implicit reward with `BETA`, even when `LOSS=simpo`, in which case they are only a monitoring view (SimPO does not train that reward).
- **Why it is here:** The loss alone cannot reveal likelihood displacement or over-optimisation; the extra columns can.

### 5. Before training  (`main.py` lines 82–87)
```python
# ## 5. Before training
print(f"loss={LOSS} beta={BETA} lr={LR}  ln2={math.log(2):.3f}")
show("test, step 0", metrics(test))
first = train[0]
pi_c, pi_r, ref_c, ref_r = pair_logps(first, grad=False)
print(f"first pair loss at step 0 = {preference_loss(pi_c, pi_r, ref_c, ref_r, 1, 1).item():.3f}  (policy == reference so the margin is 0)")
```
- **In simple words:** Before touching the weights, print the dashboard and the loss of one pair. With identical models the margin must be exactly zero and the DPO loss `ln 2`.
- **Exactly:** The `1, 1` passed as `n_c, n_r` only matter for SimPO. This is a *self-test*: if it does not print 0.693 (DPO), the reference is not an exact copy or something is mis-wired.
- **Why it is here:** The first number to check in any DPO implementation.

### 6. Train: one pair per step, accumulate 4 pairs per optimiser step  (`main.py` lines 89–112)
```python
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
```
- **In simple words:** Go through the pairs four at a time; add up their gradients, take one optimiser step per group of four, and print the dashboard after each epoch.
- **Exactly:** This is *gradient accumulation*: `loss / 4` then `loss.backward()` four times accumulates the gradient of the *mean* loss of four pairs in `.grad`, so a single `opt.step()` equals a batch of 4. `zero_grad()` is called only once per group. 60 pairs gives 15 steps per epoch. Gradient clipping at norm 1.0 guards against a rare large step. `policy.train()` enables dropout on the policy while `reference` stays in `eval()`; `policy.eval()` before `metrics` so the dashboard is deterministic. `n_c`, `n_r` are the token counts of the *reply alone* (used by SimPO). `batch_loss` is the mean loss over the four pairs (the four `loss.item()` values are already divided by 4).
- **Why it is here:** One pair per forward pass keeps memory tiny on a CPU, yet behaves like batch size 4; real trainers batch properly.

### 7. What did the policy learn? Sample greedily before vs after  (`main.py` lines 114–120)
```python
# ## 7. What did the policy learn? sample greedily before vs after
def reply(model, prompt):
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:])
for q in ("User: Where is my order?\nAssistant:", "User: Can I get a refund?\nAssistant:"):
    print(f"\n{q!r}\n  reference: {reply(reference, q)!r}\n  policy   : {reply(policy, q)!r}")
```
- **In simple words:** Ask the untouched reference and the trained policy the same two questions and compare what they write.
- **Exactly:** Greedy decoding of up to 10 new tokens with `generate`; slice off the prompt; decode. Because `reference` is a frozen copy of the starting weights, it shows the *before* behaviour without storing it separately.
- **Why it is here:** Log-prob dashboards are abstract; a changed sentence is the proof of behaviour change. These two questions are not in the training templates exactly as phrased, so it also hints at generalisation.

### 8. Variants at a glance  (`main.py` lines 122–129)
```python
# ## 8. Variants at a glance
print("""
variant  reference model?  loss in one line
DPO      yes               -log sigmoid(beta * [(log pi_c - log ref_c) - (log pi_r - log ref_r)])
IPO      yes               (margin - 1/(2 beta))^2 : squared loss, bounded target margin, resists over-fitting
KTO      yes               needs only thumbs-up / thumbs-down per example (no pairs); prospect-theory-shaped value function
ORPO     NO                SFT loss on chosen + odds-ratio penalty on rejected; one stage, no reference
SimPO    NO                length-normalised avg log-prob as reward + target margin gamma; no reference""")
```
- **In simple words:** A cheat-sheet printed at the end.
- **Exactly:** A triple-quoted string; no computation. KTO and ORPO are *not* implemented here.
- **Why it is here:** To place DPO among its siblings: which need a reference, which need pairs.

### 9. TRY IT / BREAK IT comments  (`main.py` lines 131–132)
```python
# TRY IT: BETA=0.5 python main.py (stays closer to the reference)  |  LOSS=ipo python main.py  |  LOSS=simpo python main.py
# BREAK IT: LR=5e-4 python main.py  -> look at logp(chosen): the policy wrecks BOTH replies (likelihood displacement).
```
- **In simple words:** Knobs to turn; see Experiments and Debugging below.
- **Exactly:** Comments only.
- **Why it is here:** Module convention.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `copy.deepcopy(model)` | an independent copy of the module and all its tensors |
| `p.requires_grad_(False)` | freeze one tensor (trailing `_` = in place) |
| `loss / 4; loss.backward()` | gradient accumulation: gradients add up in `.grad` |
| `with torch.no_grad():` | no autograd graph inside the block |
| `kind=LOSS` default arg | the default is fixed *when the function is defined* (module-level knob) |
| `f"{'':<18}"` | 18 spaces (an empty string left-aligned in 18 columns) |
| `sum(a > b for a, b in zip(rc, rr))` | counts pairs where `a > b` |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `seq_logprob` (`common.py`) | `log p(reply \| prompt)` summed over reply tokens | one forward pass, log-softmax, gather the true next-token log-probabilities, sum over reply positions |
| `F.logsigmoid` | `log(sigmoid(x))` | stable form `-softplus(-x)` |
| `torch.nn.utils.clip_grad_norm_` | caps the global gradient norm | rescales all gradients if the L2 norm exceeds 1.0 |
| `model.generate(do_sample=False)` | greedy decoding | argmax at each step with a KV cache |
| `copy.deepcopy` | clones the reference | duplicates parameter tensors (memory x2) |

## Execution trace (real run)
```
loss=dpo beta=0.1 lr=5e-06  ln2=0.693
test, step 0      r_chosen +0.000  r_rejected -0.000  margin +0.000  acc 0.10  logp(chosen) -34.04
first pair loss at step 0 = 0.693  (policy == reference so the margin is 0)
test, epoch 1     r_chosen +0.296  r_rejected -0.246  margin +0.542  acc 1.00  logp(chosen) -31.08
                  mean train loss 0.578
test, epoch 2     r_chosen +0.508  r_rejected -0.597  margin +1.105  acc 1.00  logp(chosen) -28.95
                  mean train loss 0.307
test, epoch 3     r_chosen +0.674  r_rejected -1.040  margin +1.714  acc 1.00  logp(chosen) -27.30
                  mean train loss 0.160

'User: Where is my order?\nAssistant:'
  reference: " I'm in the middle of a busy day."
  policy   : "\nI'm currently in the process of getting my"

'User: Can I get a refund?\nAssistant:'
  reference: ' I can get a refund.\nAssistant: I'
  policy   : ' Yes, we can get a refund.\nIf'

variant  reference model?  loss in one line
DPO      yes               -log sigmoid(beta * [(log pi_c - log ref_c) - (log pi_r - log ref_r)])
IPO      yes               (margin - 1/(2 beta))^2 : squared loss, bounded target margin, resists over-fitting
KTO      yes               needs only thumbs-up / thumbs-down per example (no pairs); prospect-theory-shaped value function
ORPO     NO                SFT loss on chosen + odds-ratio penalty on rejected; one stage, no reference
SimPO    NO                length-normalised avg log-prob as reward + target margin gamma; no reference
```
(Progress bars and the torch Redirects note omitted.)
- **Step 0**: rewards are exactly 0 and the margin is 0, and the first pair's loss is 0.693 = `ln 2`: policy and reference are identical, as designed. The dashboard prints `acc 0.10` even though nothing has been learned: accuracy uses a strict `>` and the ties (0.000 vs -0.000, up to floating-point noise) are not counted as wins, so it is a tie artefact, not a real 10%.
- **Epochs 1-3**: the margin grows (+0.54, +1.11, +1.71), the chosen implicit reward rises (+0.30, +0.51, +0.67) while the rejected one falls (-0.25, -0.60, -1.04), accuracy on the 30 held-out pairs is 1.00 from the first epoch, and the mean train loss falls 0.578, 0.307, 0.160. Half of the test pairs use unseen reply phrases.
- **`logp(chosen)` rises from -34.04 to -27.30**: no likelihood displacement in this run at `lr=5e-6`. The chosen reply became more likely, not just relatively better than the rejected one. (The BREAK IT with a huge learning rate is where this column would fall; it was not run, see Experiments.)
- **Greedy samples did not become obviously polite.** "Where is my order?" goes from `" I'm in the middle of a busy day."` to `"\nI'm currently in the process of getting my"`; "Can I get a refund?" goes from `' I can get a refund.\nAssistant: I'` to `' Yes, we can get a refund.\nIf'` (a little more helpful, still rambling). The pairs teach the model to prefer specific polite sentences over specific rude ones; greedy decoding follows the most likely next tokens, which these are not. Three epochs at a tiny learning rate on 60 pairs of a 82M model shift probabilities, not style wholesale. Read the dashboard *and* the samples: both are needed, and they can disagree.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| (measured, no model needed) DPO loss at different margins, beta 0.1 | margin 0 -> 0.693; +10 -> 0.313; +20 -> 0.127; -20 -> 2.127 | `-logsigmoid(beta * margin)`: small when the chosen log-ratio beats the rejected one, large when it loses |
| (measured) IPO loss at margin 0 and at the target `1/(2 beta) = 5` | 25.0 and 0.0 | squared loss has a finite target, unlike DPO's ever-decreasing loss |
| (not run) `BETA=0.5` | smaller implicit rewards per unit of log-ratio movement; the policy is held closer to the reference | larger `beta` is a tighter leash |
| (not run) `LOSS=ipo` / `LOSS=simpo` | same dashboard computed with a different training loss (SimPO ignores the reference) | variants differ only in `preference_loss` |
| (measured) `LR=5e-4` (100x larger) | train loss 0.069 then 0.000; margin +11.1 to +11.8 but `logp(chosen)` falls from -34.04 to -81.9, -88.0, -83.7; accuracy only 0.87 | likelihood displacement: the margin is huge because the rejected reward fell to about -16 while the chosen reward also fell to about -5 (both below the reference); the policy made both replies much less likely |

## Debugging practice
**BREAK IT (loss-level, real output).** Exercising the file's own `preference_loss` function directly (extracted from `main.py` and called on hand-made numbers) shows two classic bugs without loading a model:
```
kind='bogus' -> ValueError: bogus
sign-swapped loss at margin +10: 1.313 (rewarding the REJECTED reply)
```
How to read it: an unknown `LOSS` value raises `ValueError` with just the value as the message, so `LOSS=dpoo python main.py` fails only after the model is loaded and the step-0 dashboard is computed; check the knob spelling first. The sign-swapped version (using `(pi_r - ref_r) - (pi_c - ref_c)`) raises no error: a margin that *should* give 0.313 gives 1.313 and training would push the policy toward the rejected reply, so reward margin and accuracy would fall instead of rise. The defence is the step-0 self-check (loss = `ln 2`) *plus* watching that the margin rises in the first epoch.

**The file's own BREAK IT (`LR=5e-4`, real run).** The dashboard lines:
```
test, step 0      r_chosen +0.000  r_rejected -0.000  margin +0.000  acc 0.10  logp(chosen) -34.04
test, epoch 1     r_chosen -4.790  r_rejected -15.880  margin +11.090  acc 0.87  logp(chosen) -81.93
test, epoch 3     r_chosen -4.967  r_rejected -16.138  margin +11.171  acc 0.87  logp(chosen) -83.71
```
How to read it: no error, and the loss is 0.000, which looks like perfect training. But `logp(chosen)` dropped from -34 to -84: the policy now finds the *chosen* reply far less likely than the untouched model did, and its greedy answer to "Where is my order?" became `' Great question for help with that and here are the'` (nonsense). The margin is large only because the rejected reply was pushed down even more. Fix: a learning rate around 5e-6, a larger `beta`, an SFT term on the chosen reply, or a stopping rule on `logp(chosen)`.

## Common mistakes
- A reference that is not an exact frozen copy, or one that keeps training (or is in `train()` mode with dropout): the step-0 loss is then not `ln 2`.
- Watching only the loss. Always log chosen and rejected rewards and `log pi(chosen)`: a falling margin-loss can hide likelihood displacement.
- A learning rate that is too large for DPO (it is typically orders of magnitude smaller than SFT's).
- Computing the sequence log-probability over the *prompt* as well as the reply; the prompt cancels only if both replies share the same prompt *and* you take the difference, but the correct definition sums reply tokens only.
- Comparing summed log-probs of replies of different lengths without thinking: longer replies have lower totals (SimPO normalises by length for exactly this reason).
- Treating DPO as online: it never sees the policy's own samples, so it cannot fix mistakes the dataset does not cover.
- Using chosen/rejected pairs that differ in many ways at once (length, tone, correctness): the model learns the cheapest feature, not the one you meant.

## Rebuild checklist
1. A policy and a `deepcopy` reference with `eval()` and `requires_grad_(False)`.
2. `seq_logprob` for chosen and rejected under both models (reference under `no_grad`).
3. `logits = (pi_c - ref_c) - (pi_r - ref_r)` and `loss = -logsigmoid(beta * logits)`.
4. Dashboard: implicit rewards `beta * log-ratio`, margin, accuracy, `log pi(chosen)`.
5. The `ln 2` self-check at step 0.
6. Gradient accumulation over 4 pairs with `loss / 4`, clip, step.
7. Greedy samples from reference and policy.
8. IPO and SimPO branches.

## Say it aloud
"DPO trains the policy directly on preference pairs: the loss is minus log sigmoid of beta times the difference between the chosen and rejected log-ratios against a frozen reference, so the policy itself acts as the reward model and no separate reward model or sampling is needed. At step zero the policy equals the reference so the loss is ln 2. Beta controls how far the policy may drift. I monitor implicit rewards, margin, accuracy and the log-probability of the chosen answer, because a growing margin can come from both answers becoming less likely, which is likelihood displacement. DPO is offline, so for skills the data does not demonstrate you need RL."

## The new problem (cliffhanger)
DPO is offline: it only reshuffles probability among replies someone already wrote. It can make the model politer, but it cannot teach it to *solve* new maths problems, because no pair in the data shows a correct solution being produced by the model itself. For tasks where the answer can be checked by code, we can sample from the current policy and reward success directly: PPO, GRPO and RLVR. On to [step 06](../06_ppo_vs_grpo_rlvr/WALKTHROUGH.md).
