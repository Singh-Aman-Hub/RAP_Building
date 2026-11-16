# Step 04 — Reward model with the Bradley-Terry loss

## The story so far
[Step 03](../03_peft_family/WALKTHROUGH.md) showed how to adapt a model cheaply; [step 02](../02_sft_from_scratch_with_loss_masking/WALKTHROUGH.md) taught it to imitate good answers. Imitation has a ceiling: SFT can only say "produce *this* answer". It cannot express "this answer is better than that one" or "never answer like this".

## The pain point
Humans find it hard to write the perfect answer but easy to compare two answers ("A is more helpful than B"). That comparison data is a different shape from SFT data: `(prompt, chosen, rejected)`, with no absolute score. You need a way to turn *pairs* into a *scalar score* a training algorithm can use, and a way to check the score is not noise.

## The spark
Model the probability that a human prefers answer A over B as a logistic function of a score difference (the **Bradley-Terry** model), and train a network to output the score. A pretrained language model with one extra linear layer on top, reading the hidden state of the *last token*, is a reward model. Because only differences matter, the loss is `-log sigmoid(r_chosen - r_rejected)`.

## Goal
Train a reward model on 60 seeded preference pairs (polite reply = chosen, rude reply = rejected) with `distilgpt2` plus a 768-to-1 scalar head. Report held-out pairwise accuracy and mean margin, where half the test pairs use reply phrases never seen in training. Then run a *shuffled-label control*: the same recipe on randomly swapped labels must stay at chance, which proves the real run learned something real. Finally read the reward the model gives to concrete replies.

## The theory (read this before the interview)
**Bradley-Terry model:** each answer `y` has a latent score `r(x, y)`. The probability that `y_w` (chosen, "winner") beats `y_l` (rejected, "loser") is
```
P(y_w > y_l | x) = sigmoid( r(x, y_w) - r(x, y_l) ) = 1 / (1 + exp( -(r_w - r_l) ))
```
**Loss** (negative log-likelihood of the observed preferences):
```
L = - E[ log sigmoid( r_w - r_l ) ]
```
Properties worth stating in an interview:
- *Only differences matter*: adding a constant to every reward changes nothing, so the absolute scale is arbitrary (reward models are often normalised before RL).
- *Start value*: if the two rewards are equal, `sigmoid(0) = 0.5` and the loss is `-log 0.5 = ln 2 = 0.693`. A reward model at initialisation should therefore show a loss *near* 0.69 and pairwise accuracy *near* 0.5. A freshly initialised head is not exactly zero, so expect a little off (in the recorded run 0.811 and 0.367); a start far from `ln 2` points to a bug or a badly scaled head.
- The gradient is large when the model ranks a pair wrongly and shrinks as the margin grows.

**Architecture:** backbone (a causal LM *without* the vocabulary head, `AutoModel`) outputs hidden states `(B, L, d)`. Because the model is causal, only the last real token has seen the whole sequence, so we take *its* hidden state, `(B, d)`, and apply `Linear(d, 1)`. The result is one scalar per sequence. With right padding the last real token is at index `attention_mask.sum() - 1`, not at `-1`.

**Evaluation:** *pairwise accuracy* = fraction of held-out pairs where `r_chosen > r_rejected`; *margin* = mean of `r_chosen - r_rejected`; also the BT loss itself. Always keep unseen phrasing in the test set and add a shuffled-label control.

**Where it sits in RLHF:** (1) SFT, (2) collect preferences and train this reward model, (3) optimise the policy against the reward with PPO or GRPO plus a KL penalty to a reference model ([step 06](../06_ppo_vs_grpo_rlvr/WALKTHROUGH.md)). DPO ([step 05](../05_dpo_from_scratch/WALKTHROUGH.md)) skips the explicit reward model by showing the policy *is* a reward model.

**Caveats:** a learned reward is a proxy, so an RL policy can exploit its blind spots (reward hacking, over-optimisation); preference labels carry annotator noise and biases (length, style); a reward model trained on one distribution can be unreliable on another.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Reward model | RM, preference model, scoring model | usually | "Preference model" stresses it is trained from comparisons |
| Bradley-Terry loss | pairwise ranking loss, logistic loss on score differences, RankNet-style loss | near | All push `r_chosen` above `r_rejected` through a sigmoid |
| Chosen / rejected | preferred / dispreferred, winner / loser, `y_w` / `y_l` | yes | Naming only |
| Scalar head | value head, reward head, regression head | near | A `Linear(d, 1)` on a hidden state |
| Pairwise accuracy | preference accuracy, ranking accuracy | yes | Fraction of pairs ranked correctly |
| Margin | reward gap, `r_w - r_l` | yes | Mean score difference on held-out pairs |
| Control experiment | shuffled-label baseline, sanity check | near | Same pipeline with the signal removed |
| Right padding | pad after the text | no | Last real token is at `mask.sum() - 1`; left padding would put it at `-1` |
| Reward hacking | over-optimisation, Goodhart's law | near | The policy exploits the proxy, not the goal |

## Run it
```bash
cd 17_finetuning_alignment_and_benchmarks/04_reward_model_bradley_terry
python main.py
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/04_reward_model_bradley_terry/main.py
# knobs: SEED, EPOCHS (3), LR (2e-5)
```
Two trainings of distilgpt2; about a minute on an idle laptop CPU, varying by machine (and much slower when the CPU is shared).

## Code, step by step

Lines 1–13 of `main.py` are the header docstring (GOAL / RUN / EXPECTED OUTPUT / NEEDS) already summarised above; the walk starts after it.

### 0. Imports and settings  (`main.py` lines 14–33)
```python
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_preference_data   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 2e-5))
SHUFFLE_LABELS = int(os.environ.get("SHUFFLE_LABELS", 0))
torch.manual_seed(SEED)
random.seed(SEED)
```
- **In simple words:** Imports, the shared preference-data helper, and the knobs.
- **Exactly:** `AutoModel` (not `AutoModelForCausalLM`) returns hidden states without the vocabulary head. `make_preference_data()` returns `(train, test)` lists of dicts `{"prompt", "chosen", "rejected"}`. `SHUFFLE_LABELS` is read but not used later: the control run in section 7 is built unconditionally, so this knob is vestigial (changing it does nothing).
- **Why it is here:** Same pattern as the earlier steps: one block of knobs, seeded for repeatability.

### 1. Data and tokenizer  (`main.py` lines 35–40)
```python
# ## 1. Data and tokenizer
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
tok.padding_side = "right"
train, test = make_preference_data()
print(f"train pairs={len(train)} test pairs={len(test)}\nexample: {train[0]}")
```
- **In simple words:** Load the tokenizer, say padding goes at the end, and build 60 training and 30 test pairs.
- **Exactly:** GPT-2 has no pad token, so EOS is reused (safe here because the attention mask hides pads). `padding_side="right"` is explicit because the pooling in block 2 *assumes* it. In `common.py` the chosen replies are polite sentences, the rejected ones rude sentences; the *last two phrases of each pool appear only in the test set*, so half of the 30 test pairs are unseen phrasing.
- **Why it is here:** Data design is half the experiment: the unseen-phrase half of the test set turns "accuracy" into a generalisation measurement.

### 2. The reward model = backbone + scalar head  (`main.py` lines 42–55)
```python
# ## 2. The reward model = backbone + scalar head
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("distilgpt2")        # outputs hidden states, NOT vocab logits
        self.head = nn.Linear(self.backbone.config.n_embd, 1)           # 768 -> 1 scalar reward
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # (B, L, 768)
        last = attention_mask.sum(dim=1) - 1                            # index of the last real token in each row (B,)
        pooled = h[torch.arange(h.size(0)), last]                       # (B, 768) hidden state of that token
        return self.head(pooled).squeeze(-1)                            # (B,) one scalar per sequence
```
- **In simple words:** A language model body plus one small layer that turns "the summary vector of the last word" into a single score.
- **Exactly:** `self.backbone(...).last_hidden_state` has shape `(B, L, 768)`. `attention_mask.sum(dim=1) - 1` counts real tokens per row and subtracts one: the index of the last real token `(B,)`. `h[torch.arange(B), last]` is *advanced indexing*: it picks `h[b, last[b]]` for every row at once and returns `(B, 768)`. `self.head` maps that to `(B, 1)`; `.squeeze(-1)` gives `(B,)`. The head is initialised with a small normal (`std=0.02`) and zero bias so the initial rewards are small and nearly equal, giving a starting loss near (but not exactly at) `ln 2`. `nn.Module` subclasses register `self.backbone` and `self.head`, so `rm.parameters()` includes both and all of them are trained (this is full fine-tuning, not PEFT).
- **Why it is here:** This is the standard reward-model design (RM = LM trunk + scalar head). Taking the last token is the causal-model analogue of BERT's `[CLS]` token.

### 3. Encode a list of pairs into two padded batches  (`main.py` lines 57–61)
```python
# ## 3. Encode a list of pairs into two padded batches
def encode_pairs(pairs):
    chosen = tok([p["prompt"] + p["chosen"] for p in pairs], return_tensors="pt", padding=True)
    rejected = tok([p["prompt"] + p["rejected"] for p in pairs], return_tensors="pt", padding=True)
    return chosen, rejected
```
- **In simple words:** Turn the chosen texts and the rejected texts into two separate padded batches of token ids.
- **Exactly:** Each text is `prompt + reply`. `tok(list_of_str, return_tensors="pt", padding=True)` returns a `BatchEncoding` with `input_ids` and `attention_mask`, both `(N, L)`, padded to the longest text *in that call*. Chosen and rejected get *different* `L`; that is fine because they are scored in two separate forward passes.
- **Why it is here:** The reward is a function of the *whole* sequence (prompt and reply), so both are encoded together.

### 4. The Bradley-Terry loss  (`main.py` lines 63–66)
```python
# ## 4. The Bradley-Terry loss
def bt_loss(r_chosen, r_rejected):
    # P(chosen beats rejected) = sigmoid(r_c - r_r); the loss is -log of that probability, averaged over the batch
    return -F.logsigmoid(r_chosen - r_rejected).mean()
```
- **In simple words:** Push the chosen score above the rejected score; the further above, the smaller the penalty.
- **Exactly:** `F.logsigmoid(z)` computes `log(sigmoid(z))` in a numerically stable way (better than `torch.log(torch.sigmoid(z))`, which underflows for large negative `z`). Inputs `(B,)`, output a scalar after `.mean()`. At `z = 0` it returns `ln 2 = 0.693`.
- **Why it is here:** One line is the entire objective. DPO in [step 05](../05_dpo_from_scratch/WALKTHROUGH.md) uses the same function with a different `z`.

### 5. Held-out evaluation: accuracy, mean margin, loss  (`main.py` lines 68–74)
```python
# ## 5. Held-out evaluation: accuracy, mean margin
@torch.no_grad()
def evaluate(rm, pairs):
    rm.eval()
    c, r = encode_pairs(pairs)
    rc, rr = rm(**c), rm(**r)                                           # (N,), (N,)
    return (rc > rr).float().mean().item(), (rc - rr).mean().item(), bt_loss(rc, rr).item()
```
- **In simple words:** Score every test pair both ways and report how often chosen wins, by how much, and the loss.
- **Exactly:** `rm(**c)` unpacks the `BatchEncoding` into `input_ids=..., attention_mask=...`. `(rc > rr)` is a boolean tensor; `.float().mean()` is the fraction True, i.e. accuracy. `rm.eval()` switches off dropout so scores are deterministic. Returns three Python floats.
- **Why it is here:** Training loss going down proves nothing about generalisation; this is the number that does.

### 6. Train  (`main.py` lines 76–99)
```python
# ## 6. Train
def train_rm(pairs):
    torch.manual_seed(SEED)
    rm = RewardModel()
    opt = torch.optim.AdamW(rm.parameters(), lr=LR)
    acc, margin, loss = evaluate(rm, test)
    print(f"  before training: held-out loss {loss:.3f} (ln 2 = {math.log(2):.3f})  accuracy {acc:.3f}  margin {margin:+.3f}")
    rng = random.Random(SEED)
    for ep in range(EPOCHS):
        rm.train()
        order = list(range(len(pairs)))
        rng.shuffle(order)
        run = []
        for s in range(0, len(order), 8):
            batch = [pairs[i] for i in order[s:s + 8]]
            c, r = encode_pairs(batch)
            loss = bt_loss(rm(**c), rm(**r))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run.append(loss.item())
        acc, margin, tl = evaluate(rm, test)
        print(f"  epoch {ep + 1}: train loss {sum(run) / len(run):.3f} | held-out accuracy {acc:.3f} margin {margin:+.3f}")
    return rm
```
- **In simple words:** Build a fresh reward model, measure it before any training, then train for a few epochs on batches of 8 pairs, measuring held-out performance after each epoch.
- **Exactly:** Wrapping everything in a function lets us call it twice (real and shuffled labels) from the same seed. The "before training" line is the sanity check that the start is `ln 2` and accuracy near 0.5. In each step the chosen and rejected batches go through the model separately, the BT loss is taken on the two score vectors, and the usual `zero_grad / backward / step` follows. 60 pairs with batch 8 gives 8 steps per epoch (the last has 4 pairs). There is no gradient clipping here, unlike step 02. `rm.train()` / `rm.eval()` toggle dropout.
- **Why it is here:** The "before" line catches initialisation bugs; the per-epoch held-out line shows *when* generalisation appears.

### 7. Run for real, then the control on SHUFFLED labels  (`main.py` lines 101–108)
```python
print("\n== real labels ==")
rm = train_rm(train)

# ## 7. Control: the same recipe on SHUFFLED labels must stay near chance
print("\n== control: chosen/rejected swapped at random in training ==")
rng = random.Random(1)
shuffled = [{**p, "chosen": p["rejected"], "rejected": p["chosen"]} if rng.random() < 0.5 else p for p in train]
train_rm(shuffled)
```
- **In simple words:** Run the real training, then run it again on data where, for about half the pairs, "chosen" and "rejected" are swapped at random. If the recipe still "learns", the recipe is broken.
- **Exactly:** `rm = train_rm(train)` is the real run and keeps the trained model for block 8. `{**p, "chosen": ..., "rejected": ...}` copies the dict and overrides two keys. The control tests on the *real* held-out pairs, so with random labels the model can at best fit noise and cannot rank the test pairs correctly.
- **Why it is here:** It is the cheapest proof that your accuracy is not an artefact of data leakage or an evaluation bug. A result without a control is a claim.

### 8. What does the reward say about concrete replies?  (`main.py` lines 110–116)
```python
# ## 8. What does the reward say about concrete replies?
rm.eval()
print("\nrewards for one prompt (higher = better):")
for text in [" Sure, happy to help. Here is how.", " Excellent question, glad to help!", " No. Figure it out yourself.", " Ugh, whatever."]:
    enc = tok(["User: How do I reset my password?\nAssistant:" + text], return_tensors="pt")
    with torch.no_grad():
        print(f"  {rm(**enc).item():+7.3f}  {text!r}")
```
- **In simple words:** Ask the trained (real-label) model to score four replies to the same question, two from the training pool, one new polite phrase and one new rude phrase.
- **Exactly:** A single-text batch needs no padding; `rm(**enc)` returns shape `(1,)` and `.item()` a float. `:+7.3f` prints the sign. `rm` here is the model from the real-label run (the control's return value was not stored).
- **Why it is here:** Numbers become meaningful when you see them on sentences. The two sentences the model never saw (`Excellent question...` and `Ugh, whatever.`) test generalisation by eye.

### 9. TRY IT / BREAK IT comments  (`main.py` lines 118–119)
```python
# TRY IT: EPOCHS=1 python main.py ; LR=2e-4 python main.py (is a bigger LR better?)
# BREAK IT: remove `attention_mask.sum(dim=1) - 1` and always use position -1 with right padding: the "last token" becomes padding.
```
- **In simple words:** Two experiments; see Experiments and Debugging below.
- **Exactly:** Comments only.
- **Why it is here:** Module convention.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `h[torch.arange(B), last]` | advanced indexing: row `b` takes position `last[b]`, giving `(B, d)` |
| `attention_mask.sum(dim=1) - 1` | last real index per row |
| `.squeeze(-1)` | remove a trailing axis of size 1 |
| `rm(**c)` | unpack a `BatchEncoding` into keyword arguments |
| `{**p, "chosen": x}` | copy a dict, overriding some keys |
| `(rc > rr).float().mean()` | fraction of True entries |
| `f"{x:+.3f}"` | print with an explicit sign |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `AutoModel.from_pretrained("distilgpt2")` | loads the transformer trunk with no LM head | builds `GPT2Model`; `last_hidden_state` is the output of the final layer norm |
| `nn.Linear(768, 1)` | the scalar head | a `(1, 768)` weight and a bias; `y = xW^T + b` |
| `F.logsigmoid` | `log(sigmoid(x))` | uses `-softplus(-x)` for numerical stability |
| `nn.init.normal_`, `nn.init.zeros_` | in-place initialisation | fills the tensor under `no_grad` |
| `torch.optim.AdamW` | optimiser | per-parameter moment estimates, decoupled weight decay |

## Execution trace (real run)
```
train pairs=60 test pairs=30
example: {'prompt': 'User: How do I contact support?\nAssistant:', 'chosen': ' Thanks for asking. I can help with that.', 'rejected': ' No. Figure it out yourself.'}

== real labels ==
  before training: held-out loss 0.811 (ln 2 = 0.693)  accuracy 0.367  margin -0.163
  epoch 1: train loss 0.396 | held-out accuracy 1.000 margin +2.659
  epoch 2: train loss 0.043 | held-out accuracy 1.000 margin +5.248
  epoch 3: train loss 0.006 | held-out accuracy 1.000 margin +5.821

== control: chosen/rejected swapped at random in training ==
  before training: held-out loss 0.811 (ln 2 = 0.693)  accuracy 0.367  margin -0.163
  epoch 1: train loss 0.779 | held-out accuracy 0.333 margin -0.260
  epoch 2: train loss 0.626 | held-out accuracy 0.167 margin -0.338
  epoch 3: train loss 0.663 | held-out accuracy 0.133 margin -0.387

rewards for one prompt (higher = better):
   +6.546  ' Sure, happy to help. Here is how.'
   +2.027  ' Excellent question, glad to help!'
   -2.332  ' No. Figure it out yourself.'
   -1.571  ' Ugh, whatever.'
```
(Progress bars and the torch Redirects note on stderr are omitted.)
- **Before training**: held-out loss 0.811 (close to, not exactly, `ln 2 = 0.693`) and accuracy 0.367 with margin -0.163. The head is initialised near zero but not at zero, so the two scores of a pair are small and not equal; on 30 pairs this random start happens to rank 11 of 30 correctly (0.367). "About `ln 2` and about chance" is the right expectation, not an exact 0.693 and 0.5.
- **Real labels**: held-out accuracy is 1.000 after the first epoch and the margin grows (+2.66, +5.25, +5.82) while the training loss falls 0.396, 0.043, 0.006. Half of the 30 test pairs use reply phrases never seen in training, so this is more than memorising sentences, but the task is easy (polite versus rude) and the test set is small: say "perfect on 30 pairs", not "solved".
- **Control (random swaps)**: held-out accuracy 0.333, 0.167, 0.133, *below* chance, and the training loss stays high (0.78, 0.63, 0.66). The control did what it is for: the recipe does not reach the real run's 1.000 without real signal. It is not a perfect null: with `random.Random(1)` the swap happened to flip 33 of the 60 pairs (55%), so the noisy labels lean slightly toward "rude is chosen", and a tiny model that fits that lean scores under 0.5 on the real test pairs. Run the control with several seeds if you want an average near 0.5.
- **Concrete replies**: the two training-pool replies get the extremes (+6.5 for the polite one, -2.3 for the rude one). The two sentences never seen in training are ordered correctly too: `Excellent question, glad to help!` +2.03 and `Ugh, whatever.` -1.57. Their magnitudes are smaller than the seen ones, which is typical: the model is less sure out of distribution. Only differences matter; the absolute values have no meaning.


## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| default (lr 2e-5, 3 epochs) | held-out accuracy 1.000 from epoch 1, margin +2.66 to +5.82; control 0.13-0.33 | baseline |
| (measured) `LR=2e-4 EPOCHS=2` (10x larger LR) | real run: accuracy 1.000 after epoch 1 with margin +7.18, +12.68; rewards for the four sample replies +8.2 / +1.0 / -8.6 / -2.5; control: 0.367 then 0.133 | a bigger learning rate is not more accurate here (already 1.000) but inflates the margins and the reward scale; with RL downstream, a huge reward scale is a liability |
| (not run) `EPOCHS=1` | accuracy probably already near 1.000 (epoch 1 of the default run was 1.000) | the task is easy |

## Debugging practice
**BREAK IT:** replace `last = attention_mask.sum(dim=1) - 1` with `last = -1`. With right padding, every shorter sequence in the batch ends in padding (EOS ids hidden by the attention mask); the model then pools the hidden state of a *pad* position. There is no exception. The symptom is that scores depend on how much padding a text received, i.e. on its length relative to the longest text in the batch: chosen and rejected replies of different lengths get different amounts of padding and the learned signal becomes partly a length artefact. Detect it by printing `last` for a batch and checking that it differs per row and that `input_ids[b, last[b]]` is not the pad id; also evaluate a text alone versus inside a padded batch, which must give the same score. (Not run here; it is a design bug rather than a crash.)

## Common mistakes
- Pooling position `-1` with right padding (see above), or forgetting the padding side entirely.
- No control run: a high accuracy that came from leakage looks identical to a real one.
- Test data that shares phrasing with training, so accuracy measures memorisation.
- Reading the absolute reward value: only differences carry meaning.
- Training the reward model for too many epochs on a small set: it can overfit annotator noise and then be exploited by the policy.
- Confusing a reward model (scores an answer) with a policy (generates one): the backbone here has no LM head and cannot generate.
- Ignoring length bias: if chosen answers are systematically longer, the model can learn "longer is better".

## Rebuild checklist
1. `RewardModel`: `AutoModel` trunk, `Linear(d, 1)` head, last-real-token pooling with `attention_mask.sum(1) - 1`.
2. `encode_pairs` returning two padded batches (right padding).
3. `bt_loss = -logsigmoid(r_c - r_r).mean()`.
4. `evaluate` returning accuracy, margin, loss with `no_grad` and `eval()`.
5. A training function you can call twice, printing the `ln 2` before-line.
6. The shuffled-label control.
7. Scores for a few concrete replies.

## Say it aloud
"A reward model is a language-model trunk with a scalar head on the last real token, trained on chosen and rejected pairs with the Bradley-Terry loss, minus log sigmoid of the reward difference. Only differences matter, and at initialisation the loss is near ln 2, about 0.69. I check it on held-out pairs that include unseen phrasing, and I run a shuffled-label control, which should not learn the real signal (it stayed far below the real run's 100% here), to prove the learning is real. The reward is a proxy, so an RL policy trained against it can exploit it."

## The new problem (cliffhanger)
The reward model is a second network to train, store and serve, and RL against it needs sampling from the policy as well. Is there a way to use the same preference pairs to train the policy *directly*, with no reward model and no sampling? Yes: DPO, which shows the policy and a frozen reference already define a reward. On to [step 05](../05_dpo_from_scratch/WALKTHROUGH.md).
