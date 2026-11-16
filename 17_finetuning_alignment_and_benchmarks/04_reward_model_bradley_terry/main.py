"""
GOAL:  Train a reward model from preference pairs with the Bradley-Terry loss: a distilgpt2 backbone + one linear
       "scalar head" reading the LAST token's hidden state. Measure pairwise accuracy on held-out pairs (half of them use
       reply phrases never seen in training), and prove the learning is real with a shuffled-label control.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/04_reward_model_bradley_terry/main.py
EXPECTED OUTPUT (shortened):
    before training: loss near ln 2 (0.81 recorded), held-out accuracy near chance (0.37 recorded)
    epoch losses falling to ~0.0x ; held-out accuracy ~1.0 ; reward(chosen) >> reward(rejected)
    shuffled-label control: no real learning (held-out acc 0.13-0.33 recorded; not exactly 0.5)
NEEDS: torch, transformers; cached distilgpt2.  ~1 minute on CPU.
"""
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

# ## 1. Data and tokenizer
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
tok.padding_side = "right"
train, test = make_preference_data()
print(f"train pairs={len(train)} test pairs={len(test)}\nexample: {train[0]}")

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

# ## 3. Encode a list of pairs into two padded batches
def encode_pairs(pairs):
    chosen = tok([p["prompt"] + p["chosen"] for p in pairs], return_tensors="pt", padding=True)
    rejected = tok([p["prompt"] + p["rejected"] for p in pairs], return_tensors="pt", padding=True)
    return chosen, rejected

# ## 4. The Bradley-Terry loss
def bt_loss(r_chosen, r_rejected):
    # P(chosen beats rejected) = sigmoid(r_c - r_r); the loss is -log of that probability, averaged over the batch
    return -F.logsigmoid(r_chosen - r_rejected).mean()

# ## 5. Held-out evaluation: accuracy, mean margin
@torch.no_grad()
def evaluate(rm, pairs):
    rm.eval()
    c, r = encode_pairs(pairs)
    rc, rr = rm(**c), rm(**r)                                           # (N,), (N,)
    return (rc > rr).float().mean().item(), (rc - rr).mean().item(), bt_loss(rc, rr).item()

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

print("\n== real labels ==")
rm = train_rm(train)

# ## 7. Control: the same recipe on SHUFFLED labels must stay near chance
print("\n== control: chosen/rejected swapped at random in training ==")
rng = random.Random(1)
shuffled = [{**p, "chosen": p["rejected"], "rejected": p["chosen"]} if rng.random() < 0.5 else p for p in train]
train_rm(shuffled)

# ## 8. What does the reward say about concrete replies?
rm.eval()
print("\nrewards for one prompt (higher = better):")
for text in [" Sure, happy to help. Here is how.", " Excellent question, glad to help!", " No. Figure it out yourself.", " Ugh, whatever."]:
    enc = tok(["User: How do I reset my password?\nAssistant:" + text], return_tensors="pt")
    with torch.no_grad():
        print(f"  {rm(**enc).item():+7.3f}  {text!r}")

# TRY IT: EPOCHS=1 python main.py ; LR=2e-4 python main.py (is a bigger LR better?)
# BREAK IT: remove `attention_mask.sum(dim=1) - 1` and always use position -1 with right padding: the "last token" becomes padding.
