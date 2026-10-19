"""
GOAL:  Measure how good the model is on data it did NOT train on (validation loss + perplexity),
       then save a checkpoint and prove that the reloaded model behaves identically.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_fast_inference/03_eval_perplexity_checkpoints/main.py
EXPECTED OUTPUT (shortened):
    train blocks: 20 | validation blocks: 5
    BEFORE  val_loss 4.72  perplexity 112.2
    epoch 1 train_loss 3.99 val_loss 3.23
    ...
    AFTER   val_loss 2.52  perplexity 12.4
    saved checkpoint to .../checkpoints/distilgpt2-support
    max difference between original and reloaded logits: 0.0
NEEDS: torch, transformers (distilgpt2 is cached from step 02).  ~30 s.  Writes a ~330 MB checkpoint
       into ./checkpoints/ (safe to delete; the script recreates it).
"""

import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_device, make_blocks, read_lines

MODEL_NAME = "distilgpt2"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "distilgpt2-support"
EPOCHS, BATCH_SIZE, LR = 4, 4, 5e-5

torch.manual_seed(0)
device = get_device()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

blocks = make_blocks(tokenizer, read_lines(), 48)
# TRAIN / VALIDATION SPLIT: keep the last 20% aside.  Never train on it, or the score is meaningless.
n_val = max(1, int(0.2 * len(blocks)))
train_blocks, val_blocks = blocks[:-n_val], blocks[-n_val:]
print(f"train blocks: {len(train_blocks)} | validation blocks: {len(val_blocks)}")
train_loader = DataLoader(TensorDataset(train_blocks), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_blocks), batch_size=BATCH_SIZE)


def evaluate() -> float:
    """Average loss over the validation set, with no weight updates."""
    model.eval()                              # dropout OFF -> deterministic scores
    total, count = 0.0, 0
    with torch.no_grad():                     # skip gradient bookkeeping
        for (batch,) in val_loader:
            batch = batch.to(device)
            loss = model(input_ids=batch, labels=batch).loss
            total += loss.item() * batch.shape[0]   # weight by batch size (last batch may be smaller)
            count += batch.shape[0]
    model.train()
    return total / count


def perplexity(loss: float) -> float:
    """Perplexity = exp(loss).  Intuition: 'the model is as unsure as if choosing among N equally likely tokens'."""
    return math.exp(loss)


v = evaluate()
print(f"BEFORE  val_loss {v:.2f}  perplexity {perplexity(v):.1f}")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()
for epoch in range(1, EPOCHS + 1):
    running = 0.0
    for (batch,) in train_loader:
        batch = batch.to(device)
        loss = model(input_ids=batch, labels=batch).loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item()
    # 'train_loss' falling while 'val_loss' RISES is the classic sign of overfitting.
    print(f"epoch {epoch} train_loss {running / len(train_loader):.2f} val_loss {evaluate():.2f}")

v = evaluate()
print(f"AFTER   val_loss {v:.2f}  perplexity {perplexity(v):.1f}")

# ---- CHECKPOINTING ----------------------------------------------------------
# save_pretrained writes config.json + weights (safetensors); the tokenizer needs saving too.
CKPT_DIR.mkdir(parents=True, exist_ok=True)       # parents=True: create missing folders; exist_ok: no error if present
model.save_pretrained(CKPT_DIR)
tokenizer.save_pretrained(CKPT_DIR)
print("saved checkpoint to", CKPT_DIR)
print("files:", sorted(p.name for p in CKPT_DIR.iterdir()))

# Reload from disk and compare outputs on the same input.
reloaded = AutoModelForCausalLM.from_pretrained(CKPT_DIR).to(device).eval()
model.eval()
probe = tokenizer("Ticket: How do I sign out? Reply:", return_tensors="pt").to(device)
with torch.no_grad():
    a = model(**probe).logits   # shape: (1, seq_len, 50257)
    b = reloaded(**probe).logits   # shape: (1, seq_len, 50257)
print("max difference between original and reloaded logits:", (a - b).abs().max().item())

# The lower-level PyTorch way: only the weights dictionary (you must rebuild the architecture yourself).
state = model.state_dict()                        # dict: layer name -> tensor
print("state_dict has", len(state), "tensors, e.g.", next(iter(state)))

# TRY IT: set EPOCHS = 12 and watch val_loss start to rise while train_loss keeps falling (overfitting).
# TRY IT: change the 0.2 split to 0.4 - with so little data, how noisy does val_loss become?
# BREAK IT: remove tokenizer.save_pretrained(...) then try AutoTokenizer.from_pretrained(CKPT_DIR) -> OSError.
