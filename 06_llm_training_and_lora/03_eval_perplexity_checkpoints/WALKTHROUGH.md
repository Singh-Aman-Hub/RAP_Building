# Step 03 — Evaluation, perplexity and checkpoints

## The story so far
[Step 02](../02_training_loop/WALKTHROUGH.md) drove the training loss down. That number only tells you how well the model knows the text it has already seen.

## The pain point
Train long enough on a small file and the model memorises it: training loss near zero, new text no better, maybe worse. Without a held-out set you will not notice, and if the run crashes at hour 9 you have nothing to resume from.

## The spark
Hold out validation data and measure its loss; perplexity is `exp(loss)`, the average number of tokens the model is torn between. Save a checkpoint with `save_pretrained`, reload it, and check the numbers match.

## Goal
Know whether training *generalised* (validation loss / perplexity), and save + reload a model correctly.

## Run it
```bash
cd 06_llm_training_and_lora/03_eval_perplexity_checkpoints
python main.py      # writes ./checkpoints (~330 MB, safe to delete)
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Validation set / dev set / held-out set | eval split | Yes | Data the model never trains on, used to judge it. A test set is a second held-out set touched only at the end. |
| Perplexity / exp(loss) | PPL | Yes | `exp(average cross-entropy)`: as unsure as choosing among N equally likely tokens. |
| Checkpoint / saved model / snapshot | `save_pretrained`, weights file | Close | A checkpoint can also include optimizer state; here it is weights + config + tokenizer. |
| `save_pretrained` / `state_dict` / `torch.save` | HF vs raw PyTorch | Different | `save_pretrained` writes config + weights so `from_pretrained` rebuilds everything; `state_dict` is only the name-to-tensor dict. |
| Overfitting / memorising | high variance | Yes | Train loss keeps falling while validation loss rises. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–17)

```python
"""
GOAL:  Measure how good the model is on data it did NOT train on (validation loss + perplexity),
       then save a checkpoint and prove that the reloaded model behaves identically.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_lora/03_eval_perplexity_checkpoints/main.py
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
```

- **In simple words:** What the file does, and a warning that it writes a big folder.
- **Exactly:** Docstring; not executed. NEEDS note: the checkpoint is about 330 MB under `./checkpoints/` (it is git-ignored).
- **Why it is here:** Warns you before you fill your disk.

### 2. Imports  (`main.py` lines 19–28)

```python
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_device, make_blocks, read_lines
```

- **In simple words:** Libraries plus the shared helpers from `common.py`.
- **Exactly:** `math` for `exp`; `sys.path.insert` makes `common` importable.
- **Why it is here:** Without them the script cannot tokenize or load the model.

### 3. Settings, seed, device, model  (`main.py` lines 30–37)

```python
MODEL_NAME = "distilgpt2"
CKPT_DIR = Path(__file__).resolve().parent / "checkpoints" / "distilgpt2-support"
EPOCHS, BATCH_SIZE, LR = 4, 4, 5e-5

torch.manual_seed(0)
device = get_device()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
```

- **In simple words:** Choose the model and where to save, set the training sizes and load everything.
- **Exactly:** `CKPT_DIR = .../checkpoints/distilgpt2-support`; `EPOCHS=4, BATCH_SIZE=4, LR=5e-5`. Same loading calls as step 02.
- **Why it is here:** Same starting point as step 02 so numbers are comparable.

### 4. Train/validation split and loaders  (`main.py` lines 39–45)

```python
blocks = make_blocks(tokenizer, read_lines(), 48)
# TRAIN / VALIDATION SPLIT: keep the last 20% aside.  Never train on it, or the score is meaningless.
n_val = max(1, int(0.2 * len(blocks)))
train_blocks, val_blocks = blocks[:-n_val], blocks[-n_val:]
print(f"train blocks: {len(train_blocks)} | validation blocks: {len(val_blocks)}")
train_loader = DataLoader(TensorDataset(train_blocks), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_blocks), batch_size=BATCH_SIZE)
```

- **In simple words:** Hold back the last 20% of the data and never train on it.
- **Exactly:** `n_val = max(1, int(0.2 * 25)) = 5`; `train_blocks` is `(20, 48)`, `val_blocks` is `(5, 48)`. Real: `train blocks: 20 | validation blocks: 5`. The validation loader has no shuffle.
- **Why it is here:** Scoring on training data only measures memory; the held-out slice measures generalisation.

### 5. The evaluate function  (`main.py` lines 48–59)

```python
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
```

- **In simple words:** Compute the average loss over the held-out blocks without learning.
- **Exactly:** `model.eval()` (dropout off), `torch.no_grad()`, loss per batch weighted by `batch.shape[0]` because the last batch is smaller, then total/count. `model.train()` restores training mode.
- **Why it is here:** Without the weighting the average is slightly wrong; without eval mode scores are noisy.

### 6. Perplexity helper  (`main.py` lines 62–64)

```python
def perplexity(loss: float) -> float:
    """Perplexity = exp(loss).  Intuition: 'the model is as unsure as if choosing among N equally likely tokens'."""
    return math.exp(loss)
```

- **In simple words:** Convert the loss into a number people can picture.
- **Exactly:** `math.exp(loss)`; loss 4.72 gives 112.2.
- **Why it is here:** Perplexity is the number quoted in papers and interviews.

### 7. Score before training  (`main.py` lines 67–68)

```python
v = evaluate()
print(f"BEFORE  val_loss {v:.2f}  perplexity {perplexity(v):.1f}")
```

- **In simple words:** Baseline: how good is the untouched model?
- **Exactly:** Real: `val_loss 4.72 perplexity 112.2`.
- **Why it is here:** Without a baseline you cannot say the training helped.

### 8. Training loop with validation each epoch  (`main.py` lines 70–83)

```python
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
```

- **In simple words:** Same loop as step 02, but after each epoch we also score the held-out data.
- **Exactly:** AdamW, zero_grad, backward, clip at 1.0, step. `running` sums the loss per batch; `running / len(train_loader)` is the epoch mean. Real: train_loss 3.99, 2.91, 2.37, 2.03; val_loss 3.23, 2.79, 2.59, 2.52.
- **Why it is here:** Watching both curves is how you detect overfitting: train down and validation up.

### 9. Final score  (`main.py` lines 85–86)

```python
v = evaluate()
print(f"AFTER   val_loss {v:.2f}  perplexity {perplexity(v):.1f}")
```

- **In simple words:** Score again after training.
- **Exactly:** Real: `AFTER val_loss 2.52 perplexity 12.4`, down from 112.2.
- **Why it is here:** The headline result.

### 10. Save the checkpoint  (`main.py` lines 88–94)

```python
# ---- CHECKPOINTING ----------------------------------------------------------
# save_pretrained writes config.json + weights (safetensors); the tokenizer needs saving too.
CKPT_DIR.mkdir(parents=True, exist_ok=True)       # parents=True: create missing folders; exist_ok: no error if present
model.save_pretrained(CKPT_DIR)
tokenizer.save_pretrained(CKPT_DIR)
print("saved checkpoint to", CKPT_DIR)
print("files:", sorted(p.name for p in CKPT_DIR.iterdir()))
```

- **In simple words:** Write the model and tokenizer to disk.
- **Exactly:** `mkdir(parents=True, exist_ok=True)`; `save_pretrained` writes `config.json`, `generation_config.json`, `model.safetensors` (about 330 MB, fp32); the tokenizer writes `tokenizer.json`, `tokenizer_config.json`. Real file list printed.
- **Why it is here:** Without the tokenizer files, loading the tokenizer from the folder fails.

### 11. Reload and compare logits  (`main.py` lines 96–103)

```python
# Reload from disk and compare outputs on the same input.
reloaded = AutoModelForCausalLM.from_pretrained(CKPT_DIR).to(device).eval()
model.eval()
probe = tokenizer("Ticket: How do I sign out? Reply:", return_tensors="pt").to(device)
with torch.no_grad():
    a = model(**probe).logits   # shape: (1, seq_len, 50257)
    b = reloaded(**probe).logits   # shape: (1, seq_len, 50257)
print("max difference between original and reloaded logits:", (a - b).abs().max().item())
```

- **In simple words:** Prove the saved copy behaves exactly like the original.
- **Exactly:** `from_pretrained(CKPT_DIR)` rebuilds the model; `.eval()` on both; logits are `(1, seq_len, 50257)`; `(a - b).abs().max()` is the largest difference. Real: `0.0`.
- **Why it is here:** A save you never reload may be broken; this is the sanity test.

### 12. Peek at the state_dict  (`main.py` lines 105–107)

```python
# The lower-level PyTorch way: only the weights dictionary (you must rebuild the architecture yourself).
state = model.state_dict()                        # dict: layer name -> tensor
print("state_dict has", len(state), "tensors, e.g.", next(iter(state)))
```

- **In simple words:** See the raw dictionary of weights.
- **Exactly:** `model.state_dict()` maps layer name to tensor. Real: 77 tensors, first key `transformer.wte.weight`.
- **Why it is here:** This is the lower-level format; you must rebuild the architecture yourself to use it.

### 13. TRY IT / BREAK IT  (`main.py` lines 109–111)

```python
# TRY IT: set EPOCHS = 12 and watch val_loss start to rise while train_loss keeps falling (overfitting).
# TRY IT: change the 0.2 split to 0.4 - with so little data, how noisy does val_loss become?
# BREAK IT: remove tokenizer.save_pretrained(...) then try AutoTokenizer.from_pretrained(CKPT_DIR) -> OSError.
```

- **In simple words:** Experiments in comments.
- **Exactly:** Removing `tokenizer.save_pretrained` then `AutoTokenizer.from_pretrained(CKPT_DIR)` raises `OSError` (cannot find tokenizer files).
- **Why it is here:** Practices reading a loading error.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `blocks[:-n_val], blocks[-n_val:]` | split off the last `n_val` rows | `[1,2,3,4][:-1]` -> `[1,2,3]` |
| `def evaluate() -> float:` | function with a return type hint (a hint only, not enforced) | |
| `total += loss.item() * batch.shape[0]` | weighted running sum | |
| `path.mkdir(parents=True, exist_ok=True)` | create folders, no error if present | |
| `sorted(p.name for p in dir.iterdir())` | generator expression inside `sorted` | |
| `next(iter(dict))` | first key of a dict | `next(iter({"a":1}))` -> `'a'` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `math.exp` | built-in | e^x; perplexity = exp(loss) | float | float |
| `model.eval()` | torch | dropout off for deterministic scoring | – | model |
| `torch.no_grad()` | torch | no gradient tracking | – | context manager |
| `model.save_pretrained(dir)` | transformers | writes `config.json` + `model.safetensors` | dir | None |
| `tokenizer.save_pretrained(dir)` | transformers | writes tokenizer files | dir | None |
| `from_pretrained(dir)` | transformers | loads from a local folder exactly like a hub name | dir | model |
| `model.state_dict()` | torch | dict of layer name -> weight tensor | – | `OrderedDict` |
| `tensor.abs().max()` | torch | largest absolute entry | – | tensor |

## Execution trace (real run)
- 25 blocks -> 20 train, 5 validation.
- **Before** training: val_loss 4.72, perplexity exp(4.72) = **112.2** (the model is as unsure as choosing among ~112 tokens).
- Epoch 1: train 3.99 / val 3.23; epoch 2: 2.91 / 2.79; epoch 3: 2.37 / 2.59; epoch 4: 2.03 / 2.52.
- **After:** val_loss 2.52, perplexity **12.4**. Notice val_loss improves more slowly than train_loss and the gap widens — the start of overfitting.
- Saved files: `config.json, generation_config.json, model.safetensors, tokenizer.json, tokenizer_config.json`.
- Reloaded model vs original: max logit difference **0.0** -> the checkpoint is faithful.
- `state_dict` has 77 tensors, the first is `transformer.wte.weight` (the token embedding table from module 04).

## Experiments (change ONE thing)
1. `EPOCHS = 12`: train_loss keeps falling, val_loss flattens then rises = overfitting; the fix is fewer epochs / more data / early stopping.
2. 40% validation split: val_loss becomes a less noisy estimate but you train on less.
3. Evaluate with `model.train()` instead of `.eval()`: scores jitter between runs (dropout).

## Debugging practice
Delete the `tokenizer.save_pretrained` line, then `AutoTokenizer.from_pretrained(CKPT_DIR)`:
```
OSError: Can't load tokenizer for '.../distilgpt2-support' ...
```
Read the message: the folder exists but has no tokenizer files. `ls` the folder to confirm. Root cause: a checkpoint is *model + tokenizer + config*.

## Rebuild checklist
- [ ] split blocks into train / validation
- [ ] `evaluate()` with `eval()` + `no_grad()`
- [ ] `perplexity = math.exp(val_loss)`
- [ ] `save_pretrained` (model and tokenizer), reload, compare logits

## The new problem (cliffhanger)
Everything works, but this full fine-tune rewrote all 82 million weights and saved a 330 MB copy. For a 7B model that is 107 GB of memory and a 14 GB file per task. There must be a cheaper way: [step 04, LoRA math with NumPy](../04_lora_math_numpy/WALKTHROUGH.md).

> **Say it aloud (interview box):** Validation loss measures generalisation, and perplexity is the exponential of that loss, roughly how many tokens the model is choosing between. Training loss falling while validation loss rises means overfitting, so I keep the best checkpoint and stop early. I always reload a saved checkpoint and confirm it gives the same loss before trusting it.
