# Step 02 — The training loop

## The story so far
[Step 01](../01_dataset_preparation/WALKTHROUGH.md) produced batches. A batch alone changes nothing; weights only move if something computes how wrong the model was and nudges them.

## The pain point
A model straight from the hub is a generalist. It has never seen your data, so its loss on your text is high, and no amount of prompting changes the weights. Worse, an unclipped gradient can blow one update up and undo hours of progress.

## The spark
Four beats, repeated: forward (get the loss), `zero_grad`, backward (gradients), `clip_grad_norm_` then `optimizer.step()`. The loss falls as the weights adapt.

## Goal
Write the four-beat loop yourself — forward, loss, backward, optimizer step — and see loss fall on a real model (distilgpt2, 82M parameters).

## Run it
```bash
cd 06_llm_training_and_lora/02_training_loop
python main.py        # ~40 s on an M1 (uses the Apple GPU, "mps")
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Forward pass / inference / prediction | `model(x)`, forward | Forward pass is the computation; inference is using it without learning | In training the forward pass is followed by backward and an optimizer step. |
| Loss / cost / objective | cross-entropy, NLL, negative log-likelihood | For LMs, essentially yes | The number training tries to push down: average -log(probability of the correct next token). |
| Gradient clipping / clip_grad_norm_ | norm clipping | Yes | Rescales all gradients together when their total length exceeds `MAX_GRAD_NORM`. |
| Epoch / step / iteration / batch | pass, update | Different | A step is one optimizer update on one batch; an epoch is a full pass over the data (here 7 steps). |
| AdamW / Adam / SGD | optimizers | Related | AdamW = Adam with decoupled weight decay; SGD has no per-weight adaptive step size. |
| Learning rate / step size | lr | Yes | Multiplier on how far each update moves the weights. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–14)

```python
"""
GOAL:  Write a minimal causal-LM training loop by hand (forward -> loss -> backward -> optimizer step)
       and watch the loss fall while distilgpt2 learns our little customer-support style.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_lora/02_training_loop/main.py
EXPECTED OUTPUT (shortened; exact numbers vary a little):
    device: mps
    epoch 1 step  1 loss 5.1 grad_norm 20.6
    epoch 6 step 42 loss 1.5 grad_norm 20.0
    BEFORE training: Ticket: How do I reset my password? Reply: ...(generic text)
    AFTER  training: Ticket: How do I reset my password? Reply: Open Settings, tap Account ...
NEEDS: torch, transformers.  First run downloads distilgpt2 (~330 MB).  Takes ~20-40 s.
"""
```

- **In simple words:** The note at the top saying what the file does and what to expect.
- **Exactly:** A module docstring; not executed. Says distilgpt2 is trained on the tiny support corpus for 6 epochs and prints loss.
- **Why it is here:** It tells you what a correct run looks like so you can spot a broken one.

### 2. Imports and helper import  (`main.py` lines 16–24)

```python
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so we can `import common` from the module folder
from common import get_device, make_blocks, read_lines          # helpers built by hand in step 01
```

- **In simple words:** Load the libraries, then borrow three small helper functions we wrote in step 01.
- **Exactly:** `sys.path.insert(0, <module folder>)` lets Python find `common.py` one level up; `get_device` picks cuda/mps/cpu, `make_blocks` builds the `(n_blocks, 48)` tensor, `read_lines` reads the corpus.
- **Why it is here:** Without the path insert, `import common` fails with `ModuleNotFoundError`.

### 3. Hyper-parameters  (`main.py` lines 26–33)

```python
# ---- hyper-parameters (module 05 explains each one) ------------------------
MODEL_NAME = "distilgpt2"
BLOCK_SIZE = 48
BATCH_SIZE = 4          # blocks per optimizer step
EPOCHS = 6              # full passes over the data
LEARNING_RATE = 5e-5    # step size of the optimizer; the single most sensitive knob
MAX_GRAD_NORM = 1.0     # gradient clipping threshold (prevents one huge update from wrecking the model)
SEED = 0
```

- **In simple words:** The settings that control training, kept in one place at the top.
- **Exactly:** `BLOCK_SIZE=48`, `BATCH_SIZE=4`, `EPOCHS=6`, `LEARNING_RATE=5e-5`, `MAX_GRAD_NORM=1.0`, `SEED=0`. 25 blocks / 4 = 7 steps per epoch (last batch has 1 row), so 6 x 7 = 42 steps total.
- **Why it is here:** Named constants make experiments (TRY IT) a one-line change.

### 4. Seed and pick device  (`main.py` lines 35–37)

```python
torch.manual_seed(SEED)                       # makes shuffling/dropout repeatable
device = get_device()
print("device:", device)
```

- **In simple words:** Make randomness repeatable and choose the fastest chip.
- **Exactly:** `torch.manual_seed(0)` fixes shuffling and dropout masks. Real run printed `device: mps` (Apple GPU).
- **Why it is here:** Without a seed your loss curve differs every run, so you cannot compare experiments.

### 5. Load tokenizer and model  (`main.py` lines 39–40)

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)   # .to(device) moves weights to GPU/CPU memory
```

- **In simple words:** Download (or load from cache) the pretrained distilgpt2 and put it on the chip.
- **Exactly:** `AutoModelForCausalLM.from_pretrained("distilgpt2")` returns `GPT2LMHeadModel` with about 82M parameters in fp32 (weights are updated in the whole network). `.to(device)` copies the weights into device memory.
- **Why it is here:** Training starts from pretrained weights, not random ones, which is why 42 steps already show a big change.

### 6. Build blocks and the DataLoader  (`main.py` lines 42–43)

```python
blocks = make_blocks(tokenizer, read_lines(), BLOCK_SIZE)   # shape: (n_blocks, BLOCK_SIZE)
loader = DataLoader(TensorDataset(blocks), batch_size=BATCH_SIZE, shuffle=True)
```

- **In simple words:** Cut the corpus into 48-token chunks and prepare a batch dispenser.
- **Exactly:** `blocks` has shape `(25, 48)`; `TensorDataset(blocks)` yields 1-tuples; `DataLoader(batch_size=4, shuffle=True)` gives batches `(4, 48)`.
- **Why it is here:** Batches give the optimizer a less noisy direction than single rows.

### 7. Create the optimizer  (`main.py` lines 45–46)

```python
# AdamW = Adam optimizer with proper weight decay; model.parameters() = every weight it may update.
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

- **In simple words:** The optimizer is the part that actually changes the weights using the gradients.
- **Exactly:** `torch.optim.AdamW(model.parameters(), lr=5e-5)` keeps two extra numbers per weight (running mean and variance of gradients) and applies weight decay (default 0.01) separately from the gradient update.
- **Why it is here:** Without an optimizer, `backward()` computes gradients but nothing learns.

### 8. A greedy `sample` helper  (`main.py` lines 49–57)

```python
def sample(prompt: str) -> str:
    """Generate a short continuation (greedy = always pick the most likely token)."""
    model.eval()                                            # turns off dropout
    enc = tokenizer(prompt, return_tensors="pt").to(device)   # input_ids, attention_mask: shape (1, prompt_len)
    with torch.no_grad():                                   # no gradients needed -> less memory, faster
        out = model.generate(**enc, max_new_tokens=30, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    model.train()                                           # back to training mode
    return tokenizer.decode(out[0], skip_special_tokens=True)   # out shape: (1, prompt_len + 30)
```

- **In simple words:** A function that finishes a prompt so we can look at before and after.
- **Exactly:** `model.eval()` turns dropout off; `tokenizer(prompt, return_tensors="pt")` returns `input_ids` and `attention_mask` of shape `(1, prompt_len)`; `torch.no_grad()` skips gradient bookkeeping; `generate(max_new_tokens=30, do_sample=False)` picks the arg-max token each step; `pad_token_id=eos` silences a warning (GPT-2 has no pad). Output shape `(1, prompt_len + 30)`. `model.train()` switches dropout back on.
- **Why it is here:** Forgetting `model.train()` afterwards would silently train with dropout off; forgetting `no_grad` wastes memory.

### 9. Sample before training  (`main.py` lines 60–61)

```python
PROMPT = "Ticket: How do I reset my password? Reply:"
print("BEFORE training:", sample(PROMPT), "\n")
```

- **In simple words:** Take a photo of the model before it learns anything.
- **Exactly:** Real output: `I reset my password. Reply: I reset my password. Reply: ...` (generic repetition).
- **Why it is here:** Gives the "before" half of the before/after proof.

### 10. Epoch loop and batch unpack  (`main.py` lines 63–67)

```python
# ---- the training loop ------------------------------------------------------
model.train()                                               # dropout ON
step = 0
for epoch in range(1, EPOCHS + 1):
    for (batch,) in loader:                                 # TensorDataset yields 1-tuples -> unpack with (batch,)
```

- **In simple words:** Go through the data 6 times, one batch at a time.
- **Exactly:** `model.train()` enables dropout. `for (batch,) in loader` unpacks the 1-tuple a `TensorDataset` yields.
- **Why it is here:** Without the parentheses you would get a list, not a tensor, and the next line fails.

### 11. Forward pass and loss  (`main.py` lines 68–71)

```python
        batch = batch.to(device)                            # shape: (BATCH_SIZE, BLOCK_SIZE)
        # 1. FORWARD: labels=batch makes HF shift by one and compute cross-entropy for us.
        out = model(input_ids=batch, labels=batch)          # out.logits shape: (BATCH_SIZE, BLOCK_SIZE, 50257)
        loss = out.loss                                     # a 0-d tensor: average loss per token
```

- **In simple words:** Show the model a batch and measure how wrong it is.
- **Exactly:** `batch.to(device)` shape `(4, 48)`. `labels=batch` makes the model shift by one and compute cross-entropy over the vocab of 50257; `out.logits` is `(4, 48, 50257)`; `out.loss` is a 0-d tensor, mean over tokens. Real first loss: 5.088.
- **Why it is here:** This is the training signal; no loss, nothing to minimise.

### 12. Clear old gradients and backpropagate  (`main.py` lines 72–74)

```python
        # 2. BACKWARD: compute d(loss)/d(weight) for every weight.
        optimizer.zero_grad()                               # clear gradients from the previous step (they accumulate!)
        loss.backward()
```

- **In simple words:** Wipe the previous step's notes, then work out how each weight should change.
- **Exactly:** `zero_grad()` sets `.grad` to zero (otherwise gradients add up); `loss.backward()` fills `param.grad` for all 82M weights by the chain rule.
- **Why it is here:** Dropping `zero_grad()` makes step k use the sum of all earlier gradients and the loss misbehaves.

### 13. Clip gradients and update weights  (`main.py` lines 75–77)

```python
        # 3. CLIP then STEP: limit gradient size, then nudge the weights downhill.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
```

- **In simple words:** Cap how big a step can be, then take the step.
- **Exactly:** `clip_grad_norm_` computes the global L2 norm of all gradients, multiplies them by `1.0 / norm` if norm > 1.0 and returns the norm before clipping (real step 1: 20.57, so every update was scaled by about 1/20). `optimizer.step()` applies the AdamW update.
- **Why it is here:** One huge gradient could throw the model far from its good pretrained weights.

### 14. Count steps and log  (`main.py` lines 78–79)

```python
        step += 1
        print(f"epoch {epoch} step {step:2d} loss {loss.item():.3f} grad_norm {grad_norm:.2f}")
```

- **In simple words:** Keep a counter and print one line per step.
- **Exactly:** Real log: step 1 loss 5.088, step 42 loss 1.467. `loss.item()` converts the 0-d tensor to a Python float.
- **Why it is here:** You need the curve to know training is working.

### 15. Sample after training  (`main.py` lines 81–81)

```python
print("\nAFTER  training:", sample(PROMPT))
```

- **In simple words:** Take the "after" photo.
- **Exactly:** Real: `Open Settings, tap Account, then choose Reset Password.` The model has learned the support style.
- **Why it is here:** A falling loss number is abstract; the changed text shows it.

### 16. TRY IT / BREAK IT  (`main.py` lines 83–87)

```python
# TRY IT: set LEARNING_RATE = 5e-4 (10x) - does loss fall faster, or become unstable?
# TRY IT: set LEARNING_RATE = 5e-7 - how little does the loss move in 6 epochs?
# TRY IT: set EPOCHS = 20 and watch the model memorise the corpus (loss near 0 = overfitting).
# BREAK IT: delete the optimizer.zero_grad() line - gradients pile up; compare the loss curve.
# BREAK IT: write `out = model(input_ids=batch)` (no labels) and then `out.loss.backward()` -> AttributeError on None.
```

- **In simple words:** Experiments in comments.
- **Exactly:** Without labels, `out.loss` is `None`, so `.backward()` raises `AttributeError: 'NoneType' object has no attribute 'backward'`.
- **Why it is here:** Reading the error teaches you where the loss comes from.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `for (batch,) in loader:` | unpack a 1-item tuple | `for (x,) in [(1,),(2,)]: print(x)` |
| `with torch.no_grad():` | context manager: gradients off inside the block | see `sample()` |
| `sys.path.insert(0, folder)` | let `import` also look in `folder` | used to `import common` |
| `loss.item()` | turn a 1-element tensor into a normal Python float | `torch.tensor(2.5).item()` -> `2.5` |
| `f"{x:.3f}"`, `f"{n:2d}"` | format floats / pad ints | `f"{3.14159:.3f}"` -> `3.142` |
| nested loops | epochs outside, batches inside | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `AutoModelForCausalLM.from_pretrained` | transformers | loads weights + architecture (as in module 04) | model name | model |
| `model.to(device)` | torch | move weights to cpu / mps / cuda | device | the model |
| `torch.optim.AdamW(params, lr)` | torch | the optimizer: updates weights from gradients | `lr`, `weight_decay` | optimizer |
| `model(input_ids=..., labels=...)` | transformers | forward pass; with `labels` also returns `.loss` (shifted cross-entropy) | `input_ids`, `labels`, `attention_mask` | output object with `.loss`, `.logits` |
| `optimizer.zero_grad()` | torch | clears old gradients (they accumulate by default) | – | None |
| `loss.backward()` | torch | autograd: computes d loss / d weight into `.grad` | – | None |
| `clip_grad_norm_(params, max_norm)` | torch | rescales gradients if their total norm exceeds `max_norm` | `max_norm` | the norm *before* clipping |
| `optimizer.step()` | torch | applies the update `w -= lr * f(grad)` | – | None |
| `model.train()` / `model.eval()` | torch | dropout on / off | – | model |
| `model.generate(...)` | transformers | produce tokens autoregressively (module 05 params) | `max_new_tokens`, `do_sample` | tensor of IDs |

## Execution trace (real run)
- `device: mps`, 25 blocks, batch size 4 -> 7 steps per epoch (the last batch holds 1 block), 6 epochs = 42 steps.
- Step 1: loss **5.088**, grad_norm 20.57. Step 14: 2.004. Step 28: 1.886. Step 42: 1.467. The trend is down; single steps are noisy because each batch is only 4 blocks.
- `grad_norm` is printed *before* clipping and is ~10-20, so with `MAX_GRAD_NORM = 1.0` every step is clipped (scaled down). That is OK here; in a long, healthy run you expect it to be mostly below the threshold, and spikes are what clipping is for.
- Before training, the prompt `Ticket: How do I reset my password? Reply:` continues with a repetitive loop (`I reset my password. Reply: I reset ...`).
- After training: `Reply: Open Settings, tap Account, then choose Reset Password.` — it learned the answer style (and memorised the nearby line).

## Experiments (change ONE thing)
Measured with `solutions/lr_sweep.py` (3 epochs each, same seed, mean loss per epoch):
| learning rate | epoch 1 | epoch 2 | epoch 3 |
|---|---|---|---|
| 5e-7 | 4.71 | 4.66 | 4.64 |
| 5e-5 (default) | 3.81 | 2.66 | 2.14 |
| 5e-4 | 3.23 | 1.59 | 0.99 |
1. Too small (5e-7): the model hardly moves. Larger (5e-4): faster here because the model is small and the data tiny, but on real data a high rate can wipe out pretrained knowledge ("catastrophic forgetting") or diverge to `nan`.
2. `EPOCHS = 20`: loss keeps falling towards memorisation. Step 03 shows how to detect that with validation data.
3. `BATCH_SIZE = 1` vs `8`: noisier vs smoother loss curve; fewer vs more weight updates per epoch.

## Debugging practice
1. Remove `optimizer.zero_grad()`: loss still prints but behaves erratically — a *silent* bug (no traceback!). Lesson: some bugs only show in behaviour, so always watch the loss curve.
2. Call `model(input_ids=batch)` without labels then `out.loss.backward()`:
```
AttributeError: 'NoneType' object has no attribute 'backward'
```
`out.loss` is `None` because no labels were given, so there was nothing to compute.

## Rebuild checklist
- [ ] load tokenizer + model, move model to device
- [ ] create DataLoader of token blocks
- [ ] `AdamW(model.parameters(), lr=...)`
- [ ] loop: forward with labels -> `zero_grad` -> `backward` -> clip -> `step`
- [ ] print `loss.item()` each step

## The new problem (cliffhanger)
The loss on the training text falls, but falling training loss can mean memorising. Are we learning or cheating? Next: [step 03, evaluation, perplexity and checkpoints](../03_eval_perplexity_checkpoints/WALKTHROUGH.md).

> **Say it aloud (interview box):** Training is a loop of four beats: forward pass for the cross-entropy loss, backward pass for gradients, clip the gradient norm, and let AdamW update the weights. The loss on next-token prediction falls as the model adapts to my data. The learning rate and gradient clipping are the two settings most likely to ruin a run.
