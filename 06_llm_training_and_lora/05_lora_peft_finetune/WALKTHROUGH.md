# Step 05 — LoRA fine-tuning with `peft`

## The story so far
[Step 04](../04_lora_math_numpy/WALKTHROUGH.md) proved the idea on one matrix in NumPy. Real models have dozens of matrices, and nobody wants to write the plumbing by hand.

## The pain point
Hand-wiring low-rank layers into every attention projection of a real model is slow and error-prone. You must also freeze the base, give the optimizer only the new parameters, and later save and load just the adapter.

## The spark
Let `peft` do it: `LoraConfig(r, alpha, target_modules)` plus `get_peft_model` wraps the chosen layers, freezes the rest and leaves about 0.18% of the parameters trainable. Train with the same loop as step 02, save a 0.6 MB adapter, reload it, merge it.

## Goal
Fine-tune distilgpt2 by training only ~0.18% of the parameters, save just the adapter, reload it and merge it.

## Run it
```bash
cd 06_llm_training_and_lora/05_lora_peft_finetune
python main.py       # ~1 min; writes ./adapter_out (0.6 MB)
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| LoRA / PEFT / adapter | parameter-efficient fine-tuning | LoRA is one PEFT method | `peft` is the library; LoRA is the algorithm; the saved folder is the "adapter". |
| `lora_alpha` / scale | alpha/r | Related | The update is multiplied by `lora_alpha / r` (16/8 = 2.0). |
| `target_modules` | which layers get adapters | Yes | Names of layers to wrap; they differ per architecture (GPT-2: `c_attn`; Llama/Qwen: `q_proj`, ...). |
| Merge / `merge_and_unload` | fold adapter | Yes | Writes the update into the base weights and returns a plain model. |
| `disable_adapter()` / un-adapted base | switch LoRA off | Same effect | A context manager that makes the model behave like the base temporarily. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–16)

```python
"""
GOAL:  Fine-tune distilgpt2 with LoRA using the `peft` library: freeze the base model, train tiny adapters,
       save only the adapter (a few hundred KB), reload it, and merge it for fast inference.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_lora/05_lora_peft_finetune/main.py
EXPECTED OUTPUT (shortened):
    trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.1797
    epoch 1 loss 4.57 ... epoch 25 loss 1.40
    base      : Ticket: How do I turn on dark mode? Reply: ...generic...
    with LoRA : Ticket: How do I turn on dark mode? Reply: Open Settings, tap Settings, then choose Dark Mode.
    adapter folder size: 0.60 MB  vs full model ~330 MB
    reloaded adapter output identical to trained adapter: True
    merged model output identical: True
NEEDS: torch, transformers, peft.  ~30-60 s.
"""
```

- **In simple words:** Summary of the step and expected output.
- **Exactly:** Not executed; mentions 147,456 trainable params and the 0.60 MB adapter.
- **Why it is here:** Tells you what a correct run looks like.

### 2. Imports and settings  (`main.py` lines 18–31)

```python
import sys
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model   # peft = Parameter-Efficient Fine-Tuning library
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import get_device, make_blocks, read_lines

MODEL_NAME = "distilgpt2"
ADAPTER_DIR = Path(__file__).resolve().parent / "adapter_out"
EPOCHS, BATCH_SIZE, LR = 25, 4, 1e-3          # LoRA tolerates (and wants) a HIGHER lr than full fine-tuning
```

- **In simple words:** Load libraries, helpers and training settings.
- **Exactly:** `LoraConfig`, `PeftModel`, `get_peft_model` from `peft`. `EPOCHS=25`, `BATCH_SIZE=4`, `LR=1e-3` (higher than full fine-tuning because only tiny matrices learn).
- **Why it is here:** A full-fine-tune learning rate (about 5e-5) would barely move LoRA in 25 epochs.

### 3. Seed, device, tokenizer, base model  (`main.py` lines 33–36)

```python
torch.manual_seed(0)
device = get_device()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)      # keep an un-adapted copy on CPU to compare later
```

- **In simple words:** Load a fresh distilgpt2 on the CPU.
- **Exactly:** `base` stays on CPU, not moved to the device before wrapping.
- **Why it is here:** Keeps the original weights to compare later.

### 4. Describe and attach the LoRA adapters  (`main.py` lines 38–49)

```python
# ---- 1. Describe the adapter ---------------------------------------------------
config = LoraConfig(
    r=8,                          # rank of the update matrices (bigger = more capacity, more params)
    lora_alpha=16,                # scaling: the update is multiplied by lora_alpha / r  (here 2.0)
    lora_dropout=0.05,            # dropout applied inside the adapter path; regularisation
    target_modules=["c_attn"],    # WHICH layers get adapters: GPT-2's fused query/key/value projection
    fan_in_fan_out=True,          # GPT-2 stores this layer transposed (Conv1D); peft must know
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, config)  # wraps the model: freezes base weights, inserts trainable A and B
model.print_trainable_parameters()    # the headline number to quote in an interview
model.to(device)
```

- **In simple words:** Say where and how big the adapters should be, then wrap the model with them.
- **Exactly:** `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, `target_modules=["c_attn"]` (GPT-2's fused q/k/v, 768 -> 2304), `fan_in_fan_out=True` because GPT-2 uses `Conv1D` with transposed weights. `get_peft_model` freezes everything and inserts A and B. Real: `trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.1797` (6 layers x (768x8 + 8x2304) = 147,456).
- **Why it is here:** Wrong `target_modules` raises `ValueError`; wrong `fan_in_fan_out` gives a warning and wrong maths.

### 5. Inspect what was inserted  (`main.py` lines 51–57)

```python
# peek at what was inserted
for name, param in model.named_parameters():
    if "h.0" in name and "lora" in name:   # first adapter matrix of transformer block 0
        print("  trainable:", name, tuple(param.shape), "requires_grad =", param.requires_grad)
        break
frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"  frozen parameters: {frozen:,}")
```

- **In simple words:** Look at one adapter matrix and count frozen weights.
- **Exactly:** Real: `base_model.model.transformer.h.0.attn.c_attn.lora_A.default.weight (8, 768) requires_grad = True`; frozen parameters 81,912,576.
- **Why it is here:** Proves the base model is frozen and only adapters learn.

### 6. Data and an optimizer for trainable params only  (`main.py` lines 59–62)

```python
# ---- 2. Train (same loop as step 02, but the optimizer only sees trainable params) ----
blocks = make_blocks(tokenizer, read_lines(), 48)
loader = DataLoader(TensorDataset(blocks), batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)  # filter: trainable only
```

- **In simple words:** Same data pipeline as step 02, but AdamW only gets the adapter weights.
- **Exactly:** `blocks` is `(25, 48)`; the list comprehension keeps parameters with `requires_grad`; lr 1e-3.
- **Why it is here:** Giving the optimizer frozen weights wastes memory for its state.

### 7. Training loop  (`main.py` lines 64–74)

```python
model.train()
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for (batch,) in loader:
        batch = batch.to(device)   # shape: (BATCH_SIZE, BLOCK_SIZE)
        loss = model(input_ids=batch, labels=batch).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"epoch {epoch} loss {total / len(loader):.3f}")
```

- **In simple words:** Same loop as before, but without gradient clipping.
- **Exactly:** Per-epoch mean loss printed. Real: epoch 1 4.574, epoch 10 2.158, epoch 25 1.404. Backward only creates gradients for the adapters.
- **Why it is here:** Shows LoRA learns the style with 0.18% of the weights.

### 8. Generation helper  (`main.py` lines 77–83)

```python
def generate(m, prompt: str) -> str:
    """Greedy-generate 30 tokens from model m."""
    m.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(next(m.parameters()).device)
    with torch.no_grad():
        out = m.generate(**enc, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)
```

- **In simple words:** A function to finish a prompt with greedy decoding.
- **Exactly:** `m.eval()`; device taken from the model's first parameter; 30 new tokens, no sampling; `tokenizer.decode(out[0], skip_special_tokens=True)`.
- **Why it is here:** Works for the peft model, the reloaded one and the merged one.

### 9. Compare base vs LoRA output  (`main.py` lines 86–91)

```python
PROMPT = "Ticket: How do I turn on dark mode? Reply:"
# `with model.disable_adapter():` temporarily switches LoRA off -> you get the original behaviour.
with model.disable_adapter():
    print("\nbase      :", generate(model, PROMPT))
trained_text = generate(model, PROMPT)
print("with LoRA :", trained_text)
```

- **In simple words:** Switch adapters off for the "before" text, on for the "after" text.
- **Exactly:** `with model.disable_adapter()` temporarily bypasses A and B. Real base: `... I turn on dark mode? Reply: I turn on dark mode? ...`; with LoRA: `Open Settings, tap Settings, then choose Dark Mode.`
- **Why it is here:** Same-model before/after without loading two copies.

### 10. Save only the adapter  (`main.py` lines 93–96)

```python
# ---- 3. Save ONLY the adapter ----------------------------------------------------
model.save_pretrained(ADAPTER_DIR)      # writes adapter_config.json + adapter_model.safetensors
size_mb = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir()) / 1e6
print(f"\nadapter folder size: {size_mb:.2f} MB  vs full model ~330 MB")
```

- **In simple words:** Write a tiny file instead of a whole model.
- **Exactly:** `save_pretrained` on a PeftModel writes `adapter_config.json` and `adapter_model.safetensors`. Real: `0.60 MB` vs about 330 MB.
- **Why it is here:** Small files are easy to version and ship.

### 11. Reload adapter onto a fresh base  (`main.py` lines 98–101)

```python
# ---- 4. Reload: fresh base model + saved adapter ------------------------------------
fresh_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
reloaded = PeftModel.from_pretrained(fresh_base, ADAPTER_DIR).to(device)
print("reloaded adapter output identical to trained adapter:", generate(reloaded, PROMPT) == trained_text)
```

- **In simple words:** Rebuild the model from base + saved adapter and check it behaves the same.
- **Exactly:** `PeftModel.from_pretrained(fresh_base, ADAPTER_DIR)`. Real: `True`.
- **Why it is here:** A save you never reload may be broken.

### 12. Merge for deployment  (`main.py` lines 103–106)

```python
# ---- 5. Merge for deployment: W + (alpha/r) B A is folded into W; no peft needed at inference ----
merged = reloaded.merge_and_unload()    # returns a plain transformers model
print("merged model output identical:", generate(merged, PROMPT) == trained_text)
print("merged model type:", type(merged).__name__)
```

- **In simple words:** Fold the adapter into the base weights.
- **Exactly:** `merge_and_unload()` computes `W + (alpha/r) B A` per adapted layer and returns a plain `GPT2LMHeadModel`. Real: `merged model output identical: True`.
- **Why it is here:** No peft needed at serving, no extra latency.

### 13. TRY IT / BREAK IT  (`main.py` lines 108–112)

```python
# TRY IT: set r=2, then r=32. Compare 'trainable%' and the final loss.
# TRY IT: set target_modules=["c_attn", "c_proj", "c_fc"] to adapt more layers; what happens to trainable params?
# TRY IT: set LR = 3e-6 (typical full-fine-tune size). LoRA barely learns - why?
# BREAK IT: remove fan_in_fan_out=True and read the warning peft prints about Conv1D.
# BREAK IT: set target_modules=["q_proj"] and read the ValueError: GPT-2 has no module with that name.
```

- **In simple words:** Experiments in comments.
- **Exactly:** `target_modules=["q_proj"]` raises `ValueError: Target modules ... not found in the base model`.
- **Why it is here:** Shows that module names are architecture-specific.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `LoraConfig(r=8, ...)` | a settings object (a class with keyword arguments) | `cfg.r` |
| `[p for p in model.parameters() if p.requires_grad]` | filter: keep only trainable tensors | |
| `with model.disable_adapter():` | temporarily turn LoRA off inside the block | |
| `type(x).__name__` | class name as a string | `type(3).__name__` -> `'int'` |
| `sum(p.numel() for p in ...)` | generator expression: add up counts without building a list | |
| `next(m.parameters()).device` | where the model lives | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `LoraConfig` | peft | declares the adapter | `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `fan_in_fan_out`, `task_type` | config |
| `get_peft_model(model, config)` | peft | freezes base weights, inserts trainable A/B | model, config | `PeftModel` |
| `model.print_trainable_parameters()` | peft | prints trainable vs total | – | None |
| `model.save_pretrained(dir)` | peft | saves ONLY adapter weights + config | dir | None |
| `PeftModel.from_pretrained(base, dir)` | peft | attaches a saved adapter to a fresh base | base model, dir | `PeftModel` |
| `merge_and_unload()` | peft | folds `(alpha/r)·B·A` into W, removes adapter layers | – | plain transformers model |
| `disable_adapter()` | peft | context manager for A/B comparisons | – | context |

Parameter meaning: **r** = rank (capacity); **lora_alpha** = scaling numerator (update scaled by `alpha / r`, here 2.0); **target_modules** = which layers to adapt (GPT-2: `c_attn`; Llama/Qwen: `q_proj`, `k_proj`, `v_proj`, `o_proj`, ...).

## Execution trace (real run)
- `trainable params: 147,456 || all params: 82,060,032 || trainable%: 0.1797`. Check: 6 blocks x (`768*8 + 8*2304`) = 6 x 24,576 = 147,456. (`c_attn` maps 768 -> 2304 because Q, K and V are fused.)
- Frozen parameters: 81,912,576 (the original distilgpt2).
- Training 25 epochs at lr 1e-3: loss 4.574 (epoch 1) -> 2.158 (10) -> 1.606 (20) -> **1.404** (25).
- Prompt `Ticket: How do I turn on dark mode? Reply:`
  - base (adapter disabled): `I turn on dark mode? Reply: I turn on dark mode? ...` (repeats)
  - with LoRA: `Open Settings, tap Settings, then choose Dark Mode.` (learned the style, got the wording a bit muddled — expected from a tiny corpus)
- Adapter on disk: **0.60 MB** vs ~330 MB for the full model.
- Reloaded adapter output identical: `True`. After `merge_and_unload()` identical: `True`, and the type is `GPT2LMHeadModel` again.

## Experiments (change ONE thing)
Measured with `solutions/lora_rank_sweep.py` (10 epochs):
| r | trainable params | final loss |
|---|---|---|
| 1 | 18,432 | 2.902 |
| 8 | 147,456 | 2.145 |
| 32 | 589,824 | 1.734 |
1. Higher rank = more capacity = lower training loss, linear growth in parameters. Lower training loss is not automatically better generalisation.
2. `LR = 3e-6` (a full-fine-tune rate): LoRA barely moves because only tiny, zero-initialised matrices are trained; LoRA usually wants 10-100x larger rates.
3. Adding `c_proj`, `c_fc` to `target_modules` increases trainable parameters.

## Debugging practice
`target_modules=["q_proj"]`:
```
ValueError: Target modules {'q_proj'} not found in the base model ...
```
Fix by listing real layer names: `print([n for n, _ in base.named_modules()][:20])`. Layer names differ per architecture — the single most common peft error.

## Rebuild checklist
- [ ] `LoraConfig(r, lora_alpha, target_modules, task_type)`
- [ ] `get_peft_model(base, config)` + `print_trainable_parameters()`
- [ ] optimizer over trainable params only; same loop as step 02
- [ ] `save_pretrained`, reload with `PeftModel.from_pretrained`, `merge_and_unload()`

## The new problem (cliffhanger)
The adapter trains on a laptop. For a 7B model the frozen base weights still take 14 GB in float16, more than a small GPU holds. Compress the frozen part: [step 06, QLoRA](../06_qlora_reference/WALKTHROUGH.md).

> **Say it aloud (interview box):** With peft I wrap the attention layers in LoRA adapters, freeze the base model, and train only about 0.2 percent of the parameters with the same loop as before. The result is a tiny adapter file that I can load on top of the base model or merge into it. One base model can then serve many tasks, each with its own small adapter.
