# Step 09 - Pretraining vs fine-tuning vs inference (The Life Cycle)

## The story so far
[Step 08](../08_generation_loop/WALKTHROUGH.md) made generation fast with the KV cache. Now the model can read ([07](../07_forward_pass_logits/WALKTHROUGH.md)) and write. But we have taken one thing on faith: where did its weights come from, and can we change them?

## The pain point
A model that has never seen language predicts at random (loss near 11 for a 50,257-token vocabulary). A pretrained model knows English but not YOUR style or task. And running it (inference) and teaching it (training) are very different jobs with very different costs. Mixing them up is a common source of wrong cost estimates.

## The spark
Three stages, one loss. **Pretraining** learns general language from a huge corpus. **Fine-tuning** continues the same training on a small dataset with a small learning rate. **Inference** freezes the weights and only runs forward passes. We measure each stage with the loss from step 07.

## Goal
Use loss numbers to feel the three stages of an LLM's life, and see a complete training loop run.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/09_pretrain_vs_finetune_vs_inference/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Pretraining | foundation-model training, self-supervised pretraining, from-scratch training | Yes | Learn next-token prediction on trillions of tokens; produces the base model |
| Fine-tuning | continued training, adaptation | Umbrella term | Further training on a small dataset (full fine-tune, LoRA, QLoRA ...) |
| SFT / instruction tuning / supervised fine-tuning | chat tuning | Close | SFT on (instruction, answer) pairs is one kind of fine-tuning; it turns a base model into an assistant |
| RLHF / DPO / preference tuning | alignment tuning | Different | Learn from human preferences rather than next-token targets (module 17) |
| Inference | serving, prediction, generation, forward-only | Yes | Weights frozen; only forward passes |
| Base vs instruct vs chat model | checkpoint types | Different | Base = pretrained only; instruct/chat = additionally SFT (and often preference) tuned |
| Loss masking (`-100`) | ignore_index, label masking | Yes | Positions with label -100 are skipped by the cross-entropy loss |
| `from_pretrained` vs `from_config` | trained vs random weights | No | Loads learned weights vs builds the same architecture with random numbers |
| Overfitting vs memorising | train loss near 0 | Related | Memorising 4 sentences is extreme overfitting; check on held-out data |

## Code, step by step

### 1. Module docstring: the three life stages  (`main.py` lines 1–19)

```python
"""
GOAL:   Feel the difference between the three life stages of an LLM using loss numbers:
        (1) random weights  (2) pretrained weights  (3) fine-tuned weights, plus inference.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/09_pretrain_vs_finetune_vs_inference/main.py
EXPECTED OUTPUT (short): loss ~10.9 for random weights (= ln 50257), ~4-5 pretrained,
        then dropping below 1 on the tiny fine-tuning text (it memorises 4 sentences); generation starts to copy its style.
NEEDS:  torch + transformers (distilgpt2 cached; loaded in step 06). Takes ~20-40 s on an M1 CPU.

The three stages
  PRETRAINING : learn language from scratch on a huge corpus (weeks, thousands of GPUs, millions of $).
                Objective: predict the next token.  You will NOT do this; you USE the result.
  FINE-TUNING : continue training a pretrained model on a small, specific dataset (minutes-hours)
                so it picks up a style/task/domain.  Same loss, same loop, tiny dataset, small learning rate.
  INFERENCE   : weights frozen; just run forward passes to produce text.  What users pay latency for.
Full-size fine-tuning (LoRA, datasets) is ../../06_llm_training_and_lora/; the knobs are ../../05_llm_parameters/.
This step is a 20-second taste.
"""
```

- **In simple words:** Describes pretraining (learn language), fine-tuning (learn a style or task) and inference (just use it).
- **Exactly:** Docstring only. Promised: loss about 10.9 for random weights (= ln 50,257), 4 to 6 pretrained, below 1 after fine-tuning on four sentences.
- **Why it is here:** A clear picture of what you will do (fine-tune and serve) versus what labs do (pretrain).

### 2. Imports  (`main.py` lines 21–23)

```python
import math
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
```

- **In simple words:** Maths, PyTorch and the loaders (now including `AutoConfig` for building a random model).
- **Exactly:** `AutoConfig` gives the architecture only; `AutoModelForCausalLM.from_config` builds it with random weights.
- **Why it is here:** Random weights give the "knows nothing" baseline.

### 3. Seed and tokenizer with a pad token  (`main.py` lines 25–27)

```python
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
```

- **In simple words:** Make randomness repeatable and let the tokenizer pad short sentences.
- **Exactly:** `torch.manual_seed(0)`. GPT-2 has no pad token, so `tok.pad_token = tok.eos_token` reuses id 50256 (the attention mask still marks padding positions).
- **Why it is here:** Batching sentences of different length needs padding.

### 4. A tiny pirate-style dataset  (`main.py` lines 29–35)

```python
# A tiny "dataset": a handful of sentences in one consistent (pirate) style.
texts = [
    "Arrr, ye scurvy dog, hand over the treasure or walk the plank!",
    "Avast, matey! The sea be calm and the rum be plentiful.",
    "Hoist the sails, me hearties, for fortune awaits beyond the horizon!",
    "Shiver me timbers, that be the finest ship on the seven seas.",
]
```

- **In simple words:** Four sentences that all sound like pirates; this is our "fine-tuning data".
- **Exactly:** `texts` is a `list[str]` of 4 items with a consistent style.
- **Why it is here:** Fine-tuning teaches style or task from a small, focused set of examples.

### 5. Tokenise as a padded batch and mask the padding in the labels  (`main.py` lines 36–39)

```python
batch = tok(texts, return_tensors="pt", padding=True)   # input_ids shape: (4, longest_len) = (batch, seq_len)
# Labels = the input ids, but padding positions are set to -100 which means "ignore in the loss".
labels = batch["input_ids"].clone()                      # shape: (4, longest_len)
labels[batch["attention_mask"] == 0] = -100          # attention_mask shape: (4, longest_len); 0 = padding
```

- **In simple words:** Turn the sentences into a rectangle of ids and tell the loss to ignore padding positions.
- **Exactly:** `tok(texts, return_tensors="pt", padding=True)` gives `input_ids` `(4, longest_len)` and `attention_mask` of the same shape. `labels = input_ids.clone()`, and `labels[attention_mask == 0] = -100` marks padded positions; PyTorch's cross-entropy skips label `-100`. The model shifts labels by one internally.
- **Why it is here:** Without masking, the loss would reward predicting padding tokens (the last BREAK IT comment).

### 6. Evaluation loss helper  (`main.py` lines 41–45)

```python
def loss_of(model):
    """Average next-token cross-entropy of `model` on our texts (no gradient, evaluation only)."""
    model.eval()
    with torch.no_grad():
        return model(**batch, labels=labels).loss.item()   # loss: scalar (logits inside are (4, longest_len, 50257))
```

- **In simple words:** A function that returns the average next-token loss of any model on our four sentences.
- **Exactly:** `model.eval()` turns off dropout; `no_grad` skips the graph. `model(**batch, labels=labels).loss.item()` is a scalar float; logits inside are `(4, longest_len, 50257)`.
- **Why it is here:** A single reusable number makes the three stages comparable.

### 7. Stage 1: random weights  (`main.py` lines 47–52)

```python
# ---------------------------------------------------------------------------
# 1. Random weights = a model that never saw language.  Same architecture, untrained numbers.
# ---------------------------------------------------------------------------
cfg = AutoConfig.from_pretrained("distilgpt2")
random_model = AutoModelForCausalLM.from_config(cfg)      # from_config = architecture only, random weights
print(f"1) random weights   loss = {loss_of(random_model):.2f}   (guessing uniformly = ln(50257) = {math.log(50257):.2f})")
```

- **In simple words:** A model with the right architecture but no knowledge.
- **Exactly:** `AutoModelForCausalLM.from_config(cfg)` builds distilgpt2's architecture with random weights. Real output: `loss = 11.05` against `ln(50257) = 10.82`. A model guessing uniformly over the vocabulary has exactly that loss; initial random logits give slightly more.
- **Why it is here:** This is the starting point of pretraining and a sanity check: a loss far above 10.8 at the start of training means a bug.

### 8. Stage 2: pretrained weights  (`main.py` lines 54–58)

```python
# ---------------------------------------------------------------------------
# 2. Pretrained weights: already knows English, but not our pirate style.
# ---------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print(f"2) pretrained       loss = {loss_of(model):.2f}")
```

- **In simple words:** Load the real distilgpt2, which already knows English but not pirate style.
- **Exactly:** Real output: `loss = 5.92`. The loss is far below 10.8 because the model predicts English well, yet remains high because pirate phrases are unusual.
- **Why it is here:** Pretraining is the expensive part that someone else already did; we reuse it.

### 9. Sample text helper and a before-fine-tuning sample  (`main.py` lines 60–67)

```python
def sample_text(m, prompt="Arrr,", seed=1):
    torch.manual_seed(seed)
    m.eval()
    ids = tok(prompt, return_tensors="pt")["input_ids"]   # shape: (1, prompt_len)
    out = m.generate(ids, max_new_tokens=25, do_sample=True, top_k=40, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0])                              # out shape: (1, prompt_len + 25)

print("   sample before fine-tuning:", repr(sample_text(model)))
```

- **In simple words:** A function that generates 25 tokens from a prompt, so we can see the style change.
- **Exactly:** `torch.manual_seed(seed)` makes the sampled text repeatable. `do_sample=True, top_k=40` samples from the 40 most likely tokens. Real output before fine-tuning: `'Arrr, tux, and fudge.\n\nThis would be interesting, really, ...'`: fluent but not pirate-like.
- **Why it is here:** Comparing before and after is the simplest check that fine-tuning changed behaviour.

### 10. Stage 3: fine-tune with the five-move loop  (`main.py` lines 69–80)

```python
# ---------------------------------------------------------------------------
# 3. Fine-tune: the classic training loop in 6 lines (a toy version is in ../../03_ml_foundations/04_pytorch_training_loop; the full story is ../../06_llm_training_and_lora)
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)   # lr = learning rate: size of each update step
model.train()                                                  # train mode: dropout on
for step in range(20):
    out = model(**batch, labels=labels)       # forward pass + loss; out.loss is a scalar, out.logits (4, seq_len, 50257)
    out.loss.backward()                       # backpropagation: compute gradient of loss wrt every weight
    optimizer.step()                          # nudge each weight a little against its gradient
    optimizer.zero_grad()                     # clear gradients, otherwise they accumulate across steps
    if step % 5 == 0 or step == 19:
        print(f"   step {step:2d}  train loss = {out.loss.item():.3f}")
```

- **In simple words:** Run the training loop for 20 steps on our four sentences.
- **Exactly:** `AdamW(model.parameters(), lr=5e-5)` is the optimiser; `model.train()` turns dropout on. Each step: forward with labels, `loss.backward()`, `optimizer.step()`, `optimizer.zero_grad()`. Each step uses all four sentences (full batch). Real losses: step 0 `6.133`, step 5 `3.766`, step 10 `2.168`, step 15 `1.143`, step 19 `0.651`. Step 0 is above the 5.92 of stage 2 because dropout is on in training mode.
- **Why it is here:** This is exactly the loop from module 03; only the model and data changed. A low learning rate (5e-5) nudges the pretrained weights instead of wrecking them.

### 11. Evaluate and sample after fine-tuning  (`main.py` lines 82–85)

```python
print(f"3) after fine-tune  loss = {loss_of(model):.2f}")
print("   sample after fine-tuning :", repr(sample_text(model)))
print("   (we trained 20 steps on 4 sentences: it is memorising/overfitting. Real fine-tuning uses a")
print("    larger dataset AND a held-out validation set to catch that - ../../06_llm_training_and_lora/.)")
```

- **In simple words:** Check the new loss and generate again.
- **Exactly:** Real output: `loss = 0.53` and the sample `'Arrr, ye scurvy dog, hand over the treasure or walk the plank! This very well-known dog ...'`: it reproduces a training sentence. The printed note says we are memorising 4 sentences.
- **Why it is here:** A low loss on the training text only proves memorisation; real fine-tuning needs held-out validation data (module 06).

### 12. Inference: frozen weights, forward only  (`main.py` lines 87–93)

```python
# ---------------------------------------------------------------------------
# 4. Inference = frozen weights, forward only.  Check that no gradients are produced.
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    _ = model(**batch)
print("\n4) inference: eval() + no_grad() -> no dropout, no gradient memory, weights untouched")
```

- **In simple words:** Use the model without changing it.
- **Exactly:** `model.eval()` plus `torch.no_grad()` and one forward pass: no dropout, no gradients stored, weights untouched.
- **Why it is here:** This is what serving does. It is why inference needs far less memory than training (no gradients, no optimiser state).

### 13. Experiments and break-it comments  (`main.py` lines 95–98)

```python
# TRY IT: change lr=5e-5 to 5e-3 (100x bigger) and watch the loss: does it improve smoothly or explode? (see ../../05_llm_parameters/06_training_parameters)
# TRY IT: change range(20) to range(3) - how much style do you get with a few steps?
# BREAK IT: delete optimizer.zero_grad() - loss behaves oddly because gradients pile up.
# BREAK IT: use labels=batch["input_ids"] without the -100 masking - the loss quietly includes padding tokens.
```

- **In simple words:** Try a 100x larger learning rate, fewer steps, and remove `zero_grad` or the `-100` mask.
- **Exactly:** lr 5e-3 can make the loss jump (see module 05 step 06). Without `zero_grad` gradients accumulate. Without the mask the loss quietly includes padding tokens.
- **Why it is here:** Each of these is a real bug pattern in fine-tuning scripts.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `labels[mask == 0] = -100` | boolean-mask assignment: set entries where mask is 0 | `a[a < 0] = 0` clamps negatives |
| `.clone()` | make an independent copy of a tensor | - |
| `if step % 5 == 0 or step == 19:` | `%` = remainder; print every 5th step | `7 % 5 == 2` |
| `_ = model(**batch)` | `_` = "I ignore this result" | - |
| nested helper `def loss_of(model):` | a function using variables from the outer scope (`batch`, `labels`) | closure |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `AutoModelForCausalLM.from_config(cfg)` | transformers | model with RANDOM weights | config | model |
| `torch.optim.AdamW(params, lr)` | torch | optimizer: updates weights from gradients | `lr`, `weight_decay` | optimizer |
| `loss.backward()` | torch | backpropagation: fills `.grad` of every weight | - | None |
| `optimizer.step()` | torch | apply one update | - | None |
| `optimizer.zero_grad()` | torch | clear stored gradients | - | None |
| `model.train()` / `model.eval()` | torch | dropout on / off | - | model |
| `torch.manual_seed(n)` | torch | make randomness reproducible | seed | generator |
| `tok.pad_token = tok.eos_token` | transformers | attribute assignment: reuse EOS for padding | - | - |

## Execution trace
```
1) random weights   loss = 11.05   (guessing uniformly = ln(50257) = 10.82)
2) pretrained       loss = 5.92
   sample before fine-tuning: 'Arrr, tux, and fudge.\n\nThis would be interesting, really, and probably the least I could put down, but'
   step  0  train loss = 6.133
   step  5  train loss = 3.766
   step 10  train loss = 2.168
   step 15  train loss = 1.143
   step 19  train loss = 0.651
3) after fine-tune  loss = 0.53
   sample after fine-tuning : 'Arrr, ye scurvy dog, hand over the treasure or walk the plank! This very well-known dog will transform the room'
   (we trained 20 steps on 4 sentences: it is memorising/overfitting. ...)

4) inference: eval() + no_grad() -> no dropout, no gradient memory, weights untouched
```
- **Random weights ~ 11**: a model with no knowledge picks uniformly among 50,257 tokens: loss = ln(50257) = 10.82. Anything much lower means the model learned something.
- **Pretrained ~ 6 on pirate text**: it knows English but not this style.
- **Fine-tuned 0.5**: a few steps on 4 sentences and it reproduces them. That is memorisation (overfitting) - we only measured on the training text. Honest evaluation needs held-out data ([06_llm_training_and_lora](../../06_llm_training_and_lora/)).
- **Inference** = `eval()` + `no_grad()`: no weight updates, no dropout, no gradient memory.
- The 5-line loop (`forward -> loss -> backward -> step -> zero_grad`) is identical for 4 sentences and for a GPT-4-scale run. Only data size, hardware and tricks differ. You met the same loop on a toy network in [03_ml_foundations/04_pytorch_training_loop](../../03_ml_foundations/04_pytorch_training_loop/).

| Stage | Data | Cost | Result |
|---|---|---|---|
| Pretraining | trillions of tokens | millions of $ | general language ability |
| Fine-tuning | 100s - 100,000s of examples | $ - $$$ (cheaper with LoRA) | specific style/task |
| Inference | one prompt | ms - s per request | the product |

## Experiments
- Change `lr=5e-5` to `5e-3`: loss may spike or jump around ([05_llm_parameters/06_training_parameters](../../05_llm_parameters/06_training_parameters/) sweeps this systematically).
- Reduce to `range(3)`: how much style change after only 3 steps?
- Fine-tune on a different style (formal legal English) and sample the same prompt.

## Debugging practice
Delete `optimizer.zero_grad()`: gradients accumulate, so each update pushes with the sum of all previous gradients; loss decreases erratically or explodes. There's no exception - the code runs. This is the dangerous kind of bug; you find it by printing `loss` each step and reasoning about what *should* happen. Contrast with a crash, where the traceback tells you where.

## Rebuild checklist
- [ ] Write the 5-line training loop from memory with comments.
- [ ] Explain why `labels` uses -100 for padding.
- [ ] State the loss of a random model on a vocabulary of size V.
- [ ] Explain the difference between `train()` and `eval()`.

## The new problem (cliffhanger)
We have now seen the whole life of a model, but every number we picked was a guess: `lr=5e-5`, greedy decoding, 25 tokens, float32. Change any of them and the result changes, sometimes by a lot. How do we control the text (temperature, top-k), shrink the model (quantization), and tune training settings without guessing? Next: [05_llm_parameters](../../05_llm_parameters/README.md).

> **Say it aloud (interview box):** Pretraining teaches the model language by predicting the next token over a huge corpus; fine-tuning continues the same training on a small dataset with a low learning rate; inference freezes the weights and only runs forward passes. I measured the loss going from 11 for random weights, to 6 for pretrained, to under 1 after a few fine-tuning steps. Fine-tuning on very little data overfits, so real projects hold out validation data.
