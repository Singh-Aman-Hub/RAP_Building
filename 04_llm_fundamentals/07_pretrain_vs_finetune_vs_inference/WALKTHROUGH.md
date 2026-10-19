# Step 07 - Pretraining vs fine-tuning vs inference

## Goal
Use loss numbers to feel the three stages of an LLM's life, and see a complete training loop run.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/07_pretrain_vs_finetune_vs_inference/main.py
```

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
   sample before fine-tuning: 'Arrr, tux, and fudge.\n\nThis would be interesting, really, ...'
   step  0  train loss = 6.133
   step  5  train loss = 3.766
   step 10  train loss = 2.168
   step 15  train loss = 1.143
   step 19  train loss = 0.651
3) after fine-tune  loss = 0.53
   sample after fine-tuning : 'Arrr, ye scurvy dog, hand over the treasure or walk the plank! This very well-known ...'
```
- **Random weights ~ 11**: a model with no knowledge picks uniformly among 50,257 tokens: loss = ln(50257) = 10.82. Anything much lower means the model learned something.
- **Pretrained ~ 6 on pirate text**: it knows English but not this style.
- **Fine-tuned 0.5**: a few steps on 4 sentences and it reproduces them. That is memorisation (overfitting) - we only measured on the training text. Honest evaluation needs held-out data (module 06).
- **Inference** = `eval()` + `no_grad()`: no weight updates, no dropout, no gradient memory.
- The 5-line loop (`forward -> loss -> backward -> step -> zero_grad`) is identical for 4 sentences and for a GPT-4-scale run. Only data size, hardware and tricks differ. You met the same loop on a toy network in [03_ml_foundations/04_pytorch_training_loop](../../03_ml_foundations/04_pytorch_training_loop/).

| Stage | Data | Cost | Result |
|---|---|---|---|
| Pretraining | trillions of tokens | millions of $ | general language ability |
| Fine-tuning | 100s - 100,000s of examples | $ - $$$ (cheaper with LoRA) | specific style/task |
| Inference | one prompt | ms - s per request | the product |

## Experiments
- Change `lr=5e-5` to `5e-3`: loss may spike or jump around (module 05 step 06 sweeps this systematically).
- Reduce to `range(3)`: how much style change after only 3 steps?
- Fine-tune on a different style (formal legal English) and sample the same prompt.

## Debugging practice
Delete `optimizer.zero_grad()`: gradients accumulate, so each update pushes with the sum of all previous gradients; loss decreases erratically or explodes. There's no exception - the code runs. This is the dangerous kind of bug; you find it by printing `loss` each step and reasoning about what *should* happen. Contrast with a crash, where the traceback tells you where.

## Rebuild checklist
- [ ] Write the 5-line training loop from memory with comments.
- [ ] Explain why `labels` uses -100 for padding.
- [ ] State the loss of a random model on a vocabulary of size V.
- [ ] Explain the difference between `train()` and `eval()`.

## Say it aloud
"Pretraining teaches the model language by predicting the next token over a huge corpus; fine-tuning continues training on a small dataset for a task or style, using the same loss and loop but a low learning rate; inference freezes the weights and only runs forward passes. I demonstrated the loss going from 11 for random weights, to 6 for pretrained, to under 1 after a few fine-tuning steps. Trade-off: fine-tuning on little data overfits and can forget general knowledge, so you hold out validation data."
