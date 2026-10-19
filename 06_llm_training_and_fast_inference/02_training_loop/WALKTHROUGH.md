# Step 02 — The training loop

## Goal
Write the four-beat loop yourself — forward, loss, backward, optimizer step — and see loss fall on a real model (distilgpt2, 82M parameters).

## Run it
```bash
cd 06_llm_training_and_fast_inference/02_training_loop
python main.py        # ~40 s on an M1 (uses the Apple GPU, "mps")
```

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

## Say it aloud
"Training is a loop: run a batch forward to get the cross-entropy loss against the next tokens, backpropagate to get gradients, clip them so no single update explodes, and let AdamW nudge every weight downhill by the learning rate. I watch the loss fall and compare generations before and after. Trade-off: full fine-tuning updates all 82 million weights, so it needs memory for gradients and optimizer states too — which is exactly what LoRA avoids in step 05."
