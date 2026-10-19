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
cd 06_llm_training_and_fast_inference/03_eval_perplexity_checkpoints
python main.py      # writes ./checkpoints (~330 MB, safe to delete)
```

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
