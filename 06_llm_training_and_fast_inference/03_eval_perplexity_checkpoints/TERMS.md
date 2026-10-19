# Terms — Evaluation, Perplexity and Checkpoints

Folder: `06_llm_training_and_fast_inference/03_eval_perplexity_checkpoints` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [06 README](../README.md)

**Terms in this folder:** [Train / validation split](#train--validation-split) · [Validation loss](#validation-loss) · [Perplexity (math.exp)](#perplexity-mathexp) · [Evaluation function and weighted average](#evaluation-function-and-weighted-average) · [model.eval() and torch.no_grad() for scoring](#modeleval-and-torchno_grad-for-scoring) · [Return type hint `-> float`](#return-type-hint---float) · [Overfitting (train vs validation gap)](#overfitting-train-vs-validation-gap) · [Checkpoint](#checkpoint) · [save_pretrained](#save_pretrained) · [Safetensors file (model.safetensors)](#safetensors-file-modelsafetensors) · [from_pretrained from a local folder](#from_pretrained-from-a-local-folder) · [Path.mkdir(parents=True, exist_ok=True)](#pathmkdirparentstrue-exist_oktrue) · [Path.iterdir and generator expression](#pathiterdir-and-generator-expression) · [Logits comparison (reload check)](#logits-comparison-reload-check) · [state_dict](#state_dict) · [next(iter(dict))](#nextiterdict) · [OSError when loading](#oserror-when-loading) · [Early stopping](#early-stopping)

---

## Train / validation split  `(concept)`
**1. Definition:** A train/validation split divides the data into a part used to update the model (training set) and a held-out part used only to measure performance (validation set).
**2. In plain words:** If you test a student with the very questions they practised, the score proves nothing. The held-out 20% (5 of 25 blocks here) is the unseen exam. Never train on it. With such a small set the score is noisy, which is a limitation.
**3. Example to quote:** > "I kept the last 20% of blocks aside with `blocks[:-n_val], blocks[-n_val:]`, so I trained on 20 blocks and evaluated on 5."
```python
train, val = data[:-5], data[-5:]
```
**Seen in:** `main.py`, split section

## Validation loss  `(concept)`
**1. Definition:** Validation loss is the average loss (cross-entropy) of the model on the validation set, computed without updating the weights.
**2. In plain words:** It tells you how well the model does on text it hasn't memorised. Training loss going down while validation loss goes up means the model is memorising.
**3. Example to quote:** > "Validation loss fell from 4.72 before training to 2.52 after, but more slowly than training loss."
**Seen in:** `main.py`, `evaluate()`

## Perplexity (math.exp)  `(concept)`
**1. Definition:** Perplexity is `exp(loss)`: the exponential of the average per-token cross-entropy, interpretable as the effective number of equally likely tokens the model is choosing between at each step. `math.exp` is the standard-library function computing e raised to a power.
**2. In plain words:** Loss is hard to feel; perplexity translates it into "how confused is the model". A perplexity of 112 means roughly as unsure as picking among 112 words; 12.4 means much more confident. Lower is better.
**3. Example to quote:** > "Before fine-tuning the perplexity was `math.exp(4.72)` ≈ 112; after it was about 12.4, so it's far less confused about support-ticket text."
```python
import math
math.exp(2.52)   # about 12.4
```
**Seen in:** `main.py`, `perplexity()`

## Evaluation function and weighted average  `(concept)`
**1. Definition:** The `evaluate()` function loops over the validation loader, accumulates `loss * batch_size` and divides by the total number of examples, giving the true mean even if the last batch is smaller.
**2. In plain words:** If batches have different sizes, simply averaging batch losses would over-count the small one. Weighting each loss by its batch size fixes that.
**3. Example to quote:** > "I add `loss.item() * batch.shape[0]` and divide by `count`, because my last validation batch has 1 block, not 4."
**Seen in:** `main.py`, `evaluate()`

## model.eval() and torch.no_grad() for scoring  `(third-party: torch)`
**1. Definition:** `model.eval()` switches layers like dropout to inference behaviour, and `torch.no_grad()` stops gradient tracking; together they give deterministic, memory-light scoring.
**2. In plain words:** For a fair measurement you want no randomness and no learning. Evaluating in `train()` mode would make scores jitter between runs. Remember to call `model.train()` again afterwards.
**3. Example to quote:** > "`evaluate()` starts with `model.eval()`, runs inside `torch.no_grad()`, and ends with `model.train()` so training can continue."
**Seen in:** `main.py`, `evaluate()`

## Return type hint `-> float`  `(syntax)`
**1. Definition:** A return type hint such as `def evaluate() -> float:` documents the type a function returns; Python does not enforce it at run time.
**2. In plain words:** It's a label for readers and tools. It never changes what the function does.
**3. Example to quote:** > "`def perplexity(loss: float) -> float:` tells you it takes a float and returns one, but Python wouldn't stop you passing a string."
**Seen in:** `main.py`

## Overfitting (train vs validation gap)  `(concept)`
**1. Definition:** Overfitting appears as training loss continuing to fall while validation loss stalls or rises.
**2. In plain words:** The widening gap (epoch 4: train 2.03, val 2.52) is the early sign that the model is learning the training examples by heart. Fixes: fewer epochs, more data, early stopping.
**3. Example to quote:** > "With `EPOCHS = 12`, train loss kept falling but val loss turned upward, which is the textbook overfitting picture."
**Seen in:** `main.py` comments; `WALKTHROUGH.md`

## Checkpoint  `(concept)`
**1. Definition:** A checkpoint is a saved snapshot of a model (weights, config, and for resuming also optimizer state) that can be reloaded later.
**2. In plain words:** It's the "save game" of training. For a usable model folder you need the weights, the config and the tokenizer. A checkpoint is big: about 330 MB here, about 14 GB for a 7B model in fp16.
**3. Example to quote:** > "I saved to `checkpoints/distilgpt2-support` and verified it by reloading and comparing outputs."
**Seen in:** `main.py`, CHECKPOINTING section

## save_pretrained  `(third-party: transformers)`
**1. Definition:** `model.save_pretrained(dir)` and `tokenizer.save_pretrained(dir)` write a model's config and weights, or a tokenizer's files, into a folder in the Hugging Face format.
**2. In plain words:** You must save both model and tokenizer; if you skip the tokenizer, loading it later fails.
**3. Example to quote:** > "I call `model.save_pretrained(CKPT_DIR)` and `tokenizer.save_pretrained(CKPT_DIR)`, giving `config.json`, `model.safetensors` and tokenizer files."
**Seen in:** `main.py`

## Safetensors file (model.safetensors)  `(concept)`
**1. Definition:** Safetensors is a file format for storing tensors that is fast to load and, unlike pickle, cannot execute arbitrary code when opened.
**2. In plain words:** It's the safe, standard container for model weights. This is why newer Hugging Face checkpoints are `.safetensors` files.
**3. Example to quote:** > "The checkpoint folder contains `model.safetensors`; I prefer it over pickle-based files because loading it can't run hidden code."
**Seen in:** `WALKTHROUGH.md`, saved files list

## from_pretrained from a local folder  `(third-party: transformers)`
**1. Definition:** `from_pretrained(path)` accepts either a hub model name or a local directory, and loads config and weights from there.
**2. In plain words:** The same call you used to download distilgpt2 also reloads your own checkpoint; just pass the folder.
**3. Example to quote:** > "`AutoModelForCausalLM.from_pretrained(CKPT_DIR)` loads my fine-tuned model exactly like a hub model."
**Seen in:** `main.py`

## Path.mkdir(parents=True, exist_ok=True)  `(standard library)`
**1. Definition:** `Path.mkdir` creates a directory; `parents=True` also creates missing parent folders and `exist_ok=True` suppresses the error if it already exists.
**2. In plain words:** It makes the script safe to re-run: the folder is created the first time, and later runs don't crash.
**3. Example to quote:** > "I use `CKPT_DIR.mkdir(parents=True, exist_ok=True)` so rerunning doesn't raise FileExistsError."
**Seen in:** `main.py`

## Path.iterdir and generator expression  `(standard library)`
**1. Definition:** `Path.iterdir()` yields the entries of a directory; a generator expression `(p.name for p in ...)` lazily produces values one by one, and `sorted(...)` turns them into a sorted list.
**2. In plain words:** It's the one-liner to list file names in a folder, like `ls`.
**3. Example to quote:** > "`sorted(p.name for p in CKPT_DIR.iterdir())` printed `config.json, model.safetensors, tokenizer.json...`."
**Seen in:** `main.py`

## Logits comparison (reload check)  `(concept)`
**1. Definition:** Comparing the logits (raw output scores) of the original and the reloaded model on the same input, using `(a - b).abs().max()`, verifies the checkpoint reproduces the model.
**2. In plain words:** If the biggest difference is 0.0, the save/load round trip lost nothing. It's a cheap and powerful sanity test.
**3. Example to quote:** > "The max difference between original and reloaded logits was 0.0, so the checkpoint is faithful."
**Seen in:** `main.py`

## state_dict  `(third-party: torch)`
**1. Definition:** `model.state_dict()` returns an ordered dictionary mapping each layer's parameter name to its tensor.
**2. In plain words:** It's the model's weights without the architecture. Saving only this is the lower-level PyTorch way; you must rebuild the same architecture to load it. This model has 77 tensors, the first being `transformer.wte.weight`, the token embedding table.
**3. Example to quote:** > "`state_dict` has 77 tensors; the first key is the embedding table `transformer.wte.weight`."
**Seen in:** `main.py`, end

## next(iter(dict))  `(Python built-in)`
**1. Definition:** `next(iter(d))` returns the first key of dictionary `d`, because iterating a dict yields its keys in insertion order.
**2. In plain words:** A quick peek at one key without a loop.
**3. Example to quote:** > "I used `next(iter(state))` to show the first parameter name."
```python
next(iter({"a": 1}))   # 'a'
```
**Seen in:** `main.py`

## OSError when loading  `(concept)`
**1. Definition:** `OSError` is Python's error for failed operating-system operations; Hugging Face raises it when a model or tokenizer folder lacks the expected files.
**2. In plain words:** If you forget `tokenizer.save_pretrained`, loading the tokenizer gives "Can't load tokenizer for ...". Read the message, `ls` the folder, and notice the missing files.
**3. Example to quote:** > "I got an OSError, listed the folder, saw no tokenizer files, and concluded a checkpoint is model plus tokenizer plus config."
**Seen in:** `WALKTHROUGH.md`, Debugging practice

## Early stopping  `(concept)`
**1. Definition:** Early stopping ends training when validation loss stops improving.
**2. In plain words:** It prevents memorisation by stopping at the best point instead of running all epochs.
**3. Example to quote:** > "Since val loss stalled after epoch 3, I'd stop there, and the extra epochs would only overfit."
**Seen in:** `WALKTHROUGH.md`, Experiments
