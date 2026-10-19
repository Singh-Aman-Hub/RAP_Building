# Terms — Pretraining vs Fine-tuning vs Inference

Folder: `04_llm_fundamentals/07_pretrain_vs_finetune_vs_inference` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [Pretraining](#pretraining) · [Fine-tuning](#fine-tuning) · [Inference](#inference) · [math (module)](#math-module) · [Padding labels set to -100](#padding-labels-set-to--100) · [Boolean-mask assignment](#boolean-mask-assignment) · [.clone()](#clone) · [Helper function using outer variables (closure)](#helper-function-using-outer-variables-closure) · [model.eval() and no_grad](#modeleval-and-no_grad) · [from_config (random weights)](#from_config-random-weights) · [Loss of a random model: ln(V)](#loss-of-a-random-model-lnv) · [Sampling arguments in generate (top_k)](#sampling-arguments-in-generate-top_k) · [torch.optim.AdamW](#torchoptimadamw) · [Learning rate (lr)](#learning-rate-lr) · [model.train()](#modeltrain) · [Training loop](#training-loop) · [loss.backward()](#lossbackward) · [optimizer.step()](#optimizerstep) · [optimizer.zero_grad()](#optimizerzero_grad) · [Modulo operator (%)](#modulo-operator-) · [Overfitting / memorisation](#overfitting--memorisation) · [Validation set (held-out data)](#validation-set-held-out-data) · [Underscore as throwaway name](#underscore-as-throwaway-name) · [Silent bug (no exception)](#silent-bug-no-exception)

---

## Pretraining  `(concept)`
**1. Definition:** Training a model from random weights on a very large general text corpus with the objective of predicting the next token.
**2. In plain words:** This is where a model learns language and world knowledge. It takes weeks, thousands of GPUs and millions of dollars, so you almost never do it yourself; you use the result.
**3. Example to quote:** > "distilgpt2 was pretrained by others; I only downloaded its weights."

**Seen in:** `main.py`, docstring.

## Fine-tuning  `(concept)`
**1. Definition:** Continuing to train a pretrained model on a small, task- or style-specific dataset, using the same loss and training loop with a small learning rate.
**2. In plain words:** The model already knows English; fine-tuning teaches it your style or task, such as pirate speech or your company's tone. It takes minutes to hours. Risks: overfitting on little data, and forgetting some general knowledge.
**3. Example to quote:** > "After 20 steps on four pirate sentences, the training loss fell from 6.1 to 0.65 and the model copied the style."

**Seen in:** `main.py`, section 3.

## Inference  `(concept)`
**1. Definition:** Using a trained model with frozen weights to produce outputs; only forward passes are run.
**2. In plain words:** This is what happens when a user chats with a model. There are no weight updates, so it needs less memory than training. Latency and cost per request are what matter here.
**3. Example to quote:** > "For inference I use `model.eval()` and `torch.no_grad()`, so there is no dropout and no gradient memory."

**Seen in:** `main.py`, section 4.

## math (module)  `(standard library)`
**1. Definition:** The standard-library module with mathematical functions such as `log`, `sqrt` and `exp`.
**2. In plain words:** Used here to compute `math.log(50257)`, the loss expected from a model guessing at random.
**3. Example to quote:** > "I used `math.log(50257)` to get 10.82, the loss of uniform guessing."
```python
import math
math.log(50257)  # 10.82
```
**Seen in:** `main.py`, imports and section 1.

## Padding labels set to -100  `(concept)`
**1. Definition:** In Hugging Face loss computation, label value -100 marks positions to be ignored by the cross-entropy loss.
**2. In plain words:** Padding tokens are fake filler. If they counted in the loss, the model would be rewarded or punished for predicting filler. Setting their labels to -100 excludes them, a silent but important detail.
**3. Example to quote:** > "I set `labels[attention_mask == 0] = -100` so padding does not pollute the loss."

**Seen in:** `main.py`, labels construction.

## Boolean-mask assignment  `(syntax)`
**1. Definition:** Assigning to the elements of a tensor where a boolean condition is True, such as `labels[mask == 0] = -100`.
**2. In plain words:** It is a vectorised if-statement: "wherever the mask is 0, set the value". `labels` is a clone so the original IDs stay intact.
**3. Example to quote:** > "`a[a < 0] = 0` clamps negatives; I used the same pattern to blank out padding labels."
```python
a = torch.tensor([1, -2, 3]); a[a < 0] = 0
```
**Seen in:** `main.py`, labels construction.

## .clone()  `(third-party: torch)`
**1. Definition:** Returns an independent copy of a tensor.
**2. In plain words:** Without cloning, `labels` and `input_ids` would be the same object, and changing one would silently change the other.
**3. Example to quote:** > "I `clone()` the input IDs to create labels, then modify only the labels."

**Seen in:** `main.py`, labels construction.

## Helper function using outer variables (closure)  `(syntax)`
**1. Definition:** A function defined in a script that reads variables from the surrounding scope, such as `batch` and `labels` inside `loss_of`.
**2. In plain words:** `loss_of(model)` does not take the data as an argument; it uses `batch` and `labels` defined above it. This keeps calls short but makes the function depend on those names.
**3. Example to quote:** > "`loss_of(random_model)` and `loss_of(model)` evaluate different models on the same four sentences."

**Seen in:** `main.py`, `def loss_of`.

## model.eval() and no_grad  `(third-party: torch)`
**1. Definition:** `model.eval()` switches dropout off; `torch.no_grad()` stops gradient tracking; together they define inference mode.
**2. In plain words:** Eval makes behaviour deterministic; no_grad saves memory. The `loss_of` helper uses both because it only measures.
**3. Example to quote:** > "`loss_of` calls `model.eval()` and wraps the forward pass in `torch.no_grad()`."

**Seen in:** `main.py`, `loss_of` and section 4.

## from_config (random weights)  `(third-party: transformers)`
**1. Definition:** `AutoModelForCausalLM.from_config(cfg)` builds the architecture described by a config with randomly initialised weights.
**2. In plain words:** You get the same shape of model as distilgpt2 but with no knowledge. It is the "before training" baseline, and its loss shows what learning is worth.
**3. Example to quote:** > "A random distilgpt2 had loss 11.05, close to guessing uniformly among 50,257 tokens."

**Seen in:** `main.py`, section 1.

## Loss of a random model: ln(V)  `(concept)`
**1. Definition:** A model that assigns equal probability 1/V to each of V tokens has cross-entropy loss `ln(V)`; for V = 50257 this is 10.82.
**2. In plain words:** It is the "knows nothing" baseline. Anything clearly below it means the model learned something.
**3. Example to quote:** > "Uniform guessing costs ln(50257) = 10.82; pretrained distilgpt2 scored about 5.9 on pirate text."

**Seen in:** `main.py`, section 1.

## Sampling arguments in generate (top_k)  `(third-party: transformers)`
**1. Definition:** `generate(..., do_sample=True, top_k=40)` samples the next token randomly from only the 40 most likely candidates.
**2. In plain words:** Here it produces varied text for the before/after comparison. Sampling settings are explained fully in module 05.
**3. Example to quote:** > "I sampled with `top_k=40` to see the pirate style appear after fine-tuning."

**Seen in:** `main.py`, `sample_text`.

## torch.optim.AdamW  `(third-party: torch)`
**1. Definition:** An optimizer that updates parameters using gradients with adaptive per-parameter step sizes and decoupled weight decay.
**2. In plain words:** The optimizer is the part that actually changes the weights. AdamW is the usual choice for transformers. You give it the parameters and a learning rate.
**3. Example to quote:** > "I created `torch.optim.AdamW(model.parameters(), lr=5e-5)` to nudge the weights."
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
```
**Seen in:** `main.py`, section 3.

## Learning rate (lr)  `(concept)`
**1. Definition:** The scale of each weight update step in gradient-based training.
**2. In plain words:** Too large and the loss jumps around or explodes; too small and it learns very slowly. Fine-tuning uses small values like 5e-5 so as not to wreck pretrained knowledge.
**3. Example to quote:** > "I used `lr=5e-5`; making it 100 times bigger would risk unstable loss."

**Seen in:** `main.py`, section 3.

## model.train()  `(third-party: torch)`
**1. Definition:** Puts a model in training mode, enabling dropout and other training-only behaviours.
**2. In plain words:** It is the counterpart of `eval()`. Forgetting to switch modes is a common bug.
**3. Example to quote:** > "I call `model.train()` before the fine-tuning loop and `model.eval()` afterwards."

**Seen in:** `main.py`, section 3.

## Training loop  `(concept)`
**1. Definition:** The repeated cycle of forward pass, loss computation, backward pass, optimizer step and gradient reset.
**2. In plain words:** It is identical for four sentences and for a giant model: only data, hardware and tricks differ. You met it earlier on a toy network in module 03.
**3. Example to quote:** > "My loop is five lines: forward and loss, `backward()`, `step()`, `zero_grad()`."
```python
out = model(**batch, labels=labels)
out.loss.backward(); optimizer.step(); optimizer.zero_grad()
```
**Seen in:** `main.py`, section 3.

## loss.backward()  `(third-party: torch)`
**1. Definition:** Backpropagation: computes the gradient of the loss with respect to every parameter and stores it in each parameter's `.grad`.
**2. In plain words:** It answers "how should each weight change to lower the loss?" It changes nothing itself; the optimizer applies the changes.
**3. Example to quote:** > "`out.loss.backward()` fills the gradients; `optimizer.step()` then uses them."

**Seen in:** `main.py`, section 3.

## optimizer.step()  `(third-party: torch)`
**1. Definition:** Applies one parameter update using the stored gradients and the optimizer's rules.
**2. In plain words:** This is the line where learning actually happens, nudging each weight against its gradient.
**3. Example to quote:** > "After each `step()` my training loss decreased, from 6.1 toward 0.65."

**Seen in:** `main.py`, section 3.

## optimizer.zero_grad()  `(third-party: torch)`
**1. Definition:** Resets all stored gradients to zero.
**2. In plain words:** PyTorch adds new gradients to old ones. Without clearing, each update uses the sum of all previous gradients and training goes wrong, without any error message.
**3. Example to quote:** > "If I delete `zero_grad()` the code still runs, but the loss behaves erratically."

**Seen in:** `main.py`, section 3.

## Modulo operator (%)  `(syntax)`
**1. Definition:** `a % b` gives the remainder of dividing `a` by `b`.
**2. In plain words:** `step % 5 == 0` is True every fifth step, a handy way to print progress occasionally.
**3. Example to quote:** > "I print the loss when `step % 5 == 0` to avoid flooding the screen."
```python
7 % 5  # 2
```
**Seen in:** `main.py`, section 3.

## Overfitting / memorisation  `(concept)`
**1. Definition:** When a model fits its training data so closely (even memorising it) that it performs worse on unseen data.
**2. In plain words:** After 20 steps on four sentences the model reproduced them almost word for word. That looks great on training loss but means little. Real fine-tuning uses a validation set to catch this.
**3. Example to quote:** > "Training loss 0.53 on four sentences is memorisation, not proof the model learned pirate speak generally."

**Seen in:** `main.py`, section 3 and `WALKTHROUGH.md`.

## Validation set (held-out data)  `(concept)`
**1. Definition:** Data kept out of training and used only to measure how well the model generalises.
**2. In plain words:** It is the exam the model has not seen the answers to. Rising training accuracy with flat or worsening validation loss signals overfitting.
**3. Example to quote:** > "This demo measures loss on the training text only; module 06 adds a held-out set."

**Seen in:** `main.py`, final prints.

## Underscore as throwaway name  `(syntax)`
**1. Definition:** `_` is a conventional variable name meaning "I do not use this value".
**2. In plain words:** In `_ = model(**batch)` we run the forward pass to demonstrate inference but ignore the result.
**3. Example to quote:** > "I assign to `_` because I only care that the call runs without gradients."

**Seen in:** `main.py`, section 4.

## Silent bug (no exception)  `(concept)`
**1. Definition:** A programming error that does not raise an exception but produces wrong behaviour, such as missing `zero_grad()` or unmasked padding in the loss.
**2. In plain words:** These are the hardest bugs because there is no traceback to follow. You find them by printing values like the loss each step and reasoning about what should happen.
**3. Example to quote:** > "Without the -100 masking the loss silently included padding tokens; the code ran fine but the number was wrong."

**Seen in:** `WALKTHROUGH.md`, debugging section.
