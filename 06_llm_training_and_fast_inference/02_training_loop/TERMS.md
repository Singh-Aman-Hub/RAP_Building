# Terms — Training Loop

Folder: `06_llm_training_and_fast_inference/02_training_loop` · Files covered: `main.py`, `WALKTHROUGH.md` (and the `common.py` helpers it imports)
Back to module: [06 README](../README.md)

**Terms in this folder:** [sys.path.insert](#syspathinsert) · [Device (cpu / mps / cuda)](#device-cpu--mps--cuda) · [Hyperparameter](#hyperparameter) · [Learning rate](#learning-rate) · [Epoch](#epoch) · [Gradient clipping (clip_grad_norm_)](#gradient-clipping-clip_grad_norm_) · [torch.manual_seed](#torchmanual_seed) · [AutoModelForCausalLM.from_pretrained](#automodelforcausallmfrom_pretrained) · [model.to(device)](#modeltodevice) · [AdamW](#adamw) · [model.parameters()](#modelparameters) · [model.train() / model.eval()](#modeltrain--modeleval) · [torch.no_grad](#torchno_grad) · [Context manager (`with`)](#context-manager-with) · [model.generate and greedy decoding](#modelgenerate-and-greedy-decoding) · [Tuple unpacking `for (batch,) in loader`](#tuple-unpacking-for-batch-in-loader) · [Forward pass with labels](#forward-pass-with-labels) · [Loss (cross-entropy) and loss.item()](#loss-cross-entropy-and-lossitem) · [optimizer.zero_grad()](#optimizerzero_grad) · [loss.backward()](#lossbackward) · [Gradient norm](#gradient-norm) · [optimizer.step()](#optimizerstep) · [Training loop](#training-loop) · [Overfitting (memorisation)](#overfitting-memorisation) · [Silent bug](#silent-bug) · [Formatted f-string `:.3f` and `:2d`](#formatted-f-string-3f-and-2d)

---

## sys.path.insert  `(standard library)`
**1. Definition:** `sys.path` is the list of folders Python searches when you `import` something; `sys.path.insert(0, folder)` puts a folder at the front of that list.
**2. In plain words:** The step folders are not packages, so Python wouldn't find `common.py` in the module folder by default. Adding that folder to the search list makes `from common import ...` work. Because step folders start with digits and every step has a `main.py`, small helper files like `common.py` need unique names.
**3. Example to quote:** > "I insert the module folder into `sys.path` so that `from common import get_device, make_blocks, read_lines` works from inside the step."
**Seen in:** `main.py`, imports

## Device (cpu / mps / cuda)  `(concept)`
**1. Definition:** A device is the hardware where tensors live and computations run: `cpu`, `cuda` (NVIDIA GPU) or `mps` (Apple-silicon GPU).
**2. In plain words:** GPUs do the big matrix maths much faster than the CPU. The helper `get_device()` picks the best one available, so the same script runs on a laptop or a GPU server. The model and its data must be on the same device.
**3. Example to quote:** > "The script printed `device: mps`, so training ran on the Apple GPU of my M1."
**Seen in:** `main.py` (`get_device()` from `common.py`)

## Hyperparameter  `(concept)`
**1. Definition:** A hyperparameter is a setting you choose before training (learning rate, batch size, epochs, clipping threshold) rather than something the model learns.
**2. In plain words:** Weights are learned; hyperparameters are the dials you set. Changing them can make training faster, unstable or useless. Module 05 explains each one.
**3. Example to quote:** > "My hyperparameters here are LR 5e-5, batch size 4, 6 epochs and a clip threshold of 1.0."
**Seen in:** `main.py`, constants block

## Learning rate  `(concept)`
**1. Definition:** The learning rate (`lr`) scales how far the optimizer moves each weight in the direction suggested by the gradient on every step.
**2. In plain words:** It is the step size when walking downhill. Too small: almost no progress (5e-7 barely moved the loss). Too large: you overshoot and the loss can bounce or become `nan`. It is the most sensitive knob.
**3. Example to quote:** > "With 5e-7 the epoch-3 loss was 4.64; with 5e-5 it was 2.14; with 5e-4 it was 0.99. On a bigger real dataset the high rate could damage the pretrained knowledge."
**Seen in:** `main.py` (`LEARNING_RATE`), `WALKTHROUGH.md` experiments

## Epoch  `(concept)`
**1. Definition:** An epoch is one full pass through the entire training dataset.
**2. In plain words:** One epoch means the model has seen every block once. Several epochs give the model repeated chances to learn, but too many on a tiny dataset make it memorise. Here, 25 blocks in batches of 4 give 7 steps per epoch, and 6 epochs give 42 steps.
**3. Example to quote:** > "Six epochs over 25 blocks with batch size 4 is 7 × 6 = 42 optimizer steps."
**Seen in:** `main.py` (`EPOCHS`, outer loop)

## Gradient clipping (clip_grad_norm_)  `(third-party: torch)`
**1. Definition:** `torch.nn.utils.clip_grad_norm_(params, max_norm)` rescales all gradients together if their combined (L2) norm exceeds `max_norm`, and returns the norm measured before clipping.
**2. In plain words:** Occasionally one bad batch produces a huge gradient that would throw the weights far off course. Clipping caps the size of the update while keeping its direction. Here the printed norm is 10 to 20, so with a limit of 1.0 every step is clipped, which is fine for this toy run.
**3. Example to quote:** > "I clip at `MAX_GRAD_NORM = 1.0` so a single spike can't ruin the model; the printed `grad_norm` shows the size before clipping."
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
**Seen in:** `main.py`, inside the loop

## torch.manual_seed  `(third-party: torch)`
**1. Definition:** `torch.manual_seed(n)` sets the starting state of PyTorch's random number generator so that random operations repeat the same way.
**2. In plain words:** Shuffling and dropout use randomness. Fixing the seed makes two runs comparable, so you can tell whether a change in loss came from your edit and not from luck.
**3. Example to quote:** > "I set the seed to 0 so my learning-rate comparison used the same shuffling every time."
**Seen in:** `main.py`, `torch.manual_seed(SEED)`

## AutoModelForCausalLM.from_pretrained  `(third-party: transformers)`
**1. Definition:** It loads the architecture and pretrained weights of a named causal language model (here `distilgpt2`) into a PyTorch model object.
**2. In plain words:** Rather than training from random numbers, you start from a model that already knows English. The first run downloads about 330 MB.
**3. Example to quote:** > "I load distilgpt2, an 82-million-parameter model, with `AutoModelForCausalLM.from_pretrained('distilgpt2')` and fine-tune it on support text."
**Seen in:** `main.py`

## model.to(device)  `(third-party: torch)`
**1. Definition:** `module.to(device)` moves all of the model's parameters to the chosen device and returns the model.
**2. In plain words:** Weights start in CPU memory. Moving them to the GPU is needed for fast training; inputs must be moved with `.to(device)` as well, otherwise you get a device-mismatch error.
**3. Example to quote:** > "`model = ...from_pretrained(...).to(device)` and then `batch = batch.to(device)` keep model and data on the same device."
**Seen in:** `main.py`

## AdamW  `(third-party: torch)`
**1. Definition:** `torch.optim.AdamW` is an optimizer implementing Adam (adaptive per-weight step sizes using running averages of gradients) with decoupled weight decay.
**2. In plain words:** The optimizer is the part that actually changes the weights after the gradients are computed. AdamW is the default choice for fine-tuning transformers because it adapts the step for each weight and regularises sensibly. It stores extra numbers per weight, which costs memory.
**3. Example to quote:** > "I use `AdamW(model.parameters(), lr=5e-5)`. Full fine-tuning keeps gradients and Adam's extra state for all 82M weights, which is why LoRA later is cheaper."
**Seen in:** `main.py`

## model.parameters()  `(third-party: torch)`
**1. Definition:** `model.parameters()` returns an iterator over every learnable tensor (weights and biases) of a module.
**2. In plain words:** It is the list of things the optimizer is allowed to change, and what gradient clipping measures.
**3. Example to quote:** > "I pass `model.parameters()` to both the optimizer and the clipping function so they act on the same weights."
**Seen in:** `main.py`

## model.train() / model.eval()  `(third-party: torch)`
**1. Definition:** `model.train()` puts a module in training mode (dropout active); `model.eval()` puts it in evaluation mode (dropout off, deterministic layers).
**2. In plain words:** Dropout randomly switches off some units to prevent overfitting during training, but you want stable output when generating. The `sample` function flips to eval, generates, then flips back to train.
**3. Example to quote:** > "Inside `sample()` I call `model.eval()` before generating and `model.train()` after, so the training loop resumes with dropout on."
**Seen in:** `main.py`, `sample()` and before the loop

## torch.no_grad  `(third-party: torch)`
**1. Definition:** `torch.no_grad()` is a context manager that disables gradient tracking for the operations inside it.
**2. In plain words:** While generating you don't need to train, so skipping gradient bookkeeping saves memory and time.
**3. Example to quote:** > "I wrap the `generate` call in `with torch.no_grad():` because I'm only reading outputs, not updating weights."
**Seen in:** `main.py`, `sample()`

## Context manager (`with`)  `(syntax)`
**1. Definition:** A `with` statement runs a block inside a context manager, which sets something up on entry and cleans it up on exit, even if an error occurs.
**2. In plain words:** It guarantees that a temporary state (gradients off, a file open) is undone after the block. Here, gradients are off only inside the indented lines.
**3. Example to quote:** > "`with torch.no_grad():` turns gradients off just for that block and turns them back on automatically afterwards."
**Seen in:** `main.py`, `sample()`

## model.generate and greedy decoding  `(third-party: transformers)`
**1. Definition:** `model.generate(...)` produces new tokens one at a time, autoregressively; with `do_sample=False` it uses greedy decoding, always choosing the highest-probability next token, up to `max_new_tokens`.
**2. In plain words:** It is the "write text" function. Greedy is deterministic, so before and after training the same prompt can be compared fairly. `pad_token_id=tokenizer.eos_token_id` is passed because GPT-2 has no pad token and `generate` would warn otherwise.
**3. Example to quote:** > "Before training, greedy output repeated 'I reset my password. Reply:'; after training it said 'Open Settings, tap Account, then choose Reset Password.'"
**Seen in:** `main.py`, `sample()`

## Tuple unpacking `for (batch,) in loader`  `(syntax)`
**1. Definition:** Tuple unpacking assigns the items of a tuple to separate names; `(batch,)` unpacks a tuple with exactly one item.
**2. In plain words:** A `TensorDataset` with a single tensor yields one-item tuples. The trailing comma in `(batch,)` pulls the tensor out of each tuple. Forgetting it leaves you with a tuple and a confusing error.
**3. Example to quote:** > "The loader gives `(tensor,)` each time, so I write `for (batch,) in loader:` to get the tensor itself."
```python
for (x,) in [(1,), (2,)]:
    print(x)
```
**Seen in:** `main.py`, training loop

## Forward pass with labels  `(concept)`
**1. Definition:** A forward pass runs inputs through the model to get outputs; calling `model(input_ids=batch, labels=batch)` also makes the model shift the labels by one and return `.loss`, the average cross-entropy per token, in addition to `.logits`.
**2. In plain words:** "Forward" means input to prediction. Passing labels asks the library to also compute how wrong the predictions were. Without labels, `out.loss` is `None` and calling `.backward()` on it raises an `AttributeError`.
**3. Example to quote:** > "`out = model(input_ids=batch, labels=batch)` gives me `out.loss`, which started at 5.09 on step 1."
**Seen in:** `main.py`, step 1 of the loop

## Loss (cross-entropy) and loss.item()  `(concept)`
**1. Definition:** Loss is a single number measuring how wrong the model's predictions are; for language models it is the cross-entropy between predicted next-token probabilities and the actual next tokens. `loss.item()` converts a one-element tensor to a plain Python float.
**2. In plain words:** Lower loss means the model assigns more probability to the real next words. You watch it fall as training works. Individual steps are noisy because each batch is just 4 blocks.
**3. Example to quote:** > "Loss went from 5.088 at step 1 to 1.467 at step 42, so the model got better at predicting the support text."
**Seen in:** `main.py`, print line

## optimizer.zero_grad()  `(third-party: torch)`
**1. Definition:** `optimizer.zero_grad()` resets the stored gradient of every parameter to zero (or None).
**2. In plain words:** PyTorch adds new gradients to old ones by default. Without clearing, each step uses an accumulation of all previous batches. This is a silent bug: no error, just odd behaviour.
**3. Example to quote:** > "I call `zero_grad()` before `backward()` on every step; if I delete it the loss still prints but behaves erratically."
**Seen in:** `main.py`, step 2 of the loop

## loss.backward()  `(third-party: torch)`
**1. Definition:** `loss.backward()` runs backpropagation (autograd), computing the gradient of the loss with respect to every parameter and storing it in each parameter's `.grad`.
**2. In plain words:** It answers "for each weight, if I nudged it up a little, would the loss go up or down, and by how much?" Those answers are the gradients the optimizer uses.
**3. Example to quote:** > "After the forward pass, `loss.backward()` fills in the gradients for all 82 million weights."
**Seen in:** `main.py`

## Gradient norm  `(concept)`
**1. Definition:** The gradient norm is the overall length (L2 norm) of the vector formed by all parameter gradients.
**2. In plain words:** It tells you how big the proposed update is. Watching it helps spot exploding gradients. In this run it was about 20 before clipping.
**3. Example to quote:** > "The printed `grad_norm 20.57` is measured before clipping, then scaled down to my limit of 1.0."
**Seen in:** `main.py`, print line

## optimizer.step()  `(third-party: torch)`
**1. Definition:** `optimizer.step()` updates every parameter using its stored gradient according to the optimizer's rule (for AdamW, an adaptive step scaled by the learning rate).
**2. In plain words:** This is the moment the model actually learns something: the weights move slightly downhill.
**3. Example to quote:** > "The four-beat loop ends with `optimizer.step()`, which applies the clipped gradients to the weights."
**Seen in:** `main.py`

## Training loop  `(concept)`
**1. Definition:** The training loop repeats, for each batch: forward pass, loss, backward pass (gradients), optional clipping, optimizer step.
**2. In plain words:** Everything in deep learning training is this loop. Libraries wrap it, but knowing the four beats lets you debug it.
**3. Example to quote:** > "Forward, loss, backward, step: I run that for 7 batches per epoch for 6 epochs and print the loss."
**Seen in:** `main.py`

## Overfitting (memorisation)  `(concept)`
**1. Definition:** Overfitting is when a model fits its training data so closely, including noise, that it performs worse on new data.
**2. In plain words:** With 20 epochs on a tiny corpus the loss drops towards 0, meaning the model has memorised the lines. Only a separate validation set (the next step) reveals it.
**3. Example to quote:** > "Setting `EPOCHS = 20` pushed training loss near zero, a sign of memorisation, not understanding."
**Seen in:** `main.py` TRY IT; `WALKTHROUGH.md`

## Silent bug  `(concept)`
**1. Definition:** A silent bug is a defect that raises no error or traceback but makes the program behave incorrectly.
**2. In plain words:** Forgetting `zero_grad()` is a classic one. You only catch it by watching the loss curve, so always monitor behaviour, not just whether the script crashes.
**3. Example to quote:** > "Removing `zero_grad()` produced no traceback, only strange loss values, so I always plot or print the loss."
**Seen in:** `WALKTHROUGH.md`, Debugging practice

## Formatted f-string `:.3f` and `:2d`  `(syntax)`
**1. Definition:** Inside an f-string, a colon format spec controls display: `:.3f` shows a float with 3 decimals and `:2d` pads an integer to width 2.
**2. In plain words:** It keeps the printed log lines aligned and readable, e.g. `loss 5.088`, `step  1`.
**3. Example to quote:** > "`f\"{loss.item():.3f}\"` prints 5.088 instead of 5.08837890625."
```python
f"{3.14159:.3f}"   # '3.142'
```
**Seen in:** `main.py`, print line
