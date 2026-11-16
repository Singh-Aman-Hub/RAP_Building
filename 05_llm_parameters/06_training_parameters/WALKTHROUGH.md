# Step 06 - Training parameters: learning rate, batch size, epochs, optimizers, loss

## The story so far
[Step 05](../05_quantization/WALKTHROUGH.md) made inference fit. To change what the model knows you must train it, and training has its own dials that can silently ruin a run.

## The pain point
A fine-tune with the wrong learning rate either never learns or wrecks the pretrained knowledge: loss goes to NaN, or barely moves. Batch size and epochs decide whether you run out of memory or just memorise four sentences. Copying numbers from a from-scratch tutorial is how people fail here.

## The spark
Run controlled experiments: one knob at a time on a tiny network where results appear in seconds (learning rate, batch size, epochs, optimizer), then confirm the pattern on a real LLM fine-tune. Learn to read the loss curve as a gauge.

## Goal
See what each training knob does by running controlled experiments (tiny network for speed, then a real LLM fine-tune).

## Run it
```bash
.venv/bin/python 05_llm_parameters/06_training_parameters/main.py
```
(~1 minute; builds on [03_ml_foundations](../../03_ml_foundations/README.md) training-loop and overfitting steps.)

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Learning rate | step size, lr, alpha, eta | Yes | Multiplier on the update; typical LLM fine-tuning range is 1e-5 to 5e-4 (full fine-tune lower, LoRA higher) |
| Batch size | mini-batch size, micro-batch (per device) | Related | Examples per update; "global batch" = micro-batch x gradient accumulation x number of GPUs |
| Epoch vs step vs iteration | full pass vs one update | No | Updates per epoch = ceil(n / batch size) |
| SGD / momentum / Adam / AdamW | optimisers | Different | SGD: plain step; momentum: smoothed direction; Adam: per-weight adaptive step; AdamW: Adam with decoupled weight decay (LLM default) |
| Weight decay vs L2 regularisation | shrinking weights | Equal for plain SGD, not for Adam | AdamW applies decay directly to the weights instead of adding it to the gradient |
| CrossEntropyLoss | softmax + NLL, `F.cross_entropy` | Yes | Takes raw logits, not probabilities |
| Overfitting vs early stopping | train-val gap vs the fix | No | Overfitting is the problem; early stopping (stop at the validation minimum) is one remedy |
| Divergence | exploding loss, instability | Yes | Loss grows because steps overshoot (lr too large) |
| Hidden size / width | `hidden=64`, capacity | Yes | More units mean more capacity to memorise |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–16)

```python
"""
GOAL:   See with your own eyes what learning rate, batch size, epochs, optimizer and loss
        function do - first on a tiny fast problem, then on a real LLM fine-tune.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/06_training_parameters/main.py
EXPECTED OUTPUT (short): 5 small tables: lr sweep (tiny lr = slow, huge lr = diverges), batch-size
        sweep, epoch curve showing train/val gap, optimizer comparison, and an LLM lr sweep.
NEEDS:  torch, scikit-learn (make_moons), transformers (distilgpt2 cached). ~40-60 s.

Builds on: 03_ml_foundations/02_linear_regression_numpy (gradient descent by hand),
           03_ml_foundations/04_pytorch_training_loop (the loop used below),
           03_ml_foundations/05_overfitting_train_val (the train/val gap in section 3),
           03_ml_foundations/06_softmax_cross_entropy (the loss used here and by LLMs).
Leads to:  06_llm_training_and_lora (the same knobs for a real fine-tune, LoRA, memory).
"""
```

- **In simple words:** Says this file shows learning rate, batch size, epochs, optimiser and loss on a tiny problem and then on a real LLM.
- **Exactly:** Docstring only. Promised: five small tables. It builds on module 03 steps 02, 04, 05 and 06 and leads to module 06.
- **Why it is here:** A map of which earlier steps each section reuses.

### 2. Imports  (`main.py` lines 18–22)

```python
import time
import torch
import torch.nn as nn
from sklearn.datasets import make_moons           # a 2-D toy dataset: two interleaving half-circles
from transformers import AutoTokenizer, AutoModelForCausalLM
```

- **In simple words:** Timing, PyTorch, a toy data generator and the LLM loaders.
- **Exactly:** `make_moons` creates two interleaving half-circles, a non-linear 2-class problem that needs a small network.
- **Why it is here:** A tiny problem trains in seconds, so you can run many experiments.

### 3. Make the toy dataset and split it  (`main.py` lines 24–29)

```python
# ---------------------------------------------------------------------------
# Data: 2-D points with a label 0/1.  Split into train and validation (never train on validation!).
# ---------------------------------------------------------------------------
X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)       # numpy -> torch tensors
X_train, y_train, X_val, y_val = X[:300], y[:300], X[300:], y[300:]   # shapes: (300, 2), (300,), (300, 2), (300,)
```

- **In simple words:** 600 two-dimensional points with labels 0 or 1; the first 300 train, the other 300 validate.
- **Exactly:** `make_moons(n_samples=600, noise=0.25, random_state=0)` returns `X (600, 2)` float64 and `y (600,)`. `torch.tensor(X, dtype=torch.float32)` converts. Shapes: `X_train (300, 2)`, `y_train (300,)`, `X_val (300, 2)`, `y_val (300,)`.
- **Why it is here:** Never evaluate on training data (module 03 step 05); the same split is used everywhere below.

### 4. A small neural network  (`main.py` lines 32–34)

```python
def make_model(hidden=64):
    """A small MLP: 2 inputs -> hidden -> hidden -> 2 class scores (logits)."""
    return nn.Sequential(nn.Linear(2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
```

- **In simple words:** A stack of three linear layers with ReLU between them.
- **Exactly:** `nn.Sequential(Linear(2, hidden), ReLU(), Linear(hidden, hidden), ReLU(), Linear(hidden, 2))`. With hidden=64: shapes `(n, 2) -> (n, 64) -> (n, 64) -> (n, 2)`; the last output is two logits, one per class. Parameters: 2*64+64 + 64*64+64 + 64*2+2 = 4,482.
- **Why it is here:** ReLU gives the non-linearity; without it the three layers collapse into one linear map and cannot separate half-moons.

### 5. train(): setup, loss and the four optimisers  (`main.py` lines 37–48)

```python
def train(optimizer_name="adam", lr=0.01, batch_size=32, epochs=30, hidden=64, X=X_train, y=y_train, track=False):
    """The standard loop. Returns (final_train_loss, final_val_loss, history)."""
    torch.manual_seed(0)                                            # same initial weights for fair comparison
    model = make_model(hidden)
    loss_fn = nn.CrossEntropyLoss()                                 # softmax + negative log-likelihood (module 03/06)
    opts = {
        "sgd": lambda: torch.optim.SGD(model.parameters(), lr=lr),
        "sgd+momentum": lambda: torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9),
        "adam": lambda: torch.optim.Adam(model.parameters(), lr=lr),
        "adamw": lambda: torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01),
    }
    opt = opts[optimizer_name]()                                    # build the chosen optimizer
```

- **In simple words:** A function that trains the network with a chosen optimiser and settings, and returns the final losses.
- **Exactly:** `torch.manual_seed(0)` means every call starts from the same weights, so only the knob under test changes. `nn.CrossEntropyLoss()` is softmax plus negative log-likelihood and takes raw logits. `opts` maps names to lambdas that build `SGD`, `SGD` with `momentum=0.9`, `Adam` or `AdamW(weight_decay=0.01)`; the lambdas are evaluated only for the chosen one, after `model` exists.
- **Why it is here:** One function reused for every table keeps the comparisons fair.

### 6. The training loop  (`main.py` lines 49–57)

```python
    history = []
    for epoch in range(epochs):                                     # one epoch = one pass over ALL training data
        perm = torch.randperm(len(X))                               # shuffle each epoch
        for i in range(0, len(X), batch_size):                      # one iteration = one batch = one weight update
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(X[idx]), y[idx])                   # forward + loss
            opt.zero_grad()
            loss.backward()                                         # gradients
            opt.step()                                              # update weights
```

- **In simple words:** Shuffle, cut into batches, and for each batch predict, score, clear, differentiate, update.
- **Exactly:** `torch.randperm(len(X))` is a fresh shuffle each epoch; `range(0, len(X), batch_size)` steps by batch size, so the last batch may be smaller. `model(X[idx])` is `(batch, 2)`; `loss_fn(logits, y[idx])` compares with integer class labels `(batch,)`. `zero_grad`, `backward`, `step` are the same five moves as module 03 step 04.
- **Why it is here:** Updates per epoch is `ceil(n / batch_size)`; that number is what the batch-size table varies.

### 7. Optionally record loss curves, and return the final losses  (`main.py` lines 58–62)

```python
        if track:
            with torch.no_grad():
                history.append((loss_fn(model(X), y).item(), loss_fn(model(X_val), y_val).item()))
    with torch.no_grad():
        return loss_fn(model(X), y).item(), loss_fn(model(X_val), y_val).item(), history
```

- **In simple words:** Record train and validation loss after each epoch if asked; always return the final two numbers.
- **Exactly:** `track=True` stores `(train_loss, val_loss)` per epoch under `no_grad`. The return is `(final_train_loss, final_val_loss, history)`.
- **Why it is here:** Tracking is off by default because it costs extra forward passes.

### 8. Experiment 1: learning rate with plain SGD  (`main.py` lines 65–73)

```python
# ---------------------------------------------------------------------------
# 1. Learning rate: size of each step downhill.
# ---------------------------------------------------------------------------
print("1) LEARNING RATE (SGD, batch 32, 30 epochs)")
notes = {0.0001: "too small: barely moves", 0.001: "still slow", 0.01: "ok", 0.1: "good",
         1.0: "works here, but near the edge", 10.0: "too large: loss BLOWS UP (overshoots the valley)"}
for lr, note in notes.items():
    tr, va, _ = train("sgd", lr=lr)
    print(f"   lr={lr:<8} train loss={tr:.3f}  val loss={va:.3f}  {note}")
```

- **In simple words:** Try six step sizes from tiny to huge.
- **Exactly:** Real output: lr 0.0001 gives train 0.734 and val 0.736 (barely moves from its starting value); 0.001 gives 0.685; 0.01 gives 0.390 / 0.344; 0.1 gives 0.229 / 0.205; 1.0 gives 0.229 / 0.212 (works here); 10.0 gives 2.859 / 2.889 (diverges). The `notes` dict labels each.
- **Why it is here:** The learning rate has a safe window: too small trains slowly and too large overshoots. It is the most important hyperparameter.

### 9. Experiment 2: batch size  (`main.py` lines 75–83)

```python
# ---------------------------------------------------------------------------
# 2. Batch size: examples per update.  Small = noisy but many updates; large = smooth but few updates.
# ---------------------------------------------------------------------------
print("\n2) BATCH SIZE (Adam lr=0.01, 30 epochs) -> updates per epoch = ceil(300 / batch)")
for bs in (4, 32, 128, 300):
    t0 = time.perf_counter()
    tr, va, _ = train("adam", lr=0.01, batch_size=bs)
    print(f"   batch={bs:<4} updates/epoch={-(-300 // bs):<4} train loss={tr:.3f}  val loss={va:.3f}  time={time.perf_counter() - t0:.2f}s")
print("   Bigger batches use more memory per step (activations) - the main GPU-memory knob in LLM training.")
```

- **In simple words:** Same total epochs; vary how many examples feed each update.
- **Exactly:** `-(-300 // bs)` is ceiling division. Real output (Adam lr 0.01, 30 epochs): batch 4 has 75 updates per epoch, val 0.162, time 5.03 s; batch 32 has 10 updates, val 0.162, 0.85 s; batch 128 has 3 updates, val 0.161, 0.30 s; batch 300 has 1 update, val 0.148, 0.13 s. Final quality is similar; small batches cost far more time per epoch. The timings are from a loaded machine, so read the ratios.
- **Why it is here:** On a GPU, larger batches need more activation memory, so batch size is the main memory knob in LLM training.

### 10. Experiment 3: epochs and overfitting  (`main.py` lines 85–93)

```python
# ---------------------------------------------------------------------------
# 3. Epochs: too few = underfit; too many (with a big model, little data) = overfit.
# ---------------------------------------------------------------------------
print("\n3) EPOCHS (big model hidden=256, only 40 training points, Adam lr=0.01)")
tr, va, hist = train("adam", lr=0.01, batch_size=8, epochs=150, hidden=256, X=X_train[:40], y=y_train[:40], track=True)
print("   epoch   train loss   val loss")
for e in (0, 4, 19, 49, 99, 149):
    print(f"   {e + 1:5d}   {hist[e][0]:10.3f}   {hist[e][1]:8.3f}")
print("   Train loss keeps falling while val loss stops improving/rises = overfitting. Stop at the val minimum.")
```

- **In simple words:** Give a big network only 40 training points and train for 150 epochs, watching training and validation loss.
- **Exactly:** `hidden=256`, `X_train[:40]`, `batch_size=8`, `epochs=150`, `track=True`. Real output: epoch 1 train 0.235 / val 0.280; epoch 5 0.127 / 0.274; epoch 20 0.066 / 0.300; epoch 50 0.038 / 0.521; epoch 100 0.029 / 0.684; epoch 150 0.035 / 0.512. Train loss keeps falling while validation loss rises from epoch about 5.
- **Why it is here:** The best epoch is the one at the validation minimum, so stop there (early stopping). Too many epochs on little data memorises it.

### 11. Experiment 4: optimisers  (`main.py` lines 95–102)

```python
# ---------------------------------------------------------------------------
# 4. Optimizers: different rules for turning gradients into steps.
# ---------------------------------------------------------------------------
print("\n4) OPTIMIZERS (same lr scale, 30 epochs)")
for name, lr in [("sgd", 0.01), ("sgd+momentum", 0.01), ("adam", 0.01), ("adamw", 0.01)]:
    tr, va, _ = train(name, lr=lr)
    print(f"   {name:13} train loss={tr:.3f}  val loss={va:.3f}")
print("   Adam/AdamW adapt a step size per weight; AdamW is the default for LLMs (decoupled weight decay).")
```

- **In simple words:** Same problem, four update rules.
- **Exactly:** Real output at lr 0.01: SGD train 0.390 / val 0.344; SGD with momentum 0.237 / 0.212; Adam 0.127 / 0.162; AdamW 0.131 / 0.163. Adam adapts a step size for each weight using running averages of the gradient and its square; momentum smooths direction.
- **Why it is here:** AdamW is the default for LLM fine-tuning because decoupled weight decay behaves better than Adam with L2.

### 12. Experiment 5: the same knob on a real LLM  (`main.py` lines 104–127)

```python
# ---------------------------------------------------------------------------
# 5. The same knob on a real LLM: fine-tune distilgpt2 for 8 steps at different learning rates
# ---------------------------------------------------------------------------
print("\n5) LEARNING RATE ON distilgpt2 (8 steps on 4 sentences, AdamW)")
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
texts = ["Arrr, ye scurvy dog, hand over the treasure!", "Avast, matey! The sea be calm and the rum be plentiful.",
         "Hoist the sails, me hearties, for fortune awaits!", "Shiver me timbers, that be the finest ship afloat."]
batch = tok(texts, return_tensors="pt", padding=True)   # input_ids, attention_mask: shape (n_texts, longest_text)
labels = batch["input_ids"].clone()   # shape: (n_texts, longest_text)
labels[batch["attention_mask"] == 0] = -100                         # ignore padding in the loss
for lr in (1e-6, 5e-5, 5e-3):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(8):
        loss = model(**batch, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(round(loss.item(), 2))
    print(f"   lr={lr:<7} losses per step: {losses}")
print("   1e-6: crawls. 5e-5: healthy. 5e-3: jumps around / damages the pretrained knowledge. LLM fine-tuning")
print("   uses LOW learning rates (1e-5..5e-4) because the weights are already good - we only nudge them.")
```

- **In simple words:** Fine-tune distilgpt2 for 8 steps on four sentences at three learning rates and watch the loss.
- **Exactly:** `batch` is `(4, longest_text)`; `labels` copy `input_ids` with padding set to `-100` so padding is ignored. For each lr the model is reloaded fresh, `AdamW` is created and eight steps of forward, `backward`, `step`, `zero_grad` are run. Real output: lr 1e-6 gives `[6.49, 6.39, 6.27, 6.25, 6.28, 6.29, 6.21, 6.11]` (crawls); 5e-5 gives `[6.49, 5.76, 5.01, 4.58, 4.18, 3.87, 3.42, 3.13]` (healthy); 5e-3 gives `[6.49, 6.26, 6.81, 15.54, 7.12, 6.39, 5.09, 4.69]` (jumps to 15.54 at step 4 before recovering, a sign of damaging the pretrained weights). All three start at 6.49 because the model and seed are the same.
- **Why it is here:** Pretrained weights are already good, so fine-tuning uses small learning rates, and a too-large one damages what the model knew.

### 13. Experiments and break-it comments  (`main.py` lines 129–132)

```python
# TRY IT: in section 1 add lr=0.3 - find the largest lr that still trains smoothly.
# TRY IT: in section 3 use hidden=8 - does overfitting go away? (smaller model = less capacity)
# BREAK IT: delete opt.zero_grad() in train() and compare the loss: gradients accumulate across batches.
# BREAK IT: make the batch size 0 -> `range(0, len(X), 0)` raises ValueError: range() arg 3 must not be zero.
```

- **In simple words:** Ideas: find the largest stable learning rate, shrink the model, and remove `zero_grad`.
- **Exactly:** `range(0, len(X), 0)` raises `ValueError: range() arg 3 must not be zero`.
- **Why it is here:** Reading the exact message lets you spot the cause quickly.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `opts = {"sgd": lambda: ..., "adam": lambda: ...}` | dict of small anonymous functions - build only the chosen optimizer | `f = lambda x: x + 1` |
| `for i in range(0, len(X), batch_size):` | range with a step: jump by batch size | `range(0, 10, 4)` -> 0, 4, 8 |
| `-(-300 // bs)` | ceiling division trick | `-(-300 // 128)` = 3 |
| `perm = torch.randperm(n)` ; `X[perm[i:i+bs]]` | shuffle then slice indices | - |
| `tr != tr` | True only for NaN | `float("nan") != float("nan")` |
| `def train(optimizer_name="adam", lr=0.01, ...)` | many default arguments -> easy one-at-a-time experiments | `train(lr=0.1)` |
| `loss.backward(); opt.step(); opt.zero_grad()` | `;` puts statements on one line | - |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `sklearn.datasets.make_moons` | scikit-learn | synthetic 2-class 2-D data | `n_samples`, `noise`, `random_state` | X, y (numpy) |
| `nn.Sequential(layers...)` | torch | chain layers | - | model |
| `nn.Linear(in, out)` | torch | fully connected layer | sizes | layer |
| `nn.ReLU()` | torch | `max(0, x)` non-linearity | - | layer |
| `nn.CrossEntropyLoss()` | torch | softmax + negative log-likelihood on logits | - | loss function |
| `torch.optim.SGD / Adam / AdamW` | torch | optimizers | `lr`, `momentum`, `weight_decay` | optimizer |

## The knobs
| Knob | Meaning | Too low | Too high | Memory/speed link |
|---|---|---|---|---|
| **learning rate** | step size of an update | learns very slowly | overshoots, loss explodes | none |
| **batch size** | examples per update | noisy gradients, many slow updates | few updates, may generalise worse | **memory** (activations) |
| **epochs** | passes over the data | underfit | overfit (train loss falls, val loss rises) | time |
| **optimizer** | how gradients -> steps | SGD: simple, slow | Adam/AdamW: adaptive, default for LLMs | extra memory: Adam keeps 2 extra numbers per weight |
| **loss function** | what "wrong" means | - | - | cross-entropy for classification AND next-token prediction |

## Execution trace
```
1) LEARNING RATE (SGD, batch 32, 30 epochs)
   lr=0.0001   train loss=0.734  too small: barely moves
   lr=0.01     train loss=0.390  ok
   lr=0.1      train loss=0.229  good
   lr=10.0     train loss=2.859  too large: loss BLOWS UP

2) BATCH SIZE: batch=4 -> 75 updates/epoch, 0.56 s   |  batch=300 -> 1 update/epoch, 0.01 s
   final losses are all close (0.12-0.14): batch size mostly trades speed, noise and memory.

3) EPOCHS (40 training points, hidden=256)
   epoch   train loss   val loss
       1        0.235      0.280
       5        0.127      0.274
      20        0.066      0.300
      50        0.038      0.521
     100        0.029      0.684      <- train keeps falling, val gets WORSE = overfitting

4) OPTIMIZERS: sgd 0.390 | sgd+momentum 0.237 | adam 0.127 | adamw 0.131   (same lr=0.01)

5) LEARNING RATE ON distilgpt2 (8 steps):
   lr=1e-06   [6.49, 6.39, 6.27, 6.25, 6.28, 6.29, 6.21, 6.11]      crawls
   lr=5e-05   [6.49, 5.76, 5.01, 4.58, 4.18, 3.87, 3.42, 3.13]      healthy
   lr=0.005   [6.49, 6.26, 6.81, 15.54, 7.12, 6.39, 5.09, 4.69]     spikes to 15.5
```
- **Why fine-tuning LRs are tiny (1e-5 to 5e-4)**: the weights are already good; big steps damage ("catastrophically forget") pretrained knowledge.
- **Epochs and datasets:** for LLM fine-tuning 1-3 epochs is typical; more epochs on a small dataset memorises it (step 09 of module 04 showed loss 0.5 on 4 sentences).
- **Batch size on GPUs:** you pick the largest that fits in memory; if too big, use *gradient accumulation* (several small batches before one update). Covered in module 06.
- **Loss functions:** regression -> mean squared error; classification and LLM next-token prediction -> cross-entropy (see [03/06_softmax_cross_entropy](../../03_ml_foundations/06_softmax_cross_entropy/)).

## Experiments
- Add `lr=0.3` to the SGD sweep; find the largest stable value.
- Section 3 with `hidden=8`: overfitting should shrink (less capacity).
- Section 5 with `lr=5e-4` and 20 steps: smooth or not?

## Debugging practice
Delete `opt.zero_grad()` inside `train`: no traceback, but the training curve becomes worse/erratic since gradients accumulate. How would you *detect* this? Print the loss per step, compare with a version that works, and write down the hypothesis ("gradients accumulate") before changing code. Also try `batch_size=0`: `ValueError: range() arg 3 must not be zero` - a precise, readable error.

## Rebuild checklist
- [ ] Write the batch loop: shuffle, slice batch, forward, loss, backward, step, zero_grad.
- [ ] Explain epoch vs iteration vs batch.
- [ ] Draw the train/val loss curves of overfitting and mark where to stop.
- [ ] Explain why Adam uses more memory than SGD.

## The new problem (cliffhanger)
You can tune each knob, but impressions are not evidence: "it felt better" does not survive a code review. We need a record of every run. Next: [step 07, the experiment logger](../07_experiment_logger/WALKTHROUGH.md).

> **Say it aloud (interview box):** The learning rate is the step size: too small and nothing is learned, too large and the loss explodes, and fine-tuning wants small values. Batch size trades gradient noise against memory, and more epochs on a small dataset means memorising it. I change one knob at a time and read the loss curve to know which one is wrong.
