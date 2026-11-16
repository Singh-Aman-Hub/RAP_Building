# Step 04 — The PyTorch training loop

## The story so far
Step 03 gave us automatic gradients but a hand-rolled update. Here we package the same five moves into the standard PyTorch skeleton.

## The pain point
Updating each parameter by hand does not scale to many layers, many batches, or swapping in a better optimizer such as Adam. Every project would reinvent the same loop slightly differently and slightly wrong.

## The spark
Let `nn.Module` own the parameters, `DataLoader` serve the batches, and the optimizer do the update. What is left is five lines, always in this order: forward, loss, `zero_grad`, `backward`, `step`.

## The universal training loop
The five lines inside the batch loop are the whole secret of training:
```python
pred = model(xb)               # 1. forward
loss = loss_fn(pred, yb)        # 2. loss
optimizer.zero_grad()           # 3. clear old gradients
loss.backward()                 # 4. new gradients (autograd)
optimizer.step()                # 5. update the parameters
```
Only the model, the data and the loss change. The same five lines train this 3-parameter model, a CNN, and an LLM during fine-tuning (module [06](../../06_llm_training_and_lora/)).

## Goal
Write the canonical loop: `DataLoader` → `model(x)` → `loss_fn` → `zero_grad` → `backward` → `step`; evaluate on validation.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/04_pytorch_training_loop/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `nn.Module` | model, layer, network | Roughly | The base class: every layer and every whole model is a Module that owns parameters and a `forward` |
| `nn.Linear` | dense layer, fully connected layer, affine layer | Yes | `y = x @ W.T + b`; "dense" is the Keras name |
| Optimizer | SGD, Adam, AdamW | SGD/Adam are optimizers | The optimizer object holds the update rule; SGD is `w -= lr*grad`, Adam adds momentum and per-parameter scaling |
| `DataLoader` | data pipeline, batcher | Roughly | Wraps a Dataset to yield shuffled mini-batches |
| `model.train()` vs `model.eval()` | training mode vs inference mode | No | Switches layers like dropout and batch-norm; unrelated to `torch.no_grad()`, which stops gradient tracking |
| `zero_grad` | clear gradients | Yes | Resets `.grad` of every parameter before `backward()` |
| Logits / prediction | raw output, scores | Here yes | Output of the model before any activation like softmax (here a plain regression value) |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–9)

```python
"""
GOAL: The standard PyTorch training loop: Dataset/DataLoader -> model (nn.Module) -> loss -> optimizer
      -> forward / backward / step.  This SAME skeleton trains an LLM in module 06.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/04_pytorch_training_loop/main.py
EXPECTED OUTPUT: epoch-by-epoch train/validation loss falling, learned weights near [2, -3] and bias 1.
NEEDS: torch
"""
```

- **In simple words:** Says this file is the standard PyTorch training recipe.
- **Exactly:** Docstring only. The promise: train/val loss fall and the weights approach [2, -3] with bias 1.
- **Why it is here:** Gives the goal before the code.

### 2. Imports and a fixed random seed  (`main.py` lines 11–15)

```python
import torch
from torch import nn                                     # nn = neural-network building blocks
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(0)                                     # reproducibility
```

- **In simple words:** Load PyTorch's tools and make the random numbers repeatable.
- **Exactly:** `nn` has layers and losses; `DataLoader`, `TensorDataset`, `random_split` handle batching and splitting. `torch.manual_seed(0)` seeds the global CPU generator, which affects `randn`, the weight initialisation of `nn.Linear`, `random_split` and `shuffle=True`.
- **Why it is here:** Without a seed every run gives slightly different numbers, so you cannot compare experiments.

### 3. Create 500 samples with a known rule  (`main.py` lines 17–19)

```python
# --- 1. Data: y = 2*x1 - 3*x2 + 1 + noise ---------------------------------
X = torch.randn(500, 2)                                  # shape: (500, 2) = (samples, features) | 500 samples, 2 features, standard normal
y = (X @ torch.tensor([2.0, -3.0]) + 1.0 + 0.1 * torch.randn(500)).unsqueeze(1)  # unsqueeze -> shape: (500, 1)
```

- **In simple words:** Invent data so that the right answer is known: y = 2*x1 - 3*x2 + 1 plus a little noise.
- **Exactly:** `X = torch.randn(500, 2)` is `(500, 2)` standard normal. `X @ tensor([2., -3.])` is `(500, 2) @ (2,) -> (500,)`. Adding `1.0` and `0.1 * randn(500)` (noise std 0.1) keeps `(500,)`. `.unsqueeze(1)` inserts an axis: `(500,) -> (500, 1)`, which matches the model output shape (a column per target).
- **Why it is here:** `nn.MSELoss` compares tensors of the same shape. A `(32,)` vs `(32, 1)` mismatch would silently broadcast to `(32, 32)` and give a wrong loss.

### 4. Split and wrap in DataLoaders  (`main.py` lines 20–23)

```python
dataset = TensorDataset(X, y)                            # pairs each X row with its y
train_set, val_set = random_split(dataset, [400, 100])   # 80/20 split
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)   # yields batches of 32, reshuffled each epoch
val_loader = DataLoader(val_set, batch_size=100)
```

- **In simple words:** Pair each row with its label, split 400/100, and set up machines that hand out batches.
- **Exactly:** `TensorDataset(X, y)` indexes to `(X[i], y[i])`. `random_split(dataset, [400, 100])` gives two `Subset` objects. `train_loader` yields 400/32 = 13 batches per epoch (12 of 32 and one of 16) and reshuffles each epoch. `val_loader` uses one batch of 100 (no shuffle).
- **Why it is here:** Batching is what makes training efficient and gives noisy-but-fast updates (step 02). A separate validation set tells you whether you generalise.

### 5. Define the model as an nn.Module  (`main.py` lines 25–32)

```python
# --- 2. Model --------------------------------------------------------------
class LinearModel(nn.Module):                            # every model subclasses nn.Module
    def __init__(self):
        super().__init__()                               # required: sets up PyTorch's parameter tracking
        self.linear = nn.Linear(2, 1)                    # 2 inputs -> 1 output: weight (1,2) + bias (1,)

    def forward(self, x):                                # defines the forward pass; called via model(x)
        return self.linear(x)
```

- **In simple words:** A tiny model: take 2 numbers in, produce 1 number out, using one learned linear layer.
- **Exactly:** `super().__init__()` initialises Module's bookkeeping so assigned layers are registered as parameters. `nn.Linear(2, 1)` creates `weight` of shape `(1, 2)` and `bias` of shape `(1,)`, 3 parameters in total (the printed `number of parameters: 3`). `forward(x)` maps `(batch, 2) -> (batch, 1)` as `x @ W.T + b`. You call `model(x)`, not `model.forward(x)`, so hooks run.
- **Why it is here:** This class shape (init declares layers, forward uses them) is the template for every network, including GPT-style models in module 04.

### 6. Create the model, loss function and optimizer  (`main.py` lines 34–36)

```python
model = LinearModel()
loss_fn = nn.MSELoss()                                   # the loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # updates the parameters. Try torch.optim.Adam too.
```

- **In simple words:** Build the model, pick how to score mistakes, and pick the rule that updates the numbers.
- **Exactly:** `nn.MSELoss()` computes `mean((pred - target)**2)`. `torch.optim.SGD(model.parameters(), lr=0.05)` stores references to the 3 parameters and will do `p -= 0.05 * p.grad` on `step()`. The weights start random (from the seed).
- **Why it is here:** The optimizer must be given the parameters; otherwise `step()` changes nothing.

### 7. Outer loop over epochs  (`main.py` lines 38–41)

```python
# --- 3. Training loop ------------------------------------------------------
# The 5 lines in the inner loop (forward, loss, zero_grad, backward, step) are the UNIVERSAL training loop:
# they train this 3-parameter model and, unchanged, fine-tune an LLM.
for epoch in range(1, 11):
```

- **In simple words:** Go through the whole training set 10 times.
- **Exactly:** `range(1, 11)` gives epochs 1..10 for human-friendly printing. With 13 batches per epoch there are 130 optimizer steps in total.
- **Why it is here:** One pass is rarely enough; the first epoch's validation loss is 0.5431 and it drops below 0.012 by epoch 3.

### 8. The five-line inner loop: forward, loss, zero, backward, step  (`main.py` lines 42–48)

```python
    model.train()                                        # training mode (matters for dropout/batchnorm)
    for xb, yb in train_loader:                          # xb: (32, 2), yb: (32, 1) | one iteration = one batch
        pred = model(xb)                                 # shape: (32, 1) | forward
        loss = loss_fn(pred, yb)                         # compare with truth
        optimizer.zero_grad()                            # clear old gradients
        loss.backward()                                  # compute new gradients
        optimizer.step()                                 # update parameters
```

- **In simple words:** For every batch: guess, score, wipe old gradients, work out new gradients, move the weights.
- **Exactly:** `model.train()` sets training mode. `xb` is `(32, 2)`, `yb` is `(32, 1)`, `pred` is `(32, 1)`, `loss` is a scalar. `zero_grad()` sets each `.grad` to zero (or None), `loss.backward()` fills them and `optimizer.step()` applies SGD. Order matters: zero before backward.
- **Why it is here:** This exact skeleton fine-tunes an LLM in module 06. Remove `zero_grad` and gradients add up across batches, making training unstable.

### 9. Evaluate train and validation loss without gradients  (`main.py` lines 50–54)

```python
    model.eval()                                         # evaluation mode
    with torch.no_grad():                                # no gradients needed -> faster, less memory
        train_loss = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in train_loader) / len(train_set)
        val_loss = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in val_loader) / len(val_set)
    print(f"epoch {epoch:2d}  train loss={train_loss:.4f}  val loss={val_loss:.4f}")
```

- **In simple words:** After each epoch, switch to exam mode and measure the loss on both sets.
- **Exactly:** `model.eval()` turns off dropout and uses running stats in batch-norm (no effect for this linear model but a habit). `torch.no_grad()` skips graph building, saving memory. The generator expression weights each batch's mean loss by its size (`* len(xb)`) and divides by the set size, giving the true average even when the last batch is smaller. Real output epoch 1: `train loss=0.5938 val loss=0.5431`; epoch 10: `0.0089` and `0.0112`.
- **Why it is here:** Watching both curves is how you detect overfitting (step 05). The loss plateaus near 0.01 which is the noise floor (0.1 squared).

### 10. Inspect the learned parameters  (`main.py` lines 56–57)

```python
print("learned weight:", model.linear.weight.data.flatten().tolist(), "bias:", model.linear.bias.item())
print("number of parameters:", sum(p.numel() for p in model.parameters()))   # LLMs have billions of these
```

- **In simple words:** Print what the model learned and how many numbers it has.
- **Exactly:** `weight.data.flatten().tolist()` converts the `(1,2)` tensor to `[2.002, -2.9999]`; `bias.item()` gives `1.0064`. True values 2, -3, 1. `numel()` counts elements per parameter: 2 + 1 = 3. A GPT-2 small has about 124 million.
- **Why it is here:** Confirms the training worked and shows the model is just a handful of numbers, scaled up in LLMs.

### 11. TRY IT and BREAK IT comments  (`main.py` lines 58–60)

```python
# TRY IT: swap SGD for torch.optim.Adam(model.parameters(), lr=0.05); compare epoch-1 loss.
# TRY IT: batch_size=400 vs 8. Count how many optimizer.step() calls happen per epoch.
# BREAK IT: delete optimizer.zero_grad() -- training gets unstable because gradients accumulate across batches.
```

- **In simple words:** Suggested experiments.
- **Exactly:** Swap in Adam, change batch size to 400 (1 step per epoch) or 8 (50 steps), or delete `zero_grad()` to see gradient accumulation.
- **Why it is here:** Changing one thing at a time is how you build intuition for hyperparameters.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class LinearModel(nn.Module):` | class inheriting behaviour from `nn.Module` | |
| `super().__init__()` | run the parent's constructor | mandatory in `nn.Module` subclasses |
| `def forward(self, x)` | the computation; call it with `model(x)` | |
| `for xb, yb in loader:` | tuple unpacking per batch | |
| generator in `sum(... for ...)` | weighted average of batch losses | |
| `.unsqueeze(1)` | add a dimension: (500,) → (500,1) | |

## Libraries & functions
| Name | Origin | Does | Important args |
|---|---|---|---|
| `nn.Linear(in, out)` | torch | y = xWᵀ + b layer | in/out features |
| `nn.MSELoss()` | torch | mean squared error | |
| `torch.optim.SGD(params, lr)` | torch | gradient descent updater | `lr`, `momentum` |
| `TensorDataset`, `random_split` | torch.utils.data | pair tensors; split | lengths |
| `DataLoader(ds, batch_size, shuffle)` | torch.utils.data | batches, shuffles | `batch_size`, `shuffle` |
| `optimizer.zero_grad/step`, `loss.backward` | torch | clear / apply / compute gradients | |
| `model.train()` / `.eval()` | torch | training vs inference mode | |
| `torch.manual_seed` | torch | reproducibility | seed |

## Execution trace (real run)
Input: 500×2 features, true rule `2·x1 − 3·x2 + 1`. 400 train / 100 val, 13 batches per epoch.
```
epoch  1  train loss=0.5938  val loss=0.5431
epoch  2  train loss=0.0414  val loss=0.0393
epoch  3  train loss=0.0106  val loss=0.0126
epoch 10  train loss=0.0089  val loss=0.0112
learned weight: [2.002..., -2.9999...] bias: 1.0063
number of parameters: 3
```
Loss floor ≈ noise variance 0.01. Train ≈ validation → no overfitting.

## Experiments
1. `torch.optim.Adam(..., lr=0.05)`.
2. `batch_size=400` (1 step/epoch) vs `8` (50 steps/epoch).
3. Add a hidden layer: `nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,1))`; count parameters (33).

## Debugging practice
Pass `y` of shape `(500,)` instead of `(500,1)` to `MSELoss`: you get `UserWarning: Using a target size (torch.Size([500])) that is different to the input size (torch.Size([500, 1]))...` (a broadcasting *warning*, not an error) and a wrong loss — shape bugs may not crash. Print `pred.shape, yb.shape`.

## Rebuild checklist
- [ ] Write the 5-line loop body from memory in order.
- [ ] Explain train vs eval mode and `no_grad`.
- [ ] Count parameters of `nn.Linear(2,1)`.

## The new problem (cliffhanger)
The loop drives training loss toward the noise floor. But here the validation set looked just like the training set. What happens when the model is flexible enough to memorise instead of learn? Does it understand, or does it memorise? Next: [05_overfitting_train_val](../05_overfitting_train_val/).

## Say it aloud
> **Say it aloud (interview box):** Each step of the loop is: the model predicts, the loss scores it, `zero_grad` clears old gradients, `backward` computes new ones, and the optimizer steps the weights. I evaluate on held-out data in `eval` mode under `no_grad`. These five lines are the universal training loop, and fine-tuning an LLM reuses them unchanged with a bigger model and a text loss.
