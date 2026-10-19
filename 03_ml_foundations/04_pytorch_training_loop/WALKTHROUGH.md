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
Only the model, the data and the loss change. The same five lines train this 3-parameter model, a CNN, and an LLM during fine-tuning (module [06](../../06_llm_training_and_fast_inference/)).

## Goal
Write the canonical loop: `DataLoader` → `model(x)` → `loss_fn` → `zero_grad` → `backward` → `step`; evaluate on validation.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/04_pytorch_training_loop/main.py
```

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
