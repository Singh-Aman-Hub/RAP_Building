# Step 03 — PyTorch tensors and autograd

## The story so far
Chapter 2 ended with a wall: gradients derived by hand do not scale past one formula. Chapter 3, [The Modern Engine](../../MASTER_READING_GUIDE.md), starts here: let the library compute them.

## The pain point
Hand-written gradients are slow to derive and easy to get wrong, and each new model needs a new derivation. Plain numpy also cannot run on a GPU.

## The spark
If every operation is recorded as it happens, the chain rule can be replayed backwards automatically. Mark a number `requires_grad=True`, compute a loss, call `loss.backward()`, and read `.grad`. We check it against our step 02 formula and it matches exactly. Look at the manual loop at the bottom of `main.py`: forward, loss, clear gradients, backward, update. Those are the five lines we name next.

## Goal
Replace hand-derived gradients with `loss.backward()`; know tensors, dtype, shape, device.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/03_pytorch_tensors_autograd/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `requires_grad=True` | track operations for differentiation | `torch.tensor(1., requires_grad=True)` |
| `with torch.no_grad():` | context manager: temporarily disable tracking | used for the manual update |
| `.item()` | single-value tensor → Python number | `loss.item()` |
| `w.grad = None` | reset accumulated gradient | |
| inline conditional | `"mps" if cond else "cuda" if cond2 else "cpu"` | |

## Libraries & functions
| Name | Origin | Does | Returns |
|---|---|---|---|
| `torch.tensor(data)` | PyTorch (3rd-party) | create tensor | Tensor |
| `torch.zeros(n)` | PyTorch | zeros | Tensor |
| `tensor.backward()` | PyTorch | compute gradients of a scalar | None (fills `.grad`) |
| `tensor.numpy()` / `torch.from_numpy` | PyTorch | convert to/from numpy (share memory) | |
| `torch.backends.mps.is_available()` | PyTorch | Apple GPU present? | bool |

## Execution trace (real run)
x=[1,2,3,4], y=[3,5,7,9], w=b=0 → y_hat all 0, loss 41.0.
```
loss: 41.0
autograd  dL/dw = -35.0  dL/db = -12.0
by hand   dL/dw = -35.0  dL/db = -12.0
after 200 steps: w=2.005 bias=0.986 loss=0.000034
```
The negative gradient says "increase w to reduce the loss". Autograd gave exactly the numpy formula's answer.

## Experiments
1. Add `torch.float64` dtype: `torch.tensor([1,2], dtype=torch.float64)`.
2. Lower lr to 0.005 and watch 200 steps become too few.
3. Remove `w.grad = None…` and observe the effect of accumulating gradients.

## Debugging practice
Delete `with torch.no_grad():` → `RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.` Meaning: updating a tracked parameter must happen outside the graph.

## Rebuild checklist
- [ ] Make a parameter tensor, compute loss, call `.backward()`, read `.grad`.
- [ ] Explain why you must zero gradients.
- [ ] Move a tensor between devices (`.to(device)`).

## The new problem (cliffhanger)
Autograd gives us gradients, but we are still hand-updating each parameter one by one. A real model has dozens of parameter tensors and data that arrives in batches. We need a tidy, reusable skeleton: [04_pytorch_training_loop](../04_pytorch_training_loop/).

## Say it aloud
> **Say it aloud (interview box):** A tensor is a numpy-like array that can live on a GPU, and with `requires_grad=True` PyTorch records the operations so `loss.backward()` fills in every parameter's gradient without me deriving anything. The result matched my hand formula exactly. Gradients accumulate by default, so I clear them every step, and the parameter update must happen under `no_grad`.
