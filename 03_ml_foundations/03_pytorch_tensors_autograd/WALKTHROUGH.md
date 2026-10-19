# Step 03 — PyTorch tensors and autograd

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

## Say it aloud
"A tensor is a numpy-like array that can live on a GPU. With `requires_grad=True` PyTorch records the operations, and `loss.backward()` computes every parameter's gradient automatically, so I don't derive formulas. Gradients accumulate by default, so I clear them each step."
