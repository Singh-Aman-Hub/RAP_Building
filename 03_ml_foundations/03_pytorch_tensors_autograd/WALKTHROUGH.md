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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Tensor | n-d array, ndarray (numpy) | Nearly | A tensor is a numpy-like array that can live on a GPU and remember its operations for autograd |
| Autograd | automatic differentiation, reverse-mode AD, backprop | Close | Autograd is the PyTorch engine; backpropagation is the algorithm it runs when you call `.backward()` |
| `requires_grad` | "track me", trainable flag | Yes | Switch that makes PyTorch record operations on that tensor |
| `.grad` | gradient, dL/dw | Yes | The stored derivative after `backward()` |
| `torch.no_grad()` | inference mode, `detach()` | Related | `no_grad` disables recording inside a block; `inference_mode` is a stricter, faster version; `detach` cuts one tensor out of the graph |
| mps / cuda / cpu | device, backend | No | Different hardware a tensor can live on (Apple GPU, NVIDIA GPU, processor) |
| Forward / backward pass | inference / backprop | Roughly | Forward computes outputs and loss; backward computes gradients |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–9)

```python
"""
GOAL: Learn PyTorch's two core ideas: tensors (numpy-like arrays) and autograd
      (automatic gradients) -- replacing the hand-written derivatives from step 02.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/03_pytorch_tensors_autograd/main.py
EXPECTED OUTPUT: tensor shapes, d(loss)/dw values that match the hand formula, a few manual updates.
NEEDS: torch
"""
```

- **In simple words:** Says the step replaces hand-written derivatives with PyTorch's automatic ones.
- **Exactly:** Docstring only; no runtime effect.
- **Why it is here:** States the goal: the same learning as step 02 with less hand maths.

### 2. Import numpy and torch  (`main.py` lines 11–12)

```python
import numpy as np
import torch  # third-party library: tensors + autograd + neural-network building blocks
```

- **In simple words:** Bring in numpy (for conversion demo) and PyTorch.
- **Exactly:** `torch` provides `Tensor`, autograd, devices and the `torch.nn` building blocks used in step 04 and in every later module.
- **Why it is here:** PyTorch is the framework that Hugging Face models (modules 04 and 06) run on.

### 3. Create tensors and convert to and from numpy  (`main.py` lines 14–18)

```python
# --- Tensors ---------------------------------------------------------------
a = torch.tensor([1.0, 2.0, 3.0])               # shape: (3,) | create from a Python list
b = torch.zeros(3)                              # 3 zeros
print("a + 2 =", a + 2, "| a * a =", a * a, "| dtype:", a.dtype, "| shape:", a.shape)
print("from numpy:", torch.from_numpy(np.array([1.0, 2.0])), "| back:", a.numpy())
```

- **In simple words:** A tensor is an array. You can do arithmetic on it and swap it with numpy.
- **Exactly:** `torch.tensor([1.,2.,3.])` has shape `(3,)` and default dtype `torch.float32`. `a + 2` broadcasts a scalar: `[3,4,5]`; `a * a` is element-wise `[1,4,9]`. `torch.from_numpy` shares memory with the numpy array and keeps its dtype (`float64` in the real output), while `a.numpy()` returns an array sharing memory with `a`. `torch.zeros(3)` creates `b`, unused later.
- **Why it is here:** Numpy-to-torch conversion is constant in practice. Remember torch defaults to float32 but numpy to float64, which can cause dtype mismatch errors.

### 4. Pick the best device  (`main.py` lines 20–22)

```python
# Devices: tensors live on cpu / mps (Apple GPU) / cuda (NVIDIA). Operands must be on the SAME device.
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("best available device:", device, "(we use cpu in this module to keep things simple)")
```

- **In simple words:** Check whether a faster chip is available.
- **Exactly:** A conditional expression chains three choices. On this machine `torch.backends.mps.is_available()` is True so the printed device is `mps`, but nothing is moved to it; later tensors stay on CPU. Operands of one operation must be on the same device or PyTorch raises a RuntimeError.
- **Why it is here:** Teaches the device habit you need for LLMs in module 04 and beyond.

### 5. Create data and two trainable parameters  (`main.py` lines 24–28)

```python
# --- Autograd: PyTorch records operations and can differentiate them -------
x = torch.tensor([1.0, 2.0, 3.0, 4.0])          # shape: (4,)
y = torch.tensor([3.0, 5.0, 7.0, 9.0])          # rule: y = 2x + 1
w = torch.tensor(0.0, requires_grad=True)       # requires_grad=True -> "track me, I'm a parameter"
bias = torch.tensor(0.0, requires_grad=True)
```

- **In simple words:** Four data points on the rule y = 2x + 1 and two numbers w and bias that PyTorch should be allowed to adjust.
- **Exactly:** `x` is `(4,)`, `y` is `(4,)`. `w` and `bias` are 0-dim tensors (shape `()`), float32, with `requires_grad=True`, so they become leaf nodes of the computation graph.
- **Why it is here:** Without `requires_grad=True`, `loss.backward()` raises an error and `w.grad` stays `None`.

### 6. Forward pass and loss  (`main.py` lines 30–32)

```python
y_hat = w * x + bias                            # shape: (4,) | forward pass (builds a computation graph)
loss = ((y_hat - y) ** 2).mean()                # shape: () scalar | MSE as a single number
print("loss:", loss.item())                     # .item() -> plain Python float
```

- **In simple words:** Predict, then turn the prediction errors into one number, as in step 02.
- **Exactly:** `w * x + bias` gives shape `(4,)` (the scalar w broadcasts). Each operation is recorded in a graph. `loss` has shape `()` and `.item()` converts it to a Python float. With w=0, b=0: errors are [-3,-5,-7,-9], mean square = (9+25+49+81)/4 = 41.0, matching the real output `loss: 41.0`.
- **Why it is here:** The loss must be a scalar, since `backward()` with no argument only works on a scalar.

### 7. Backward pass: let PyTorch compute the gradients  (`main.py` lines 34–35)

```python
loss.backward()                                 # backward pass: fills w.grad and bias.grad
print("autograd  dL/dw =", w.grad.item(), " dL/db =", bias.grad.item())
```

- **In simple words:** One call walks the recorded graph backwards and fills in how much each parameter affects the loss.
- **Exactly:** `loss.backward()` applies the chain rule and stores results in `w.grad` and `bias.grad` (same shape as the parameters). Real values: `dL/dw = -35.0`, `dL/db = -12.0`. Negative gradients mean increasing w or bias would lower the loss.
- **Why it is here:** This replaces the hand-derived `grad_w`/`grad_b` of step 02 and scales to billions of parameters.

### 8. Verify against the hand formula  (`main.py` lines 37–39)

```python
# Check against the formula we used in numpy step 02: 2*mean(error*x), 2*mean(error)
error = (0.0 * x + 0.0) - y
print("by hand   dL/dw =", (2 * (error * x).mean()).item(), " dL/db =", (2 * error.mean()).item())
```

- **In simple words:** Check that autograd gives the same answer as the formulas from step 02.
- **Exactly:** With w=0, b=0 the error is `-y` = [-3,-5,-7,-9]. `2*mean(error*x)` = 2*(-3-10-21-36)/4 = -35.0; `2*mean(error)` = -12.0. Both match autograd exactly.
- **Why it is here:** Builds trust: autograd is not magic, it equals the maths.

### 9. The universal training loop, written by hand  (`main.py` lines 41–52)

```python
# --- A manual training loop with autograd ----------------------------------
# The 5 moves below (forward, loss, clear grads, backward, update) are the UNIVERSAL training loop: step 04 names them, and
# the same five lines fine-tune an LLM in module 06.
for step in range(200):
    y_hat = w * x + bias
    loss = ((y_hat - y) ** 2).mean()
    w.grad = None; bias.grad = None             # gradients ACCUMULATE by default; clear them each step!
    loss.backward()
    with torch.no_grad():                       # don't record the update itself in the graph
        w -= 0.05 * w.grad
        bias -= 0.05 * bias.grad
print(f"after 200 steps: w={w.item():.3f} bias={bias.item():.3f} loss={loss.item():.6f}")
```

- **In simple words:** Repeat 200 times: predict, measure loss, clear old gradients, compute new gradients, step downhill.
- **Exactly:** Five moves: forward (`y_hat`), loss, clear (`w.grad = None`), `backward()`, update inside `torch.no_grad()` with `w -= 0.05 * w.grad`. Gradients accumulate (are added to) on every `backward()` unless cleared. `no_grad` stops PyTorch from recording the in-place update (and from raising an in-place-on-leaf error). Real output: `after 200 steps: w=2.005 bias=0.986 loss=0.000034`.
- **Why it is here:** Step 04 puts a name on each of these five moves (`zero_grad`, `step`, ...) and the same five lines fine-tune an LLM in module 06.

### 10. TRY IT and BREAK IT comments  (`main.py` lines 53–54)

```python
# TRY IT: remove the `w.grad = None; bias.grad = None` line. Do you still converge? (gradients pile up -> trouble)
# BREAK IT: delete `with torch.no_grad():` and read the "a leaf Variable that requires grad is being used in an in-place operation" RuntimeError.
```

- **In simple words:** Two suggested experiments in comments.
- **Exactly:** Removing the clear line makes gradients pile up so the effective step grows and training misbehaves; deleting `torch.no_grad()` raises "a leaf Variable that requires grad is being used in an in-place operation".
- **Why it is here:** Reproducing the error once helps you recognise it in real training code.

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
