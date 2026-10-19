# Terms — PyTorch tensors and autograd

Folder: `03_ml_foundations/03_pytorch_tensors_autograd` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [ML foundations README](../README.md)

**Terms in this folder:** [PyTorch](#pytorch--third-party-torch) · [Tensor](#tensor--third-party-torch) · [torch.tensor() and torch.zeros()](#torchtensor-and-torchzeros--third-party-torch) · [dtype and shape of a tensor](#dtype-and-shape-of-a-tensor--third-party-torch) · [NumPy interoperability (`from_numpy`, `.numpy()`)](#numpy-interoperability-from_numpy-numpy--third-party-torch) · [Device (cpu / mps / cuda)](#device-cpu--mps--cuda--concept) · [torch.backends.mps.is_available()](#torchbackendsmpsis_available--third-party-torch) · [Inline conditional expression](#inline-conditional-expression--syntax) · [Autograd (automatic differentiation)](#autograd-automatic-differentiation--third-party-torch) · [requires_grad=True](#requires_gradtrue--third-party-torch) · [Computation graph](#computation-graph--concept) · [loss.backward()](#lossbackward--third-party-torch) · [.grad and gradient accumulation](#grad-and-gradient-accumulation--third-party-torch) · [.item()](#item--third-party-torch) · [torch.no_grad()](#torchno_grad--third-party-torch) · [with statement (context manager)](#with-statement-context-manager--syntax) · [Manual gradient-descent loop with autograd](#manual-gradient-descent-loop-with-autograd--concept) · [Mean (`.mean()`) on tensors](#mean-mean-on-tensors--third-party-torch) · [RuntimeError (in-place on leaf variable)](#runtimeerror-in-place-on-leaf-variable--concept)

---

## PyTorch  `(third-party: torch)`
**1. Definition:** PyTorch is an open-source library for tensors, automatic differentiation and neural-network building blocks, used to train most modern deep-learning models including LLMs.
**2. In plain words:** It is NumPy that can also run on a GPU and work out gradients for you. Hugging Face Transformers, used from module 04, is built on top of it.
**3. Example to quote:**
> "I use PyTorch because it computes gradients automatically, so I do not have to derive them by hand like in the NumPy step."

**Seen in:** `main.py`, `import torch`

## Tensor  `(third-party: torch)`
**1. Definition:** A tensor is PyTorch's multi-dimensional array, like a NumPy `ndarray`, that can live on a GPU and take part in automatic differentiation.
**2. In plain words:** A number, a list of numbers, a table, and a stack of tables are all tensors with different numbers of dimensions. Operations such as `a + 2` or `a * a` act element by element.
**3. Example to quote:**
> "`torch.tensor([1.0, 2.0, 3.0])` is a tensor; `a * a` gives `[1, 4, 9]`."

**Seen in:** `main.py`, `a = torch.tensor(...)`

## torch.tensor() and torch.zeros()  `(third-party: torch)`
**1. Definition:** `torch.tensor(data)` builds a tensor from Python data; `torch.zeros(n)` builds a tensor of n zeros.
**2. In plain words:** The two common ways to create tensors. Pass `requires_grad=True` to `torch.tensor` to mark a learnable parameter.
**3. Example to quote:**
> "`torch.zeros(3)` gives `[0, 0, 0]`, a handy starting value for parameters."

**Seen in:** `main.py`, `a`, `b`

## dtype and shape of a tensor  `(third-party: torch)`
**1. Definition:** `tensor.dtype` is the element data type (for example `torch.float32`), and `tensor.shape` is the size of each dimension.
**2. In plain words:** Many PyTorch errors come from mismatched dtype (int vs float) or shape. Check both with a print when something fails.
**3. Example to quote:**
> "`a.dtype` is `torch.float32` and `a.shape` is `torch.Size([3])`."

**Seen in:** `main.py`, first print

## NumPy interoperability (`from_numpy`, `.numpy()`)  `(third-party: torch)`
**1. Definition:** `torch.from_numpy(arr)` makes a tensor from a NumPy array and `tensor.numpy()` makes a NumPy array from a CPU tensor; they share the same memory.
**2. In plain words:** You can move between NumPy and PyTorch cheaply. Because memory is shared, changing one changes the other.
**3. Example to quote:**
> "I convert a NumPy array with `torch.from_numpy` and back with `.numpy()` without copying."

**Seen in:** `main.py`, from-numpy print

## Device (cpu / mps / cuda)  `(concept)`
**1. Definition:** The device is the hardware where a tensor's data lives and computations run: `cpu`, `mps` (Apple GPU) or `cuda` (NVIDIA GPU).
**2. In plain words:** All tensors in an operation must be on the same device, or PyTorch raises an error. This module stays on the CPU for simplicity; large models need a GPU for speed.
**3. Example to quote:**
> "On my Mac `torch.backends.mps.is_available()` is True, so the best device is `mps`. If tensors were on different devices, PyTorch would complain."

**Seen in:** `main.py`, `device = ...`

## torch.backends.mps.is_available()  `(third-party: torch)`
**1. Definition:** Returns True if PyTorch can use Apple's Metal Performance Shaders GPU backend on this machine.
**2. In plain words:** A check for "do I have an Apple GPU?". The line picks `mps`, then `cuda`, otherwise `cpu`.
**3. Example to quote:**
> "I choose the device with a fallback chain: MPS if available, otherwise CUDA, otherwise CPU."

**Seen in:** `main.py`, `device`

## Inline conditional expression  `(syntax)`
**1. Definition:** `A if cond else B` picks A when the condition is true, else B; they can be chained.
**2. In plain words:** The device line `"mps" if ... else "cuda" if ... else "cpu"` reads as a fallback ladder in a single expression.
**3. Example to quote:**
> "`'adult' if age >= 18 else 'minor'` is the same pattern I use to choose the device."

**Seen in:** `main.py`, `device`

## Autograd (automatic differentiation)  `(third-party: torch)`
**1. Definition:** Autograd is PyTorch's engine that records tensor operations in a computation graph and computes gradients of a scalar with respect to parameters automatically.
**2. In plain words:** You write the forward calculation; PyTorch works out the derivatives. It matched the hand-derived NumPy formula exactly (`-35.0` and `-12.0`).
**3. Example to quote:**
> "I never wrote the derivative. `loss.backward()` computed `dL/dw = -35.0`, identical to my hand formula."

**Seen in:** `main.py`, autograd section

## requires_grad=True  `(third-party: torch)`
**1. Definition:** A flag on a tensor telling PyTorch to track operations on it so gradients can be computed for it.
**2. In plain words:** It marks a tensor as "a parameter I want to learn". Inputs and labels do not need it.
**3. Example to quote:**
> "`w = torch.tensor(0.0, requires_grad=True)` tells PyTorch that `w` is a learnable parameter."

**Seen in:** `main.py`, `w`, `bias`

## Computation graph  `(concept)`
**1. Definition:** A computation graph is the record of operations that produced a tensor, which autograd walks backwards to compute gradients.
**2. In plain words:** Each time you compute `w * x + bias` with tracked tensors, PyTorch remembers the steps. That is why updates must be done inside `torch.no_grad()`: you do not want to record the update itself.
**3. Example to quote:**
> "The forward pass builds a graph from `w` and `bias` to the loss, and `backward()` follows it in reverse."

**Seen in:** `main.py`, comments

## loss.backward()  `(third-party: torch)`
**1. Definition:** `loss.backward()` computes the gradient of the scalar `loss` with respect to every tracked tensor and stores it in each tensor's `.grad`.
**2. In plain words:** It is the "backward pass". After calling it, `w.grad` tells you how to change `w`. Gradients add up across calls, so you must clear them each step.
**3. Example to quote:**
> "After `loss.backward()`, `w.grad` is -35.0, which says increasing `w` will reduce the loss."

**Seen in:** `main.py`, `loss.backward()`

## .grad and gradient accumulation  `(third-party: torch)`
**1. Definition:** `tensor.grad` holds the gradient computed by `backward()`; PyTorch adds new gradients to it rather than overwriting.
**2. In plain words:** If you forget to reset it (`w.grad = None`, or `optimizer.zero_grad()` later), old gradients pile up and updates become wrong. This is the most common PyTorch beginner bug.
**3. Example to quote:**
> "I set `w.grad = None` before each `backward()` so gradients from the previous step do not accumulate."

**Seen in:** `main.py`, manual loop

## .item()  `(third-party: torch)`
**1. Definition:** `tensor.item()` converts a one-element tensor into a plain Python number.
**2. In plain words:** Useful for printing or logging a loss. Calling it on a multi-element tensor raises an error.
**3. Example to quote:**
> "`loss.item()` gives the plain float `41.0` for printing."

**Seen in:** `main.py`, loss prints

## torch.no_grad()  `(third-party: torch)`
**1. Definition:** `with torch.no_grad():` is a context manager that disables gradient tracking for the operations inside it.
**2. In plain words:** Updating a tracked parameter in place without it raises a `RuntimeError`, because you would be modifying a leaf variable that needs gradients. It is also used at inference to save memory and time.
**3. Example to quote:**
> "I apply the update `w -= 0.05 * w.grad` inside `torch.no_grad()` so PyTorch does not record the update."

**Seen in:** `main.py`, manual training loop

## with statement (context manager)  `(syntax)`
**1. Definition:** `with obj:` runs a block between the object's enter and exit actions, guaranteeing cleanup or restoring state afterwards.
**2. In plain words:** The same keyword opens files safely. Here it temporarily switches off gradient tracking and automatically switches it back on after the block.
**3. Example to quote:**
> "`with torch.no_grad():` turns tracking off only for the indented block."

**Seen in:** `main.py`, update step

## Manual gradient-descent loop with autograd  `(concept)`
**1. Definition:** A hand-written training loop that, per step, computes predictions and loss, calls `backward()`, updates parameters under `no_grad`, and resets gradients.
**2. In plain words:** It reproduces step 02 but with automatic gradients: 200 steps took `w` to 2.005 and `bias` to 0.986 for the rule `y = 2x + 1`. Next step wraps this in higher-level PyTorch tools.
**3. Example to quote:**
> "After 200 steps the weights are `w=2.005` and `bias=0.986`, close to the true 2 and 1."

**Seen in:** `main.py`, `for step in range(200)`

## Mean (`.mean()`) on tensors  `(third-party: torch)`
**1. Definition:** `tensor.mean()` returns the average of all elements as a scalar tensor.
**2. In plain words:** Turns the squared errors into a single loss number, which `backward()` requires.
**3. Example to quote:**
> "`((y_hat - y) ** 2).mean()` is my MSE loss as a single tensor."

**Seen in:** `main.py`, `loss`

## RuntimeError (in-place on leaf variable)  `(concept)`
**1. Definition:** PyTorch raises a `RuntimeError` when you modify, in place, a tensor that requires gradients and is a leaf of the graph.
**2. In plain words:** Deleting `torch.no_grad()` from the update triggers it. The message says "a leaf Variable that requires grad is being used in an in-place operation". It is PyTorch protecting its graph.
**3. Example to quote:**
> "Without `no_grad`, `w -= ...` fails with a RuntimeError about an in-place operation on a leaf variable."

**Seen in:** `main.py`, BREAK IT comment
