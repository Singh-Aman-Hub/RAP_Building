# Terms — A tiny autograd engine

Folder: `15_neural_networks/03_tiny_autograd_engine` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Automatic differentiation (autodiff, autograd)](#automatic-differentiation-autodiff-autograd) · [Computational graph](#computational-graph) · [Reverse-mode differentiation](#reverse-mode-differentiation) · [Local derivative](#local-derivative) · [Closure](#closure) · [Topological sort](#topological-sort) · [Gradient accumulation (`+=`)](#gradient-accumulation-) · [Seeding the backward pass](#seeding-the-backward-pass) · [Zeroing gradients (`zero_grad`)](#zeroing-gradients-zerograd) · [Define-by-run (dynamic graph, eager mode)](#define-by-run-dynamic-graph-eager-mode) · [Leaf and parent nodes](#leaf-and-parent-nodes) · [Operator overloading (`__add__`, `__mul__`, `__radd__`)](#operator-overloading-add-mul-radd) · [Neuron, layer, MLP as objects](#neuron-layer-mlp-as-objects) · [Sum of squared errors loss](#sum-of-squared-errors-loss) · [TypeError: unsupported operand type(s) for ** or pow()](#typeerror-unsupported-operand-types-for--or-pow) · [Non-determinism from set ordering](#non-determinism-from-set-ordering)

---

## Automatic differentiation (autodiff, autograd)  `(concept)`
**1. Definition:** A technique that computes exact derivatives of a program by applying the chain rule to the sequence of elementary operations it executed.
**2. In plain words:** The computer differentiates your code for you; not symbolic algebra and not finite differences.
**3. Example to quote:** > "In PyTorch: `loss.backward()`."
**Seen in:** `main.py`

## Computational graph  `(concept)`
**1. Definition:** A directed acyclic graph whose nodes are values and whose edges point from inputs to the operations that used them.
**2. In plain words:** The family tree of a number: who it was computed from.
**3. Example to quote:** > "`_prev` holds the parents of a Value."
**Seen in:** `main.py`, block 1

## Reverse-mode differentiation  `(concept)`
**1. Definition:** Autodiff that propagates derivatives from the output to the inputs, giving the gradient of one output with respect to all inputs in one sweep.
**2. In plain words:** Start with 'the answer depends on itself by 1' and spread blame backwards.
**3. Example to quote:** > "Ideal when there are many parameters and one loss."
**Seen in:** `main.py`, `backward`

## Local derivative  `(concept)`
**1. Definition:** The derivative of a single operation's output with respect to its inputs, ignoring the rest of the graph.
**2. In plain words:** 'If I wiggle this input, how much does this one step wiggle?'
**3. Example to quote:** > "For `a*b`: `b` and `a`."
**Seen in:** `main.py`, block 2

## Closure  `(syntax)`
**1. Definition:** A function defined inside another function that keeps access to the outer function's variables after it returns.
**2. In plain words:** A function with a memory of where it was created.
**3. Example to quote:** > "Each `_backward` remembers `self`, `other`, `out`."
```python
def _backward():
    self.grad += other.data * out.grad
```
**Seen in:** `main.py`, block 2

## Topological sort  `(concept)`
**1. Definition:** An ordering of the nodes of a directed acyclic graph where every node comes after all its predecessors.
**2. In plain words:** Do the prerequisites first.
**3. Example to quote:** > "Reverse it to run backprop."
**Seen in:** `main.py`, `backward`

## Gradient accumulation (`+=`)  `(concept)`
**1. Definition:** Adding, not assigning, each incoming gradient contribution to a node's `.grad`.
**2. In plain words:** Several uses each report their blame; the totals add.
**3. Example to quote:** > "`y = x*x + x` gives 7.0."
**Seen in:** `main.py`, block 5

## Seeding the backward pass  `(concept)`
**1. Definition:** Setting `d output / d output = 1` to start the chain rule.
**2. In plain words:** The loss depends on itself exactly once.
**3. Example to quote:** > "`root.grad = 1.0`"
**Seen in:** `main.py`, `backward`

## Zeroing gradients (`zero_grad`)  `(concept)`
**1. Definition:** Resetting every `.grad` to 0 before a new backward pass because gradients accumulate.
**2. In plain words:** Wipe the whiteboard before the next calculation.
**3. Example to quote:** > "`p.grad = 0.0` in the loop; `optimizer.zero_grad()` in PyTorch."
**Seen in:** `main.py`, block 7

## Define-by-run (dynamic graph, eager mode)  `(concept)`
**1. Definition:** The graph is built while the code runs, so it can differ on every call.
**2. In plain words:** Plain Python control flow works inside models.
**3. Example to quote:** > "A fresh graph is built each training step."
**Seen in:** `main.py`, block 7

## Leaf and parent nodes  `(concept)`
**1. Definition:** Leaves have no inputs (parameters, data, constants); other nodes have parents recorded in `_prev`.
**2. In plain words:** The roots of the family tree.
**3. Example to quote:** > "`a, b, c`, the constants `1.0` and `5` are leaves."
**Seen in:** `main.py`, block 4

## Operator overloading (`__add__`, `__mul__`, `__radd__`)  `(syntax)`
**1. Definition:** Defining what `+`, `*` etc. do for your own class by implementing special methods.
**2. In plain words:** Lets `a * b + c` build the graph with ordinary syntax.
**3. Example to quote:** > "`Value.__add__ = _add`"
**Seen in:** `main.py`, block 2

## Neuron, layer, MLP as objects  `(concept)`
**1. Definition:** A neuron holds weights and a bias; a layer is a list of neurons; an MLP is a list of layers; `parameters()` collects every Value to train.
**2. In plain words:** `nn.Module`s, one scalar at a time.
**3. Example to quote:** > "`MLP([2, 4, 1])` has 17 parameters."
**Seen in:** `main.py`, block 6

## Sum of squared errors loss  `(concept)`
**1. Definition:** `sum((pred - target)^2)` over the samples; a regression loss.
**2. In plain words:** The total squared miss.
**3. Example to quote:** > "Used here instead of cross-entropy to keep the engine small."
**Seen in:** `main.py`, block 7

## TypeError: unsupported operand type(s) for ** or pow()  `(error)`
**1. Definition:** Python cannot apply `**` between the given types.
**2. In plain words:** You asked a float to be raised to a Value; our `_pow` handles number exponents only.
**3. Example to quote:** > "`x ** Value(2.0)` raises it."
**Seen in:** `WALKTHROUGH.md`, debugging

## Non-determinism from set ordering  `(concept)`
**1. Definition:** Iteration order of a `set` of objects depends on their memory addresses and can change between runs.
**2. In plain words:** Floating-point sums in a different order differ in the last digits, which chaotic training can amplify.
**3. Example to quote:** > "We store `_prev` in a tuple."
**Seen in:** `main.py`, block 1
