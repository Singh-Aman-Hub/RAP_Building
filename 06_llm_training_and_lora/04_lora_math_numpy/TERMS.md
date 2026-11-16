# Terms — LoRA Math with NumPy

Folder: `06_llm_training_and_fast_inference/04_lora_math_numpy` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [06 README](../README.md)

**Terms in this folder:** [NumPy](#numpy) · [np.random.default_rng](#nprandomdefault_rng) · [LoRA (low-rank adaptation)](#lora-low-rank-adaptation) · [Rank (r)](#rank-r) · [Frozen weights](#frozen-weights) · [Matrix multiplication `@`](#matrix-multiplication-) · [Transpose `.T`](#transpose-t) · [np.zeros and zero initialisation of B](#npzeros-and-zero-initialisation-of-b) · [rng.normal](#rngnormal) · [Down-projection A and up-projection B](#down-projection-a-and-up-projection-b) · [alpha and scaling (alpha / r)](#alpha-and-scaling-alpha--r) · [Order of multiplication (A @ x first)](#order-of-multiplication-a--x-first) · [np.abs(...).max()](#npabsmax) · [Mean squared error `(err ** 2).mean()`](#mean-squared-error-err--2mean) · [Hand-derived gradient](#hand-derived-gradient) · [In-place update `-=`](#in-place-update--) · [Merging an adapter](#merging-an-adapter) · [Parameter-count arithmetic and `:,` formatting](#parameter-count-arithmetic-and--formatting) · [Target modules (q, k, v, o)](#target-modules-q-k-v-o) · [ValueError: matmul mismatch](#valueerror-matmul-mismatch)

---

## NumPy  `(third-party: numpy)`
**1. Definition:** NumPy is a Python library providing the `ndarray` n-dimensional array and fast vectorised maths and linear algebra.
**2. In plain words:** It lets you do matrix maths on whole grids of numbers in one line instead of slow Python loops. Here it simulates LoRA without needing any model or download.
**3. Example to quote:** > "I used NumPy to build a toy layer and show LoRA's idea with plain matrices, no GPU needed."
```python
import numpy as np
np.zeros((2, 3)).shape   # (2, 3)
```
**Seen in:** `main.py`, import

## np.random.default_rng  `(third-party: numpy)`
**1. Definition:** `np.random.default_rng(seed)` creates a NumPy random-number generator object; with a fixed seed it produces the same sequence every run.
**2. In plain words:** It makes the "random" matrices repeatable so your numbers match the walkthrough.
**3. Example to quote:** > "`rng = np.random.default_rng(0)` makes my experiment reproducible."
**Seen in:** `main.py`, top

## LoRA (low-rank adaptation)  `(concept)`
**1. Definition:** LoRA fine-tunes a model by freezing the pretrained weight matrix W and learning a small update expressed as the product of two thin matrices, ΔW = (alpha / r) · B · A.
**2. In plain words:** Instead of changing all 4 million numbers in a big layer, you learn only two skinny matrices (32,768 numbers for r=8 on a 2048×2048 layer, 0.78%). Empirically the useful change in fine-tuning is low-rank, so this works well. Limitation: low rank caps how much new behaviour the adapter can absorb.
**3. Example to quote:** > "A full 2048×2048 update is 4,194,304 parameters; LoRA with r=8 is 2048·8 + 8·2048 = 32,768, so about 0.78% of that."
**Seen in:** `main.py`, section 1

## Rank (r)  `(concept)`
**1. Definition:** In LoRA, the rank r is the inner dimension shared by B and A, i.e. how many columns B and rows A have.
**2. In plain words:** It's the adapter's capacity dial. Bigger r fits more complex changes but costs more trainable parameters. With r=1 the toy error cannot reach about 0 because the target change has rank 4.
**3. Example to quote:** > "My target change is rank 4, so with r=1 the error plateaus, and r=16 costs four times the parameters of r=4 for little gain."
**Seen in:** `main.py` (`r`), TRY IT

## Frozen weights  `(concept)`
**1. Definition:** Frozen weights are parameters excluded from training, so their values never change.
**2. In plain words:** The original matrix W stays exactly as pretrained; only A and B learn. That saves memory for gradients and optimizer state and preserves the base model.
**3. Example to quote:** > "In the training loop W is never updated; only `B` and `A` change."
**Seen in:** `main.py`, section 3

## Matrix multiplication `@`  `(syntax)`
**1. Definition:** The `@` operator performs matrix multiplication between arrays.
**2. In plain words:** It's how a layer turns an input vector into an output: `W @ x`. Shapes must line up: columns of the left matrix equal rows of the right.
**3. Example to quote:** > "`W @ x` is the frozen layer's output, and `B @ (A @ x)` is the LoRA correction added to it."
**Seen in:** `main.py`

## Transpose `.T`  `(syntax)`
**1. Definition:** `.T` on a NumPy array returns its transpose, swapping rows and columns.
**2. In plain words:** It is needed when computing gradients so the shapes line up.
**3. Example to quote:** > "In the gradient for B I use `(A @ x).T` to match shapes."
**Seen in:** `main.py`, gradients

## np.zeros and zero initialisation of B  `(third-party: numpy)`
**1. Definition:** `np.zeros(shape)` creates an array of zeros; LoRA initialises B with zeros.
**2. In plain words:** Because B is zero, the update B·A starts at exactly 0, so the adapted model begins identical to the pretrained one and training can only move away gradually. The output difference at init is 0.0.
**3. Example to quote:** > "At init `max |LoRA output - base output| = 0.0` because B is all zeros, so a fresh adapter changes nothing."
**Seen in:** `main.py`, section 2

## rng.normal  `(third-party: numpy)`
**1. Definition:** `rng.normal(size=...)` draws random numbers from a Gaussian (bell-curve) distribution.
**2. In plain words:** It makes the fake weights, inputs and target change. Multiplying by 0.01 makes A start very small.
**3. Example to quote:** > "`A = rng.normal(size=(r, d)) * 0.01` gives a small random down-projection."
**Seen in:** `main.py`

## Down-projection A and up-projection B  `(concept)`
**1. Definition:** In LoRA, A (shape r × d_in) projects the input down to r dimensions and B (shape d_out × r) projects it back up to the output size.
**2. In plain words:** The signal goes through a narrow bottleneck: wide to narrow to wide. That bottleneck is why there are so few parameters.
**3. Example to quote:** > "A squeezes the 64-dim input into 4 numbers, and B expands them back to 64."
**Seen in:** `main.py`, section 2

## alpha and scaling (alpha / r)  `(concept)`
**1. Definition:** `alpha` is a constant by which the LoRA update is scaled, in practice multiplied by `alpha / r`.
**2. In plain words:** It controls how strongly the adapter influences the layer, and keeps the effective strength more stable when you change r.
**3. Example to quote:** > "I use `alpha = 8`, `r = 4`, so the update is multiplied by 2."
**Seen in:** `main.py`, section 2

## Order of multiplication (A @ x first)  `(concept)`
**1. Definition:** Computing `B @ (A @ x)` multiplies A with the input first, producing a small (r × batch) matrix, instead of building the full d × d matrix `B @ A`.
**2. In plain words:** Same numbers, much less memory and compute for large d. For d=4096, building the full matrix first wastes memory.
**3. Example to quote:** > "I write `B @ (A @ x)` rather than `(B @ A) @ x` so I never create the huge full matrix."
**Seen in:** `main.py`, comments

## np.abs(...).max()  `(third-party: numpy)`
**1. Definition:** `np.abs(x)` returns the absolute value of each element and `.max()` returns the largest.
**2. In plain words:** Together they measure the biggest difference between two arrays, a handy "are these equal?" check.
**3. Example to quote:** > "`np.abs(lora_out - base_out).max()` printed 0.0 so the outputs match exactly at init."
**Seen in:** `main.py`

## Mean squared error `(err ** 2).mean()`  `(concept)`
**1. Definition:** Mean squared error averages the squares of the differences between prediction and target.
**2. In plain words:** It's the toy problem's loss: lower means the layer imitates the target better. It fell from 3.0181 to 0.0184.
**3. Example to quote:** > "My error went from 3.0181 at step 0 to 0.0184 at step 300 while only A and B were trained."
**Seen in:** `main.py`, training loop

## Hand-derived gradient  `(concept)`
**1. Definition:** A gradient derived by hand uses the chain rule to write the formula for how the loss changes with each trainable matrix.
**2. In plain words:** PyTorch's autograd does this for you. Writing it out shows what happens under the hood and that only A and B receive gradients.
**3. Example to quote:** > "I computed `grad_B` and `grad_A` manually; PyTorch's `loss.backward()` would do the same automatically."
**Seen in:** `main.py`, section 3

## In-place update `-=`  `(syntax)`
**1. Definition:** `x -= y` subtracts `y` from `x` in place; for arrays it modifies the existing array.
**2. In plain words:** `B -= lr * grad_B` is the plain gradient-descent step: move against the gradient by a learning rate.
**3. Example to quote:** > "`B -= lr * grad_B` is the same step an optimizer's `step()` does."
**Seen in:** `main.py`

## Merging an adapter  `(concept)`
**1. Definition:** Merging folds the learned update into the base weights: W_merged = W + (alpha / r) · B · A.
**2. In plain words:** After merging, the model is a single ordinary matrix again, so inference costs the same as the original model. The difference from the unmerged version is only rounding (about 1e-14).
**3. Example to quote:** > "After training I merge with `W + scale * (B @ A)` and inference has no extra layers."
**Seen in:** `main.py`, section 4

## Parameter-count arithmetic and `:,` formatting  `(syntax)`
**1. Definition:** The count of trainable parameters is the product of matrix dimensions; `f"{n:,}"` formats a number with thousands separators.
**2. In plain words:** `2048*8 + 8*2048 = 32,768` is the entire argument for LoRA's efficiency, and `:,` just makes big numbers readable.
**3. Example to quote:** > "`f'{4194304:,}'` prints `4,194,304`, the full update size."
**Seen in:** `main.py`, section 1

## Target modules (q, k, v, o)  `(concept)`
**1. Definition:** Target modules are the specific layers of a transformer to which LoRA adapters are attached, commonly the attention projections query (q), key (k), value (v) and output (o).
**2. In plain words:** You don't adapt every weight, only chosen matrices. For a 7B-style model with r=8 on q, k, v, o in 32 layers, that is about 8.4M trainable parameters, 0.12% of the model.
**3. Example to quote:** > "Adapting q, k, v, o in 32 layers at r=8 gives about 8.4 million trainable parameters versus roughly 7 billion total."
**Seen in:** `main.py`, section 5

## ValueError: matmul mismatch  `(concept)`
**1. Definition:** NumPy raises `ValueError: matmul: Input operand ... mismatch in its core dimension` when the inner dimensions of a matrix product do not match.
**2. In plain words:** If A is `(d, r)` instead of `(r, d)`, `A @ x` can't work. Print the `.shape` of every operand before multiplying.
**3. Example to quote:** > "I read the shapes in the error, saw A had the wrong orientation, and fixed it to `(r, d)`."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
