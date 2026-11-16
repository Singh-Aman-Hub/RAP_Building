# Step 04 — LoRA math with NumPy

## The story so far
[Step 03](../03_eval_perplexity_checkpoints/WALKTHROUGH.md) closed with a bill: full fine-tuning updates every weight, stores every gradient and optimizer state, and saves a full copy per task.

## The pain point
A single weight matrix in a 7B model is 4096 by 4096, about 16.8 million numbers. Fine-tuning changes each of them, but the useful change for one task is small and simple. We are paying for 16.8 million degrees of freedom when we need very few.

## The spark
Freeze the original `W0` and learn the update as a product of two thin matrices: `W = W0 + B @ A`, with `A` of shape (r, d) and `B` of shape (d, r), r around 8. That is 2*d*r numbers instead of d*d, under 1% for this matrix. Start `B` at zero so the model begins exactly as before.

## Goal
See *why* LoRA trains so few parameters: freeze `W`, learn a low-rank update `B @ A`.

## Run it
```bash
cd 06_llm_training_and_lora/04_lora_math_numpy
python main.py       # instant, numpy only
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Rank r / bottleneck width | LoRA rank | Yes | The inner size of the two skinny matrices; the main capacity knob. |
| LoRA / low-rank adaptation / adapter | PEFT, delta weights | LoRA is one kind of adapter | Other adapters insert small MLP layers; LoRA adds a low-rank product to existing weights and can be merged. |
| `@` / matmul / dot | matrix multiplication | Yes (for 2-D) | Python operator for matrix product; `*` is element-wise. |
| Frozen / non-trainable / requires_grad=False | locked weights | Yes | The weights are never updated. |
| Merge / fold / fuse | merge_and_unload | Yes | Add `scale * B @ A` into W once, so inference has no extra layers. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–14)

```python
"""
GOAL:  Understand LoRA with pure NumPy: instead of updating a huge weight matrix W, learn two skinny
       matrices B and A whose product B @ A is the update.  Count the savings and watch it learn.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_lora/04_lora_math_numpy/main.py
EXPECTED OUTPUT (shortened):
    full update parameters : 4,194,304 (2048 x 2048)
    LoRA   parameters (r=8): 32,768   -> 0.78% of full
    at init  max |LoRA output - base output| = 0.0
    step    0 error 3.0181
    step  300 error 0.0184
NEEDS: numpy only (no download).  < 2 s.
"""
```

- **In simple words:** What LoRA is in one paragraph and what output to expect.
- **Exactly:** Not executed; says full update 4,194,304 params vs LoRA 32,768 (0.78%).
- **Why it is here:** Orients you before the maths.

### 2. Import NumPy and seed  (`main.py` lines 16–18)

```python
import numpy as np   # 3rd-party: fast arrays + linear algebra

rng = np.random.default_rng(0)       # random generator with a fixed seed -> same numbers every run
```

- **In simple words:** Use NumPy and a fixed random generator so every run gives the same numbers.
- **Exactly:** `np.random.default_rng(0)` creates a seeded `Generator`.
- **Why it is here:** Reproducible output lets this walkthrough quote exact numbers.

### 3. Count parameters  (`main.py` lines 20–25)

```python
# ---- 1. The parameter-count argument ----------------------------------------
d_out, d_in, r = 2048, 2048, 8       # r = the "rank": width of the skinny matrices (the LoRA knob)
full = d_out * d_in                  # fine-tuning W directly changes every entry
lora = d_out * r + r * d_in          # B is (d_out x r), A is (r x d_in)
print(f"full update parameters : {full:,} ({d_out} x {d_in})")
print(f"LoRA   parameters (r={r}): {lora:,}   -> {100 * lora / full:.2f}% of full")
```

- **In simple words:** Compare changing the whole matrix with changing two skinny matrices.
- **Exactly:** `full = 2048*2048 = 4,194,304`; `lora = 2048*8 + 8*2048 = 32,768`; ratio 0.78%.
- **Why it is here:** This is the whole reason LoRA exists: far fewer numbers to train and store.

### 4. Create W, A, B  (`main.py` lines 27–34)

```python
# ---- 2. The forward pass: y = W x + (alpha / r) * B (A x) --------------------
# Use a smaller size for the learning demo so it runs instantly.
d, r = 64, 4
W = rng.normal(size=(d, d))                 # shape: (d_out, d_in) = (64, 64); pretend: a frozen pretrained weight matrix
A = rng.normal(size=(r, d)) * 0.01          # shape: (r, d_in) = (4, 64); LoRA "down-projection": random small init
B = np.zeros((d, r))                        # shape: (d_out, r) = (64, 4); LoRA "up-projection": ZERO init  => update starts at exactly 0
alpha = 8                                   # scaling constant; the update is multiplied by alpha / r
scale = alpha / r
```

- **In simple words:** Pretend W is a frozen pretrained matrix; make A small random and B all zeros.
- **Exactly:** `d=64, r=4`. `W` is `(64, 64)` normal; `A` is `(4, 64)` times 0.01; `B` is `(64, 4)` zeros; `scale = alpha/r = 8/4 = 2.0`.
- **Why it is here:** Zero B means the adapter starts as "no change", so training begins from the pretrained behaviour.

### 5. Forward pass at initialisation  (`main.py` lines 36–40)

```python
x = rng.normal(size=(d, 10))                # shape: (d_in, n_examples) = (64, 10); 10 example input vectors (columns)
base_out = W @ x                            # shape: (d_out, n_examples) = (64, 10); @ is matrix multiplication in Python
lora_out = W @ x + scale * (B @ (A @ x))    # shape: (64, 10); A @ x is (r, 10), B @ that is (64, 10); note the order: A @ x first keeps everything small (r x 10)
print("\nat init  max |LoRA output - base output| =", np.abs(lora_out - base_out).max())
# Because B = 0, a fresh LoRA adapter changes nothing: training starts from the pretrained behaviour.
```

- **In simple words:** Run the layer with and without the adapter and show they match.
- **Exactly:** `x` is `(64, 10)`. `A @ x` is `(4, 10)`, `B @ that` is `(64, 10)`. Real: max difference `0.0`.
- **Why it is here:** Multiplying `A @ x` first avoids building a 64 x 64 matrix; and the zero check shows the adapter is harmless at the start.

### 6. Make a target that is a rank-4 change  (`main.py` lines 42–46)

```python
# ---- 3. Train ONLY A and B so the layer imitates a new target ----------------
# Pretend fine-tuning wants W_target = W + (a rank-r change). Real fine-tuning updates are
# empirically low-rank too, which is why LoRA works.
true_delta = rng.normal(size=(d, r)) @ rng.normal(size=(r, d)) * 0.1   # shape: (64, 64), a rank-4 change
y_target = (W + true_delta) @ x                        # shape: (64, 10)
```

- **In simple words:** Invent a "fine-tuned" layer that differs from W by a simple low-rank change.
- **Exactly:** `true_delta` is `(64,4) @ (4,64) * 0.1` = rank 4; `y_target = (W + true_delta) @ x`, shape `(64, 10)`.
- **Why it is here:** Gives the training something that LoRA can in principle fit perfectly.

### 7. Train only A and B by hand  (`main.py` lines 48–60)

```python
lr = 0.05                                               # learning rate for this toy problem
print()
for step in range(301):
    y_pred = W @ x + scale * (B @ (A @ x))              # forward; shape: (64, 10)
    err = y_pred - y_target                              # shape: (d, 10) = (64, 10)
    loss = (err ** 2).mean()
    # Gradients derived by hand with the chain rule (PyTorch autograd does this for you):
    grad_B = scale * (err @ (A @ x).T) / err.size * 2   # shape: (64, 4) same as B
    grad_A = scale * (B.T @ err @ x.T) / err.size * 2   # shape: (4, 64) same as A
    B -= lr * grad_B                                # W is NEVER touched: it stays frozen
    A -= lr * grad_A
    if step % 100 == 0:
        print(f"step {step:4d} error {loss:.4f}")
```

- **In simple words:** Repeat predict, measure error, nudge A and B downhill, never touching W.
- **Exactly:** Loss is mean squared error. `grad_B = scale * err @ (A@x).T / err.size * 2` has shape `(64,4)`, `grad_A = scale * B.T @ err @ x.T / err.size * 2` has shape `(4,64)`; plain gradient descent with lr 0.05. Real: error 3.0181, 0.2478, 0.0326, 0.0184 at steps 0, 100, 200, 300.
- **Why it is here:** This is exactly what autograd does in PyTorch; writing it once shows nothing is magic.

### 8. Merge the adapter into W  (`main.py` lines 62–66)

```python
# ---- 4. Merge: fold the adapter back into W for zero-overhead inference -------
W_merged = W + scale * (B @ A)                          # shape: (64, 64); (64,4) @ (4,64) -> (64,64)
merged_out = W_merged @ x                               # shape: (64, 10)
print("\nmerged vs unmerged max difference:", np.abs(merged_out - (W @ x + scale * (B @ (A @ x)))).max())
print("=> after merging, inference costs the same as the original model (no extra layers).")
```

- **In simple words:** Fold the learned change back into W so deployment is as fast as the original.
- **Exactly:** `W_merged = W + scale * (B @ A)`, shape `(64,64)`. Real difference to the unmerged output: `2.1e-14` (float rounding only).
- **Why it is here:** Merging removes the extra matrix multiplies at serving time.

### 9. Scale up to a 7B model  (`main.py` lines 68–73)

```python
# ---- 5. What it means for a 7B model ------------------------------------------
layers, hidden = 32, 4096
per_matrix = 2 * hidden * 8                     # r = 8 on a hidden x hidden matrix
adapted = layers * 4 * per_matrix               # adapt q, k, v, o in each of 32 layers
print(f"\n7B-style model, r=8 on q,k,v,o: ~{adapted / 1e6:.1f}M trainable params vs ~7000M total "
      f"({100 * adapted / 7e9:.2f}%)")
```

- **In simple words:** Estimate how many numbers LoRA trains on a realistic model.
- **Exactly:** `per_matrix = 2*4096*8 = 65,536`; times 4 matrices times 32 layers = 8,388,608. Real print: `~8.4M trainable params vs ~7000M total (0.12%)`.
- **Why it is here:** Gives the number to quote in an interview.

### 10. TRY IT / BREAK IT  (`main.py` lines 75–80)

```python
# TRY IT: change r (the second `d, r = 64, 4`) to 1 - the target has rank 4, so error cannot reach ~0. Why?
# TRY IT: change r to 16 - does it fit better or faster?
# TRY IT: initialise B with rng.normal(...) instead of zeros and re-check the "at init" difference line.
# TRY IT: replace `B @ (A @ x)` by `(B @ A) @ x` - same numbers, but it builds a full d x d matrix first.
#         For d=4096 that wastes memory; this is why LoRA code multiplies A @ x FIRST.
# BREAK IT: make A shape (d, r) instead of (r, d) and read the 'matmul: Input operand ... mismatch' ValueError.
```

- **In simple words:** Experiments in comments.
- **Exactly:** Making A shaped `(d, r)` makes `B @ (A @ x)` fail with a `ValueError` about matmul dimensions.
- **Why it is here:** Shape errors are the most common LoRA bug.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `A @ B` | matrix multiplication | `np.eye(2) @ np.ones((2,1))` |
| `A.T` | transpose | shape `(r,d)` -> `(d,r)` |
| `np.random.default_rng(0)` | seeded random generator (reproducible) | `rng.normal(size=(2,2))` |
| `B -= lr * grad` | in-place update, shorthand for `B = B - lr*grad` | |
| `f"{n:,}"` | thousands separators | `f"{4194304:,}"` -> `4,194,304` |
| `x.size` | number of elements | `np.zeros((3,4)).size` -> `12` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `np.zeros(shape)` | numpy | array of zeros (LoRA's `B` init) | shape | ndarray |
| `rng.normal(size=...)` | numpy | Gaussian random numbers | `size`, `scale` | ndarray |
| `np.abs(x).max()` | numpy | largest absolute value | – | float |
| `(x ** 2).mean()` | numpy | mean squared error pattern | – | float |

## Execution trace (real run)
- Full update of a 2048x2048 matrix: **4,194,304** parameters. LoRA r=8: `2048*8 + 8*2048 = ` **32,768** (0.78%).
- At init `B = 0`, so `max |LoRA output - base output| = 0.0`: the adapter starts as a no-op.
- Training only `A` and `B` toward a rank-4 target: error 3.0181 -> 0.2478 (step 100) -> 0.0326 -> **0.0184** (step 300).
- Merge `W + (alpha/r) B A`: difference vs unmerged is ~1e-14 (float rounding) -> merged inference is free.
- 7B-style estimate: ~8.4M trainable parameters, 0.12% of the model.

## Experiments (change ONE thing)
1. `r = 1`: the target has rank 4, so the error plateaus above zero — capacity limit.
2. `r = 16`: fits as well or better but costs 4x the adapter parameters.
3. Initialise `B` with random numbers: the "at init" difference is no longer 0, so training would start by damaging the pretrained model.

## Debugging practice
Make `A` shape `(d, r)`:
```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 ...
```
Read the shapes: `A @ x` needs `A` columns == `x` rows. Print `.shape` of every operand before multiplying.

## Rebuild checklist
- [ ] `W` frozen, `A` small random `(r, d)`, `B` zeros `(d, r)`
- [ ] forward: `W @ x + (alpha/r) * B @ (A @ x)`
- [ ] gradients for A and B only
- [ ] merge: `W + (alpha/r) * B @ A`

## The new problem (cliffhanger)
On paper the numbers are tiny. Does it work on a real model with a real training loop? Next: [step 05, LoRA fine-tuning with peft](../05_lora_peft_finetune/WALKTHROUGH.md).

> **Say it aloud (interview box):** LoRA freezes the pretrained weight matrix and learns its change as the product of two thin matrices, W equals W0 plus B times A, where the rank r is small. That trains about one percent of the parameters or less, and B starts at zero so training begins from the original model. After training I can merge B times A into W0 so inference costs nothing extra.
