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
cd 06_llm_training_and_fast_inference/04_lora_math_numpy
python main.py       # instant, numpy only
```

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
