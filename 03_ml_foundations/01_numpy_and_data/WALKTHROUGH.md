# Step 01 — numpy and what "data" means in ML

## The story so far
This is Chapter 1, the start of the journey. [Module 02](../../02_dsa_arrays_hashmaps/) taught you to hold numbers in arrays and think about cost. Now we use arrays for something new: holding *data* that a machine will learn from.

## The pain point
A normal program is a pile of rules you wrote by hand. For "predict an exam score from hours studied" nobody knows the rule, and a Python loop over every row is slow and clumsy. Worse, shapes like `(3,2)` against `(2,)` fail in confusing ways if you do not know the layout.

## The spark
Treat the data as a matrix: rows are samples, columns are features, and the model is one vectorised expression (`w * X + b`) applied to every row at once. One number, the loss, says how wrong the guess is. If we can *measure* wrongness, we can later reduce it.

## Goal
Know features (X), labels (y), parameters, predictions, loss, shapes, broadcasting and train/validation splits.

## Run it
```bash
python main.py
.venv/bin/python 03_ml_foundations/01_numpy_and_data/main.py   # from repo root
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `import numpy as np` | import with a short alias | `np.array([1,2])` |
| `X ** 2` | element-wise power | `np.array([1,2]) ** 2` → `[1 4]` |
| `X2 @ weights` | matrix multiplication | `(3,2) @ (2,)` → `(3,)` |
| `arr[:4]`, `arr[4:]` | slicing | first 4 / from index 4 |
| `float(x)` / `round(x, 3)` | convert numpy scalar / round | |
| `axis=0` | operate down the rows (per column) | `X2.mean(axis=0)` |

## Libraries & functions
| Name | Origin | Does | Important args | Returns |
|---|---|---|---|---|
| `np.array` | numpy (3rd-party) | build array from list | `dtype` | ndarray |
| `np.mean`, `.std` | numpy | average / spread | `axis` | scalar/array |
| `.shape` | ndarray attribute | dimensions | | tuple |
| `np.random.default_rng(seed)` | numpy | reproducible random generator | `seed` | Generator |
| `rng.permutation(n)` | numpy | shuffled `0..n-1` | | array |

## Execution trace (real run)
Input: hours `[1..5]`, scores `[52,58,61,68,74]`, guess `w=5, b=45`.
Processing: `y_hat = 5*X + 45 = [50 55 60 65 70]`; errors `[-2 -3 -1 -3 -4]`; squared `[4 9 1 9 16]`; mean 7.8.
Output:
```
X shape: (5,) | y shape: (5,)
predictions: [50. 55. 60. 65. 70.]
MSE loss: 7.8
X2 shape (samples, features): (3, 2)
X2 @ weights = [1.5 2.5 2.7]
mean: [  2.         123.33333333] std: [ 0.81649658 20.54804668]
scaled columns now have mean ~0: [0. 0.]
train rows: [0, 2, 3, 4] validation rows: [1]
```

## Experiments
1. `w = 6.0` → loss changes (try it; which direction?).
2. Remove the scaling line: compare feature ranges before/after.
3. Different `seed` → different split. Same seed → same split (reproducibility).

## Debugging practice
`X2 @ np.array([0.5, 0.01, 1.0])` → `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, ... (size 3 is different from 2)`. It tells you the shapes disagree: (3,2) needs a length-2 vector. Print `.shape` of both sides first.

## Rebuild checklist
- [ ] Compute MSE for a line by hand-typed numpy.
- [ ] Explain (samples, features) layout.
- [ ] Standardise columns with mean/std.

## The new problem (cliffhanger)
We can hold data as matrices and score a guess with the loss. But `w=5, b=45` was typed by hand. How does a machine *find* the right numbers by itself? That is the next chapter: [02_linear_regression_numpy](../02_linear_regression_numpy/).

## Say it aloud
> **Say it aloud (interview box):** A model is a function with adjustable numbers called parameters, the data is features in and labels out, and the loss is one number saying how wrong the predictions are. I keep a validation slice aside so I can measure on examples the model never saw. The limit is that a model can only be as good as its data, and I still have to guess the parameters by hand until the next step.
