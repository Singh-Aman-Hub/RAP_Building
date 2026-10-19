# Step 01 — numpy and what "data" means in ML

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
train rows: [0, 2, 3, 4] validation rows: [1]
```

## Experiments
1. `w = 6.0` → loss changes (try it; which direction?).
2. Remove the scaling line: compare feature ranges before/after.
3. Different `seed` → different split. Same seed → same split (reproducibility).

## Debugging practice
`X2 @ np.array([0.5, 0.01, 1.0])` → `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0...`. It tells you the shapes disagree: (3,2) needs a length-2 vector. Print `.shape` of both sides first.

## Rebuild checklist
- [ ] Compute MSE for a line by hand-typed numpy.
- [ ] Explain (samples, features) layout.
- [ ] Standardise columns with mean/std.

## Say it aloud
"A model is a function with adjustable numbers called parameters. The data has features as input and labels as the correct outputs. The loss measures how wrong predictions are, and training changes the parameters to lower it. I keep some data aside as validation to measure on unseen examples. Limitation: the model can only be as good as its data."
