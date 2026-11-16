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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Feature(s) | input, predictor, independent variable, X, covariate | Yes, same thing | "Feature" is the ML word; statisticians say "predictor"; `X` is the variable name in code |
| Label | target, y, ground truth, dependent variable | Yes, same thing | The known right answer the model is trained to reproduce |
| Weight and bias | parameters, coefficients, theta | Weight and bias are both parameters | Weight multiplies a feature; bias is added once (the intercept) |
| Loss | cost, objective, error function | Mostly | "Loss" is per example or batch; "cost" often means the average over the whole dataset |
| MSE | mean squared error, L2 loss | Same formula | `mean((y_hat - y)**2)`; "L2 loss" sometimes omits the mean (sum of squares) |
| Validation set | dev set, held-out set | Mostly | Used to tune choices; the test set is a separate third split touched once at the end |
| Normalising / scaling | standardising, z-scoring | Here yes | `(x - mean) / std` gives mean 0, std 1; "min-max scaling" is a different formula to [0,1] |

## Code, step by step

### 1. Module docstring: goal, how to run, needs  (`main.py` lines 1–9)

```python
"""
GOAL: Understand what a machine-learning dataset looks like (features X, labels y)
      and the numpy operations every later step relies on.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/01_numpy_and_data/main.py
EXPECTED OUTPUT: shapes, a prediction, "mean"/"std" values.
NEEDS: numpy
"""
```

- **In simple words:** A comment-like note at the top that says what the file is for and how to run it.
- **Exactly:** A triple-quoted string at the top of a module is its docstring, stored in `__doc__`. Python never executes it as logic.
- **Why it is here:** It tells the next reader what to expect so they do not have to read all the code first.

### 2. Import numpy as np  (`main.py` line 11)

```python
import numpy as np  # numpy = fast arrays; `np` is the universal nickname
```

- **In simple words:** Load the array library and give it the short name everyone uses.
- **Exactly:** `import numpy as np` binds the module to the name `np`. NumPy provides `ndarray`, a fixed-type n-dimensional array stored in one contiguous memory block, with C-speed operations.
- **Why it is here:** Without it none of the `np.` calls below exist (`NameError: name 'np' is not defined`).

### 3. Build a tiny dataset: hours studied vs exam score  (`main.py` lines 13–20)

```python
# --- ML in one picture ------------------------------------------------------
# A model is a function: prediction = f(features).  "Learning" = finding f's numbers (parameters)
# so predictions match the known answers (labels) in the training data.
#
# Tiny dataset: predict an exam score (label y) from hours studied (feature x).
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])        # features: shape (5,)  -> 5 samples
y = np.array([52.0, 58.0, 61.0, 68.0, 74.0])   # labels (the "right answers")
print("X shape:", X.shape, "| y shape:", y.shape)
```

- **In simple words:** Five students: how many hours each studied (X) and what they scored (y). The model must learn the link.
- **Exactly:** `X` is a 1-D `float64` array of shape `(5,)`; `y` also `(5,)`. Run output: `X shape: (5,) | y shape: (5,)`. A 1-D array has one axis (length 5), so it is neither a row nor a column vector. The comments are the whole ML idea: prediction = f(features), learning = finding f's parameters.
- **Why it is here:** Every ML task starts as arrays of inputs and known answers; this is the smallest possible example.

### 4. Predict with a hand-picked straight line  (`main.py` lines 22–25)

```python
# --- The model: a straight line y_hat = w * x + b ---------------------------
w, b = 5.0, 45.0                                # parameters (weight and bias) - guessed by hand for now
y_hat = w * X + b                               # shape: (5,) | vectorised: applies to ALL 5 samples at once (no loop!)
print("predictions:", y_hat)
```

- **In simple words:** Guess a slope and offset, then predict all five scores at once.
- **Exactly:** `w, b = 5.0, 45.0` is tuple unpacking into two Python floats. `y_hat = w * X + b` is vectorised: the scalar `w` is broadcast over the `(5,)` array, then `b` is added, giving `(5,)`. Real output: `predictions: [50. 55. 60. 65. 70.]` (for X=1 it is 5*1+45=50).
- **Why it is here:** Vectorised code replaces a Python `for` loop. It is 10-100x faster and is how every later model (including LLMs) is written.

### 5. Measure the error with mean squared error  (`main.py` lines 27–31)

```python
# --- How wrong is the model?  Loss = mean squared error ---------------------
errors = y_hat - y                              # shape: (5,) | element-wise subtraction
mse = np.mean(errors ** 2)                      # square so +/- errors don't cancel; mean over samples
print("MSE loss:", round(float(mse), 3))
# TRY IT: change w to 6.0. Does the loss go up or down?  That "up or down" signal is what training uses.
```

- **In simple words:** Compute how far each guess is from the truth, then average the squared misses into one number.
- **Exactly:** `errors = y_hat - y` is element-wise, shape `(5,)`: `[-2, -3, -1, -3, -4]`. `errors ** 2` squares each, `np.mean` averages to a scalar: (4+9+1+9+16)/5 = 7.8. Printed: `MSE loss: 7.8`. `float(...)` turns the `numpy.float64` into a Python float, `round(..., 3)` trims it.
- **Why it is here:** Training needs a single number to push down. Squaring stops positive and negative errors cancelling and punishes big misses more.

### 6. A 2-D feature matrix and matrix multiplication  (`main.py` lines 33–40)

```python
# --- Shapes: the #1 source of beginner bugs ---------------------------------
X2 = np.array([[1.0, 100.0],                    # 2-D: rows = samples, columns = features
               [2.0, 150.0],
               [3.0, 120.0]])
print("X2 shape (samples, features):", X2.shape)
weights = np.array([0.5, 0.01])                 # shape: (2,) | one weight per feature
print("X2 @ weights =", X2 @ weights)           # @ is matrix multiplication: (3,2) @ (2,) -> (3,)
# BREAK IT: use weights = np.array([0.5, 0.01, 1.0]) and read the "matmul: Input operand 1 has a mismatch" ValueError.
```

- **In simple words:** Real data has several columns (features) per row (sample). The `@` operator combines each row with a weight per column.
- **Exactly:** `X2` is `(3, 2)`: 3 samples, 2 features. `weights` is `(2,)`. `X2 @ weights` is matmul `(3,2) @ (2,) -> (3,)`: row i gives `X2[i,0]*0.5 + X2[i,1]*0.01`. Printed `[1.5 2.5 2.7]` (1*0.5+100*0.01=1.5). The inner dimensions (2 and 2) must match.
- **Why it is here:** Shape mismatch is the most common ML bug. The BREAK IT comment shows the error you get when weights has 3 entries.

### 7. Normalise each feature column (broadcasting)  (`main.py` lines 42–47)

```python
# --- Broadcasting & normalising features ------------------------------------
mean, std = X2.mean(axis=0), X2.std(axis=0)     # shape: (2,) each | axis=0 -> statistics PER COLUMN
print("mean:", mean, "std:", std)
X2_scaled = (X2 - mean) / std                   # shape: (3, 2) | (3,2) - (2,) broadcasts the row across every sample
print("scaled columns now have mean ~0:", X2_scaled.mean(axis=0).round(6))
# Why scale? Features with very different sizes (1..3 vs 100..150) make gradient descent zig-zag (see step 02).
```

- **In simple words:** Put both columns on a similar scale so neither dominates just because its numbers are bigger.
- **Exactly:** `X2.mean(axis=0)` collapses axis 0 (rows), giving one mean per column, shape `(2,)`: `[2., 123.333]`; std `[0.816, 20.548]`. `(X2 - mean) / std` subtracts a `(2,)` row from every row of the `(3,2)` matrix (broadcasting: size-1 or missing leading axes are stretched). Result `(3,2)` with column means `[0. 0.]` after `.round(6)`.
- **Why it is here:** Features sized 1-3 next to 100-150 make gradient descent zig-zag (step 02). Scaling makes the loss surface rounder, so one learning rate suits every feature.

### 8. Shuffle and split into train and validation  (`main.py` lines 49–53)

```python
# --- Splitting data ---------------------------------------------------------
rng = np.random.default_rng(seed=0)             # seeded random generator -> reproducible results
indices = rng.permutation(len(X))               # shuffled positions 0..4
train_idx, val_idx = indices[:4], indices[4:]   # 80% train / 20% validation
print("train rows:", sorted(train_idx.tolist()), "validation rows:", val_idx.tolist())
```

- **In simple words:** Mix up the row order with a fixed seed, keep 4 rows for learning and hold 1 back to test on.
- **Exactly:** `np.random.default_rng(seed=0)` creates a seeded `Generator`; `permutation(5)` returns a shuffled copy of `[0..4]`. Slicing `[:4]` and `[4:]` gives 4 train and 1 validation index. Real output: `train rows: [0, 2, 3, 4] validation rows: [1]`. Same seed gives the same split every run.
- **Why it is here:** Judging a model on data it trained on hides overfitting (step 05). The seed makes the split reproducible, so results can be compared.

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
