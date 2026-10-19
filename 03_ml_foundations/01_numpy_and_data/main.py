"""
GOAL: Understand what a machine-learning dataset looks like (features X, labels y)
      and the numpy operations every later step relies on.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/01_numpy_and_data/main.py
EXPECTED OUTPUT: shapes, a prediction, "mean"/"std" values.
NEEDS: numpy
"""

import numpy as np  # numpy = fast arrays; `np` is the universal nickname

# --- ML in one picture ------------------------------------------------------
# A model is a function: prediction = f(features).  "Learning" = finding f's numbers (parameters)
# so predictions match the known answers (labels) in the training data.
#
# Tiny dataset: predict an exam score (label y) from hours studied (feature x).
X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])        # features: shape (5,)  -> 5 samples
y = np.array([52.0, 58.0, 61.0, 68.0, 74.0])   # labels (the "right answers")
print("X shape:", X.shape, "| y shape:", y.shape)

# --- The model: a straight line y_hat = w * x + b ---------------------------
w, b = 5.0, 45.0                                # parameters (weight and bias) - guessed by hand for now
y_hat = w * X + b                               # vectorised: applies to ALL 5 samples at once (no loop!)
print("predictions:", y_hat)

# --- How wrong is the model?  Loss = mean squared error ---------------------
errors = y_hat - y                              # element-wise subtraction
mse = np.mean(errors ** 2)                      # square so +/- errors don't cancel; mean over samples
print("MSE loss:", round(float(mse), 3))
# TRY IT: change w to 6.0. Does the loss go up or down?  That "up or down" signal is what training uses.

# --- Shapes: the #1 source of beginner bugs ---------------------------------
X2 = np.array([[1.0, 100.0],                    # 2-D: rows = samples, columns = features
               [2.0, 150.0],
               [3.0, 120.0]])
print("X2 shape (samples, features):", X2.shape)
weights = np.array([0.5, 0.01])                 # one weight per feature
print("X2 @ weights =", X2 @ weights)           # @ is matrix multiplication: (3,2) @ (2,) -> (3,)
# BREAK IT: use weights = np.array([0.5, 0.01, 1.0]) and read the "matmul: Input operand 1 has a mismatch" ValueError.

# --- Broadcasting & normalising features ------------------------------------
mean, std = X2.mean(axis=0), X2.std(axis=0)     # axis=0 -> statistics PER COLUMN
print("mean:", mean, "std:", std)
X2_scaled = (X2 - mean) / std                   # (3,2) - (2,) broadcasts the row across every sample
print("scaled columns now have mean ~0:", X2_scaled.mean(axis=0).round(6))
# Why scale? Features with very different sizes (1..3 vs 100..150) make gradient descent zig-zag (see step 02).

# --- Splitting data ---------------------------------------------------------
rng = np.random.default_rng(seed=0)             # seeded random generator -> reproducible results
indices = rng.permutation(len(X))               # shuffled positions 0..4
train_idx, val_idx = indices[:4], indices[4:]   # 80% train / 20% validation
print("train rows:", sorted(train_idx.tolist()), "validation rows:", val_idx.tolist())
