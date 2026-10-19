"""
GOAL: See overfitting: a flexible model memorises noisy training data (train loss tiny)
      but does badly on unseen validation data.  Learn to spot it from the two loss curves.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/05_overfitting_train_val/main.py
EXPECTED OUTPUT: table of polynomial degree vs train/validation error; train keeps falling, validation turns up.
NEEDS: numpy
"""

import numpy as np

rng = np.random.default_rng(1)
x = rng.uniform(-1, 1, 30)
y = np.sin(3 * x) + rng.normal(0, 0.2, 30)               # true curve + noise
x_train, y_train = x[:15], y[:15]                        # model sees these
x_val, y_val = x[15:], y[15:]                            # model never trains on these

print(f"{'degree':>6} {'train MSE':>10} {'val MSE':>10}")
for degree in (1, 3, 5, 9, 12):                          # higher degree = more flexible curve
    coefs = np.polyfit(x_train, y_train, degree)         # least-squares polynomial fit (a model with degree+1 parameters)
    train_mse = np.mean((np.polyval(coefs, x_train) - y_train) ** 2)
    val_mse = np.mean((np.polyval(coefs, x_val) - y_val) ** 2)
    print(f"{degree:>6} {train_mse:>10.4f} {val_mse:>10.4f}")

# Reading the table:
#  degree 1  -> underfitting: both errors high (model too simple)
#  degree 3  -> good fit: both low and close
#  degree 12 -> overfitting: train error ~0, validation error large (memorised noise)
# Remedies: more data, simpler model, regularisation, early stopping (stop when validation loss starts rising),
# dropout, data augmentation.
# TRY IT: split at 25 instead of 15 (x[:25] train, x[25:] val) and re-run: overfitting shrinks. More data helps.
# BREAK IT: evaluate "val" on x_train by mistake -> the gap disappears and you fool yourself. That's data leakage.
