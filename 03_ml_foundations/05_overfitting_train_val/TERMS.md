# Terms — Overfitting and train/validation

Folder: `03_ml_foundations/05_overfitting_train_val` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [ML foundations README](../README.md)

**Terms in this folder:** [Overfitting](#overfitting--concept) · [Underfitting](#underfitting--concept) · [Generalisation](#generalisation--concept) · [Training set vs validation set](#training-set-vs-validation-set--concept) · [Data leakage](#data-leakage--concept) · [Model flexibility (polynomial degree)](#model-flexibility-polynomial-degree--concept) · [np.polyfit()](#nppolyfit--third-party-numpy) · [np.polyval()](#nppolyval--third-party-numpy) · [Synthetic data with noise](#synthetic-data-with-noise--concept) · [Early stopping and other remedies](#early-stopping-and-other-remedies--concept) · [f-string alignment (`:>6`, `:>10.4f`)](#f-string-alignment-6-104f--syntax) · [Slicing train/validation (`x[:15]`, `x[15:]`)](#slicing-trainvalidation-x15-x15--syntax) · [Mean squared error as the comparison metric](#mean-squared-error-as-the-comparison-metric--concept)

---

## Overfitting  `(concept)`
**1. Definition:** Overfitting is when a model fits the training data (including its random noise) so closely that it performs poorly on new, unseen data.
**2. In plain words:** It is like a student who memorises last year's exam answers: perfect on those exact questions, lost on new ones. In the table, a degree-12 polynomial has a training error of 0.0069 but a validation error of 430.7. The telltale sign is a training loss that keeps falling while the validation loss rises.
**3. Example to quote:**
> "At degree 12 my training error was almost zero but validation error exploded to 430, so the model memorised the noise instead of the pattern."

**Seen in:** `main.py`, degree loop

## Underfitting  `(concept)`
**1. Definition:** Underfitting is when a model is too simple to capture the real pattern, so it has high error on both training and validation data.
**2. In plain words:** A straight line (degree 1) cannot follow a sine curve, so both errors are high (0.17 and 0.20). More flexibility or better features fixes it.
**3. Example to quote:**
> "Degree 1 has high error on train and validation alike, so the model is too simple. That is underfitting."

**Seen in:** `main.py`, results comments

## Generalisation  `(concept)`
**1. Definition:** Generalisation is a model's ability to perform well on data it has not seen during training.
**2. In plain words:** The real goal of machine learning is good performance on new data, not on the training set. Validation error is the estimate of it.
**3. Example to quote:**
> "Degrees 3 to 5 have low training and validation error that are close together, so they generalise well."

**Seen in:** `main.py`, header and table

## Training set vs validation set  `(concept)`
**1. Definition:** The training set is the data the model learns from; the validation set is held-out data used only to measure performance on unseen samples.
**2. In plain words:** Here the first 15 points train the model and the last 15 are never used for fitting. Evaluating on training data hides overfitting.
**3. Example to quote:**
> "I fit on `x[:15]` and score on `x[15:]`, which the model never saw."

**Seen in:** `main.py`, `x_train`, `x_val`

## Data leakage  `(concept)`
**1. Definition:** Data leakage is when information from outside the training set (for example validation data) influences the model or its evaluation, producing misleadingly good scores.
**2. In plain words:** The BREAK IT drill evaluates "validation" on the training points by mistake, the gap vanishes, and you fool yourself. Always keep evaluation data separate, and fit scalers and vectorisers on training data only.
**3. Example to quote:**
> "If I accidentally score on `x_train`, the overfit model looks perfect. That is leakage, and the score is meaningless."

**Seen in:** `main.py`, BREAK IT comment

## Model flexibility (polynomial degree)  `(concept)`
**1. Definition:** Flexibility (capacity) is how complex a function a model can represent; for a polynomial it grows with the degree.
**2. In plain words:** Degree 1 is a line, degree 12 can wiggle through nearly every training point. More flexibility reduces training error but risks overfitting, which is the central bias-variance trade-off.
**3. Example to quote:**
> "A degree-12 polynomial has 13 parameters for only 15 training points, so it can almost memorise them."

**Seen in:** `main.py`, `for degree in (1, 3, 5, 9, 12)`

## np.polyfit()  `(third-party: numpy)`
**1. Definition:** `np.polyfit(x, y, deg)` fits a polynomial of degree `deg` to the points using least squares and returns its coefficients.
**2. In plain words:** It is a quick model with `deg + 1` parameters, found without a training loop. Very high degrees give numerically unstable fits.
**3. Example to quote:**
> "`np.polyfit(x_train, y_train, 3)` returns the four coefficients of the best cubic curve for my training points."

**Seen in:** `main.py`, `coefs`

## np.polyval()  `(third-party: numpy)`
**1. Definition:** `np.polyval(coefs, x)` evaluates a polynomial with the given coefficients at the points `x`.
**2. In plain words:** It is the "predict" step: plug x values into the fitted polynomial to get predictions, on training or validation points.
**3. Example to quote:**
> "`np.polyval(coefs, x_val)` gives predictions at the validation points so I can compute validation error."

**Seen in:** `main.py`, `train_mse`, `val_mse`

## Synthetic data with noise  `(concept)`
**1. Definition:** Synthetic data is artificially generated, here as a known curve `sin(3x)` plus random Gaussian noise.
**2. In plain words:** Because you know the true curve, you can see exactly when a model is learning the pattern versus the noise.
**3. Example to quote:**
> "I generate `y = sin(3x)` plus noise of 0.2 so I know the real pattern and can see the polynomial chase the noise."

**Seen in:** `main.py`, data generation

## Early stopping and other remedies  `(concept)`
**1. Definition:** Remedies for overfitting include more data, a simpler model, regularisation, dropout, data augmentation, and early stopping (halting training when validation loss starts rising).
**2. In plain words:** Early stopping is the cheapest: watch the validation loss each epoch and stop at its lowest point. More data is usually the best fix: splitting at 25 instead of 15 shrinks the problem.
**3. Example to quote:**
> "If validation loss rises for a few epochs in a row, I stop and keep the earlier checkpoint. That is early stopping."

**Seen in:** `main.py`, closing comments

## f-string alignment (`:>6`, `:>10.4f`)  `(syntax)`
**1. Definition:** In an f-string, `{value:>6}` right-aligns the value in a field 6 characters wide, and `{value:>10.4f}` also shows 4 decimals.
**2. In plain words:** Makes the printed table's columns line up so you can compare numbers at a glance.
**3. Example to quote:**
> "`f"{degree:>6} {train_mse:>10.4f}"` prints each degree and error in neat columns."

**Seen in:** `main.py`, table print

## Slicing train/validation (`x[:15]`, `x[15:]`)  `(syntax)`
**1. Definition:** `x[:15]` takes the first 15 items and `x[15:]` takes everything after.
**2. In plain words:** A simple, ordered split. Because the data was generated in random order it is safe here; with ordered data you should shuffle first.
**3. Example to quote:**
> "`x[:15]` is my training half and `x[15:]` is my validation half."

**Seen in:** `main.py`, split lines

## Mean squared error as the comparison metric  `(concept)`
**1. Definition:** MSE, `mean((prediction - truth)²)`, is used on both sets so their errors can be compared.
**2. In plain words:** If validation MSE is far above training MSE, the model does not generalise.
**3. Example to quote:**
> "Degree 9 has train MSE 0.023 and validation MSE 5.9: a gap that big signals overfitting."

**Seen in:** `main.py`, `train_mse`, `val_mse`
