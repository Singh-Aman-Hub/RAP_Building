# Terms — NumPy and data

Folder: `03_ml_foundations/01_numpy_and_data` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [ML foundations README](../README.md)

**Terms in this folder:** [Machine learning (model, parameters, learning)](#machine-learning-model-parameters-learning--concept) · [Features and labels (X and y)](#features-and-labels-x-and-y--concept) · [Sample](#sample--concept) · [import numpy as np](#import-numpy-as-np--syntax) · [NumPy array (ndarray)](#numpy-array-ndarray--third-party-numpy) · [Shape (`.shape`)](#shape-shape--third-party-numpy) · [Weight and bias (parameters)](#weight-and-bias-parameters--concept) · [Vectorisation](#vectorisation--concept) · [Prediction (y_hat)](#prediction-y_hat--concept) · [Loss and mean squared error (MSE)](#loss-and-mean-squared-error-mse--concept) · [np.mean() and power (`** 2`)](#npmean-and-power--2--third-party-numpy) · [Element-wise subtraction](#element-wise-subtraction--concept) · [Matrix multiplication (`@`)](#matrix-multiplication---syntax) · [Axis (`axis=0`)](#axis-axis0--third-party-numpy) · [Standard deviation (`.std`)](#standard-deviation-std--third-party-numpy) · [Broadcasting](#broadcasting--concept) · [Feature scaling (standardisation)](#feature-scaling-standardisation--concept) · [Train/validation split](#trainvalidation-split--concept) · [Random generator with a seed (`default_rng`)](#random-generator-with-a-seed-default_rng--third-party-numpy) · [rng.permutation()](#rngpermutation--third-party-numpy) · [Slicing arrays (`arr[:4]`, `arr[4:]`)](#slicing-arrays-arr4-arr4--syntax) · [float() and round()](#float-and-round--python-built-in) · [ValueError (shape mismatch)](#valueerror-shape-mismatch--concept)

---

## Machine learning (model, parameters, learning)  `(concept)`
**1. Definition:** Machine learning is building a function `prediction = f(features)` whose numbers (parameters) are adjusted automatically so that its predictions match known answers in the training data.
**2. In plain words:** Instead of writing rules by hand, you show the computer examples and let it tune the dials. Here the model is a straight line and "learning" means finding a good slope and intercept. The model can only be as good as the examples it sees.
**3. Example to quote:**
> "My model predicts an exam score from hours studied. Learning means finding the slope and offset so its predictions land close to the real scores."

**Seen in:** `main.py`, header comment

## Features and labels (X and y)  `(concept)`
**1. Definition:** Features (`X`) are the input values describing each sample; labels (`y`) are the known correct outputs the model should predict.
**2. In plain words:** X is the question and y is the answer key. Here X is hours studied and y is the exam score. In a text classifier, X would be the text and y the category.
**3. Example to quote:**
> "X is `[1, 2, 3, 4, 5]` hours and y is `[52, 58, 61, 68, 74]` scores. Each pair is one sample."

**Seen in:** `main.py`, `X`, `y`

## Sample  `(concept)`
**1. Definition:** A sample (or example/row) is one data point, a single set of feature values with its label.
**2. In plain words:** Five students means five samples. In a 2-D feature array the rows are samples and the columns are features.
**3. Example to quote:**
> "`X.shape` is `(5,)` so I have five samples; in `X2` the shape `(3, 2)` means 3 samples with 2 features each."

**Seen in:** `main.py`, shape prints

## import numpy as np  `(syntax)`
**1. Definition:** `import numpy as np` loads the third-party NumPy library and gives it the short alias `np`.
**2. In plain words:** NumPy provides fast arrays and maths, and every ML library builds on it. The alias `np` is a universal convention so everyone's code reads alike.
**3. Example to quote:**
> "I write `import numpy as np` and then call `np.array(...)`, the standard nickname."

**Seen in:** `main.py`, first line

## NumPy array (ndarray)  `(third-party: numpy)`
**1. Definition:** `np.array(list)` creates an `ndarray`, a fixed-type, multi-dimensional array supporting fast element-wise operations.
**2. In plain words:** It looks like a list but does maths on the whole thing at once and uses much less time and memory than Python loops. All elements share one type (here floats).
**3. Example to quote:**
> "`np.array([1.0, 2.0, 3.0])` is an array, and `X * 2` doubles every item without a loop."

**Seen in:** `main.py`, `X`, `y`, `X2`

## Shape (`.shape`)  `(third-party: numpy)`
**1. Definition:** `array.shape` is a tuple giving the size of each dimension, for example `(5,)` for a vector of 5 items or `(3, 2)` for 3 rows and 2 columns.
**2. In plain words:** Shape mismatches are the number one beginner bug in ML code. When something breaks, print the shapes first.
**3. Example to quote:**
> "`X2.shape` is `(3, 2)`: three samples, two features. A weight vector must have 2 items to multiply with it."

**Seen in:** `main.py`, shape prints

## Weight and bias (parameters)  `(concept)`
**1. Definition:** In the linear model `y_hat = w * x + b`, `w` (weight) scales the input and `b` (bias) shifts the output; both are learned parameters.
**2. In plain words:** The weight is the slope of the line and the bias is where it crosses the axis. Here they are guessed by hand (5 and 45); later steps learn them. An LLM has billions of such numbers.
**3. Example to quote:**
> "With `w=5, b=45` the line predicts 50 for one hour and 70 for five hours."

**Seen in:** `main.py`, `w, b = 5.0, 45.0`

## Vectorisation  `(concept)`
**1. Definition:** Vectorisation means applying an operation to a whole array at once instead of looping over elements in Python.
**2. In plain words:** `w * X + b` computes all five predictions in one expression. It is shorter, less error-prone and much faster because NumPy loops in compiled code.
**3. Example to quote:**
> "`y_hat = w * X + b` gives all five predictions at once with no for loop."

**Seen in:** `main.py`, `y_hat = w * X + b`

## Prediction (y_hat)  `(concept)`
**1. Definition:** `y_hat` is the model's output for the given inputs, as opposed to `y`, the true value.
**2. In plain words:** The little hat means "estimated". Comparing `y_hat` with `y` tells you how wrong the model is.
**3. Example to quote:**
> "My predictions are `[50, 55, 60, 65, 70]` and the true scores are `[52, 58, 61, 68, 74]`, so I am a few points off each time."

**Seen in:** `main.py`, `y_hat`

## Loss and mean squared error (MSE)  `(concept)`
**1. Definition:** A loss is a single number measuring how wrong predictions are; mean squared error is the average of the squared differences `(y_hat - y)²`.
**2. In plain words:** Squaring stops positive and negative errors from cancelling and punishes big mistakes more. Lower is better. Training is the search for parameters that make the loss small, and "does the loss go up or down?" is the signal it follows.
**3. Example to quote:**
> "The errors are -2, -3, -1, -3, -4, which square to 4, 9, 1, 9, 16. The mean is 7.8, so my MSE loss is 7.8."

```python
mse = np.mean((y_hat - y) ** 2)
```
**Seen in:** `main.py`, `mse`

## np.mean() and power (`** 2`)  `(third-party: numpy)`
**1. Definition:** `np.mean(arr)` returns the average of the values; `arr ** 2` squares every element.
**2. In plain words:** These two operations build MSE. NumPy applies `** 2` to each item separately, unlike a plain list.
**3. Example to quote:**
> "`np.array([1, 2]) ** 2` is `[1, 4]`, and `np.mean` of that is 2.5."

**Seen in:** `main.py`, `mse`

## Element-wise subtraction  `(concept)`
**1. Definition:** Subtracting two arrays of the same shape subtracts matching positions: `a - b` gives `[a0-b0, a1-b1, ...]`.
**2. In plain words:** `y_hat - y` yields one error per sample.
**3. Example to quote:**
> "`errors = y_hat - y` gives me the error for each of the five students."

**Seen in:** `main.py`, `errors`

## Matrix multiplication (`@`)  `(syntax)`
**1. Definition:** The `@` operator performs matrix multiplication; a `(3, 2)` array times a `(2,)` vector gives a `(3,)` result.
**2. In plain words:** Each row of features is multiplied by the weights and summed, giving one prediction per row. Inner dimensions must match; mixing 2 features with 3 weights raises a `ValueError`.
**3. Example to quote:**
> "`X2 @ weights` takes 3 samples times 2 weights and returns 3 predictions: `[1.5, 2.5, 2.7]`."

**Seen in:** `main.py`, `X2 @ weights`

## Axis (`axis=0`)  `(third-party: numpy)`
**1. Definition:** The `axis` argument tells NumPy which dimension to collapse; `axis=0` operates down the rows, giving one result per column.
**2. In plain words:** `X2.mean(axis=0)` returns the mean of each feature column. Forgetting `axis` gives one number for the entire array.
**3. Example to quote:**
> "`X2.mean(axis=0)` gives me one average per feature: `[2.0, 123.33]`."

**Seen in:** `main.py`, `X2.mean(axis=0)`

## Standard deviation (`.std`)  `(third-party: numpy)`
**1. Definition:** The standard deviation measures how spread out values are around their mean.
**2. In plain words:** A feature like income (huge spread) and a feature like age (small spread) live on different scales. Knowing each column's std lets you scale them to be comparable.
**3. Example to quote:**
> "`X2.std(axis=0)` is `[0.82, 20.55]`: the second feature varies about 25 times more than the first."

**Seen in:** `main.py`, `X2.std(axis=0)`

## Broadcasting  `(concept)`
**1. Definition:** Broadcasting is NumPy's rule for combining arrays of different shapes by stretching the smaller one across the larger, such as subtracting a `(2,)` row from every row of a `(3, 2)` array.
**2. In plain words:** `(X2 - mean) / std` subtracts the column mean from every sample without a loop. It only works when the shapes line up from the right; otherwise NumPy raises an error.
**3. Example to quote:**
> "`(X2 - mean) / std` subtracts the per-column mean from all three rows at once because NumPy broadcasts the 2-item row."

**Seen in:** `main.py`, `X2_scaled`

## Feature scaling (standardisation)  `(concept)`
**1. Definition:** Standardising features subtracts the mean and divides by the standard deviation so each column has mean about 0 and spread 1.
**2. In plain words:** If one feature ranges 1-3 and another 100-150, gradient descent zig-zags because one parameter needs tiny steps and the other large ones. Scaling makes training smoother and faster. Always compute the mean and std on training data only.
**3. Example to quote:**
> "After scaling, the column means print as about 0, so both features are on a comparable scale."

**Seen in:** `main.py`, `X2_scaled`

## Train/validation split  `(concept)`
**1. Definition:** Splitting data into a training set (used to fit the model) and a validation set (held out to measure how well it generalises).
**2. In plain words:** Testing on the data you trained on is like grading a student on the exact questions they memorised. Keep some data aside to see real performance. Here 4 of 5 rows train and 1 validates (80/20).
**3. Example to quote:**
> "I shuffle the indices and use 4 for training and 1 for validation, so the model is judged on a sample it has not seen."

**Seen in:** `main.py`, `train_idx`, `val_idx`

## Random generator with a seed (`default_rng`)  `(third-party: numpy)`
**1. Definition:** `np.random.default_rng(seed)` creates a pseudo-random number generator; the same seed always yields the same sequence.
**2. In plain words:** Seeding makes experiments reproducible: you and a colleague get the same shuffle. Change the seed to see how results vary.
**3. Example to quote:**
> "With `seed=0` the shuffle always gives train rows `[0, 2, 3, 4]`, so my results are repeatable."

**Seen in:** `main.py`, `rng`

## rng.permutation()  `(third-party: numpy)`
**1. Definition:** `rng.permutation(n)` returns the integers `0..n-1` in random order.
**2. In plain words:** A shuffled list of row positions, used to pick which rows go to training and validation without bias from the original order.
**3. Example to quote:**
> "`rng.permutation(5)` gives a shuffled `[0..4]` that I slice 4 and 1 for the split."

**Seen in:** `main.py`, `indices`

## Slicing arrays (`arr[:4]`, `arr[4:]`)  `(syntax)`
**1. Definition:** `arr[:4]` takes the first four items and `arr[4:]` takes everything from index 4 onwards.
**2. In plain words:** A one-line way to cut the shuffled indices into train and validation parts.
**3. Example to quote:**
> "`indices[:4]` is my training rows and `indices[4:]` is the validation row."

**Seen in:** `main.py`, split lines

## float() and round()  `(Python built-in)`
**1. Definition:** `float(x)` converts a value to a Python float; `round(x, n)` rounds to n decimal places.
**2. In plain words:** NumPy returns special number types that print awkwardly; converting and rounding gives a clean `7.8`.
**3. Example to quote:**
> "`round(float(mse), 3)` prints the loss as `7.8` instead of a long NumPy number."

**Seen in:** `main.py`, `print("MSE loss:", ...)`

## ValueError (shape mismatch)  `(concept)`
**1. Definition:** NumPy raises a `ValueError` when operand shapes are incompatible, for example matmul between `(3, 2)` and `(3,)`.
**2. In plain words:** Read the message: it names the mismatch. The fix is almost always to print `.shape` for each operand and correct the one that is wrong.
**3. Example to quote:**
> "Using three weights for two features raised a matmul `ValueError`; printing the shapes showed `(3, 2)` against `(3,)`."

**Seen in:** `main.py`, BREAK IT comment
