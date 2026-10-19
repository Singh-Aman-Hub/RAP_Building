# Terms — Linear regression in NumPy

Folder: `03_ml_foundations/02_linear_regression_numpy` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [ML foundations README](../README.md)

**Terms in this folder:** [Linear regression](#linear-regression--concept) · [Gradient descent](#gradient-descent--concept) · [Gradient](#gradient--concept) · [Learning rate](#learning-rate--concept) · [Epoch](#epoch--concept) · [Batch size and mini-batch](#batch-size-and-mini-batch--concept) · [Forward pass](#forward-pass--concept) · [Training loop (forward → loss → gradient → update)](#training-loop-forward--loss--gradient--update--concept) · [Hyperparameter](#hyperparameter--concept) · [Divergence (exploding loss)](#divergence-exploding-loss--concept) · [Noise floor](#noise-floor--concept) · [Default parameter values](#default-parameter-values--syntax) · [Returning multiple values (tuple unpacking)](#returning-multiple-values-tuple-unpacking--syntax) · [range() with a step](#range-with-a-step--python-built-in) · [Fancy indexing](#fancy-indexing--third-party-numpy) · [Shuffling each epoch (`rng.permutation`)](#shuffling-each-epoch-rngpermutation--third-party-numpy) · [Random data generation (`rng.uniform`, `rng.normal`)](#random-data-generation-rnguniform-rngnormal--third-party-numpy) · [Compound update (`w -= lr * grad`)](#compound-update-w---lr--grad--syntax) · [Ceiling division trick (`-(-a // b)`)](#ceiling-division-trick---a--b--syntax) · [`or` for default values (`batch_size or n`)](#or-for-default-values-batch_size-or-n--syntax) · [Loss history](#loss-history--concept)

---

## Linear regression  `(concept)`
**1. Definition:** Linear regression fits a straight-line model `y_hat = w * x + b` to data by choosing `w` and `b` that minimise the squared prediction error.
**2. In plain words:** It draws the best straight line through the points. The data here follows `y = 2x + 1` plus a little noise, so training should find a weight near 2 and a bias near 1. It cannot model curves unless you add features.
**3. Example to quote:**
> "After 50 epochs my line learned `w = 1.943` and `b = 0.997`, close to the true 2 and 1."

**Seen in:** `main.py`, `train`

## Gradient descent  `(concept)`
**1. Definition:** Gradient descent is an optimisation method that repeatedly changes parameters in the direction that reduces the loss, using the gradient (the slope of the loss).
**2. In plain words:** Imagine standing on a hill in fog and always stepping downhill. The gradient tells you which way is downhill, and the learning rate decides the step size. Every neural network, including LLMs, is trained with a version of this.
**3. Example to quote:**
> "Each update computes the gradient and moves against it: `w -= learning_rate * grad_w`. Repeating that lowered the loss from 2.04 to 0.01."

**Seen in:** `main.py`, update lines

## Gradient  `(concept)`
**1. Definition:** The gradient is the derivative of the loss with respect to a parameter: how much the loss changes if that parameter changes slightly.
**2. In plain words:** A positive gradient says "increasing this parameter increases the loss", so you decrease it. Here it is computed by hand: `2 * mean(error * x)` for the weight and `2 * mean(error)` for the bias. PyTorch computes it automatically in the next step.
**3. Example to quote:**
> "`grad_w = 2 * np.mean(error * xb)` tells me which direction to nudge the weight to reduce the squared error."

**Seen in:** `main.py`, `grad_w`, `grad_b`

## Learning rate  `(concept)`
**1. Definition:** The learning rate is the multiplier on the gradient that sets the size of each parameter update.
**2. In plain words:** Too small and training crawls (lr 0.001 barely moved); too large and each step overshoots so the loss explodes (lr 1.1 gave a loss of 7.5e+08). A good value converges quickly (lr 0.1). It is the first hyperparameter to tune.
**3. Example to quote:**
> "With lr=0.001 the loss was still 2.4 after 50 epochs; with lr=1.1 it diverged to 7.5e+08. The learning rate controls step size."

**Seen in:** `main.py`, `learning_rate` parameter, Experiment A

## Epoch  `(concept)`
**1. Definition:** An epoch is one complete pass of the training procedure over the whole training dataset.
**2. In plain words:** If you have 200 samples, one epoch has seen all 200. More epochs usually lower the loss until it levels off at the noise floor, and too many on a big model can overfit.
**3. Example to quote:**
> "After 1 epoch the loss was high; after 50 epochs it flattened near 0.011. More epochs help only until the loss stops improving."

**Seen in:** `main.py`, `epochs`, Experiment C

## Batch size and mini-batch  `(concept)`
**1. Definition:** Batch size is the number of samples used to compute one gradient update; a mini-batch is a subset smaller than the full dataset.
**2. In plain words:** Full-batch (200 samples) makes one smooth update per epoch; batch size 1 makes 200 noisy updates per epoch. Smaller batches give more updates per epoch but noisier steps; larger batches need more memory.
**3. Example to quote:**
> "With batch size 200 I got 5 updates in 5 epochs and loss 0.75; with batch size 32 I got 35 updates and loss 0.017. More updates per epoch means faster progress."

**Seen in:** `main.py`, `batch_size`, Experiment B

## Forward pass  `(concept)`
**1. Definition:** The forward pass computes the model's predictions from its inputs and current parameters.
**2. In plain words:** Step one of every training iteration: `y_hat = w * xb + b`. Then you measure the error and work backwards to update.
**3. Example to quote:**
> "The forward pass is just `y_hat = w * xb + b`; the later steps decide how to fix `w` and `b`."

**Seen in:** `main.py`, `y_hat = w * xb + b`

## Training loop (forward → loss → gradient → update)  `(concept)`
**1. Definition:** The training loop repeats four steps: predict, measure the loss, compute gradients, update parameters.
**2. In plain words:** This is the heart of all machine learning. The same four steps train a straight line here and a billion-parameter LLM in module 06; only the model and scale differ.
**3. Example to quote:**
> "Each iteration: forward pass, compute error, compute gradients, update weights. I repeat it per mini-batch for several epochs."

**Seen in:** `main.py`, `train`

## Hyperparameter  `(concept)`
**1. Definition:** A hyperparameter is a setting chosen by you before training (such as learning rate, epochs or batch size) rather than learned from the data.
**2. In plain words:** The model learns `w` and `b`; you pick `lr`, `epochs` and `batch_size`. Tuning them is half of practical ML. This folder compares each one systematically by changing only one at a time.
**3. Example to quote:**
> "The weights are learned, but learning rate, epochs and batch size are hyperparameters I set and compare."

**Seen in:** `main.py`, `train` docstring

## Divergence (exploding loss)  `(concept)`
**1. Definition:** Divergence is when updates overshoot so badly that the loss grows instead of shrinking, often because the learning rate is too high.
**2. In plain words:** Each step jumps past the valley and lands higher up the other side, and it gets worse every step until numbers become huge. If your loss goes up or becomes `nan`, lower the learning rate first.
**3. Example to quote:**
> "At lr=1.1 the weight went to -3347 and the loss to 7.5e+08. That is divergence."

**Seen in:** `main.py`, Experiment A

## Noise floor  `(concept)`
**1. Definition:** The noise floor is the minimum loss a model can reach because the data itself contains random variation that no model can predict.
**2. In plain words:** The data has noise of standard deviation 0.1, so the best possible MSE is about 0.01. If your loss plateaus there, the model is not broken; the remaining error is noise.
**3. Example to quote:**
> "My loss flattens at about 0.011, matching the noise variance 0.1², so it cannot get meaningfully lower."

**Seen in:** `main.py`, TRY IT comment

## Default parameter values  `(syntax)`
**1. Definition:** In `def train(X, y, learning_rate=0.1, ...)`, the `=0.1` part is a default used when the caller does not pass that argument.
**2. In plain words:** It lets you call `train(X, y)` simply, or override just one setting: `train(X, y, learning_rate=0.9)`.
**3. Example to quote:**
> "I call `train(X, y, learning_rate=0.001)` and the other arguments fall back to their defaults."

**Seen in:** `main.py`, `train` signature

## Returning multiple values (tuple unpacking)  `(syntax)`
**1. Definition:** `return w, b, history` returns a tuple, which the caller can unpack into separate names.
**2. In plain words:** `w, b, history = train(...)` receives three results at once. Use `_` for any you do not need.
**3. Example to quote:**
> "`w_, b_, h = train(...)` unpacks the learned weight, bias and loss history."

**Seen in:** `main.py`, `train`

## range() with a step  `(Python built-in)`
**1. Definition:** `range(start, stop, step)` counts from start up to stop in increments of step.
**2. In plain words:** `range(0, n, batch_size)` yields the start of each mini-batch: 0, 32, 64 and so on, so slicing `order[start:start + batch_size]` walks through the data in chunks.
**3. Example to quote:**
> "`list(range(0, 10, 4))` is `[0, 4, 8]`, the start index of each batch of 4."

**Seen in:** `main.py`, `train`

## Fancy indexing  `(third-party: numpy)`
**1. Definition:** Indexing a NumPy array with an array of integers (`X[idx]`) selects those specific positions.
**2. In plain words:** With `idx` a shuffled slice of positions, `X[idx]` and `y[idx]` give the matching shuffled batch for features and labels.
**3. Example to quote:**
> "`xb, yb = X[idx], y[idx]` gives me the same random rows from both features and labels."

**Seen in:** `main.py`, `train`

## Shuffling each epoch (`rng.permutation`)  `(third-party: numpy)`
**1. Definition:** Re-ordering the training samples randomly at the start of each epoch before forming batches.
**2. In plain words:** Without shuffling, the model sees the same batches in the same order every time and can pick up that order as a pattern. Shuffling gives each epoch fresh batches.
**3. Example to quote:**
> "`order = rng.permutation(n)` reshuffles the sample order at the start of every epoch."

**Seen in:** `main.py`, `train`

## Random data generation (`rng.uniform`, `rng.normal`)  `(third-party: numpy)`
**1. Definition:** `rng.uniform(lo, hi, size)` draws numbers evenly between two limits; `rng.normal(mean, std, size)` draws bell-curve (Gaussian) numbers.
**2. In plain words:** The script fabricates data: inputs between -1 and 1, and labels `2x + 1` plus small Gaussian noise. Synthetic data lets you know the true answer and check the model recovers it.
**3. Example to quote:**
> "I generate `y = 2x + 1` plus noise of standard deviation 0.1, so I know the right answer is `w=2, b=1`."

**Seen in:** `main.py`, data generation lines

## Compound update (`w -= lr * grad`)  `(syntax)`
**1. Definition:** `w -= learning_rate * grad_w` subtracts the scaled gradient from `w` in place.
**2. In plain words:** Subtracting moves downhill. Writing `+=` by mistake walks uphill and the loss climbs, which is the BREAK IT drill.
**3. Example to quote:**
> "The minus sign is the 'descent': I move against the gradient. With `+=` the loss would climb."

**Seen in:** `main.py`, update step

## Ceiling division trick (`-(-a // b)`)  `(syntax)`
**1. Definition:** `-(-a // b)` computes `ceil(a / b)` using integer floor division on the negated value.
**2. In plain words:** A batch size of 32 over 200 samples needs 7 batches (the last is smaller). The trick counts updates per epoch without importing `math`.
**3. Example to quote:**
> "`-(-200 // 32)` is 7, so batch size 32 takes 7 updates per epoch."

**Seen in:** `main.py`, Experiment B

## `or` for default values (`batch_size or n`)  `(syntax)`
**1. Definition:** `a or b` returns `a` if it is truthy, otherwise `b`.
**2. In plain words:** `batch_size = batch_size or n` means "use the whole dataset if no batch size was given" (None is falsy). A subtle catch: it would also replace 0, so check for `None` explicitly when 0 is valid.
**3. Example to quote:**
> "If I pass `batch_size=None`, the line falls back to `n`, so training runs full-batch."

**Seen in:** `main.py`, `train`

## Loss history  `(concept)`
**1. Definition:** A loss history is a list recording the loss after each epoch, used to see whether training is converging.
**2. In plain words:** Plotting or printing it is the quickest way to spot problems: falling means learning, flat means converged, rising means divergence.
**3. Example to quote:**
> "`history[-1]` is the final loss, and printing it for each learning rate shows which settings converged."

**Seen in:** `main.py`, `history`
