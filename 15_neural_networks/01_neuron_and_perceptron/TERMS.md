# Terms — The neuron and the perceptron

Folder: `15_neural_networks/01_neuron_and_perceptron` · Files covered: `main.py` (perceptron + logistic regression), `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Artificial neuron](#artificial-neuron) · [Weights and bias (parameters)](#weights-and-bias-parameters) · [Pre-activation (logit, net input)](#pre-activation-logit-net-input) · [Activation function](#activation-function) · [Step function (Heaviside)](#step-function-heaviside) · [Perceptron learning rule](#perceptron-learning-rule) · [Perceptron convergence theorem](#perceptron-convergence-theorem) · [Linear separability](#linear-separability) · [Decision boundary (hyperplane)](#decision-boundary-hyperplane) · [XOR problem](#xor-problem) · [Hidden layer](#hidden-layer) · [Feature cross](#feature-cross) · [Universal approximation theorem](#universal-approximation-theorem) · [Epoch](#epoch) · [Matrix product operator @](#matrix-product-operator-) · [In-place update (+=)](#in-place-update-) · [ValueError: matmul ... mismatch in its core dimension](#valueerror-matmul--mismatch-in-its-core-dimension) · [Logistic regression (logit model, sigmoid classifier)](#logistic-regression-logit-model-sigmoid-classifier--concept) · [Sigmoid (logistic function, expit)](#sigmoid-logistic-function-expit--function) · [Binary cross-entropy (BCE, log loss, negative log-likelihood)](#binary-cross-entropy-bce-log-loss-negative-log-likelihood--function) · [Gradient (p - y) x](#gradient-p---y-x--concept) · [Log-odds (logit) and odds](#log-odds-logit-and-odds--concept) · [Probability calibration (reliability, ECE, temperature scaling)](#probability-calibration-reliability-ece-temperature-scaling--concept) · [Numerical gradient check (central difference)](#numerical-gradient-check-central-difference--function) · [scikit-learn LogisticRegression (C, L-BFGS)](#scikit-learn-logisticregression-c-l-bfgs--library)

---

## Artificial neuron  `(concept)`
**1. Definition:** A function `a = f(x . w + b)`: a weighted sum of the inputs plus a bias, passed through an activation function `f`.
**2. In plain words:** One voter per input with a voting power (weight), an entry fee (bias) and a decision rule (activation).
**3. Example to quote:** > "A neuron computes one dot product, adds a bias and applies a nonlinearity; a layer is many of them side by side."
```python
def neuron(x, w, b, activation=step):
    return activation(x @ w + b)
```
**Seen in:** `main.py`, block 1

## Weights and bias (parameters)  `(concept)`
**1. Definition:** Weights `w` are the numbers multiplying each input; the bias `b` is a number added to the sum. Together with other learned numbers they are the model's parameters.
**2. In plain words:** Weights: how much each input matters. Bias: the default lean when all inputs are zero.
**3. Example to quote:** > "The decision boundary is `x . w + b = 0`: `w` sets its direction, `b` its offset."
**Seen in:** `main.py`, blocks 1-2

## Pre-activation (logit, net input)  `(concept)`
**1. Definition:** The raw value `z = x . w + b` before the activation is applied.
**2. In plain words:** The score before it is squashed or thresholded. In the last layer of a classifier it is called logits.
**3. Example to quote:** > "Here `z = 0.1` for `x=[1,2]`, and the step turned it into 1."
**Seen in:** `main.py`, block 1

## Activation function  `(concept)`
**1. Definition:** A (usually nonlinear) function applied to `z` to produce the neuron's output.
**2. In plain words:** The part that makes layers more powerful than one matrix. Without it, depth collapses into a single linear map.
**3. Example to quote:** > "Step 04 compares sigmoid, tanh, ReLU, GELU and SiLU."
**Seen in:** `main.py`, `step`

## Step function (Heaviside)  `(concept)`
**1. Definition:** `step(z) = 1 if z >= 0 else 0`.
**2. In plain words:** A hard on/off switch. Its derivative is 0 everywhere except at 0, so gradient descent cannot use it.
**3. Example to quote:** > "This is why modern networks use smooth activations and backpropagation instead of the perceptron rule."
**Seen in:** `main.py`, `step`

## Perceptron learning rule  `(concept)`
**1. Definition:** For each sample: `w += lr * (y - pred) * x` and `b += lr * (y - pred)`.
**2. In plain words:** Fix mistakes by moving the line toward the missed point; do nothing when right.
**3. Example to quote:** > "It converged on AND after 6 epochs and never on XOR."
```python
update = lr * (yi - pred)
w += update * xi
b += update
```
**Seen in:** `main.py`, `train_perceptron`

## Perceptron convergence theorem  `(concept)`
**1. Definition:** If the data are linearly separable with margin `gamma` and all inputs have norm at most `R`, the perceptron makes at most `(R/gamma)^2` mistakes and then stops.
**2. In plain words:** A guarantee: separable data means it will finish; nothing is promised otherwise.
**3. Example to quote:** > "XOR is outside the guarantee, so it cycles forever."
**Seen in:** `WALKTHROUGH.md`, theory

## Linear separability  `(concept)`
**1. Definition:** Two classes are linearly separable if a hyperplane has all of one class on one side and all of the other on the other side.
**2. In plain words:** 'Can one straight cut split them?'
**3. Example to quote:** > "AND and OR: yes. XOR: no (best line: 3 of 4)."
**Seen in:** `main.py`, blocks 3 and 5

## Decision boundary (hyperplane)  `(concept)`
**1. Definition:** The set of points where `x . w + b = 0`; a line in 2-D, a plane in 3-D, a hyperplane in n-D.
**2. In plain words:** The fence between the two predictions.
**3. Example to quote:** > "For the blobs the fence was `1.91*x1 + 2.09*x2 - 1 = 0`."
**Seen in:** `main.py`, block 4

## XOR problem  `(concept)`
**1. Definition:** XOR outputs 1 when exactly one of two binary inputs is 1; the two classes lie on opposite diagonals, so no line separates them.
**2. In plain words:** The classic example showing a single neuron is too weak.
**3. Example to quote:** > "Minsky and Papert (1969); a hidden layer solves it."
**Seen in:** `main.py`, blocks 3, 5, 6

## Hidden layer  `(concept)`
**1. Definition:** A layer between input and output whose values are not observed directly; it re-represents the input.
**2. In plain words:** A set of helper neurons that compute useful intermediate features, here OR and AND.
**3. Example to quote:** > "`h = step(X @ W1 + b1)` with shape `(4, 2)`."
**Seen in:** `main.py`, block 6

## Feature cross  `(concept)`
**1. Definition:** A hand-made feature that multiplies inputs (`x1*x2`) to make a problem linearly separable.
**2. In plain words:** A manual hidden unit. Networks learn these automatically.
**3. Example to quote:** > "With `x1*x2` added, XOR converged in 8 epochs."
**Seen in:** `WALKTHROUGH.md`, experiments

## Universal approximation theorem  `(concept)`
**1. Definition:** A feed-forward network with one hidden layer and a non-polynomial activation can approximate any continuous function on a compact set to any accuracy, given enough hidden units.
**2. In plain words:** Networks are expressive enough; it does not say how to train them or how big they must be.
**3. Example to quote:** > "Say it and add: 'existence, not learnability'."
**Seen in:** `WALKTHROUGH.md`, theory

## Epoch  `(concept)`
**1. Definition:** One full pass over the training data.
**2. In plain words:** Every sample seen once. The inner `for xi, yi in zip(X, y)` loop is one epoch.
**3. Example to quote:** > "AND needed 6 epochs."
**Seen in:** `main.py`, `train_perceptron`

## Matrix product operator @  `(syntax)`
**1. Definition:** `a @ b` is matrix multiplication (`np.matmul`); on 1-D arrays it is the dot product.
**2. In plain words:** The 'multiply and add up' of a neuron in one symbol.
**3. Example to quote:** > "`x @ w + b`"
**Seen in:** `main.py`, block 1

## In-place update (+=)  `(syntax)`
**1. Definition:** `w += v` modifies the array object `w` instead of creating a new one.
**2. In plain words:** Edits the same array; be careful if another variable points to it.
**3. Example to quote:** > "`w += update * xi`"
**Seen in:** `main.py`, block 2

## ValueError: matmul ... mismatch in its core dimension  `(error)`
**1. Definition:** Raised when the inner dimensions of `@` do not agree.
**2. In plain words:** The number of columns on the left must equal the number of rows on the right.
**3. Example to quote:** > "`(4,2) @ (3,)` fails because 2 != 3."
**Seen in:** `WALKTHROUGH.md`, debugging

## Logistic regression (logit model, sigmoid classifier)  `(concept)`
**1. Definition:** A binary classifier `p = sigmoid(w.x + b)` trained by minimising binary cross-entropy; equivalently one neuron with a sigmoid output. Its logit `w.x + b` is the log-odds of class 1.
**2. In plain words:** A straight-line classifier that reports a probability instead of a hard yes/no.
**3. Example to quote:** > "Logistic regression is a single neuron with a sigmoid, trained with log-loss; the boundary is where w.x + b = 0."
```python
p = sigmoid(X @ w + b)
```
**Seen in:** `main.py`, blocks 7-9

## Sigmoid (logistic function, expit)  `(function)`
**1. Definition:** `sigmoid(z) = 1 / (1 + e^-z)`, which maps any real number to (0, 1); derivative `p (1 - p)`.
**2. In plain words:** A smooth switch that turns a score into a probability.
**3. Example to quote:** > "Compute it from exp(-|z|) so it never overflows."
```python
ez = np.exp(-np.abs(z))
np.where(z >= 0, 1 / (1 + ez), ez / (1 + ez))
```
**Seen in:** `main.py`, `sigmoid`

## Binary cross-entropy (BCE, log loss, negative log-likelihood)  `(function)`
**1. Definition:** `L = -[y ln p + (1-y) ln(1-p)]`, averaged over samples; the negative log-likelihood of a Bernoulli model, minimised by maximum likelihood.
**2. In plain words:** Heavily punishes being confident and wrong.
**3. Example to quote:** > "A coin-flip prediction scores ln 2 = 0.693; a confident miss costs a lot."
```python
-np.mean(y*np.log(p) + (1-y)*np.log(1-p))
```
**Seen in:** `main.py`, `bce`

## Gradient (p - y) x  `(concept)`
**1. Definition:** For sigmoid + BCE, `dL/dw = (p - y) x` and `dL/db = p - y`: the sigmoid derivative cancels the loss's `1/p` terms, leaving the residual times the input.
**2. In plain words:** Move each weight in proportion to how wrong the probability was and how large that feature was.
**3. Example to quote:** > "Same residual form as linear regression and softmax cross-entropy."
```python
gw = X.T @ (p - y) / len(y)
```
**Seen in:** `main.py`, `logreg_grad`

## Log-odds (logit) and odds  `(concept)`
**1. Definition:** Odds = `p/(1-p)`; log-odds (logit) = `ln(p/(1-p))`, the inverse of the sigmoid. In logistic regression the logit is linear: `w.x + b`.
**2. In plain words:** How many times likelier yes is than no, on a log scale.
**3. Example to quote:** > "A weight is the change in log-odds per unit of that feature; e^w multiplies the odds."
```python
odds = np.exp(z)   # = p / (1 - p)
```
**Seen in:** `main.py`, block 9

## Probability calibration (reliability, ECE, temperature scaling)  `(concept)`
**1. Definition:** A model is calibrated if predictions of `p` are right about a fraction `p` of the time. ECE is the count-weighted gap between mean predicted and observed frequency per bin; temperature scaling divides the logits by a scalar `T` fitted on validation data.
**2. In plain words:** Does '80% sure' mean right 8 times out of 10?
**3. Example to quote:** > "Same accuracy, worse calibration: scaling the logits by 3 doubled the ECE."
```python
over = sigmoid(3 * (X @ w + b))
```
**Seen in:** `main.py`, block 11

## Numerical gradient check (central difference)  `(function)`
**1. Definition:** Estimate `dL/dtheta` as `(L(theta+h) - L(theta-h)) / 2h` and compare with the analytic gradient; a max difference around 1e-10 (for h = 1e-6) means the formula is right.
**2. In plain words:** Measure the slope with a ruler to prove the algebra.
**3. Example to quote:** > "Always gradient-check a hand-derived backward pass."
```python
(bce(w + h*e) - bce(w - h*e)) / (2*h)
```
**Seen in:** `main.py`, block 7

## scikit-learn LogisticRegression (C, L-BFGS)  `(library)`
**1. Definition:** `sklearn.linear_model.LogisticRegression`: fits logistic regression with an L2 penalty of strength `1/C` (default C=1.0) using an optimiser such as L-BFGS; `predict_proba` returns `(n, 2)` probabilities.
**2. In plain words:** The standard library version, regularised by default.
**3. Example to quote:** > "C=1e6 means almost no penalty, so it matches our maximum-likelihood fit."
```python
LogisticRegression(C=1e6).fit(X, y).coef_
```
**Seen in:** `main.py`, block 12
