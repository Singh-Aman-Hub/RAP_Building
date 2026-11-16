# Terms — Regularisation

Folder: `15_neural_networks/07_regularization` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Regularisation](#regularisation) · [Generalisation gap](#generalisation-gap) · [Overfitting and underfitting](#overfitting-and-underfitting) · [L2 regularisation (ridge)](#l2-regularisation-ridge) · [Decoupled weight decay (AdamW)](#decoupled-weight-decay-adamw) · [Weight norm](#weight-norm) · [Dropout](#dropout) · [Inverted dropout](#inverted-dropout) · [Early stopping](#early-stopping) · [Patience](#patience) · [Selection bias of the validation set](#selection-bias-of-the-validation-set) · [Data augmentation](#data-augmentation) · [Label smoothing](#label-smoothing) · [Calibration and over-confidence](#calibration-and-over-confidence) · [train() and eval() modes](#train-and-eval-modes) · [state_dict (checkpoint)](#statedict-checkpoint) · [Cloning tensors in a snapshot](#cloning-tensors-in-a-snapshot) · [Hyperparameter vs parameter](#hyperparameter-vs-parameter)

---

## Regularisation  `(concept)`
**1. Definition:** Any technique that constrains a model to reduce overfitting (variance) at some cost in fit.
**2. In plain words:** Making the model prefer simpler explanations.
**3. Example to quote:** > "Nine variants compared in block 4."
**Seen in:** `main.py`

## Generalisation gap  `(concept)`
**1. Definition:** The difference between a metric on the training data and on unseen data.
**2. In plain words:** Exam score minus homework score.
**3. Example to quote:** > "Baseline gap 0.183."
**Seen in:** `main.py`, block 4

## Overfitting and underfitting  `(concept)`
**1. Definition:** Overfitting: low training error, high validation error (memorising). Underfitting: both errors high (too simple or too regularised).
**2. In plain words:** Memorising answers versus not learning enough.
**3. Example to quote:** > "Adam + L2 0.5 collapsed to chance (underfit)."
**Seen in:** `main.py`, experiments

## L2 regularisation (ridge)  `(concept)`
**1. Definition:** Adding `(wd/2) * ||w||^2` to the loss, i.e. `wd * w` to the gradient.
**2. In plain words:** Large weights are taxed.
**3. Example to quote:** > "`Adam(weight_decay=0.05)`"
**Seen in:** `main.py`, block 3

## Decoupled weight decay (AdamW)  `(concept)`
**1. Definition:** Shrinking weights by `lr * wd` each step independently of the gradient.
**2. In plain words:** A fixed percentage shrink, not tied to gradient size.
**3. Example to quote:** > "Weight norm stayed 29.7 with `wd=0.05`."
**Seen in:** `main.py`, block 3

## Weight norm  `(concept)`
**1. Definition:** The L2 norm of all weight tensors, a measure of model 'size'.
**2. In plain words:** Smaller usually means smoother functions.
**3. Example to quote:** > "30.34 baseline vs 1.83 with Adam+L2."
**Seen in:** `main.py`, block 3

## Dropout  `(concept)`
**1. Definition:** Randomly zeroing each unit with probability `p` during training.
**2. In plain words:** Training with random absences so no unit can be relied upon.
**3. Example to quote:** > "`nn.Dropout(0.3)`"
**Seen in:** `main.py`, block 5

## Inverted dropout  `(concept)`
**1. Definition:** Scaling the surviving activations by `1/(1-p)` during training so inference needs no rescaling.
**2. In plain words:** Compensate now, not later.
**3. Example to quote:** > "`x * mask / (1 - p)`"
**Seen in:** `main.py`, block 5

## Early stopping  `(concept)`
**1. Definition:** Stop training when validation loss stops improving; restore the best checkpoint.
**2. In plain words:** Quit while ahead.
**3. Example to quote:** > "Epoch 78 of 500; val loss 0.315."
**Seen in:** `main.py`, block 3

## Patience  `(concept)`
**1. Definition:** The number of epochs to wait for an improvement before stopping.
**2. In plain words:** How many bad days before giving up.
**3. Example to quote:** > "`patience=40`"
**Seen in:** `main.py`, block 3

## Selection bias of the validation set  `(concept)`
**1. Definition:** Choosing the epoch or hyperparameters by validation score makes that score optimistic; use a separate untouched test set to report.
**2. In plain words:** Grading yourself on the exam you used to pick your notes.
**3. Example to quote:** > "0.875 is upward-biased."
**Seen in:** `WALKTHROUGH.md`, theory

## Data augmentation  `(concept)`
**1. Definition:** Training on label-preserving transformed copies of the data.
**2. In plain words:** More examples for free.
**3. Example to quote:** > "Gaussian input noise std 0.15."
**Seen in:** `main.py`, block 3

## Label smoothing  `(concept)`
**1. Definition:** Replacing one-hot targets by `1-e` for the true class and spreading `e` over the classes.
**2. In plain words:** Never claim 100 percent certainty.
**3. Example to quote:** > "`label_smoothing=0.1`"
```python
F.cross_entropy(logits, y, label_smoothing=0.1)
```
**Seen in:** `main.py`, block 3

## Calibration and over-confidence  `(concept)`
**1. Definition:** A model is calibrated if predicted probability matches observed frequency; overfit nets are over-confident.
**2. In plain words:** Saying 99 percent and being right 80 percent of the time.
**3. Example to quote:** > "Val loss 1.64 at 81.7 percent accuracy."
**Seen in:** `WALKTHROUGH.md`, trace

## train() and eval() modes  `(library)`
**1. Definition:** Switches layers such as Dropout and BatchNorm between training and inference behaviour.
**2. In plain words:** Practice versus exam mode.
**3. Example to quote:** > "`net.eval()` in `evaluate`."
**Seen in:** `main.py`, block 3

## state_dict (checkpoint)  `(library)`
**1. Definition:** A dict of parameter and buffer tensors; `.clone()` copies them for later restoring.
**2. In plain words:** A save-game of the weights.
**3. Example to quote:** > "`best = (..., {k: v.clone() ...})`"
**Seen in:** `main.py`, block 3

## Cloning tensors in a snapshot  `(syntax)`
**1. Definition:** `state_dict()` returns references; without `.clone()` the 'saved' weights keep changing with training.
**2. In plain words:** A photo versus a window.
**3. Example to quote:** > "`v.clone()`"
**Seen in:** `main.py`, block 3

## Hyperparameter vs parameter  `(concept)`
**1. Definition:** Parameters are learned (weights); hyperparameters are chosen (`wd`, `p`, `patience`).
**2. In plain words:** Settings versus learned knowledge.
**3. Example to quote:** > "All the knobs in `runs`."
**Seen in:** `main.py`, block 4
