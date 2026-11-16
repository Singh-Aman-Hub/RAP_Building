# Terms — Initialisation, normalisation and residual connections

Folder: `15_neural_networks/05_initialization_and_normalization` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Weight initialisation](#weight-initialisation) · [Xavier / Glorot initialisation](#xavier--glorot-initialisation) · [He / Kaiming initialisation](#he--kaiming-initialisation) · [fan_in and fan_out](#fanin-and-fanout) · [Symmetry breaking](#symmetry-breaking) · [Variance propagation](#variance-propagation) · [Normalisation layer](#normalisation-layer) · [Batch normalisation (BatchNorm)](#batch-normalisation-batchnorm) · [Layer normalisation (LayerNorm)](#layer-normalisation-layernorm) · [RMS normalisation (RMSNorm)](#rms-normalisation-rmsnorm) · [Gamma and beta (scale and shift)](#gamma-and-beta-scale-and-shift) · [Epsilon in normalisation](#epsilon-in-normalisation) · [Running mean and variance (BatchNorm buffers)](#running-mean-and-variance-batchnorm-buffers) · [Train mode vs eval mode](#train-mode-vs-eval-mode) · [Residual (skip) connection](#residual-skip-connection) · [Pre-LN and post-LN](#pre-ln-and-post-ln) · [Internal covariate shift](#internal-covariate-shift) · [In-place tensor methods (trailing underscore)](#in-place-tensor-methods-trailing-underscore) · [ValueError: Expected more than 1 value per channel when training](#valueerror-expected-more-than-1-value-per-channel-when-training)

---

## Weight initialisation  `(concept)`
**1. Definition:** The rule used to set parameters before training; it fixes the starting scale of activations and gradients.
**2. In plain words:** Where you place the ball at the top of the hill matters if the hill has 50 steps.
**3. Example to quote:** > "Block 1 compares five rules."
**Seen in:** `main.py`, block 1

## Xavier / Glorot initialisation  `(concept)`
**1. Definition:** Weights with variance `2 / (fan_in + fan_out)`, designed to keep activation and gradient variance constant for linear/tanh units.
**2. In plain words:** A compromise between keeping the forward signal and the backward signal the same size.
**3. Example to quote:** > "`nn.init.xavier_normal_`"
**Seen in:** `main.py`, block 1

## He / Kaiming initialisation  `(concept)`
**1. Definition:** Weights with variance `2 / fan_in`, compensating for ReLU zeroing half of its inputs.
**2. In plain words:** Xavier times sqrt 2 for ReLU networks.
**3. Example to quote:** > "ReLU std stays around 0.4 to 0.8 over 50 layers."
**Seen in:** `main.py`, block 1

## fan_in and fan_out  `(concept)`
**1. Definition:** The number of input / output connections of a unit's layer.
**2. In plain words:** Width of the layer before / after.
**3. Example to quote:** > "`nn.Linear(256, 256)`: both 256."
**Seen in:** `main.py`, block 1

## Symmetry breaking  `(concept)`
**1. Definition:** Random initial weights make units differ, so they receive different gradients and can specialise.
**2. In plain words:** Identical twins given the same instructions never diverge.
**3. Example to quote:** > "Zero init: 8 hidden units are copies (difference 0.0)."
**Seen in:** `main.py`, block 2

## Variance propagation  `(concept)`
**1. Definition:** `Var(y) = n_in * Var(w) * Var(x)` for a linear layer with independent zero-mean weights and inputs.
**2. In plain words:** The arithmetic behind picking weight scales.
**3. Example to quote:** > "Choose `Var(w) = 1 / n_in` to keep variance constant."
**Seen in:** `WALKTHROUGH.md`, theory

## Normalisation layer  `(concept)`
**1. Definition:** A layer that rescales activations to a fixed mean and variance (or RMS) then applies a learned scale and shift.
**2. In plain words:** Resets the volume knob at every layer.
**3. Example to quote:** > "BatchNorm, LayerNorm, RMSNorm."
**Seen in:** `main.py`, block 3

## Batch normalisation (BatchNorm)  `(concept)`
**1. Definition:** Normalises each feature over the batch (and spatial dims); keeps running mean/variance for inference.
**2. In plain words:** Compares each sample with the rest of its batch.
**3. Example to quote:** > "`my_batchnorm` uses `dim=0`."
**Seen in:** `main.py`, blocks 3-4

## Layer normalisation (LayerNorm)  `(concept)`
**1. Definition:** Normalises each sample over its feature dimension, independent of the batch.
**2. In plain words:** Each token standardises itself.
**3. Example to quote:** > "`my_layernorm` uses `dim=-1`."
**Seen in:** `main.py`, block 3

## RMS normalisation (RMSNorm)  `(concept)`
**1. Definition:** `y = x / sqrt(mean(x^2) + eps) * gamma`: LayerNorm without mean subtraction and without bias.
**2. In plain words:** Only rescale, do not recentre.
**3. Example to quote:** > "Output has root-mean-square 1 but mean 0.86."
**Seen in:** `main.py`, block 3

## Gamma and beta (scale and shift)  `(concept)`
**1. Definition:** Learned per-feature parameters applied after normalising (`weight` and `bias` in PyTorch).
**2. In plain words:** Lets the network undo the normalisation where that helps.
**3. Example to quote:** > "RMSNorm has gamma only."
**Seen in:** `main.py`, block 3

## Epsilon in normalisation  `(concept)`
**1. Definition:** A tiny constant added under the square root to avoid division by zero.
**2. In plain words:** A safety net.
**3. Example to quote:** > "`1e-5` in all three."
**Seen in:** `main.py`, block 3

## Running mean and variance (BatchNorm buffers)  `(concept)`
**1. Definition:** Exponential moving averages of the batch statistics updated in training and used in eval mode.
**2. In plain words:** BatchNorm's memory of typical data.
**3. Example to quote:** > "`bn.running_mean ~ 5`, `running_var ~ 9`."
**Seen in:** `main.py`, block 4

## Train mode vs eval mode  `(concept)`
**1. Definition:** `model.train()` / `model.eval()` switch layers such as BatchNorm and Dropout between training and inference behaviour.
**2. In plain words:** Practice behaviour versus match behaviour.
**3. Example to quote:** > "Forgetting `eval()` makes outputs depend on the batch."
**Seen in:** `main.py`, block 4

## Residual (skip) connection  `(concept)`
**1. Definition:** Adding a layer's input to its output: `h + f(h)`.
**2. In plain words:** A shortcut lane around each layer.
**3. Example to quote:** > "Sigmoid net 50 layers: gradient ratio 0.0 -> 1.27."
**Seen in:** `main.py`, block 5

## Pre-LN and post-LN  `(concept)`
**1. Definition:** Pre-LN applies the norm to the sublayer input (`x + F(Norm(x))`); post-LN to the sum (`Norm(x + F(x))`).
**2. In plain words:** Is the shortcut lane clean (pre) or normalised (post)?
**3. Example to quote:** > "Most modern LLMs use pre-LN (or variants)."
**Seen in:** `WALKTHROUGH.md`, theory

## Internal covariate shift  `(concept)`
**1. Definition:** The original (disputed) explanation of BatchNorm: layer input distributions drift during training.
**2. In plain words:** Better to say 'norms stabilise scale and smooth optimisation'.
**3. Example to quote:** > "Do not assert it as settled."
**Seen in:** `WALKTHROUGH.md`, theory

## In-place tensor methods (trailing underscore)  `(syntax)`
**1. Definition:** Methods ending in `_` modify the tensor in place, e.g. `W.normal_()`, `nn.init.zeros_`.
**2. In plain words:** No new tensor is created.
**3. Example to quote:** > "`W.zero_()`"
**Seen in:** `main.py`, block 1

## ValueError: Expected more than 1 value per channel when training  `(error)`
**1. Definition:** BatchNorm in train mode cannot compute a variance from a single sample.
**2. In plain words:** Use eval mode, a bigger batch or LayerNorm.
**3. Example to quote:** > "`bn.train(); bn(sample)`"
**Seen in:** `WALKTHROUGH.md`, debugging
