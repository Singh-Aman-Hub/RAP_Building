# Terms — Activations and vanishing gradients

Folder: `15_neural_networks/04_activations_and_vanishing_gradients` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [15 README](../README.md)

**Terms in this folder:** [Activation function (nonlinearity)](#activation-function-nonlinearity) · [Sigmoid (logistic function)](#sigmoid-logistic-function) · [Tanh](#tanh) · [ReLU (rectified linear unit)](#relu-rectified-linear-unit) · [Leaky ReLU (PReLU)](#leaky-relu-prelu) · [GELU](#gelu) · [SiLU (Swish)](#silu-swish) · [Gated linear unit (GLU, GeGLU, SwiGLU)](#gated-linear-unit-glu-geglu-swiglu) · [Vanishing gradient](#vanishing-gradient) · [Exploding gradient](#exploding-gradient) · [Saturation](#saturation) · [Dead ReLU (dying ReLU)](#dead-relu-dying-relu) · [Non-leaf tensor and retain_grad](#non-leaf-tensor-and-retaingrad) · [Xavier (Glorot) vs He (Kaiming) initialisation](#xavier-glorot-vs-he-kaiming-initialisation) · [Frobenius norm](#frobenius-norm) · [Float32 underflow](#float32-underflow) · [intermediate_size (hidden size of the feed-forward block)](#intermediatesize-hidden-size-of-the-feed-forward-block) · [AttributeError: 'NoneType' object has no attribute 'norm'](#attributeerror-nonetype-object-has-no-attribute-norm)

---

## Activation function (nonlinearity)  `(concept)`
**1. Definition:** A nonlinear function applied elementwise to a layer's pre-activation; without it stacked layers collapse to one linear map.
**2. In plain words:** The bend that lets a network draw curves.
**3. Example to quote:** > "Six compared in block 1."
**Seen in:** `main.py`, block 1

## Sigmoid (logistic function)  `(concept)`
**1. Definition:** `sigma(z) = 1 / (1 + e^-z)`, range (0, 1), derivative `sigma(1 - sigma)`, at most 0.25.
**2. In plain words:** An S-curve that squashes any number into a probability-like value.
**3. Example to quote:** > "Slope 0.0025 at z = +-6."
**Seen in:** `main.py`, block 2

## Tanh  `(concept)`
**1. Definition:** `tanh(z)`, range (-1, 1), derivative `1 - tanh^2`, max 1 at 0.
**2. In plain words:** A zero-centred sigmoid.
**3. Example to quote:** > "Stable at depth 20 with Xavier (ratio 0.51)."
**Seen in:** `main.py`

## ReLU (rectified linear unit)  `(concept)`
**1. Definition:** `max(0, z)`; derivative 1 for z > 0, 0 for z < 0.
**2. In plain words:** Pass positives, block negatives.
**3. Example to quote:** > "Cheap, non-saturating on the right."
**Seen in:** `main.py`

## Leaky ReLU (PReLU)  `(concept)`
**1. Definition:** `z` if `z > 0` else `a * z` with small fixed `a` (PReLU: learned `a`).
**2. In plain words:** ReLU that lets a trickle through on the left.
**3. Example to quote:** > "Gradient into always-off units `9e-05` instead of 0."
**Seen in:** `main.py`, block 5

## GELU  `(concept)`
**1. Definition:** `z * Phi(z)` with `Phi` the standard normal CDF; a smooth, slightly non-monotone ReLU.
**2. In plain words:** Scales each input by the probability it is 'on'.
**3. Example to quote:** > "BERT, GPT-2."
**Seen in:** `main.py`

## SiLU (Swish)  `(concept)`
**1. Definition:** `z * sigmoid(z)`.
**2. In plain words:** A smooth ReLU that dips slightly below zero for negative input.
**3. Example to quote:** > "The gate of SwiGLU."
**Seen in:** `main.py`

## Gated linear unit (GLU, GeGLU, SwiGLU)  `(concept)`
**1. Definition:** `act(x W_gate) * (x W_up)`, elementwise product of a gated and an ungated projection (Shazeer 2020).
**2. In plain words:** One projection decides how much of the other gets through, per feature.
**3. Example to quote:** > "LLaMA-style feed-forward blocks."
```python
def forward(self, x):
    return self.down(F.silu(self.gate(x)) * self.up(x))
```
**Seen in:** `main.py`, block 6

## Vanishing gradient  `(concept)`
**1. Definition:** The gradient reaching early layers is exponentially smaller than at late layers because each layer multiplies it by a factor below 1.
**2. In plain words:** The early layers hear no feedback.
**3. Example to quote:** > "Sigmoid 20 layers: ratio 4e-13."
**Seen in:** `main.py`, block 4

## Exploding gradient  `(concept)`
**1. Definition:** The gradient grows exponentially with depth because per-layer factors exceed 1.
**2. In plain words:** Feedback so loud that updates overshoot, giving `inf` or `nan` losses.
**3. Example to quote:** > "Cured by init, normalisation, clipping."
**Seen in:** `WALKTHROUGH.md`, theory

## Saturation  `(concept)`
**1. Definition:** The region of an activation where the slope is near zero.
**2. In plain words:** The flat parts of an S-curve: output stops reacting to input.
**3. Example to quote:** > "Sigmoid and tanh at |z| > 4."
**Seen in:** `main.py`, block 2

## Dead ReLU (dying ReLU)  `(concept)`
**1. Definition:** A ReLU unit whose input is negative for all samples; its output and gradient are 0 so it never recovers.
**2. In plain words:** A switch stuck in the off position.
**3. Example to quote:** > "14 of 32 units dead at `lr=5`."
**Seen in:** `main.py`, block 5

## Non-leaf tensor and retain_grad  `(library)`
**1. Definition:** A tensor computed from others; its `.grad` is discarded unless you call `retain_grad()`.
**2. In plain words:** PyTorch only keeps the gradient of tensors you created.
**3. Example to quote:** > "`h.retain_grad()`"
**Seen in:** `main.py`, block 4

## Xavier (Glorot) vs He (Kaiming) initialisation  `(concept)`
**1. Definition:** Weight variance `2 / (fan_in + fan_out)` (Xavier, for tanh/sigmoid) versus `2 / fan_in` (He, for ReLU).
**2. In plain words:** Pick the starting scale so the signal neither shrinks nor grows.
**3. Example to quote:** > "Full treatment in step 05."
**Seen in:** `main.py`, block 3

## Frobenius norm  `(concept)`
**1. Definition:** The square root of the sum of squares of all entries of a tensor, i.e. the L2 norm of the flattened tensor.
**2. In plain words:** One number for 'how big is this gradient'.
**3. Example to quote:** > "`o.grad.norm()`"
**Seen in:** `main.py`, block 4

## Float32 underflow  `(concept)`
**1. Definition:** A value too small for the format becomes exactly 0 (smallest normal float32 is about 1.2e-38).
**2. In plain words:** A very tiny number is rounded to zero, so a gradient 'vanishes' literally.
**3. Example to quote:** > "Sigmoid at 50 layers: `0.00e+00`."
**Seen in:** `WALKTHROUGH.md`, experiments

## intermediate_size (hidden size of the feed-forward block)  `(concept)`
**1. Definition:** The width of the hidden layer in a transformer MLP, config field `intermediate_size`.
**2. In plain words:** How wide the MLP is inside. 4x d for classic, about 8/3 x d for gated.
**3. Example to quote:** > "SwiGLU `int(8 * 64 / 3) = 170`."
**Seen in:** `main.py`, block 6

## AttributeError: 'NoneType' object has no attribute 'norm'  `(error)`
**1. Definition:** You called a method on `None`; here `.grad` was `None` because the tensor was not a leaf and `retain_grad()` was missing.
**2. In plain words:** Something you expected to exist was never created.
**3. Example to quote:** > "See Debugging practice."
**Seen in:** `WALKTHROUGH.md`, debugging
