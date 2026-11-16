# Step 04 — Activations and vanishing gradients

## The story so far
[Step 03](../03_tiny_autograd_engine/WALKTHROUGH.md) trained a tiny network with `tanh`. Swapping `tanh` for a sigmoid (experiment 2 there) stalled the loss near 1.0. That was not a bug in the engine; it was a property of the sigmoid. Here we measure it.

## The pain point
Add layers and a network should get more powerful. In the 1990s and 2000s it often got *worse*: the layers near the input barely learned. With sigmoid, the gradient reaching layer 1 of a 20-layer net is about 1e-13 of the one at the top, so those weights effectively never change. The same arithmetic happens along the time axis in RNNs (step 09).

## The spark
The gradient of a deep net is a product of per-layer factors: `weight x activation slope`. If each factor is below 1 the product shrinks exponentially (*vanishing gradient*); above 1 it explodes. Sigmoid's slope is at most 0.25, so it shrinks by at least a factor of 4 per layer before the weights even count. ReLU-like activations have slope 1 on the active side, so the product holds up (given a matching initialisation, step 05).

## Goal
Print the shape and slopes of six activations; measure the gradient at every layer of a 20-layer net for each, with Xavier and He initialisation; see ReLU units die and leaky ReLU keep a trickle of gradient; implement the gated unit (SwiGLU) that LLM feed-forward blocks use and compare its parameter count with the classic 4x MLP.

## The theory (read this before the interview)
**Why an activation at all.** Without a nonlinearity, `W2 (W1 x) = (W2 W1) x`, one linear layer. The nonlinearity lets depth add expressive power ([step 01](../01_neuron_and_perceptron/WALKTHROUGH.md)).

| Activation | Formula | Range | Max slope | Notes |
|---|---|---|---|---|
| sigmoid | `1 / (1 + e^-z)` | (0, 1) | 0.25 | not zero-centred; saturates; now used only for gates and binary outputs |
| tanh | `(e^z - e^-z) / (e^z + e^-z)` | (-1, 1) | 1 | zero-centred; saturates; used in LSTM/GRU cells |
| ReLU | `max(0, z)` | [0, inf) | 1 | cheap, no saturation for z > 0; can 'die' |
| LeakyReLU | `z if z > 0 else a z` (a = 0.01) | (-inf, inf) | 1 | small slope on the negative side keeps a gradient |
| GELU | `z * Phi(z)` (Phi = standard normal CDF) | about [-0.17, inf) | about 1.13 | smooth ReLU; BERT, GPT-2, ViT |
| SiLU / Swish | `z * sigmoid(z)` | about [-0.28, inf) | about 1.10 | smooth ReLU; used in SwiGLU feed-forward blocks |

**The gradient is a product.** For layers `h_l = f(W_l h_{l-1})`, the gradient reaching layer `k` from the loss is `dL/dh_L * prod_{l=k+1..L} (diag(f'(z_l)) W_l)`. A product of 20 matrices whose typical factor is below 1 shrinks exponentially. With sigmoid, `f' <= 0.25`; saturated units (large `|z|`) have slope near 0, as the table printed (`0.0025` at +-6).

**Measured here (20 layers, width 64, ratio = gradient at layer 1 over layer 20):**
- sigmoid: 4e-13 with Xavier, 1.5e-10 with He. Vanishes whatever the init.
- tanh with Xavier: 0.51. Fine, because tanh has slope 1 near 0 and Xavier is designed for it.
- ReLU / leaky with He: about 4. Fine. With Xavier: 5e-3 (the signal itself halves each layer, a preview of step 05).
- GELU / SiLU with Xavier: about 1e-5; with He: 5 to 6.

**Exploding gradient** is the other failure (factors above 1); the cures are initialisation (step 05), normalisation (step 05), residual connections (step 05) and gradient clipping (step 10).

**Dead ReLU.** If a unit's pre-activation is negative for every input, its output is 0 and its slope is 0, so no gradient reaches its weights and it stays dead. A large learning rate step that pushes a bias very negative does this. Measured: at `lr=5` 14 of 32 units were always off; ReLU's gradient into them was exactly `0.00`, leaky ReLU's was `9e-05`: small but alive. Leaky ReLU, PReLU (learned slope), ELU, GELU and SiLU all keep a nonzero gradient for negative inputs.

**Gated linear units (GLU family).** A GLU multiplies two projections of the input, one passed through a nonlinearity that acts as a gate (Shazeer 2020, 'GLU Variants Improve Transformer'):
```
classic MLP:  down( act(x W_up) )                    two matrices, hidden = 4 * d
GLU family:   down( act(x W_gate) * (x W_up) )       three matrices; act = sigmoid (GLU), GELU (GeGLU), SiLU (SwiGLU)
```
`*` is elementwise: the gate decides, per hidden unit, how much of the value passes. To keep parameters equal the hidden size is cut to about `8/3 d` (`2 * 4d = 8d` matrix entries versus `3 * (8/3 d) = 8d`). Measured: 32,768 versus 32,640 parameters at `d = 64`. SwiGLU is used in the LLaMA family and many other open models; check a model's config (`hidden_act: silu`, `intermediate_size`) rather than assuming.

**What to use in 2026.** Transformers: SwiGLU or GeGLU in the feed-forward block, GELU in older models. CNNs: ReLU, SiLU or GELU variants. Gates inside LSTMs: sigmoid and tanh. Output layers: none (logits) with softmax in the loss, or sigmoid for independent binary labels.

**Vanishing gradient remedies, in order of impact:** (1) non-saturating activations (ReLU family), (2) matching initialisation (step 05), (3) residual connections and normalisation (step 05), (4) gated architectures in RNNs (step 09), (5) gradient clipping for the exploding side (step 10).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| SiLU | Swish, swish-1, `x * sigmoid(x)` | Same (Swish with beta = 1) | Swish allows a learned `beta`; PyTorch's `F.silu` fixes beta = 1 |
| GELU | Gaussian error linear unit, `x * Phi(x)`; 'tanh approximation' | Exact vs approximate | `F.gelu(approximate='tanh')` is a cheaper approximation to the exact erf version |
| LeakyReLU | PReLU, RReLU, 'parametric ReLU' | Related | Leaky: fixed slope; PReLU: slope is learned; RReLU: random slope in training |
| SwiGLU | gated MLP, GLU variant, `silu(xW) * (xV)` | SwiGLU is one GLU variant | GeGLU uses GELU as the gate, ReGLU uses ReLU, plain GLU uses sigmoid |
| vanishing gradient | gradient decay, saturation problem | Same problem | Opposite of exploding gradient |
| dead ReLU | dying ReLU, dead neuron | Same | A unit that never fires and gets zero gradient; not the same as 'sparse activation', which is healthy |
| feed-forward / MLP block in a transformer | FFN, position-wise feed-forward, 'MLP' | Same block | GPT-2: `Linear -> GELU -> Linear`; LLaMA-style: three matrices with SiLU gate |
| saturation | flat region of the activation | Cause of vanishing gradient | Sigmoid and tanh saturate for large `|z|`; ReLU only on one side |

## Run it
```bash
cd 15_neural_networks/04_activations_and_vanishing_gradients
python main.py       # a few seconds on CPU, torch only
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-20)
```python
"""
GOAL:  Compare activation functions (sigmoid, tanh, ReLU, LeakyReLU, GELU, SiLU) by their SLOPES, then
       measure the gradient that reaches each layer of a 20-layer network. See sigmoid's gradient vanish,
       watch a ReLU unit die, and meet the gated unit (GLU/SwiGLU) used in modern LLM feed-forward blocks.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/04_activations_and_vanishing_gradients/main.py
EXPECTED OUTPUT (shortened):
    sigmoid 0.250 max slope; with xavier init the layer-1 / layer-20 gradient ratio is ~4e-13
    relu with he init: ratio ~4; always-off ReLU units get gradient exactly 0, leaky ReLU does not
    classic MLP (4x, GELU): 32768 params | SwiGLU (8/3 x): 32640 params
NEEDS: Python + torch. A few seconds on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

```
- **In simple words:** Import torch and its layer/function modules, and seed the random numbers.
- **Exactly:** `torch.nn` holds layers (`nn.Linear`), `torch.nn.functional` (`F`) holds stateless functions (`F.relu`, `F.gelu`, `F.silu`). `torch.manual_seed(0)` seeds torch's global generator.
- **Why it is here:** Reproducible initial weights, so the printed ratios are the same on every run.

### 1. The activation zoo, each as a plain function of a tensor  (`main.py` lines 21-29)
```python
# [1] The activation zoo, each as a plain function of a tensor
ACTS = {
    "sigmoid": torch.sigmoid,                       # 1 / (1 + e^-z), range (0, 1)
    "tanh": torch.tanh,                             # range (-1, 1), zero-centred
    "relu": F.relu,                                 # max(0, z)
    "leaky_relu": lambda z: F.leaky_relu(z, 0.01),  # z if z > 0 else 0.01 z
    "gelu": F.gelu,                                 # z * Phi(z), Phi = normal CDF (BERT, GPT-2)
    "silu": F.silu,                                 # z * sigmoid(z), a.k.a. swish (LLaMA-style MLPs)
}
```
- **In simple words:** Collect six activation functions in one dictionary so we can loop over them and compare.
- **Exactly:** Each value is a function `Tensor -> Tensor` of the same shape. `leaky_relu` is wrapped in a lambda to fix the negative slope at 0.01. The dictionary preserves insertion order, so tables print in this order.
- **Why it is here:** Looping over a dict of functions keeps every experiment identical except for the activation, the only thing we want to vary.

### 2. Slopes: the derivative decides how much gradient flows through each unit  (`main.py` lines 32-39)
```python
# [2] Slopes: the derivative decides how much gradient flows through each unit
z = torch.linspace(-6, 6, 1201, requires_grad=True)             # (1201,) grid of inputs
print(f"{'activation':<11}{'f(-2)':>8}{'f(0)':>8}{'f(2)':>8}{'max slope':>11}{'slope@-6':>10}{'slope@+6':>10}")
for name, fn in ACTS.items():
    out = fn(z)
    (slope,) = torch.autograd.grad(out.sum(), z)                # d out_i / d z_i for every grid point
    f_at = lambda v: fn(torch.tensor(float(v))).item()
    print(f"{name:<11}{f_at(-2):>8.3f}{f_at(0):>8.3f}{f_at(2):>8.3f}{slope.max().item():>11.3f}{slope[0].item():>10.4f}{slope[-1].item():>10.4f}")
```
- **In simple words:** Draw each activation's slope from -6 to +6 and print a few values. The slope tells how much of the blame passes through.
- **Exactly:** `z` is a `(1201,)` leaf tensor with `requires_grad=True`. `torch.autograd.grad(out.sum(), z)` returns `d(sum out)/dz`, which, since each output depends only on its own input, is the elementwise derivative. Printed: sigmoid max slope `0.250` and `0.0025` at +-6; tanh `1.000` and `0.0000`; ReLU `0.0000` at -6, `1.0000` at +6; leaky `0.0100` at -6; GELU max `1.129`; SiLU max `1.100` and `-0.0123` at -6 (SiLU has slope slightly below 0 far out on the left).
- **Why it is here:** It makes saturation visible: sigmoid and tanh pass almost nothing at the extremes; ReLU passes everything or nothing; GELU and SiLU are smooth versions of ReLU whose slope can exceed 1 near zero.

### 3. A deep MLP with a chosen activation and a chosen initialisation (step 05 explains the two inits)  (`main.py` lines 42-51)
```python
# [3] A deep MLP with a chosen activation and a chosen initialisation (step 05 explains the two inits)
def make_deep(act, init, depth=20, width=64):
    layers = [nn.Linear(width, width) for _ in range(depth)]
    for lin in layers:
        if init == "xavier":
            nn.init.xavier_normal_(lin.weight)                  # variance 2 / (fan_in + fan_out): for tanh / sigmoid
        else:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")   # variance 2 / fan_in: for ReLU-like units
        nn.init.zeros_(lin.bias)
    return layers
```
- **In simple words:** Build a stack of 20 equally wide layers. The initialisation can be Xavier or He (step 05 explains the difference).
- **Exactly:** 20 `nn.Linear(64, 64)` layers. `xavier_normal_`: weights `N(0, 2 / (fan_in + fan_out))` = variance 1/64 here. `kaiming_normal_(nonlinearity='relu')`: variance `2 / fan_in` = 2/64. Biases zero.
- **Why it is here:** Depth is what exposes vanishing gradients; initialisation is needed because a bad scale would hide or exaggerate the effect of the activation.

### 4. Per-layer gradient: how big is dLoss/d(output of layer l) at every depth, after ONE backward pass?  (`main.py` lines 54-74)
```python
# [4] Per-layer gradient: how big is dLoss/d(output of layer l) at every depth, after ONE backward pass?
def activation_grad_norms(act_name, init, depth=20, width=64):
    torch.manual_seed(0)
    act = ACTS[act_name]
    layers = make_deep(act, init, depth, width)
    h = torch.randn(256, width)                                 # (batch, width) random inputs
    outs = []
    for lin in layers:
        h = act(lin(h))                                         # (256, 64) after every layer
        h.retain_grad()                                         # keep .grad on this non-leaf tensor
        outs.append(h)
    h.pow(2).mean().backward()                                  # any scalar loss will do
    return [o.grad.norm().item() for o in outs]                 # index 0 = first layer (far from the loss)


for init in ("xavier", "he"):
    print(f"\nnorm of dLoss/d(layer output), 20 layers, {init} init (layer 1 is farthest from the loss):")
    print(f"{'activation':<11}{'layer 1':>12}{'layer 10':>12}{'layer 20':>12}{'ratio L1/L20':>14}")
    for name in ACTS:
        g = activation_grad_norms(name, init)
        print(f"{name:<11}{g[0]:>12.2e}{g[9]:>12.2e}{g[19]:>12.2e}{g[0] / g[19]:>14.2e}")
```
- **In simple words:** Run one forward and backward pass, and read off how large the blame is at each layer. Compare the far end (layer 1) with the near end (layer 20).
- **Exactly:** Each layer's output `h` has shape `(256, 64)`. `h.retain_grad()` is required because `h` is not a leaf and PyTorch normally discards its gradient. After `backward()` on `h.pow(2).mean()`, `o.grad.norm()` is the Frobenius norm of `dLoss/dh_l`. Ratio L1/L20: Xavier: sigmoid `4.19e-13`, tanh `0.513`, relu `5.47e-03`, gelu `1.90e-05`, silu `9.07e-06`. He: sigmoid `1.47e-10`, tanh `1.98`, relu `3.96`, leaky `4.04`, gelu `5.79`, silu `5.46`.
- **Why it is here:** This is the experiment behind the claim 'sigmoid networks cannot be deep'. Measuring the gradient at layer outputs (not weights) isolates the backward signal from how large the stored forward activations are.

### 5. Dead ReLU: a unit whose input is negative for every sample gets gradient exactly 0 and cannot recover  (`main.py` lines 77-103)
```python
# [5] Dead ReLU: a unit whose input is negative for every sample gets gradient exactly 0 and cannot recover
torch.manual_seed(0)
x = torch.randn(512, 2)
y = (x[:, 0] * x[:, 1] > 0).float().unsqueeze(1)                # XOR-like labels on the plane


def train_and_count_dead(act, lr, steps=200):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 32), nn.Identity(), nn.Linear(32, 1))
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in range(steps):
        h = act(net[0](x))
        loss = F.binary_cross_entropy_with_logits(net[2](h), y)
        opt.zero_grad(); loss.backward(); opt.step()
    h = act(net[0](x))                                          # (512, 32) hidden activations
    loss = F.binary_cross_entropy_with_logits(net[2](h), y)
    opt.zero_grad(); loss.backward()
    off = (net[0](x) <= 0).all(dim=0)                           # (32,) units that are negative for EVERY sample
    grad_into_off = net[0].weight.grad[off].abs().max().item() if off.any() else float("nan")
    return loss.item(), int(off.sum()), grad_into_off


print()
for lr in (0.1, 1.0, 5.0, 10.0):
    for name, act in (("relu", F.relu), ("leaky_relu", ACTS["leaky_relu"])):
        loss, n_off, g = train_and_count_dead(act, lr)
        print(f"lr={lr:<5} {name:<11} loss {loss:.3f}  always-off units {n_off:>2}/32  max |grad| into those units {g:.2e}")
```
- **In simple words:** Train a small network and check whether some hidden units have switched off for good. Compare ReLU, which stops sending any blame to them, with leaky ReLU.
- **Exactly:** Labels are XOR-like: positive when the two coordinates have the same sign, on 512 random points. After 200 SGD steps, `off` marks hidden units whose pre-activation is `<= 0` for all 512 samples. For ReLU, the largest gradient entering those units' weights is exactly `0.00e+00`; for leaky ReLU it is nonzero (`9.00e-05` at `lr=5`). Results: `lr=5`: relu 14/32 always-off, loss 0.093; `lr=10`: relu 11/32, loss 0.851, leaky 9/32, loss 0.608. At `lr=0.1` and `1.0` none died.
- **Why it is here:** Dead units are wasted capacity and the cause is a too-high learning rate or a bad init. It is the practical reason leaky ReLU, GELU and SiLU exist.

### 6. Gated units (GLU family): out = activation(x W) * (x V), two projections multiplied elementwise  (`main.py` lines 106-127)
```python
# [6] Gated units (GLU family): out = activation(x W) * (x V), two projections multiplied elementwise
class SwiGLU(nn.Module):
    """LLaMA-style feed-forward: down( silu(gate(x)) * up(x) ). The gate decides what passes through."""
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)    # x W
        self.up = nn.Linear(d_model, d_hidden, bias=False)      # x V
        self.down = nn.Linear(d_hidden, d_model, bias=False)    # back to d_model

    def forward(self, x):                                       # x: (batch, d_model)
        return self.down(F.silu(self.gate(x)) * self.up(x))


d = 64
plain = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))   # classic 4x MLP
swiglu = SwiGLU(d, int(8 * d / 3))                              # hidden ~ 8/3 d keeps the parameter count similar
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"\nclassic MLP (4x, GELU): {count(plain)} params | SwiGLU (8/3 x): {count(swiglu)} params")
print("SwiGLU output shape:", tuple(swiglu(torch.randn(5, d)).shape))

# TRY IT: add "softplus" (F.softplus) or "mish" (F.mish) to ACTS and re-run blocks 2 and 4 (and add it to the `ACTS` dict first).
# BREAK IT: call activation_grad_norms("sigmoid", "xavier", depth=50): the layer-1 gradient underflows to exactly 0.0.
```
- **In simple words:** A gated unit: two parallel projections, one goes through a smooth gate function and multiplies the other, then project back down. Like a tap whose opening is decided by the input itself.
- **Exactly:** `gate`, `up`: `(d_model, d_hidden)` with no bias; `down`: `(d_hidden, d_model)`. Forward: `down(silu(gate(x)) * up(x))`, with `x: (batch, d_model)` and output the same shape `(5, 64)`. For `d = 64`: classic MLP `64*256*2 = 32768` params; SwiGLU with `int(8*64/3) = 170` hidden: `3*64*170 = 32640` params.
- **Why it is here:** This is the feed-forward block of LLaMA-style models you will see in [module 04 step 05](../../04_llm_fundamentals/05_transformer_block/WALKTHROUGH.md) and in config files (`intermediate_size`). The `8/3` factor is a bookkeeping rule to match parameters.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `lambda z: F.leaky_relu(z, 0.01)` | anonymous function fixing an argument | `ACTS['leaky_relu']` |
| `(slope,) = torch.autograd.grad(...)` | unpack a one-element tuple | `grad` returns a tuple, one entry per input |
| `h.retain_grad()` | keep `.grad` for a non-leaf tensor | otherwise `h.grad` is `None` |
| `(h <= 0).all(dim=0)` | True where every sample's value is <= 0 (reduce over the batch) | shape `(32,)` |
| `net.weight.grad[off]` | boolean mask indexing on rows | gradient rows of the 'off' units |
| `f"{x:>12.2e}"` | right-align width 12, scientific notation with 2 decimals | `4.19e-13` |
| `count = lambda m: sum(p.numel() for p in m.parameters())` | parameter counter | 32768 |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `torch.sigmoid`, `torch.tanh` | torch | elementwise activations | tensor | tensor |
| `F.relu`, `F.leaky_relu(z, negative_slope)`, `F.gelu`, `F.silu` | torch.nn.functional | activations | tensor (+ slope) | tensor |
| `nn.init.xavier_normal_`, `nn.init.kaiming_normal_` | torch.nn.init | in-place weight initialisers | weight, `nonlinearity` / `gain` | the tensor |
| `torch.autograd.grad(outputs, inputs)` | torch | gradients without touching `.grad` | scalar output, input tensors | tuple of tensors |
| `tensor.norm()` | torch | Frobenius / L2 norm | `p=2` default | scalar tensor |
| `F.binary_cross_entropy_with_logits` | torch.nn.functional | sigmoid + BCE in one stable op | logits, float labels | scalar |

## Execution trace (real run)
```
activation    f(-2)    f(0)    f(2)  max slope  slope@-6  slope@+6
sigmoid       0.119   0.500   0.881      0.250    0.0025    0.0025
tanh         -0.964   0.000   0.964      1.000    0.0000    0.0000
relu          0.000   0.000   2.000      1.000    0.0000    1.0000
leaky_relu   -0.020   0.000   2.000      1.000    0.0100    1.0000
gelu         -0.046   0.000   1.954      1.129   -0.0000    1.0000
silu         -0.238   0.000   1.762      1.100   -0.0123    1.0123

norm of dLoss/d(layer output), 20 layers, xavier init (layer 1 is farthest from the loss):
activation      layer 1    layer 10    layer 20  ratio L1/L20
sigmoid        3.31e-15    2.79e-09    7.90e-03      4.19e-13
tanh           1.12e-03    3.85e-03    2.18e-03      5.13e-01
relu           6.53e-08    1.20e-06    1.19e-05      5.47e-03
leaky_relu     6.66e-08    1.21e-06    1.19e-05      5.57e-03
gelu           4.57e-13    6.06e-11    2.40e-08      1.90e-05
silu           1.52e-13    4.63e-11    1.67e-08      9.07e-06

norm of dLoss/d(layer output), 20 layers, he init (layer 1 is farthest from the loss):
activation      layer 1    layer 10    layer 20  ratio L1/L20
sigmoid        1.18e-12    6.46e-08    8.00e-03      1.47e-10
tanh           1.74e-02    1.34e-02    8.83e-03      1.98e+00
relu           4.84e-02    3.93e-02    1.22e-02      3.96e+00
leaky_relu     4.94e-02    3.96e-02    1.22e-02      4.04e+00
gelu           3.57e-02    2.30e-02    6.16e-03      5.79e+00
silu           7.66e-03    5.02e-03    1.40e-03      5.46e+00

lr=0.1   relu        loss 0.326  always-off units  0/32  max |grad| into those units nan
lr=0.1   leaky_relu  loss 0.329  always-off units  0/32  max |grad| into those units nan
lr=1.0   relu        loss 0.070  always-off units  0/32  max |grad| into those units nan
lr=1.0   leaky_relu  loss 0.070  always-off units  0/32  max |grad| into those units nan
lr=5.0   relu        loss 0.093  always-off units 14/32  max |grad| into those units 0.00e+00
lr=5.0   leaky_relu  loss 0.099  always-off units 13/32  max |grad| into those units 9.00e-05
lr=10.0  relu        loss 0.851  always-off units 11/32  max |grad| into those units 0.00e+00
lr=10.0  leaky_relu  loss 0.608  always-off units  9/32  max |grad| into those units 7.96e-04

classic MLP (4x, GELU): 32768 params | SwiGLU (8/3 x): 32640 params
SwiGLU output shape: (5, 64)
```
- First table: sigmoid's slope never exceeds 0.25 and is 0.0025 at +-6; tanh and ReLU reach 1; GELU and SiLU slightly exceed 1 (1.129 and 1.100).
- Xavier table: sigmoid's layer-1 gradient is 3.31e-15 against 7.90e-03 at layer 20 (ratio 4e-13). tanh is flat across depth (0.51). ReLU, GELU and SiLU with Xavier shrink (the unit halves signal variance, which Xavier does not compensate), which is the cue for He init.
- He table: ReLU-like activations now hold their gradient (ratios 4 to 6; the gradient is slightly larger at the bottom). Sigmoid is still 1e-10: no init fixes a 0.25 slope.
- Dead-unit table: nothing dies at small lr; at `lr=5` 14/32 ReLU units are off for every sample and receive gradient exactly 0.00; leaky ReLU's off units still receive 9e-05.
- Last lines: 32,768 versus 32,640 parameters, and the SwiGLU output shape.

## Experiments (change ONE thing)
1. Depth for sigmoid (Xavier): layer-1 gradient 2.15e-05 at 5 layers, 1.13e-08 at 10, 3.31e-15 at 20, exactly 0.00e+00 at 50 and 100 (float32 underflow), while the last layer stays near 8e-03 throughout.
2. Depth for ReLU with He: ratio 3.96 at 20 layers and 0.54 at 100 layers. The gradient survives even 100 layers (1.08e-04 at layer 1), though it is not perfectly constant. tanh with Xavier at 100 layers: ratio 0.42.
3. Learning rate for dead units: at `lr=5`, 14 of 32 ReLU units always off; at `lr=10`, 11 of 32 and a worse loss (0.851), so too-high learning rates both kill units and disturb the live ones.

## Debugging practice
BREAK IT: delete the line `h.retain_grad()` in `activation_grad_norms`:
```
  File ".../main.py", line 66, in activation_grad_norms
    return [o.grad.norm().item() for o in outs]                 # index 0 = first layer (far from the loss)
            ^^^^^^^^^^^
AttributeError: 'NoneType' object has no attribute 'norm'
```
Read it: `o.grad` is `None`, so `None.norm()` fails. PyTorch frees the gradient of non-leaf tensors (anything computed from parameters or inputs) to save memory; only leaves keep `.grad`. The fix is `retain_grad()` before `backward()` (or a hook: `h.register_hook(...)`). Second BREAK IT, a silent one: `activation_grad_norms("sigmoid", "xavier", depth=50)` returns `0.0` at layer 1: float32 underflow, a 'vanished' gradient in the literal sense.

## Rebuild checklist
- [ ] write sigmoid, tanh, ReLU, GELU, SiLU and their slopes from memory
- [ ] explain the product-of-factors argument for vanishing gradients
- [ ] list four remedies
- [ ] explain dead ReLU and two cures
- [ ] write SwiGLU and explain the 8/3 factor

## The new problem (cliffhanger)
He init rescued ReLU and Xavier rescued tanh, and a wrong pairing cost five orders of magnitude (GELU with Xavier: 1.9e-05). Where do those two formulas come from, and how do we keep the signal stable through hundreds of layers? Next: [step 05, initialisation, normalisation and residual connections](../05_initialization_and_normalization/WALKTHROUGH.md).

> **Say it aloud (interview box):** A deep net's gradient is a product of per-layer factors, weight times activation slope. Sigmoid's slope is at most 0.25, so with 20 layers the gradient at the bottom was about 1e-13 of the top's, and no initialisation fixes it. ReLU has slope 1 where active, so with He initialisation the gradient holds even at 100 layers, but a unit with always-negative input is dead. Leaky ReLU, GELU and SiLU keep a gradient for negative inputs. Modern transformers use SwiGLU: a gated MLP where silu of one projection multiplies another, with hidden size 8/3 d to match the parameter count of a 4x MLP.
