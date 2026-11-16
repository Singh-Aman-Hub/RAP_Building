# Step 05 — Initialisation, normalisation and residual connections

## The story so far
[Step 04](../04_activations_and_vanishing_gradients/WALKTHROUGH.md) showed that matching the initialisation to the activation changes the gradient by five orders of magnitude (GELU with Xavier 1.9e-05, with He 5.8) and that sigmoid vanishes whatever you do. Now we ask where those init formulas come from and what else keeps a 100-layer network trainable.

## The pain point
Pick weights too small and the signal shrinks to nothing layer by layer; too large and it explodes or saturates. All-zero weights make every unit identical forever. A network that cannot get its first forward pass numerically sane cannot be trained at all, however good the optimizer. Also, activation statistics drift during training (the old 'internal covariate shift' story), and very deep stacks lose gradient.

## The spark
Treat each layer as a random linear map and keep the *variance* of what flows through it constant, forwards and backwards (Xavier, He). Then enforce it during training instead of hoping: normalise activations (BatchNorm, LayerNorm, RMSNorm). Finally give the signal and the gradient a shortcut that skips the layer entirely (residual connection). Together these three ideas are why transformers with hundreds of layers train.

## Goal
Measure activation standard deviation through 50 layers for five inits and two activations; prove zero init makes hidden units copies; implement BatchNorm, LayerNorm and RMSNorm by hand and match torch to 1e-6; show BatchNorm's train/eval difference and its batch-size-1 failure; show residuals rescue gradients in a 50-layer sigmoid net.

## The theory (read this before the interview)
**Variance propagation (the derivation behind Xavier and He).** For a layer `y = W x` with independent zero-mean weights of variance `Var(w)` and `n_in` inputs of variance `Var(x)`:
```
Var(y) = n_in * Var(w) * Var(x)
```
To keep `Var(y) = Var(x)` choose `Var(w) = 1 / n_in`. The backward pass has the same structure with `n_out`. Compromise (**Xavier / Glorot**, 2010, for tanh/sigmoid): `Var(w) = 2 / (n_in + n_out)`. ReLU zeroes half of its inputs, halving the second moment, so compensate with a factor 2 (**He / Kaiming**, 2015): `Var(w) = 2 / n_in`.

| Init | Weight variance | Use with |
|---|---|---|
| zero | 0 | biases only; never weights (symmetry) |
| small random (0.01) | 1e-4 | nothing deep: signal dies |
| unit random | 1 | nothing: `n_in * 1` blow-up |
| Xavier / Glorot | `2 / (n_in + n_out)` | tanh, sigmoid, linear, attention projections |
| He / Kaiming | `2 / n_in` | ReLU, leaky ReLU (and a sensible default for GELU/SiLU) |
| LLM practice | normal, std about 0.02 (GPT-2 style), often scaled by `1 / sqrt(2 * n_layers)` on residual-output projections | transformers with LayerNorm/RMSNorm; check the model's `initializer_range` |

PyTorch's `nn.Linear` default is Kaiming-uniform with `a = sqrt(5)`, which works out to variance `1 / (3 n_in)`: a bit small, a deliberate choice that is fine with normalisation. Our run shows it collapses a 50-layer ReLU stack (std 3e-20 at layer 50) when used bare.

**Symmetry.** With equal initial weights, every hidden unit computes the same value and receives the same gradient, so all units remain identical (measured: maximum difference 0.0 after 200 steps). Random init breaks the tie. Biases can be zero safely.

**Normalisation layers.** Each rescales activations to zero mean and unit variance (or unit RMS), then applies a learned scale `gamma` (and shift `beta`):
```
BatchNorm:  mean/var over the BATCH (per feature), keeps running averages for inference
LayerNorm:  mean/var over the FEATURES of ONE sample (no batch dependence)
RMSNorm:    x / sqrt(mean(x^2) + eps) * gamma   (no mean subtraction, no beta)
```
| | Normalises over | Batch dependence | Train vs eval differ | Parameters | Typical home |
|---|---|---|---|---|---|
| BatchNorm | batch (and space for conv) | yes | **yes** (running stats) | gamma, beta | CNNs |
| LayerNorm | features of each token | no | no | gamma, beta | transformers (GPT-2, BERT) |
| RMSNorm | features of each token | no | no | gamma only | most modern LLMs (LLaMA family and others) |
| GroupNorm | groups of channels | no | no | gamma, beta | small-batch vision |

RMSNorm drops the mean-centring and the bias: slightly cheaper and found to work as well (Zhang and Sennrich, 2019). BatchNorm breaks with a batch of 1 in training (variance of one number), mixes information between samples, and needs care with distributed batches; transformers process variable-length sequences with small per-device batches, so they use LayerNorm/RMSNorm.

**Where the norm goes.** *Post-LN* (original Transformer): `x = Norm(x + Sublayer(x))`. *Pre-LN* (GPT-2 onwards): `x = x + Sublayer(Norm(x))`. Pre-LN keeps an un-normalised identity path from input to output, so gradients flow well and training is stable without delicate warm-up; it is the common default for deep LLMs.

**Residual connections.** `h_{l+1} = h_l + f(h_l)` (He et al., ResNet 2015). The derivative of the output with respect to the input is `I + f'(h)`: the identity term gives the gradient a path with factor exactly 1, so it cannot vanish through that path. Measured: plain sigmoid net, layer-1 gradient exactly 0.0 at 50 layers; same net with residuals, ratio 1.27. Also called skip connections; DenseNet concatenates instead of adding; the transformer's 'residual stream' is the running sum all layers read and write.

**Other practical points.** Initialise biases to 0 (output-layer bias can be set to the log class prior). Do not apply weight decay to norm gains and biases (step 07). BatchNorm before or after the activation is debated; LayerNorm in transformers sits at the block input (pre-LN). Mixed precision (step 10) computes norms in float32.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Xavier init | Glorot init, `xavier_uniform_`, `xavier_normal_` | Same family; uniform and normal variants | Both have variance `2 / (fan_in + fan_out)`; uniform uses a limit `sqrt(6 / (fan_in + fan_out))` |
| He init | Kaiming init, `kaiming_normal_`, `kaiming_uniform_` | Same | Variance `2 / fan_in` (default `mode='fan_in'`); `fan_out` mode preserves backward variance |
| fan_in / fan_out | number of inputs / outputs of a layer | Same | For `nn.Linear(in, out)`: `fan_in = in`, `fan_out = out`; for conv: channels x kernel area |
| BatchNorm | BN, `nn.BatchNorm1d/2d` | Different axis from LayerNorm | Statistics across the batch; has running stats; train != eval |
| LayerNorm | LN, `nn.LayerNorm` | Different axis from BatchNorm | Statistics across features of one sample; same in train and eval |
| RMSNorm | root-mean-square norm | LayerNorm without mean-centring and without bias | `x / rms(x) * gamma`; cheaper; used in many open LLMs |
| gamma / beta | scale and shift, `weight` and `bias` of a norm layer, 'gain' | Same | PyTorch names them `.weight` and `.bias`; Keras `gamma`, `beta` |
| residual connection | skip connection, shortcut, identity path, 'residual stream' (transformer view) | Same idea | ResNet adds `h + f(h)`; DenseNet concatenates; highway nets gate the sum |
| pre-LN / post-LN | norm-first / norm-last | Opposite placements | Pre-LN `x + F(Norm(x))`, post-LN `Norm(x + F(x))`; pre-LN is more stable for deep models |
| internal covariate shift | the original BatchNorm motivation | Disputed | Later work found BN smooths the loss landscape; do not state ICS as settled fact |

## Run it
```bash
cd 15_neural_networks/05_initialization_and_normalization
python main.py       # a few seconds on CPU, torch only
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-26)
```python
"""
GOAL:  (1) See how weight initialisation decides whether signals shrink, explode or survive through 50 layers
       (zero / small / large / Xavier / He). (2) Implement BatchNorm, LayerNorm and RMSNorm by hand and match
       torch. (3) Show that a residual connection keeps gradients alive in a deep stack.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/05_initialization_and_normalization/main.py
EXPECTED OUTPUT (shortened):
    tanh   small init: std 0.156 -> 1e-40 after 50 layers; large: ~0.97 (saturated); xavier 0.63 -> 0.09
    relu   he: 0.82 -> 0.39; xavier: 0.58 -> 1e-8; large -> nan
    zero init: the 8 hidden units differ by at most 0.0
    my_batchnorm / my_layernorm / my_rmsnorm vs torch: max abs diff ~1e-6
    train mode, batch size 1 -> Expected more than 1 value per channel when training ...
    sigmoid plain: layer-1 gradient 0.0, residual: ratio ~1.3
NEEDS: Python + torch. A few seconds on CPU.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


```
- **In simple words:** Import torch and the usual helpers; `math` is imported for convenience.
- **Exactly:** `nn` for layers and `nn.init`, `F` for functions. A global seed; each experiment reseeds so the experiments do not depend on each other.
- **Why it is here:** Reseeding per experiment keeps every number below reproducible and independent of execution order.

### 1. Push random data through 50 layers and watch the standard deviation of the activations  (`main.py` lines 27-53)
```python
# [1] Push random data through 50 layers and watch the standard deviation of the activations
def activation_stds(act, init, depth=50, width=256, n=512):
    torch.manual_seed(0)
    h = torch.randn(n, width)                                    # (512, 256), unit variance input
    stds = []
    for _ in range(depth):
        W = torch.empty(width, width)
        if init == "zero":
            W.zero_()
        elif init == "small":
            W.normal_(0, 0.01)                                   # naive: tiny random numbers
        elif init == "large":
            W.normal_(0, 1.0)                                    # naive: unit-variance weights
        elif init == "xavier":
            nn.init.xavier_normal_(W)                            # std = sqrt(2 / (fan_in + fan_out)) = sqrt(1/256)
        elif init == "he":
            nn.init.kaiming_normal_(W, nonlinearity="relu")      # std = sqrt(2 / fan_in)
        h = act(h @ W)                                           # (512, 256); no bias to keep it clean
        stds.append(h.std().item())
    return stds


for act_name, act in (("tanh", torch.tanh), ("relu", F.relu)):
    print(f"\n{act_name}: std of activations after layer 1 / 10 / 25 / 50")
    for init in ("zero", "small", "large", "xavier", "he"):
        s = activation_stds(act, init)
        print(f"  {init:<7}{s[0]:>10.4f}{s[9]:>10.4g}{s[24]:>10.4g}{s[49]:>10.4g}")
```
- **In simple words:** Feed random numbers through 50 matrix multiplications and print how spread-out the numbers are after layers 1, 10, 25 and 50. If it goes to zero the signal died; if it goes to infinity the signal exploded.
- **Exactly:** Input `(512, 256)` with unit variance. Each layer: a fresh `W (256, 256)` built by one of five rules, `h = act(h @ W)` with no bias. `h.std()` after each layer is recorded. tanh results (layer 1 / 10 / 25 / 50): zero `0 / 0 / 0 / 0`; small (std 0.01) `0.156 / 1.1e-08 / 1.2e-20 / 1.4e-40`; large (std 1) `0.97` everywhere (every unit saturated at +-1); xavier `0.63 / 0.23 / 0.14 / 0.094`; he `0.72 / 0.56 / 0.55 / 0.56`. ReLU: small collapses to exactly 0; large `9.3 / 2.9e10 / 1.5e26 / nan`; xavier `0.58 / 0.026 / 1.2e-4 / 1.2e-8`; he `0.82 / 0.84 / 0.70 / 0.39`.
- **Why it is here:** It turns the variance derivation into numbers. Xavier is right for tanh (decay is slow, caused by tanh's own compression); He is right for ReLU (Xavier loses a factor of 2 in variance per layer, giving 1e-8).

### 2. Zero init breaks symmetry: every hidden unit gets the same gradient and stays a copy of the others  (`main.py` lines 56-70)
```python
# [2] Zero init breaks symmetry: every hidden unit gets the same gradient and stays a copy of the others
torch.manual_seed(0)
X = torch.randn(64, 2)
y = (X[:, :1] * X[:, 1:] > 0).float()                            # (64, 1) XOR-like target
net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
for lin in (net[0], net[2]):
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
opt = torch.optim.SGD(net.parameters(), lr=0.5)
for _ in range(200):
    loss = F.binary_cross_entropy_with_logits(net(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
rows = net[0].weight                                             # (8, 2): one row per hidden unit
print(f"\nzero init: loss after 200 steps {loss.item():.4f}; the 8 hidden units differ by at most "
      f"{(rows - rows[0]).abs().max().item():.1e}, i.e. they are all copies of one unit")
```
- **In simple words:** Start a tiny network with all weights equal to zero and train it. Every hidden unit does exactly the same thing, so we effectively have a network with one hidden unit.
- **Exactly:** A `(2 -> 8 -> 1)` tanh net with all weights and biases set to 0 trains 200 SGD steps. `rows = net[0].weight` has shape `(8, 2)`; `(rows - rows[0]).abs().max()` is `0.0`: all 8 rows are identical. The loss only reached `0.6887`, near `ln 2 = 0.693`, the loss of guessing 50/50.
- **Why it is here:** It proves why random initialisation is mandatory: identical units get identical gradients (by symmetry of the computation), so gradient descent cannot make them different.

### 3. Three normalisations by hand: they differ in WHAT they average over and whether they centre  (`main.py` lines 73-107)
```python
# [3] Three normalisations by hand: they differ in WHAT they average over and whether they centre
def my_batchnorm(x, gamma, beta, eps=1e-5):                      # x: (batch, features)
    mean = x.mean(dim=0, keepdim=True)                           # (1, features): over the BATCH
    var = x.var(dim=0, unbiased=False, keepdim=True)             # biased variance, as torch uses for the forward pass
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def my_layernorm(x, gamma, beta, eps=1e-5):                      # x: (batch, features)
    mean = x.mean(dim=-1, keepdim=True)                          # (batch, 1): over the FEATURES of each sample
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def my_rmsnorm(x, gamma, eps=1e-5):                              # x: (batch, features); no mean, no bias
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)  # root mean square over the features
    return gamma * x / rms


torch.manual_seed(0)
x = torch.randn(32, 16) * 3 + 5                                  # (batch=32, features=16), mean 5, std 3
gamma, beta = torch.rand(16) + 0.5, torch.randn(16)              # arbitrary learned scale and shift

bn = nn.BatchNorm1d(16); bn.weight.data, bn.bias.data = gamma.clone(), beta.clone()
ln = nn.LayerNorm(16); ln.weight.data, ln.bias.data = gamma.clone(), beta.clone()
print("\nmy_batchnorm vs torch BatchNorm1d (train mode):", (my_batchnorm(x, gamma, beta) - bn(x)).abs().max().item())
print("my_layernorm vs torch LayerNorm               :", (my_layernorm(x, gamma, beta) - ln(x)).abs().max().item())
if hasattr(nn, "RMSNorm"):
    rn = nn.RMSNorm(16, eps=1e-5); rn.weight.data = gamma.clone()
    print("my_rmsnorm   vs torch nn.RMSNorm              :", (my_rmsnorm(x, gamma) - rn(x)).abs().max().item())

y_ln = my_layernorm(x, torch.ones(16), torch.zeros(16))
y_rms = my_rmsnorm(x, torch.ones(16))
print(f"\ninput          : per-sample mean {[round(v, 3) for v in x.mean(-1)[:3].tolist()]}")
print(f"LayerNorm out  : per-sample mean {[round(v, 6) for v in y_ln.mean(-1)[:3].tolist()]}, std {[round(v, 4) for v in y_ln.std(-1, unbiased=False)[:3].tolist()]}")
print(f"RMSNorm out    : per-sample mean {[round(v, 3) for v in y_rms.mean(-1)[:3].tolist()]} (not centred), root-mean-square {[round(v, 4) for v in (y_rms ** 2).mean(-1).sqrt()[:3].tolist()]}")
```
- **In simple words:** Write the three normalisers yourself. They differ only in which numbers they average over and whether they subtract the mean.
- **Exactly:** BatchNorm: `x.mean(dim=0)` averages down the batch, giving `(1, features)`; LayerNorm: `x.mean(dim=-1)` averages along features, giving `(batch, 1)`; RMSNorm: root-mean-square over features, no centring, no beta. `unbiased=False` divides by `n` rather than `n-1`, which is what PyTorch uses. `eps=1e-5` avoids dividing by 0. Max absolute difference to torch: BatchNorm `9.5e-07`, LayerNorm `7.2e-07`, RMSNorm `2.4e-07` (float32 rounding). Output statistics: LayerNorm output per-sample mean `0.0`, std `1.0`; RMSNorm output mean `0.86` (not centred) and root-mean-square `1.0`.
- **Why it is here:** Normalising makes the scale of each layer's input independent of what earlier layers did, so the init and learning rate matter less. Writing it by hand is a common interview exercise, and the axis (`dim=0` versus `dim=-1`) is the whole difference between BatchNorm and LayerNorm.

### 4. BatchNorm behaves differently in train and eval mode (running averages); LayerNorm does not  (`main.py` lines 110-123)
```python
# [4] BatchNorm behaves differently in train and eval mode (running averages); LayerNorm does not
bn = nn.BatchNorm1d(16)
for _ in range(50):
    bn(torch.randn(32, 16) * 3 + 5)                              # feed batches so it learns running mean ~5, var ~9
sample = torch.randn(1, 16) * 3 + 5                              # ONE sample
bn.eval()
print("\nBatchNorm running_mean (should be ~5):", [round(v, 2) for v in bn.running_mean[:3].tolist()],
      "| running_var (should be ~9):", [round(v, 1) for v in bn.running_var[:3].tolist()])
print("eval-mode output for a single sample works:", bn(sample).shape)
bn.train()
try:
    bn(sample)                                                   # batch of ONE in train mode: variance undefined
except ValueError as e:
    print("train mode, batch size 1 ->", e)
```
- **In simple words:** BatchNorm keeps a running memory of the statistics it saw in training and uses that memory at inference. It cannot compute statistics from one sample in training mode.
- **Exactly:** After 50 batches `running_mean` is about 5 (`[4.8, 5.02, 4.98]`) and `running_var` about 9 (`[9.7, 8.9, 8.9]`), the true values of the data (mean 5, std 3). `bn.eval()` uses them, so a single sample works (`(1, 16)`). In `bn.train()` mode a batch of 1 raises `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 16])`.
- **Why it is here:** This is the classic source of 'works in training, breaks in serving' bugs: a forgotten `model.eval()` makes outputs depend on the other items in the batch. LayerNorm and RMSNorm have no such mode.

### 5. Residual connections: h_{l+1} = h_l + f(h_l). The identity path gives gradients a highway  (`main.py` lines 126-149)
```python
# [5] Residual connections: h_{l+1} = h_l + f(h_l). The identity path gives gradients a highway
def grad_ratio(act, residual, depth=50, width=64):
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
    for lin in layers:
        nn.init.xavier_normal_(lin.weight); nn.init.zeros_(lin.bias)
    h = torch.randn(128, width)
    outs = []
    for lin in layers:
        h = h + act(lin(h)) if residual else act(lin(h))         # the ONLY difference: "h +"
        h.retain_grad(); outs.append(h)
    h.pow(2).mean().backward()
    g = [o.grad.norm().item() for o in outs]
    return g[0], g[-1], g[0] / g[-1]


print("\ngradient at layer 1 versus layer 50 (ratio = layer 1 / layer 50):")
for name, act in (("sigmoid", torch.sigmoid), ("tanh", torch.tanh)):
    for residual in (False, True):
        g1, gL, r = grad_ratio(act, residual)
        print(f"  {name:<8}{'residual' if residual else 'plain   '} |grad| layer 1 {g1:.2e}, layer 50 {gL:.2e}, ratio {r:.2e}")

# TRY IT: in block 1, change `n` or `width` and check that the xavier/he std barely moves.
# BREAK IT: in block 3, use `unbiased=True` in my_layernorm and compare with torch (small but visible difference).
```
- **In simple words:** Build a 50-layer stack twice. In the second, each layer *adds* its result to its input instead of replacing it. Then compare the blame that arrives at layer 1 and at layer 50.
- **Exactly:** `h = h + act(lin(h))` versus `h = act(lin(h))`. Results (Xavier, width 64): sigmoid plain: layer-1 gradient `0.00e+00`, layer-50 `1.15e-02`, ratio 0; sigmoid residual: `6.99e-01 / 5.49e-01`, ratio `1.27`. tanh plain: ratio `0.54` (already healthy); tanh residual: ratio `28.8` (gradient grows toward the input because each layer's contribution adds up).
- **Why it is here:** It is why modern networks are deep at all: the gradient through `h + f(h)` is `1 + f'(h)`, which always contains the identity. Note the side effect in the tanh row: plain residual stacks can have growing activations, which is why transformers add norms and scale init.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `torch.empty(w, w)` then `.normal_(0, 0.01)` | allocate an uninitialised tensor and fill it in place; trailing `_` means in-place | `W.normal_(0, 1.0)` |
| `x.mean(dim=-1, keepdim=True)` | average over the last axis, keep it as size 1 | `(32,16) -> (32,1)` |
| `x.var(unbiased=False)` | variance dividing by n instead of n-1 | matches `nn.LayerNorm` |
| `bn.weight.data = ...` | overwrite parameter values without autograd tracking | copy the same gamma into torch's layer |
| `hasattr(nn, 'RMSNorm')` | check the library version has the class | added in recent torch |
| `try: ... except ValueError as e:` | catch an error and print its message | BatchNorm batch-size-1 |
| `h = h + act(...) if residual else act(...)` | conditional expression; precedence: `(h + act(...)) if residual else act(...)` | the single line that differs |
| `nn.ModuleList` | a list of modules registered as sub-modules | makes `.parameters()` find them |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `nn.init.xavier_normal_(W)` | torch.nn.init | fill `W` with `N(0, 2/(fan_in+fan_out))` | tensor, `gain` | the tensor |
| `nn.init.kaiming_normal_(W, nonlinearity)` | torch.nn.init | fill with `N(0, 2/fan_in)` (default `mode='fan_in'`) | `a`, `mode`, `nonlinearity` | the tensor |
| `nn.BatchNorm1d(n)` | torch | batch norm over `(batch, n)`; keeps `running_mean`, `running_var` | `eps`, `momentum`, `affine` | module |
| `nn.LayerNorm(shape)` | torch | normalise over the last dim(s) | `normalized_shape`, `eps` | module |
| `nn.RMSNorm(shape)` | torch | root-mean-square norm (recent torch versions) | `normalized_shape`, `eps` | module |
| `F.layer_norm` | torch.nn.functional | functional LayerNorm | input, shape | tensor |

## Execution trace (real run)
```

tanh: std of activations after layer 1 / 10 / 25 / 50
  zero       0.0000         0         0         0
  small      0.1561 1.102e-08 1.209e-20 1.422e-40
  large      0.9745    0.9743    0.9742    0.9744
  xavier     0.6276    0.2321    0.1408   0.09351
  he         0.7208    0.5592     0.554    0.5595

relu: std of activations after layer 1 / 10 / 25 / 50
  zero       0.0000         0         0         0
  small      0.0933  2.88e-10 1.533e-24         0
  large      9.3279  2.88e+10 1.533e+26       nan
  xavier     0.5830   0.02619 0.0001209 1.151e-08
  he         0.8245    0.8381    0.7003    0.3863

zero init: loss after 200 steps 0.6887; the 8 hidden units differ by at most 0.0e+00, i.e. they are all copies of one unit

my_batchnorm vs torch BatchNorm1d (train mode): 9.5367431640625e-07
my_layernorm vs torch LayerNorm               : 7.152557373046875e-07
my_rmsnorm   vs torch nn.RMSNorm              : 2.384185791015625e-07

input          : per-sample mean [4.642, 5.398, 5.193]
LayerNorm out  : per-sample mean [-0.0, 0.0, 0.0], std [1.0, 1.0, 1.0]
RMSNorm out    : per-sample mean [0.863, 0.857, 0.891] (not centred), root-mean-square [1.0, 1.0, 1.0]

BatchNorm running_mean (should be ~5): [4.8, 5.02, 4.98] | running_var (should be ~9): [9.7, 8.9, 8.9]
eval-mode output for a single sample works: torch.Size([1, 16])
train mode, batch size 1 -> Expected more than 1 value per channel when training, got input size torch.Size([1, 16])

gradient at layer 1 versus layer 50 (ratio = layer 1 / layer 50):
  sigmoid plain    |grad| layer 1 0.00e+00, layer 50 1.15e-02, ratio 0.00e+00
  sigmoid residual |grad| layer 1 6.99e-01, layer 50 5.49e-01, ratio 1.27e+00
  tanh    plain    |grad| layer 1 1.05e-03, layer 50 1.95e-03, ratio 5.40e-01
  tanh    residual |grad| layer 1 4.09e+00, layer 50 1.42e-01, ratio 2.88e+01
```
- tanh: **zero** stays 0; **small** decays by 10 orders of magnitude per 10 layers (1e-8, 1e-20, 1e-40); **large** has std about 0.97 at every layer, which is saturation (all outputs at +-1), so its slope is near 0 and it would not learn; **xavier** shrinks gently; **he** is about 0.55.
- ReLU: **large** explodes to 2.9e10 at layer 10 and becomes `nan` at 50; **xavier** loses a factor of about 10 per 10 layers; **he** stays near 0.4 to 0.8.
- Zero init: all rows identical (difference 0.0), loss 0.6887, barely below the chance level 0.693.
- The three `my_*` lines: all within float32 rounding (about 1e-6) of torch.
- LayerNorm output has mean 0 and std 1 per sample; RMSNorm output has root-mean-square 1 but a nonzero mean: no centring.
- BatchNorm's running estimates approach the true mean 5 and variance 9; the `ValueError` shows the batch-size-1 failure.
- Gradient ratios: plain sigmoid 0.0 versus residual 1.27.

## Experiments (change ONE thing)
1. `unbiased=True` in `my_layernorm`: the difference to torch grows from 7e-07 to 9.1e-02, because dividing by `n-1 = 15` instead of `n = 16` rescales the output by `sqrt(16/15)` (about 3.3 percent). PyTorch uses the biased estimator in its normalisation layers.
2. He init at width 64 / 256 / 1024 (ReLU, 50 layers): final std 0.24 / 0.35 / 0.64. The same formula behaves better at larger width because the law of large numbers kicks in; small nets have visible random fluctuations in their activation scale.
3. PyTorch's default `nn.Linear` init (bias-free, ReLU, width 256): std 0.34 / 8.0e-05 / 1.2e-10 / 3.0e-20 at layers 1 / 10 / 25 / 50. Fine for 12-layer nets with normalisation, but not for a 50-layer bare stack.
4. Xavier init, ReLU and a `LayerNorm` after every layer: std 1.0000 at every checked layer (1, 10, 25, 50), the bad init is fully compensated. Normalisation makes training far less sensitive to initialisation.

## Debugging practice
BREAK IT: BatchNorm in train mode with one sample.
```
bn.train(); bn(sample)   # sample.shape == (1, 16)
ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 16])
```
Read it: in train mode BatchNorm needs the variance of each feature over the batch; with one row the variance is undefined (0/0 for the unbiased version). Fixes: use `eval()` for inference, use LayerNorm/GroupNorm, or `drop_last=True` so the last tiny batch of an epoch is skipped. The same mistake in a service shows up as 'predictions change depending on batch size' when someone forgot `model.eval()`.

Second BREAK IT, silent: use `unbiased=True` in `my_layernorm`: no error, but outputs differ from `nn.LayerNorm` by 9.1e-02. A reimplementation that is 'almost' equal needs a comparison with a tolerance you chose on purpose: `torch.allclose(a, b, atol=1e-5)`.

## Rebuild checklist
- [ ] derive `Var(y) = n_in Var(w) Var(x)` and the Xavier and He variances
- [ ] say why zero init fails and why zero bias is fine
- [ ] write BatchNorm and LayerNorm from memory, naming the axis
- [ ] write RMSNorm and say what it drops
- [ ] explain train versus eval for BatchNorm and the batch-size-1 error
- [ ] explain why `h + f(h)` helps the gradient and where pre-LN puts the norm

## The new problem (cliffhanger)
A network now starts healthily and keeps its signals sane through depth. But we have only used plain gradient descent, `p -= lr * grad`, on loss surfaces with narrow valleys. Is there something smarter than one global learning rate, and what do `Adam`, `AdamW`, warm-up and cosine decay actually do? Next: [step 06, optimisers and schedules](../06_optimizers_and_schedules/WALKTHROUGH.md).

> **Say it aloud (interview box):** Initialisation keeps the variance of activations and gradients constant across layers: variance of the weights is 1/n_in for a linear map, Xavier uses 2/(n_in+n_out) for tanh, He uses 2/n_in because ReLU halves the second moment. Zero weights give identical units forever, so use random weights and zero biases. BatchNorm normalises each feature over the batch and needs running statistics at inference; LayerNorm normalises each sample over its features and is what transformers use; RMSNorm drops the mean and bias and is cheaper. Residual connections add an identity path so the gradient through a layer is one plus something, which is why very deep nets train, and pre-LN keeps that path clean.
