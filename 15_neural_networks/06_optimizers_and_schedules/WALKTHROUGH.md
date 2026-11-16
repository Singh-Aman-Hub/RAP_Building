# Step 06 — Optimisers and learning-rate schedules

## The story so far
Every step so far used `p -= lr * grad` (full-batch gradient descent). [Module 03 step 02](../../03_ml_foundations/02_linear_regression_numpy/WALKTHROUGH.md) showed that the learning rate alone decides between slow, good and diverged. [Step 05](../05_initialization_and_normalization/WALKTHROUGH.md) made the start of training healthy. Now we improve the walk itself.

## The pain point
Loss surfaces of real problems are long narrow valleys: steep across, flat along. One global learning rate must be small enough not to bounce across the valley and is then painfully slow along it. Different parameters (embeddings of rare tokens, biases, attention weights) also need different step sizes, and the gradient from one mini-batch is noisy.

## The spark
Use the history. Momentum averages past gradients so that the bouncing cancels and the steady direction accumulates. RMSProp divides each parameter's step by the typical size of its recent gradients, so every parameter moves at a comparable speed. Adam does both and corrects the start-up bias. AdamW then fixes how weight decay is applied. On top, a learning-rate schedule (warm-up, then decay) controls the global step size over time.

## Goal
Write SGD, momentum, Nesterov, RMSProp, Adam and AdamW from scratch, check each against `torch.optim` (difference about 1e-16), see how they behave on Rosenbrock, implement warm-up plus cosine decay (difference to torch 0.0), and train a small net with my AdamW versus torch's (same loss to six digits).

## The theory (read this before the interview)
**Notation.** `g_t` gradient at step `t`, `lr` learning rate, `p` parameter.

| Optimizer | Update | Extra state | Key idea |
|---|---|---|---|
| SGD | `p -= lr * g` | none | follow the slope |
| Momentum | `v = mu v + g; p -= lr v` | `v` | accumulate velocity; damp oscillation (mu = 0.9) |
| Nesterov | `v = mu v + g; p -= lr (g + mu v)` | `v` | gradient 'looks ahead' (a correction that often helps slightly) |
| RMSProp | `s = a s + (1-a) g^2; p -= lr g / (sqrt(s) + eps)` | `s` | per-parameter step normalisation |
| Adam | `m = b1 m + (1-b1) g; v = b2 v + (1-b2) g^2; p -= lr m_hat / (sqrt(v_hat) + eps)` | `m`, `v`, `t` | momentum + RMSProp + bias correction |
| AdamW | Adam with `p *= (1 - lr * wd)` applied separately | `m`, `v`, `t` | decoupled weight decay |

Defaults for Adam: `b1 = 0.9`, `b2 = 0.999`, `eps = 1e-8`. LLM training often uses `b2 = 0.95`, `eps = 1e-8` and `wd = 0.1`; check each paper or config.

**Bias correction.** `m` and `v` start at 0, so early on they underestimate the true averages (after 1 step `m = 0.1 g`, `v = 0.001 g^2`). Dividing by `1 - b^t` fixes that. Without it, the first step is `0.1 g / sqrt(0.001 g^2) = 3.16` times too large: measured, the first Adam step on Rosenbrock is `[0.02, 0.02]`, without the correction `[0.063, 0.063]`.

**Why Adam's step is about `lr` in size.** `m_hat / sqrt(v_hat)` is roughly a signed number of magnitude at most about 1. So with Adam, `lr` is approximately the maximum per-step change of each weight, regardless of gradient scale. SGD's step scales with the gradient, which is why the two need very different learning rates (1e-3 for Adam in transformers, 1e-1 to 1e-2 for SGD in CNNs).

**L2 regularisation vs weight decay (details in step 07).** Adding `wd * p` to the gradient (L2) and then running Adam divides that term by `sqrt(v)`, so weights with large gradients are decayed *less*. AdamW (Loshchilov and Hutter, 2017/2019) shrinks the weight directly, `p *= 1 - lr * wd`, so the decay is independent of the gradient history. For plain SGD the two are the same up to a factor `lr`.

**Learning-rate schedules.**
```
warm-up:  lr rises linearly from ~0 to peak over the first W steps
decay:    cosine from peak down to a floor (10 percent of peak is common) by the last step
```
Why warm-up: at the start Adam's `v` estimate is based on few samples and the weights are random; a big step can destabilise training. Cosine decay (Loshchilov and Hutter, SGDR) gives large steps early to explore, small steps late to settle. Other shapes: linear decay, constant with a final 'cooldown' (warmup-stable-decay), and no decay at all in the schedule-free approach below.

**Batch size and learning rate.** Larger batches have less noisy gradients and usually tolerate a larger learning rate; the linear-scaling rule (`lr` proportional to batch) works up to a point, `sqrt` scaling is another heuristic. Treat both as starting guesses to tune.

**Optimizer memory.** Adam keeps two extra tensors per parameter (`m`, `v`), commonly in float32: with fp32 weights, 4 bytes (weights) + 4 (gradient) + 8 (`m`, `v`) = 16 bytes per parameter, before activations. This is the root of the 'training takes about 16 bytes per parameter' rule in [module 06](../../06_llm_training_and_lora/README.md) and why 8-bit optimizers and Adafactor exist.

**Newer optimizers (state as of Nov 2026; verify before relying).**
- **Muon**: for 2-D hidden weight matrices it orthogonalises the momentum update with a few Newton-Schulz iterations; biases, norms and embeddings stay on AdamW. It has been used in some large-model training reports, and PyTorch ships `torch.optim.Muon` (documented from 2.9; this machine's torch 2.14.1 reports `hasattr(torch.optim, 'Muon') == True`). Its docs state it is for 2-D parameters of hidden layers; the signature in the 2.11 docs is `Muon(params, lr=0.001, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, ns_steps=5, adjust_lr_fn=None)`.
- **Schedule-Free AdamW / SGD** (Defazio et al., 'The Road Less Scheduled', NeurIPS 2024): replaces momentum by interpolation and averaging so no decay schedule or known total step count is needed; official code `facebookresearch/schedule_free`; part of the winning AlgoPerf self-tuning entry.
- **Lion** (Chen et al., 2023, 'Symbolic Discovery of Optimization Algorithms'): sign of an interpolated momentum, one state tensor instead of two. `torch.optim.Lion` does not exist here (printed `False`); it comes from third-party packages. Not re-verified in this session: check the paper.
- **Adafactor** (factored second moments, less memory) exists as `torch.optim.Adafactor` in this torch version.
- **8-bit Adam / paged optimizers** (bitsandbytes) cut optimizer memory; see [module 06](../../06_llm_training_and_lora/README.md).
In an interview: 'AdamW with warm-up and cosine decay is still the default; Muon and schedule-free are credible alternatives I would try on a benchmark of my own, not assume.'

**Measured here on Rosenbrock (2000 steps, start (-1.5, 2)).** Momentum `1.4e-08`, Nesterov `5.8e-09`, Adam (lr 0.02) `2.2e-05`, plain SGD `0.195`, AdamW with `wd=0.1` `0.046` (the decay pulls toward 0, away from the optimum at (1, 1)), RMSProp at its tested learning rate still jittering at `2.44`. It is one toy problem: do not read a general ranking from it.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| learning rate | step size, `lr`, `alpha`, `eta` | Same | Scalar multiplier of the update; schedules change it over time |
| momentum | heavy-ball, `mu`, `beta`, velocity | Same idea | PyTorch `SGD(momentum=0.9)` uses `v = mu v + g`; some texts use `v = mu v - lr g`; equivalent for constant lr |
| Nesterov momentum | NAG, Nesterov accelerated gradient | A variant of momentum | Uses a look-ahead gradient; `nesterov=True` |
| Adam | adaptive moment estimation | - | Momentum on the gradient plus RMS scaling plus bias correction |
| AdamW | Adam with decoupled weight decay | Not the same as Adam + L2 | Decay applied to weights directly, not through the gradient |
| weight decay | L2 regularisation, ridge, `wd`, `lambda` | Equal for SGD, different for Adam | Under adaptive optimizers L2 and weight decay diverge |
| warm-up | LR ramp, linear warm-up | Same | A schedule phase, not an optimizer |
| cosine decay | cosine annealing, `CosineAnnealingLR`, SGDR (with restarts) | Close | SGDR adds warm restarts; plain cosine is one half-wave |
| epsilon (Adam) | `eps`, damping | Same | Added outside the square root in PyTorch; matters in low precision |
| optimizer state | slots, moments, `m` and `v`, `state_dict()['state']` | Same | Must be saved in checkpoints to resume exactly |
| schedule-free | Road Less Scheduled | A method, not a scheduler | Removes the LR decay schedule via averaging |

## Run it
```bash
cd 15_neural_networks/06_optimizers_and_schedules
python main.py       # about 10 seconds on CPU, numpy + torch
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-22)
```python
"""
GOAL:  Implement SGD, momentum, Nesterov, RMSProp, Adam and AdamW from scratch, race them on the Rosenbrock
       valley, and prove each matches torch.optim to float precision. Then implement learning-rate warm-up +
       cosine decay and compare with a torch scheduler.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/06_optimizers_and_schedules/main.py
EXPECTED OUTPUT (shortened):
    optimizer   final loss  dist to (1,1)  max |mine - torch|
    momentum     1.357e-08         0.0003           2.22e-16
    adam         2.171e-05         0.0104           2.22e-16
    max |torch LambdaLR - mine| over 1000 steps: 0.0
    small net, 300 steps: my AdamW loss 0.107021 | torch AdamW loss 0.107021 | plain SGD loss 0.6858
NEEDS: Python + numpy + torch. A few seconds on CPU.
"""

import math

import numpy as np
import torch


```
- **In simple words:** Import numpy for the from-scratch maths and torch to check against the real optimisers.
- **Exactly:** `math` for `cos`/`pi` in the schedule. The numpy code runs in float64, and so does torch (`dtype=torch.float64`), so differences are at the 1e-16 level.
- **Why it is here:** Using float64 on both sides removes rounding as an excuse: any bigger difference would be a real bug.

### 1. The test problem: Rosenbrock's banana valley. Minimum at (1, 1), a long narrow curved valley  (`main.py` lines 23-34)
```python
# [1] The test problem: Rosenbrock's banana valley. Minimum at (1, 1), a long narrow curved valley
def rosen(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosen_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])   # hand-derived gradient


START = np.array([-1.5, 2.0])
```
- **In simple words:** A test landscape with a long, curved, narrow valley. Easy to find the valley, hard to walk along it.
- **Exactly:** `rosen(p) = (1-x)^2 + 100 (y - x^2)^2`, minimum 0 at (1, 1); the start `(-1.5, 2)` has loss `12.5`. `rosen_grad` returns `[df/dx, df/dy]` derived by hand (`-2(1-x) - 400 x (y-x^2)` and `200 (y - x^2)`).
- **Why it is here:** A famous stress test for optimisers: a single learning rate is either too big across the valley or too small along it.

### 2. Each optimizer is a small class: state lives in the object, step(p, g) returns the new parameters  (`main.py` lines 37-74)
```python
# [2] Each optimizer is a small class: state lives in the object, step(p, g) returns the new parameters
class SGD:
    def __init__(self, lr): self.lr = lr
    def step(self, p, g): return p - self.lr * g


class Momentum:                                       # PyTorch convention: v = mu*v + g ; p -= lr*v
    def __init__(self, lr, mu=0.9, nesterov=False):
        self.lr, self.mu, self.nesterov, self.v = lr, mu, nesterov, 0.0
    def step(self, p, g):
        self.v = self.mu * self.v + g                 # velocity: an exponential sum of past gradients
        d = g + self.mu * self.v if self.nesterov else self.v   # Nesterov: look one step ahead
        return p - self.lr * d


class RMSProp:                                        # divide by a running RMS of the gradient, per parameter
    def __init__(self, lr, alpha=0.99, eps=1e-8):
        self.lr, self.alpha, self.eps, self.sq = lr, alpha, eps, 0.0
    def step(self, p, g):
        self.sq = self.alpha * self.sq + (1 - self.alpha) * g ** 2
        return p - self.lr * g / (np.sqrt(self.sq) + self.eps)


class Adam:                                           # momentum (m) + RMSProp scaling (v) + bias correction
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, decoupled=False):
        self.lr, self.b1, self.b2, self.eps, self.wd, self.decoupled = lr, b1, b2, eps, weight_decay, decoupled
        self.m, self.v, self.t = 0.0, 0.0, 0
    def step(self, p, g):
        self.t += 1
        if self.decoupled:
            p = p * (1 - self.lr * self.wd)           # AdamW: shrink the weights directly, outside the adaptive update
        elif self.wd:
            g = g + self.wd * p                       # classic L2: the penalty gradient goes THROUGH the adaptive scaling
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)      # bias correction: m and v start at 0, so they start too small
        v_hat = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```
- **In simple words:** Six small classes, one per optimiser. Each remembers whatever history it needs and returns the new position.
- **Exactly:** `SGD`: no state. `Momentum`: `v` (initially the number 0.0, broadcast to the parameter shape on first use) with the PyTorch convention; Nesterov uses `g + mu*v`. `RMSProp`: running average `sq` of `g^2` (alpha 0.99), `eps` added outside the sqrt. `Adam`: first moment `m`, second moment `v`, step count `t`, bias-corrected `m_hat`, `v_hat`; `decoupled=True` gives AdamW (`p * (1 - lr*wd)` before the Adam step) and `decoupled=False` with `weight_decay` gives Adam with L2 (adds `wd*p` to `g`).
- **Why it is here:** Writing the updates yourself shows that every optimiser is a few lines on top of the gradient, and that state (`m`, `v`) is extra memory per parameter.

### 3. Race: same start, same number of steps, and compare with the torch.optim equivalent  (`main.py` lines 77-109)
```python
# [3] Race: same start, same number of steps, and compare with the torch.optim equivalent
def run_mine(opt, steps=2000):
    p = START.copy()
    for _ in range(steps):
        p = opt.step(p, rosen_grad(p))
    return p


def run_torch(make_opt, steps=2000):
    p = torch.tensor(START, dtype=torch.float64, requires_grad=True)
    opt = make_opt([p])
    for _ in range(steps):
        opt.zero_grad()
        x, y = p
        ((1 - x) ** 2 + 100 * (y - x ** 2) ** 2).backward()
        opt.step()
    return p.detach().numpy()


T = torch.optim
contestants = {
    "sgd":      (SGD(1e-3),                        lambda ps: T.SGD(ps, lr=1e-3)),
    "momentum": (Momentum(1e-3),                   lambda ps: T.SGD(ps, lr=1e-3, momentum=0.9)),
    "nesterov": (Momentum(1e-3, nesterov=True),    lambda ps: T.SGD(ps, lr=1e-3, momentum=0.9, nesterov=True)),
    "rmsprop":  (RMSProp(1e-3),                    lambda ps: T.RMSprop(ps, lr=1e-3)),
    "adam":     (Adam(0.02),                       lambda ps: T.Adam(ps, lr=0.02)),
    "adamw":    (Adam(0.02, weight_decay=0.1, decoupled=True), lambda ps: T.AdamW(ps, lr=0.02, weight_decay=0.1)),
}
print(f"start {START.tolist()}, loss {rosen(START):.1f}; optimum (1, 1)")
print(f"{'optimizer':<10}{'final loss':>12}{'dist to (1,1)':>15}{'max |mine - torch|':>21}")
for name, (mine, make) in contestants.items():
    p_mine, p_torch = run_mine(mine), run_torch(make)
    print(f"{name:<10}{rosen(p_mine):>12.3e}{np.linalg.norm(p_mine - 1):>15.4f}{np.abs(p_mine - p_torch).max():>21.2e}")
```
- **In simple words:** Race all six from the same start for 2000 steps, and also run the official torch versions with the same settings. They should land on the same point.
- **Exactly:** `run_mine` loops `p = opt.step(p, grad)`. `run_torch` makes a float64 tensor with `requires_grad`, calls `backward()` on the same formula, then `opt.step()`. Matching settings: `T.SGD(momentum=0.9)`, `T.RMSprop(lr=1e-3)` (alpha default 0.99), `T.Adam(lr=0.02)`, `T.AdamW(lr=0.02, weight_decay=0.1)`. Max differences: `2.78e-16`, `2.22e-16`, `2.22e-16`, `5.55e-16`, `2.22e-16`, `1.11e-16`. Final losses: sgd `1.951e-01`, momentum `1.357e-08`, nesterov `5.778e-09`, rmsprop `2.443e+00`, adam `2.171e-05`, adamw `4.611e-02`.
- **Why it is here:** It establishes that our formulas are exactly PyTorch's, including the details people get wrong (bias correction, eps outside the sqrt, decoupled decay).

### 4. Learning-rate schedule: linear warm-up, then cosine decay to a floor  (`main.py` lines 112-129)
```python
# [4] Learning-rate schedule: linear warm-up, then cosine decay to a floor
def lr_at(step, total=1000, warmup=100, peak=1.0, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup                              # ramp up from ~0: Adam's v estimate is noisy at first
    progress = (step - warmup) / (total - warmup)                      # 0 -> 1 over the decay phase
    return floor * peak + (1 - floor) * peak * 0.5 * (1 + math.cos(math.pi * progress))


print("\nlr schedule (peak 1.0, warm-up 100 steps, cosine to 0.1 over 1000 steps):")
print("  " + "  ".join(f"step {s}: {lr_at(s):.3f}" for s in (0, 50, 99, 100, 300, 550, 800, 999)))
dummy = torch.nn.Parameter(torch.zeros(1))
sched_opt = T.SGD([dummy], lr=1.0)
sched = T.lr_scheduler.LambdaLR(sched_opt, lambda s: lr_at(s))      # torch multiplies the base lr by our function
worst = 0.0
for s in range(1000):
    worst = max(worst, abs(sched_opt.param_groups[0]["lr"] - lr_at(s)))
    sched_opt.step(); sched.step()
print("max |torch LambdaLR - mine| over 1000 steps:", worst)
```
- **In simple words:** A learning-rate curve over time: climb quickly during the first 10 percent, then glide down along a half-cosine to a floor.
- **Exactly:** `lr_at(step)` returns `peak*(step+1)/warmup` for `step < 100`, then `floor*peak + (1-floor)*peak*0.5*(1+cos(pi*progress))`. Printed: step 0 `0.010`, 50 `0.510`, 99 `1.000`, 100 `1.000`, 300 `0.895`, 550 `0.550`, 800 `0.205`, 999 `0.100`. `LambdaLR(opt, fn)` sets `lr = base_lr * fn(step)`; stepping both for 1000 steps gives maximum difference `0.0`. (`opt.step()` before `sched.step()` is the required order.)
- **Why it is here:** Nearly every LLM training run uses warm-up plus a decay; the shape is a hyperparameter you can read in any training config (`lr_scheduler_type`, `warmup_steps`).

### 5. A small neural net: my Adam/AdamW on torch parameters against the real thing  (`main.py` lines 132-167)
```python
# [5] A small neural net: my Adam/AdamW on torch parameters against the real thing
torch.manual_seed(0)
Xd = torch.randn(256, 2, dtype=torch.float64)
yd = (Xd[:, :1] * Xd[:, 1:] > 0).double()                           # XOR-like labels (256, 1)


def make_net():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).double()


def train_net(opt_factory, steps=300):
    net = make_net()
    opt = opt_factory(net)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xd), yd)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), torch.cat([p.detach().flatten() for p in net.parameters()])


class MyAdamW:                                                      # the same Adam class, applied to torch tensors
    def __init__(self, params, lr, wd):
        self.params = list(params)
        self.opts = [Adam(lr, weight_decay=wd, decoupled=True) for _ in self.params]
    def zero_grad(self):
        for p in self.params: p.grad = None
    @torch.no_grad()
    def step(self):
        for p, o in zip(self.params, self.opts):
            p.copy_(torch.from_numpy(o.step(p.numpy(), p.grad.numpy())))


l_mine, w_mine = train_net(lambda n: MyAdamW(n.parameters(), 0.01, 0.1))
l_ref, w_ref = train_net(lambda n: T.AdamW(n.parameters(), lr=0.01, weight_decay=0.1))
l_sgd, _ = train_net(lambda n: T.SGD(n.parameters(), lr=0.01))
print(f"\nsmall net, 300 steps: my AdamW loss {l_mine:.6f} | torch AdamW loss {l_ref:.6f} | max weight diff {(w_mine - w_ref).abs().max():.1e} | plain SGD loss {l_sgd:.4f}")
```
- **In simple words:** Use the same Adam class inside a real neural network and check it against torch's AdamW. Also train with plain SGD at the same learning rate to see the difference.
- **Exactly:** A `(2 -> 16 -> 1)` tanh net on 256 XOR-like points in float64. `MyAdamW` keeps one `Adam` per parameter tensor and `copy_`s the result into the parameter under `no_grad`. After 300 steps: my AdamW loss `0.107021`, torch AdamW `0.107021`, max weight difference `5.3e-15`; plain SGD at lr 0.01 reaches only `0.6858` (about chance, `ln 2 = 0.693`).
- **Why it is here:** Optimiser choice is not cosmetic: at the same learning rate, Adam's per-parameter scaling makes progress where SGD has barely moved.

### 6. What else exists in torch today?  (`main.py` lines 170-175)
```python
# [6] What else exists in torch today?
print("\ntorch", torch.__version__, "| torch.optim has Muon:", hasattr(T, "Muon"), "| Adafactor:", hasattr(T, "Adafactor"),
      "| Lion:", hasattr(T, "Lion"))

# TRY IT: change Adam's lr to 0.2, then 0.002, and note the loss after 2000 steps.
# BREAK IT: in Adam.step, delete the two bias-correction lines (use self.m and self.v directly) and compare to torch.
```
- **In simple words:** Ask the installed torch which newer optimisers it ships.
- **Exactly:** Prints `torch 2.14.1 | torch.optim has Muon: True | Adafactor: True | Lion: False`.
- **Why it is here:** Library support changes quickly; checking `hasattr` is better than trusting a blog post.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `def step(self, p, g): return p - self.lr * g` | one-line method | `SGD` |
| `self.lr, self.mu, self.v = lr, mu, 0.0` | tuple assignment of several attributes | Momentum |
| `d = a if cond else b` | conditional expression | Nesterov vs plain |
| `self.b1 ** self.t` | power; `b1^t` for bias correction | 0.9^1 = 0.9 |
| `lambda ps: T.SGD(ps, lr=1e-3)` | factory function that builds an optimiser for given params | `contestants` dict |
| `@torch.no_grad()` | decorator: run the function without recording gradients | `MyAdamW.step` |
| `p.copy_(tensor)` | in-place copy into an existing tensor | keeps the parameter object |
| `x, y = p` on a tensor | unpack along dim 0 | `p` has shape `(2,)` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `torch.optim.SGD(params, lr, momentum, nesterov, weight_decay)` | torch.optim | SGD with options | see signature | optimizer |
| `torch.optim.RMSprop`, `Adam`, `AdamW` | torch.optim | adaptive optimisers; AdamW default `weight_decay=0.01` | `lr`, `betas`, `eps`, `weight_decay` | optimizer |
| `opt.zero_grad()`, `loss.backward()`, `opt.step()` | torch | the three calls of the training loop | - | None |
| `torch.optim.lr_scheduler.LambdaLR(opt, fn)` | torch.optim | lr = base_lr * fn(step) | `fn: int -> float` | scheduler |
| `opt.param_groups[0]['lr']` | torch | current learning rate | - | float |
| `np.linalg.norm` | numpy | vector length | - | float |

## Execution trace (real run)
```
start [-1.5, 2.0], loss 12.5; optimum (1, 1)
optimizer   final loss  dist to (1,1)   max |mine - torch|
sgd          1.951e-01         0.8188             2.78e-16
momentum     1.357e-08         0.0003             2.22e-16
nesterov     5.778e-09         0.0002             2.22e-16
rmsprop      2.443e+00         1.6978             5.55e-16
adam         2.171e-05         0.0104             2.22e-16
adamw        4.611e-02         0.4397             1.11e-16

lr schedule (peak 1.0, warm-up 100 steps, cosine to 0.1 over 1000 steps):
  step 0: 0.010  step 50: 0.510  step 99: 1.000  step 100: 1.000  step 300: 0.895  step 550: 0.550  step 800: 0.205  step 999: 0.100
max |torch LambdaLR - mine| over 1000 steps: 0.0

small net, 300 steps: my AdamW loss 0.107021 | torch AdamW loss 0.107021 | max weight diff 5.3e-15 | plain SGD loss 0.6858

torch 2.14.1 | torch.optim has Muon: True | Adafactor: True | Lion: False
```
- `max |mine - torch|` is 1e-16 for all six: the implementations are exact.
- momentum and Nesterov converge to the valley floor within 2000 steps (1e-8). Plain SGD at the same lr is still crawling (0.195). Adam reaches 2e-05.
- RMSProp (lr 1e-3, no momentum, no bias correction) is still noisy at 2.44, and AdamW with `wd=0.1` stops at 0.046 because the decay drags the weights toward 0, away from (1, 1). The ranking is specific to this toy.
- Schedule: starts at 0.010 and hits 1.000 at step 99, then decays smoothly to 0.100.
- Small net: identical losses to six digits. SGD at the same lr 0.01 is stuck at 0.6858.
- The last line records the library state of this machine.

## Experiments (change ONE thing)
1. Adam learning rate on Rosenbrock (2000 steps): `lr=0.2` loss `1.4e-26`; `lr=0.02` loss `2.2e-05`; `lr=0.002` loss `3.9` (distance 1.97, not there yet). Here a bigger lr wins because the problem is deterministic; with noisy mini-batches a large lr instead raises the noise floor.
2. RMSProp: `lr=1e-4` loss 5.74, `lr=3e-4` loss 5.15, `lr=1e-3` loss 2.44. Lower values just move slower within 2000 steps.
3. Adam with L2 (`weight_decay=0.1` through the gradient) ends at loss `1.31e-02` versus AdamW's `4.6e-02` with the same number: the two are different algorithms. With `wd=0` AdamW equals Adam (`2.171e-05`).
4. Remove the bias correction: the first step grows from `[0.02, 0.02]` to `[0.063, 0.063]` (3.16x). On this problem the final loss is `1.6e-05`, so no disaster here, but the early steps of a big network can blow up, which is why warm-up helps too.

## Debugging practice
BREAK IT: in `Adam.step` use `self.m` and `self.v` directly (no bias correction) and compare with `torch.optim.Adam`. No exception: the `max |mine - torch|` column simply stops being 1e-16 (the first step is already 3.16 times too large, shown in experiment 4). A reimplementation must be compared to the reference at the first few steps, because early divergence is the symptom of a missing bias correction; at step 2000 the difference may have shrunk and hide it.

Common real error: calling `sched.step()` before `opt.step()` gives `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()` and skips the first value of the schedule. Order: `opt.step()` then `sched.step()`.

## Rebuild checklist
- [ ] write SGD, momentum, RMSProp and Adam updates from memory
- [ ] explain bias correction and show the 3.16x first-step error
- [ ] state the AdamW difference in one sentence
- [ ] write warm-up + cosine as a function of the step
- [ ] explain Adam's memory cost per parameter
- [ ] name two optimisers newer than AdamW and say what to verify

## The new problem (cliffhanger)
AdamW takes a weight-decay number (we used 0.1), but we have not yet said what it does to *generalisation*. Our nets so far were trained and judged on the same data. How do we stop them memorising: dropout, weight decay, early stopping, augmentation, label smoothing? Next: [step 07, regularisation](../07_regularization/WALKTHROUGH.md).

> **Say it aloud (interview box):** SGD follows the gradient. Momentum adds a velocity so oscillations cancel; RMSProp divides by a running RMS of the gradient so each parameter has a similar step size; Adam combines both with bias correction because its moment estimates start at zero. AdamW applies weight decay directly to the weights instead of through the gradient, which matters because Adam rescales the gradient. In practice I use AdamW with linear warm-up and cosine decay, and I know Adam costs two extra float32 tensors per parameter. Muon and schedule-free methods are newer alternatives I would benchmark, not assume.
