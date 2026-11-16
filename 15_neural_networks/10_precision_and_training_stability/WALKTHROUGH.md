# Step 10 — Numerical precision and training stability

## The story so far
Steps 04, 05 and 09 each ended in numbers like `0.0` and `nan`: sigmoid gradients that underflow, ReLU activations that overflow to `nan`, RNN gradients of 2.6e+05. They were not abstract; they are limits of number formats and of optimisation. [Module 06](../../06_llm_training_and_lora/README.md) states that LLMs train in bfloat16 and are served in 8-bit or 4-bit: this step shows why and what can go wrong.

## The pain point
A 7B-parameter model in float32 needs 28 GB just for weights. Halving the bits halves memory and speeds up matrix multiplies on modern hardware, but fewer bits mean a smaller range (overflow, underflow) or coarser steps (rounding). The result is a loss curve that suddenly becomes `nan` after three days of training, with no obvious cause.

## The spark
Know the three numbers of each format (max, smallest normal, epsilon), keep the dangerous parts in float32 (master weights, accumulations, softmax, norms), use a format with the range of float32 (bfloat16) or rescale the loss (float16), clip gradients, and keep a debugging checklist.

## Goal
Print `torch.finfo` for five dtypes; trigger overflow, underflow and absorption in float16 and bfloat16; implement loss scaling and a dynamic scaler; implement gradient clipping and match `clip_grad_norm_`; produce and locate a NaN with `detect_anomaly`; run bfloat16 autocast on CPU and compare to float32.

## The theory (read this before the interview)
**Floating-point anatomy.** A float is `sign * mantissa * 2^exponent`. More **exponent bits = larger range**, more **mantissa bits = finer precision** (relative step `eps`).

| Format | Bits (sign / exp / mantissa) | Max | Smallest normal | eps | Typical use |
|---|---|---|---|---|---|
| float32 (fp32) | 1 / 8 / 23 | 3.4e+38 | 1.2e-38 | 1.2e-07 | master weights, optimizer state, accumulations |
| float16 (fp16) | 1 / 5 / 10 | 65,504 | 6.1e-05 | 9.8e-04 | older mixed precision (needs loss scaling), inference |
| bfloat16 (bf16) | 1 / 8 / 7 | 3.4e+38 | 1.2e-38 | 7.8e-03 | **default for LLM training** on current GPUs/TPUs: fp32 range, low precision |
| fp8 e4m3 (`float8_e4m3fn`) | 1 / 4 / 3 | 448 | 1.6e-02 | 0.125 | forward weights/activations in fp8 training and inference (with scaling) |
| fp8 e5m2 | 1 / 5 / 2 | 57,344 | 6.1e-05 | 0.25 | gradients in fp8 training (more range) |

(The values are from the `torch.finfo` table printed by `main.py`.) Hardware support for fp8 and fp4 formats depends on the accelerator generation: check the vendor documentation rather than assuming. Smaller integer or 4-bit formats for *inference* are covered in [module 06](../../06_llm_training_and_lora/README.md).

**Failure modes (all demonstrated).**
- *Overflow*: `float16(70000) = inf`; `inf - inf` or `inf / inf` gives `nan`. Naive softmax of `[12, 1, 0.5]` in float16: `exp(12) = 162755` is above 65,504, so the result is `[nan, 0, 0]`. Cure: subtract the max (the stable softmax) and compute softmax in float32.
- *Underflow*: `float16(1e-8) = 0.0`: tiny gradients vanish. bfloat16 keeps `1.0e-08`.
- *Absorption*: adding a number smaller than half an `eps` to a larger one does nothing: float16 `1 + 0.0001 = 1.0`, bfloat16 `1 + 0.001 = 1.0`. Summing `0.0001` ten thousand times gives `1.0001` in float32 and `0.2500` in float16 (the running sum stops growing once the increment is below half a spacing). That is why optimizers keep **fp32 master weights** and why reductions/accumulations run in fp32.
- *Catastrophic cancellation*, `log(0)`, `sqrt(0)` gradients, division by a tiny variance: guard with `eps`, `clamp`, log-sum-exp, or fused stable ops (`F.cross_entropy` on logits, not `log(softmax)`).

**Loss scaling (needed for fp16, usually not for bf16).** Gradients can be far below fp16's smallest values. Multiply the loss by `S` (e.g. 2^16) before backward so gradients are `S` times larger, then divide by `S` before the optimizer step. Measured: a gradient of `1e-8` stored in float16 is `0.0`; scaled by 2^16 it is stored as `0.000655` and unscales to `9.997e-09`. *Dynamic* scaling (what `torch.amp.GradScaler` does) starts high, halves `S` and skips the step when any gradient is `inf`/`nan`, and doubles `S` after a streak of good steps: the printed history `False->32768, False->16384, True ... True->65536, False->32768`.

**Mixed precision.** Keep fp32 *master weights*; run matmuls and convolutions in a lower precision (autocast chooses per-operation); keep reductions, norms, softmax and the loss in fp32 where the framework decides to. On CPU, `torch.autocast('cpu', dtype=torch.bfloat16)` demonstrates the mechanics: weights stay float32, Linear outputs are bfloat16; measured max difference to fp32 on a tiny net was `0.0072` (relative 0.33 percent). On CUDA the autocast op lists differ (several ops are forced to float32, check the PyTorch autocast documentation for the current list). Float16 autocast on CPU also works here (difference 0.00086, finer than bf16 because it has more mantissa bits) but is not the practice for training because of the narrow range.

**Memory arithmetic.** Parameters: fp32 4 bytes, bf16/fp16 2, fp8 1 (our 4,938-parameter toy: 19,752 / 9,876 / 4,938 bytes). Training with Adam in mixed precision typically keeps bf16 weights/activations plus fp32 master weights and two fp32 moments, about 16 bytes per parameter in total (check your framework: ZeRO/FSDP shard these).

**Gradient clipping.** `scale = min(1, max_norm / (global_norm + 1e-6))`; multiply every gradient by it. It preserves direction and bounds the step. Measured: global norm 15.50 clipped to 1.0000; our manual version matches `clip_grad_norm_` exactly (difference 0.0). The function returns the *pre-clip* norm: log it, because a rising gradient norm is the earliest warning of instability.

**Training-stability toolkit for transformers.** Pre-LN or RMSNorm (step 05), warm-up (step 06), smaller `b2` (0.95) with Adam, gradient clipping at 1.0, bf16 (not fp16), z-loss or logit capping on the output softmax, QK-norm in attention for large models, scaled residual init (`1/sqrt(2L)`), data checks (no NaN/inf in batches), skipping or rewinding on loss spikes. Several of these appear in recent open-model reports; verify details per model card.

**NaN debugging checklist.**
1. Does it happen at step 0 or after hours? Step 0 means data, init or a bad op; later means instability or a bad batch.
2. `torch.isfinite(x).all()` on inputs, loss, activations, gradients: find the *first* non-finite tensor.
3. `torch.autograd.set_detect_anomaly(True)` (or `with detect_anomaly():`) names the backward function that produced NaN (here `SqrtBackward0`) and prints the forward traceback line; it is slow, use it only to debug.
4. Suspect: `log(0)`, `sqrt(0)` (its gradient is infinite at 0), `0/0`, `exp` of large numbers, softmax without max-subtraction, division by a variance, `x.pow(0.5)`, masked attention rows that are all `-inf`.
5. Lower the learning rate, add warm-up, clip gradients, switch fp16 -> bf16, run a small batch in fp32 to compare.
6. Reproduce on a tiny batch with a fixed seed, then bisect the model by layer.

**What differs for inference.** Serving uses fp16/bf16 weights or quantised int8/int4/fp8 weights with higher-precision accumulation; quality loss is measured on your own evaluation set (module 06).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| bfloat16 | bf16, brain float, 'bfloat' | Different from float16 | Same 8 exponent bits as fp32 (huge range), only 7 mantissa bits; float16 has 5 exponent bits and 10 mantissa bits |
| float16 | fp16, half precision, `torch.half` | - | Needs loss scaling; narrow range (max 65,504) |
| fp8 | float8, e4m3 / e5m2, FP8 | Two formats | e4m3: more precision; e5m2: more range; both need per-tensor scaling |
| mixed precision | AMP, autocast, `torch.amp` | Same idea | Mixed = some ops in low precision, others and master weights in fp32; not a model-quality statement |
| master weights | fp32 copy of the parameters | - | The optimiser updates the fp32 copy; low-precision copies are used for compute |
| loss scaling | gradient scaling, `GradScaler` | Same | Static: fixed `S`; dynamic: adjusted automatically; rarely needed with bf16 |
| gradient clipping | clip by norm, clip by value | Two methods | `clip_grad_norm_` rescales the whole vector; `clip_grad_value_` clamps each element |
| NaN vs inf | not-a-number vs infinity | Different | `inf - inf`, `0 * inf`, `inf / inf` and `0 / 0` give NaN; NaN then spreads through everything it touches |
| overflow / underflow | range errors | Opposite ends | Overflow: too large -> inf. Underflow: too small -> 0 (or subnormal) |
| quantisation | int8/int4 weights, GPTQ/AWQ-style compression | Not the same as mixed precision | Quantisation stores weights in fewer bits for inference; mixed precision is a training/inference compute setting |
| eps (machine epsilon) | relative precision, `finfo.eps` | Not the `eps` in LayerNorm | Machine eps: smallest relative step; norm eps: a constant added for safety |

## Run it
```bash
cd 15_neural_networks/10_precision_and_training_stability
python main.py       # a few seconds on CPU, torch only (fp8 is inspected, not computed)
```

## Code, step by step
Every block below is copied from `main.py` (line numbers are exact). Read the block, then the three bullets.

### 0. Docstring and imports  (`main.py` lines 1-25)
```python
"""
GOAL:  Learn the number formats of deep learning (fp32, fp16, bf16, fp8), see overflow, underflow and rounding
       happen, implement loss scaling and gradient clipping by hand, find a NaN with autograd's anomaly mode, and
       run mixed precision with autocast (bfloat16 on CPU).

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/10_precision_and_training_stability/main.py
EXPECTED OUTPUT (shortened):
    dtype table: float16 max 65504, bfloat16 max 3.39e+38, float32 eps 1.19e-07 ...
    float16(70000) = inf, float16(1e-8) = 0.0, bfloat16(1 + 0.001) = 1.0
    loss scaling rescues a gradient that float16 flushes to zero
    clip_grad_norm_ matches my manual clipping
    anomaly mode names the backward op that produced NaN
    autocast on CPU: Linear/LayerNorm/softmax outputs bfloat16 here; weights stay float32
NEEDS: Python + torch. A few seconds on CPU. (fp8 is only inspected, not computed.)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

```
- **In simple words:** Import torch; the module docstring lists what is run.
- **Exactly:** `math` is imported for completeness; `F` gives the stable softmax; `nn` gives layers and `clip_grad_norm_`.
- **Why it is here:** Everything here runs on a CPU in seconds, so you can poke at the formats without a GPU.

### 1. The formats: bits for the exponent (range) and the mantissa (precision)  (`main.py` lines 26-30)
```python
# [1] The formats: bits for the exponent (range) and the mantissa (precision)
print(f"{'dtype':<22}{'bits':>5}{'max':>12}{'smallest normal':>17}{'eps (precision)':>17}")
for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
    fi = torch.finfo(dt)
    print(f"{str(dt):<22}{fi.bits:>5}{fi.max:>12.4g}{fi.smallest_normal:>17.4g}{fi.eps:>17.4g}")
```
- **In simple words:** Print the specifications of each number format: how big, how small and how fine.
- **Exactly:** `torch.finfo(dtype)` returns `bits`, `max`, `smallest_normal`, `eps`. Printed: float32 max `3.403e+38`, eps `1.192e-07`; float16 max `6.55e+04`, smallest normal `6.104e-05`, eps `0.0009766`; bfloat16 max `3.39e+38`, eps `0.007812`; `float8_e4m3fn` max `448`, eps `0.125`; `float8_e5m2` max `5.734e+04`, eps `0.25`.
- **Why it is here:** These three numbers (max, smallest normal, eps) predict every failure below. bfloat16 trades 3 decimal digits of precision for the full float32 range.

### 2. What goes wrong: overflow to inf, underflow to 0, and rounding that swallows small increments  (`main.py` lines 33-48)
```python
# [2] What goes wrong: overflow to inf, underflow to 0, and rounding that swallows small increments
big, tiny = torch.tensor(70000.0), torch.tensor(1e-8)
print("\nfloat16(70000)      =", big.half().item(), "  (max is 65504 -> overflow)")
print("float16(1e-8)       =", tiny.half().item(), "  (below the smallest subnormal ~6e-8 -> underflow)")
print("bfloat16(70000)     =", big.bfloat16().item(), "  (wide range, so it survives, but rounded)")
print("bfloat16(1e-8)      =", tiny.bfloat16().item())
one = torch.tensor(1.0)
for dt in (torch.float32, torch.float16, torch.bfloat16):
    print(f"{str(dt):<16} 1 + 0.001 = {(one.to(dt) + torch.tensor(0.001).to(dt)).item():.6f}   1 + 0.0001 = {(one.to(dt) + torch.tensor(0.0001).to(dt)).item():.6f}")
x = torch.tensor(0.0)
for _ in range(10000):                                                    # add 0.0001 ten thousand times: true answer 1.0
    x = x + torch.tensor(1e-4)
xh = torch.tensor(0.0, dtype=torch.float16)
for _ in range(10000):
    xh = xh + torch.tensor(1e-4, dtype=torch.float16)
print(f"sum of 10,000 x 0.0001: float32 {x.item():.4f} | float16 {xh.item():.4f} (the increments stopped registering)")
```
- **In simple words:** Break numbers on purpose: too big, too small, and adding a small number to a big one.
- **Exactly:** `.half()` is float16, `.bfloat16()` is bfloat16. 70000 becomes `inf` in float16 but `70144.0` in bfloat16; `1e-8` becomes `0.0` in float16 but `1.0011717677116394e-08` in bfloat16. `1 + 0.001` is `1.000977` in float16 and `1.0` in bfloat16; `1 + 0.0001` is `1.0` in both. Ten thousand additions of `0.0001`: float32 `1.0001`, float16 `0.2500`.
- **Why it is here:** It turns the finfo table into observed behaviour, including the surprising one: a running sum in float16 stalls. That is why accumulation is done in float32.

### 3. Loss scaling: small gradients underflow in float16; multiply the loss by S, then divide the grads by S  (`main.py` lines 51-78)
```python
# [3] Loss scaling: small gradients underflow in float16; multiply the loss by S, then divide the grads by S
w = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
true_grad = 1e-8                                                          # a legitimate tiny gradient, e.g. deep in a network
g_half = (torch.tensor(true_grad) * 1.0).half()                           # what a float16 backward pass would store
print(f"\ntiny gradient {true_grad:g} stored in float16: {g_half.item()}   (lost)")
S = 2.0 ** 16
g_scaled = (torch.tensor(true_grad) * S).half()                           # scale BEFORE it is rounded to float16
print(f"scaled by 2^16 -> {g_scaled.item():.6g} in float16; unscaled in float32: {g_scaled.float().item() / S:.3e}   (recovered)")


class DynamicLossScaler:
    """What torch.amp.GradScaler does: start high, halve on inf/nan (skip that step), double after a streak of good steps."""
    def __init__(self, init=2.0 ** 16, growth_interval=3):
        self.scale, self.good, self.interval = init, 0, growth_interval
    def update(self, grads_finite):
        if not grads_finite:
            self.scale /= 2; self.good = 0                                # overflow: back off and skip the optimizer step
        else:
            self.good += 1
            if self.good == self.interval:
                self.scale *= 2; self.good = 0


scaler = DynamicLossScaler()
print("dynamic loss scale history (finite? -> scale):", end=" ")
for finite in (False, False, True, True, True, True, True, True, False):
    scaler.update(finite); print(f"{finite}->{scaler.scale:g}", end="  ")
print()
```
- **In simple words:** Rescue a gradient that float16 would round to zero: make it larger first, compute, then shrink it back. Add a controller that adapts the scale.
- **Exactly:** `g_half = tiny.half()` gives `0.0`. With `S = 2**16`, the scaled value `6.5536e-04` is representable in float16 and `float()/S` returns `9.997e-09`. `DynamicLossScaler.update(finite)` halves on a non-finite gradient, otherwise counts good steps and doubles after `growth_interval = 3`. History printed: `32768, 16384, 16384, 16384, 32768, 32768, 32768, 65536, 32768`.
- **Why it is here:** This is the logic inside `torch.amp.GradScaler`; knowing it explains the log lines 'skipping step, loss scale reduced' you see in fp16 runs.

### 4. Gradient clipping: rescale the whole gradient if its global norm exceeds a threshold  (`main.py` lines 81-90)
```python
# [4] Gradient clipping: rescale the whole gradient if its global norm exceeds a threshold
net = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1))
x_in, y_in = torch.randn(16, 8) * 50, torch.randn(16, 1)                  # huge inputs -> a huge gradient
F.mse_loss(net(x_in), y_in).backward()
norm_before = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters())).item()
manual = [p.grad.clone() * min(1.0, 1.0 / (norm_before + 1e-6)) for p in net.parameters()]    # scale = max_norm / (norm + eps), capped at 1
returned = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)       # clips in place, returns the norm BEFORE clipping
norm_after = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters())).item()
diff = max((p.grad - m).abs().max().item() for p, m in zip(net.parameters(), manual))
print(f"\ngradient norm before clipping {norm_before:.2f} (clip_grad_norm_ returned {returned.item():.2f}), after {norm_after:.4f}; max diff to my manual version {diff:.1e}")
```
- **In simple words:** Put a speed limit on how big one update can be: if the whole gradient is too long, shrink it, keeping its direction.
- **Exactly:** With inputs scaled by 50, the gradient norm is `15.50`. The manual scale `min(1, 1 / (norm + 1e-6))` times each gradient matches `clip_grad_norm_(params, max_norm=1.0)` (max difference `0.0`); the function returns the norm *before* clipping (`15.50`) and the gradient norm afterwards is `1.0000`.
- **Why it is here:** One bad batch can produce a gradient 100x normal; unclipped, it throws the weights far away and the run may never recover. The returned norm is the single most useful training-health metric.

### 5. Finding a NaN: naive softmax / log in low precision, then autograd's anomaly detector  (`main.py` lines 93-109)
```python
# [5] Finding a NaN: naive softmax / log in low precision, then autograd's anomaly detector
logits = torch.tensor([12.0, 1.0, 0.5])
naive = lambda z: torch.exp(z) / torch.exp(z).sum()                       # no max-subtraction
print("\nnaive softmax in float32:", naive(logits).tolist())
print("naive softmax in float16:", naive(logits.half()).tolist(), "  (exp(12) = 162755 > 65504 -> inf / inf = nan)")
print("stable softmax in float16:", [round(v, 6) for v in F.softmax(logits.half(), dim=0).tolist()])

p = torch.tensor([0.0, 1.0], requires_grad=True)
try:
    with torch.autograd.detect_anomaly():
        (torch.sqrt(p).sum() * 0.0).backward()                            # sqrt'(0) = inf, times 0 -> nan: a classic hidden NaN
except RuntimeError as e:
    print("anomaly mode:", str(e).splitlines()[0])
print("p.grad without anomaly mode:", end=" ")
q = torch.tensor([0.0, 1.0], requires_grad=True)
(torch.sqrt(q).sum() * 0.0).backward()
print(q.grad.tolist(), "-> finite?", bool(torch.isfinite(q.grad).all()))
```
- **In simple words:** See a NaN being born, then ask PyTorch to point at its birthplace.
- **Exactly:** Naive softmax of `[12, 1, 0.5]`: float32 `[0.99997, 1.67e-05, 1.01e-05]`; float16 `[nan, 0.0, 0.0]` because `exp(12) = 162755` overflows to `inf` and `inf/inf = nan`; `F.softmax` in float16 is stable: `[1.0, 1.7e-05, 1e-05]`. Then `sqrt([0, 1])` times `0.0`: the backward of `sqrt` at 0 is `inf`, and `inf * 0 = nan`. Inside `detect_anomaly()` the error is `Function 'SqrtBackward0' returned nan values in its 0th output.` (PyTorch also prints a warning with the forward line that created it to stderr). Without anomaly mode the gradient is silently `[nan, 0.0]`.
- **Why it is here:** Two typical NaN sources: overflow in `exp` and a function whose derivative is infinite at the boundary. Anomaly mode converts a silent `nan` into a located error.

### 6. Mixed precision: compute in bfloat16 where it is safe, keep float32 master weights and sensitive ops  (`main.py` lines 112-128)
```python
# [6] Mixed precision: compute in bfloat16 where it is safe, keep float32 master weights and sensitive ops
model = nn.Sequential(nn.Linear(64, 64), nn.LayerNorm(64), nn.Linear(64, 10))
x_mp = torch.randn(32, 64)
out32 = model(x_mp)
with torch.autocast("cpu", dtype=torch.bfloat16):
    h = model[0](x_mp)
    n = model[1](h)
    out16 = model[2](n)
    probs = F.softmax(out16, dim=-1)
print(f"\nautocast bf16: Linear output {h.dtype} | LayerNorm output {n.dtype} | final Linear {out16.dtype} | softmax {probs.dtype}")
print("weights stay", model[0].weight.dtype, "| max |fp32 output - autocast output| =", f"{(out32 - out16.float()).abs().max().item():.4f}",
      "| relative", f"{((out32 - out16.float()).abs().max() / out32.abs().max()).item():.4f}")
n_params = sum(p.numel() for p in model.parameters())
print(f"memory for {n_params:,} params: fp32 {n_params * 4:,} bytes | bf16 {n_params * 2:,} bytes | fp8 {n_params:,} bytes")

# TRY IT: change `dtype=torch.bfloat16` to `torch.float16` in autocast on CPU and compare the error to fp32.
# BREAK IT: in block 5 multiply `logits` by 10 and compare the naive float32 softmax with the stable float16 one; which one survives?
```
- **In simple words:** Run a small model in 'half-price' arithmetic and check how far the answer moves.
- **Exactly:** `torch.autocast('cpu', dtype=torch.bfloat16)` makes matrix-multiply-like ops (here `Linear`) output bfloat16; the weights remain float32 (`torch.float32`). On this CPU build the printed dtypes are bfloat16 for Linear, LayerNorm and softmax. Max absolute difference to the fp32 output `0.0072`, relative `0.0033` (0.33 percent). The toy model has 4,938 parameters: 19,752 bytes fp32, 9,876 bf16, 4,938 fp8.
- **Why it is here:** A concrete feel for the cost of bf16: about three significant digits. The CUDA autocast policy keeps more operations in fp32 than the CPU one; do not generalise this line to GPU behaviour.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `torch.finfo(dtype)` | float-format information object | `.max`, `.eps`, `.smallest_normal` |
| `tensor.half()` / `.bfloat16()` / `.float()` | cast to float16 / bfloat16 / float32 | `big.half()` |
| `x.to(dtype)` | general cast | `one.to(torch.float16)` |
| `2.0 ** 16` | float power; 65,536 | loss scale |
| `f"{v:.6g}"` | 6 significant digits, general format | `0.000655174` |
| `print(..., end=' ')` | no newline; used for inline history | - |
| `with torch.autograd.detect_anomaly():` | context that checks every backward op for NaN | slow, debug only |
| `with torch.autocast('cpu', dtype=...):` | context in which supported ops run in the lower precision | - |
| `try / except RuntimeError as e: str(e).splitlines()[0]` | catch and print only the first line of a long message | anomaly message |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `torch.finfo` | torch | limits of a floating dtype | dtype | object |
| `nn.utils.clip_grad_norm_(params, max_norm)` | torch | clip the global gradient norm in place | `max_norm`, `norm_type=2` | norm before clipping |
| `torch.autograd.detect_anomaly` | torch | raise on NaN in backward and print the forward trace | - | context manager |
| `torch.isfinite(x)` | torch | elementwise `not nan and not inf` | - | bool tensor |
| `torch.autocast(device_type, dtype)` | torch | mixed-precision context | `'cpu'`/`'cuda'`, `bfloat16`/`float16` | context manager |
| `torch.amp.GradScaler` | torch | dynamic loss scaler (for CUDA float16) | `init_scale`, `growth_interval` | scaler (not run here) |

## Execution trace (real run)
```
dtype                  bits         max  smallest normal  eps (precision)
torch.float32            32   3.403e+38        1.175e-38        1.192e-07
torch.float16            16    6.55e+04        6.104e-05        0.0009766
torch.bfloat16           16    3.39e+38        1.175e-38         0.007812
torch.float8_e4m3fn       8         448          0.01562            0.125
torch.float8_e5m2         8   5.734e+04        6.104e-05             0.25

float16(70000)      = inf   (max is 65504 -> overflow)
float16(1e-8)       = 0.0   (below the smallest subnormal ~6e-8 -> underflow)
bfloat16(70000)     = 70144.0   (wide range, so it survives, but rounded)
bfloat16(1e-8)      = 1.0011717677116394e-08
torch.float32    1 + 0.001 = 1.001000   1 + 0.0001 = 1.000100
torch.float16    1 + 0.001 = 1.000977   1 + 0.0001 = 1.000000
torch.bfloat16   1 + 0.001 = 1.000000   1 + 0.0001 = 1.000000
sum of 10,000 x 0.0001: float32 1.0001 | float16 0.2500 (the increments stopped registering)

tiny gradient 1e-08 stored in float16: 0.0   (lost)
scaled by 2^16 -> 0.000655174 in float16; unscaled in float32: 9.997e-09   (recovered)
dynamic loss scale history (finite? -> scale): False->32768  False->16384  True->16384  True->16384  True->32768  True->32768  True->32768  True->65536  False->32768  

gradient norm before clipping 15.50 (clip_grad_norm_ returned 15.50), after 1.0000; max diff to my manual version 0.0e+00

naive softmax in float32: [0.9999731183052063, 1.670125129749067e-05, 1.012982102110982e-05]
naive softmax in float16: [nan, 0.0, 0.0]   (exp(12) = 162755 > 65504 -> inf / inf = nan)
stable softmax in float16: [1.0, 1.7e-05, 1e-05]
anomaly mode: Function 'SqrtBackward0' returned nan values in its 0th output.
p.grad without anomaly mode: [nan, 0.0] -> finite? False

autocast bf16: Linear output torch.bfloat16 | LayerNorm output torch.bfloat16 | final Linear torch.bfloat16 | softmax torch.bfloat16
weights stay torch.float32 | max |fp32 output - autocast output| = 0.0072 | relative 0.0033
memory for 4,938 params: fp32 19,752 bytes | bf16 9,876 bytes | fp8 4,938 bytes
```
- The dtype table: bfloat16 and float32 share the same maximum and smallest normal number; float16 has a max of only 65,504.
- Overflow, underflow and absorption lines; the float16 running sum stalls at 0.25.
- Loss scaling: `0.0` without it, `9.997e-09` recovered with it; the history of scales follows the halve-on-overflow, double-after-3-good rule.
- Clipping: 15.50 -> 1.0000, manual and library versions agree exactly.
- Softmax in float16: `nan` for the naive version, finite for the stable one. Sqrt gradient: `nan` located by `SqrtBackward0` in anomaly mode. (Two `UserWarning` messages on stderr come from anomaly mode: they are expected.)
- Autocast: weights float32, outputs bfloat16, error 0.0072 (0.33 percent).

## Experiments (change ONE thing)
1. Autocast with `dtype=torch.float16` on CPU (this torch build supports it): output dtype float16, max difference to fp32 `0.00086` versus `0.0090` for bfloat16 in the same test (the standalone check used a different random input from the `main.py` run, whose bfloat16 difference was 0.0072). float16 is more precise, bfloat16 is safer for range.
2. Multiply the logits by 10 (`[120, 10, 5]`): naive float32 softmax gives `[nan, 0, 0]` (`exp(120)` overflows float32 too: it exceeds 3.4e+38), stable float16 softmax gives `[1.0, 0.0, 0.0]`. A stable algorithm in low precision beats a naive one in high precision.
3. Loss scale: with `S = 2**10` the gradient `1e-8` becomes about `1.0e-05`, which float16 can store only as a subnormal (smallest normal is `6.1e-05`) and with reduced precision; so a good scale pushes values into the normal range (idea shown by block 3, number not run here).

## Debugging practice
BREAK IT: in block 5 use `logits * 10` with the naive float32 softmax: `torch.exp(120.0)` is about 1.3e+52, beyond float32's 3.4e+38, so the result is `inf` and the division gives `nan`:
```
naive softmax in float32: [nan, 0.0, 0.0]
```
No exception is raised. NaNs propagate silently, which is why you must check for them: `assert torch.isfinite(loss), 'loss is not finite'` inside the training loop stops the run at the first bad step instead of 10,000 steps later, when every weight is `nan`.

Real traceback you will meet with anomaly mode on (shown here as the summary line our script prints): `Function 'SqrtBackward0' returned nan values in its 0th output.` It names the backward node (`Sqrt`), so look in your forward code for the `sqrt`/`norm`/`pow(0.5)` that can receive 0, and add `+ eps` inside it.

## Rebuild checklist
- [ ] give max, smallest normal and eps for fp32, fp16 and bf16 (roughly) and say why LLMs use bf16
- [ ] demonstrate overflow, underflow and absorption in one line each
- [ ] write loss scaling and the dynamic rule from memory
- [ ] write gradient clipping by global norm
- [ ] explain why softmax subtracts the max
- [ ] recite the NaN checklist and use `detect_anomaly`

## The new problem (cliffhanger)
That completes the neural-network toolbox: neurons, backprop, autograd, activations, init and normalisation, optimisers, regularisation, convolutions, recurrence and numerics. Every ingredient reappears in a transformer: residual streams, RMSNorm, SwiGLU, AdamW with warm-up and cosine decay, bf16 with gradient clipping, and attention replacing the recurrence. Next: [module 04, LLM fundamentals](../../04_llm_fundamentals/README.md), starting with [attention from scratch](../../04_llm_fundamentals/04_attention_from_scratch/WALKTHROUGH.md).

> **Say it aloud (interview box):** Floats trade range for precision: fp16 has 5 exponent bits, max 65,504, and needs loss scaling; bf16 has the fp32 range with 7 mantissa bits, so LLMs train in bf16 with fp32 master weights and optimiser state. Failures are overflow to inf, underflow to zero, and absorption of small increments, so reductions and softmax stay in fp32 and softmax subtracts the max. Gradient clipping by global norm bounds each update, and I log the pre-clip norm as a health signal. To debug a NaN I find the first non-finite tensor, use anomaly mode to locate the backward op, and suspect log(0), sqrt(0), exp overflow and division by tiny variances.
