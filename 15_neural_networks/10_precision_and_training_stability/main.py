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

# [1] The formats: bits for the exponent (range) and the mantissa (precision)
print(f"{'dtype':<22}{'bits':>5}{'max':>12}{'smallest normal':>17}{'eps (precision)':>17}")
for dt in (torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
    fi = torch.finfo(dt)
    print(f"{str(dt):<22}{fi.bits:>5}{fi.max:>12.4g}{fi.smallest_normal:>17.4g}{fi.eps:>17.4g}")


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
