"""
GOAL:  Exercise 10 solution (PyTorch). Show fp16 vs bf16 range and precision, absorption in summation, fp16
       gradient underflow fixed by loss scaling, and how to locate a NaN.

RUN (from this folder):  python ex10_precision_loss_scaling.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex10_precision_loss_scaling.py
EXPECTED OUTPUT (shortened):
    fp16 max 65504.0 ; bf16 max ...
    ...
NEEDS: torch (CPU). < 1 s.
"""
import warnings
import torch

warnings.filterwarnings("ignore")

for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(f"{str(dt):<16} max {fi.max:<12.4g} smallest normal {fi.tiny:<12.3g} eps {fi.eps:.3g}")

# 1. overflow: fp16 maxes out at 65,504, bf16 keeps fp32's range
big = torch.tensor(70000.0)
print("70000 in fp16:", big.to(torch.float16).item(), "| in bf16:", big.to(torch.bfloat16).item())

# 2. absorption: adding a small number to a big one is lost when the mantissa is short
one = torch.tensor(256.0, dtype=torch.bfloat16)
print("bf16: 256 + 1 =", (one + 1).item(), "(bf16 has 8 bits of precision: spacing at 256 is 2, so +1 rounds away)")
print("fp32: 2**24 + 1 =", (torch.tensor(2.0 ** 24) + 1).item())

# 3. underflow of small gradients in fp16, fixed by scaling the loss (and unscaling the gradient)
w = torch.tensor(1.0, dtype=torch.float32)
tiny_grad = 1e-8                       # a realistic small gradient
print("\n1e-8 stored in fp16:", torch.tensor(tiny_grad).to(torch.float16).item(), "(underflow to 0)")
scale = 2.0 ** 14
scaled = (torch.tensor(tiny_grad) * scale).to(torch.float16)
print(f"scaled by 2^14 then fp16: {scaled.item():.4g}; unscaled in fp32: {scaled.float().item() / scale:.4g} (recovered)")

# 4. NaN hunting: find the first non-finite tensor
x = torch.tensor([1.0, 0.0, -1.0])
steps = {"log": torch.log(x), "sqrt": torch.sqrt(x), "exp(100)*exp(100)": torch.exp(torch.tensor(100.0)) * torch.exp(torch.tensor(100.0))}
for name, t in steps.items():
    print(f"{name:<18} finite? {bool(torch.isfinite(t).all())}  {t.tolist() if t.numel() > 1 else t.item()}")
try:
    with torch.autograd.detect_anomaly():
        a = torch.tensor([0.0], requires_grad=True)
        torch.sqrt(a).sum().backward()          # d sqrt(0) = inf; times 0 elsewhere gives nan downstream
        (a.grad * 0).sum()
        print("sqrt'(0) =", a.grad.item(), "(inf, which turns into nan as soon as it is multiplied by 0)")
except RuntimeError as e:
    print("anomaly mode raised:", str(e)[:100])
