"""Exercise 07 (Explain): SLERP keeps the norm of unit vectors, linear interpolation shrinks it.
RUN: .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s07_slerp_check.py"""
import torch
torch.manual_seed(0)
a, b = torch.randn(1000), torch.randn(1000)
a, b = a / a.norm(), b / b.norm()
omega = torch.acos((a * b).sum())
slerp = (torch.sin(0.5 * omega) * a + torch.sin(0.5 * omega) * b) / torch.sin(omega)
lin = 0.5 * a + 0.5 * b
print(f"angle {omega.item():.3f} rad | norm slerp {slerp.norm():.3f} | norm linear {lin.norm():.3f}  (random high-dim vectors are nearly orthogonal: linear shrinks to ~0.707)")
