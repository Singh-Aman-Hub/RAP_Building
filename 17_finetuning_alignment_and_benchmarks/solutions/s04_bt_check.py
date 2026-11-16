"""Exercise 04 (Explain): Bradley-Terry loss facts you can verify numerically.
RUN: .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s04_bt_check.py"""
import math, torch, torch.nn.functional as F
bt = lambda rc, rr: -F.logsigmoid(rc - rr)
print("equal rewards -> loss", bt(torch.tensor(1.0), torch.tensor(1.0)).item(), " ln2 =", math.log(2))
print("shift-invariant:", torch.allclose(bt(torch.tensor(3.0), torch.tensor(1.0)), bt(torch.tensor(103.0), torch.tensor(101.0))))
print("margin 2 ->", round(bt(torch.tensor(2.0), torch.tensor(0.0)).item(), 4), " margin -2 ->", round(bt(torch.tensor(-2.0), torch.tensor(0.0)).item(), 4), "(wrong order is punished more)")
