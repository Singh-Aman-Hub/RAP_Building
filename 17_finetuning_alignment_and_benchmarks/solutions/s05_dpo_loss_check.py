"""Exercise 05 (Rebuild/Explain): DPO loss from numbers, no model needed.
RUN: .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s05_dpo_loss_check.py"""
import math, torch, torch.nn.functional as F
def dpo(pi_c, pi_r, ref_c, ref_r, beta=0.1):
    return -F.logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))
t = torch.tensor
print("policy == reference:", dpo(t(-10.), t(-12.), t(-10.), t(-12.)).item(), "ln2 =", round(math.log(2), 4))
print("chosen up by 5 nats:", round(dpo(t(-5.), t(-12.), t(-10.), t(-12.)).item(), 4))
print("rejected down by 5 :", round(dpo(t(-10.), t(-17.), t(-10.), t(-12.)).item(), 4), "(same loss: only the DIFFERENCE of log-ratios matters)")
print("both DOWN, rejected more:", round(dpo(t(-11.), t(-20.), t(-10.), t(-12.)).item(), 4), "<- loss falls although chosen log-prob dropped (likelihood displacement)")
for beta in (0.01, 0.1, 1.0):
    print(f"beta={beta}: loss at margin 5 nats = {dpo(t(-5.), t(-12.), t(-10.), t(-12.), beta).item():.4f}")
