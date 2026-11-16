"""
GOAL:  Exercise 9 solution (PyTorch). Train a small MLP with AdamW, linear warm-up + cosine decay (written as a
       function, plugged into LambdaLR) and gradient clipping; log lr, gradient norm and loss. Compare with
       a constant high lr that diverges more easily.

RUN (from this folder):  python ex09_warmup_cosine_clip.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex09_warmup_cosine_clip.py
EXPECTED OUTPUT (shortened):
    step   lr   grad_norm(before clip)  loss
    ...
NEEDS: torch (CPU). ~3 s.
"""
import math
import torch
import torch.nn as nn

STEPS, WARM, PEAK, FLOOR = 300, 30, 3e-2, 0.1


def lr_factor(step):
    """Multiplier on the peak lr: linear ramp 0 -> 1, then cosine from 1 down to FLOOR."""
    if step < WARM:
        return (step + 1) / WARM
    progress = (step - WARM) / (STEPS - WARM)
    return FLOOR + (1 - FLOOR) * 0.5 * (1 + math.cos(math.pi * progress))


def run(use_schedule, clip, peak=PEAK, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(512, 10)
    y = (X[:, :3].sum(dim=1, keepdim=True) > 0).float()
    net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=peak, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor) if use_schedule else None
    log = []
    for step in range(STEPS):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(net(X), y)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0 if clip else float("inf"))  # returns the norm BEFORE clipping
        opt.step()
        if sched:
            sched.step()
        log.append((step, opt.param_groups[0]["lr"], gnorm.item(), loss.item()))
    return log


log = run(True, True)
print(f"{'step':>5}{'lr':>10}{'grad_norm':>11}{'loss':>9}")
for s, lr, g, l in log[::40] + [log[-1]]:
    print(f"{s:>5}{lr:>10.5f}{g:>11.3f}{l:>9.4f}")
assert abs(log[WARM - 1][1] - PEAK) < 1e-9 or abs(log[WARM][1] - PEAK) < 1e-6
assert log[0][1] < log[WARM][1]                  # warm-up ramps up
assert log[-1][1] < 0.2 * PEAK                   # cosine decayed near the floor
print(f"\nfinal loss with schedule+clip: {log[-1][3]:.4f}")
const = run(False, False, peak=PEAK * 10)
print(f"constant lr {PEAK*10:g}, no clip: final loss {const[-1][3]:.4f}, max loss over run {max(l for *_, l in const):.3f}")
