"""Exercise 06 (Modify): group-normalised advantages, zero-variance groups, and the no-std-division variant.
RUN: .venv/bin/python 17_finetuning_alignment_and_benchmarks/solutions/s06_group_advantage.py"""
import torch
groups = {"all wrong": [0., 0, 0, 0], "one right": [1., 0, 0, 0], "half right": [1., 1, 0, 0], "all right": [1., 1, 1, 1]}
for name, r in groups.items():
    r = torch.tensor(r)
    std_adv = (r - r.mean()) / (r.std() + 1e-4)       # original GRPO
    mean_adv = r - r.mean()                           # mean only (no std division)
    print(f"{name:<11} std-normalised {[round(x, 2) for x in std_adv.tolist()]}   mean-only {[round(x, 2) for x in mean_adv.tolist()]}")
print("Note: 'one right' gets +1.5 for the winner with std division but only +0.75 without: std division up-weights hard/easy prompts.")
