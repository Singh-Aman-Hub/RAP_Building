"""Solution ex06: a 'learning-rate range test' - train briefly at many learning rates and pick the best.
RUN: .venv/bin/python 05_llm_parameters/solutions/ex06_lr_range_test.py"""
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

X, y = make_moons(400, noise=0.25, random_state=1)
X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y)


def final_loss(lr, steps=100):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):                                # full-batch steps keep the test simple
        loss = nn.functional.cross_entropy(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


results = {lr: final_loss(lr) for lr in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)}   # dict comprehension
for lr, loss in results.items():
    print(f"lr={lr:<7} loss after 100 steps = {loss:.3f}")
best = min(results, key=results.get)                      # key= tells min() to compare by loss, returns the lr
print("best lr:", best, "(a common rule: pick a value a bit BELOW the lr where loss is lowest)")
