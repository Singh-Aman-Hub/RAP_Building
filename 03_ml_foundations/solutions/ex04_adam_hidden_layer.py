"""Exercise 04: Adam + a hidden layer on a NON-linear target.  Run: python ex04_adam_hidden_layer.py"""
import torch
from torch import nn

torch.manual_seed(0)
X = torch.linspace(-3, 3, 400).unsqueeze(1)
y = torch.sin(X) + 0.05 * torch.randn_like(X)           # a curve a straight line cannot fit
perm = torch.randperm(400)
tr, va = perm[:320], perm[320:]


def fit(model, epochs=300):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X[tr]), y[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        return loss_fn(model(X[tr]), y[tr]).item(), loss_fn(model(X[va]), y[va]).item()


linear = nn.Linear(1, 1)
hidden = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))   # activation function = the non-linearity
print("linear  train/val:", [round(v, 4) for v in fit(linear)])
print("hidden  train/val:", [round(v, 4) for v in fit(hidden)])
print("hidden parameters:", sum(p.numel() for p in hidden.parameters()))
