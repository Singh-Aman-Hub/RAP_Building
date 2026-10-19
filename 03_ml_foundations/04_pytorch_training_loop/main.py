"""
GOAL: The standard PyTorch training loop: Dataset/DataLoader -> model (nn.Module) -> loss -> optimizer
      -> forward / backward / step.  This SAME skeleton trains an LLM in module 06.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/04_pytorch_training_loop/main.py
EXPECTED OUTPUT: epoch-by-epoch train/validation loss falling, learned weights near [2, -3] and bias 1.
NEEDS: torch
"""

import torch
from torch import nn                                     # nn = neural-network building blocks
from torch.utils.data import DataLoader, TensorDataset, random_split

torch.manual_seed(0)                                     # reproducibility

# --- 1. Data: y = 2*x1 - 3*x2 + 1 + noise ---------------------------------
X = torch.randn(500, 2)                                  # 500 samples, 2 features, standard normal
y = (X @ torch.tensor([2.0, -3.0]) + 1.0 + 0.1 * torch.randn(500)).unsqueeze(1)  # unsqueeze -> shape (500, 1)
dataset = TensorDataset(X, y)                            # pairs each X row with its y
train_set, val_set = random_split(dataset, [400, 100])   # 80/20 split
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)   # yields batches of 32, reshuffled each epoch
val_loader = DataLoader(val_set, batch_size=100)

# --- 2. Model --------------------------------------------------------------
class LinearModel(nn.Module):                            # every model subclasses nn.Module
    def __init__(self):
        super().__init__()                               # required: sets up PyTorch's parameter tracking
        self.linear = nn.Linear(2, 1)                    # 2 inputs -> 1 output: weight (1,2) + bias (1,)

    def forward(self, x):                                # defines the forward pass; called via model(x)
        return self.linear(x)

model = LinearModel()
loss_fn = nn.MSELoss()                                   # the loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # updates the parameters. Try torch.optim.Adam too.

# --- 3. Training loop ------------------------------------------------------
for epoch in range(1, 11):
    model.train()                                        # training mode (matters for dropout/batchnorm)
    for xb, yb in train_loader:                          # one iteration = one batch
        pred = model(xb)                                 # forward
        loss = loss_fn(pred, yb)                         # compare with truth
        optimizer.zero_grad()                            # clear old gradients
        loss.backward()                                  # compute new gradients
        optimizer.step()                                 # update parameters

    model.eval()                                         # evaluation mode
    with torch.no_grad():                                # no gradients needed -> faster, less memory
        train_loss = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in train_loader) / len(train_set)
        val_loss = sum(loss_fn(model(xb), yb).item() * len(xb) for xb, yb in val_loader) / len(val_set)
    print(f"epoch {epoch:2d}  train loss={train_loss:.4f}  val loss={val_loss:.4f}")

print("learned weight:", model.linear.weight.data.flatten().tolist(), "bias:", model.linear.bias.item())
print("number of parameters:", sum(p.numel() for p in model.parameters()))   # LLMs have billions of these
# TRY IT: swap SGD for torch.optim.Adam(model.parameters(), lr=0.05); compare epoch-1 loss.
# TRY IT: batch_size=400 vs 8. Count how many optimizer.step() calls happen per epoch.
# BREAK IT: delete optimizer.zero_grad() -- training gets unstable because gradients accumulate across batches.
