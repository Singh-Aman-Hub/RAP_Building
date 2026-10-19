"""
GOAL: Learn PyTorch's two core ideas: tensors (numpy-like arrays) and autograd
      (automatic gradients) -- replacing the hand-written derivatives from step 02.

RUN (from this folder):  python main.py
RUN (from repo root):    .venv/bin/python 03_ml_foundations/03_pytorch_tensors_autograd/main.py
EXPECTED OUTPUT: tensor shapes, d(loss)/dw values that match the hand formula, a few manual updates.
NEEDS: torch
"""

import numpy as np
import torch  # third-party library: tensors + autograd + neural-network building blocks

# --- Tensors ---------------------------------------------------------------
a = torch.tensor([1.0, 2.0, 3.0])               # create from a Python list
b = torch.zeros(3)                              # 3 zeros
print("a + 2 =", a + 2, "| a * a =", a * a, "| dtype:", a.dtype, "| shape:", a.shape)
print("from numpy:", torch.from_numpy(np.array([1.0, 2.0])), "| back:", a.numpy())

# Devices: tensors live on cpu / mps (Apple GPU) / cuda (NVIDIA). Operands must be on the SAME device.
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("best available device:", device, "(we use cpu in this module to keep things simple)")

# --- Autograd: PyTorch records operations and can differentiate them -------
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y = torch.tensor([3.0, 5.0, 7.0, 9.0])          # rule: y = 2x + 1
w = torch.tensor(0.0, requires_grad=True)       # requires_grad=True -> "track me, I'm a parameter"
bias = torch.tensor(0.0, requires_grad=True)

y_hat = w * x + bias                            # forward pass (builds a computation graph)
loss = ((y_hat - y) ** 2).mean()                # MSE as a single number
print("loss:", loss.item())                     # .item() -> plain Python float

loss.backward()                                 # backward pass: fills w.grad and bias.grad
print("autograd  dL/dw =", w.grad.item(), " dL/db =", bias.grad.item())

# Check against the formula we used in numpy step 02: 2*mean(error*x), 2*mean(error)
error = (0.0 * x + 0.0) - y
print("by hand   dL/dw =", (2 * (error * x).mean()).item(), " dL/db =", (2 * error.mean()).item())

# --- A manual training loop with autograd ----------------------------------
for step in range(200):
    y_hat = w * x + bias
    loss = ((y_hat - y) ** 2).mean()
    w.grad = None; bias.grad = None             # gradients ACCUMULATE by default; clear them each step!
    loss.backward()
    with torch.no_grad():                       # don't record the update itself in the graph
        w -= 0.05 * w.grad
        bias -= 0.05 * bias.grad
print(f"after 200 steps: w={w.item():.3f} bias={bias.item():.3f} loss={loss.item():.6f}")
# TRY IT: remove the `w.grad = None; bias.grad = None` line. Do you still converge? (gradients pile up -> trouble)
# BREAK IT: delete `with torch.no_grad():` and read the "a leaf Variable that requires grad is being used in an in-place operation" RuntimeError.
