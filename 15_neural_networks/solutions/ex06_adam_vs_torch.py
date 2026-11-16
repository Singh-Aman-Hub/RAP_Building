"""
GOAL:  Exercise 6 solution. Adam and AdamW from scratch in numpy; show the first step has size ~lr thanks to
       bias correction; match torch.optim.Adam/AdamW; show L2-in-gradient differs from decoupled decay.

RUN (from this folder):  python ex06_adam_vs_torch.py
RUN (from repo root):    .venv/bin/python 15_neural_networks/solutions/ex06_adam_vs_torch.py
EXPECTED OUTPUT (shortened):
    first Adam step with bias correction: ...
    max |mine - torch| Adam ... AdamW ...
NEEDS: numpy, torch (CPU). < 2 s.
"""
import numpy as np
import torch


def adam_run(grad_fn, p0, steps, lr=0.1, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=True, bias_correct=True):
    p, m, v = p0.copy(), np.zeros_like(p0), np.zeros_like(p0)
    first_step = None
    for t in range(1, steps + 1):
        g = grad_fn(p)
        if wd and not decoupled:
            g = g + wd * p                       # L2 regularisation: decay goes THROUGH the adaptive scaling
        if wd and decoupled:
            p = p * (1 - lr * wd)                # AdamW: shrink the weight directly, outside m and v
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh, vh = (m / (1 - b1 ** t), v / (1 - b2 ** t)) if bias_correct else (m, v)
        step = lr * mh / (np.sqrt(vh) + eps)
        p = p - step
        if t == 1:
            first_step = np.abs(step).max()
    return p, first_step


grad_fn = lambda p: 2 * (p - np.array([3.0, -2.0, 0.5])) * np.array([1.0, 10.0, 100.0])  # ill-scaled quadratic
p0 = np.array([0.0, 0.0, 0.0])

_, s_bc = adam_run(grad_fn, p0, 1)
_, s_no = adam_run(grad_fn, p0, 1, bias_correct=False)
print(f"first Adam step with bias correction: {s_bc:.4f} (= lr, whatever the gradient scale)")
print(f"first Adam step WITHOUT correction   : {s_no:.4f} (only (1-b1)/sqrt(1-b2) = {0.1 / np.sqrt(0.001):.2f} x lr: far too big, m and v start at 0 and are biased)")


def torch_run(opt_cls, steps, **kw):
    p = torch.tensor(p0, dtype=torch.float64, requires_grad=True)
    opt = opt_cls([p], lr=0.1, **kw)
    target = torch.tensor([3.0, -2.0, 0.5], dtype=torch.float64)
    scale = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float64)
    for _ in range(steps):
        opt.zero_grad()
        (scale * (p - target) ** 2).sum().backward()
        opt.step()
    return p.detach().numpy()


mine_adam, _ = adam_run(grad_fn, p0, 50, wd=0.1, decoupled=False)        # torch Adam(weight_decay) is L2
mine_adamw, _ = adam_run(grad_fn, p0, 50, wd=0.1, decoupled=True)
t_adam = torch_run(torch.optim.Adam, 50, weight_decay=0.1)
t_adamw = torch_run(torch.optim.AdamW, 50, weight_decay=0.1)
print("max |mine - torch|  Adam(L2):", np.abs(mine_adam - t_adam).max(), " AdamW:", np.abs(mine_adamw - t_adamw).max())
assert np.allclose(mine_adam, t_adam, atol=1e-6) and np.allclose(mine_adamw, t_adamw, atol=1e-6)
print("Adam(L2) result  :", mine_adam.round(3))
print("AdamW result     :", mine_adamw.round(3))
print("Target is [3, -2, 0.5]. Under Adam(L2) the decay term wd*p is added to the gradient and then divided by sqrt(v), so "
      "coordinates with large gradients feel almost no decay; AdamW multiplies every weight by (1 - lr*wd) regardless. "
      "Here that makes AdamW end further from the target on the large-gradient coordinate (2.71 vs 3.05).")
