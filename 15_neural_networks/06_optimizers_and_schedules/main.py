"""
GOAL:  Implement SGD, momentum, Nesterov, RMSProp, Adam and AdamW from scratch, race them on the Rosenbrock
       valley, and prove each matches torch.optim to float precision. Then implement learning-rate warm-up +
       cosine decay and compare with a torch scheduler.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/06_optimizers_and_schedules/main.py
EXPECTED OUTPUT (shortened):
    optimizer   final loss  dist to (1,1)  max |mine - torch|
    momentum     1.357e-08         0.0003           2.22e-16
    adam         2.171e-05         0.0104           2.22e-16
    max |torch LambdaLR - mine| over 1000 steps: 0.0
    small net, 300 steps: my AdamW loss 0.107021 | torch AdamW loss 0.107021 | plain SGD loss 0.6858
NEEDS: Python + numpy + torch. A few seconds on CPU.
"""

import math

import numpy as np
import torch


# [1] The test problem: Rosenbrock's banana valley. Minimum at (1, 1), a long narrow curved valley
def rosen(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosen_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])   # hand-derived gradient


START = np.array([-1.5, 2.0])


# [2] Each optimizer is a small class: state lives in the object, step(p, g) returns the new parameters
class SGD:
    def __init__(self, lr): self.lr = lr
    def step(self, p, g): return p - self.lr * g


class Momentum:                                       # PyTorch convention: v = mu*v + g ; p -= lr*v
    def __init__(self, lr, mu=0.9, nesterov=False):
        self.lr, self.mu, self.nesterov, self.v = lr, mu, nesterov, 0.0
    def step(self, p, g):
        self.v = self.mu * self.v + g                 # velocity: an exponential sum of past gradients
        d = g + self.mu * self.v if self.nesterov else self.v   # Nesterov: look one step ahead
        return p - self.lr * d


class RMSProp:                                        # divide by a running RMS of the gradient, per parameter
    def __init__(self, lr, alpha=0.99, eps=1e-8):
        self.lr, self.alpha, self.eps, self.sq = lr, alpha, eps, 0.0
    def step(self, p, g):
        self.sq = self.alpha * self.sq + (1 - self.alpha) * g ** 2
        return p - self.lr * g / (np.sqrt(self.sq) + self.eps)


class Adam:                                           # momentum (m) + RMSProp scaling (v) + bias correction
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.0, decoupled=False):
        self.lr, self.b1, self.b2, self.eps, self.wd, self.decoupled = lr, b1, b2, eps, weight_decay, decoupled
        self.m, self.v, self.t = 0.0, 0.0, 0
    def step(self, p, g):
        self.t += 1
        if self.decoupled:
            p = p * (1 - self.lr * self.wd)           # AdamW: shrink the weights directly, outside the adaptive update
        elif self.wd:
            g = g + self.wd * p                       # classic L2: the penalty gradient goes THROUGH the adaptive scaling
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)      # bias correction: m and v start at 0, so they start too small
        v_hat = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# [3] Race: same start, same number of steps, and compare with the torch.optim equivalent
def run_mine(opt, steps=2000):
    p = START.copy()
    for _ in range(steps):
        p = opt.step(p, rosen_grad(p))
    return p


def run_torch(make_opt, steps=2000):
    p = torch.tensor(START, dtype=torch.float64, requires_grad=True)
    opt = make_opt([p])
    for _ in range(steps):
        opt.zero_grad()
        x, y = p
        ((1 - x) ** 2 + 100 * (y - x ** 2) ** 2).backward()
        opt.step()
    return p.detach().numpy()


T = torch.optim
contestants = {
    "sgd":      (SGD(1e-3),                        lambda ps: T.SGD(ps, lr=1e-3)),
    "momentum": (Momentum(1e-3),                   lambda ps: T.SGD(ps, lr=1e-3, momentum=0.9)),
    "nesterov": (Momentum(1e-3, nesterov=True),    lambda ps: T.SGD(ps, lr=1e-3, momentum=0.9, nesterov=True)),
    "rmsprop":  (RMSProp(1e-3),                    lambda ps: T.RMSprop(ps, lr=1e-3)),
    "adam":     (Adam(0.02),                       lambda ps: T.Adam(ps, lr=0.02)),
    "adamw":    (Adam(0.02, weight_decay=0.1, decoupled=True), lambda ps: T.AdamW(ps, lr=0.02, weight_decay=0.1)),
}
print(f"start {START.tolist()}, loss {rosen(START):.1f}; optimum (1, 1)")
print(f"{'optimizer':<10}{'final loss':>12}{'dist to (1,1)':>15}{'max |mine - torch|':>21}")
for name, (mine, make) in contestants.items():
    p_mine, p_torch = run_mine(mine), run_torch(make)
    print(f"{name:<10}{rosen(p_mine):>12.3e}{np.linalg.norm(p_mine - 1):>15.4f}{np.abs(p_mine - p_torch).max():>21.2e}")


# [4] Learning-rate schedule: linear warm-up, then cosine decay to a floor
def lr_at(step, total=1000, warmup=100, peak=1.0, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup                              # ramp up from ~0: Adam's v estimate is noisy at first
    progress = (step - warmup) / (total - warmup)                      # 0 -> 1 over the decay phase
    return floor * peak + (1 - floor) * peak * 0.5 * (1 + math.cos(math.pi * progress))


print("\nlr schedule (peak 1.0, warm-up 100 steps, cosine to 0.1 over 1000 steps):")
print("  " + "  ".join(f"step {s}: {lr_at(s):.3f}" for s in (0, 50, 99, 100, 300, 550, 800, 999)))
dummy = torch.nn.Parameter(torch.zeros(1))
sched_opt = T.SGD([dummy], lr=1.0)
sched = T.lr_scheduler.LambdaLR(sched_opt, lambda s: lr_at(s))      # torch multiplies the base lr by our function
worst = 0.0
for s in range(1000):
    worst = max(worst, abs(sched_opt.param_groups[0]["lr"] - lr_at(s)))
    sched_opt.step(); sched.step()
print("max |torch LambdaLR - mine| over 1000 steps:", worst)


# [5] A small neural net: my Adam/AdamW on torch parameters against the real thing
torch.manual_seed(0)
Xd = torch.randn(256, 2, dtype=torch.float64)
yd = (Xd[:, :1] * Xd[:, 1:] > 0).double()                           # XOR-like labels (256, 1)


def make_net():
    torch.manual_seed(1)
    return torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).double()


def train_net(opt_factory, steps=300):
    net = make_net()
    opt = opt_factory(net)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xd), yd)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item(), torch.cat([p.detach().flatten() for p in net.parameters()])


class MyAdamW:                                                      # the same Adam class, applied to torch tensors
    def __init__(self, params, lr, wd):
        self.params = list(params)
        self.opts = [Adam(lr, weight_decay=wd, decoupled=True) for _ in self.params]
    def zero_grad(self):
        for p in self.params: p.grad = None
    @torch.no_grad()
    def step(self):
        for p, o in zip(self.params, self.opts):
            p.copy_(torch.from_numpy(o.step(p.numpy(), p.grad.numpy())))


l_mine, w_mine = train_net(lambda n: MyAdamW(n.parameters(), 0.01, 0.1))
l_ref, w_ref = train_net(lambda n: T.AdamW(n.parameters(), lr=0.01, weight_decay=0.1))
l_sgd, _ = train_net(lambda n: T.SGD(n.parameters(), lr=0.01))
print(f"\nsmall net, 300 steps: my AdamW loss {l_mine:.6f} | torch AdamW loss {l_ref:.6f} | max weight diff {(w_mine - w_ref).abs().max():.1e} | plain SGD loss {l_sgd:.4f}")


# [6] What else exists in torch today?
print("\ntorch", torch.__version__, "| torch.optim has Muon:", hasattr(T, "Muon"), "| Adafactor:", hasattr(T, "Adafactor"),
      "| Lion:", hasattr(T, "Lion"))

# TRY IT: change Adam's lr to 0.2, then 0.002, and note the loss after 2000 steps.
# BREAK IT: in Adam.step, delete the two bias-correction lines (use self.m and self.v directly) and compare to torch.
