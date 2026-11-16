"""
GOAL:  Compare activation functions (sigmoid, tanh, ReLU, LeakyReLU, GELU, SiLU) by their SLOPES, then
       measure the gradient that reaches each layer of a 20-layer network. See sigmoid's gradient vanish,
       watch a ReLU unit die, and meet the gated unit (GLU/SwiGLU) used in modern LLM feed-forward blocks.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/04_activations_and_vanishing_gradients/main.py
EXPECTED OUTPUT (shortened):
    sigmoid 0.250 max slope; with xavier init the layer-1 / layer-20 gradient ratio is ~4e-13
    relu with he init: ratio ~4; always-off ReLU units get gradient exactly 0, leaky ReLU does not
    classic MLP (4x, GELU): 32768 params | SwiGLU (8/3 x): 32640 params
NEEDS: Python + torch. A few seconds on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# [1] The activation zoo, each as a plain function of a tensor
ACTS = {
    "sigmoid": torch.sigmoid,                       # 1 / (1 + e^-z), range (0, 1)
    "tanh": torch.tanh,                             # range (-1, 1), zero-centred
    "relu": F.relu,                                 # max(0, z)
    "leaky_relu": lambda z: F.leaky_relu(z, 0.01),  # z if z > 0 else 0.01 z
    "gelu": F.gelu,                                 # z * Phi(z), Phi = normal CDF (BERT, GPT-2)
    "silu": F.silu,                                 # z * sigmoid(z), a.k.a. swish (LLaMA-style MLPs)
}


# [2] Slopes: the derivative decides how much gradient flows through each unit
z = torch.linspace(-6, 6, 1201, requires_grad=True)             # (1201,) grid of inputs
print(f"{'activation':<11}{'f(-2)':>8}{'f(0)':>8}{'f(2)':>8}{'max slope':>11}{'slope@-6':>10}{'slope@+6':>10}")
for name, fn in ACTS.items():
    out = fn(z)
    (slope,) = torch.autograd.grad(out.sum(), z)                # d out_i / d z_i for every grid point
    f_at = lambda v: fn(torch.tensor(float(v))).item()
    print(f"{name:<11}{f_at(-2):>8.3f}{f_at(0):>8.3f}{f_at(2):>8.3f}{slope.max().item():>11.3f}{slope[0].item():>10.4f}{slope[-1].item():>10.4f}")


# [3] A deep MLP with a chosen activation and a chosen initialisation (step 05 explains the two inits)
def make_deep(act, init, depth=20, width=64):
    layers = [nn.Linear(width, width) for _ in range(depth)]
    for lin in layers:
        if init == "xavier":
            nn.init.xavier_normal_(lin.weight)                  # variance 2 / (fan_in + fan_out): for tanh / sigmoid
        else:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")   # variance 2 / fan_in: for ReLU-like units
        nn.init.zeros_(lin.bias)
    return layers


# [4] Per-layer gradient: how big is dLoss/d(output of layer l) at every depth, after ONE backward pass?
def activation_grad_norms(act_name, init, depth=20, width=64):
    torch.manual_seed(0)
    act = ACTS[act_name]
    layers = make_deep(act, init, depth, width)
    h = torch.randn(256, width)                                 # (batch, width) random inputs
    outs = []
    for lin in layers:
        h = act(lin(h))                                         # (256, 64) after every layer
        h.retain_grad()                                         # keep .grad on this non-leaf tensor
        outs.append(h)
    h.pow(2).mean().backward()                                  # any scalar loss will do
    return [o.grad.norm().item() for o in outs]                 # index 0 = first layer (far from the loss)


for init in ("xavier", "he"):
    print(f"\nnorm of dLoss/d(layer output), 20 layers, {init} init (layer 1 is farthest from the loss):")
    print(f"{'activation':<11}{'layer 1':>12}{'layer 10':>12}{'layer 20':>12}{'ratio L1/L20':>14}")
    for name in ACTS:
        g = activation_grad_norms(name, init)
        print(f"{name:<11}{g[0]:>12.2e}{g[9]:>12.2e}{g[19]:>12.2e}{g[0] / g[19]:>14.2e}")


# [5] Dead ReLU: a unit whose input is negative for every sample gets gradient exactly 0 and cannot recover
torch.manual_seed(0)
x = torch.randn(512, 2)
y = (x[:, 0] * x[:, 1] > 0).float().unsqueeze(1)                # XOR-like labels on the plane


def train_and_count_dead(act, lr, steps=200):
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(2, 32), nn.Identity(), nn.Linear(32, 1))
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in range(steps):
        h = act(net[0](x))
        loss = F.binary_cross_entropy_with_logits(net[2](h), y)
        opt.zero_grad(); loss.backward(); opt.step()
    h = act(net[0](x))                                          # (512, 32) hidden activations
    loss = F.binary_cross_entropy_with_logits(net[2](h), y)
    opt.zero_grad(); loss.backward()
    off = (net[0](x) <= 0).all(dim=0)                           # (32,) units that are negative for EVERY sample
    grad_into_off = net[0].weight.grad[off].abs().max().item() if off.any() else float("nan")
    return loss.item(), int(off.sum()), grad_into_off


print()
for lr in (0.1, 1.0, 5.0, 10.0):
    for name, act in (("relu", F.relu), ("leaky_relu", ACTS["leaky_relu"])):
        loss, n_off, g = train_and_count_dead(act, lr)
        print(f"lr={lr:<5} {name:<11} loss {loss:.3f}  always-off units {n_off:>2}/32  max |grad| into those units {g:.2e}")


# [6] Gated units (GLU family): out = activation(x W) * (x V), two projections multiplied elementwise
class SwiGLU(nn.Module):
    """LLaMA-style feed-forward: down( silu(gate(x)) * up(x) ). The gate decides what passes through."""
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.gate = nn.Linear(d_model, d_hidden, bias=False)    # x W
        self.up = nn.Linear(d_model, d_hidden, bias=False)      # x V
        self.down = nn.Linear(d_hidden, d_model, bias=False)    # back to d_model

    def forward(self, x):                                       # x: (batch, d_model)
        return self.down(F.silu(self.gate(x)) * self.up(x))


d = 64
plain = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))   # classic 4x MLP
swiglu = SwiGLU(d, int(8 * d / 3))                              # hidden ~ 8/3 d keeps the parameter count similar
count = lambda m: sum(p.numel() for p in m.parameters())
print(f"\nclassic MLP (4x, GELU): {count(plain)} params | SwiGLU (8/3 x): {count(swiglu)} params")
print("SwiGLU output shape:", tuple(swiglu(torch.randn(5, d)).shape))

# TRY IT: add "softplus" (F.softplus) or "mish" (F.mish) to ACTS and re-run blocks 2 and 4 (and add it to the `ACTS` dict first).
# BREAK IT: call activation_grad_norms("sigmoid", "xavier", depth=50): the layer-1 gradient underflows to exactly 0.0.
