"""
GOAL:  (1) See how weight initialisation decides whether signals shrink, explode or survive through 50 layers
       (zero / small / large / Xavier / He). (2) Implement BatchNorm, LayerNorm and RMSNorm by hand and match
       torch. (3) Show that a residual connection keeps gradients alive in a deep stack.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 15_neural_networks/05_initialization_and_normalization/main.py
EXPECTED OUTPUT (shortened):
    tanh   small init: std 0.156 -> 1e-40 after 50 layers; large: ~0.97 (saturated); xavier 0.63 -> 0.09
    relu   he: 0.82 -> 0.39; xavier: 0.58 -> 1e-8; large -> nan
    zero init: the 8 hidden units differ by at most 0.0
    my_batchnorm / my_layernorm / my_rmsnorm vs torch: max abs diff ~1e-6
    train mode, batch size 1 -> Expected more than 1 value per channel when training ...
    sigmoid plain: layer-1 gradient 0.0, residual: ratio ~1.3
NEEDS: Python + torch. A few seconds on CPU.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# [1] Push random data through 50 layers and watch the standard deviation of the activations
def activation_stds(act, init, depth=50, width=256, n=512):
    torch.manual_seed(0)
    h = torch.randn(n, width)                                    # (512, 256), unit variance input
    stds = []
    for _ in range(depth):
        W = torch.empty(width, width)
        if init == "zero":
            W.zero_()
        elif init == "small":
            W.normal_(0, 0.01)                                   # naive: tiny random numbers
        elif init == "large":
            W.normal_(0, 1.0)                                    # naive: unit-variance weights
        elif init == "xavier":
            nn.init.xavier_normal_(W)                            # std = sqrt(2 / (fan_in + fan_out)) = sqrt(1/256)
        elif init == "he":
            nn.init.kaiming_normal_(W, nonlinearity="relu")      # std = sqrt(2 / fan_in)
        h = act(h @ W)                                           # (512, 256); no bias to keep it clean
        stds.append(h.std().item())
    return stds


for act_name, act in (("tanh", torch.tanh), ("relu", F.relu)):
    print(f"\n{act_name}: std of activations after layer 1 / 10 / 25 / 50")
    for init in ("zero", "small", "large", "xavier", "he"):
        s = activation_stds(act, init)
        print(f"  {init:<7}{s[0]:>10.4f}{s[9]:>10.4g}{s[24]:>10.4g}{s[49]:>10.4g}")


# [2] Zero init breaks symmetry: every hidden unit gets the same gradient and stays a copy of the others
torch.manual_seed(0)
X = torch.randn(64, 2)
y = (X[:, :1] * X[:, 1:] > 0).float()                            # (64, 1) XOR-like target
net = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
for lin in (net[0], net[2]):
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
opt = torch.optim.SGD(net.parameters(), lr=0.5)
for _ in range(200):
    loss = F.binary_cross_entropy_with_logits(net(X), y)
    opt.zero_grad(); loss.backward(); opt.step()
rows = net[0].weight                                             # (8, 2): one row per hidden unit
print(f"\nzero init: loss after 200 steps {loss.item():.4f}; the 8 hidden units differ by at most "
      f"{(rows - rows[0]).abs().max().item():.1e}, i.e. they are all copies of one unit")


# [3] Three normalisations by hand: they differ in WHAT they average over and whether they centre
def my_batchnorm(x, gamma, beta, eps=1e-5):                      # x: (batch, features)
    mean = x.mean(dim=0, keepdim=True)                           # (1, features): over the BATCH
    var = x.var(dim=0, unbiased=False, keepdim=True)             # biased variance, as torch uses for the forward pass
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def my_layernorm(x, gamma, beta, eps=1e-5):                      # x: (batch, features)
    mean = x.mean(dim=-1, keepdim=True)                          # (batch, 1): over the FEATURES of each sample
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def my_rmsnorm(x, gamma, eps=1e-5):                              # x: (batch, features); no mean, no bias
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)  # root mean square over the features
    return gamma * x / rms


torch.manual_seed(0)
x = torch.randn(32, 16) * 3 + 5                                  # (batch=32, features=16), mean 5, std 3
gamma, beta = torch.rand(16) + 0.5, torch.randn(16)              # arbitrary learned scale and shift

bn = nn.BatchNorm1d(16); bn.weight.data, bn.bias.data = gamma.clone(), beta.clone()
ln = nn.LayerNorm(16); ln.weight.data, ln.bias.data = gamma.clone(), beta.clone()
print("\nmy_batchnorm vs torch BatchNorm1d (train mode):", (my_batchnorm(x, gamma, beta) - bn(x)).abs().max().item())
print("my_layernorm vs torch LayerNorm               :", (my_layernorm(x, gamma, beta) - ln(x)).abs().max().item())
if hasattr(nn, "RMSNorm"):
    rn = nn.RMSNorm(16, eps=1e-5); rn.weight.data = gamma.clone()
    print("my_rmsnorm   vs torch nn.RMSNorm              :", (my_rmsnorm(x, gamma) - rn(x)).abs().max().item())

y_ln = my_layernorm(x, torch.ones(16), torch.zeros(16))
y_rms = my_rmsnorm(x, torch.ones(16))
print(f"\ninput          : per-sample mean {[round(v, 3) for v in x.mean(-1)[:3].tolist()]}")
print(f"LayerNorm out  : per-sample mean {[round(v, 6) for v in y_ln.mean(-1)[:3].tolist()]}, std {[round(v, 4) for v in y_ln.std(-1, unbiased=False)[:3].tolist()]}")
print(f"RMSNorm out    : per-sample mean {[round(v, 3) for v in y_rms.mean(-1)[:3].tolist()]} (not centred), root-mean-square {[round(v, 4) for v in (y_rms ** 2).mean(-1).sqrt()[:3].tolist()]}")


# [4] BatchNorm behaves differently in train and eval mode (running averages); LayerNorm does not
bn = nn.BatchNorm1d(16)
for _ in range(50):
    bn(torch.randn(32, 16) * 3 + 5)                              # feed batches so it learns running mean ~5, var ~9
sample = torch.randn(1, 16) * 3 + 5                              # ONE sample
bn.eval()
print("\nBatchNorm running_mean (should be ~5):", [round(v, 2) for v in bn.running_mean[:3].tolist()],
      "| running_var (should be ~9):", [round(v, 1) for v in bn.running_var[:3].tolist()])
print("eval-mode output for a single sample works:", bn(sample).shape)
bn.train()
try:
    bn(sample)                                                   # batch of ONE in train mode: variance undefined
except ValueError as e:
    print("train mode, batch size 1 ->", e)


# [5] Residual connections: h_{l+1} = h_l + f(h_l). The identity path gives gradients a highway
def grad_ratio(act, residual, depth=50, width=64):
    torch.manual_seed(0)
    layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
    for lin in layers:
        nn.init.xavier_normal_(lin.weight); nn.init.zeros_(lin.bias)
    h = torch.randn(128, width)
    outs = []
    for lin in layers:
        h = h + act(lin(h)) if residual else act(lin(h))         # the ONLY difference: "h +"
        h.retain_grad(); outs.append(h)
    h.pow(2).mean().backward()
    g = [o.grad.norm().item() for o in outs]
    return g[0], g[-1], g[0] / g[-1]


print("\ngradient at layer 1 versus layer 50 (ratio = layer 1 / layer 50):")
for name, act in (("sigmoid", torch.sigmoid), ("tanh", torch.tanh)):
    for residual in (False, True):
        g1, gL, r = grad_ratio(act, residual)
        print(f"  {name:<8}{'residual' if residual else 'plain   '} |grad| layer 1 {g1:.2e}, layer 50 {gL:.2e}, ratio {r:.2e}")

# TRY IT: in block 1, change `n` or `width` and check that the xavier/he std barely moves.
# BREAK IT: in block 3, use `unbiased=True` in my_layernorm and compare with torch (small but visible difference).
