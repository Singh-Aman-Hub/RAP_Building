"""
GOAL:  Solutions for step 06: top_k=1/no shared expert statistics, learning rates for the aux-loss router, bias
       balancing with gamma 0.1 (oscillation), the top_k=9 error, and the dispatch loop + loss E * sum(f * P).

RUN (from this folder):  python s06_moe.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s06_moe.py
NEEDS: torch (CPU). A few seconds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, d=64, h=128, n_experts=8, top_k=2, n_shared=1):
        super().__init__()
        self.k, self.n = top_k, n_experts
        self.router = nn.Linear(d, n_experts, bias=False)
        self.experts = nn.ModuleList(Expert(d, h) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d, h) for _ in range(n_shared))
        self.register_buffer("bias", torch.zeros(n_experts))

    def forward(self, x):
        probs = self.router(x).softmax(-1)
        top_idx = (probs + self.bias).topk(self.k, dim=-1).indices      # bias only decides WHO is chosen
        top_w = probs.gather(-1, top_idx)                                # raw probabilities decide HOW MUCH each counts
        top_w = top_w / top_w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e in range(self.n):                                          # dispatch: nonzero finds the tokens routed to e
            tok, slot = (top_idx == e).nonzero(as_tuple=True)
            if len(tok):
                out[tok] += top_w[tok, slot, None] * self.experts[e](x[tok])   # indexed add (tok has no duplicates per expert)
        for s in self.shared:
            out = out + s(x)
        self.last = dict(probs=probs, top_idx=top_idx)
        return out

    def counts(self):
        return torch.bincount(self.last["top_idx"].flatten(), minlength=self.n)


def balance_loss(probs, top_idx, n):
    f = torch.bincount(top_idx.flatten(), minlength=n).float() / top_idx.numel()   # fraction of routed slots (not differentiable)
    P = probs.mean(0)                                                              # mean router probability (differentiable)
    return n * (f * P).sum()


count = lambda m: sum(p.numel() for p in m.parameters())
torch.manual_seed(0)
x = torch.randn(2048, 64) + torch.randn(1, 64)       # a shared offset makes the untrained router imbalanced

# 1. top_k=1, no shared expert
for k, sh in ((2, 1), (1, 0)):
    m = MoE(top_k=k, n_shared=sh)
    m(x)
    active = count(m.router) + k * count(m.experts[0]) + count(m.shared)
    c = m.counts()
    print(f"top_k={k}, shared={sh}: active/total {active / count(m):.1%}; untrained loads min {c.min().item()} max {c.max().item()} of {2048 * k} slots")

# 2. aux loss at two learning rates
init = MoE().state_dict()
for lr in (0.05, 0.01):
    A = MoE(); A.load_state_dict(init)
    opt = torch.optim.Adam(A.router.parameters(), lr=lr)
    for _ in range(200):
        A(x)
        loss = balance_loss(A.last["probs"], A.last["top_idx"], 8)
        opt.zero_grad(); loss.backward(); opt.step()
    A(x)
    print(f"aux loss lr={lr}: loads {A.counts().tolist()} loss {balance_loss(A.last['probs'], A.last['top_idx'], 8).item():.3f}")
print("lr 0.05 overshoots: three experts end with zero tokens (dead experts: they get no routed tokens, hence no gradient, and f*P pushes only the others), while lr 0.01 reaches the ideal loss 1.0.")

# 3. bias balancing with gamma 0.01 and 0.1: track max/min load over time
for gamma in (0.01, 0.1):
    B = MoE(); B.load_state_dict(init)
    hist = []
    for step in range(300):
        B(x)
        c = B.counts().float()
        B.bias -= gamma * torch.sign(c - c.mean())
        hist.append((c.max() / c.min().clamp(min=1)).item())
    print(f"bias gamma={gamma}: max/min load at step 50 {hist[50]:.2f}, last 50 steps mean {sum(hist[-50:]) / 50:.2f}, std {torch.tensor(hist[-50:]).std():.3f}")

# 4. top_k larger than number of experts
try:
    MoE(top_k=9)(x)
except RuntimeError as e:
    print("top_k=9 with 8 experts ->", str(e)[:70])
