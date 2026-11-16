"""
GOAL:  Build a sparse Mixture-of-Experts layer: router, top-k gating, expert MLPs, shared expert, capacity,
       the Switch-style load-balancing loss, and DeepSeek-V3's auxiliary-loss-free bias balancing.  Show expert
       utilisation and total vs ACTIVE parameters.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 16_modern_llm_architecture/06_mixture_of_experts/main.py
EXPECTED OUTPUT (shortened):
    total params ... active params per token ...
    utilisation before balancing: [..] ; after bias balancing: [..]
NEEDS: torch only.  ~10 s.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class Expert(nn.Module):                                               # an ordinary SwiGLU MLP (see step 03)
    def __init__(self, d, h):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, h, bias=False), nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    def __init__(self, d=64, h=128, n_experts=8, top_k=2, n_shared=1):
        super().__init__()
        self.k, self.n = top_k, n_experts
        self.router = nn.Linear(d, n_experts, bias=False)              # one score per expert, per token
        self.experts = nn.ModuleList(Expert(d, h) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d, h) for _ in range(n_shared))   # always-on experts (DeepSeek, Qwen-MoE)
        self.register_buffer("bias", torch.zeros(n_experts))           # aux-loss-free balancing: used for SELECTION only

    def forward(self, x):                                              # x: (tokens, d)
        logits = self.router(x)                                        # (tokens, n_experts)
        probs = logits.softmax(-1)                                     # router probabilities
        top_idx = (probs + self.bias).topk(self.k, dim=-1).indices     # (tokens, k) choose with the bias...
        top_w = probs.gather(-1, top_idx)                              # ...but weight with the RAW probabilities
        top_w = top_w / top_w.sum(-1, keepdim=True)                    # renormalise the k weights to sum to 1
        out = torch.zeros_like(x)
        for e in range(self.n):                                        # loop over experts (real systems use grouped GEMMs)
            tok, slot = (top_idx == e).nonzero(as_tuple=True)          # which tokens picked expert e, in which slot
            if len(tok):
                out[tok] += top_w[tok, slot, None] * self.experts[e](x[tok])
        for s in self.shared:
            out = out + s(x)                                           # every token goes through shared experts
        self.last = dict(probs=probs, top_idx=top_idx)
        return out

    def counts(self):                                                  # how many tokens each expert received
        return torch.bincount(self.last["top_idx"].flatten(), minlength=self.n)


def balance_loss(probs, top_idx, n):
    """Switch/GShard aux loss: n * sum_e f_e * P_e. f_e = fraction of routed slots to e, P_e = mean router prob of e.
    Minimum (=1) when both are uniform."""
    f = torch.bincount(top_idx.flatten(), minlength=n).float() / top_idx.numel()
    P = probs.mean(0)
    return n * (f * P).sum()


# ---- 1. Total vs active parameters ---------------------------------------------------------------------------------
moe = MoE()
count = lambda m: sum(p.numel() for p in m.parameters())
per_expert = count(moe.experts[0])
total = count(moe)
active = count(moe.router) + moe.k * per_expert + count(moe.shared)
dense_equiv = per_expert
print(f"one expert {per_expert:,} | router {count(moe.router):,} | total layer {total:,}")
print(f"active per token (router + {moe.k} routed + 1 shared) = {active:,}  ->  {active / total:.1%} of the total")

# ---- 2. A skewed token batch: tokens cluster around a few directions so some experts get overloaded ------------------------
torch.manual_seed(1)
centers = torch.randn(3, 64) * 2                                       # 3 "topics"
x = centers[torch.randint(0, 3, (2048,))] + 0.5 * torch.randn(2048, 64)    # (tokens, d)
y = moe(x)
print("\noutput", tuple(y.shape), "| tokens x k =", 2048 * 2, "routed slots")
print("utilisation, untrained router      :", moe.counts().tolist())
print(f"aux balance loss (1.0 is perfect)  : {balance_loss(moe.last['probs'], moe.last['top_idx'], 8).item():.3f}")

# ---- 3. Capacity: drop tokens beyond capacity_factor * average load ---------------------------------------------------------
cap = int(1.25 * 2048 * moe.k / moe.n)                                 # capacity factor 1.25
dropped = (moe.counts() - cap).clamp(min=0).sum().item()
print(f"capacity per expert (factor 1.25) = {cap}; slots that would be DROPPED = {dropped} ({dropped / (2048 * 2):.1%})")

# ---- 4. Method A: train the router with the auxiliary loss -----------------------------------------------------------------------
torch.manual_seed(2)
A = MoE()
A.load_state_dict(moe.state_dict())
opt = torch.optim.Adam(A.router.parameters(), lr=0.01)                 # only the router, to isolate the effect
for step in range(200):
    A(x)
    loss = balance_loss(A.last["probs"], A.last["top_idx"], 8)
    opt.zero_grad()
    loss.backward()
    opt.step()
A(x)
print("\nA) aux-loss trained router, utilisation:", A.counts().tolist(), f"| loss {balance_loss(A.last['probs'], A.last['top_idx'], 8).item():.3f}")

# ---- 5. Method B: DeepSeek-V3 auxiliary-loss-free balancing -------------------------------------------------------------------------
B = MoE()
B.load_state_dict(moe.state_dict())
gamma = 0.01                                                           # bias update speed
for step in range(300):
    B(x)
    c = B.counts().float()
    B.bias -= gamma * torch.sign(c - c.mean())                         # overloaded -> bias down; underloaded -> bias up
B(x)
print("B) bias-balanced (no extra loss), utilisation:", B.counts().tolist(), "| bias:", B.bias.numpy().round(2))
print("   router weights untouched by balancing:", torch.equal(B.router.weight, moe.router.weight))
print(f"   max/min load: untrained {moe.counts().max() / max(moe.counts().min(), 1):.1f}x | A {A.counts().max() / max(A.counts().min(), 1):.2f}x | B {B.counts().max() / B.counts().min():.2f}x")

# ---- 6. Real models: total vs active (from public model cards) ----------------------------------------------------------------------
print("\nmodel                 total params   active/token   experts (routed + shared), top-k")
for row in [("Mixtral 8x7B", "46.7B", "12.9B", "8 + 0, top-2"),
            ("DeepSeek-V3", "671B", "37B", "256 + 1, top-8"),
            ("Qwen3-30B-A3B", "30.5B", "3.3B", "128 + 0, top-8")]:
    print(f"{row[0]:<22}{row[1]:<15}{row[2]:<15}{row[3]}")

# TRY IT: set top_k=1 and n_shared=0 and compare utilisation and the active-parameter percentage; change gamma to 0.1.
# BREAK IT: set top_k=9 with 8 experts. topk raises RuntimeError "selected index k out of range".
