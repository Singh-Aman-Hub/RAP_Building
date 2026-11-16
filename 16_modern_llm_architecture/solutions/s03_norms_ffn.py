"""
GOAL:  Solutions for step 03: effect of RMSNorm eps, SwiGLU 4d vs 8/3 d parameter ratio, keepdim bug, pre-norm
       block parameter count at d=768.

RUN (from this folder):  python s03_norms_ffn.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s03_norms_ffn.py
NEEDS: torch (CPU). < 1 s.
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


def rmsnorm(x, g, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g


x = torch.randn(4, 768) * 0.5            # input rms about 0.5
g = torch.ones(768)
for eps in (1e-5, 1e-1):
    y = rmsnorm(x, g, eps)
    print(f"eps {eps:g}: output rms {y.pow(2).mean(-1).sqrt().mean():.4f}  (ideal 1.0; |error| {abs(1 - y.pow(2).mean(-1).sqrt().mean()):.4f})")
print("eps 0.1 is comparable to mean(x^2)=0.25, so it adds 40% to the denominator's variance: the output is under-scaled.")

d = 768
gelu_mlp = 2 * d * 4 * d
for name, h in (("8d/3 = 2048", round(8 * d / 3)), ("4d = 3072", 4 * d)):
    swi = 3 * d * h
    print(f"SwiGLU hidden {name:<22} params {swi:>10,} vs GELU MLP {gelu_mlp:,}: ratio {swi / gelu_mlp:.3f}")

# keepdim bug
try:
    bad = x.pow(2).mean(-1)               # (4,) instead of (4, 1)
    x / bad                                # (4, 768) / (4,) -> broadcasts on the LAST axis: error since 768 != 4
except RuntimeError as e:
    print("keepdim dropped ->", str(e)[:80])
sq = torch.randn(5, 5)
r = sq.pow(2).mean(-1)
print("silent when the feature size equals the batch size (5x5): output shape", (sq / r).shape, "-> normalises the WRONG axis, no error")


class Block(nn.Module):
    def __init__(self, d, n_heads, n_kv, ff):
        super().__init__()
        hd = d // n_heads
        self.n1, self.n2 = nn.Parameter(torch.ones(d)), nn.Parameter(torch.ones(d))
        self.q, self.k, self.v, self.o = (nn.Linear(d, n_heads * hd, bias=False), nn.Linear(d, n_kv * hd, bias=False),
                                          nn.Linear(d, n_kv * hd, bias=False), nn.Linear(n_heads * hd, d, bias=False))
        self.gate, self.up, self.down = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)


b = Block(768, 12, 12, 2048)
print("pre-norm block params at d=768 (MHA 12 heads, SwiGLU 2048, 2 RMSNorms):", f"{sum(p.numel() for p in b.parameters()):,}")
print("= attention 4*768^2 + MLP 3*768*2048 + norms 2*768 =", f"{4 * 768 ** 2 + 3 * 768 * 2048 + 2 * 768:,}")
