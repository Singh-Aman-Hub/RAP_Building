"""
GOAL:  Implement LayerNorm vs RMSNorm, post-norm vs pre-norm, GELU-MLP vs SwiGLU (param-matched at 8/3*d) and
       QK-norm in torch; count parameters; show why pre-norm keeps the residual stream clean.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 16_modern_llm_architecture/03_norms_and_ffn/main.py
EXPECTED OUTPUT (shortened):
    RMSNorm matches torch.nn.RMSNorm: True
    GELU MLP params 4,722,432 | SwiGLU params 4,718,592 ...
NEEDS: torch only.  ~3 s.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
D = 768


# ---- 1. LayerNorm vs RMSNorm -------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))                      # learned gain only: no bias, no mean
        self.eps = eps

    def forward(self, x):                                              # x: (batch, seq, d)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()   # 1 / sqrt(mean(x^2) + eps)  (batch, seq, 1)
        return x * rms * self.weight


x = torch.randn(2, 5, D) * 3 + 1.5                                     # deliberately NOT centred, not unit scale
ln, rms = nn.LayerNorm(D, eps=1e-6), RMSNorm(D)
y_ln, y_rms = ln(x), rms(x)
print("input mean/std      :", round(x.mean().item(), 3), round(x.std().item(), 3))
print("LayerNorm mean/std  :", round(y_ln.mean().item(), 4), round(y_ln.std().item(), 4), "(centres AND rescales)")
print("RMSNorm  mean/std   :", round(y_rms.mean().item(), 4), round(y_rms.std().item(), 4), "(rescales only; mean stays)")
print("RMS of RMSNorm out  :", round(y_rms.pow(2).mean(-1).sqrt().mean().item(), 4))
print("RMSNorm matches torch.nn.RMSNorm:", torch.allclose(y_rms, nn.RMSNorm(D, eps=1e-6)(x), atol=1e-6))
print("params: LayerNorm", sum(p.numel() for p in ln.parameters()), "| RMSNorm", sum(p.numel() for p in rms.parameters()))


# ---- 2. FFN variants ------------------------------------------------------------------------------------------
class GeluMLP(nn.Module):                                              # GPT-2 style: d -> 4d -> d
    def __init__(self, d):
        super().__init__()
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class SwiGLU(nn.Module):                                               # Llama style: down( silu(gate(x)) * up(x) )
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):                                              # (batch, seq, d) -> (batch, seq, d)
        return self.down(F.silu(self.gate(x)) * self.up(x))            # the gate decides, elementwise, what passes


count = lambda m: sum(p.numel() for p in m.parameters())
gelu = GeluMLP(D)
hidden = int(8 * D / 3)                                                # 2048: three matrices of d*h ~ two of d*4d
swi = SwiGLU(D, hidden)
print(f"\nGELU MLP : hidden {4 * D}, 2 matrices + biases = {count(gelu):,} params")
print(f"SwiGLU   : hidden {hidden}, 3 matrices, no biases = {count(swi):,} params  (ratio {count(swi) / count(gelu):.4f})")
naive = SwiGLU(D, 4 * D)
print(f"SwiGLU with hidden 4d would cost {count(naive):,} params ({count(naive) / count(gelu):.2f}x) -> that is why hidden is cut to 8/3 d")
for name, d, h in [("Llama-2-7B", 4096, 11008), ("Llama-3-8B", 4096, 14336), ("Qwen2.5-0.5B", 896, 4864)]:
    print(f"  {name}: d={d}, intermediate={h}, ratio {h / d:.3f}  (8/3 = 2.667)")

# ---- 3. Post-norm vs pre-norm: what is on the residual path? ------------------------------------------------------
class Sub(nn.Module):                                                  # stand-in for attention or MLP
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


def run_stack(kind, depth=24, d=64):
    torch.manual_seed(1)
    layers = nn.ModuleList(Sub(d) for _ in range(depth))
    norms = nn.ModuleList(RMSNorm(d) for _ in range(depth))
    h = torch.randn(1, 4, d)
    start = h.clone()
    for sub, nm in zip(layers, norms):
        h = nm(h + sub(h)) if kind == "post" else h + sub(nm(h))     # post: norm(x + f(x)); pre: x + f(norm(x))
    # does the original input still travel, unchanged, along the residual stream?
    return h.pow(2).mean().sqrt().item(), F.cosine_similarity(h.flatten(), start.flatten(), dim=0).item()


for kind in ("post", "pre"):
    rms_out, cos = run_stack(kind)
    print(f"{kind}-norm, 24 layers: output RMS {rms_out:.2f}, cosine(out, input) {cos:.3f}")
print("pre-norm: the residual stream is an unbroken sum, so gradients reach early layers directly (identity path).")

# gradient reaching the first layer's weights at init
def grad_first(kind, depth=48, d=64):
    torch.manual_seed(1)
    layers = nn.ModuleList(Sub(d) for _ in range(depth))
    norms = nn.ModuleList(RMSNorm(d) for _ in range(depth))
    h = torch.randn(1, 4, d)
    for sub, nm in zip(layers, norms):
        h = nm(h + sub(h)) if kind == "post" else h + sub(nm(h))
    h.pow(2).sum().backward()
    return layers[0].lin.weight.grad.norm().item(), layers[-1].lin.weight.grad.norm().item()


for kind in ("post", "pre"):
    g0, gN = grad_first(kind)
    print(f"{kind}-norm, 48 layers: grad norm first layer {g0:.2e} | last layer {gN:.2e} | ratio first/last {g0 / gN:.3f}")


# ---- 4. QK-norm: normalise q and k per head before the dot product so attention logits cannot explode -----------
def attn_logits(q, k):
    return (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5               # (heads, seq, seq)


q, k = torch.randn(4, 6, 64) * 8, torch.randn(4, 6, 64) * 8             # big activations, as in late training
qn, kn = RMSNorm(64), RMSNorm(64)
print(f"\nmax |logit| without QK-norm {attn_logits(q, k).abs().max():.1f} | with QK-norm {attn_logits(qn(q), kn(k)).abs().max():.1f}")
print("softmax of the large logits is almost one-hot (entropy):",
      round(-(F.softmax(attn_logits(q, k), -1) * F.log_softmax(attn_logits(q, k), -1)).sum(-1).mean().item(), 3),
      "vs with QK-norm:", round(-(F.softmax(attn_logits(qn(q), kn(k)), -1) * F.log_softmax(attn_logits(qn(q), kn(k)), -1)).sum(-1).mean().item(), 3))

# ---- 5. Bias terms: modern models drop most of them (Qwen2.5 keeps bias only on q, k, v) -------------------------
print("bias params in a GELU MLP:", 4 * D + D, "of", count(gelu), "| SwiGLU has none")

# TRY IT: change `hidden` to 4 * D//1 and compare parameter counts; or set eps=1e-2 in RMSNorm and see the output RMS drift.
# BREAK IT: in RMSNorm.forward drop keepdim=True. The (batch, seq) rms no longer broadcasts against (batch, seq, d)
#           and you get a RuntimeError (or silently wrong broadcasting when batch == d).
