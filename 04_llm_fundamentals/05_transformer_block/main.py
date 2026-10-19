"""
GOAL:   Build ONE GPT-style transformer block by hand (multi-head causal attention + residual +
        LayerNorm + feed-forward MLP), stack it N times, and watch the shape stay constant.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/05_transformer_block/main.py
EXPECTED OUTPUT (short): shapes (2, 10, 64) after every block; one block has 12*d_model^2 weights
        plus a few biases; removing the residual makes the gradient vanish; removing LayerNorm
        makes activations grow with depth; d_model not divisible by n_heads raises an error;
        a real distilgpt2 block has exactly the same parameter structure.
NEEDS:  torch (+ transformers and the cached distilgpt2 for the last section only).

One attention layer is a single mixing step.  A block wraps it with the pieces that make
DEPTH possible:  residual (keep the signal), LayerNorm (keep the scale), MLP (think per token).
Notation: B = batch, T = seq_len, C = d_model, H = n_heads, D = C // H = head_dim.
"""

import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. LayerNorm by hand: normalise each token vector to mean 0 / std 1, then learn a scale+shift
# ---------------------------------------------------------------------------
class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))    # shape: (C,)  learned scale
        self.beta = nn.Parameter(torch.zeros(d_model))    # shape: (C,)  learned shift
        self.eps = eps

    def forward(self, x):                                  # x shape: (B, T, C)
        mean = x.mean(dim=-1, keepdim=True)                # shape: (B, T, 1)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # shape: (B, T, 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)    # shape: (B, T, C)
        return self.gamma * x_hat + self.beta              # shape: (B, T, C)

# ---------------------------------------------------------------------------
# 2. Multi-head causal self-attention (the idea from ../04_attention_from_scratch, batched + split in heads)
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads                  # D
        self.qkv = nn.Linear(d_model, 3 * d_model)          # weight shape: (C, 3C)  Q, K, V packed
        self.proj = nn.Linear(d_model, d_model)             # weight shape: (C, C)   mixes the heads back

    def forward(self, x):                                   # x shape: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)              # each shape: (B, T, C)
        # split C into H heads of size D, then move heads before time so each head is independent
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # shape: (B, H, T, D)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # shape: (B, H, T, D)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # shape: (B, H, T, D)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)     # shape: (B, H, T, T)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)  # shape: (T, T) True = future
        scores = scores.masked_fill(mask, float("-inf"))                # shape: (B, H, T, T) no peeking ahead
        weights = torch.softmax(scores, dim=-1)                         # shape: (B, H, T, T) rows sum to 1
        out = weights @ v                                               # shape: (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, C)            # shape: (B, T, C) glue heads together
        return self.proj(out)                                           # shape: (B, T, C)

# ---------------------------------------------------------------------------
# 3. Feed-forward MLP: expand 4x, GELU, shrink back.  Applied to each token separately.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)           # weight shape: (C, 4C)
        self.proj = nn.Linear(4 * d_model, d_model)         # weight shape: (4C, C)

    def forward(self, x):                                   # x shape: (B, T, C)
        h = nn.functional.gelu(self.fc(x))                  # shape: (B, T, 4C)
        return self.proj(h)                                 # shape: (B, T, C)

# ---------------------------------------------------------------------------
# 4. The block: pre-norm GPT layout.  x = x + attn(LN(x));  x = x + mlp(LN(x))
#    use_residual / use_ln switches exist ONLY so experiments can break the block.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model, n_heads, use_residual=True, use_ln=True):
        super().__init__()
        self.ln1 = MyLayerNorm(d_model) if use_ln else nn.Identity()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = MyLayerNorm(d_model) if use_ln else nn.Identity()
        self.mlp = MLP(d_model)
        self.use_residual = use_residual

    def forward(self, x):                                   # x shape: (B, T, C)
        a = self.attn(self.ln1(x))                          # shape: (B, T, C)
        x = x + a if self.use_residual else a               # shape: (B, T, C) residual = add the input back
        m = self.mlp(self.ln2(x))                           # shape: (B, T, C)
        x = x + m if self.use_residual else m               # shape: (B, T, C)
        return x

def count_params(module):
    return sum(p.numel() for p in module.parameters())

B, T, C, H = 2, 10, 64, 4                                   # batch, seq_len, d_model, n_heads
x = torch.randn(B, T, C)                                    # shape: (B, T, C) fake token vectors

print("== 1. one block ==")
print("input shape :", tuple(x.shape))
block = Block(C, H)
print("output shape:", tuple(block(x).shape), " <- identical: that is what lets us stack blocks")

# sanity check: our LayerNorm equals torch's
print("MyLayerNorm == nn.LayerNorm:", torch.allclose(MyLayerNorm(C)(x), nn.LayerNorm(C)(x), atol=1e-5))

# causal check: changing the LAST token must not change earlier outputs
x2 = x.clone(); x2[:, -1, :] += 5.0                         # shape: (B, T, C)
same_before = torch.allclose(block(x)[:, :-1], block(x2)[:, :-1], atol=1e-5)
print("earlier tokens unaffected by a change in the last token (causal):", same_before)

# ---------------------------------------------------------------------------
# 5. Stack N blocks: shape stays (B, T, C) all the way
# ---------------------------------------------------------------------------
print("\n== 2. stack of 6 blocks ==")
stack = nn.ModuleList([Block(C, H) for _ in range(6)])
h = x                                                       # shape: (B, T, C)
for i, blk in enumerate(stack):
    h = blk(h)                                              # shape: (B, T, C)
    print(f"after block {i + 1}: shape {tuple(h.shape)}")

# ---------------------------------------------------------------------------
# 6. Parameter count: 12 * d_model^2 weights (+ 13 * d_model biases / LayerNorm params)
#    attention: qkv 3C^2 + proj C^2 = 4C^2 ; MLP: C*4C + 4C*C = 8C^2 ; total 12C^2
# ---------------------------------------------------------------------------
print("\n== 3. parameter count of one block ==")
n_block = count_params(block)
formula = 12 * C * C
extra = 13 * C   # biases: qkv 3C + proj C + fc 4C + proj2 C = 9C ; two LayerNorms 2 x 2C = 4C
print(f"d_model={C}: counted {n_block:,} | 12*d^2 = {formula:,} | 13*d (biases+LN) = {extra:,} | sum = {formula + extra:,}")
C_big = 768
print(f"d_model={C_big}: 12*d^2 = {12 * C_big**2:,}  (+ 13*d = {13 * C_big:,}  ->  {12 * C_big**2 + 13 * C_big:,})")

# ---------------------------------------------------------------------------
# 7. Experiment A: remove the residual -> the gradient reaching the input dies with depth
# ---------------------------------------------------------------------------
def grad_at_input(depth, use_residual, use_ln=True):
    torch.manual_seed(1)
    blocks = nn.Sequential(*[Block(C, H, use_residual, use_ln) for _ in range(depth)])
    inp = torch.randn(B, T, C, requires_grad=True)          # shape: (B, T, C)
    out = blocks(inp)                                       # shape: (B, T, C)
    out.pow(2).mean().backward()                            # a dummy loss; gradient flows back through every block
    return inp.grad.norm().item()                           # scalar: size of the gradient at the very first input

print("\n== 4. experiment A: gradient norm at the input (depth 24) ==")
print(f"with residual   : {grad_at_input(24, True):.3e}")
print(f"without residual: {grad_at_input(24, False):.3e}   <- signal from the loss barely reaches the early layers")

# ---------------------------------------------------------------------------
# 8. Experiment B: remove LayerNorm -> activation scale drifts with depth
# ---------------------------------------------------------------------------
def std_per_depth(use_ln, depth=24):
    torch.manual_seed(2)
    h = torch.randn(B, T, C)                                # shape: (B, T, C)
    stds = []
    with torch.no_grad():
        for _ in range(depth):
            h = Block(C, H, True, use_ln)(h)                # shape: (B, T, C)  new random block each time
            stds.append(h.std().item())
    return stds

print("\n== 5. experiment B: std of activations after depth 1, 6, 12, 24 ==")
for use_ln in (True, False):
    s = std_per_depth(use_ln)
    label = "with LayerNorm   " if use_ln else "without LayerNorm"
    print(label, [f"{s[d - 1]:.2f}" for d in (1, 6, 12, 24)])

# ---------------------------------------------------------------------------
# 9. Break it: d_model not divisible by n_heads
# ---------------------------------------------------------------------------
print("\n== 6. break it ==")
try:
    CausalSelfAttention(d_model=64, n_heads=5)
except AssertionError as e:
    print("AssertionError:", e)
try:
    nn.MultiheadAttention(embed_dim=64, num_heads=5)
except AssertionError as e:
    print("torch's own layer says:", e)

# ---------------------------------------------------------------------------
# 10. Compare with a REAL block: distilgpt2 transformer.h[0]
# ---------------------------------------------------------------------------
print("\n== 7. real distilgpt2 block vs ours ==")
from transformers import AutoModelForCausalLM
real = AutoModelForCausalLM.from_pretrained("distilgpt2").eval().transformer.h[0]
for name, p in real.named_parameters():
    print(f"  {name:22} {tuple(p.shape)}")                 # GPT-2 stores Linear weights as (in, out)
n_real = count_params(real)
print(f"real block params: {n_real:,} | ours at d_model=768: {12 * 768**2 + 13 * 768:,} | equal: {n_real == 12 * 768**2 + 13 * 768}")
print("real block param names: ln_1, attn.c_attn (=our qkv), attn.c_proj, ln_2, mlp.c_fc, mlp.c_proj -> same 6 pieces")

# TRY IT: change n_heads to 8 or 1 - the shape and the parameter count do not change (heads only split C).
# TRY IT: depth 48 in experiment A - the no-residual gradient shrinks further.
# BREAK IT: remove `.transpose(1, 2)` before `.view(B, T, C)` in CausalSelfAttention -> shapes still fit, output is silently scrambled.
# BREAK IT: set n_heads=5 -> AssertionError with a readable message (see section 6).
