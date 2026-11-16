"""
GOAL:  Solutions for step 04: KV bytes per token for extra models, GQA-2 vs GQA-4, the divisibility assertion, and a
       proof that MQA equals MHA whose K/V heads are all copies of one head.

RUN (from this folder):  python s04_kv_and_gqa.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s04_kv_and_gqa.py
NEEDS: torch (CPU). < 1 s.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes          # K and V


configs = {"Llama-2-7B (MHA)": (32, 32, 128), "Llama-3-8B (GQA 8)": (32, 8, 128), "Qwen2.5-0.5B (GQA 2)": (24, 2, 64)}
print(f"{'model':<24}{'KB/token':>10}{'GB at 128k':>12}")
for name, (L, kv, hd) in configs.items():
    b = kv_bytes_per_token(L, kv, hd)
    print(f"{name:<24}{b / 1024:>10.1f}{b * 131072 / 1e9:>12.1f}")


class Attention(nn.Module):
    def __init__(self, d, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "each KV head must serve a whole number of query heads"
        self.h, self.g, self.hd = n_heads, n_kv_heads, d // n_heads
        self.wq = nn.Linear(d, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, d, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(b, t, self.g, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, t, self.g, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.h // self.g, dim=1)           # (b, h, t, hd): each kv head serves h/g query heads
        v = v.repeat_interleave(self.h // self.g, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(b, t, -1))


for g in (2, 4, 8):
    a = Attention(512, 8, g)
    print(f"GQA-{g}: params {sum(p.numel() for p in a.parameters()):,}, cache floats/token/layer {2 * g * 64:,}")

try:
    Attention(512, 8, 3)
except AssertionError as e:
    print("Attention(512, 8, 3) ->", e)

# MQA == MHA with all K/V heads copied from the single head
torch.manual_seed(0)
mqa = Attention(512, 8, 1)
mha = Attention(512, 8, 8)
with torch.no_grad():
    mha.wq.weight.copy_(mqa.wq.weight); mha.wo.weight.copy_(mqa.wo.weight)
    mha.wk.weight.copy_(mqa.wk.weight.repeat(8, 1))                # stack the one head's rows 8 times
    mha.wv.weight.copy_(mqa.wv.weight.repeat(8, 1))
x = torch.randn(2, 10, 512)
print("MQA vs MHA-with-copied-heads max diff:", (mqa(x) - mha(x)).abs().max().item())
assert torch.allclose(mqa(x), mha(x), atol=1e-6)
