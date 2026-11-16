"""
GOAL:  Implement MHA, MQA, GQA and a simplified MLA (latent KV compression) in one torch module; compute
       KV-cache bytes per token for real model configs; implement sliding-window masks and attention sinks.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 16_modern_llm_architecture/04_attention_variants_and_kv_cache/main.py
EXPECTED OUTPUT (shortened):
    MHA 1,048,576 params ... GQA-2 ... MQA ... MLA 16x smaller ...
    model                 layers kv_heads head_dim   KB/token   GB @128k
NEEDS: torch only.  ~2 s.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---- 1. One attention class covers MHA, GQA and MQA: only n_kv_heads changes -----------------------------------
class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0                               # each KV head is shared by a whole group of Q heads
        self.h, self.kvh, self.hd = n_heads, n_kv_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, n_heads * self.hd, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)   # fewer K/V heads -> smaller projection AND cache
        self.wv = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.wo = nn.Linear(n_heads * self.hd, d_model, bias=False)

    def forward(self, x, mask=None):                                   # x: (batch, seq, d_model)
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.h, self.hd).transpose(1, 2)     # (batch, heads, seq, hd)
        k = self.wk(x).view(b, t, self.kvh, self.hd).transpose(1, 2)   # (batch, kv_heads, seq, hd)  <- this is what is cached
        v = self.wv(x).view(b, t, self.kvh, self.hd).transpose(1, 2)
        k = k.repeat_interleave(self.h // self.kvh, dim=1)             # broadcast each KV head to its group: (batch, heads, seq, hd)
        v = v.repeat_interleave(self.h // self.kvh, dim=1)
        if mask is None:
            mask = torch.ones(t, t, dtype=torch.bool).tril()           # causal
        att = (q @ k.transpose(-1, -2)) / self.hd ** 0.5               # (batch, heads, seq, seq)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        return self.wo((att @ v).transpose(1, 2).reshape(b, t, -1))

    def cache_floats_per_token(self):
        return 2 * self.kvh * self.hd                                  # K and V, per layer


D, H = 512, 8
x = torch.randn(2, 10, D)
print(f"{'variant':<10}{'n_kv_heads':>11}{'params':>10}{'cache floats/token/layer':>26}")
for name, kvh in [("MHA", 8), ("GQA-2", 2), ("MQA", 1)]:
    m = Attention(D, H, kvh)
    assert m(x).shape == (2, 10, D)
    print(f"{name:<10}{kvh:>11}{sum(p.numel() for p in m.parameters()):>10,}{m.cache_floats_per_token():>26,}")

# MQA/GQA with kv_heads == heads must equal MHA: same class, same maths
a, b = Attention(D, H, 8), Attention(D, H, 8)
b.load_state_dict(a.state_dict())
print("same weights, same output:", torch.allclose(a(x), b(x)))


# ---- 2. Simplified MLA (DeepSeek): cache ONE small latent per token, up-project to K and V when needed ---------
class MLASimple(nn.Module):
    """Teaching version: no decoupled RoPE branch (the real one adds a small separate rotary key part)."""
    def __init__(self, d_model, n_heads, d_latent):
        super().__init__()
        self.h, self.hd = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.down = nn.Linear(d_model, d_latent, bias=False)           # compress: this (batch, seq, d_latent) IS the cache
        self.up_k = nn.Linear(d_latent, d_model, bias=False)           # expand latent -> per-head keys
        self.up_v = nn.Linear(d_latent, d_model, bias=False)           # expand latent -> per-head values
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        latent = self.down(x)                                          # (batch, seq, d_latent)  cached
        q = self.wq(x).view(b, t, self.h, self.hd).transpose(1, 2)
        k = self.up_k(latent).view(b, t, self.h, self.hd).transpose(1, 2)
        v = self.up_v(latent).view(b, t, self.h, self.hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(o.transpose(1, 2).reshape(b, t, -1))


mla = MLASimple(D, H, d_latent=64)
print(f"\nMLA: out {tuple(mla(x).shape)}, cache floats/token/layer = 64 (MHA would be {2 * D}) -> {2 * D / 64:.0f}x smaller")


# ---- 3. KV-cache size formula and real configs ---------------------------------------------------------------------
# bytes per token = 2 (K and V) * n_layers * n_kv_heads * head_dim * bytes_per_value
def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes


configs = [  # (name, layers, kv_heads, head_dim)  all read from the public config.json files
    ("GPT-2 small (MHA)", 12, 12, 64),
    ("Llama-2-7B (MHA)", 32, 32, 128),
    ("Llama-3-8B (GQA)", 32, 8, 128),
    ("Llama-3-70B (GQA)", 80, 8, 128),
    ("Qwen2.5-0.5B (GQA)", 24, 2, 64),
]
print(f"\n{'model':<22}{'layers':>7}{'kv_heads':>9}{'head_dim':>9}{'KB/token':>10}{'GB @128k tokens':>17}")
for name, L, kv, hd in configs:
    b = kv_bytes_per_token(L, kv, hd)
    print(f"{name:<22}{L:>7}{kv:>9}{hd:>9}{b / 1024:>10.1f}{b * 131072 / 1e9:>17.1f}")
# MLA (DeepSeek-V3): per layer cache = kv_lora_rank 512 + rope key dim 64 = 576 values (not x2: one shared latent)
mla_b = 61 * (512 + 64) * 2
print(f"{'DeepSeek-V3 (MLA)':<22}{61:>7}{'latent':>9}{'576':>9}{mla_b / 1024:>10.1f}{mla_b * 131072 / 1e9:>17.1f}")
mha_equiv = 61 * 128 * (192 + 128) * 2                  # layers x heads x (K dim 128+64 rope, V dim 128) x 2 bytes (bf16)
print(f"  (DeepSeek-V3 with plain MHA, 128 heads x (192 K + 128 V) dims, would need {mha_equiv / 1024:.0f} KB/token = {mha_equiv / mla_b:.0f}x more)")

# ---- 4. Sliding-window and local/global masks ------------------------------------------------------------------------
def sliding_window_mask(t, window):
    i, j = torch.arange(t)[:, None], torch.arange(t)[None, :]
    return (j <= i) & (j > i - window)                                 # causal AND within the last `window` tokens


print("\nsliding window 3, T=8 (1 = visible):")
print(sliding_window_mask(8, 3).int())
# Gemma-3 style: 5 local layers (window 1024) for every 1 global layer -> cache of local layers is capped
T, win, layers, glob_every = 128_000, 1024, 48, 6
n_global = layers // glob_every
local_tokens = min(T, win)
frac = (n_global * T + (layers - n_global) * local_tokens) / (layers * T)
print(f"5:1 local:global, T={T:,}, window={win}: cache is {frac:.1%} of an all-global model")

# ---- 5. Attention sinks: the first tokens soak up attention; keep them when you evict old tokens ------------------------
def sink_mask(t, window, n_sink):
    m = sliding_window_mask(t, window)
    m[:, :n_sink] |= torch.arange(t)[:, None] >= torch.arange(n_sink)[None, :]   # always allow the first n_sink tokens (causally)
    return m


print("\nwindow 3 + 1 sink token, T=8:")
print(sink_mask(8, 3, 1).int())
# prove the sink phenomenon on gpt2: how much attention does token 0 receive on average?
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok, lm = AutoTokenizer.from_pretrained("gpt2"), AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
ids = tok("The quick brown fox jumps over the lazy dog because it was a sunny day in the park.", return_tensors="pt")
with torch.no_grad():
    attn = lm(**ids, output_attentions=True).attentions               # tuple of 12 x (1, heads, seq, seq)
seq = ids["input_ids"].shape[1]
first = torch.stack([a[0, :, 1:, 0].mean() for a in attn])            # mean weight on token 0 from later tokens, per layer
print(f"gpt2: attention paid to token 0 by later tokens, per layer: {first.numpy().round(2)} (uniform attention would give {sum(1 / (i + 1) for i in range(1, seq)) / (seq - 1):.2f})")

# TRY IT: change n_kv_heads from 2 to 4 in GQA-2 and read the cache column; add a 5th row to `configs` for a model you know.
# BREAK IT: Attention(512, 8, 3). 8 % 3 != 0 so the AssertionError fires: query heads cannot be split into equal groups.
