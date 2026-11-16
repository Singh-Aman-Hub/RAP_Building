"""
GOAL:  A reference table of how real models differ from GPT-2, COMPUTED from a dict of public config values
       (layers, heads, kv heads, head_dim, window pattern, total/active params).  Computes KV-cache bytes per token
       and at 128k context, the active-parameter ratio, and a "what changed since GPT-2" summary.
       Every number below was read from the model's config.json / model card (see References in the README).
       Where a value could not be verified it is None and the table prints "n/a" -- never guessed.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 16_modern_llm_architecture/10_architecture_landscape_2026/main.py
EXPECTED OUTPUT (shortened):
    model  total  active  active%  attention  KV KB/token  KV GB @128k
NEEDS: python standard library only.  < 1 s.
"""
# ---- 1. The data: public numbers only. None = not stated in the config/card we checked ---------------------------------------------
# layers L, kv_heads, head_dim for standard (GQA/MHA) attention; `global_layers` = layers that attend to ALL tokens,
# `local_window` = window of the other (sliding/chunked) layers.  MLA models store a latent: mla_dim = kv_lora_rank + rope dim.
MODELS = {
    "GPT-2 small (2019)":    dict(total=0.124e9, active=0.124e9, L=12, kv_heads=12, head_dim=64, attn="MHA", ffn="GELU MLP", pos="learned abs", ctx=1_024, norm="LayerNorm pre"),
    "Llama 3.1 8B (2024)":   dict(total=8.0e9, active=8.0e9, L=32, kv_heads=8, head_dim=128, attn="GQA", ffn="SwiGLU", pos="RoPE 5e5 + scaling", ctx=131_072, norm="RMSNorm pre"),
    "Qwen2.5-0.5B (2024)":   dict(total=0.494e9, active=0.494e9, L=24, kv_heads=2, head_dim=64, attn="GQA", ffn="SwiGLU", pos="RoPE 1e6", ctx=32_768, norm="RMSNorm pre"),
    "Qwen3-8B (2025)":       dict(total=8.2e9, active=8.2e9, L=36, kv_heads=8, head_dim=128, attn="GQA + QK-norm", ffn="SwiGLU", pos="RoPE 1e6", ctx=40_960, norm="RMSNorm pre"),
    "Qwen3-235B-A22B":       dict(total=235e9, active=22e9, L=94, kv_heads=4, head_dim=128, attn="GQA + QK-norm", ffn="MoE 128e top-8", pos="RoPE 1e6", ctx=40_960, norm="RMSNorm pre"),
    "gpt-oss-120b (2025)":   dict(total=116.8e9, active=5.1e9, L=36, kv_heads=8, head_dim=64, attn="GQA, alt. sliding(128)/full", ffn="MoE 128e top-4", pos="RoPE+YaRN", ctx=131_072, norm="RMSNorm pre",
                                  global_layers=18, local_window=128),
    "Gemma 3 27B (2025)":    dict(total=27e9, active=27e9, L=62, kv_heads=16, head_dim=128, attn="GQA, 5 local(1024):1 global", ffn="GeGLU dense", pos="RoPE (local 1e4 / global 1e6)", ctx=131_072, norm="RMSNorm pre+post",
                                  global_layers=10, local_window=1024),
    "Llama 4 Scout (2025)":  dict(total=109e9, active=17e9, L=48, kv_heads=8, head_dim=128, attn="GQA, chunked(8192) + 1/4 NoPE global", ffn="MoE 16e top-1 (+shared)", pos="RoPE / NoPE", ctx=10_485_760, norm="RMSNorm pre + QK-norm",
                                  global_layers=12, local_window=8192),
    "DeepSeek-V3 (2024-12)": dict(total=671e9, active=37e9, L=61, mla_dim=512 + 64, attn="MLA", ffn="MoE 256e+1 shared top-8", pos="RoPE+YaRN (decoupled)", ctx=163_840, norm="RMSNorm pre"),
    "Kimi K2 (2025)":        dict(total=1.0e12, active=32e9, L=61, mla_dim=512 + 64, attn="MLA (DeepSeek-V3 design)", ffn="MoE 384e+1 shared top-8", pos="RoPE+YaRN", ctx=131_072, norm="RMSNorm pre"),
    "Qwen3.5-397B-A17B":     dict(total=397e9, active=17e9, L=60, kv_heads=2, head_dim=256, attn="3 Gated DeltaNet : 1 gated attn", ffn="MoE 512e+1 shared, 10 routed", pos="RoPE (attn layers)", ctx=262_144, norm="RMSNorm",
                                  global_layers=15, local_window=0),         # DeltaNet layers keep a fixed-size state: 0 per-token cache
    "GLM-5 (2026)":          dict(total=744e9, active=40e9, L=78, attn="MLA + DeepSeek Sparse Attention", ffn="MoE 256e, 8 routed + 1 shared", pos=None, ctx=None, norm=None),
    "DeepSeek-V4-Pro (2026)": dict(total=1.6e12, active=49e9, L=None, attn="Compressed Sparse + Heavily Compressed attn", ffn="MoE", pos=None, ctx=1_000_000, norm=None),
    "Kimi K2.5 (2026)":      dict(total=1.0e12, active=None, L=None, attn="MLA (K2-style) + vision", ffn="MoE", pos=None, ctx=None, norm=None),  }


# ---- 2. KV cache arithmetic -------------------------------------------------------------------------------------------------------------------------
def kv_bytes(m, tokens, dtype_bytes=2):
    """Total KV-cache bytes for ONE sequence of `tokens` tokens. None when the config does not let us compute it."""
    if m.get("L") is None:
        return None
    if "mla_dim" in m:                                                            # MLA: one shared latent per layer per token
        return m["L"] * m["mla_dim"] * dtype_bytes * tokens
    if "kv_heads" not in m:
        return None
    per_layer_token = 2 * m["kv_heads"] * m["head_dim"] * dtype_bytes             # K and V
    g = m.get("global_layers", m["L"])
    local_layers = m["L"] - g
    local_tokens = min(tokens, m.get("local_window", tokens))                     # local layers never keep more than `window` tokens
    return per_layer_token * (g * tokens + local_layers * local_tokens)


def fmt(x, unit, nd=1):
    return "n/a" if x is None else f"{x / unit:.{nd}f}"


# ---- 3. The table --------------------------------------------------------------------------------------------------------------------------------------
print(f"{'model':<25}{'total B':>9}{'active B':>9}{'active%':>8}{'layers':>7}{'KV KB/tok @1k':>15}{'KV GB @128k':>13}  attention")
for name, m in MODELS.items():
    ratio = None if m["active"] is None else m["active"] / m["total"]
    k1k = kv_bytes(m, 1_000)
    k128 = kv_bytes(m, 131_072)
    print(f"{name:<25}{fmt(m['total'], 1e9):>9}{fmt(m['active'], 1e9):>9}{('n/a' if ratio is None else f'{ratio:.0%}'):>8}"
          f"{m['L'] if m['L'] else 'n/a':>7}{fmt(None if k1k is None else k1k / 1_000, 1024):>15}{fmt(k128, 1e9):>13}  {m['attn']}")
print("\nKV columns assume bf16/fp16 cache, batch 1.  KB/tok @1k = bytes per token when the context is 1,000 tokens (local layers fully used).")
print("n/a = the number was not stated in the config or card we checked (we do not guess).")

# ---- 4. What changed since GPT-2: count the columns ------------------------------------------------------------------------------------------------------
print("\nComponent       GPT-2 (2019)        2025-26 open models")
for row in [("tokenizer", "BPE 50k", "byte-level BPE, 128k-262k vocab"),
            ("position", "learned absolute", "RoPE (+YaRN scaling), sometimes NoPE layers"),
            ("norm", "LayerNorm", "RMSNorm, pre-norm (sometimes also post / QK-norm)"),
            ("FFN", "GELU MLP 4d", "SwiGLU/GeGLU, or MoE with shared + routed experts"),
            ("attention", "MHA", "GQA, MLA, sliding/chunked, hybrid with linear attention"),
            ("kernel", "naive softmax", "FlashAttention / SDPA / paged KV cache"),
            ("decoding", "1 token/step", "speculative decoding, multi-token prediction"),
            ("training", "next-token", "+ RL for reasoning (test-time compute)")]:
    print(f"{row[0]:<16}{row[1]:<20}{row[2]}")

# ---- 5. Total vs active: the MoE bargain ----------------------------------------------------------------------------------------------------------------------
print("\nMoE bargain (memory ~ total params, compute per token ~ active params):")
for n in ("Qwen3-235B-A22B", "gpt-oss-120b (2025)", "DeepSeek-V3 (2024-12)", "Kimi K2 (2025)", "Qwen3.5-397B-A17B"):
    m = MODELS[n]
    print(f"  {n:<24} weights in bf16 ~ {m['total'] * 2 / 1e9:>6.0f} GB | FLOPs/token ~ 2 x {m['active'] / 1e9:.0f}B = {2 * m['active'] / 1e9:.0f} GFLOP | a dense model with the same FLOPs has {m['active'] / m['total']:.0%} of the weights")

# TRY IT: add a model you care about: copy a config.json's num_hidden_layers / num_key_value_heads / head_dim into a new dict row.
# BREAK IT: give a model local_window but forget global_layers. kv_bytes() then treats ALL layers as global and over-reports the cache.
