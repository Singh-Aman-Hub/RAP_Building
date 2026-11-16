"""Exercise P6 solution: at what (batch x context) do KV-cache bytes per decode step equal the weight bytes? (Llama-2-7B-like, fp16)"""
layers, kv_heads, head_dim, bytes_ = 32, 32, 128, 2            # full multi-head attention (no GQA)
kv_per_token = 2 * layers * kv_heads * head_dim * bytes_
weights = 6.74e9 * bytes_
print(f"KV bytes per token: {kv_per_token} = {kv_per_token / 2 ** 20:.2f} MiB | weights {weights / 1e9:.1f} GB")
tokens_equal = weights / kv_per_token
print(f"KV bytes == weight bytes when batch x context ~ {tokens_equal:,.0f} tokens (e.g. batch 8 at context {tokens_equal / 8:,.0f})")
assert kv_per_token == 524288
# with GQA (8 KV heads) the KV is 4x smaller, so the crossover moves 4x higher
gqa = 2 * layers * 8 * head_dim * bytes_
print(f"with 8 KV heads (GQA): {gqa} B/token -> crossover at {weights / gqa:,.0f} tokens")
assert gqa * 4 == kv_per_token
print("OK")
