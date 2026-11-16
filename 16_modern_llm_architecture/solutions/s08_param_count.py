"""
GOAL:  Solution for step 08 "Rebuild": compute the parameter count of Qwen2.5-0.5B from its config numbers with a
       formula, compare with 494,032,768, and show the effect of untying the embeddings.
       (The equality-with-HF modifications (untie / rope_theta / strict=False) need the HF Llama model: run step 08
       with the edit; answers are in solutions.md.)

RUN (from this folder):  python s08_param_count.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s08_param_count.py
NEEDS: python stdlib only. < 1 s.
"""
V, d, L, h, kv, hd, ff = 151936, 896, 24, 14, 2, 64, 4864        # Qwen2.5-0.5B config.json values (step 08)

embed = V * d
attn = d * h * hd + 2 * d * kv * hd + h * hd * d                  # q, k, v, o weights
attn_bias = h * hd + 2 * kv * hd                                  # Qwen2 has biases on q, k, v (not on o)
mlp = 3 * d * ff                                                  # gate, up, down
norms = 2 * d                                                     # two RMSNorm gains per layer
per_layer = attn + attn_bias + mlp + norms
total_tied = embed + L * per_layer + d                            # + final norm; lm_head shares the embedding
print(f"embedding {embed:,} ({embed / total_tied:.1%}) | per layer {per_layer:,} (attention {attn + attn_bias:,}, mlp {mlp:,}, norms {norms:,})")
print(f"total (tied head) = {total_tied:,}   expected 494,032,768 -> {'MATCH' if total_tied == 494_032_768 else 'MISMATCH'}")
print(f"if lm_head were untied: {total_tied + embed:,} (+{embed:,}, the output matrix counted separately)")
print(f"KV cache bytes/token bf16 = {2 * L * kv * hd * 2:,}")
assert total_tied == 494_032_768
