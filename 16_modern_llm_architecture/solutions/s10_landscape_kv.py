"""
GOAL:  Solution for step 10 "Modify": KV-cache GB at 1M tokens for models whose numbers are stated in this module,
       and which cannot fit one 80 GB GPU together with weights. Only well-established numbers are used; anything
       else would be None (do not guess).

RUN (from this folder):  python s10_landscape_kv.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s10_landscape_kv.py
NEEDS: python stdlib only. < 1 s.
"""
GPU = 80e9
# name: (KV bytes per token in bf16, weight bytes). Values from steps 04 and 08 / README of this module.
rows = {
    "Llama-2-7B (MHA)":       (2 * 32 * 32 * 128 * 2, 7e9 * 2),
    "Llama-3-8B (GQA 8)":     (2 * 32 * 8 * 128 * 2, 8e9 * 2),
    "Qwen2.5-0.5B (GQA 2)":   (2 * 24 * 2 * 64 * 2, 0.494e9 * 2),
    "DeepSeek-V3 (MLA 576)":  (61 * 576 * 2, 671e9 * 1),        # 61 layers x 576 cached values x 2 bytes; weights FP8 (1 byte)
}
print(f"{'model':<24}{'KV GB @1M':>11}{'weights GB':>12}{'total GB':>10}  fits 80 GB?")
for name, (kvb, w) in rows.items():
    kv = kvb * 1_000_000
    print(f"{name:<24}{kv / 1e9:>11.1f}{w / 1e9:>12.1f}{(kv + w) / 1e9:>10.1f}  {'yes' if kv + w < GPU else 'NO'}")
print("Llama-2-7B (MHA) needs 524 GB of cache alone; Llama-3-8B 131 GB: both fail even though the weights fit. "
      "DeepSeek-V3 caches only 70 GB but its weights alone are far beyond one GPU (it needs a multi-GPU node). "
      "Only the tiny GQA model fits (12 GB cache + 1 GB weights).")
