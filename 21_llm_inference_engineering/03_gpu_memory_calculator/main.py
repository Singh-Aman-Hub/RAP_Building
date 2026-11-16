"""
GOAL:  Estimate GPU memory for serving and training an LLM with back-of-envelope arithmetic, then check
       the KV-cache formula against a real model.  This is THE skill for "will it fit on my GPU?" questions.

This file is the CANONICAL weights + KV-cache memory formula of the whole repo
(modules 16/04, 21/05, 21/07, 21/09 and 21/16 link here instead of re-deriving it).

RUN (from this step folder):  python main.py                 (prints the 7B tables)
                              python main.py --preset llama-3-8b --seq 8192 --batch 4 --kv-dtype fp8
RUN (from repo root):         .venv/bin/python 21_llm_inference_engineering/03_gpu_memory_calculator/main.py
EXPECTED OUTPUT (shortened):
    Model: llama-7b  (6.70B params) | seq 4096 | batch 8 | KV dtype fp16
    precision   weights(GB)   KV cache(GB)   total~(GB)
    fp16             13.4          17.2          33.6
    int4              3.4          17.2          22.6
    KV-cache dtype (weights fp16): fp16 17.2 GB | fp8 8.6 GB | int4 4.3 GB
    GQA effect: MHA (kv_heads=32) 17.2 GB | GQA (kv_heads=8) 4.3 GB | MQA (kv_heads=1) 0.5 GB
    TRAINING memory: full fine-tune 107.2 GB | LoRA 13.5 GB | QLoRA 3.5 GB
    KV cache check on distilgpt2 (32 tokens, fp32): formula 1,179,648 bytes | measured 1,179,648 bytes
NEEDS: Python stdlib for the tables; torch+transformers only for the final real-model check (distilgpt2 cached; --skip-check avoids it).  ~3 s.
"""

import argparse   # built-in: parse command-line options like --seq 4096

# bytes needed per parameter / per cached number at each precision
BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}   # fp8 = e4m3/e5m2: 1 byte each

# name -> (parameters, layers, kv_heads, head_dim).  kv_heads < attention heads means "grouped-query attention" (GQA).
PRESETS = {
    "llama-7b":      (6.7e9, 32, 32, 128),   # classic multi-head attention: every head has its own K,V
    "llama-3-8b":    (8.0e9, 32, 8, 128),    # GQA: only 8 K,V heads -> KV cache is 4x smaller
    "qwen2.5-0.5b":  (0.5e9, 24, 2, 64),
    "distilgpt2":    (82e6, 6, 12, 64),
}

GB = 1e9


def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, batch, bytes_per_value):
    """KV cache = 2 (keys AND values) x layers x kv_heads x head_dim x tokens x sequences x bytes."""
    return int(2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per_value)


def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value):
    """The cache cost of ONE token of ONE sequence (the number to memorise: llama-3-8b fp16 = 131,072 B = 128 KiB)."""
    return int(2 * layers * kv_heads * head_dim * bytes_per_value)


def inference_table(name, params, layers, kv_heads, head_dim, seq, batch, kv_dtype="fp16"):
    print(f"\nModel: {name}  ({params / 1e9:.2f}B params) | seq {seq} | batch {batch} | KV dtype {kv_dtype}")
    print(f"{'precision':<10}{'weights(GB)':>13}{'KV cache(GB)':>15}{'total~(GB)':>13}")
    for prec in ("fp32", "fp16", "int8", "int4"):
        weights = params * BYTES[prec]
        # Common practice: weights may be int4/int8, but the KV cache usually stays fp16 unless you quantise it too.
        kv = kv_cache_bytes(layers, kv_heads, head_dim, seq, batch, BYTES[kv_dtype])
        overhead = 0.10 * (weights + kv)        # rough 10% for activations, CUDA context, fragmentation
        print(f"{prec:<10}{weights / GB:>13.1f}{kv / GB:>15.1f}{(weights + kv + overhead) / GB:>13.1f}")


def kv_dtype_table(params, layers, kv_heads, head_dim, seq, batch):
    """Same model, same weights (fp16): only the KV-cache dtype changes. fp8 halves the cache, int4 quarters it
    (ignoring the small per-group scale overhead; quality impact is the subject of step 07)."""
    row = []
    for dt in ("fp16", "fp8", "int4"):
        row.append(f"{dt} {kv_cache_bytes(layers, kv_heads, head_dim, seq, batch, BYTES[dt]) / GB:.1f} GB")
    print("KV-cache dtype (weights fp16): " + " | ".join(row))


def gqa_table(layers, head_dim, seq, batch, n_attn_heads=32):
    """Same layers/head_dim, vary how many K,V heads exist: MHA = one per query head, GQA = groups share, MQA = one."""
    row = []
    for label, h in (("MHA", n_attn_heads), ("GQA", n_attn_heads // 4), ("MQA", 1)):
        row.append(f"{label} (kv_heads={h}) {kv_cache_bytes(layers, h, head_dim, seq, batch, BYTES['fp16']) / GB:.1f} GB")
    print("GQA effect (fp16 cache, 32 query heads): " + " | ".join(row))


def max_tokens_in_budget(gpu_gb, params, layers, kv_heads, head_dim, weight_dtype="fp16", kv_dtype="fp16", reserve=0.10):
    """How many cached tokens (summed over all sequences) fit once the weights are loaded: the serving-capacity question."""
    free = gpu_gb * GB * (1 - reserve) - params * BYTES[weight_dtype]
    return max(0, int(free // kv_bytes_per_token(layers, kv_heads, head_dim, BYTES[kv_dtype])))


def training_table(params, lora_fraction=0.0012):
    """Rule of thumb for mixed-precision AdamW: weights 2 + grads 2 + fp32 master 4 + Adam m 4 + Adam v 4 = 16 B/param."""
    full = params * 16
    # LoRA: frozen base in fp16 (2 B/param) + 16 B only for the tiny trainable fraction (activations extra, ignored here)
    lora = params * 2 + params * lora_fraction * 16
    # QLoRA: frozen base in 4-bit (0.5 B/param) + same tiny trainable part
    qlora = params * 0.5 + params * lora_fraction * 16
    print("\nTRAINING memory (weights + gradients + optimizer states; activations extra):")
    print(f"  full fine-tune : {full / GB:6.1f} GB   (16 bytes per parameter)")
    print(f"  LoRA           : {lora / GB:6.1f} GB")
    print(f"  QLoRA (4-bit)  : {qlora / GB:6.1f} GB")


def check_against_real_model():
    """Run distilgpt2 once and compare the real cache size with our formula."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
    ids = tok("hello " * 31, return_tensors="pt")            # 31 words -> a known token count
    seq = ids["input_ids"].shape[1]
    with torch.no_grad():
        out = model(**ids, use_cache=True)                    # use_cache=True returns the cached K,V
    cache = out.past_key_values
    # `cache` holds, per layer, a keys tensor and a values tensor. Sum their bytes.
    measured = sum(t.numel() * t.element_size()
                   for layer in cache.layers for t in (layer.keys, layer.values))
    _, layers, kv_heads, head_dim = PRESETS["distilgpt2"]
    formula = kv_cache_bytes(layers, kv_heads, head_dim, seq, 1, BYTES["fp32"])   # model runs in fp32 here
    print(f"\nKV cache check on distilgpt2 ({seq} tokens, fp32): formula {formula:,} bytes | measured {measured:,} bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM GPU memory estimator")
    ap.add_argument("--preset", default="llama-7b", choices=PRESETS)
    ap.add_argument("--seq", type=int, default=4096, help="tokens per sequence (prompt + generated)")
    ap.add_argument("--batch", type=int, default=8, help="concurrent sequences")
    ap.add_argument("--kv-dtype", default="fp16", choices=["fp16", "bf16", "fp8", "int8", "int4"], help="dtype of the cached K,V")
    ap.add_argument("--gpu-gb", type=float, default=24.0, help="GPU size for the 'how many tokens fit' line")
    ap.add_argument("--skip-check", action="store_true", help="skip loading the real model")
    args = ap.parse_args()

    p, layers, kv_heads, head_dim = PRESETS[args.preset]
    inference_table(args.preset, p, layers, kv_heads, head_dim, args.seq, args.batch, args.kv_dtype)
    print(f"KV cost per token: {kv_bytes_per_token(layers, kv_heads, head_dim, BYTES[args.kv_dtype]):,} bytes ({args.kv_dtype})")
    kv_dtype_table(p, layers, kv_heads, head_dim, args.seq, args.batch)
    gqa_table(layers, head_dim, args.seq, args.batch)
    fit = max_tokens_in_budget(args.gpu_gb, p, layers, kv_heads, head_dim, "fp16", args.kv_dtype)
    print(f"On a {args.gpu_gb:.0f} GB GPU with fp16 weights, ~{fit:,} cached tokens fit (e.g. {fit // max(args.seq, 1)} sequences of {args.seq})")
    training_table(p)
    if not args.skip_check:
        check_against_real_model()

    print("\nRead the table: at long context and big batch the KV cache can be BIGGER than the weights.")
    print("That is why PagedAttention (vLLM), GQA and KV-cache quantisation exist.")

# TRY IT: python main.py --batch 1 vs --batch 32 : which column grows?
# TRY IT: python main.py --preset llama-3-8b : compare its KV cache with llama-7b at the same seq/batch (GQA effect).
# TRY IT: python main.py --seq 32768 --batch 1 : long context alone can fill a 24 GB GPU.
# TRY IT: python main.py --preset llama-3-8b --kv-dtype fp8 --gpu-gb 80 : how many tokens fit in an 80 GB card?
# BREAK IT: python main.py --preset gpt5 and read the argparse "invalid choice" error message.
