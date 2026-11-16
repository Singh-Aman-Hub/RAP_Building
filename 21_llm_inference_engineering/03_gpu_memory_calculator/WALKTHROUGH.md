# Step 03 — GPU memory calculator: weights + KV cache on paper

## The story so far
[Step 01](../01_checkpoint_formats_and_loading/WALKTHROUGH.md) showed what is inside a checkpoint and how it is loaded; [step 02](../02_gguf_and_llamacpp_formats/WALKTHROUGH.md) showed the GGUF world and a size table (params x bits per weight / 8). Those are file sizes. A running model needs more than its file: a cache that grows with every token, plus working memory. Before touching quantization (step 04) you need one formula you can do on paper.

## The pain point
Guessing wastes hours. You launch a server, wait for the model to load, and get `CUDA out of memory` as soon as the second user arrives. Or you rent an 80 GB GPU you did not need. Nobody told you that the KV cache of one long conversation can be bigger than the model, or that fine-tuning costs about 16 bytes per parameter, not 2.

## The spark
Memory is arithmetic. Serving = weights (params x bytes) + KV cache (2 x layers x kv_heads x head_dim x tokens x batch x bytes) + overhead. Training adds gradients and AdamW state: about 16 bytes per parameter for a full fine-tune. Compute it on paper first, then check one case against a real model.

> **This step is the canonical memory formula of the repo.** [16/04 (attention variants and KV cache)](../../16_modern_llm_architecture/04_attention_variants_and_kv_cache/WALKTHROUGH.md), [05 (activation quantization)](../05_activation_quant_and_calibration/WALKTHROUGH.md), [07 (KV-cache quantization)](../07_kv_cache_quantization/WALKTHROUGH.md), [09 (roofline)](../09_prefill_decode_roofline/WALKTHROUGH.md) and [16 (cost per token)](../16_cost_per_token/WALKTHROUGH.md) all link here instead of re-deriving it.

## Goal
Answer "will it fit?" with arithmetic: weights + KV cache + overhead for serving (including the KV-cache dtype and the effect of GQA), the number of tokens a GPU can hold, and weights + gradients + optimizer states for training.

## The theory (read this before the interview)

```
weights bytes   = parameters x bytes_per_param        (fp32 4, fp16/bf16 2, fp8 1, int8 1, int4 0.5)
KV cache bytes  = 2 x layers x kv_heads x head_dim x tokens x batch x bytes_per_value
                  (2 = keys AND values;  tokens = prompt + generated;  batch = concurrent sequences)
KV per token    = 2 x layers x kv_heads x head_dim x bytes_per_value
total (serving) ~ (weights + KV cache) x 1.1        (10% rough allowance: activations, CUDA context, fragmentation)
training (AdamW, mixed precision) ~ 16 bytes/param = 2 weights + 2 grads + 4 fp32 master + 4 Adam m + 4 Adam v
LoRA training   ~ frozen base (2 B/param) + 16 B x tiny trainable fraction     (activations extra)
QLoRA training  ~ frozen base (0.5 B/param) + 16 B x tiny trainable fraction   (activations extra)
```

**Worked examples (fp16 cache, 2 bytes per value):**
| Model | layers | kv_heads | head_dim | KV bytes per token | 8k tokens, 1 sequence |
|---|---|---|---|---|---|
| llama-7b style (MHA) | 32 | 32 | 128 | 524,288 (512 KiB) | 4.3 GB |
| Llama-3-8B (GQA) | 32 | 8 | 128 | 131,072 (128 KiB) | 1.07 GB |
| Qwen2.5-0.5B | 24 | 2 | 64 | 12,288 (12 KiB) | 0.10 GB |
| Llama-3-70B (GQA) | 80 | 8 | 128 | 327,680 (320 KiB) | 2.7 GB |

**Three levers that shrink the cache** (they multiply):
1. **GQA / MQA** (fewer KV heads): cache scales linearly with `kv_heads`. 32 -> 8 heads is 4x smaller.
2. **KV-cache dtype**: fp16 -> fp8 halves it, int4 quarters it (plus a small scale overhead that this calculator ignores). Engines expose this as, for example, vLLM's `--kv-cache-dtype fp8`; quality impact is step 07.
3. **Better memory management**: PagedAttention does not shrink the data but removes the waste from fragmentation and over-reservation ([step 11](../11_paged_attention_fragmentation/WALKTHROUGH.md)).

**What the formula leaves out** (so never promise exact numbers): activations and temporary buffers, CUDA/cuBLAS workspace, framework context, allocator fragmentation, tensor-parallel replicas (e.g. embeddings), CUDA graphs, and the draft model if you use speculative decoding. Hence the 10% fudge factor, and real engines are usually told to use a fraction of GPU memory (for example vLLM's `--gpu-memory-utilization`, around 0.9 by default) and give the rest of that to the KV pool.

**Capacity question** (what interviewers actually ask): `cached tokens that fit = (usable GPU bytes - weights) / KV per token`. Divide by the average sequence length for the number of concurrent sequences.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| KV cache (key-value cache) | `past_key_values`, attention cache | Yes | Stored keys and values of past tokens so they are not recomputed. |
| GQA / MQA / MHA | grouped-query / multi-query / multi-head attention | Related | They differ in how many K,V heads exist: MHA = one per query head, GQA = groups share, MQA = one for all. Fewer KV heads means a smaller cache. |
| `kv_heads` | `num_key_value_heads` (HF config), n_kv_head (llama.cpp) | Yes | The number that goes in the formula; not `num_attention_heads`. |
| fp16 / bf16 / fp8 / int8 / int4 | precision, dtype | Different formats | Bytes per value: 2 / 2 / 1 / 1 / 0.5. |
| KV-cache dtype | cache quantization, `--kv-cache-dtype` | Related | Precision of the cached K,V, independent of the weight precision. |
| Weights memory / activation memory / KV memory | VRAM budget | Different parts | Weights are fixed by model size; KV grows with tokens x batch; activations are temporary. |
| GB / GiB | 1e9 / 2^30 bytes | No | This calculator uses GB (1e9); nvidia-smi reports MiB (2^20), so numbers differ by about 5-7%. |
| LoRA / QLoRA / full fine-tune memory | training footprint | Different | Full = 16 B/param; LoRA = frozen fp16 base + tiny trainable part; QLoRA = frozen 4-bit base + tiny trainable part. |

## Run it
```bash
cd 21_llm_inference_engineering/03_gpu_memory_calculator
python main.py                                                   # 7B tables, ~3 s (loads distilgpt2 for the check)
python main.py --preset llama-3-8b --batch 1 --seq 8192
python main.py --preset llama-3-8b --kv-dtype fp8 --gpu-gb 80 --skip-check
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/03_gpu_memory_calculator/main.py
```

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–21)

```python
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
```

- **In simple words:** What the calculator does, that it is the repo's canonical memory formula, and the expected output.
- **Exactly:** Docstring only. It lists the two run modes, including the new `--kv-dtype` flag, and the numbers the default run prints.
- **Why it is here:** Other steps (16/04, 21/05, 21/07, 21/09, 21/16) link here instead of re-deriving the formula.

### 2. Imports and bytes per precision  (`main.py` lines 23–26)

```python
import argparse   # built-in: parse command-line options like --seq 4096

# bytes needed per parameter / per cached number at each precision
BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5}   # fp8 = e4m3/e5m2: 1 byte each
```

- **In simple words:** A lookup of how many bytes each number format uses.
- **Exactly:** `BYTES` maps a dtype name to bytes per value: fp32 4, fp16/bf16 2, fp8 1 (e4m3 or e5m2), int8 1, int4 0.5. `argparse` parses command-line flags.
- **Why it is here:** Memory = number of values x bytes per value. This table is used for weights AND for the cache.

### 3. Model presets and the GB constant  (`main.py` lines 28–36)

```python
# name -> (parameters, layers, kv_heads, head_dim).  kv_heads < attention heads means "grouped-query attention" (GQA).
PRESETS = {
    "llama-7b":      (6.7e9, 32, 32, 128),   # classic multi-head attention: every head has its own K,V
    "llama-3-8b":    (8.0e9, 32, 8, 128),    # GQA: only 8 K,V heads -> KV cache is 4x smaller
    "qwen2.5-0.5b":  (0.5e9, 24, 2, 64),
    "distilgpt2":    (82e6, 6, 12, 64),
}

GB = 1e9
```

- **In simple words:** A small catalogue of models with the four numbers the formulas need.
- **Exactly:** Tuple `(parameters, layers, kv_heads, head_dim)`. llama-7b has 32 KV heads (multi-head attention); llama-3-8b has 8 (GQA); qwen2.5-0.5b has 2. `GB = 1e9` (decimal gigabytes, not GiB).
- **Why it is here:** The KV-cache formula needs exactly these four numbers; `kv_heads`, not the number of query heads, is what counts.

### 4. The KV-cache formula  (`main.py` lines 39–41)

```python
def kv_cache_bytes(layers, kv_heads, head_dim, seq_len, batch, bytes_per_value):
    """KV cache = 2 (keys AND values) x layers x kv_heads x head_dim x tokens x sequences x bytes."""
    return int(2 * layers * kv_heads * head_dim * seq_len * batch * bytes_per_value)
```

- **In simple words:** Count how many bytes the model keeps about the conversation so far.
- **Exactly:** `2 x layers x kv_heads x head_dim x seq_len x batch x bytes_per_value`. The 2 is keys AND values. For llama-7b, seq 4096, batch 8, fp16: 2*32*32*128*4096*8*2 = 17,179,869,184 bytes = 17.2 GB.
- **Why it is here:** This is the canonical formula for "how many users fit on this GPU".

### 5. KV bytes per token  (`main.py` lines 44–46)

```python
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value):
    """The cache cost of ONE token of ONE sequence (the number to memorise: llama-3-8b fp16 = 131,072 B = 128 KiB)."""
    return int(2 * layers * kv_heads * head_dim * bytes_per_value)
```

- **In simple words:** The cache cost of one token of one sequence.
- **Exactly:** Same formula with `seq_len = batch = 1`. llama-3-8b fp16: 2*32*8*128*2 = 131,072 bytes = 128 KiB per token; llama-7b (MHA) fp16 is 512 KiB per token.
- **Why it is here:** A number worth memorising: total cache = per-token cost x total cached tokens.

### 6. Inference memory table  (`main.py` lines 49–57)

```python
def inference_table(name, params, layers, kv_heads, head_dim, seq, batch, kv_dtype="fp16"):
    print(f"\nModel: {name}  ({params / 1e9:.2f}B params) | seq {seq} | batch {batch} | KV dtype {kv_dtype}")
    print(f"{'precision':<10}{'weights(GB)':>13}{'KV cache(GB)':>15}{'total~(GB)':>13}")
    for prec in ("fp32", "fp16", "int8", "int4"):
        weights = params * BYTES[prec]
        # Common practice: weights may be int4/int8, but the KV cache usually stays fp16 unless you quantise it too.
        kv = kv_cache_bytes(layers, kv_heads, head_dim, seq, batch, BYTES[kv_dtype])
        overhead = 0.10 * (weights + kv)        # rough 10% for activations, CUDA context, fragmentation
        print(f"{prec:<10}{weights / GB:>13.1f}{kv / GB:>15.1f}{(weights + kv + overhead) / GB:>13.1f}")
```

- **In simple words:** For each weight precision print weights, KV cache and a rough total; the KV dtype is now a parameter.
- **Exactly:** `weights = params x BYTES[prec]`; `kv` uses `BYTES[kv_dtype]` (default fp16, so quantising weights does not touch it); `overhead = 10%` of weights+KV is a rough allowance for activations, CUDA context and fragmentation (real engines differ).
- **Why it is here:** Shows that quantising weights helps, but the cache then dominates unless the cache dtype is reduced too.

### 7. KV-cache dtype table (new)  (`main.py` lines 60–66)

```python
def kv_dtype_table(params, layers, kv_heads, head_dim, seq, batch):
    """Same model, same weights (fp16): only the KV-cache dtype changes. fp8 halves the cache, int4 quarters it
    (ignoring the small per-group scale overhead; quality impact is the subject of step 07)."""
    row = []
    for dt in ("fp16", "fp8", "int4"):
        row.append(f"{dt} {kv_cache_bytes(layers, kv_heads, head_dim, seq, batch, BYTES[dt]) / GB:.1f} GB")
    print("KV-cache dtype (weights fp16): " + " | ".join(row))
```

- **In simple words:** Same model, same fp16 weights, only the cache number format changes.
- **Exactly:** fp16 -> fp8 halves the cache, fp16 -> int4 quarters it. Scale/zero-point overhead of quantised caches is ignored here (small: per-head or per-group scales), and the accuracy impact is the subject of [step 07](../07_kv_cache_quantization/WALKTHROUGH.md). vLLM exposes this as `--kv-cache-dtype fp8`.
- **Why it is here:** Steps 05/07/16 refer to this block when they talk about KV-cache dtype.

### 8. GQA table (new)  (`main.py` lines 69–74)

```python
def gqa_table(layers, head_dim, seq, batch, n_attn_heads=32):
    """Same layers/head_dim, vary how many K,V heads exist: MHA = one per query head, GQA = groups share, MQA = one."""
    row = []
    for label, h in (("MHA", n_attn_heads), ("GQA", n_attn_heads // 4), ("MQA", 1)):
        row.append(f"{label} (kv_heads={h}) {kv_cache_bytes(layers, h, head_dim, seq, batch, BYTES['fp16']) / GB:.1f} GB")
    print("GQA effect (fp16 cache, 32 query heads): " + " | ".join(row))
```

- **In simple words:** Same layers and head size, vary how many K,V heads exist.
- **Exactly:** MHA = 32 KV heads (one per query head), GQA = 8 (groups of 4 query heads share one K,V head), MQA = 1 (all query heads share). Cache scales linearly with `kv_heads`, so GQA with 8 groups is 4x smaller and MQA 32x smaller.
- **Why it is here:** This is the architectural way to shrink the cache; see [16/04](../../16_modern_llm_architecture/04_attention_variants_and_kv_cache/WALKTHROUGH.md) for how attention uses it.

### 9. How many tokens fit in a GPU (new)  (`main.py` lines 77–80)

```python
def max_tokens_in_budget(gpu_gb, params, layers, kv_heads, head_dim, weight_dtype="fp16", kv_dtype="fp16", reserve=0.10):
    """How many cached tokens (summed over all sequences) fit once the weights are loaded: the serving-capacity question."""
    free = gpu_gb * GB * (1 - reserve) - params * BYTES[weight_dtype]
    return max(0, int(free // kv_bytes_per_token(layers, kv_heads, head_dim, BYTES[kv_dtype])))
```

- **In simple words:** Subtract the weights from the usable GPU memory and divide by the cost of one cached token.
- **Exactly:** `free = gpu_gb x 1e9 x (1 - reserve) - weight bytes`; `tokens = free // kv_bytes_per_token`. Default reserve is 10%. The result is the total cached tokens over ALL sequences, which is what a serving engine's KV pool holds.
- **Why it is here:** Turns the formula into the capacity question interviewers ask: "how many concurrent users on an 80 GB card?"

### 10. Training memory table  (`main.py` lines 83–93)

```python
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
```

- **In simple words:** Estimate memory for full fine-tuning versus LoRA versus QLoRA.
- **Exactly:** Full = 16 bytes per parameter (2 weights + 2 gradients + 4 fp32 master copy + 4 Adam m + 4 Adam v) = 107.2 GB for 6.7B. LoRA = 2 B/param frozen fp16 base + 16 B on a 0.12% trainable fraction = 13.5 GB. QLoRA = 0.5 B/param + same = 3.5 GB. Activations are not counted.
- **Why it is here:** Explains why a 7B full fine-tune does not fit on one 24 GB GPU but QLoRA does.

### 11. Check the formula against a real model  (`main.py` lines 96–113)

```python
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
```

- **In simple words:** Run distilgpt2 once and measure the real cache.
- **Exactly:** Imports are inside the function so the tables work without torch. `"hello " * 31` gives 32 tokens. `model(**ids, use_cache=True).past_key_values` is a cache object; `cache.layers` holds layers with `.keys` and `.values` of shape `(1, 12, 32, 64)`; bytes = `numel x element_size`. (`cache.layers` is the transformers v5 API; older versions returned tuples.)
- **Why it is here:** A formula you have not checked against a real model is a guess.

### 12. Command-line interface  (`main.py` lines 116–124)

```python
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LLM GPU memory estimator")
    ap.add_argument("--preset", default="llama-7b", choices=PRESETS)
    ap.add_argument("--seq", type=int, default=4096, help="tokens per sequence (prompt + generated)")
    ap.add_argument("--batch", type=int, default=8, help="concurrent sequences")
    ap.add_argument("--kv-dtype", default="fp16", choices=["fp16", "bf16", "fp8", "int8", "int4"], help="dtype of the cached K,V")
    ap.add_argument("--gpu-gb", type=float, default=24.0, help="GPU size for the 'how many tokens fit' line")
    ap.add_argument("--skip-check", action="store_true", help="skip loading the real model")
    args = ap.parse_args()
```

- **In simple words:** Let the user choose model, sequence length, batch, cache dtype and GPU size.
- **Exactly:** `choices=PRESETS` validates the model name; `--kv-dtype` selects the cache dtype; `--gpu-gb` sets the capacity line; `--skip-check` avoids loading distilgpt2.
- **Why it is here:** Try your own numbers without editing the file.

### 13. Run the tables and print the lesson  (`main.py` lines 126–138)

```python
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
```

- **In simple words:** Call every function with the chosen preset and print the takeaway.
- **Exactly:** `if __name__ == "__main__":` runs only when executed directly, so other steps can `import` the functions.
- **Why it is here:** The last two lines are the message: the KV cache can exceed the weights.

### 14. TRY IT / BREAK IT  (`main.py` lines 140–144)

```python
# TRY IT: python main.py --batch 1 vs --batch 32 : which column grows?
# TRY IT: python main.py --preset llama-3-8b : compare its KV cache with llama-7b at the same seq/batch (GQA effect).
# TRY IT: python main.py --seq 32768 --batch 1 : long context alone can fill a 24 GB GPU.
# TRY IT: python main.py --preset llama-3-8b --kv-dtype fp8 --gpu-gb 80 : how many tokens fit in an 80 GB card?
# BREAK IT: python main.py --preset gpt5 and read the argparse "invalid choice" error message.
```

- **In simple words:** Experiments in comments.
- **Exactly:** `--preset gpt5` gives `error: argument --preset: invalid choice: 'gpt5' (choose from llama-7b, llama-3-8b, qwen2.5-0.5b, distilgpt2)`.
- **Why it is here:** argparse error messages list the valid values.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `argparse.ArgumentParser` | read `--flags` from the command line | `ap.add_argument("--seq", type=int, default=4096)` |
| `choices=PRESETS` | only dictionary keys are accepted | |
| `action="store_true"` | a flag with no value; present -> `True` | `--skip-check` |
| dict of tuples + unpacking | `p, layers, kv_heads, head_dim = PRESETS[name]` | |
| `f"{x:<10}{y:>13.1f}"` | left / right alignment and decimals | |
| `a // b` | integer (floor) division | tokens that fit |
| `if __name__ == "__main__":` | run only when executed directly, not when imported | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Internally |
|---|---|---|---|
| `argparse` | built-in | command-line options | builds a parser from `add_argument` calls and validates `choices` |
| `out.past_key_values` | transformers | the actual cache returned by a forward pass (`.layers[i].keys / .values`) | per layer, two tensors of shape (batch, kv_heads, seq, head_dim) |
| `tensor.numel()`, `.element_size()` | torch | count and bytes per element | metadata lookups, no data scan |

## Execution trace (real run)
Default command (`python main.py`, llama-7b style, seq 4096, batch 8, distilgpt2 check included):
```

Model: llama-7b  (6.70B params) | seq 4096 | batch 8 | KV dtype fp16
precision   weights(GB)   KV cache(GB)   total~(GB)
fp32               26.8           17.2         48.4
fp16               13.4           17.2         33.6
int8                6.7           17.2         26.3
int4                3.4           17.2         22.6
KV cost per token: 524,288 bytes (fp16)
KV-cache dtype (weights fp16): fp16 17.2 GB | fp8 8.6 GB | int4 4.3 GB
GQA effect (fp16 cache, 32 query heads): MHA (kv_heads=32) 17.2 GB | GQA (kv_heads=8) 4.3 GB | MQA (kv_heads=1) 0.5 GB
On a 24 GB GPU with fp16 weights, ~15,640 cached tokens fit (e.g. 3 sequences of 4096)

TRAINING memory (weights + gradients + optimizer states; activations extra):
  full fine-tune :  107.2 GB   (16 bytes per parameter)
  LoRA           :   13.5 GB
  QLoRA (4-bit)  :    3.5 GB

KV cache check on distilgpt2 (32 tokens, fp32): formula 1,179,648 bytes | measured 1,179,648 bytes

Read the table: at long context and big batch the KV cache can be BIGGER than the weights.
That is why PagedAttention (vLLM), GQA and KV-cache quantisation exist.
```
And `python main.py --preset llama-3-8b --seq 8192 --batch 4 --kv-dtype fp8 --skip-check`:
```

Model: llama-3-8b  (8.00B params) | seq 8192 | batch 4 | KV dtype fp8
precision   weights(GB)   KV cache(GB)   total~(GB)
fp32               32.0            2.1         37.6
fp16               16.0            2.1         20.0
int8                8.0            2.1         11.2
int4                4.0            2.1          6.8
KV cost per token: 65,536 bytes (fp8)
KV-cache dtype (weights fp16): fp16 4.3 GB | fp8 2.1 GB | int4 1.1 GB
GQA effect (fp16 cache, 32 query heads): MHA (kv_heads=32) 17.2 GB | GQA (kv_heads=8) 4.3 GB | MQA (kv_heads=1) 0.5 GB
On a 24 GB GPU with fp16 weights, ~85,449 cached tokens fit (e.g. 10 sequences of 8192)

TRAINING memory (weights + gradients + optimizer states; activations extra):
  full fine-tune :  128.0 GB   (16 bytes per parameter)
  LoRA           :   16.2 GB
  QLoRA (4-bit)  :    4.2 GB

Read the table: at long context and big batch the KV cache can be BIGGER than the weights.
That is why PagedAttention (vLLM), GQA and KV-cache quantisation exist.
```
- Quantising weights shrinks only the weights column; the fp16 KV cache stays 17.2 GB and dominates.
- Same cache, different dtype: fp16 17.2 GB, fp8 8.6 GB, int4 4.3 GB (scales ignored).
- GQA: 32 -> 8 KV heads takes 17.2 GB to 4.3 GB; MQA (1 head) 0.5 GB. Llama-3-8B uses 8.
- A 24 GB GPU with fp16 7B weights holds only about 15.6k cached tokens (3 sequences of 4096); Llama-3-8B with an fp8 cache holds about 85k.
- Training a 6.7B model: full fine-tune 107.2 GB, LoRA 13.5 GB, QLoRA 3.5 GB.
- Formula check on distilgpt2 (32 tokens, fp32): formula = measured = 1,179,648 bytes.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `--batch 1` vs `--batch 32` | only the KV column scales | KV grows with sequences; weights do not |
| `--seq 32768 --batch 1` | 17.2 GB of cache for llama-7b style: one long context equals eight 4k ones | linear in total tokens |
| `--preset llama-3-8b` vs `llama-7b` | cache 4x smaller | GQA, 8 vs 32 KV heads |
| `--kv-dtype fp8` | cache halves, weights unchanged | 1 byte per cached value |
| `--gpu-gb 80` | capacity line grows | more free bytes after the weights |

## Debugging practice
`python main.py --preset gpt5`:
```
main.py: error: argument --preset: invalid choice: 'gpt5' (choose from llama-7b, llama-3-8b, qwen2.5-0.5b, distilgpt2)
```
argparse errors list the valid choices: read to the end of the line. A subtler bug: using the number of **attention** heads instead of **KV** heads in the formula overstates a GQA model's cache by 4x (Llama-3-8B: 32 vs 8). Always read `num_key_value_heads` from the model config ([16/08](../../16_modern_llm_architecture/08_read_a_real_config/WALKTHROUGH.md)).

## Common mistakes
- Using attention heads instead of KV heads.
- Forgetting the 2 (keys and values).
- Using the model's maximum context for `tokens` when the average sequence is much shorter, or the reverse: a long-context request that arrives later and OOMs the server.
- Quoting weights-only memory as "what it needs".
- Treating the 10% overhead as exact; always leave headroom and measure with `nvidia-smi` on the real stack.
- Mixing GB (1e9) and GiB (2^30).

## Rebuild checklist
1. Write `kv_cache_bytes(...)` from memory, and say what each factor is.
2. Compute Llama-3-8B bytes per token (should be 131,072 for fp16).
3. Explain why KV cache grows with batch and length but weights do not.
4. Verify the formula against `past_key_values` for a small model.
5. Add the cache-dtype and GQA variations.

## Say it aloud
- "Serving memory is weights plus KV cache plus overhead, and I compute each term from parameters, layers, KV heads, head dimension and bytes per value."
- "Llama-3-8B costs 128 KiB of cache per token in fp16 because it has only 8 KV heads; with fp8 that is 64 KiB."
- "Full fine-tuning with AdamW costs about 16 bytes per parameter, so a 7B model needs around 107 GB, while LoRA and QLoRA need a small fraction of that."
- "I always do this arithmetic before renting hardware, then confirm with the real engine's memory report."

## The new problem (cliffhanger)
The calculator shows that bytes per value is the lever: weights at 2 bytes are 13 GB, at 0.5 bytes they are 3 GB. But how do you actually squeeze a float into 4 or 8 bits without wrecking the model? Rounding rules, scales, outliers: [step 04, quantization math](../04_quantization_math/WALKTHROUGH.md).
