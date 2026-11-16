# Step 09 — Prefill vs decode and the roofline model

## The story so far
[Step 08](../08_hf_inference_batching_kvcache/WALKTHROUGH.md) showed that the KV cache and batching make Hugging Face generation fast, and it measured one prefill and one decode step. [Step 03](../03_gpu_memory_calculator/WALKTHROUGH.md) gave the memory formulas (bytes per parameter, KV bytes per token). Both left one question open: *why* is one forward pass over 500 prompt tokens only a few times slower than a forward pass over a single token, and why does batching 16 users cost almost nothing extra per step?

## The pain point
"Decode is memory-bound, prefill is compute-bound" is the most repeated sentence in LLM serving, and most people can say it but cannot prove it. Without a model of the hardware you cannot predict whether a change (bigger batch, int4 weights, a faster GPU) will help.

## The spark
Every operation is limited by one of two roofs: how many FLOPs per second the chip can do (**compute roof**) or how many bytes per second it can pull from memory (**bandwidth roof**). Divide the two and you get the **ridge point**: the number of FLOPs you must do per byte moved to keep the compute units busy. A workload below the ridge waits for memory; above it, it waits for arithmetic. Count FLOPs and bytes per token and you can place prefill and decode on that picture, then check it against a stopwatch.

## Goal
1. Measure this laptop: matmul GFLOP/s and copy GB/s, giving a roofline and a ridge point.
2. Compute FLOPs and bytes per token from the parameter count and the arithmetic intensity of decode at each batch size.
3. Measure prefill time versus prompt length and decode time per token versus batch size, on distilgpt2 and a Qwen2.5-0.5B-shaped model.

## The theory (read this before the interview)

**Two phases of one request**

```
prompt: "Explain KV cache ..."  (L tokens)                  output tokens
|---------------- PREFILL ----------------|---- DECODE: 1 token per forward pass ----|
 ONE forward pass over all L tokens        t1 -> t2 -> t3 -> ... (each needs the previous one)
 fills the KV cache; many tokens at once   reads ALL weights + the whole KV cache every step
 => big matrix x matrix products            => matrix x VECTOR products (batch 1)
 => compute-bound                           => memory-bound
 latency felt by user = TTFT                latency felt by user = ITL / TPOT
```

**FLOPs per token.** A matrix multiply of an `(m x k)` by a `(k x n)` matrix costs `2*m*k*n` FLOPs (each output = k multiplies + k adds). Every weight in the model is used once per token, as one multiply and one add. So

`FLOPs per token  ~  2 * N_params`   (ignores the small attention-score term and the embedding lookup)

A 7B model: about 14 GFLOP per token. A prompt of L tokens costs `2*N*L` FLOPs in prefill.

**Bytes per decode step.** To produce one token the GPU must stream every weight from memory at least once, plus the KV cache of every sequence in the batch:

`bytes per decode step  ~  N_params * bytes_per_param  +  B * context * kv_bytes_per_token`

(7B in fp16: 14 GB of weights. KV formula: [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md).)

**Arithmetic intensity** = FLOPs / bytes moved (unit: FLOP per byte).

```
decode, batch B, weights only:   (2*N*B) / (N*bytes_per_param)  =  2*B / bytes_per_param
   fp32 (4 B): B/2    fp16 (2 B): B     int8 (1 B): 2B     int4 (0.5 B): 4B   FLOP/byte
prefill, L tokens in one pass:   (2*N*L) / (N*bytes_per_param)  =  2*L / bytes_per_param   (weights are read ONCE for L tokens)
```

So **batch size B in decode plays the same role that prompt length L plays in prefill: it is how many tokens share one read of the weights.** That is why prefill (L in the hundreds) is compute-bound and decode at B=1 (intensity 1 in fp16) is memory-bound.

**Roofline model.**

```
attainable FLOP/s = min( peak_FLOPs ,  bandwidth * intensity )

 FLOP/s
   ^            peak compute ------------------------
   |                         /
   |                       /   <- slope = memory bandwidth
   |                     /
   |                   /
   |  decode B=1 ->  /                 <- prefill, big batches
   +-----------------+---------------------------> intensity (FLOP/byte)
                ridge point = peak_FLOPs / bandwidth
```

Ridge point examples (vendor spec sheets, dense bf16, quoted as reference only): A100 80GB 312 TFLOP/s and 2.04 TB/s gives about 153 FLOP/byte; H100 SXM 989 TFLOP/s and 3.35 TB/s gives about 295. Decode at B=1 has intensity about 1 in fp16, so it uses well under 1 percent of the compute on a GPU. Batching "up" to B ~ 150-300 is nearly free, until the KV cache bytes (which grow with B) start to matter.

**Why batching is almost free in decode.** Step time ~ max(time to stream weights, time to do the math). Streaming weights costs the same at B=1 and B=16, and the math for 16 tokens still fits in the idle compute. So tokens/s rises almost linearly with B while time per step is flat. It stops being free (a) when the intensity crosses the ridge, or (b) when KV cache reads dominate, or (c) when the KV cache does not fit in memory ([step 11](../11_paged_attention_fragmentation/WALKTHROUGH.md)).

**Prefill time versus prompt length.** Time is about `2*N*L / achieved_FLOPs` plus an attention term that grows with L squared: `~4 * layers * d_model * L^2` FLOPs. For small L the matrix products are too small to use the chip well, so ms per token falls as L grows (low efficiency, fixed overheads). For medium L it is near-linear. For very long L (thousands of tokens) the L-squared attention term takes over and time grows faster than linear. For distilgpt2 the two terms are equal around L ~ 4,600, which is beyond its 1,024-token window, so you will *not* see the superlinear part in this step's measurements; for a 128k-token context you will.

**Chunked prefill.** A long prompt (8,000 tokens) prefilled in one forward pass occupies the GPU for hundreds of milliseconds, and every running request's next token waits for it (a spike in ITL for everyone). Chunked prefill splits the prompt into chunks of a fixed token budget (say 512) and puts one chunk plus all the running decode tokens in every scheduler iteration. The prefill still finishes (slightly later) but nobody stalls. vLLM and SGLang both do this; the scheduler is simulated in [step 12](../12_continuous_batching_prefix_caching/WALKTHROUGH.md).

**Disaggregated prefill/decode (concept).** Because the phases want different hardware (compute for prefill, bandwidth and memory for decode) large deployments run them on separate pools of GPUs: prefill workers compute the KV cache for a prompt and ship it over a fast interconnect to decode workers. Benefit: TTFT and ITL targets can be tuned independently and neither phase interferes with the other. Cost: moving the KV cache (size from step 03) between machines. This is current industry practice at large scale, not something a single laptop or small deployment needs. Engines are surveyed in [step 15](../15_serving_engines_landscape/WALKTHROUGH.md).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Prefill (prompt processing, context encoding, prompt phase) | "pre-fill", TTFT phase | Yes | One parallel forward pass over the whole prompt; fills the KV cache. |
| Decode (generation phase, autoregressive generation, token generation) | "decoding" (also means sampling strategy elsewhere) | Yes | One token per forward pass, each step needs the previous token. |
| Compute-bound (math-bound, FLOP-bound) | ALU-bound | Yes | Time set by peak FLOP/s. |
| Memory-bound (bandwidth-bound, HBM-bound) | bytes-bound | Yes | Time set by bytes moved / memory bandwidth. |
| Arithmetic intensity (operational intensity, FLOPs per byte) | roofline x-axis | Yes | FLOPs done per byte moved from memory. |
| Ridge point (machine balance, knee of the roofline) | | Yes | Intensity where the two roofs meet = peak FLOP/s / bandwidth. |
| HBM bandwidth vs DRAM bandwidth | memory bandwidth | Same idea, different chips | HBM (GPU) is 10-40x faster than laptop DRAM. |
| Chunked prefill (prompt chunking, split-fuse) | Sarathi, SplitFuse | Near-synonyms | Prompt split into token-budget chunks mixed with decode tokens. |
| Disaggregated serving (prefill/decode disaggregation, PD split) | splitwise | Yes | Prefill and decode run on different workers; KV cache transferred. |

## Run it
```bash
cd 21_llm_inference_engineering/09_prefill_decode_roofline
python main.py                                  # both models
MODELS=distilgpt2 python main.py                # skip the Qwen-shaped model (faster)
# from repo root, through the heavy-run lock if other jobs are running:
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/09_prefill_decode_roofline/main.py
```
About 1.5 minutes on CPU when the machine is idle. Honest caveat: Qwen/Qwen2.5-0.5B has only its config and tokenizer in the local cache (no weight file, and nothing is downloaded), so the script builds it with **random weights from the config**. Speed and memory depend on tensor shapes, not weight values, so timings are valid; the text it would produce is meaningless and nothing here prints text.

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–18)

```python
"""
GOAL:  See, with numbers from YOUR laptop, why prefill is compute-bound and decode is memory-bound:
       (1) measure the machine (numpy matmul GFLOP/s + memory-copy GB/s) -> a roofline,
       (2) compute FLOPs and bytes per token from the parameter count,
       (3) MEASURE prefill time vs prompt length and decode time/token vs batch size on distilgpt2 and Qwen2.5-0.5B.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/09_prefill_decode_roofline/main.py
EXPECTED OUTPUT (shortened; numbers vary by machine):
    machine: peak ~N GFLOP/s (numpy fp32 matmul) | ~M GB/s (copy) | ridge point = N/M FLOP per byte
    reference GPUs (spec sheet): A100 ridge ~153, H100 ridge ~295 FLOP/byte
    [distilgpt2] prefill: ms/token falls then flattens as the prompt grows (matmuls get efficient)
    [distilgpt2] decode:  ms per step ~flat for batch 1..8, then rises; tokens/s per step keeps growing
    bytes/step / measured time = achieved GB/s (compare to the copy bandwidth)
NEEDS: torch, numpy, transformers (cached distilgpt2; Qwen2.5-0.5B has only config cached -> random weights, same timing; no downloads).  ~1.5 min on CPU.
       Qwen can be skipped:  MODELS=distilgpt2 python main.py     (heavy: run through the global lock)
Builds on: 08_hf_inference_batching_kvcache (prefill/decode vocabulary), 03_gpu_memory_calculator (bytes/param, KV formula).
"""
```

- **In simple words:** What the script measures and what it prints.
- **Exactly:** A module docstring in the required format; it names the cached models and the one caveat (Qwen weights absent, random init).
- **Why it is here:** So you can compare your own run with the expected shape.

### 2. Imports, offline mode and thread count  (`main.py` lines 19–34)

```python
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import benchlib` works from any cwd
import benchlib

MODELS = os.environ.get("MODELS", "distilgpt2,Qwen/Qwen2.5-0.5B").split(",")
THREADS = int(os.environ.get("THREADS", 4))
torch.set_num_threads(THREADS)

```

- **In simple words:** Load libraries, forbid downloads, and pin the number of CPU threads.
- **Exactly:** `HF_HUB_OFFLINE=1` makes Hugging Face fail rather than download. `sys.path.insert` puts the module folder first so `import benchlib` finds `../benchlib.py` from any directory. `torch.set_num_threads(4)` fixes the thread pool, since CPU timing depends strongly on it.
- **Why it is here:** A benchmark whose thread count is not stated cannot be reproduced.

### 3. Measure the compute roof, the memory roof and a timer  (`main.py` lines 36–59)

```python
# ---- 1. Measure the machine: compute roof and memory roof ------------------------------------------------------------
def measure_gflops(n=1536, reps=5):
    """Best of `reps` fp32 matmuls: FLOPs = 2*n^3 (n^2 outputs, each n multiplies + n adds)."""
    a, b = torch.randn(n, n), torch.randn(n, n)
    a @ b                                                        # warm-up
    best = min(_time(lambda: a @ b) for _ in range(reps))
    return 2 * n ** 3 / best / 1e9


def measure_gbps(mb=256, reps=5):
    """Copy a big tensor (read mb + write mb). Bigger than any cache, so it measures DRAM, not cache."""
    src = torch.empty(mb * 1024 * 1024 // 4)                    # fp32 elements
    src.fill_(1.0)
    dst = torch.empty_like(src)
    dst.copy_(src)
    best = min(_time(lambda: dst.copy_(src)) for _ in range(reps))
    return 2 * mb * 1024 * 1024 / best / 1e9                    # bytes moved = read + write


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t

```

- **In simple words:** Two tiny benchmarks: a big matrix multiply (math speed) and a big copy (memory speed).
- **Exactly:** `measure_gflops`: an `n x n` by `n x n` matmul does `2*n^3` FLOPs; best of 5 timed runs after one warm-up gives GFLOP/s. `measure_gbps`: a 256 MB fp32 tensor is bigger than any cache, so `copy_` is limited by DRAM; bytes moved = read + write = `2 * mb` MB. `_time` just times one call.
- **Why it is here:** These two numbers ARE your machine's roofline; everything else is compared to them.

### 4. Ridge point, reference GPUs and the roofline function  (`main.py` lines 61–72)

```python
GFLOPS, GBPS = measure_gflops(), measure_gbps()
RIDGE = GFLOPS / GBPS                                           # FLOP per byte where the two roofs meet
print(f"threads={THREADS} | machine: ~{GFLOPS:.0f} GFLOP/s fp32 matmul | ~{GBPS:.0f} GB/s copy | ridge point = {RIDGE:.1f} FLOP/byte")
print("reference (vendor spec sheets, NOT measured here; dense bf16):")
for name, tflops, tbps in (("A100 80GB", 312, 2.039), ("H100 SXM", 989, 3.35)):
    print(f"  {name:10s} {tflops} TFLOP/s | {tbps} TB/s | ridge ~{tflops / tbps:.0f} FLOP/byte")


def roofline_gflops(intensity):
    """Attainable GFLOP/s = min(compute roof, bandwidth * FLOP-per-byte). This single line IS the roofline model."""
    return min(GFLOPS, GBPS * intensity)

```

- **In simple words:** Divide the roofs to get the ridge, print GPU spec-sheet numbers for comparison, define the roofline formula.
- **Exactly:** `RIDGE = GFLOPS / GBPS` in FLOP per byte. The A100/H100 numbers are vendor spec-sheet dense bf16 peaks, labelled as reference, not measured. `roofline_gflops(i) = min(GFLOPS, GBPS * i)`.
- **Why it is here:** The one-line roofline model used in the next block.

### 5. Per-model analysis: params, FLOPs, bytes, intensity  (`main.py` lines 74–87)

```python
# ---- 2. Per-token FLOPs and bytes from the parameter count -----------------------------------------------------------
def analyse(name):
    model, tok = benchlib.load_causal_lm(name, threads=THREADS)
    n_params = sum(p.numel() for p in model.parameters())      # tied embeddings are counted once by .parameters()
    weight_bytes = n_params * 4                                # fp32
    flops_per_token = 2 * n_params                             # each weight: 1 multiply + 1 add per token
    print(f"\n[{name}] params {n_params / 1e6:.0f}M | weights {weight_bytes / 1e6:.0f} MB fp32 | FLOPs/token ~ 2*params = {flops_per_token / 1e9:.2f} GFLOP")
    print("  arithmetic intensity of ONE decode step by batch size B  =  2*P*B / (4*P)  = B/2 FLOP/byte (KV bytes ignored):")
    for B in (1, 8, 64, 512):
        ai = flops_per_token * B / weight_bytes
        side = "memory-bound" if ai < RIDGE else "compute-bound"
        print(f"    B={B:<4d} intensity {ai:6.1f} -> {side} on this machine (ridge {RIDGE:.0f}); roofline allows {roofline_gflops(ai):.0f} GFLOP/s")
    return model, tok, n_params, weight_bytes, flops_per_token

```

- **In simple words:** Load a model, count parameters, compute FLOPs per token and weight bytes, and classify decode at several batch sizes.
- **Exactly:** `sum(p.numel() for p in model.parameters())` counts each tied tensor once; `flops_per_token = 2 * n_params`; fp32 weights cost `4 * n_params` bytes. Intensity of a decode step with batch B = `2*P*B / (4*P) = B/2`. Returns the model and the numbers.
- **Why it is here:** Turns the theory into numbers for a real model (82M and 494M parameters).

### 6. Measure prefill time versus prompt length  (`main.py` lines 89–104)

```python
# ---- 3. Measure prefill time vs prompt length ------------------------------------------------------------------------
def best_of(fn, n=3):
    fn()                                                       # warm-up
    return min(_time(fn) for _ in range(n))


@torch.no_grad()
def prefill_sweep(model, flops_per_token, lengths):
    print(f"  PREFILL (one forward over L prompt tokens, batch 1):")
    print(f"  {'L':>5} {'time ms':>9} {'ms/token':>9} {'GFLOP/s':>9} {'% of matmul peak':>17}")
    for L in lengths:
        ids = torch.randint(100, 5000, (1, L))
        t = best_of(lambda: model(input_ids=ids, use_cache=True))
        g = flops_per_token * L / t / 1e9
        print(f"  {L:>5} {t * 1e3:>9.1f} {t * 1e3 / L:>9.3f} {g:>9.0f} {100 * g / GFLOPS:>16.0f}%")

```

- **In simple words:** Time one forward pass over L random tokens for several L and print ms per token and achieved GFLOP/s.
- **Exactly:** `best_of` runs the function once untimed then keeps the minimum of 3. `model(input_ids=ids, use_cache=True)` with `ids` shape `(1, L)` is a prefill; achieved GFLOP/s = `2*P*L / time`, shown also as percent of the matmul peak.
- **Why it is here:** Shows efficiency rising with L (compute-bound regime) and near-linear total time.

### 7. Measure decode time versus batch size  (`main.py` lines 106–117)

```python
# ---- 4. Measure decode step time vs batch size -----------------------------------------------------------------------
def decode_sweep(model, weight_bytes, flops_per_token, batches, ctx=16):
    print(f"  DECODE (context {ctx} tokens, TPOT = mean time per generated token, via benchlib.timed_generate):")
    print(f"  {'batch':>5} {'ms/step':>9} {'tokens/s':>9} {'GB/s weights':>13} {'GFLOP/s':>9} {'step x vs B=1':>14}")
    base = None
    for B in batches:
        ids = torch.randint(100, 5000, (B, ctx))
        tr = benchlib.aggregate(benchlib.run_trials(model, ids, new_tokens=9, warmup=1, trials=3))
        step = tr["tpot_s"]
        base = base or step
        print(f"  {B:>5} {step * 1e3:>9.1f} {B / step:>9.0f} {weight_bytes / step / 1e9:>13.1f} {flops_per_token * B / step / 1e9:>9.0f} {step / base:>13.2f}x")

```

- **In simple words:** For batch sizes 1 to 128 run a short generation and report the time per generated token.
- **Exactly:** `benchlib.run_trials` (warm-up, 3 trials) over `(B, 16)` token prompts generating 9 tokens with a KV cache; `aggregate(...)['tpot_s']` is the mean time per output token excluding the first. `weight_bytes / step` is the effective weight-streaming rate; `step / base` is the slowdown versus B=1.
- **Why it is here:** The experiment that proves 'batching is almost free until the ridge'.

### 8. Run both models  (`main.py` lines 119–125)

```python
for name in MODELS:
    model, tok, P, WB, FPT = analyse(name)
    big = P > 2e8
    prefill_sweep(model, FPT, (8, 32, 128, 512) if big else (8, 16, 32, 64, 128, 256, 512, 960))
    decode_sweep(model, WB, FPT, (1, 4, 16, 64) if big else (1, 2, 4, 8, 16, 32, 64, 128))
    del model

```

- **In simple words:** Loop over the models; use a smaller sweep for the big one.
- **Exactly:** `big = P > 2e8` picks fewer lengths and batches to keep Qwen under the time budget; `del model` frees ~2 GB before anything else loads.
- **Why it is here:** Keeps the mandatory path within the RAM and time limits.

### 9. KV-cache bytes per decode step  (`main.py` lines 126–132)

```python
# ---- 5. KV-cache bytes also move every decode step (grows with context) ---------------------------------------------
# Formula (full derivation: 03_gpu_memory_calculator): kv_bytes_per_token = 2 (K and V) * layers * kv_heads * head_dim * bytes
# Qwen2.5-0.5B: 24 layers, 2 KV heads (GQA), head_dim 64, fp32 -> 2*24*2*64*4 = 24,576 bytes/token = 24 KiB.
kv = 2 * 24 * 2 * 64 * 4
print(f"\nKV bytes/token for Qwen2.5-0.5B fp32 = {kv} B; at context 2048 that is {kv * 2048 / 1e6:.0f} MB read per decode step per sequence")
print("(vs 1976 MB of fp32 weights: tiny here, but for a 7B MHA model at 32k context the KV bytes exceed the weights.)")

```

- **In simple words:** Compute how many KV bytes one sequence adds to each decode step.
- **Exactly:** `2 (K and V) * layers * kv_heads * head_dim * 4 bytes` for Qwen2.5-0.5B (24 layers, 2 KV heads from grouped-query attention, head_dim 64): 24,576 bytes per token. Full derivation in [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md).
- **Why it is here:** The second term of 'bytes per step': negligible here, dominant for big MHA models at long context.

### 10. Chunked prefill arithmetic and TRY IT / BREAK IT  (`main.py` lines 133–142)

```python
# ---- 6. Chunked prefill: why a long prompt must not freeze everyone else's decode (concept, simple arithmetic) -------
# A 8,000-token prompt prefilled in one go occupies the GPU for T_pre. Every running request's next token waits T_pre.
# Splitting it into chunks of C tokens and ALSO running the decode tokens in each iteration caps the stall at ~T_pre*C/8000.
T_PRE_MS, PROMPT, CHUNK = 800.0, 8000, 512                      # illustrative numbers, not measured
print(f"\nchunked prefill (illustrative): unchunked stall {T_PRE_MS:.0f} ms vs chunk of {CHUNK} -> ~{T_PRE_MS * CHUNK / PROMPT:.0f} ms per iteration")

# TRY IT: THREADS=1 python main.py (compute roof drops ~4x; ridge moves; does batch-1 decode time change much?)
# TRY IT: MODELS=distilgpt2 python main.py and compare 'GB/s weights' at batch 1 with the copy GB/s measured at the top.
# BREAK IT: change `ctx=16` in decode_sweep to 900 for distilgpt2: step time rises because attention must read a bigger KV cache.
# BREAK IT: delete the warm-up call in best_of(): the first (smallest) length looks much slower than it really is.
```

- **In simple words:** Illustrate with made-up numbers why chunking caps the decode stall; then the experiments.
- **Exactly:** Stall without chunking is the whole prefill time (800 ms illustrative); with chunk 512 of 8000 tokens it is about `800 * 512 / 8000 = 51 ms` per iteration. The numbers are illustrative, not measured; the scheduler simulation is in step 12.
- **Why it is here:** Gives you the shape of the argument for chunked prefill without needing a GPU.

## Syntax explained

| Syntax | Meaning | Tiny example |
|---|---|---|
| `os.environ.get("MODELS", "a,b").split(",")` | read a knob from the environment with a default | `MODELS=distilgpt2 python main.py` |
| `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` | make the parent folder importable (finds `benchlib.py`) | `parents[1]` = the module folder |
| `lambda: a @ b` | tiny anonymous function so `_time` can time any call | `_time(lambda: a @ b)` |
| `min(_time(f) for _ in range(reps))` | best of N runs (the least-disturbed run) | |
| `2 * n ** 3 / best / 1e9` | FLOPs divided by seconds, scaled to GFLOP/s | |
| `a @ b` | matrix multiplication operator | `(n,n) @ (n,n)` |
| `del model` | drop the reference so memory can be freed before loading the next model | |
| f-string `{x:>9.1f}` | right-align in width 9, one decimal | |

## Libraries & functions

| Name | What it does | Internally |
|---|---|---|
| `torch.randn(n, n) @ ...` | fp32 matrix multiply | calls the platform BLAS (Accelerate on macOS), multi-threaded and cache-blocked |
| `tensor.copy_(src)` | copy a tensor's bytes | a `memcpy`: reads and writes every byte once, so it measures DRAM bandwidth |
| `torch.set_num_threads(n)` | CPU threads for ops | sets the intra-op thread pool size |
| `benchlib.load_causal_lm(name)` | load model + tokenizer (random weights from config if no weight file cached) | `AutoModelForCausalLM.from_pretrained` or `.from_config` |
| `model(input_ids=ids, use_cache=True)` | one forward pass | returns logits `(B, L, vocab)` and the KV cache |
| `benchlib.run_trials` / `aggregate` | warm-up + repeated timed decode, then mean and percentiles | see `../benchlib.py` |

## Execution trace (real run)

Real output, this laptop (Apple M1, 4 threads, other jobs were sharing the CPU, so the Qwen rows are noisy; yours will differ):

```
threads=4 | machine: ~937 GFLOP/s fp32 matmul | ~45 GB/s copy | ridge point = 20.6 FLOP/byte
reference (vendor spec sheets, NOT measured here; dense bf16):
  A100 80GB  312 TFLOP/s | 2.039 TB/s | ridge ~153 FLOP/byte
  H100 SXM   989 TFLOP/s | 3.35 TB/s | ridge ~295 FLOP/byte

[distilgpt2] params 82M | weights 328 MB fp32 | FLOPs/token ~ 2*params = 0.16 GFLOP
  arithmetic intensity of ONE decode step by batch size B  =  2*P*B / (4*P)  = B/2 FLOP/byte (KV bytes ignored):
    B=1    intensity    0.5 -> memory-bound on this machine (ridge 21); roofline allows 23 GFLOP/s
    B=8    intensity    4.0 -> memory-bound on this machine (ridge 21); roofline allows 182 GFLOP/s
    B=64   intensity   32.0 -> compute-bound on this machine (ridge 21); roofline allows 937 GFLOP/s
    B=512  intensity  256.0 -> compute-bound on this machine (ridge 21); roofline allows 937 GFLOP/s
  PREFILL (one forward over L prompt tokens, batch 1):
      L   time ms  ms/token   GFLOP/s  % of matmul peak
      8      24.3     3.035        54                6%
     16      22.0     1.373       119               13%
     32      27.2     0.851       192               21%
     64      42.2     0.660       248               26%
    128      60.4     0.472       347               37%
    256      91.4     0.357       459               49%
    512     155.7     0.304       539               57%
    960     310.3     0.323       507               54%
  DECODE (context 16 tokens, TPOT = mean time per generated token, via benchlib.timed_generate):
  batch   ms/step  tokens/s  GB/s weights   GFLOP/s  step x vs B=1
      1      14.1        71          23.2        12          1.00x
      2      18.3       109          17.9        18          1.30x
      4      19.2       208          17.0        34          1.36x
      8      20.0       400          16.4        66          1.42x
     16      22.4       713          14.6       117          1.59x
     32      30.8      1039          10.6       170          2.18x
     64      45.4      1410           7.2       231          3.22x
    128      83.7      1529           3.9       250          5.93x
[benchlib] no weights cached for Qwen/Qwen2.5-0.5B: using RANDOM weights from config (fine for timing, useless for text)

[Qwen/Qwen2.5-0.5B] params 494M | weights 1976 MB fp32 | FLOPs/token ~ 2*params = 0.99 GFLOP
  arithmetic intensity of ONE decode step by batch size B  =  2*P*B / (4*P)  = B/2 FLOP/byte (KV bytes ignored):
    B=1    intensity    0.5 -> memory-bound on this machine (ridge 21); roofline allows 23 GFLOP/s
    B=8    intensity    4.0 -> memory-bound on this machine (ridge 21); roofline allows 182 GFLOP/s
    B=64   intensity   32.0 -> compute-bound on this machine (ridge 21); roofline allows 937 GFLOP/s
    B=512  intensity  256.0 -> compute-bound on this machine (ridge 21); roofline allows 937 GFLOP/s
  PREFILL (one forward over L prompt tokens, batch 1):
      L   time ms  ms/token   GFLOP/s  % of matmul peak
      8     203.7    25.463        39                4%
     32     130.5     4.077       242               26%
    128     231.8     1.811       546               58%
    512    1667.7     3.257       303               32%
  DECODE (context 16 tokens, TPOT = mean time per generated token, via benchlib.timed_generate):
  batch   ms/step  tokens/s  GB/s weights   GFLOP/s  step x vs B=1
      1     121.5         8          16.3         8          1.00x
      4     112.6        36          17.5        35          0.93x
     16     199.3        80           9.9        79          1.64x
     64     608.0       105           3.3       104          5.01x

KV bytes/token for Qwen2.5-0.5B fp32 = 24576 B; at context 2048 that is 50 MB read per decode step per sequence
(vs 1976 MB of fp32 weights: tiny here, but for a 7B MHA model at 32k context the KV bytes exceed the weights.)

chunked prefill (illustrative): unchunked stall 800 ms vs chunk of 512 -> ~51 ms per iteration
```

Reading the numbers:
- **The machine:** about 940 GFLOP/s and 45 GB/s give a ridge near 21 FLOP/byte. Reference GPUs sit at 153 (A100) and 295 (H100): GPUs need far larger batches before decode stops being memory-bound.
- **Prefill (distilgpt2):** ms per token falls from 3.0 ms at 8 tokens to about 0.3 ms from 256 tokens on, and efficiency climbs from 6% to 55-60% of the matmul peak. Small prompts do not fill the chip (fixed overheads, tiny matrices). Total time is near-linear in L beyond about 256 tokens. No superlinear growth shows up below 960 tokens, exactly as the L-squared estimate predicts (crossover ~4,600).
- **Decode (distilgpt2):** step time is 14 ms at B=1 and only 1.6x that (22 ms) at B=16, while tokens per second goes 71 to 713 (10x). That is "batching is almost free". Past B=32 time per step rises (3.2x at B=64, 5.9x at B=128 in the run) because the matmuls now keep the CPU busy: the intensity of B/2 passes this machine's ridge (~21) at B ~ 40. Tokens/s flattens (1,400 to 1,500 at B=64 to 128).
- **Bandwidth check:** at B=1 the weights are streamed at about 23 GB/s against a 45 GB/s copy test. Copy counts read and write, so one-way read bandwidth is higher, and a single decode step also has non-weight overheads (Python, small ops). The takeaway is the right order of magnitude, not an exact match.
- **Qwen-shaped model (494M params, 1.98 GB fp32):** B=1 decode is 120 ms per step, about 16 GB/s of weights. On an overloaded machine the Qwen sweep is noisy (the 512-token prefill row ran slower per token than the 128 row). Repeat on an idle machine before quoting.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `THREADS=1` | compute roof drops (~4x), batch-1 decode barely changes | decode at B=1 is waiting for memory, not for FLOPs |
| `MODELS=distilgpt2` only | same shapes, finishes in ~30 s | |
| `ctx=900` in `decode_sweep` | step time rises | the KV cache read per step grows with context |
| bf16 weights (`dtype=torch.bfloat16` in `load_causal_lm`) | halves weight bytes; CPU speed may not improve (no fast bf16 path on every CPU) | fewer bytes help only where bf16 math is supported and memory-bound |
| change the prefill lengths to (1, 2, 4, 8) | ms per token rises sharply | decode-like regime: no reuse of weights across tokens |

## Debugging practice
**BREAK IT:** delete the warm-up call `fn()` in `best_of`. Symptom: the first row (L=8) looks 5-10x slower than it should (the real run above already shows a 24 ms floor, which is the fixed per-call cost), and the curve of ms per token is steeper than reality. Fix: always discard at least one warm-up call before timing (lazy initialisation, thread pool start, cache cold).

**BREAK IT 2:** `MODELS=Qwen/Qwen2.5-7B python main.py` raises `OSError: ... does not appear to have a file named pytorch_model.bin or model.safetensors` (with offline mode there is no download) and, before that, its config is also missing. Check `ls ~/.cache/huggingface/hub` first.

## Common mistakes
- Quoting a speed-up from batching without saying whether the workload was memory-bound or compute-bound.
- Mixing up FLOP (a count) and FLOP/s (a rate), and GB (size) and GB/s (rate).
- Counting a copy as "bandwidth" but forgetting it moves each byte twice (read + write) when comparing with a vendor's one-way spec.
- Forgetting the KV cache in "bytes per step": at long context and large batch it can exceed the weights.
- Believing prefill and decode run at the same speed per token: they differ by 10x or more.
- Using a spec-sheet peak (sparse FP8 with 2:4 sparsity, for example) as if it were dense bf16.

## Rebuild checklist
1. Write `_time(fn)` with `perf_counter` and a `best of N` wrapper that does one warm-up call.
2. Measure GFLOP/s with `2*n^3 / seconds` on a big matmul, and GB/s with a big `copy_` counting read plus write.
3. Compute `ridge = GFLOPS / GBPS`, and `roofline(i) = min(GFLOPS, GBPS * i)`.
4. Compute `params`, `flops_per_token = 2*params`, `weight_bytes = params * 4` and intensity `= 2*B / 4` per decode step.
5. Time a prefill forward for several lengths and print ms per token and achieved GFLOP/s.
6. Time decode at several batch sizes (via a manual KV-cache loop) and print tokens/s and ms per step.

## Say it aloud
- "Prefill processes the whole prompt in one forward pass of big matrix-matrix products, so it is compute-bound and sets the time to first token; decode makes one token per pass, so it re-reads all the weights and the KV cache for a single token's worth of math and is memory-bound."
- "FLOPs per token are about two times the parameter count, bytes per decode step are about the weight bytes, so intensity is roughly 2B over bytes per parameter: about B FLOP per byte in fp16, far below a GPU's ridge of 150 to 300."
- "That is why batching is almost free in decode: the step time is set by streaming the weights, which is the same for 1 or 32 sequences, until the intensity reaches the ridge or the KV cache fills memory."
- "Quantizing weights helps decode directly because it moves fewer bytes; it barely helps prefill."
- "Chunked prefill and disaggregated prefill/decode exist because the two phases fight over the same GPU in different ways."

## The new problem (cliffhanger)
Now we can say *why* batching raises throughput, but a real benchmark needs more than "tokens per second": TTFT, inter-token latency, tail percentiles, warm-up, variance, and a way to say whether the throughput is *useful* to users. [Step 10](../10_latency_throughput_benchmark/WALKTHROUGH.md) builds that measurement with `benchlib.py`.
