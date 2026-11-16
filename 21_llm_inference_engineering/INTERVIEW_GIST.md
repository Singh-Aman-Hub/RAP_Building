# Interview Gist — Module 21: LLM inference engineering

Cram sheet. Ordered like the module: **A** loading and storage (01-03), **B** quantization (04-07), **C** performance (08-13), **D** engines, parallelism, cost, report (14-17). "In our run" numbers come from the step WALKTHROUGH traces (distilgpt2 on an 8 GB CPU laptop unless noted; they vary by machine, so quote ratios, not decimals). Full explanations: [README.md](README.md).

**The one-paragraph mental model.** Inference is three resources and a handful of levers. *Memory* (weights + KV cache) decides what fits; *memory bandwidth* decides how fast decode runs (each step streams the weights); *compute* decides how fast prefill runs. Quantization shrinks weights and bytes moved, batching amortises weight reads, paging stops KV waste, prefix caching and speculation avoid work, parallelism adds devices, and $/token = GPU price / delivered tokens.

---

## 1. Key formulas

```
Weights bytes      = params x bytes_per_param           fp32 4 | fp16/bf16 2 | int8/fp8 1 | int4 0.5
Bits per weight    = bits + (scale_bits + zero_bits) / group     int4 g128 fp16 scale = 4 + 16/128 = 4.125
File size          = params x bpw / 8                   7B at Q4_K_M (~4.85 bpw) ~ 4.1 GB vs 13.5 GB fp16
KV bytes / token   = 2 x layers x kv_heads x head_dim x bytes_per_value      (2 = K and V)
KV total           = KV bytes/token x seq_len x batch
Serving memory     = weights + KV + ~10% overhead (activations, CUDA context, fragmentation)
Full AdamW train   = 16 B/param = 2 (w) + 2 (grad) + 4 (fp32 master) + 4 (m) + 4 (v)   + activations
Affine quant       = q = round(x/s) + z ,  x_hat = (q - z) x s ,  s = (max-min)/(2^b - 1) ,  max error s/2
SQNR (dB)          = 10 log10( sum x^2 / sum (x - x_hat)^2 )     ~ +6 dB per extra bit
Arithmetic intens. = FLOPs / bytes moved ;  ridge = peak FLOP/s / peak bytes/s  (A100 153, H100 295 FLOP/byte)
Decode intensity   ~ 2B / bytes_per_param  ->  compute-bound only when  B > ridge x bytes_per_param / 2
Decode step time   ~ (weight bytes + KV bytes read) / memory bandwidth           (memory-bound regime)
TTFT = queue + prefill ;  ITL = gap between consecutive tokens ;  TPOT = (t_last - t_first)/(n-1)
E2E ~ TTFT + TPOT x (n - 1) ;  throughput = batch x tokens / s ;  goodput = tokens/s of requests meeting the SLO
Speculative        E[tokens per target pass] = (1 - a^(k+1)) / (1 - a)   (a = acceptance, k = draft length)
                   speedup = (1 - a^(k+1)) / ((1 - a)(k c + 1))          (c = draft cost / target cost)
Pipeline bubble    (p - 1) / (m + p - 1)      p stages, m micro-batches
TP comms           2 all-reduces per layer per token (after attention and after MLP)
ZeRO per GPU       stage 1 shards optimizer, 2 + gradients, 3 + parameters
$ / 1M tokens      = GPU $/h / (tokens/s x 3600 x utilisation) x 1e6
Break-even         = fixed monthly GPU $ / (blended API $ per 1M)  (blended = 0.8 x input + 0.2 x output)
```

---

## 2. Part A — loading and storage (steps 01-03)

### Formats (step 01-02)
| Format | Contents | Safe? | Used by |
|---|---|---|---|
| pickle (`.bin`, `.pt`) | Python objects, may run code on load (`REDUCE` calls any callable) | **No** for untrusted files | legacy PyTorch |
| **safetensors** | 8-byte header length + JSON header (dtype, shape, offsets) + raw bytes | Yes (numbers only), mmap-able | Hugging Face default |
| **GGUF** | one file: metadata + tensors (block-quantized), 32-byte aligned, dims stored reversed | Yes | llama.cpp, Ollama |

- `torch.load(weights_only=True)` is the default in current PyTorch and refused our booby-trapped pickle (`UnpicklingError`).
- **Lazy loading:** in our run a 300 MB file loaded eagerly grew RSS by ~261 MB (peak 628 MB); `safe_open` reading one 50 MB tensor grew it by 50 MB; a raw `np.memmap` view by 0 MB. `from_pretrained` of distilgpt2 grew RSS by only 21 MB (mapped lazily); a bf16 conversion forces a real copy.
- **Shards:** `model-0000X-of-0000N.safetensors` + `model.safetensors.index.json` (`weight_map` tensor to shard). `device_map` / `meta` device place or size layers without materialising weights.
- **GGUF blocks:** Q8_0 = 32 weights + fp16 scale = 34 bytes (**8.5 bpw**); Q4_0 = 18 bytes (**4.5 bpw**). In our run Q8_0 had ~0.5% relative RMSE and Q4_0 ~8.6% on Gaussian weights, hence **k-quants** (Q4_K_M, 256-weight super-blocks with sub-scales, mixed precision per tensor) and an **importance matrix** (imatrix, calibration-weighted rounding). Q4_K_M on a 7B ~4.1 GB vs 13.5 GB fp16 (~3.3x).
- `convert_hf_to_gguf.py` then `llama-quantize` is the pipeline; Ollama wraps llama.cpp with a registry and a REST API on port 11434.

### GPU memory arithmetic (step 03)
- **Llama-7B-style MHA fp16:** 2 x 32 x 32 x 128 x 2 B = **524,288 B = 0.5 MiB per token**; 2048 tokens = 1 GiB per request.
- **Llama-3-8B (GQA, 8 KV heads) fp16:** 128 KiB per token; 8192 tokens x batch 4 = 4.3 GB (fp8 2.1 GB, int4 1.1 GB).
- **Table in our run (6.7B, seq 4096, batch 8):** fp32 48.4 GB, fp16 **33.6 GB**, int8 26.3 GB, int4 **22.6 GB**; KV is a constant 17.2 GB. Quantising weights shrinks only the weights column; at long context or big batch **the KV cache exceeds the weights**.
- **GQA effect (32 query heads, 8192 x 4):** MHA 17.2 GB, GQA-8 4.3 GB, MQA 0.5 GB.
- **24 GB GPU, fp16 7B weights:** only ~15.6k cached tokens (3 x 4096); Llama-3-8B with an fp8 cache holds ~85k.
- **Training:** 6.7B full fine-tune 107.2 GB, LoRA 13.5 GB, QLoRA 3.5 GB. Formula check on distilgpt2 (32 tokens, fp32): 1,179,648 bytes = measured.
- GB (1e9) vs GiB (2^30) differ by ~7%.

---

## 3. Part B — quantization (steps 04-07)

**Weights vs activations vs KV.** Weights are static (quantize offline, can be sophisticated); activations are dynamic with outlier channels; KV cache grows with context. Naming: **W4A16** = int4 weights, fp16 activations (weight-only, memory-bound win); **W8A8** = both int8 (needs integer GEMM, helps compute-bound prefill too).

| Idea | Rule of thumb | In our run |
|---|---|---|
| Symmetric vs asymmetric | symmetric wastes half the grid on one-sided data (post-ReLU) | int4 post-ReLU 15.11 dB -> 21.71 dB asymmetric (~1 bit) |
| Granularity | per-tensor < per-row < per-group; group 128 is the default | int4 per-tensor 8.11 dB, per-row 17.06, g128 18.58, g64 19.33, g32 20.24 dB (4.0 / 4.125 / 4.25 / 4.5 bpw) |
| Clipping | absmax is rarely MSE-optimal at 4 bits | MSE-optimal clip ~0.30-0.5 of absmax at int4, MSE up to ~10x lower on a real weight |
| FP8 vs int8 | int8 wins on Gaussian, E4M3 on heavy tails | Gaussian: int8 39.3 vs E4M3 31.6 dB; heavy-tailed: E4M3 31.5 vs int8 20.1 dB |
| 4-bit families (~4.25 bpw) | asym int4 g64 ~ NF4 ~ NVFP4 > sym int4 g64 > MXFP4 (power-of-two scale) | 20.81 / 20.69 / 20.44 / 19.33 / 18.78 dB |
| Double quantization | quantise the scales too (QLoRA) | 4.25 -> 4.127 bpw, error unchanged |
| Activation outliers | few channels 10-100x larger set the scale | max/median up to 58.8x on distilgpt2; per-tensor W8A8 23.7 dB vs per-token 37.0 |
| SmoothQuant | divide X by s, multiply W by s (product unchanged), alpha ~0.5 | per-tensor W8A8 23.5 -> 31.9 dB; alpha 0.9 only 24.2 |
| Calibration | MSE rule > min-max; percentile can clip the one real outlier | MSE 25.8 dB, min-max 24.2, 99.9th percentile 21.7; static percentile perplexity 141.68 |
| PTQ vs QAT | QAT wins at very low bits (STE passes gradients through round) | 2-bit: float 0.998, PTQ 0.619, QAT 1.000 |
| GPTQ vs AWQ vs RTN | GPTQ: Hessian-based error compensation; AWQ: scale salient channels; RTN: plain rounding | 4-bit g128 layer SQNR RTN 22.17, AWQ 23.16, GPTQ 27.54 dB |

- **Perplexity (distilgpt2, fp32 25.44):** 4-bit g128 RTN 29.40, AWQ 28.42, **GPTQ 26.16**; 3-bit RTN 42.24, AWQ 39.14, GPTQ 29.00. Methods differ little at 4 bits and a lot at 3 bits. Also W8A16 25.42 (free), W8A8 per-token 25.55 (almost free), W8A8 per-tensor 32.97. Tiny model, tiny text: trust the ordering, not the digits.
- **KV-cache quantization (step 07):** **K wants per-channel scales** (outlier channels, KIVI), **V wants per-token**. In our run per-channel K raised SQNR from 19.9 to 24.1 dB at int4. int8 and fp8 KV are essentially free (KL ~0); fp8 E4M3 is the engine default (no zero-point, hardware support). A 32k Llama-3-8B cache: 4.29 GB fp16, 2.15 GB fp8, 1.34 GB int4 (scales add 25% to the ideal 0.5 B/value at group 32). Greedy-token agreement is a brittle metric: use KL.
- **Why weight-only helps decode:** decode is bandwidth-bound, so fewer bytes per weight is a direct speedup; it hardly helps prefill (compute-bound) unless activations are quantized too.

---

## 4. Part C — performance (steps 08-13)

### Inference basics (step 08)
| Lever | What it does | In our run |
|---|---|---|
| `model.eval()` + `torch.no_grad()` | no dropout, no autograd graph | - |
| **KV cache** (`use_cache=True`) | reuse past K and V; O(n^2) -> O(n) token-forwards | 100 tokens: **1.6 s vs 4.2 s** (~2.6x, gap widens with length) |
| **Batching** (left padding) | many prompts per forward | 4 prompts: **0.8 s vs 2.1 s** one by one |
| **dtype** | bf16/fp16 half the bytes | 328 MB -> 164 MB (CPU speed not better) |

- **Why K and V but not Q:** only the current token asks the question; past queries are never reused.
- **Prefill vs decode, our run:** 121 prompt tokens in one pass 73 ms (0.6 ms/token); one decode step 17 ms, so prefill is ~30x cheaper per token.
- **Left padding** for batched generation with decoder-only models (they continue from the last position); right padding gives `'Hello, my name is\n\n\n\n...'` with only a warning. Debug habit: batch result must equal the batch-of-1 result. GPT-2 needs `pad_token = eos_token`.
- **bf16 vs fp16:** bf16 has fp32's exponent range (no overflow, less precision); fp16 has more precision, small range (needs loss scaling in training).
- **Hygiene:** warm up, equal work (`min_new_tokens = max_new_tokens`), synchronise GPU before reading the clock.

### Roofline: prefill vs decode (step 09)
- **Prefill** = one parallel pass over the prompt, big matmuls, **compute-bound**, sets TTFT. **Decode** = one token per step, streams all weights (+ KV) for little math, **memory-bound**, sets TPOT/ITL.
- **Ridge point** = peak FLOP/s / bandwidth: A100 **153**, H100 **295** FLOP/byte (our laptop ~21: ~940 GFLOP/s, ~45 GB/s). Below the ridge you are bandwidth-limited.
- **Decode in our run:** B=1 14 ms/step, B=16 22 ms (1.6x) while tokens/s goes 71 -> 713 (10x): batching is almost free until B passes the ridge (~B 40 here); then step time grows and tokens/s flattens (~1,400-1,500).
- **Crossover batch** `B* = ridge x bytes_per_param / 2`: A100 fp16 ~153, int8 ~77, int4 ~38. Quantization shrinks the memory-bound region.
- **KV bytes dominate eventually:** for a Llama-2-7B-style model KV bytes read per step equal the weights at ~25.7k batch x context tokens (8 KV heads: ~103k).
- Prefill efficiency grows with length (6% of peak at 8 tokens to 55-60% from 256 tokens) because small prompts do not fill the chip.

### Benchmarking (step 10)
- **Metrics:** TTFT (queue + prefill), ITL (per-token gap), TPOT (mean ITL after the first token), E2E, p50/p95/p99, throughput, **goodput**. Quote percentiles, not means: in our toy request the mean TPOT was 45 ms but one 120 ms stall hid inside it (p95 ITL 105 ms).
- **In our run (distilgpt2, CPU):** batch 1 to 16: tok/s 65 -> 644 (~10x); per-user speed 64 -> 40 tok/s; TPOT 15.5 -> 23 ms; TTFT 22 -> 84 ms (prefill handles batch x prompt tokens). Run-to-run CV up to 14%: **report the spread**.
- **Goodput:** with TTFT <= 400 ms and TPOT <= 22 ms, goodput equalled throughput up to batch 4 but at batch 16 only 129 of 644 tok/s met the SLO. Throughput can climb while useful throughput collapses.
- Always report threads/hardware; a manual KV loop matches `generate` token for token (engines write their own loop for lower overhead).

### Paged KV cache (step 11)
- **Problems:** *internal* fragmentation (reserved but unused: reserve 2048, use ~300), *external* fragmentation (free holes too small). The vLLM paper measured existing systems using 20-40% of KV memory for real tokens.
- **PagedAttention:** fixed 16-token blocks, allocated on demand, **block table** per request (logical to physical, like an OS page table; token t in block `table[t // 16]` at offset `t % 16`). The kernel gathers K/V through the table. Waste = last block's tail only (< 16 slots per request).
- **In our simulation (8 GB = 16,384 slots, 300 requests, mean length 306):** contiguous 8 requests (85% waste) vs paged **53** (2.4% waste) = **6.6x**; dynamic run: memory used 13.8% vs 80.3%, ~**2.9x** throughput (2268 vs 6494 steps). Block size trade-off: small = more table overhead, large = waste returns (default 16). `MAX_LEN=4096` hurts contiguous, barely moves paged.
- **Failure mode:** a `free()` that does not return blocks drains the pool until nothing is admitted.
- **Enables:** prefix sharing (ref counts), copy-on-write (parallel sampling/beam), preemption (swap to CPU or recompute).

### Scheduling and caching (step 12)
- **Static batching:** a batch runs until its slowest member ends. **Continuous (iteration-level, in-flight) batching:** after every decode step finished sequences leave and waiting ones join. By hand (2/8/2/2-token requests, 2 slots): static finishes at [8,8,10,10], continuous [2,8,4,6] (mean latency 9 vs 5).
- **In our simulation:** light load all schedulers reach the same throughput but static TTFT 4.0 s vs continuous 0.06 s; heavy load static/dynamic saturate ~165 tok/s (mean latency 36 s, p95 74 s) while continuous holds 269 tok/s (2.1 s mean, 5.5 s p95).
- **Chunked prefill:** split long prompts so decode steps are not starved: worst inter-token gap 500 -> 64 ms for a higher mean TTFT (0.17 -> 0.27 s).
- **Automatic prefix caching:** hash each full block with a **chained hash** `h = hash((h_prev, block_tokens))` so a block only hits if everything before it matches; only full blocks are cacheable (50 shared tokens = 3 blocks = 48). **RadixAttention** (SGLang) uses a radix tree and matches token-exactly (500 vs 496 in our run). LRU eviction: a cache smaller than the working set thrashes (0% hits at 16 blocks; 63% at 128 blocks, 2.6x faster TTFT; 85% at 4,096, 4.0x). Real reuse check: prefilling only the 7 unshared tokens was ~3x faster than all 154 with matching logits.
- **Disaggregated prefill/decode:** run the two phases on separate GPU pools (different bottlenecks, no interference), at the cost of moving the KV cache.

### Speculative decoding (step 13)
- A cheap **draft** proposes k tokens; the **target** verifies all in one pass and keeps the longest correct prefix plus one token. Greedy speculative output is **identical** to target-only (the correctness test). Sampling uses accept with prob `min(1, p/q)` and resample from `max(p - q, 0)`: exact target distribution.
- **Expected tokens per pass** `(1 - a^(k+1)) / (1 - a)`: a=0.8, k=4 -> 3.36 (Monte Carlo 3.364 vs 3.362). Best k: (a, c) = (0.8, 0.05) -> k=8, 3.09x; (0.8, 0.3) -> k=3, 1.55x; (0.9, 0.02) -> k=19, 6.37x. A costly draft shortens k; high acceptance lengthens it.
- **In our run:** acceptance 0.75 / 0.62 / 0.47 and tokens per pass 2.4 / 3.3 / 4.3 for gamma 2 / 4 / 8 (target passes 53, 39, 30 vs 128), yet wall-clock gain only 1.11x (gamma 4, gpt2 target with distilgpt2 draft, no KV cache) because the draft cost 65% of a target step (c = 0.65). The draft must be much cheaper than the target.
- Variants: **prompt lookup / n-gram** (no draft model), **Medusa / EAGLE-3 / MTP** (extra heads that predict several tokens). Best when decode is memory-bound and latency matters; gains shrink at large batch.

---

## 5. Part D — engines, parallelism, cost, report (steps 14-17)

### Parallelism and offloading (step 14)
| Scheme | Splits | Communication | Where |
|---|---|---|---|
| **Data parallel (DP)** | requests / batch (full model per replica) | none at inference | scale throughput |
| **Tensor parallel (TP)** | each weight matrix (column then row split, Megatron) | 2 all-reduces per layer per token | **inside a node** (NVLink) |
| **Pipeline parallel (PP)** | consecutive layers | activations between stages; bubble `(p-1)/(m+p-1)` | across nodes |
| **Expert parallel (EP)** | MoE experts | all-to-all token routing; load imbalance stalls the busiest device | MoE models |
| **Sequence / context parallel** | the sequence dimension | attention K/V exchange | very long context |
| **ZeRO / FSDP** | optimizer (1), + grads (2), + params (3) | all-gather / reduce-scatter | training |
| **Offloading** | layers to CPU RAM or disk | PCIe / disk reads (slow) | when nothing else fits |

- **Megatron MLP:** first matrix split by **columns** (GELU stays local), second by **rows** (one all-reduce at the end). Splitting the first by rows would need a sum before GELU: our "wrong" variant was off by tens, not 1e-14.
- **TP comms in our estimate** (80 layers, hidden 8192, TP=8): 160 all-reduces of 16 KB, ~4.6 MB per token per GPU: 10 us of bandwidth over NVLink vs 382 us over 100 GbE, plus 160 latencies, so keep TP inside a node. TP degree must divide heads/hidden (N=5 failed).
- **Bubble:** p=8 needs m >= 63 micro-batches for < 10%; in our simulation 4 stages needed ~16 micro-batches to drop below 16%.
- **ZeRO example:** 13B on 16 GPUs: ZeRO-0 208 GB, -1 61.8, -2 37.4, **-3 13.0 GB** per GPU (only stage 3 fits 32 GB).
- Offloading in our run: 5 modules on CPU, 6 on disk, logits unchanged (works, slow).

### Serving engines (step 15)
| Engine | Strength | Pick it when |
|---|---|---|
| **vLLM** | PagedAttention + continuous batching + prefix caching + OpenAI-compatible server | default general GPU server, many users |
| **SGLang** | RadixAttention prefix reuse, strong structured/grammar decoding | shared prompts, agents, JSON/grammar output |
| **TensorRT-LLM** | NVIDIA-specific compiled kernels, FP8/FP4 | NVIDIA-only fleet, cost per token is the KPI, accepts build complexity |
| **llama.cpp / Ollama** | GGUF, CPU/Metal/CUDA, one-command setup | laptop, edge, single user |
| **HF Transformers / TGI** | simplest, widest model coverage | learning, experiments, prototypes |

- vLLM knobs: `gpu_memory_utilization` (0.9: memory left after weights becomes KV blocks), `max_model_len` (bounds per-sequence cache). Existing OpenAI clients just change the base URL. The decision depends on **workload shape**, not only hardware (our helper flipped vLLM -> SGLang when structured output was required).
- Say vLLM paper numbers as "up to ~24x vs HF `generate`, 2-4x vs earlier servers", not as a promise for your workload.

### Cost per token (step 16)
- **Formula** at $3.50/h: 500 tok/s at 100% busy = $1.94/1M; 2,000 tok/s = **$0.49/1M**; 8,000 tok/s = $0.12/1M. At 30% busy multiply by 1/0.3 (2,000 tok/s -> $1.62; 20% busy = 5x).
- **Output tokens cost more than input tokens** (our laptop: ~34x, since decode is tens of tok/s and prefill thousands) and APIs price output several times input for the same reason. **Batching** made a decoded token ~5x cheaper (batch 1 -> 8) because the rented hardware and weight reads are shared.
- **Estimate (8B, H100-class, roofline, 50% efficiency; not a measurement):** int4 weights b=1 $2.48 vs fp16 $9.44 per 1M output tokens (~4x), but b=64 only $0.19 vs $0.30 (~1.6x): at large batch KV and compute dominate. fp8 KV b=64: $0.30 -> $0.22.
- **Prefix caching:** an 83% hit rate cut a 3000-in/300-out request from $185 to $106 per 1M requests (~40%).
- **Break-even:** one always-on H100 ($2,555/month) beats a $1.60/1M blended API above ~1.6B tokens/month in our example; below that the API wins on cost (no idle time, ops or on-call). Two $3/h GPUs vs a $0.96 blended API: 4,562M tokens/month, 6,844M with 50% redundancy.

### Capstone report (step 17)
- Methodology beats numbers: one subprocess per config (peak RSS only grows within a process), warm-up, repeats, a **quality gate** (perplexity) before "fastest wins". In our CPU report: KV cache on beat off by ~2x tok/s at batch 1 and ~3x at batch 4; batch 4 gave ~2.4x system tok/s for ~1.6x ITL; fake-int8 perplexity 42.53 -> 42.62 (kept quality) but no speed gain (needs real int8 kernels); bf16 was much slower on this CPU (no native bf16 math) and slightly worse (42.84).

---

## 6. Fundamentals vs current practice vs emerging vs outdated

| Class | Items |
|---|---|
| **Fundamentals (stable, derive them)** | KV bytes/token formula; weights + KV + overhead; prefill compute-bound vs decode memory-bound; roofline and ridge; arithmetic intensity; affine quantization, scales, bpw; latency vs throughput vs goodput; fragmentation; TP/PP/DP communication patterns; utilisation-driven $/token |
| **Current industry practice** | safetensors checkpoints; GGUF k-quants for local; weight-only int4 (AWQ/GPTQ-style) and FP8 weights+activations on recent NVIDIA GPUs; fp8/int8 KV cache; GQA models; vLLM / SGLang / TensorRT-LLM with paged KV, continuous batching, chunked prefill, automatic prefix caching; OpenAI-compatible endpoints; TP inside a node; TTFT/ITL/p99 plus goodput against an SLO |
| **Emerging (verify before asserting)** | FP4 formats (MXFP4, NVFP4) on newest hardware; speculative decoding with EAGLE-3 / MTP heads in production engines; disaggregated prefill/decode; KV offload to CPU/SSD and cache-aware routing; context parallelism for very long context; per-channel/KIVI-style low-bit KV |
| **Outdated or legacy** | pickle `.bin` checkpoints from untrusted sources; static batching; contiguous max-length KV reservation; Q4_0/Q5_0 as the default GGUF choice (k-quants/i-quants beat them); reporting only mean latency or only raw throughput; fp16 training without loss scaling |

---

## 7. Numbers to remember (with our run)

| Fact | Value |
|---|---|
| 7B weights | 14 GB fp16, 7 GB int8, 3.5 GB int4; 70B fp16 = 140 GB |
| int4 g128 / g64 / g32 bpw | 4.125 / 4.25 / 4.5 |
| Q8_0 / Q4_0 / Q4_K_M bpw | 8.5 / 4.5 / ~4.85 |
| KV per token | 0.5 MiB (Llama-2-7B MHA), 128 KiB (Llama-3-8B GQA fp16), 64 KiB fp8 |
| Ridge A100 / H100 | 153 / 295 FLOP/byte |
| Full AdamW | 16 B/param (7B ~ 107-112 GB) |
| KV cache on vs off (our run, 100 tokens) | 1.6 s vs 4.2 s |
| Batching 4 prompts (our run) | 0.8 s vs 2.1 s |
| PagedAttention sim | 8 vs 53 requests in 8 GB (6.6x), 2.9x throughput |
| Prefix cache working set | hit rate 0% -> 63% -> 85% as capacity passes the working set |
| Speculative | a=0.8, k=4 -> 3.36 tokens/pass |
| $/1M tokens | 2,000 tok/s on $3.50/h = $0.49 (busy), 5x at 20% utilisation |

---

## 8. Rapid-fire Q&A

**Q: Why cache K and V but not Q?** A: Each step needs only the current token's query; past queries are never reused, but all past keys and values are.

**Q: Complexity with and without a KV cache?** A: Without it each step recomputes the prefix: O(n^2) token-forwards over n tokens. With it each step processes one token (attention still reads all t cached tokens). In our run 4.2 s vs 1.6 s for 100 tokens.

**Q: Why is the first token slow and the rest fast?** A: Prefill processes the whole prompt in one compute-bound pass (TTFT); decode then streams the weights once per token (memory-bound, ITL).

**Q: Why is batching almost free during decode?** A: Decode is bandwidth-bound: the weights read once per step serve every sequence, so batch 16 cost 1.6x the step time of batch 1 for 10x the tokens/s in our run, until intensity passes the ridge.

**Q: Why left padding?** A: Decoder-only models continue from the last position; right padding puts pad tokens there and gives junk with only a warning.

**Q: Memory to serve a 7B model, 4k context, batch 8?** A: ~13.4 GB fp16 weights + 17.2 GB KV (MHA) + ~10% = ~34 GB; int4 weights still ~23 GB because KV dominates.

**Q: How do you shrink the KV cache?** A: GQA/MQA (fewer KV heads), KV quantization (fp8/int8/int4), sliding window, MLA, and PagedAttention to stop wasting it.

**Q: What does 'bits per weight' include?** A: Scales and zero-points: int4 g128 with an fp16 scale is 4.125 bpw, not 4.

**Q: Why not just use a smaller group size?** A: Each halving of the group adds scale overhead (4.125 -> 4.25 -> 4.5 bpw) for diminishing quality gain; 128 is the usual compromise.

**Q: What are activation outliers and why do they break W8A8?** A: A few channels are 10-100x larger and set the per-tensor scale, crushing the rest (23.7 dB vs 37.0 dB per-token in our run). Fixes: per-token scales, SmoothQuant, LLM.int8-style mixed precision.

**Q: How does SmoothQuant work?** A: It divides each activation channel by s and multiplies the matching weight column by s, leaving the product unchanged but moving difficulty from activations into weights.

**Q: GPTQ vs AWQ vs RTN?** A: RTN rounds independently. AWQ scales salient channels found from activations before rounding. GPTQ rounds column by column using the layer Hessian and pushes each error onto not-yet-quantized columns: best at 3 bits (perplexity 29.0 vs RTN 42.2 in our run), most expensive.

**Q: Does weight quantization speed up prefill?** A: Not much; prefill is compute-bound. Weight-only helps decode (fewer bytes streamed); W8A8/FP8 can speed both.

**Q: Why per-channel scales for K and per-token for V?** A: K has a few channels that are large in every token; V varies by token with similar channels (KIVI).

**Q: Is MXFP4 or NVFP4 better?** A: NVFP4 uses smaller blocks (16) and a finer scale (our run 20.44 dB vs 18.78 dB for MXFP4) at +0.25 bpw; both need hardware support to run fast.

**Q: Why safetensors over pickle?** A: Pickle can execute code on load; safetensors is JSON plus raw bytes, safe, and memory-mappable for lazy or fast loading.

**Q: What is GGUF?** A: llama.cpp's single-file format: metadata, tokenizer and block-quantized tensors, aligned for mmap. Q4_K_M is the common size/quality pick.

**Q: Latency vs throughput vs goodput?** A: Latency is per request (TTFT + TPOT x n), throughput is system tokens/s, goodput counts only tokens from requests that meet the SLO. Throughput can rise while goodput collapses (644 vs 129 tok/s at batch 16 in our run).

**Q: What are TTFT, ITL and TPOT?** A: TTFT = time to first token (queue + prefill). ITL = gap between tokens. TPOT = mean time per output token after the first. Quote p95/p99, not means.

**Q: What problem does PagedAttention solve?** A: KV waste from contiguous max-length reservations: internal fragmentation (reserved unused) and external (holes). Fixed 16-token blocks plus a block table cut waste to the last block's tail; 6.6x more requests in our simulation.

**Q: What is continuous batching?** A: Scheduling every decode step: finished sequences leave and queued ones join, so short requests do not wait for long ones (mean TTFT 0.06 s vs 4.0 s for static in our light-load run).

**Q: How does automatic prefix caching work?** A: Hash each full KV block with a chained hash (so position and history match), map equal hashes to the same physical block with ref counts, and evict with LRU. Only full blocks are cached.

**Q: Chunked prefill, why?** A: A long prompt would block every decode step for its whole prefill; chunking interleaves them: worst inter-token gap 500 -> 64 ms for slightly higher TTFT.

**Q: Why does speculative decoding preserve the output?** A: The target verifies every draft token; greedy keeps only matches, sampling uses accept `min(1, p/q)` and resamples from the residual, giving exactly the target distribution.

**Q: When does speculative decoding not help?** A: When the draft is not cheap enough (c = 0.65 gave only 1.11x), acceptance is low, or at large batch where the target is already compute-bound.

**Q: Why TP inside a node and PP across nodes?** A: TP needs 2 all-reduces per layer per token (latency-sensitive, wants NVLink); PP sends only activations between stages but pays a bubble `(p-1)/(m+p-1)`.

**Q: Why does Megatron split column then row?** A: Column split keeps the non-linearity local; the row split needs one all-reduce at the end. Row-splitting first would require a sum before the activation function.

**Q: What does ZeRO stage 3 do?** A: Shards parameters, gradients and optimizer states across GPUs (13B on 16 GPUs: 208 GB per GPU at stage 0 vs 13 GB at stage 3).

**Q: vLLM vs SGLang vs TensorRT-LLM vs llama.cpp?** A: vLLM general default; SGLang for shared prefixes and structured output; TensorRT-LLM for NVIDIA-only cost-optimised fleets; llama.cpp/Ollama for local GGUF.

**Q: When would you not use vLLM?** A: Local or CPU/Mac use (llama.cpp/Ollama), unsupported model or quant formats, or a single-user latency-bound setting where its throughput advantage is irrelevant.

**Q: What does `gpu_memory_utilization=0.9` mean?** A: vLLM reserves 90% of GPU memory; after the weights, the rest becomes the KV block pool. Higher means more concurrency, less headroom (start-up OOM risk).

**Q: What does a token cost?** A: $/1M = GPU $/h / (tokens/s x 3600 x utilisation) x 1e6: 2,000 tok/s on $3.50/h is $0.49 at full load, 5x at 20% utilisation. Output costs more than input since decode is far slower than prefill.

**Q: API or self-host?** A: Compute break-even = fixed monthly GPU cost / blended API price; below it (about 1.6B tokens/month for one H100 in our example) the API wins, and add redundancy and ops to the self-host side.

**Q: How do you benchmark fairly?** A: Warm up, fix output length, synchronise the GPU, sweep batch/concurrency, repeat runs and report spread, one process per configuration, report p50/p95/p99 TTFT/ITL and goodput, and gate on quality.

**Q: Memory to full fine-tune 7B?** A: ~16 B/param ~ 107-112 GB plus activations; LoRA ~14 GB, QLoRA ~4 GB, or shard with ZeRO/FSDP.

---

## 9. Confusable pairs

| Pair | Difference in one line |
|---|---|
| Prefill vs decode | all prompt tokens in parallel (compute-bound, TTFT) vs one token per step (memory-bound, ITL) |
| TTFT vs TPOT vs ITL vs E2E | first token vs mean gap after it vs individual gap vs whole request |
| Throughput vs goodput | all tokens/s vs tokens/s that met the SLO |
| Latency vs throughput | one user's wait vs system tokens/s; batching trades the first for the second |
| Static vs continuous batching | batch ends with its slowest member vs membership changes every step |
| Internal vs external fragmentation | reserved-but-unused inside an allocation vs free holes too small to use |
| Block table vs page table | same idea: logical block to physical block map (KV vs virtual memory) |
| Block hash vs radix tree prefix cache | full-block granularity (496 of 500) vs token-exact (500) |
| Chunked prefill vs disaggregated prefill/decode | interleave phases on one GPU vs separate GPU pools |
| MHA vs GQA vs MQA | KV heads = query heads vs shared by groups vs a single one |
| Weight-only (W4A16) vs W8A8 | fewer bytes streamed (decode) vs integer math on both (also prefill) |
| PTQ vs QAT | quantize after training (cheap) vs train with fake quantization and STE |
| RTN vs GPTQ vs AWQ | round vs Hessian error compensation vs activation-aware scaling |
| Symmetric vs asymmetric | zero-point fixed at 0 vs free zero-point for one-sided data |
| Per-tensor vs per-channel vs per-group | one scale vs one per row/channel vs one per 32-128 weights |
| Dynamic vs static activation scales | computed at runtime per token vs fixed from calibration |
| MXFP4 vs NVFP4 | block 32 with power-of-two scale vs block 16 with finer fp8 scale |
| GPTQ/AWQ (file formats) vs GGUF k-quants | GPU-oriented quantization methods vs llama.cpp block formats |
| safetensors vs pickle vs GGUF | safe tensors-only vs code-capable vs llama.cpp single file |
| TP vs PP vs DP vs EP | split matrices vs split layers vs replicate model vs split experts |
| ZeRO-1/2/3 | shard optimizer / + gradients / + parameters |
| Speculative vs Medusa/EAGLE | separate small draft model vs extra decoding heads on the target |
| vLLM vs SGLang | paged KV + continuous batching default vs radix prefix reuse + structured decoding focus |
| bf16 vs fp16 | fp32's range, less precision vs more precision, small range |
| GB vs GiB | 1e9 vs 2^30 bytes (~7% apart) |
