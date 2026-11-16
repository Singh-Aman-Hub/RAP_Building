# Module 21 — LLM Inference Engineering

How a trained model becomes a fast, cheap, reliable service: how checkpoints are stored and loaded, how quantization shrinks them, how memory and latency are estimated and measured, how batching, paged memory, prefix caching and speculation raise throughput, how a model is split across devices, which serving engine to pick, and what a token really costs. Seventeen steps, all runnable on a laptop CPU (the GPU-only engines are covered as references).

> **Restructure note (Nov 2026).** Steps **03, 08, 10, 11, 15** moved here from module 06 (old steps 08, 07, 09, 11, 10), and step **13** (speculative decoding) moved here from module 16 (old step 09). Links from other modules were updated; the old module-06 serving steps no longer exist.

## The story of this module

Module [06](../06_llm_training_and_lora/README.md) taught you to train and adapt a model. Module [16](../16_modern_llm_architecture/README.md) showed what a modern decoder looks like inside. Now someone asks you to *run* it for real users.
- **Pain point:** `model.generate()` works on your laptop, and then the questions arrive. Will the 70B model fit on one GPU? Why is the first token slow but the next ones fast? Why did throughput triple when we batched, and why did one user's latency get worse? What is GGUF and why do people send each other `Q4_K_M` files? Should we use vLLM, SGLang, Ollama or TensorRT-LLM? What will a million tokens cost us, and is the API cheaper? Without a mental model these are guesses, and each wrong guess costs GPU-hours.
- **Spark:** inference is a small number of levers acting on three resources. *Memory* (weights plus KV cache) decides what fits; *memory bandwidth* decides how fast decode runs; *compute* decides how fast prefill runs. Every technique (quantization, batching, paging, caching, speculation, parallelism) pushes on one of these, and each can be measured or computed on paper.
- **What you build:** a safetensors parser and a mmap-vs-eager loading experiment; a GGUF writer and Q8_0/Q4_0 block quantizer; the canonical weights-plus-KV memory calculator; every quantization scheme from scratch (int8/int4/FP8/NF4/FP4 blocks), activation quantization with SmoothQuant and calibration, GPTQ and AWQ on real distilgpt2 layers, KV-cache quantization on a real cache; HF generation with KV cache and batching; a measured roofline of your laptop; a TTFT/ITL/p99 benchmark harness (`benchlib.py`); a paged-allocator simulation; a continuous-batching scheduler with chunked prefill and prefix caching; real speculative decoding with a provably identical output; NumPy proofs of tensor, pipeline and expert parallelism; an engine decision helper; a cost-per-token model; and a capstone report.
- **Cliffhanger:** you can now serve and price a model. The system *around* it (API, architecture, containers, a full product) comes next: modules [09](../09_api_development/README.md), [10](../10_ml_system_architecture/README.md), [11](../11_docker/README.md) and [13](../13_capstone_project/README.md).

**Reading position:** after [04_llm_fundamentals](../04_llm_fundamentals/), [05_llm_parameters](../05_llm_parameters/) (including [05/05 quantization intro](../05_llm_parameters/05_quantization/)) and [16_modern_llm_architecture](../16_modern_llm_architecture/); before [06_llm_training_and_lora](../06_llm_training_and_lora/README.md) (recommended order: ...04 -> 05 -> 16 -> **21** -> 06 -> 17 -> ...).

**Short on time?** Read [INTERVIEW_GIST.md](INTERVIEW_GIST.md). Practice: [exercises.md](exercises.md), [solutions/solutions.md](solutions/solutions.md).

## 1. Why this matters

"How would you deploy this model?" is the most common senior-level LLM interview question, and every good answer is arithmetic plus vocabulary: *"8B in fp16 is 16 GB of weights, the KV cache is 128 KiB per token with GQA, decode is memory-bound so int4 weights help at small batch, continuous batching raises GPU use, prefix caching cuts prefill for the shared system prompt, and at 2,000 tokens/s on a $3.5/hour GPU that is about $0.50 per million output tokens."* This module gives you each of those sentences with a script that produced the numbers.

## 2. Where this fits

**Builds on**
- [04_llm_fundamentals](../04_llm_fundamentals/) ([generation loop](../04_llm_fundamentals/08_generation_loop/), logits, tokens): what is being sped up.
- [05_llm_parameters](../05_llm_parameters/) ([05 quantization intro](../05_llm_parameters/05_quantization/), [context window](../05_llm_parameters/04_max_new_tokens_and_context_window/)): symmetric int8 and group-wise scaling, which step 04 generalises.
- [16_modern_llm_architecture](../16_modern_llm_architecture/) ([04 attention variants and KV cache](../16_modern_llm_architecture/04_attention_variants_and_kv_cache/), [05 FlashAttention](../16_modern_llm_architecture/05_flash_attention_online_softmax/), [06 MoE](../16_modern_llm_architecture/06_mixture_of_experts/)): GQA/MLA shrink the cache that steps 03, 07, 11 size; FlashAttention is the kernel engines use; MoE is why expert parallelism exists.
- [10_ml_system_architecture/03_serialization](../10_ml_system_architecture/03_serialization/): pickle vs safer formats, generalised in step 01.

**Leads to**
- [06_llm_training_and_lora](../06_llm_training_and_lora/README.md) and [17_finetuning_alignment_and_benchmarks](../17_finetuning_alignment_and_benchmarks/README.md): QLoRA uses step 04's NF4; merged LoRA weights get quantized and served here.
- [09_api_development](../09_api_development/README.md): the API layer in front of an OpenAI-compatible engine.
- [10_ml_system_architecture](../10_ml_system_architecture/README.md): capacity planning uses steps 03, 10, 16.
- [11_docker](../11_docker/README.md): packaging an engine with GPU drivers.
- [13_capstone_project](../13_capstone_project/README.md): the full product that calls a served model.
- [20_llm_system_lld](../20_llm_system_lld/): rate limits, queues and budgets rest on the latency and cost models here.

## 3. Concept map

```
                      checkpoint on disk                                   what must fit / what is slow
 01 formats + loading ---> weights in memory ----------+                    --------------------------
   safetensors, mmap       02 GGUF, k-quants           |                    weights bytes        (03)
   pickle risk             (llama.cpp world)           v                    KV cache bytes       (03, 07, 11)
                                              03 memory calculator <------- activations, overhead
                                                       |
            shrink weights                             |                  shrink / speed up
   04 quant math -> 05 activations+calibration         |                  ---------------------
        -> 06 GPTQ / AWQ -> 07 KV-cache quant ---------+                  quantize (04-07)
                                                       |                  batch + cache (08, 12)
                       measure                         v                  page memory (11)
   09 prefill = compute-bound / decode = memory-bound (roofline)          speculate (13)
   10 TTFT, ITL, p99, throughput, goodput (benchlib)                      split devices (14)
                                                       |
        use memory well -> 11 paging -> 12 continuous batching + prefix caching -> 13 speculation
                                                       |
                 more than one device: 14 TP / PP / EP / offloading
                                                       |
            choose the tool: 15 vLLM | SGLang | TensorRT-LLM | llama.cpp | Ollama | HF
                                                       |
            price it: 16 $/1M tokens, utilisation, break-even   ->   17 capstone report
```

## 4. Prerequisites

- Python and NumPy comfort ([00_python_foundations](../00_python_foundations/README.md), [03_ml_foundations](../03_ml_foundations/README.md)).
- What a transformer layer and attention are ([04_llm_fundamentals](../04_llm_fundamentals/), [15_neural_networks](../15_neural_networks/)).
- Tokens, logits, sampling ([05_llm_parameters](../05_llm_parameters/)); KV cache and GQA from [16/04](../16_modern_llm_architecture/04_attention_variants_and_kv_cache/).
- Offline cached models: `distilgpt2`, `gpt2`, `sshleifer/tiny-gpt2` (no downloads; run with `HF_HUB_OFFLINE=1`). Qwen2.5-0.5B has only its config cached on this laptop, so steps that need it use random weights with the right shapes (timing depends on shapes, not values).

## 5. Study path

Run from the repo root with `HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/<step>/main.py`. "CPU" = runs here; "ref" = reference code or notes that need a GPU or binaries not installed.

| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| **Part A: loading and storage** | | | | |
| 01 | [01_checkpoint_formats_and_loading](01_checkpoint_formats_and_loading/WALKTHROUGH.md) | pickle risk, safetensors layout parsed by hand, mmap vs eager, shards, HF cache layout | CPU | ~30-60 s |
| 02 | [02_gguf_and_llamacpp_formats](02_gguf_and_llamacpp_formats/WALKTHROUGH.md) | write/read GGUF, Q8_0/Q4_0 blocks, k-quant sizes | CPU | <2 s |
| 03 | [03_gpu_memory_calculator](03_gpu_memory_calculator/WALKTHROUGH.md) | **canonical** weights + KV + training memory formula | CPU | ~3 s |
| **Part B: quantization** | | | | |
| 04 | [04_quantization_math](04_quantization_math/WALKTHROUGH.md) | affine int8/int4, granularity, FP8, NF4, block FP4, bits per weight | CPU | ~10 s |
| 05 | [05_activation_quant_and_calibration](05_activation_quant_and_calibration/WALKTHROUGH.md) | activation outliers, calibration, SmoothQuant, LLM.int8, PTQ vs QAT | CPU | ~40 s |
| 06 | [06_gptq_awq_from_scratch](06_gptq_awq_from_scratch/WALKTHROUGH.md) | RTN vs simplified GPTQ vs AWQ on distilgpt2 | CPU | 1-2 min |
| 07 | [07_kv_cache_quantization](07_kv_cache_quantization/WALKTHROUGH.md) | per-channel K vs per-token V, KIVI idea, real cache | CPU | ~30-60 s |
| **Part C: performance** | | | | |
| 08 | [08_hf_inference_batching_kvcache](08_hf_inference_batching_kvcache/WALKTHROUGH.md) | `no_grad`, dtype, padding, KV cache measured | CPU | ~30 s |
| 09 | [09_prefill_decode_roofline](09_prefill_decode_roofline/WALKTHROUGH.md) | compute-bound prefill vs memory-bound decode, a measured roofline | CPU | ~1.5 min |
| 10 | [10_latency_throughput_benchmark](10_latency_throughput_benchmark/WALKTHROUGH.md) | TTFT, ITL/TPOT, p50/p95/p99, goodput; `benchlib.py` | CPU | ~1 min |
| 11 | [11_paged_attention_fragmentation](11_paged_attention_fragmentation/WALKTHROUGH.md) | contiguous vs paged KV allocation simulation | CPU (sim) | seconds |
| 12 | [12_continuous_batching_prefix_caching](12_continuous_batching_prefix_caching/WALKTHROUGH.md) | static/dynamic/continuous batching, chunked prefill, prefix cache | CPU | ~15 s |
| 13 | [13_speculative_decoding](13_speculative_decoding/WALKTHROUGH.md) | draft + target, acceptance rate, provably identical output | CPU | ~1 min |
| **Part D: engines and cost** | | | | |
| 14 | [14_parallelism_and_offloading](14_parallelism_and_offloading/WALKTHROUGH.md) | TP/PP/EP proven in NumPy, bubble, comm volume, ZeRO, real `device_map` offload | CPU | ~10 s |
| 15 | [15_serving_engines_landscape](15_serving_engines_landscape/WALKTHROUGH.md) | HF vs llama.cpp vs Ollama vs vLLM vs SGLang vs TensorRT-LLM vs TGI; decision helper | CPU helper + **ref** (vLLM, SGLang, TensorRT-LLM, llama.cpp) | <1 s |
| 16 | [16_cost_per_token](16_cost_per_token/WALKTHROUGH.md) | $/1M tokens, utilisation, input vs output, break-even, quantization/caching effects | CPU | ~25 s |
| 17 | [17_inference_report_project](17_inference_report_project/WALKTHROUGH.md) | capstone: benchmark matrix + generated report ([project README](17_inference_report_project/README.md)) | CPU | 1.5-3 min |

Shared helpers (not steps): [`benchlib.py`](benchlib.py) (timing: TTFT/ITL/percentiles/RSS) and [`quantlib.py`](quantlib.py) (fake quantization, perplexity).

## 6. The concepts, one by one

**Checkpoint formats (01, 02).** A checkpoint is named tensors plus metadata. Pickle (`.bin`, `.pt`) can execute code on load; **safetensors** is an 8-byte length, a JSON header (`dtype`, `shape`, `data_offsets`) and raw bytes, loadable by memory-mapping with no code execution. Big models are sharded with an `index.json` mapping tensor names to files. **GGUF** is llama.cpp's single-file format: magic, version, typed metadata (including the tokenizer), tensor infos, aligned data, with quantized blocks such as Q8_0 (32 weights share one scale) and k-quants (super-blocks of 256).

**Memory arithmetic (03).** Serving memory = weights (params x bytes) + KV cache (`2 x layers x kv_heads x head_dim x tokens x batch x bytes`) + about 10% overhead; training with AdamW needs about 16 bytes per parameter. Llama-3-8B with GQA costs 128 KiB of cache per token in fp16. The KV cache, not the weights, often limits concurrency.

**Quantization (04 to 07).** Store numbers in fewer bits with a scale: `x_hat = scale x q` (affine adds a zero-point). Granularity (per-tensor, per-channel, per-group) trades accuracy against scale overhead. *Weights* quantize easily (W4A16: 4-bit weights, 16-bit activations); *activations* have outliers that need calibration or SmoothQuant. **GPTQ** minimises layer-output error with second-order information from calibration data; **AWQ** scales the roughly 1% salient channels found from activations. **KV-cache quantization** compresses the other big tensor: keys have channel outliers (scale per channel), values do not (per token). Formats: int8/int4, FP8 (E4M3/E5M2), NF4 (QLoRA), NVFP4/MXFP4 block-scaled FP4 on newer hardware.

**Why decode is slow (08, 09, 10).** Prefill processes the whole prompt in one big matrix multiplication: compute-bound. Decode produces one token per step but must read all weights and the KV cache every time: memory-bound, so tokens/s is about `bandwidth / bytes read per token`. Metrics: **TTFT** (prefill + first token), **ITL/TPOT** (per-token gap), E2E latency, throughput, **goodput** (throughput meeting the SLO), tail percentiles. Measure with warm-up and repeated trials.

**Using memory and the GPU well (11, 12).** Contiguous per-request KV slabs fragment (internal waste from over-reserving, external from holes); **PagedAttention** allocates fixed-size blocks through a block table. **Continuous batching** refills finished slots every step instead of waiting for the slowest request; **chunked prefill** stops a long prompt from stalling decodes; **prefix caching** reuses the KV of shared prompt prefixes (chained block hashes in vLLM, a radix tree in SGLang).

**Speculative decoding (13).** A small draft model proposes k tokens, the large target verifies all of them in one forward pass and accepts the longest correct prefix. With greedy decoding the output is *identical* to the target alone; the gain is fewer target passes. Variants: draft model, n-gram/prompt lookup, Medusa, EAGLE-3, multi-token-prediction heads (MTP).

**Parallelism and offloading (14).** Tensor parallelism cuts every matrix (column-parallel outputs concatenate, row-parallel outputs are summed by an all-reduce; Megatron's column-then-row MLP needs one all-reduce per block), pipeline parallelism cuts layers (bubble `(p-1)/(m+p-1)`), expert parallelism spreads MoE experts (all-to-all), data parallelism replicates. TP inside an NVLink node, PP across nodes. Offloading (`device_map`, `-ngl`) trades speed for capacity.

**Engines (15).** transformers (reference), llama.cpp/Ollama (local, GGUF), vLLM (PagedAttention, continuous batching, prefix caching), SGLang (RadixAttention, structured output), TensorRT-LLM (NVIDIA-tuned), TGI (maintenance mode), MLX/LM Studio (Apple). Almost all expose an OpenAI-compatible API.

**Cost (16).** `$/1M tokens = ($/hour) / (tokens/s x 3600 x utilisation) x 1e6`, with input and output priced separately. Self-hosting is a fixed cost that wins above a break-even volume and only if the GPU stays busy.

## 7. Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| TTFT | time to first token, first-token latency, prefill latency (approximately) | Mostly | Prefill plus the first sampling step; what the user feels as "start-up". |
| ITL | inter-token latency, per-token latency | Yes | Gap between two consecutive output tokens (one value per gap). |
| TPOT | time per output token, decode latency per token | Almost | Mean ITL of a request, TTFT excluded; ITL keeps the distribution (p95/p99). |
| E2E latency | request latency, total latency | Yes | Start to last token = TTFT + (n-1) x TPOT. |
| Throughput | tokens/s, output tok/s, req/s | Depends | System-wide rate; says nothing about one user's speed. |
| Goodput | SLO-attaining throughput | Related | Throughput counting only requests that met TTFT/ITL targets. |
| PTQ | post-training quantization | Yes | Quantize a finished model, usually with a small calibration set. |
| QAT | quantization-aware training | Yes | Train (or fine-tune) with simulated quantization in the loop. |
| W8A8 / W4A16 / W4A8 | weight bits / activation bits | Notation | "W4A16" = 4-bit weights, 16-bit activations (weight-only); W8A8 quantizes both. |
| Weight-only quantization | W4A16, GPTQ/AWQ style | Yes | Memory-saving: dequantize in the kernel, compute in fp16. |
| RTN | round-to-nearest, naive quantization | Yes | Baseline with no calibration. |
| GPTQ / AWQ / HQQ | PTQ algorithms | Different | Second-order error minimisation / activation-aware scaling / calibration-free. |
| NF4 | NormalFloat4, QLoRA data type | Yes | 4-bit levels placed at normal-distribution quantiles. |
| FP8 | E4M3 / E5M2 | Related | 8-bit floats; E4M3 for weights/activations, E5M2 wider range (gradients). |
| NVFP4 / MXFP4 | FP4 block formats | Different | NVFP4: block 16, FP8 scale, Blackwell; MXFP4: OCP standard, block 32, power-of-two scale. |
| KV cache | key-value cache, `past_key_values`, attention cache | Yes | Stored K and V of all earlier tokens so decode does not recompute them. |
| Continuous batching | in-flight batching, iteration-level scheduling, dynamic batching (loosely) | Mostly | Requests join and leave the batch at every step. |
| Static batching | fixed batch, `generate` on a list | Yes | The batch runs until its slowest member finishes. |
| PagedAttention | paged KV cache, block table | Yes | KV stored in fixed-size blocks mapped by a table, like OS pages. |
| Prefix caching | automatic prefix caching (APC), prompt caching, KV reuse | Yes | Reuse KV of an identical prompt prefix. |
| RadixAttention | radix-tree prefix cache (SGLang) | Same goal, other mechanism | Prefix lookup by tree walk at any length. |
| Chunked prefill | chunked context, split-fuse | Yes | Process long prompts in slices mixed with decode. |
| Speculative decoding | assisted generation (HF), draft-and-verify | Yes | Draft proposes, target verifies in one pass. |
| EAGLE / Medusa / MTP | draft heads / multi-token prediction | Variants | Predict several tokens with extra heads instead of a separate draft model. |
| TP / PP / EP / DP | tensor / pipeline / expert / data parallel | Different | Cut matrices / cut layers / spread experts / replicate. |
| ZeRO / FSDP | sharded data parallel | Close | Shard training state across data-parallel ranks. |
| `-ngl` / `device_map` / offloading | layer offloading, CPU offload | Same idea | How many layers live on the fast device. |
| GGUF | GGML Universal Format (successor to GGML/GGJT) | Yes | llama.cpp's single-file quantized model format. |
| k-quants / i-quants | Q4_K_M, Q5_K_S ...; IQ4_XS ... | Related | Block quantization with super-block scales / importance-matrix-based codebooks. |
| safetensors | `.safetensors` | Yes | Header + raw tensors, mmap-friendly, no code execution. |
| OpenAI-compatible API | `/v1/chat/completions`, "drop-in" | Yes | Same JSON shape; swap `base_url`. |
| Cost per token | $/1M tokens, price per Mtok | Yes | Rental price / tokens per hour, per utilisation. |

## 8. Fundamentals vs current practice vs emerging vs outdated

| Class | Items |
|---|---|
| **Fundamentals (will not change)** | KV cache; prefill compute-bound vs decode memory-bound; the memory formula; quantization as scale + integers; TTFT/ITL/throughput trade-off; collective communication (all-reduce, all-gather, all-to-all); cost = price / throughput |
| **Current industry practice (2026)** | safetensors and GGUF; vLLM / SGLang for GPU serving, llama.cpp / Ollama locally; FP8 on Hopper-class GPUs; AWQ/GPTQ 4-bit weights; continuous batching, chunked prefill, automatic prefix caching; fp8 KV cache; TP within a node, PP/EP across; OpenAI-compatible endpoints; llm-compressor to produce quantized checkpoints; speculative decoding in production engines |
| **Emerging** | NVFP4/MXFP4 weights and KV cache; disaggregated prefill/decode with KV transfer; EAGLE-3 and MTP speculation; `transformers serve` with continuous batching; Blackwell-specific kernels |
| **Outdated / maintenance** | pickle checkpoints for sharing; static batching servers; TGI for new projects (maintenance mode); GGML (pre-GGUF) files; hand-rolled `generate` behind FastAPI for real traffic; judging engines by one leaderboard number |

Versions move fast: check each project's release page (as of Nov 2026, vLLM was around v0.33 and SGLang around v0.5.x, medium confidence).

## 9. Common mistakes

- Quoting the weight size and forgetting the KV cache (and that it scales with batch x context).
- Believing int4 weights shrink the KV cache (they do not; see steps 03 and 07).
- Benchmarking once, without warm-up, or reporting only tokens/s without TTFT and tail latency.
- Confusing throughput (system) with latency (user); batching raises one and costs the other.
- Treating fake quantization as a speedup, or skipping the perplexity check after quantizing.
- Using TP across slow links or expecting offloading to be fast.
- Choosing an engine from a blog benchmark rather than your own traffic.
- Pricing input and output tokens the same, or ignoring utilisation and engineer time in break-even.
- Loading untrusted pickle checkpoints.
- Forgetting left padding for decoder-only batched generation.

## 10. Interview Q&A

1. **Will a 70B model fit on one 80 GB GPU?** Weights in fp16 are about 140 GB, so no. At int4 they are about 35 GB, which fits, leaving about 40 GB for KV cache and overhead. Or split with TP=2 at fp16.
2. **How big is the KV cache?** `2 x layers x kv_heads x head_dim x tokens x batch x bytes`. Llama-3-8B (32 layers, 8 KV heads, head_dim 128) is 128 KiB per token in fp16, so one 8k-token sequence is about 1 GiB.
3. **Why is decode memory-bound and prefill compute-bound?** Prefill multiplies a big (prompt x hidden) matrix by weights: lots of FLOPs per byte. Decode multiplies one token vector by the same weights: each weight byte is used once, so speed is set by bandwidth.
4. **TTFT vs ITL vs throughput?** TTFT is time to first token (prefill-dominated), ITL the gap between tokens (decode-dominated), throughput the system-wide tokens per second. Batching raises throughput and raises ITL/TTFT.
5. **What does continuous batching do?** It lets finished requests leave and waiting ones join at every decode step, so the GPU never waits for the slowest request; static batching wastes slots.
6. **What does PagedAttention fix?** KV fragmentation: contiguous per-request slabs waste memory; fixed-size blocks plus a block table allocate on demand and enable sharing prefixes.
7. **Prefix caching vs RadixAttention?** Same goal (reuse the KV of a shared prefix); vLLM hashes full blocks, SGLang walks a radix tree so any shared prefix length is found.
8. **How does quantization help speed?** Decode is bandwidth-bound, so fewer bytes per weight means fewer bytes read per token; compute still runs in fp16 for weight-only schemes after dequantizing in the kernel. Gains are largest at small batch.
9. **GPTQ vs AWQ?** GPTQ quantizes column by column, using the Hessian of the layer's output error from calibration data to compensate error; AWQ scales the salient channels (those with large activations) before rounding. Both are 4-bit weight-only PTQ.
10. **Why are activations harder to quantize than weights?** Activations have large outlier channels that stretch the scale; SmoothQuant migrates the outliers into the weights, LLM.int8 handles outlier columns in higher precision.
11. **Why per-channel for K and per-token for V?** Keys have a few channels with large magnitude (so scale per channel); values vary mostly by token (scale per token): the KIVI observation.
12. **How does speculative decoding keep the output identical?** The target verifies every draft token in one pass and accepts only those it would have produced itself (greedy), or uses rejection sampling to preserve the distribution (sampling). Speed comes from more tokens per target pass.
13. **Explain tensor parallelism.** Column-parallel layers split output columns (concatenate), row-parallel layers split input rows (all-reduce a sum). Megatron's MLP uses column then row so there is one all-reduce per block; attention shards by heads.
14. **TP vs PP?** TP: fine-grained, two all-reduces per layer per token, needs NVLink, so keep it within a node. PP: layers per stage, one activation per boundary, tolerates slower links but has a bubble `(p-1)/(m+p-1)`.
15. **When to offload to CPU or disk?** When it must run but does not fit; expect decode to be bounded by PCIe or disk bandwidth, 10 to 100 times slower. It is not a serving strategy.
16. **vLLM vs llama.cpp vs TensorRT-LLM vs SGLang?** vLLM: general GPU serving; llama.cpp: local/CPU/edge with GGUF; TensorRT-LLM: peak NVIDIA performance with more engineering; SGLang: prefix-heavy and structured workloads. Benchmark your own traffic.
17. **How do you compute cost per million tokens?** GPU $/hour divided by tokens per hour (throughput x 3600 x utilisation), times one million, priced separately for input (prefill) and output (decode).
18. **Self-host or API?** Compute the break-even: monthly GPU cost divided by the API's blended $/token. Above it self-hosting can win, if the GPU stays busy and engineering and redundancy are counted; privacy, latency control or a fine-tuned model can justify it earlier.
19. **How would you benchmark an endpoint?** Fixed prompt/output length distributions, warm-up, repeated trials or an open-loop request generator, report TTFT, ITL percentiles, throughput and goodput against an SLO; compare engines on the same load.
20. **Why safetensors over pickle?** Pickle can execute arbitrary code at load; safetensors is data only, mmap-able with lazy loading.

## 11. Self-check

- [ ] Compute weights and KV cache for a model and context from its config without a calculator app.
- [ ] Explain, with the roofline, why batch size barely changes decode step time at small batch.
- [ ] Define TTFT, ITL, TPOT, E2E, throughput, goodput and say which a chat UI cares about.
- [ ] Say what a safetensors header contains and why mmap makes loading fast.
- [ ] Write the affine quantize/dequantize formulas and explain per-group scales.
- [ ] Explain why SmoothQuant exists and what GPTQ and AWQ each optimise.
- [ ] Draw a block table and explain internal vs external fragmentation.
- [ ] Prove (on paper or in NumPy) that column-then-row splitting equals the unsplit MLP.
- [ ] Pick an engine for three different scenarios and defend it.
- [ ] Compute $/1M tokens and the break-even volume from a price and a measured throughput.

## 12. Files, exercises, solutions

- [INTERVIEW_GIST.md](INTERVIEW_GIST.md): one-page cram sheet.
- [exercises.md](exercises.md) and [solutions/solutions.md](solutions/solutions.md) (runnable `q*.py`, `d*.py` solution scripts in [solutions/](solutions/)).
- Shared code: [benchlib.py](benchlib.py), [quantlib.py](quantlib.py).
- Capstone: [17_inference_report_project/README.md](17_inference_report_project/README.md).
