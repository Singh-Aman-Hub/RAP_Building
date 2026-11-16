# Track B — Inference engineering report (portfolio piece)

**One sentence:** take the benchmark-and-recommendation project of module 21 and turn it from "a script I ran in a course" into "a short report I can put in front of an interviewer", then (optionally) extend it with a real serving engine on a GPU machine.

Back to the capstone: [13 README](../README.md).

## What this track is, and what it is not
- The project itself lives in module 21: [17_inference_report_project](../../21_llm_inference_engineering/17_inference_report_project/README.md) (code, how it measures, its own interview rubric and two-minute story). **This README does not repeat that.** It adds the portfolio layer: what to deliver, how to judge it, how to present the trade-offs, and how to go beyond CPU.
- Prerequisites: the steps of [module 21](../../21_llm_inference_engineering/INTERVIEW_GIST.md) that the report uses: [08 KV cache and batching](../../21_llm_inference_engineering/08_hf_inference_batching_kvcache/WALKTHROUGH.md), [09 prefill/decode roofline](../../21_llm_inference_engineering/09_prefill_decode_roofline/WALKTHROUGH.md), [10 TTFT/ITL benchmark](../../21_llm_inference_engineering/10_latency_throughput_benchmark/WALKTHROUGH.md), [04 quantization math](../../21_llm_inference_engineering/04_quantization_math/WALKTHROUGH.md), [16 cost per token](../../21_llm_inference_engineering/16_cost_per_token/WALKTHROUGH.md). Foundations: [16 modern LLM architecture](../../16_modern_llm_architecture/README.md) (KV cache, attention variants).
- Runs on the laptop (CPU, distilgpt2). Needs no GPU, no download, no paid API.

## Deliverables (what "done" means)
1. **Reproducible run:** `HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/17_inference_report_project/main.py` (use the repo's global-lock workflow for timing runs) produces `inference_report.md` and `.csv`. Keep the exact command and the machine description (CPU model, RAM, thread count) at the top of your report.
2. **One-page report** (rewrite the generated Markdown in your own words): question, setup, results table, 3-4 findings, one recommendation per constraint, caveats. If it does not fit on a page, cut rows, not caveats.
3. **One chart or table you can draw on a whiteboard:** tokens/s vs batch size with KV cache on/off, or the latency-vs-throughput curve. Matplotlib is not installed in this repo; a table, a Markdown bar made of block characters, or hand-drawn is fine.
4. **A "what would change my answer" list** (GPU vs CPU, longer prompts, real int8/int4 kernels, concurrent users).
5. **A 2-minute spoken version** and **a 5-minute version** with one slide-equivalent (the table).

## Rubric (portfolio level; 20 points)
Module 21's README has the interview-answer rubric. This one grades the *artefact*.

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| Question stated | none | "benchmark inference" | a decision ("cheapest config within 2% perplexity at p95 ITL under X ms") |
| Method | one run, no warm-up | warm-up + repeats | repeats, median, spread shown, threads/hardware recorded, one process per config |
| Metrics | tokens/s only | + latency | TTFT, ITL, throughput, peak memory, quality (perplexity), cost, each defined |
| Separation of prefill and decode | not mentioned | mentioned | used to explain why batch and KV cache behave as they do ([09](../../21_llm_inference_engineering/09_prefill_decode_roofline/WALKTHROUGH.md)) |
| Quality guard | none | quality measured | a gate decides what is allowed to win |
| Honest limits | none | listed | each limit tied to what it would change (fake int8, CPU vs GPU, noisy perplexity, example price) |
| Recommendation | "use X" | a pick | a pick per constraint, with a trigger to switch |
| Communication | table dump | readable | a reader can repeat the conclusion after 30 seconds |
| Extension | none | listed | one extension actually run (below) |
| Reproducibility | not possible | commands listed | fresh clone + command reproduces the table shape |

## What to measure (and what each number lets you say)
| Metric | Definition | Say this | Do not say |
|---|---|---|---|
| TTFT (time to first token) | request start to first generated token = queueing + prefill | "prefill is compute-bound; TTFT grows with prompt length and with load" | "TTFT is model speed" |
| ITL (inter-token latency, TPOT) | time between consecutive output tokens | "decode is memory-bandwidth-bound: each step reads all weights and the KV cache" | that ITL is constant under load |
| Throughput (tokens/s, system-wide) | output tokens over wall time across all requests | "batching raises throughput; per-request latency rises" | quote tokens/s without batch size |
| Peak memory (RSS / VRAM) | weights + KV cache + activations + runtime | "KV cache grows with batch x sequence length; quantization shrinks weights" | that int8 saved memory if you only simulated it |
| Quality | perplexity delta vs fp32 (and, on a real run, a task metric) | "quantization is accepted only inside a quality gate" | that a 0.3% perplexity change proves task quality |
| Cost | $/hour divided by tokens/hour, per 1M output tokens ([16](../../21_llm_inference_engineering/16_cost_per_token/WALKTHROUGH.md)) | "cost follows utilisation: an idle GPU is expensive tokens" | an absolute price (the price here is an example) |

Measure at least: **vary one axis at a time** (batch, prompt length, output length, KV cache on/off, dtype) and report the spread over repeats, not a single run.

## How to present trade-offs
Tell it as three tensions, each with one number from your table:
1. **Latency vs throughput:** batching moves you along one curve. Show the batch-1 pick (interactive) and the batch-N pick (offline/cost) as two points on it, not one "best".
2. **Memory vs quality:** quantization buys weights memory (and, with real kernels, bandwidth) at a quality cost. Show the gate and the config it allowed.
3. **Cost vs headroom:** the cheapest config at full utilisation is not the cheapest at 20% utilisation; say what load you assumed.

Template sentence: "For <workload> with <constraint>, I recommend <config>: it gives <TTFT/ITL/throughput> at <memory> within <quality gate>, at about <$/1M> under <assumed utilisation>. I would switch to <config B> if <trigger>, and this result does not transfer to <GPU/long prompts/real kernels> because <reason>."

## Extending with a real engine on a GPU box (not run in this course)
The laptop report simulates quantization and cannot show paged attention, continuous batching or real kernels. On a machine with a GPU (rented hour is enough), re-run the same *questions* against a real engine and keep the same report format so the two reports can be compared:
- **vLLM** (continuous batching, PagedAttention, prefix caching): serve a small model (for example a 0.5-1.5B instruct model), then drive it with its own benchmark script or a simple async client; record TTFT/ITL/throughput at several concurrency levels. Background: [12 continuous batching](../../21_llm_inference_engineering/12_continuous_batching_prefix_caching/WALKTHROUGH.md), [11 paged attention](../../21_llm_inference_engineering/11_paged_attention_fragmentation/WALKTHROUGH.md), [15 engines landscape](../../21_llm_inference_engineering/15_serving_engines_landscape/WALKTHROUGH.md).
- **llama.cpp** (GGUF, CPU/Metal/CUDA, k-quants): run the same prompts at two or three quantization levels (for example Q8 vs Q4 variants) and compare size, speed and perplexity. Background: [02 GGUF and llama.cpp formats](../../21_llm_inference_engineering/02_gguf_and_llamacpp_formats/WALKTHROUGH.md).
- **Quantized weights with real kernels** (GPTQ/AWQ in an engine that supports them): compare against fp16 under your quality gate. Background: [06 GPTQ/AWQ from scratch](../../21_llm_inference_engineering/06_gptq_awq_from_scratch/WALKTHROUGH.md).
- **Add two axes the laptop cannot show:** concurrency (open-loop request arrival, p95 latency vs request rate) and prompt-length sweep (prefill cost). Report goodput (requests meeting your latency target per second), not only peak throughput.
- Keep the honesty rules: state engine versions and flags used, GPU model, and that numbers are from one machine. Version numbers and flags change quickly; check the engine's own documentation rather than copying a command from a blog.

Reference code to adapt, rather than write from scratch: [vllm_reference.py](../../21_llm_inference_engineering/15_serving_engines_landscape/vllm_reference.py) and the other `*_reference` files in [15](../../21_llm_inference_engineering/15_serving_engines_landscape/WALKTHROUGH.md) (labelled "not run in this course").

## Where Track B connects to Track A
Track A's generator is a mock. If you serve a real model for it, this report is the evidence for *which* serving configuration to pick: the OpenAI-compatible switch in [Track A](../track_a_document_qa/README.md) (`DOCQA_LLM_BASE_URL`) is the plug.
