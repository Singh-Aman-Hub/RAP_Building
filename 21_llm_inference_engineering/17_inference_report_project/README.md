# Project — Inference engineering report (capstone of module 21)

**Goal:** turn everything in module 21 into one artefact an interviewer can read in five minutes: a benchmark matrix, a measured table, findings, and a recommendation with caveats.

```
python main.py                       # from this folder, ~1-2 min on a laptop CPU; writes inference_report.md and inference_report.csv
python main.py --model gpt2          # bigger model
python main.py --quick               # fp32 only
PRICE_PER_HOUR=3.5 python main.py    # change the example price
```

Repo-root form (use the global lock in this repo's workflow): `HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/17_inference_report_project/main.py`

## What it measures (one subprocess per configuration, so peak RSS and load time are clean)

| Axis | Values | Which step explains it |
|---|---|---|
| dtype | fp32, bf16 | [01](../01_checkpoint_formats_and_loading/WALKTHROUGH.md), [03](../03_gpu_memory_calculator/WALKTHROUGH.md) |
| weight quantization | none, fake int8 (per-channel round-to-nearest) | [04](../04_quantization_math/WALKTHROUGH.md) |
| batch size | 1, 4 | [10](../10_latency_throughput_benchmark/WALKTHROUGH.md), [12](../12_continuous_batching_prefix_caching/WALKTHROUGH.md) |
| KV cache | on, off | [08](../08_hf_inference_batching_kvcache/WALKTHROUGH.md) |
| metrics | load time, TTFT, ITL, tokens/s, peak RSS, perplexity, example $/1M tokens | [10](../10_latency_throughput_benchmark/WALKTHROUGH.md), [16](../16_cost_per_token/WALKTHROUGH.md) |

The recommendation applies a **quality gate** (perplexity within 2% of the fp32 baseline) and then picks the cheapest configuration at batch 4 (throughput pick) and the lowest TTFT + 16 decode steps at batch 1 (latency pick).

## How to present this in an interview (rubric)

| Criterion | What a strong answer shows |
|---|---|
| **Question and constraint** | "I wanted to know which configuration is cheapest at acceptable quality and latency" (not "I ran many benchmarks"). |
| **Method** | Warm-up, median of repeated trials, one subprocess per config for clean RSS, fixed prompt/new-token lengths, threads reported. |
| **Metrics vocabulary** | TTFT, ITL (= TPOT), tokens/s, perplexity, $/1M tokens; knows decode is memory-bound and prefill compute-bound. |
| **Findings, not numbers** | The KV cache is the biggest single win; batching raises throughput but raises per-step latency; bf16 is hardware-dependent on CPU. |
| **Honesty about limits** | int8 is *fake* quantization (measures quality, not speed); tiny model on CPU is not a GPU server; perplexity on a few hundred tokens is noisy; prices are examples. |
| **Recommendation** | One throughput pick and one latency pick, each tied to a constraint, with what would change the answer. |
| **Next steps** | Real int8/int4 kernels (GPTQ/AWQ in vLLM, GGUF in llama.cpp), a bigger model, prefix caching, continuous batching under load ([step 12](../12_continuous_batching_prefix_caching/WALKTHROUGH.md)), re-pricing with real GPU prices ([step 16](../16_cost_per_token/WALKTHROUGH.md)). |

**Two-minute story:** "I built a harness that benchmarks distilgpt2 across dtype, weight quantization, batch size and KV cache, one process per config. The KV cache gave about 2x tokens/s even at this tiny size; batch 4 gave about 2.4x system throughput for a modest per-token latency increase; int8 weight-only kept perplexity within 0.3% (but since I simulate it, I do not claim a speedup); bf16 was slower on this CPU. My throughput pick was ..., with the quality gate at 2% perplexity; for a GPU I would redo the pricing with step 16's model."

(Replace the numbers in the story with the ones from *your* `inference_report.md`.)

## Files

| File | Role |
|---|---|
| `main.py` | orchestrator and worker; run it |
| `WALKTHROUGH.md` | line-by-line explanation |
| `TERMS.md` | glossary |
| `inference_report.md` / `inference_report.csv` | generated outputs (rewritten at each run) |

Back to the module: [21 README](../README.md).
