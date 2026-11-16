# Terms — Inference report project

Folder: `21_llm_inference_engineering/17_inference_report_project` · Files covered: `main.py`, `README.md`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Benchmark matrix](#benchmark-matrix) · [Subprocess per configuration](#subprocess-per-configuration) · [Worker mode (--worker)](#worker-mode---worker) · [Peak RSS](#peak-rss) · [TTFT / ITL / tok/s](#ttft--itl--toks) · [Perplexity gate](#perplexity-gate) · [Fake (simulated) int8 quantization](#fake-simulated-int8-quantization) · [Throughput pick vs latency pick](#throughput-pick-vs-latency-pick) · [PRICE_PER_HOUR](#price_per_hour)

---

## Benchmark matrix  `(concept)`
**1. Definition:** A grid of configurations (dtype x quantization x batch x KV cache) each measured with the same method.
**2. In plain words:** Many variants measured fairly.
**3. Example to quote:** > "I benchmarked a matrix of nine configurations with one subprocess each."
**Seen in:** `main.py` section S2

## Subprocess per configuration  `(concept)`
**1. Definition:** Running each configuration in a fresh process so load time and peak RSS are not contaminated by earlier runs.
**2. In plain words:** Clean memory numbers.
**3. Example to quote:** > "Peak RSS only grows within a process, so I isolate each config."
**Seen in:** `main.py` section S2

## Worker mode (--worker)  `(syntax)`
**1. Definition:** The same script re-invoked with a JSON config to measure one configuration and print one `RESULT` line.
**2. In plain words:** The script calls itself as a measuring tool.
**3. Example to quote:** > "The orchestrator parses each worker's RESULT line."
**Seen in:** `main.py` section S1

## Peak RSS  `(concept)`
**1. Definition:** Maximum resident set size of a process (macOS reports bytes, Linux kilobytes) via `resource.getrusage`.
**2. In plain words:** How much RAM the run really needed.
**3. Example to quote:** > "Without KV cache the batch-4 run used more RSS because of recomputed activations."
**Seen in:** `../benchlib.py`

## TTFT / ITL / tok/s  `(concept)`
**1. Definition:** Time to first token; inter-token latency; output tokens per second.
**2. In plain words:** Start-up delay, per-token pace, and total speed.
**3. Example to quote:** > "Batching raised throughput 2.4x at 1.6x ITL."
**Seen in:** `main.py`, report

## Perplexity gate  `(concept)`
**1. Definition:** Only configurations whose perplexity is within a tolerance (2%) of the fp32 baseline are eligible for the recommendation.
**2. In plain words:** Never trade away quality silently.
**3. Example to quote:** > "I ranked only configurations inside the quality gate."
**Seen in:** `main.py` section S3

## Fake (simulated) int8 quantization  `(concept)`
**1. Definition:** Quantize then dequantize weights while keeping float math: measures quality impact, not speed or memory.
**2. In plain words:** Pretend int8 to check quality.
**3. Example to quote:** > "Perplexity changed by 0.2%, but I did not claim a speedup because the math is still float."
**Seen in:** `main.py` section S1

## Throughput pick vs latency pick  `(concept)`
**1. Definition:** Two recommendations: lowest $/1M tokens at batch 4 and lowest TTFT plus 16 decode steps at batch 1.
**2. In plain words:** Different goals give different winners.
**3. Example to quote:** > "Batch jobs get the throughput pick; chat gets the latency pick."
**Seen in:** `main.py` section S3

## PRICE_PER_HOUR  `(syntax)`
**1. Definition:** Env-var knob for the example hourly price used in the report's dollar column.
**2. In plain words:** The assumption that scales cost.
**3. Example to quote:** > "Cost scales linearly with price, so rankings do not change."
**Seen in:** `main.py`
