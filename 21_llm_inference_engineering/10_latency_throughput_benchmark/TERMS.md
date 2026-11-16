# Terms — Latency, throughput and goodput benchmark

Folder: `21_llm_inference_engineering/10_latency_throughput_benchmark` · Files covered: `main.py`, `WALKTHROUGH.md`, `results.csv`
Back to module: [21 README](../README.md)

**Terms in this folder:** [TTFT (time to first token)](#ttft-time-to-first-token) · [ITL (inter-token latency)](#itl-inter-token-latency) · [TPOT (time per output token)](#tpot-time-per-output-token) · [E2E latency](#e2e-latency) · [Throughput vs per-user speed](#throughput-vs-per-user-speed) · [Percentiles p50 / p95 / p99](#percentiles-p50--p95--p99) · [Goodput and SLO](#goodput-and-slo) · [Benchmark hygiene](#benchmark-hygiene) · [Peak RSS](#peak-rss) · [Device synchronisation](#device-synchronisation) · [min_new_tokens = max_new_tokens](#min_new_tokens--max_new_tokens) · [torch.set_num_threads](#torchset_num_threads) · [argparse options](#argparse-options) · [csv.DictWriter](#csvdictwriter) · [Real benchmark tools (vllm bench serve, GenAI-Perf, guidellm, llama-bench)](#real-benchmark-tools-vllm-bench-serve-genai-perf-guidellm-llama-bench) · [Error: Torch not compiled with CUDA enabled](#error-torch-not-compiled-with-cuda-enabled)

---

## TTFT (time to first token)  `(concept)`
**1. Definition:** Time from the start of a request to its first output token: queueing + prefill + first sampling step.
**2. In plain words:** How long you stare at a blank screen.
**3. Example to quote:** > "TTFT grew from 22 ms at batch 1 to 84 ms at batch 16."
**Seen in:** `main.py`, sweep; `benchlib.summarize_trial`

## ITL (inter-token latency)  `(concept)`
**1. Definition:** Gap between consecutive output tokens; a list per request.
**2. In plain words:** How smooth the stream is.
**3. Example to quote:** > "I report p95 ITL, not only the mean."
**Seen in:** `benchlib.summarize_trial`

## TPOT (time per output token)  `(concept)`
**1. Definition:** `(E2E - TTFT) / (N - 1)`: the mean ITL.
**2. In plain words:** Average time per word after the first.
**3. Example to quote:** > "TPOT excludes TTFT."
**Seen in:** `benchlib.summarize_trial`

## E2E latency  `(concept)`
**1. Definition:** Total time from start to the last token.
**2. In plain words:** How long the whole answer takes.
**3. Example to quote:** > "E2E = TTFT + (N-1) x TPOT."
**Seen in:** `main.py`, table

## Throughput vs per-user speed  `(concept)`
**1. Definition:** Throughput: total tokens/s across the batch. Per-user speed: tokens/s one request sees (1/TPOT).
**2. In plain words:** System speed vs the speed one reader feels.
**3. Example to quote:** > "Throughput up about 10x while per-user speed fell."
**Seen in:** `main.py`, sweep

## Percentiles p50 / p95 / p99  `(concept)`
**1. Definition:** The value below which 50, 95 or 99 percent of samples fall.
**2. In plain words:** The tail behaviour the mean hides.
**3. Example to quote:** > "p99 of 100 samples is the maximum."
**Seen in:** `benchlib.percentile`

## Goodput and SLO  `(concept)`
**1. Definition:** Goodput: throughput counting only requests that meet the SLO targets (e.g. TTFT and TPOT limits).
**2. In plain words:** Useful throughput.
**3. Example to quote:** > "Goodput was 129 vs throughput 644 at batch 16 under a 22 ms TPOT target."
**Seen in:** `main.py`, goodput

## Benchmark hygiene  `(concept)`
**1. Definition:** Warm-up, fixed output length, repeats with variance, pinned threads/dtype/device, GPU sync.
**2. In plain words:** The rules that make numbers reproducible.
**3. Example to quote:** > "Without warm-up the first row looks slow."
**Seen in:** `main.py`

## Peak RSS  `(concept)`
**1. Definition:** Highest resident memory of the process; bytes on macOS, kilobytes on Linux in `ru_maxrss`.
**2. In plain words:** Memory high-water mark.
**3. Example to quote:** > "`peak_rss_mb` converts per platform."
**Seen in:** `benchlib.peak_rss_mb`

## Device synchronisation  `(function)`
**1. Definition:** `torch.cuda.synchronize()` / `torch.mps.synchronize()` wait for queued kernels before reading the clock.
**2. In plain words:** Wait for the work to finish before the stopwatch.
**3. Example to quote:** > "Without it you time only kernel launches."
**Seen in:** `benchlib.sync`

## min_new_tokens = max_new_tokens  `(concept)`
**1. Definition:** Force every sequence to produce exactly N tokens, ignoring EOS.
**2. In plain words:** Equal-length answers.
**3. Example to quote:** > "Needed for fair batch comparison."
**Seen in:** `benchlib.timed_generate`; `main.py` sanity check

## torch.set_num_threads  `(function)`
**1. Definition:** Sets how many CPU threads PyTorch uses for ops.
**2. In plain words:** Cores given to the job.
**3. Example to quote:** > "Always report the thread count."
**Seen in:** `main.py`

## argparse options  `(syntax)`
**1. Definition:** `ap.add_argument(name, default=, choices=)` declares CLI flags.
**2. In plain words:** Command-line switches.
**3. Example to quote:** > "`--device mps`"
**Seen in:** `main.py`

## csv.DictWriter  `(function)`
**1. Definition:** Writes dict rows with a header; `newline=""` avoids blank lines on Windows.
**2. In plain words:** Dicts to CSV.
**3. Example to quote:** > "`benchlib.to_csv(rows, path)`"
**Seen in:** `benchlib.to_csv`

## Real benchmark tools (vllm bench serve, GenAI-Perf, guidellm, llama-bench)  `(concept)`
**1. Definition:** Load generators that implement TTFT/ITL/TPOT/throughput/goodput against servers or local models.
**2. In plain words:** The industrial versions of this script.
**3. Example to quote:** > "Reference only: not run in this course."
**Seen in:** `WALKTHROUGH.md`, theory

## Error: Torch not compiled with CUDA enabled  `(error)`
**1. Definition:** Raised when you select a cuda device on a build without CUDA.
**2. In plain words:** This machine has no NVIDIA GPU.
**3. Example to quote:** > "Check `torch.cuda.is_available()` first."
**Seen in:** `main.py`, BREAK IT
