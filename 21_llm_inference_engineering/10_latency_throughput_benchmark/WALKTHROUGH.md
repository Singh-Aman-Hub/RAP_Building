# Step 10 — Latency, throughput and goodput benchmark (TTFT, ITL, TPOT, p95)

## The story so far
[Step 09](../09_prefill_decode_roofline/WALKTHROUGH.md) explained *why* prefill is compute-bound and decode is memory-bound, so batching decode is almost free. This step turns that into a proper measurement. [Step 08](../08_hf_inference_batching_kvcache/WALKTHROUGH.md) had already shown that batching and the KV cache help, but "it feels faster" is not an engineering answer.

## The pain point
"Tokens per second" hides everything a user cares about. One number cannot tell you how long until the first word appears, how smooth the stream is, or how bad the worst requests are. Worse, a careless benchmark (no warm-up, one run, unknown thread count, answers of different lengths) produces numbers that are wrong by 2x and cannot be reproduced.

## The spark
Break a request's timeline into named pieces, record a timestamp for every generated token, and report the pieces separately: TTFT for the wait, ITL/TPOT for the stream, E2E for the total, percentiles for the tail, throughput for the cost, and goodput for "throughput that met the latency targets". Use one shared helper ([`../benchlib.py`](../benchlib.py)) so every later step measures the same way.

## Goal
Sweep batch sizes 1, 2, 4, 8, 16 on distilgpt2 and report TTFT, TPOT, ITL p95, E2E, throughput, per-user speed, variance across repeated trials, and goodput under an example SLO. Keep writing `results.csv`.

## The theory (read this before the interview)

**One request's timeline**
```
arrival ─► [queue wait] ─► [prefill] ─► tok1 ──ITL──► tok2 ──ITL──► tok3 ... tokN
                                         │                                     │
                      TTFT ──────────────┘                                     │
                      E2E latency ─────────────────────────────────────────────┘
TPOT = (E2E - TTFT) / (N - 1)   (mean ITL; TTFT excluded)
```
In this benchmark there is no queue (no server), so TTFT = prefill + first sampling step. In a server TTFT also includes queueing, which is why it is the metric that explodes under load.

**Definitions**
- **TTFT** (time to first token): start until the first output token. Driven by prompt length, queueing and cache hits. Users perceive it as "is it responding?".
- **ITL** (inter-token latency): gap between two consecutive output tokens. It is a *distribution*: look at p50, p95, p99.
- **TPOT** (time per output token): `(E2E - TTFT) / (N - 1)`, the average ITL. Some tools define ITL per token and TPOT per request; the mean of ITLs equals TPOT.
- **E2E latency** (end-to-end, request latency): `TTFT + (N-1) * TPOT`.
- **Throughput**: output tokens per second for the whole system (`batch * N / E2E` here), or requests per second. Drives cost per token ([step 16](../16_cost_per_token/WALKTHROUGH.md)).
- **Per-user speed**: `1 / TPOT` tokens per second, what one reader perceives (humans read about 5 to 10 tokens per second, so beyond that smoothness matters more than speed).
- **Percentiles** p50/p95/p99: the value 50, 95 or 99 percent of requests/tokens are below. The mean hides the tail. p99 with 100 samples is just the max, so collect enough samples.
- **Goodput**: throughput counting only requests that met the SLO (service-level objective, e.g. TTFT under 400 ms and TPOT under 22 ms). A system can post high throughput by overloading, with all requests slow; goodput punishes that. It is the metric for deciding how many users a GPU really serves.

**The trade-off.** Larger batch: higher throughput (the step 09 effect), but each decode step takes longer, so per-user speed and p95 ITL get worse, and TTFT grows (a bigger prefill). The right operating point is the biggest batch that still meets the SLO: that is exactly the point where goodput peaks.

**Benchmarking hygiene (the part interviewers probe)**
1. **Warm-up**: discard the first runs (lazy initialisation, allocator growth, caches, JIT, CUDA graph capture).
2. **Fixed output length**: force every sequence to produce N tokens (`min_new_tokens == max_new_tokens`, or ignore EOS) so batch sizes are comparable; otherwise a "faster" run may have just stopped early.
3. **Repeat and report variance**: at least 5 trials; report mean and standard deviation (or p50/p95), never one run.
4. **Pin and report the environment**: thread count (`torch.set_num_threads`), dtype, device, library versions, prompt length, other load on the machine.
5. **Synchronise on GPU**: kernels are asynchronous, so call `torch.cuda.synchronize()` before reading the clock (done in `benchlib`).
6. **Separate prefill from decode**: report TTFT and TPOT, not just total time.
7. **Realistic load shape**: single-request loops measure latency; for capacity measure a *request-rate* sweep with Poisson arrivals and realistic prompt/output length distributions, and watch the knee where TTFT blows up.
8. **Same tokens**: check outputs match across the implementations you compare.

**Real tools (reference only; not run in this course):** `vllm bench serve` (load generator for an OpenAI-compatible server; reports TTFT, TPOT, ITL, throughput, goodput with SLO flags), NVIDIA **GenAI-Perf** (and its successor tooling) for Triton/NIM/OpenAI endpoints, **guidellm** (SLO-oriented load sweeps), and llama.cpp's **`llama-bench`** (prompt-processing tokens/s `pp` and token-generation `tg` for local GGUF models). They implement the same metrics defined above, with concurrent clients and request rates.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| TTFT (time to first token) | first-token latency, time to first byte (roughly) | Yes | Wait until the first token appears; includes queueing and prefill. |
| ITL (inter-token latency) | token gap, decode step latency | Yes | One gap between consecutive tokens; a distribution. |
| TPOT (time per output token) | per-token latency, mean ITL | Near | Per-request mean of ITL, TTFT excluded. |
| E2E latency | request latency, total latency | Yes | Arrival to last token. |
| Throughput | tokens/s, TPS, req/s | Depends | System-wide output rate; say tokens or requests. |
| Per-user speed | tokens/s per user, 1/TPOT | Related | What one stream feels. |
| Goodput | SLO-attaining throughput | Different | Only counts requests that met latency targets. |
| p50 / p95 / p99 | median, tail latency | Yes | Percentiles of the same distribution. |
| SLO / SLA | service-level objective / agreement | Different | SLO = internal target; SLA = contractual promise. |
| Peak RSS / VRAM | high-water memory | Different places | CPU RAM of the process vs GPU memory. |
| Warm-up | first-call overhead | Yes | Untimed runs before measuring. |

## Run it
```bash
cd 21_llm_inference_engineering/10_latency_throughput_benchmark
python main.py                 # CPU (reference)
python main.py --device mps    # Apple GPU: noisy for such a small model
# from repo root (timing run: use the heavy-run lock if other jobs share the machine):
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/10_latency_throughput_benchmark/main.py
```
About 40 to 60 seconds on CPU. Knobs: `--new-tokens`, `--prompt-len`, `--threads`, `--trials`, and env `SLO_TTFT_MS`, `SLO_TPOT_MS`. The timing loop lives in `../benchlib.py` (documented there).

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–16)

```python
"""
GOAL:  Benchmark generation properly: TTFT, ITL/TPOT, E2E latency, p50/p95/p99, throughput and goodput at batch sizes
       1..16, with warm-up, repeated trials and variance. Saves results.csv.  Uses ../benchlib.py for the timing.

RUN (from this step folder):  python main.py            (options: --device mps|cuda, --new-tokens N, --threads N, --trials N)
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/10_latency_throughput_benchmark/main.py
EXPECTED OUTPUT (shortened; numbers vary by machine and are noisy):
    batch  TTFT ms  TPOT ms  ITL p95 ms  E2E s  tok/s  per-user tok/s  +-std tok/s  goodput tok/s
        1     ..       ..        ..       ..     ..        ..             ..             ..
       16     ..       ..        ..       ..     ..        ..             ..             ..
    => throughput rises with batch size while per-user speed falls: the trade-off. TTFT grows with batch (prefill).
    thread-count check: same workload with 1, 2, 4 threads
    sanity: manual KV-cache loop vs model.generate (same tokens, similar time)
NEEDS: torch, transformers (distilgpt2 cached, no downloads).  ~40-60 s on CPU. Timing run: use the global lock.
Builds on: 09_prefill_decode_roofline (why decode is flat in batch), 08_hf_inference_batching_kvcache (KV cache, generate).
"""
```

- **In simple words:** What is measured and what is printed.
- **Exactly:** Docstring in the required format.
- **Why it is here:** Orientation.

### 2. Imports and the shared helper  (`main.py` lines 17–27)

```python
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find ../benchlib.py
import benchlib
```

- **In simple words:** Standard imports plus `benchlib`.
- **Exactly:** `HF_HUB_OFFLINE=1` forbids downloads; `sys.path.insert` makes `../benchlib.py` importable from any folder.
- **Why it is here:** One timing implementation for every step.

### 3. Command-line options  (`main.py` lines 29–35)

```python
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
ap.add_argument("--new-tokens", type=int, default=32)
ap.add_argument("--prompt-len", type=int, default=16)
ap.add_argument("--threads", type=int, default=4)           # torch CPU threads: ALWAYS report it with a benchmark
ap.add_argument("--trials", type=int, default=5)
args = ap.parse_args()
```

- **In simple words:** Device, output length, prompt length, threads, trials.
- **Exactly:** Defaults: cpu, 32 new tokens, 16-token prompts, 4 threads, 5 trials. `--device cuda` on a Mac raises an AssertionError.
- **Why it is here:** Lets you rerun with different settings without editing code.

### 4. Setup: model, SLO and synthetic prompts  (`main.py` lines 37–49)

```python
# ---- 1. Setup: model, device, fixed-length synthetic prompts ---------------------------------------------------------
device = torch.device(args.device)
model, tokenizer = benchlib.load_causal_lm("distilgpt2", threads=args.threads)
model = model.to(device)
torch.manual_seed(0)
SLO_TTFT_MS = float(os.environ.get("SLO_TTFT_MS", 400))          # example service-level objective (SLO): BOTH must hold
SLO_TPOT_MS = float(os.environ.get("SLO_TPOT_MS", 22))


def make_ids(batch):
    """Identical-length prompts (no padding needed): `batch` rows of prompt-len random tokens."""
    return torch.randint(100, 5000, (batch, args.prompt_len), device=device)

```

- **In simple words:** Load distilgpt2, define an example SLO, and a prompt generator.
- **Exactly:** `load_causal_lm` returns `(model, tokenizer)` offline with the requested thread count. The SLO is read from env vars. `make_ids(B)` makes `B` identical-length random-token prompts so no padding or attention mask is needed.
- **Why it is here:** Equal-length prompts keep the comparison fair and the loop simple.

### 5. The batch-size sweep with trials, percentiles and goodput  (`main.py` lines 51–68)

```python
# ---- 2. The batch-size sweep: warm-up (inside run_trials) + repeated trials -------------------------------------------
print(f"device={args.device} threads={args.threads} prompt={args.prompt_len} tokens, {args.new_tokens} new tokens (forced: no EOS stop), "
      f"{args.trials} trials after 2 warm-ups | SLO: TTFT<={SLO_TTFT_MS:.0f} ms and TPOT<={SLO_TPOT_MS:.0f} ms\n")
print(f"{'batch':>5} {'TTFT ms':>8} {'TPOT ms':>8} {'ITLp95 ms':>10} {'E2E s':>7} {'tok/s':>7} {'user tok/s':>11} {'+-std':>6} {'goodput':>8}")
rows = []
for bs in (1, 2, 4, 8, 16):
    trials = benchlib.run_trials(model, make_ids(bs), args.new_tokens, warmup=2, trials=args.trials)
    a = benchlib.aggregate(trials)
    # GOODPUT: only count tokens from trials that met BOTH latency targets (throughput that users are happy with).
    met = [t for t in trials if t["ttft_s"] * 1e3 <= SLO_TTFT_MS and t["tpot_s"] * 1e3 <= SLO_TPOT_MS]
    goodput = a["tok_per_s"] * len(met) / len(trials)
    rows.append(dict(batch=bs, ttft_ms=a["ttft_s"] * 1e3, tpot_ms=a["tpot_s"] * 1e3, itl_p50_ms=a["itl_p50_s"] * 1e3,
                     itl_p95_ms=a["itl_p95_s"] * 1e3, itl_p99_ms=a["itl_p99_s"] * 1e3, e2e_s=a["e2e_s"],
                     latency_s=a["e2e_s"], tokens_per_s=a["tok_per_s"], tok_s_std=a["tok_per_s_std"],
                     per_user_tokens_per_s=a["n_tokens"] / a["e2e_s"], goodput_tok_s=goodput, peak_rss_mb=a["peak_rss_mb"]))
    r = rows[-1]
    print(f"{bs:>5} {r['ttft_ms']:>8.1f} {r['tpot_ms']:>8.1f} {r['itl_p95_ms']:>10.1f} {r['e2e_s']:>7.2f} {r['tokens_per_s']:>7.0f} "
          f"{r['per_user_tokens_per_s']:>11.1f} {r['tok_s_std']:>6.0f} {goodput:>8.0f}")
```

- **In simple words:** For each batch size: warm-up, 5 timed trials, summary, goodput.
- **Exactly:** `run_trials` discards 2 warm-ups then returns one summary per trial (TTFT, ITL list, TPOT, E2E, tok/s). `aggregate` gives means, std and pooled p50/p95/p99. Goodput = throughput times the fraction of trials meeting both SLO targets. `per_user = n_tokens / E2E`.
- **Why it is here:** The measurement that earlier versions of this step lacked.

### 6. Save results.csv and report the spread  (`main.py` lines 70–74)

```python
out_file = Path(__file__).resolve().parent / "results.csv"
benchlib.to_csv(rows, out_file)                               # results.csv for your notes
print("\nsaved", out_file)
print("=> throughput rises with batch size while per-user speed falls: the trade-off. TTFT grows with batch (bigger prefill).")
print(f"   coefficient of variation of tok/s (std/mean): {max(r['tok_s_std'] / r['tokens_per_s'] for r in rows):.0%} worst case: report spread, not one number")
```

- **In simple words:** Write the table and print the worst coefficient of variation.
- **Exactly:** `benchlib.to_csv` writes one row per batch size with all columns; CV = std / mean of tok/s.
- **Why it is here:** Without a spread, one lucky run looks like a result.

### 7. Thread-count check  (`main.py` lines 76–83)

```python
# ---- 3. Hygiene check: thread count changes the numbers ---------------------------------------------------------------
if device.type == "cpu":
    print("\nthread-count check (batch 4):")
    for th in (1, 2, 4):
        torch.set_num_threads(th)
        a = benchlib.aggregate(benchlib.run_trials(model, make_ids(4), 16, warmup=1, trials=3))
        print(f"  threads={th}: TPOT {a['tpot_s'] * 1e3:6.1f} ms | tok/s {a['tok_per_s']:6.0f}")
    torch.set_num_threads(args.threads)
```

- **In simple words:** Repeat batch 4 with 1, 2 and 4 threads.
- **Exactly:** `torch.set_num_threads` changes the intra-op thread pool for subsequent operations; reset afterwards.
- **Why it is here:** Shows why thread count must be reported.

### 8. Sanity check against model.generate  (`main.py` lines 85–95)

```python
# ---- 4. Sanity: our manual loop vs the library's generate() -----------------------------------------------------------
ids = make_ids(1)
t = benchlib.timed_generate(model, ids, args.new_tokens)
s = time.perf_counter()
with torch.no_grad():
    ref = model.generate(ids, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
gen_s = time.perf_counter() - s
print(f"\nsanity: same tokens as model.generate: {torch.equal(t['tokens'], ref[:, args.prompt_len:])} | "
      f"manual loop E2E {1e3 * (t['token_times'][-1] - t['t0']):.0f} ms vs generate {gen_s * 1e3:.0f} ms (generate adds logits processors / stopping checks)")
print("real tools do this at scale (not run here): `vllm bench serve`, NVIDIA GenAI-Perf, guidellm, llama.cpp `llama-bench`.")
```

- **In simple words:** Confirm the manual loop produces the same tokens as the library.
- **Exactly:** `torch.equal` compares the generated ids; both timed once. Ends by naming the real benchmark tools.
- **Why it is here:** A benchmark harness must be validated against a trusted reference.

### 9. TRY IT / BREAK IT  (`main.py` lines 97–101)

```python
# TRY IT: python main.py --new-tokens 128 : TPOT stays flat, E2E grows linearly; TTFT does not change.
# TRY IT: python main.py --prompt-len 512 : TTFT jumps (prefill is compute-bound, step 09) while TPOT barely moves.
# TRY IT: python main.py --threads 1 then --threads 8 (more than your cores): compare tok/s and the +-std column.
# BREAK IT: set trials=1 and warmup=0 in the loop: batch 1 (the first measured row) looks slow and the std column is 0 = fake confidence.
# BREAK IT: python main.py --device cuda on this Mac -> AssertionError 'Torch not compiled with CUDA enabled'.
```

- **In simple words:** Experiments in comments.
- **Exactly:** Longer outputs, longer prompts, threads; no warm-up; wrong device.
- **Why it is here:** Practice.

## Syntax explained

| Syntax | Meaning | Tiny example |
|---|---|---|
| `ap.add_argument("--device", default="cpu", choices=[...])` | command-line option with allowed values | `--device mps` |
| `float(os.environ.get("SLO_TPOT_MS", 22))` | environment knob with a default | `SLO_TPOT_MS=30 python main.py` |
| `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` | import `benchlib.py` from the module folder | |
| `[t for t in trials if cond]` | list comprehension filtering trials that met the SLO | goodput |
| `dict(a=1, b=2)` | build a row for the CSV | one row per batch size |
| f-string `{x:>8.1f}` | align and format numbers | table columns |
| `torch.randint(100, 5000, (B, L))` | random token ids, shape `(B, L)` | synthetic prompts |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `benchlib.load_causal_lm` | load model and tokenizer offline | `from_pretrained`; sets the CPU thread count |
| `benchlib.run_trials(model, ids, N, warmup, trials)` | warm-up runs then timed runs | each run is `timed_generate`: prefill pass, then N-1 decode passes with a KV cache, timestamping every token |
| `benchlib.aggregate` | mean, std, pooled p50/p95/p99 of ITL | sorts the pooled gaps and interpolates |
| `benchlib.to_csv` | write dict rows to CSV | `csv.DictWriter`, `newline=""` |
| `torch.set_num_threads` | CPU thread pool size | intra-op parallelism of BLAS and elementwise ops |
| `model.generate(..., min_new_tokens=N)` | library generation loop | adds logits processors, stopping criteria; used as a cross-check |

## Execution trace (real run)

Real output (Apple M1, CPU; other jobs may have shared the machine; your numbers will differ):

```
device=cpu threads=4 prompt=16 tokens, 32 new tokens (forced: no EOS stop), 5 trials after 2 warm-ups | SLO: TTFT<=400 ms and TPOT<=22 ms

batch  TTFT ms  TPOT ms  ITLp95 ms   E2E s   tok/s  user tok/s  +-std  goodput
    1     21.6     15.5       20.6    0.50      65        63.8      9       65
    2     23.0     18.8       21.2    0.61     106        52.9      4      106
    4     32.3     18.6       21.3    0.61     210        52.5      8      210
    8     53.6     20.5       28.8    0.69     375        46.4     42      300
   16     84.0     23.0       25.3    0.80     644        40.1     38      129

saved /Users/aman/Desktop/Project_resume/RAP_interview/21_llm_inference_engineering/10_latency_throughput_benchmark/results.csv
=> throughput rises with batch size while per-user speed falls: the trade-off. TTFT grows with batch (bigger prefill).
   coefficient of variation of tok/s (std/mean): 14% worst case: report spread, not one number

thread-count check (batch 4):
  threads=1: TPOT   17.3 ms | tok/s    218
  threads=2: TPOT   17.8 ms | tok/s    214
  threads=4: TPOT   18.5 ms | tok/s    207

sanity: same tokens as model.generate: True | manual loop E2E 442 ms vs generate 562 ms (generate adds logits processors / stopping checks)
real tools do this at scale (not run here): `vllm bench serve`, NVIDIA GenAI-Perf, guidellm, llama.cpp `llama-bench`.
```

Reading the numbers:
- **Throughput up, per-user speed down:** tok/s grows from 65 at batch 1 to 644 at batch 16 (about 10x) while per-user speed drops only from 64 to 40 tok/s. TPOT rises from 15.5 ms to 23 ms (1.5x) for a 16x larger batch: batching raises throughput much faster than it costs latency (step 09).
- **TTFT grows with batch:** 22 ms at batch 1, 84 ms at batch 16: prefill processes `batch x 16` prompt tokens in one pass.
- **Variance is real:** the `+-std` column ranges from 4 to 42 tok/s and the worst coefficient of variation is 14 percent; batches 8 and 16 have the largest spread. A single run would have misled you: report the spread.
- **Goodput:** with the SLO in the header line (TTFT 400 ms, TPOT 22 ms) goodput equals throughput for batch 1 to 4, but at batch 8 only some trials stay under 22 ms (goodput 300 of 375) and at batch 16 few do (129 of 644). Raw throughput kept climbing while the number of tokens delivered *within the target* collapsed. That is the point of goodput: raw throughput says nothing about whether users were happy.
- **Threads:** for this small model at batch 4 the thread count changes little (1 thread was even marginally faster than 4); the matrices are too small to parallelise well. For large models the effect is large, so always report it.
- **Manual loop vs `generate`:** identical tokens; `generate` is somewhat slower because of per-step bookkeeping, which is why serving engines write their own loops.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `--new-tokens 128` | E2E grows ~linearly, TPOT flat, TTFT unchanged | decode cost is per token |
| `--prompt-len 512` | TTFT jumps, TPOT barely moves | prefill is compute-bound and scales with prompt length (step 09) |
| `SLO_TPOT_MS=30` | goodput equals throughput for more batch sizes | looser target |
| `--threads 1` vs `8` | little change on this small model | not enough work per op to parallelise |
| add `32` to the batch tuple | throughput flattens, per-user speed keeps falling | the ridge point of the machine is near |
| `--trials 1` | std column becomes 0 | no variance information: fake confidence |

## Debugging practice
**BREAK IT:** set `warmup=0` and `trials=1` in the `run_trials` call. Symptom: the batch-1 row (the first thing measured) comes out slower than batch 2 in per-user speed, and the std column is 0 so there is no sign the number is unreliable. Fix: always warm up and repeat, and print the spread.

**BREAK IT 2:** `--device cuda` on this Mac raises `AssertionError: Torch not compiled with CUDA enabled`. Check `torch.cuda.is_available()` and `torch.backends.mps.is_available()` first. (On a real GPU, forgetting `synchronize` instead produces timings that are far too small: you time the kernel launch, not the kernel.)

## Common mistakes
- Quoting throughput without TTFT/TPOT, or latency without the batch size and load.
- Comparing runs whose outputs had different lengths (an early EOS makes a run look fast).
- One run, no warm-up, no variance, unknown thread count.
- Reporting the mean latency only: the tail (p95/p99) is what breaks SLOs.
- Confusing TPOT and ITL, or including TTFT in TPOT.
- Measuring at a request rate the server cannot sustain and calling the queueing delay "latency".
- Benchmarking on a busy laptop. Check the load first.

## Rebuild checklist
1. Write `timed_generate`: prefill pass, then a decode loop, `perf_counter` after each token (sync on GPU).
2. Turn timestamps into TTFT, ITL list, TPOT = mean ITL, E2E, tok/s.
3. Add warm-up runs and 5 trials; aggregate mean, std, pooled p50/p95/p99.
4. Sweep batch sizes; compute throughput `B*N/E2E` and per-user speed.
5. Define an SLO and compute goodput.
6. Write the rows to `results.csv` with `csv.DictWriter`.

## Say it aloud
- "I report TTFT for responsiveness, ITL or TPOT for stream smoothness, p95/p99 for the tail, and throughput for cost, because any one of them alone misleads."
- "Batching raises throughput nearly linearly while decode is memory-bound, but per-user speed and TTFT get worse, so the right batch is the largest one that still meets the SLO: that is where goodput peaks."
- "A valid benchmark warms up, fixes the output length, repeats trials and reports variance, and states threads, dtype, device and prompt length."
- "For serving-level numbers I would use `vllm bench serve`, GenAI-Perf or guidellm with a realistic arrival process, rather than a closed loop on one process."

## The new problem (cliffhanger)
The sweep shows throughput rising with batch size until something stops it. On a real GPU that something is KV-cache memory: every request in the batch needs its own cache, and a naive server reserves far more than it uses. [Step 11, PagedAttention and fragmentation](../11_paged_attention_fragmentation/WALKTHROUGH.md).
