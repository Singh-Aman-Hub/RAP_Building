# Step 17 — Inference report project (capstone)

## The story so far
[Step 16](../16_cost_per_token/WALKTHROUGH.md) priced tokens. Steps 01 to 15 gave you formats, quantization, memory arithmetic, a proper benchmark, KV cache, batching, caching, speculation, parallelism and engines. Each was a separate experiment.

## The pain point
In an interview "I know quantization and batching" is cheap. What convinces is one artefact where you *measured* several choices on the same model with the same method, wrote down what you found, said what you could not conclude, and made a recommendation. Doing that from memory is hard: the traps (warm-up, noisy single runs, polluted memory numbers, fake quantization mistaken for real speedups) are exactly the things an experienced interviewer probes.

## The spark
Treat inference tuning as an experiment: a matrix of configurations, one clean process per configuration, a fixed metric set, a quality gate, and a generated report. The script writes the report itself (`inference_report.md` and `.csv`), so the evidence and the prose cannot drift apart.

## Goal
Run a CPU benchmark matrix on distilgpt2 (dtype x weight quantization x batch x KV cache), measure load time, TTFT, ITL, tokens/s, peak RSS, perplexity and example $/1M tokens, and generate a report with findings, a recommendation and caveats. Then practise presenting it ([README rubric](README.md)).

## The theory (read this before the interview)

**Experiment design.**
- *Isolation:* peak RSS never decreases inside a process, so measuring many configs in one process makes later rows wrong. One subprocess per config gives clean RSS and load time.
- *Warm-up and repetition:* the first runs allocate memory and initialise kernels. Discard a warm-up, take the median of 3 trials.
- *Fixed workload:* same prompt length (32), same new tokens (16), same threads (4), reported.
- *Quality gate:* a faster configuration is only a candidate if perplexity stays within a tolerance (2%) of the baseline.
- *Honesty:* report what was *simulated* (fake int8), what is *machine-dependent* (CPU bf16) and what is *noise* (differences smaller than run-to-run variation).

**Metrics (names side by side).**

```
TTFT (time to first token)       = prefill + first sampling step           <- user feels this as "how long until it starts"
ITL  (inter-token latency)       = gap between consecutive output tokens   ~= TPOT (time per output token)
tok/s (system throughput)        = batch x tokens / end-to-end time
load time                        = from_pretrained + (fake) quantization + dtype conversion
peak RSS                         = maximum resident memory of the process  (weights + activations + KV cache + allocator slack)
perplexity                       = exp(mean next-token cross-entropy) on held-out text (quality)
$/1M output tokens               = price per hour / (tok/s x 3600) x 1e6   (step 16; example price)
```

**What to expect (and why).**
- KV cache on: each decode step feeds one token instead of the whole sequence, so ITL falls and tokens/s rises; RSS also differs slightly.
- Batch 4: the weights are read once per step for four sequences, so system throughput rises but each step takes longer: throughput up, per-user speed down (step 10).
- Fake int8 weight-only: perplexity barely moves; speed and memory do not improve because the arithmetic is still float32 (a *real* int8 path would shrink weight bytes 4x vs fp32).
- bf16 on CPU: many laptop CPUs lack fast bf16 matmul, so bf16 can be *slower* than fp32 even though it halves weight bytes. On GPUs it is the opposite.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| ITL | TPOT (time per output token), per-token latency, decode latency | Mostly | ITL is each gap; TPOT is the mean over a request (excluding TTFT). |
| Weight-only quantization | W8A16, "int8 weights, fp16 activations", `qdq` | Yes | Weights stored low-bit; compute in higher precision (step 04/05). |
| Fake quantization | simulated quantization, quantize-dequantize (QDQ) | Yes | Measures error, gives no speedup. |
| Throughput pick / latency pick | cost-optimised / interactive configuration | Different goals | One minimises $/token, the other time to a response. |
| RSS | resident set size, "memory used" | Yes | RAM actually held by the process (not virtual size). |
| Benchmark matrix | ablation grid, sweep | Yes | Vary factors systematically. |
| Quality gate | acceptance threshold, eval guardrail | Yes | A configuration that fails it cannot win. |

## Run it

```
cd 21_llm_inference_engineering/17_inference_report_project
python main.py                      # about 1.5-3 min on a laptop CPU; writes inference_report.md and inference_report.csv
python main.py --quick              # fp32 only, 8 configs
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/17_inference_report_project/main.py   # from repo root
```

This is a timing run: in this repo, run it through the global heavy-run lock so nothing else competes for CPU and RAM.

## Code, step by step

### 1. Header, constants, imports  (`main.py` lines 1–34)

```python
"""
GOAL:  Capstone of module 21. Benchmark a MATRIX of inference configurations on CPU -- dtype (fp32/bf16) x weight quantization
       (none / fake int8 weight-only) x batch (1/4) x KV cache (on/off) -- measuring load time, TTFT, ITL, tok/s, peak RSS,
       perplexity and estimated $/1M tokens, then write inference_report.md + inference_report.csv with a recommendation.
       Each configuration runs in its OWN subprocess so peak RSS and load time are clean.

RUN (from this step folder):  python main.py                (options: --model gpt2, --new-tokens 16, --quick)
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/17_inference_report_project/main.py
EXPECTED OUTPUT (shortened; numbers vary by machine):
    running 9 configs (subprocess each) ...
    dtype  quant  batch kv   load_s ttft_ms itl_ms tok/s rss_MB ppl   $/1M
    fp32   none   1     on   ...
    ... (10 rows: fp32 x {none,int8} x batch {1,4} x KV {on,off}, plus bf16 x batch 1 with KV on)
    wrote inference_report.md and inference_report.csv ; recommendation: ...
NEEDS: torch, transformers (distilgpt2 cached; no downloads), ../benchlib.py, ../quantlib.py.  ~1-2 min on CPU.
       Timing run: use the global lock.
Honest limits: int8 here is FAKE quantization (quantize->dequantize, still float math): it measures QUALITY (perplexity), not
speed or RSS. CPU timings of tiny models say little about GPU serving; the point is the method and the report.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                      # find ../benchlib.py and ../quantlib.py

PRICE_PER_HOUR = float(os.environ.get("PRICE_PER_HOUR", 0.10))   # EXAMPLE price of a small cloud CPU box, $/hour: edit it
PROMPT_LEN = 32
```

- **In simple words:** Imports, offline mode, and two constants: the example price and the prompt length.
- **Exactly:** `sys.path.insert(0, HERE.parent)` makes `benchlib` and `quantlib` importable. `PRICE_PER_HOUR` reads an environment variable so you can re-price without editing code. The docstring states the honest limits up front.
- **Why it is here:** Constants that affect results belong at the top, visible and editable.

### 2. The worker: measure one configuration  (`main.py` lines 37–86)

```python
# ---- [S1] WORKER: measure ONE configuration (runs in a fresh process) -----------------------------------------------------
def worker(cfg):
    import torch
    import benchlib
    import quantlib as ql

    t_load = time.perf_counter()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg["dtype"]]
    model, tok = benchlib.load_causal_lm(cfg["model"], dtype=torch.float32, threads=4)   # load fp32 first so quantization sees fp32
    if cfg["quant"] == "int8":                          # fake weight-only int8: per-output-channel symmetric RTN (step 04)
        for _, mod in ql.conv1d_layers(model):
            w = ql.get_w_out_in(mod)
            ql.set_w_out_in(mod, ql.qdq(w, bits=8, symmetric=True, gran="row"))
    model = model.to(dtype)
    load_s = time.perf_counter() - t_load

    # quality: perplexity on text the quantizer never "saw" (same helper as steps 04-07)
    _, eval_text = ql.corpus_split()
    ppl = ql.perplexity(model, tok, eval_text[:3000], ctx=64)

    # speed: manual greedy loop so KV cache can be switched OFF (benchlib.timed_generate always uses the cache)
    torch.manual_seed(0)
    ids = torch.randint(100, 5000, (cfg["batch"], PROMPT_LEN))

    @torch.no_grad()
    def run(n_new):
        t0 = time.perf_counter()
        cur, cache, stamps = ids, None, []
        for step in range(n_new):
            if cfg["kv"] == "on":
                out = model(input_ids=cur, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                cur = nxt                               # feed only the NEW token
            else:                                       # no cache: re-run the whole growing sequence every step
                out = model(input_ids=cur, use_cache=False)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                cur = torch.cat([cur, nxt], 1)
            stamps.append(time.perf_counter())
            if step == 0:
                t_pref = stamps[0]
        return dict(t0=t0, t_prefill_end=t_pref, token_times=stamps, batch=cfg["batch"], prompt_len=PROMPT_LEN)

    run(4)                                              # warm-up
    trials = sorted((benchlib.summarize_trial(run(cfg["new_tokens"])) for _ in range(3)), key=lambda s: s["e2e_s"])
    s = trials[1]                                       # median trial
    itl_ms = 1000 * sorted(s["itl_s"])[len(s["itl_s"]) // 2]
    tps = s["tok_per_s"]
    return dict(cfg, load_s=load_s, ttft_ms=1000 * s["ttft_s"], itl_ms=itl_ms, tok_s=tps,
                rss_mb=benchlib.peak_rss_mb(), ppl=ppl, cost_per_mtok=PRICE_PER_HOUR / (tps * 3600) * 1e6)
```

- **In simple words:** Load the model, optionally fake-quantize its weights to int8, convert dtype, compute perplexity, then time generation with the KV cache on or off, and return one dictionary of metrics.
- **Exactly:** `benchlib.load_causal_lm` loads fp32. For int8, `quantlib.qdq(w, bits=8, symmetric=True, gran="row")` quantizes each output channel with its own scale and dequantizes back to float (step 04); `conv1d_layers` finds GPT-2's four weight matrices per block. `model.to(dtype)` converts to bf16 if asked. `ql.perplexity` scores held-out lines (not the calibration split). The `run` function is a manual greedy loop: with `kv == "on"` it feeds only the newest token plus `past_key_values`; with `"off"` it re-feeds the whole growing sequence. `stamps` collects a timestamp after each token and `benchlib.summarize_trial` turns them into TTFT, ITL, tokens/s. We discard a warm-up `run(4)` and take the median of 3 trials. `benchlib.peak_rss_mb()` is read at the end of this fresh process.
- **Why it is here:** All the measuring logic lives in one function that runs in isolation, so every configuration is treated identically.

### 3. The orchestrator: run the matrix  (`main.py` lines 89–111)

```python
# ---- [S2] ORCHESTRATOR: run the matrix, one subprocess per configuration ---------------------------------------------------
def run_matrix(model, new_tokens, quick):
    dtypes, quants, batches, kvs = ["fp32", "bf16"], ["none", "int8"], [1, 4], ["on", "off"]
    if quick:
        dtypes, quants = ["fp32"], ["none", "int8"]
    # Pruned matrix (keeps the run under ~3 min): bf16 is tried only without quantization and WITH the cache. On CPUs
    # without native bf16 math, bf16 + no cache was ~1.3 s per token in development, which would dominate the run time.
    configs = [dict(model=model, dtype=d, quant=q, batch=b, kv=k, new_tokens=new_tokens)
               for d, q, b, k in itertools.product(dtypes, quants, batches, kvs)
               if not (d == "bf16" and (q != "none" or k == "off" or b != 1))]
    print(f"running {len(configs)} configs, one subprocess each ...", flush=True)
    rows = []
    for c in configs:
        r = subprocess.run([sys.executable, __file__, "--worker", json.dumps(c)], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        if r.returncode != 0 or not line:
            print(f"  FAILED {c}: {r.stderr[-300:]}")
            continue
        rows.append(json.loads(line[0][7:]))
        x = rows[-1]
        print(f"  {x['dtype']:<5}{x['quant']:<5}b={x['batch']} kv={x['kv']:<3} ttft {x['ttft_ms']:6.1f} ms  itl {x['itl_ms']:6.1f} ms  "
              f"{x['tok_s']:6.1f} tok/s  rss {x['rss_mb']:5.0f} MB  ppl {x['ppl']:.2f}", flush=True)
    return rows
```

- **In simple words:** Build the list of configurations, run each as `python main.py --worker '<json>'` in a subprocess, and collect the `RESULT` line each prints.
- **Exactly:** `itertools.product` makes the Cartesian product; the pruning `if` drops bf16 with int8 or without cache (bf16 without a cache was about 1.3 s per token on this CPU in development, which would dominate the run time) and bf16 at batch 4. `subprocess.run(..., capture_output=True, text=True)` runs the child and captures stdout/stderr; the parent keeps only the line starting with `RESULT ` and parses it with `json.loads`. A failed child prints the tail of its stderr and is skipped.
- **Why it is here:** The subprocess is what makes load time and peak RSS trustworthy.

### 4. The report: table, CSV, recommendation  (`main.py` lines 114–183)

```python
# ---- [S3] REPORT: table, CSV, recommendation -------------------------------------------------------------------------------
def recommend(rows, ppl_tolerance=0.02):
    base = min(r["ppl"] for r in rows if r["dtype"] == "fp32" and r["quant"] == "none")
    # quality gate: within 2% of the fp32 baseline.  Fake-int8 rows are excluded from the SPEED ranking: they run float math,
    # so any speed difference vs fp32 is measurement noise (reported separately as a quality / memory note).
    ok = [r for r in rows if r["ppl"] <= base * (1 + ppl_tolerance) and r["quant"] == "none"]
    thr = min((r for r in ok if r["batch"] == 4), key=lambda r: r["cost_per_mtok"])
    lat = min((r for r in ok if r["batch"] == 1), key=lambda r: r["ttft_ms"] + 16 * r["itl_ms"])   # TTFT + 16 decode steps
    return base, thr, lat


def describe(r):
    return f"{r['dtype']} / weights {r['quant']} / batch {r['batch']} / KV cache {r['kv']}"


def write_report(rows, args):
    import benchlib
    cols = ["dtype", "quant", "batch", "kv", "load_s", "ttft_ms", "itl_ms", "tok_s", "rss_mb", "ppl", "cost_per_mtok"]
    benchlib.to_csv([{k: r[k] for k in ["model", *cols]} for r in rows], HERE / "inference_report.csv")
    base, thr, lat = recommend(rows)
    nocache = [r for r in rows if r["kv"] == "off" and r["dtype"] == "fp32" and r["quant"] == "none" and r["batch"] == 1]
    cache = [r for r in rows if r["kv"] == "on" and r["dtype"] == "fp32" and r["quant"] == "none" and r["batch"] == 1]
    md = [f"# Inference report: {args.model} on CPU", "",
          f"Machine-dependent numbers; 4 torch threads; prompt {PROMPT_LEN} tokens, {args.new_tokens} new tokens, median of 3 trials; "
          f"cost uses an EXAMPLE price of ${PRICE_PER_HOUR}/hour (edit PRICE_PER_HOUR).", "",
          benchlib.to_markdown(rows, cols), "",
          "## Findings", ""]
    if cache and nocache:
        md.append(f"- **KV cache:** at batch 1 the cache gives {cache[0]['tok_s'] / nocache[0]['tok_s']:.1f}x the tokens/s of re-running the whole sequence each step.")
    b1 = [r for r in rows if r["batch"] == 1 and r["kv"] == "on" and r["dtype"] == "fp32" and r["quant"] == "none"]
    b4 = [r for r in rows if r["batch"] == 4 and r["kv"] == "on" and r["dtype"] == "fp32" and r["quant"] == "none"]
    if b1 and b4:
        md.append(f"- **Batching:** batch 4 gives {b4[0]['tok_s'] / b1[0]['tok_s']:.1f}x system throughput at {b4[0]['itl_ms'] / b1[0]['itl_ms']:.1f}x the per-step latency.")
    q = [r for r in rows if r["quant"] == "int8" and r["dtype"] == "fp32"]
    if q:
        md.append(f"- **int8 weight-only (fake):** perplexity {q[0]['ppl']:.2f} vs {base:.2f} for fp32, so quality is preserved; "
                  "speed and RSS are NOT improved here because the weights stay float (a real int8 kernel or GPU memory-bound decode would gain).")
    bf = [r for r in rows if r["dtype"] == "bf16" and r["quant"] == "none"]
    if bf:
        md.append(f"- **bf16 on CPU:** perplexity {bf[0]['ppl']:.2f}; speed vs fp32 depends on whether this CPU has native bf16 math (often slower on laptops).")
    md += ["", "## Recommendation", "",
           f"- **Throughput / cost pick (quality within 2% of fp32 perplexity {base:.2f}):** {describe(thr)} "
           f"-> {thr['tok_s']:.0f} tok/s, ${thr['cost_per_mtok']:.3f} per 1M output tokens (example price).",
           f"- **Interactive / latency pick:** {describe(lat)} -> TTFT {lat['ttft_ms']:.0f} ms, ITL {lat['itl_ms']:.1f} ms.",
           f"- **Memory add-on:** int8 weights cost only {100 * (q[0]['ppl'] / base - 1):.1f}% perplexity here and would shrink weights 4x vs fp32 "
           "(2x vs fp16) with a real int8 kernel: worth it when memory, not compute, is the limit." if q else "",
           "- Always keep the KV cache on; use the largest batch the latency target allows; quantize only after checking perplexity.",
           "", "## Caveats", "",
           "- Tiny model on CPU: decode is cheap here, so batching gains are smaller than on a GPU where decode is memory-bound.",
           "- Fake quantization measures quality only. Real speedups need kernels (GPTQ/AWQ/FP8 in vLLM, GGUF in llama.cpp).",
           "- Perplexity is measured on a few hundred tokens: differences below ~1% are noise."]
    (HERE / "inference_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-12:]))
    print("wrote inference_report.md and inference_report.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", help="internal: JSON config, measure it and print one RESULT line")
    ap.add_argument("--model", default="distilgpt2", choices=["distilgpt2", "gpt2", "sshleifer/tiny-gpt2"])
    ap.add_argument("--new-tokens", type=int, default=16)
    ap.add_argument("--quick", action="store_true", help="fp32 only (8 configs)")
    args = ap.parse_args()
    if args.worker:
        print("RESULT " + json.dumps(worker(json.loads(args.worker))))
    else:
        t = time.perf_counter()
        rows = run_matrix(args.model, args.new_tokens, args.quick)
        write_report(rows, args)
        print(f"total {time.perf_counter() - t:.0f} s")
```

- **In simple words:** Pick the winners (inside the quality gate), compute headline findings from the table, write CSV and Markdown, and print the summary.
- **Exactly:** `recommend` computes the best fp32 perplexity, keeps only unquantized rows within `ppl_tolerance` (fake-int8 speed numbers are excluded because they are noise), then picks the lowest $/1M at batch 4 (throughput) and the lowest TTFT + 16 x ITL at batch 1 (latency). `write_report` writes the CSV with `benchlib.to_csv`, builds the Markdown with `benchlib.to_markdown`, and computes ratios (cache speed-up, batching gain) from the rows so prose and numbers always agree. The `__main__` block parses flags, dispatches to the worker or the orchestrator, and prints total time.
- **Why it is here:** Generating the prose from the data is the habit that keeps a report honest.

### 5. Try it / break it  (`main.py` lines 185–189)

```python
# TRY IT: `--model gpt2` (bigger) or `--model sshleifer/tiny-gpt2` (tiny, load-time dominated); compare the recommendation.
# TRY IT: set PRICE_PER_HOUR=3.5 and note that $/1M scales linearly with price, so rankings do not change.
# TRY IT: change `ppl_tolerance` in recommend() to 0.001 and see which configs the quality gate rejects.
# BREAK IT: remove the quality gate in recommend(): the "fastest" config may be one that hurts perplexity.
# BREAK IT: measure RSS in one process for all configs instead of subprocesses: peak RSS only ever grows, so later rows are wrong.
```

- **In simple words:** Experiments and deliberate bugs.
- **Exactly:** comments only.
- **Why it is here:** Change one thing and predict the outcome first.

## Syntax explained

| Syntax | Meaning |
|---|---|
| `itertools.product(a, b, c)` | all combinations of the lists |
| `subprocess.run([sys.executable, __file__, "--worker", json.dumps(c)], capture_output=True, text=True)` | start a child Python process with the same interpreter and script, capture its output as text |
| `print("RESULT " + json.dumps(d))` / `line[0][7:]` | a tiny protocol: prefix, then JSON; strip the 7-character prefix to parse |
| `@torch.no_grad()` decorator | no autograd graph during inference (less memory, faster) |
| `sorted(..., key=lambda s: s["e2e_s"])[1]` | median of three: sort and take the middle |
| `min(iterable, key=...)` | the element with the smallest key (the recommendation) |
| `ap.add_argument("--quick", action="store_true")` | boolean flag |
| `if __name__ == "__main__":` | run only when executed, not imported |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `quantlib.qdq(w, bits, symmetric, gran)` | fake-quantize a matrix | per-row absmax scale, round to integers in [-127, 127], multiply back |
| `quantlib.perplexity(model, tok, text, ctx)` | held-out perplexity | windows of `ctx` tokens, mean cross-entropy, exp |
| `benchlib.summarize_trial` | TTFT, ITL, tok/s | differences of timestamps |
| `benchlib.peak_rss_mb` | peak memory | `resource.getrusage(RUSAGE_SELF).ru_maxrss`, with the macOS/Linux unit difference handled |
| `benchlib.to_csv / to_markdown` | tables | `csv.DictWriter` / string join |
| `subprocess.run` | child process | fork/exec, pipes for stdout and stderr, waits for exit |

## Execution trace (real run)

```
running 9 configs, one subprocess each ...
  fp32 none b=1 kv=on  ttft   25.3 ms  itl   13.5 ms    70.1 tok/s  rss   828 MB  ppl 42.53
  fp32 none b=1 kv=off ttft   27.3 ms  itl   32.0 ms    30.9 tok/s  rss   837 MB  ppl 42.53
  fp32 none b=4 kv=on  ttft   58.2 ms  itl   21.4 ms   165.6 tok/s  rss   860 MB  ppl 42.53
  fp32 none b=4 kv=off ttft   51.7 ms  itl   68.5 ms    56.5 tok/s  rss  1225 MB  ppl 42.53
  fp32 int8 b=1 kv=on  ttft   33.4 ms  itl   16.2 ms    56.1 tok/s  rss  1009 MB  ppl 42.62
  fp32 int8 b=1 kv=off ttft   20.7 ms  itl   26.1 ms    38.2 tok/s  rss  1011 MB  ppl 42.62
  fp32 int8 b=4 kv=on  ttft   42.8 ms  itl   17.2 ms   212.2 tok/s  rss  1087 MB  ppl 42.62
  fp32 int8 b=4 kv=off ttft   44.2 ms  itl   57.3 ms    72.2 tok/s  rss  1403 MB  ppl 42.62
  bf16 none b=1 kv=on  ttft 1009.0 ms  itl   35.7 ms    10.4 tok/s  rss   822 MB  ppl 42.84
## Recommendation

- **Throughput / cost pick (quality within 2% of fp32 perplexity 42.53):** fp32 / weights none / batch 4 / KV cache on -> 166 tok/s, $0.168 per 1M output tokens (example price).
- **Interactive / latency pick:** fp32 / weights none / batch 1 / KV cache on -> TTFT 25 ms, ITL 13.5 ms.
- **Memory add-on:** int8 weights cost only 0.2% perplexity here and would shrink weights 4x vs fp32 (2x vs fp16) with a real int8 kernel: worth it when memory, not compute, is the limit.
- Always keep the KV cache on; use the largest batch the latency target allows; quantize only after checking perplexity.

## Caveats

- Tiny model on CPU: decode is cheap here, so batching gains are smaller than on a GPU where decode is memory-bound.
- Fake quantization measures quality only. Real speedups need kernels (GPTQ/AWQ/FP8 in vLLM, GGUF in llama.cpp).
- Perplexity is measured on a few hundred tokens: differences below ~1% are noise.
wrote inference_report.md and inference_report.csv
total 93 s
```

The generated `inference_report.md` (table part) has the same numbers; see the file in this folder.

- Total about 90 to 165 seconds across runs on this laptop (the first run was 164 s before bf16 batch 4 was dropped); numbers vary by machine and run.
- KV cache on beats off by about 2x tokens/s at batch 1 and by about 3x at batch 4, where recomputing four growing sequences is even more wasteful (RSS is also higher without the cache: 1225 MB vs 860 MB).
- Batch 4 gives about 2.4x the system tokens/s of batch 1 for about 1.6x the per-step latency (ITL).
- int8 (fake) moved perplexity from 42.53 to 42.62 (+0.2%): quality kept. Its speed and RSS differ from fp32 only by noise or temporary copies (RSS is even higher because the quantized copies are made during loading): this is why the recommendation ignores fake-int8 speed.
- bf16 is much slower on this CPU (TTFT around 1 s, about 10 tokens/s): halving the bytes does not help without native bf16 math, and its perplexity is slightly higher (42.84).

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `--model gpt2` | longer load, lower tokens/s, similar ratios | more parameters, same shapes of trade-offs |
| `--model sshleifer/tiny-gpt2` | load time dominates, ppl very high | a toy model: nothing meaningful about quality |
| `PRICE_PER_HOUR=3.5` | all $/1M x 35 | cost is linear in price; ranking unchanged |
| `ppl_tolerance=0.001` in `recommend` | int8 and bf16 rows fail the gate | their perplexity moved more than 0.1% |
| `--new-tokens 64` | KV-off penalty grows | the recomputed sequence gets longer every step |
| Re-run the same command twice | table values wobble a few percent | machine noise: why we use medians |

## Debugging practice

**BREAK IT:** remove the quality gate (`ok = rows`).
- **Symptom:** no crash, but a lower-quality row could win the recommendation. With only fake int8 and fp32 the winner may be a row whose "speedup" is noise.
- **Fix:** keep the gate and also exclude simulated rows from speed rankings.

**BREAK IT:** measure all configurations in one process (call `worker` in a loop).
- **Symptom:** `rss_mb` is non-decreasing down the table and load times shrink after the first row (the files are cached).
- **Fix:** subprocess per configuration, as in the real code.

**Real-world error:** `FAILED {...}: ... OSError` printed by the orchestrator means the child could not load the model (for example the model is not in the offline cache). The parent continues; fix the model name or cache.

## Common mistakes

- Reporting a single run instead of a median of repeated trials.
- Claiming a speedup from simulated quantization.
- Measuring peak RSS across several configs in one process.
- Comparing configurations with different prompt lengths or thread counts.
- Skipping the quality metric (a faster model that is wrong is not a win).
- Presenting laptop-CPU numbers as GPU serving numbers.
- Leaving out the caveats (the rubric in the README treats honesty as a scoring criterion).

## Rebuild checklist

1. Define the configuration axes and the metric list.
2. Write `worker(cfg)`: load, optional quantization, perplexity, warm-up, 3 timed trials, median.
3. Write the KV on/off decode loop with a timestamp per token.
4. Write the orchestrator: `itertools.product`, `subprocess.run`, parse `RESULT`.
5. Write `recommend` with a quality gate.
6. Generate Markdown and CSV from the rows; compute findings from the data.
7. Add caveats and run it twice to see the noise.

## Say it aloud

- "I benchmarked distilgpt2 over dtype, weight quantization, batch size and KV cache, one process per configuration so that peak memory and load time are clean."
- "The KV cache was the largest single win, batching raised throughput but also per-step latency, and fake int8 kept perplexity within 0.2%, though I did not claim a speed-up because it is simulated."
- "My recommendation is gated on perplexity and split into a throughput pick and a latency pick, and I state the caveats: tiny model, CPU, example prices."
- "On a GPU I would redo it with vLLM or SGLang, real AWQ/FP8 weights and prefix caching, and price it with the cost model from step 16."

## The new problem (cliffhanger)

The module is complete: you can load, quantize, size, benchmark, batch, cache, parallelise, choose an engine and price it. What remains is the system around the model. The API layer that wraps an engine is in [module 09](../../09_api_development/README.md), the architecture around it in [module 10](../../10_ml_system_architecture/README.md), packaging in [module 11](../../11_docker/README.md), and the full product in [module 13](../../13_capstone_project/README.md). Back to the [module 21 README](../README.md).
