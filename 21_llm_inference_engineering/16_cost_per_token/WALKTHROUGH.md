# Step 16 — Cost per token: turning tokens/s into dollars

## The story so far
[Step 15](../15_serving_engines_landscape/WALKTHROUGH.md) showed which engine pulls which performance lever. Each lever (batching, quantization, prefix caching, parallelism) changed a number you measured: tokens per second, memory, latency. Money is the one number your manager and your interviewer ultimately care about.

## The pain point
"Self-host or use an API?" gets answered with feelings. A GPU rented at a fixed price per hour costs the same whether it is busy or idle, an API charges per token, and the two sides of a token (input and output) cost very different amounts of GPU time. Without a model you cannot say when self-hosting wins, or what a 4-bit quantization or a prefix cache is worth in dollars.

## The spark
Cost is a unit conversion. Dollars per hour divided by tokens per hour is dollars per token. Utilisation (how often the GPU is busy) divides the denominator. Input and output are separate because prefill is compute-bound and fast per token, while decode is memory-bound and slow per token. Everything else (batching, quantization, caching) is just a way to raise tokens per second or lower the tokens you must compute.

## Goal
Compute $/1M tokens from a measured throughput, with utilisation and a separate input/output price; compute per-request cost; find the self-host vs API break-even volume; quantify batching, quantization (int4 weights, fp8 KV) and prefix caching; and know which numbers in the model are measured and which are estimates.

## The theory (read this before the interview)

```
$/1M tokens         = ($ per GPU-hour) / (tokens/s x 3600 x utilisation) x 1,000,000
tokens/s            = measured with benchlib (step 10): prefill tok/s for input, decode tok/s for output
request cost        = (uncached_in x P_in + cached_in x P_cached + out x P_out) / 1,000,000
self-host fixed     = n_GPUs x $/hour x 730 hours/month          (paid even when idle)
break-even volume   = fixed monthly cost / API blended $ per token = (n x $/h x 730) / blended $ per 1M   (in M tokens/month)
blended API price   = share_in x P_in + share_out x P_out        (RAG/chat is often ~80-90% input tokens)
```

**Why input is cheaper than output.** Prefill processes all prompt tokens in parallel in big matrix multiplications (compute-bound, thousands of tokens/s per GPU for small models). Decode produces one token per step and must read all weights and the KV cache every step (memory-bound, see [step 09](../09_prefill_decode_roofline/WALKTHROUGH.md)); batching amortises the weight reads across users. So a GPU makes far more input tokens than output tokens per second, and API prices reflect that (output is typically several times more expensive than input).

**The levers and what they do to cost.**

| Lever | Mechanism | Effect on $/1M |
|---|---|---|
| Bigger batch (continuous batching, [step 12](../12_continuous_batching_prefix_caching/WALKTHROUGH.md)) | weight reads shared by more users | decode cost falls roughly in proportion until compute- or KV-bound; latency rises |
| Weight quantization ([04](../04_quantization_math/WALKTHROUGH.md)) | fewer bytes read per step | helps most at small batch (memory-bound); less at large batch where KV and compute dominate |
| KV-cache quantization ([07](../07_kv_cache_quantization/WALKTHROUGH.md)) | smaller cache per token | helps at large batch x long context |
| Prefix caching (step 12) | skip prefill for repeated prompt prefixes | input cost falls with hit rate; API vendors price cached input at a discount |
| Higher utilisation | fewer idle paid hours | cost falls as 1/utilisation: traffic shape and autoscaling matter |
| Faster GPU | more tokens/s but higher $/h | only wins if tokens/s rises more than the price |

**Self-host vs API.** An API is a variable cost (pay per token, zero when idle). A GPU is a fixed cost. So there is a volume above which owning wins; below it the API wins, and the break-even ignores engineers' time, on-call, and redundancy, which often decide the question in the API's favour for small teams. Reasons to self-host regardless of price: data residency/privacy, latency control, a fine-tuned model (module 17), or no API for that model.

**Where the numbers come from in this step.**

| Number | Source | Status |
|---|---|---|
| distilgpt2 prefill and decode tokens/s on this CPU | **measured** with `benchlib` | real, machine-dependent |
| 8B-on-H100 tokens/s | **roofline estimate** from the [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md) KV formula and example bandwidth/FLOPs | model, not a benchmark |
| GPU $/hour and API $/1M tokens | **example values in two editable dicts** | volatile: check current pricing |

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| $/1M tokens | cost per million tokens, price per Mtok | Yes | The standard unit for API and self-host comparison. |
| Input / prompt tokens | prefill tokens | Yes | Tokens you send; priced lower. |
| Output / completion tokens | decode tokens, generated tokens | Yes | Tokens produced; priced higher. |
| Cached input | prompt caching, prefix-cache hit, cache read | Mostly | Input tokens whose KV is reused; billed at a discount. |
| Utilisation | GPU busy fraction, load factor, MFU (different!) | **No** | Utilisation = time busy; MFU (model FLOPs utilisation) = fraction of peak FLOPs achieved while busy. |
| On-demand / spot / reserved | pricing tiers | Different | Same GPU, different price and availability guarantees. |
| TCO | total cost of ownership | Broader | Adds engineers, networking, redundancy, idle time. |
| Goodput | useful throughput meeting the SLO | Related | Throughput counting only requests inside the latency target (step 10). |

## Run it

```
cd 21_llm_inference_engineering/16_cost_per_token
python main.py                                   # THREADS=2 NEW_TOKENS=32 python main.py  to change the measurement
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/16_cost_per_token/main.py    # from repo root
```

About 25 seconds on CPU (it loads distilgpt2 and times it; use the global lock in this repo's workflow). Section [1] and sections [3] to [5] are arithmetic only.

## Code, step by step

### 1. Header, imports and `benchlib`  (`main.py` lines 1–26)

```python
"""
GOAL:  Turn throughput into money. Cost model: $/1M tokens = GPU $/hour / (tokens/s * 3600) * 1e6, with utilisation, separate
       input (prefill) and output (decode) prices, per-request cost, self-host vs API break-even, and the effect of
       batching, quantization and prefix caching.  Throughput comes from a REAL local measurement (../benchlib.py).

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/16_cost_per_token/main.py
EXPECTED OUTPUT (shortened; timings vary by machine):
    [1] pure formula examples ...
    [2] REAL measurement distilgpt2 on this CPU: prefill tok/s ..., decode tok/s at batch 1 / 8 ...
        $/1M output tokens at batch 1 vs 8 (example CPU price) ...
    [3] estimated 8B model on an example GPU: fp16 vs int4 weights, batch sweep (roofline model, NOT measured)
    [4] utilisation / prefix-cache hit rate effect ...   [5] break-even: ... M tokens/month
NEEDS: torch, transformers (distilgpt2 cached; no downloads).  ~25 s on CPU. Timing run: use the global lock.
!! ALL PRICES BELOW ARE EXAMPLE VALUES. GPU and API prices change often: edit the two dicts and check current pricing. !!
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find ../benchlib.py (built in step 10 era; shared helper)
import benchlib
```

- **In simple words:** Standard setup, plus an import of the shared timing helper from step 10's family.
- **Exactly:** `HF_HUB_OFFLINE=1` stops any download. `sys.path.insert(0, parent)` makes `import benchlib` find `21_llm_inference_engineering/benchlib.py`. The docstring states in capitals that all prices are examples.
- **Why it is here:** The cost model must stand on a measurement, not on invented speeds, and the module has one timing helper.

### 2. Editable price tables  (`main.py` lines 28–42)

```python
# ---- [S1] Editable price tables: EXAMPLE PRICES -- CHECK CURRENT PRICING ---------------------------------------------
# GPU rental $/hour (on-demand cloud list prices differ by provider and change; spot/reserved are cheaper). Examples only.
GPU_PRICE_PER_HOUR = {            # example prices -- check current pricing
    "laptop-cpu (example)": 0.10,
    "L4-24GB (example)": 0.80,
    "A100-80GB (example)": 2.00,
    "H100-80GB (example)": 3.50,
}
# API prices in $ per 1M tokens: (input, output, cached input). Invented round numbers in the usual ratio
# (output costs several x input; cached input is a fraction). example prices -- check current pricing
API_PRICE_PER_MTOK = {
    "small-model API (example)": (0.15, 0.60, 0.04),
    "mid-model API (example)": (1.00, 4.00, 0.25),
    "frontier-model API (example)": (5.00, 20.00, 1.25),
}
```

- **In simple words:** Two dictionaries of made-up round prices. Change them to current prices before you trust any dollar figure.
- **Exactly:** GPU prices are `$ per hour`; API prices are `(input, output, cached input)` in `$ per 1M tokens`. They are **examples**: provider prices differ and change often, so the file makes no claim about today's real prices.
- **Why it is here:** Separating assumptions (prices) from logic (formulas) lets you re-run the whole analysis with your own numbers.

### 3. The core formulas  (`main.py` lines 45–61)

```python
# ---- [S2] The core formulas ------------------------------------------------------------------------------------------
def cost_per_mtok(gpu_per_hour, tokens_per_s, utilisation=1.0):
    """$/1M tokens = ($/hour) / (tokens/s * 3600 s) * 1e6, divided by the fraction of time the GPU does useful work.
    You pay for the GPU every hour, busy or idle, so idle time inflates the price of the tokens you did produce."""
    return gpu_per_hour / (tokens_per_s * 3600 * utilisation) * 1e6


def request_cost(in_tokens, out_tokens, in_price, out_price, cached_tokens=0, cached_price=None):
    """Per-request cost with different prices for input, cached input, and output tokens (prices in $/1M tokens)."""
    cached_price = in_price if cached_price is None else cached_price
    return ((in_tokens - cached_tokens) * in_price + cached_tokens * cached_price + out_tokens * out_price) / 1e6


print("[1] pure formula (example price $3.50/h):")
for tps in (500, 2000, 8000):
    print(f"      {tps:>5} tok/s at 100% busy -> ${cost_per_mtok(3.50, tps):6.2f}/1M tok | at 30% busy -> ${cost_per_mtok(3.50, tps, 0.3):6.2f}/1M tok")
assert abs(cost_per_mtok(3.6, 1000) - 1.0) < 1e-9          # $3.6/h at 1000 tok/s = 3.6M tokens/h -> exactly $1 per 1M tokens
```

- **In simple words:** `cost_per_mtok` turns "$ per hour and tokens per second" into "$ per million tokens". `request_cost` prices one request with three token types. The assert checks a hand-calculable case.
- **Exactly:** `tokens_per_s * 3600` is tokens per hour; dividing by `utilisation` (0 to 1) inflates cost when the GPU idles; `* 1e6` rescales to millions. `3.6 $/h` at `1000 tok/s` is 3.6M tokens/hour, so exactly `$1` per 1M tokens: the assert proves the function.
- **Why it is here:** Two lines of arithmetic are the whole model; the rest is feeding it good numbers.

### 4. A real measurement on this machine  (`main.py` lines 63–89)

```python
# ---- [S3] REAL measurement on this machine (distilgpt2, CPU) ---------------------------------------------------------
# Prefill (input) and decode (output) have different speeds, so they get different cost per token. We time each separately.
threads = int(os.environ.get("THREADS", 4))
model, tok = benchlib.load_causal_lm("distilgpt2", threads=threads)
torch.manual_seed(0)
PROMPT, NEW, TRIALS = 128, int(os.environ.get("NEW_TOKENS", 24)), 3


def measure(batch):
    ids = torch.randint(100, 5000, (batch, PROMPT))
    benchlib.timed_generate(model, ids, 4)                              # warm-up (discarded)
    runs = [benchlib.summarize_trial(benchlib.timed_generate(model, ids, NEW)) for _ in range(TRIALS)]
    runs.sort(key=lambda r: r["e2e_s"])
    mid = runs[len(runs) // 2]                                          # median trial: robust against one slow outlier
    return dict(batch=batch, prefill_tps=batch * PROMPT / mid["prefill_s"], decode_tps=mid["decode_tok_per_s"])


meas = {b: measure(b) for b in (1, 8)}
print(f"[2] REAL measurement: distilgpt2, CPU, {threads} threads, prompt {PROMPT} tokens, {NEW} new tokens (median of {TRIALS})")
price = GPU_PRICE_PER_HOUR["laptop-cpu (example)"]
for b, m in meas.items():
    print(f"      batch {b}: prefill {m['prefill_tps']:8.0f} tok/s -> ${cost_per_mtok(price, m['prefill_tps']):6.2f}/1M input"
          f" | decode {m['decode_tps']:7.0f} tok/s -> ${cost_per_mtok(price, m['decode_tps']):6.2f}/1M output")
ratio = cost_per_mtok(price, meas[1]["decode_tps"]) / cost_per_mtok(price, meas[8]["decode_tps"])
print(f"      batching 1 -> 8 makes a decoded token {ratio:.1f}x cheaper (the hardware is rented either way)")
print(f"      output tokens cost {cost_per_mtok(price, meas[1]['decode_tps']) / cost_per_mtok(price, meas[1]['prefill_tps']):.1f}x"
      " more than input tokens here (same reason APIs price output higher)")
```

- **In simple words:** Time distilgpt2 at batch 1 and batch 8, separately for prefill (input) and decode (output), and price both at an example CPU rate.
- **Exactly:** `benchlib.load_causal_lm` loads the cached model with `threads` CPU threads. `timed_generate` runs a greedy decode loop with a KV cache and records a timestamp per token; `summarize_trial` turns timestamps into `prefill_s` and `decode_tok_per_s`. We take the median of 3 trials (sorted by end-to-end time) after one warm-up. Prefill tokens/s is `batch x prompt_len / prefill_s`.
- **Why it is here:** It shows the two key facts with real numbers: batching makes decoded tokens cheaper, and output tokens cost more than input tokens.

### 5. Estimated GPU economics for an 8B model  (`main.py` lines 92–113)

```python
# ---- [S4] Estimated GPU economics for an 8B model (roofline model, NOT measured) -----------------------------------------
# Decode step time ~ max(memory time, compute time).  memory = (weights + KV of all sequences)/bandwidth;
# compute = 2*params*batch / FLOPs.  KV bytes per token per sequence: 2*layers*kv_heads*head_dim*bytes (formula from step 03).
def est_decode_tps(batch, params=8e9, w_bytes=2.0, ctx=2048, layers=32, kv_heads=8, head_dim=128, kv_bytes=2.0,
                   bw=3.35e12, flops=0.99e15, eff=0.5):
    """Example hardware: ~3.35 TB/s HBM, ~1e15 dense fp16 FLOP/s (H100-class), eff = fraction actually achieved. Illustrative."""
    kv = 2 * layers * kv_heads * head_dim * kv_bytes * ctx * batch
    t_mem = (params * w_bytes + kv) / (bw * eff)
    t_cmp = 2 * params * batch / (flops * eff)
    return batch / max(t_mem, t_cmp)


h100 = GPU_PRICE_PER_HOUR["H100-80GB (example)"]
print("[3] ESTIMATE, 8B model on an example H100-class GPU (roofline, ctx 2048, 50% efficiency; not a measurement):")
for label, wb in (("fp16 weights", 2.0), ("int4 weights", 0.5)):
    row = []
    for b in (1, 16, 64):
        tps = est_decode_tps(b, w_bytes=wb)
        row.append(f"b={b}: {tps:6.0f} tok/s ${cost_per_mtok(h100, tps):6.2f}")
    print(f"      {label}: " + " | ".join(row) + "   ($/1M output tokens)")
print("      fp8 KV cache halves the KV term (matters at big batch x long context):",
      f"b=64 fp16-KV ${cost_per_mtok(h100, est_decode_tps(64)):.2f} -> fp8-KV ${cost_per_mtok(h100, est_decode_tps(64, kv_bytes=1.0)):.2f}")
```

- **In simple words:** A back-of-envelope model of an 8B model on an H100-class card: how many tokens per second at each batch size, in fp16 and int4 weights, and the resulting dollars.
- **Exactly:** decode step time is the larger of a memory time (read all weights and all sequences' KV caches at `bw x eff`) and a compute time (`2 x params x batch` FLOPs at `flops x eff`). Tokens/s = `batch / step_time`. The KV bytes use the [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md) formula `2 x layers x kv_heads x head_dim x bytes x tokens x batch`. Bandwidth, FLOPs and efficiency are illustrative round numbers.
- **Why it is here:** You cannot measure an H100 on a laptop; a transparent model lets you reason about int4 and fp8-KV effects. Say clearly that this is an estimate, not a benchmark.

### 6. Utilisation, prefill vs decode, prefix caching  (`main.py` lines 115–127)

```python
# ---- [S5] Utilisation, input/output mix and prefix caching ----------------------------------------------------------
tps_dec = est_decode_tps(64)
tps_pre = 0.99e15 * 0.5 / (2 * 8e9)               # prefill is compute-bound: tokens/s ~ FLOPs / (2*params)
print("[4] one chat request: 3000 input tokens (2500 are a shared system prompt + history), 300 output tokens, example H100 price")
print(f"      est. prefill {tps_pre:,.0f} tok/s, decode (batch 64) {tps_dec:,.0f} tok/s per GPU")
for hit in (0.0, 0.5, 0.83):
    # A prefix-cache hit skips prefill compute for the cached tokens, so only (1-hit) of the input needs computing.
    # tps_dec is the AGGREGATE decode rate of the batch, so 300 tokens at that rate is this request's share of GPU time.
    gpu_seconds = 3000 * (1 - hit) / tps_pre + 300 / tps_dec
    dollars = gpu_seconds / 3600 * h100
    print(f"      prefix-cache hit {hit:>4.0%}: GPU time {gpu_seconds * 1000:7.2f} ms -> ${dollars:.6f}/request (${dollars * 1e6:,.0f} per 1M requests)")
for u in (1.0, 0.5, 0.2):
    print(f"      utilisation {u:>4.0%}: effective ${cost_per_mtok(h100, tps_dec, u):7.2f}/1M output tokens")
```

- **In simple words:** Price one chat request with a long shared prompt, at three prefix-cache hit rates, and show what idle time does to the output-token price.
- **Exactly:** prefill is compute-bound, so its tokens/s is about `FLOPs x eff / (2 x params)`. A cache hit means those input tokens need no prefill compute: GPU seconds = `uncached_in / prefill_tps + out / decode_tps`. `decode_tps` is the *aggregate* decode rate, so `out / decode_tps` is this request's share of the GPU.
- **Why it is here:** It converts prefix caching and utilisation into dollars.

### 7. Self-host vs API break-even  (`main.py` lines 129–142)

```python
# ---- [S6] Self-host vs API break-even ----------------------------------------------------------------------------------
def break_even_mtok_per_month(gpus, gpu_per_hour, api_blended_per_mtok, hours=730):
    """Monthly volume (M tokens) above which owning `gpus` always-on GPUs beats paying the API per token.
    Fixed cost = gpus * $/h * 730 h (you pay even when idle). Ignores engineers' time, which is often the largest cost."""
    return gpus * gpu_per_hour * hours / api_blended_per_mtok


in_p, out_p, _ = API_PRICE_PER_MTOK["mid-model API (example)"]
blended = (0.8 * in_p + 0.2 * out_p)                      # 80% input / 20% output token mix, typical for RAG/chat
be = break_even_mtok_per_month(1, h100, blended)
cap = tps_dec * 3600 * 730 / 1e6                           # M output tokens one GPU can make per month at 100% busy
print(f"[5] break-even vs 'mid' example API (blended ${blended:.2f}/1M, 80/20 in/out): 1 always-on H100 = ${h100 * 730:,.0f}/month")
print(f"      -> self-hosting wins above ~{be:,.0f}M tokens/month; one GPU's capacity is ~{cap:,.0f}M output tokens/month at 100% busy")
print("      below that, the API is cheaper (you pay only for tokens, not for idle hours, ops or on-call time)")
```

- **In simple words:** Monthly GPU cost divided by what the API would charge per token gives the monthly volume where both cost the same.
- **Exactly:** `gpus x $/h x 730` is the always-on monthly cost. `blended` is the API price for an 80/20 input/output token mix. Break-even tokens (in millions) = fixed cost / blended price. The capacity line checks that one GPU can actually produce that volume at 100% busy.
- **Why it is here:** It gives the interview answer "self-hosting wins above roughly X tokens per month, provided the GPU is kept busy and ops cost is small".

### 8. Try it / break it  (`main.py` lines 144–148)

```python
# TRY IT: edit GPU_PRICE_PER_HOUR to your cloud's real price and re-run [3]-[5]; THREADS=1 or 2 changes the CPU numbers.
# TRY IT: change the 80/20 input/output mix to 50/50 and see how the break-even moves.
# TRY IT: set est_decode_tps(..., ctx=16384) and see why long context at big batch is expensive (KV term).
# BREAK IT: set utilisation to 0 in cost_per_mtok -> ZeroDivisionError; an idle GPU has infinite cost per token.
# BREAK IT: price prefill and decode with the SAME tokens/s (use decode for both): the input price comes out far too high.
```

- **In simple words:** Experiments and deliberate bugs.
- **Exactly:** comments only.
- **Why it is here:** Change one assumption and watch which conclusion moves.

## Syntax explained

| Syntax | Meaning |
|---|---|
| `os.environ.get("THREADS", 4)` | read an env-var knob with a default |
| `{k: v for ...}` / `{b: measure(b) for b in (1, 8)}` | dict comprehension: batch size to its measurement |
| `runs.sort(key=lambda r: r["e2e_s"])` | sort dicts by one field; `runs[len(runs)//2]` is the median |
| `f"{x:>9.2f}"`, `f"{hit:>4.0%}"` | right-align, 2 decimals / percentage format |
| `1e6`, `0.99e15` | scientific notation floats |
| `max(t_mem, t_cmp)` | the slower of the two limits wins (roofline logic) |

## Libraries & functions

| Function | What it does | Internally |
|---|---|---|
| `benchlib.load_causal_lm(name, threads=)` | load a cached HF model | `AutoModelForCausalLM.from_pretrained(..., dtype=)`, `.eval()`, sets torch threads |
| `benchlib.timed_generate(model, ids, n)` | greedy decode with timestamps | one prefill forward, then `n-1` single-token forwards using the KV cache; stamps after each |
| `benchlib.summarize_trial(trial)` | TTFT, TPOT, tok/s from stamps | differences of timestamps; decode tok/s = `b (n-1) / (E2E - TTFT)` |
| `torch.randint(100, 5000, (b, L))` | random token ids | avoids tokenizer padding issues; speed depends on shapes, not words |

## Execution trace (real run)

```
[1] pure formula (example price $3.50/h):
        500 tok/s at 100% busy -> $  1.94/1M tok | at 30% busy -> $  6.48/1M tok
       2000 tok/s at 100% busy -> $  0.49/1M tok | at 30% busy -> $  1.62/1M tok
       8000 tok/s at 100% busy -> $  0.12/1M tok | at 30% busy -> $  0.41/1M tok

[2] REAL measurement: distilgpt2, CPU, 4 threads, prompt 128 tokens, 24 new tokens (median of 3)
      batch 1: prefill     2215 tok/s -> $  0.01/1M input | decode      65 tok/s -> $  0.43/1M output
      batch 8: prefill     3518 tok/s -> $  0.01/1M input | decode     322 tok/s -> $  0.09/1M output
      batching 1 -> 8 makes a decoded token 5.0x cheaper (the hardware is rented either way)
      output tokens cost 34.3x more than input tokens here (same reason APIs price output higher)
[3] ESTIMATE, 8B model on an example H100-class GPU (roofline, ctx 2048, 50% efficiency; not a measurement):
      fp16 weights: b=1:    103 tok/s $  9.44 | b=16:   1321 tok/s $  0.74 | b=64:   3231 tok/s $  0.30   ($/1M output tokens)
      int4 weights: b=1:    392 tok/s $  2.48 | b=16:   3231 tok/s $  0.30 | b=64:   5061 tok/s $  0.19   ($/1M output tokens)
      fp8 KV cache halves the KV term (matters at big batch x long context): b=64 fp16-KV $0.30 -> fp8-KV $0.22
[4] one chat request: 3000 input tokens (2500 are a shared system prompt + history), 300 output tokens, example H100 price
      est. prefill 30,938 tok/s, decode (batch 64) 3,231 tok/s per GPU
      prefix-cache hit   0%: GPU time  189.82 ms -> $0.000185/request ($185 per 1M requests)
      prefix-cache hit  50%: GPU time  141.34 ms -> $0.000137/request ($137 per 1M requests)
      prefix-cache hit  83%: GPU time  109.34 ms -> $0.000106/request ($106 per 1M requests)
      utilisation 100%: effective $   0.30/1M output tokens
      utilisation  50%: effective $   0.60/1M output tokens
      utilisation  20%: effective $   1.50/1M output tokens
[5] break-even vs 'mid' example API (blended $1.60/1M, 80/20 in/out): 1 always-on H100 = $2,555/month
      -> self-hosting wins above ~1,597M tokens/month; one GPU's capacity is ~8,491M output tokens/month at 100% busy
      below that, the API is cheaper (you pay only for tokens, not for idle hours, ops or on-call time)
```

- Section [2] is measured on this laptop and varies run to run (two runs gave decode 73 and 65 tok/s at batch 1): the *ratios* are what to trust. Batching 1 to 8 made a decoded token about 5x cheaper because the weights are read once per step regardless of batch.
- Output tokens cost about 35x more than input tokens here because decode (tens to hundreds of tok/s) is far slower than prefill (thousands): the same logic behind API output prices being several times input prices.
- Section [3] is an estimate: int4 weights help most at batch 1 (about 4x cheaper) and much less at batch 64 (about 1.6x), because at large batch the KV cache and compute, not the weights, dominate. fp8 KV helps only when the KV term is large.
- Prefix caching at an 83% hit rate cut the per-request GPU time and cost by about 40% in this example; low utilisation multiplies the price by 1/utilisation (20% busy = 5x).
- Break-even for the example prices is on the order of a billion-plus tokens per month: below that the pay-per-token API wins on cost alone.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `THREADS=1` | lower tok/s, higher $/1M | fewer cores, same rent |
| `NEW_TOKENS=64` | per-token numbers barely move | the steady-state decode rate dominates |
| `GPU_PRICE_PER_HOUR["H100-80GB (example)"] = 2.0` | all GPU dollar values scale by 2.0/3.5 | cost is linear in price; break-even moves with it |
| `ctx=16384` in `est_decode_tps` | big-batch cost jumps | the KV term grows with context x batch |
| 50/50 input/output mix | blended API price rises, break-even volume falls | output tokens are pricier |
| utilisation 0.2 | cost x5 | you pay for idle hours |

## Debugging practice

**BREAK IT:** `cost_per_mtok(3.5, 1000, 0)`.
- **Symptom:** `ZeroDivisionError: float division by zero`.
- **Fix:** a GPU that is never busy has no finite cost per token; clamp utilisation to a small positive number or fix the traffic assumption.

**BREAK IT:** price input tokens with the *decode* tokens/s.
- **Symptom:** the input price comes out about 35x too high, and a long-prompt RAG workload looks far more expensive than it is.
- **Fix:** time prefill and decode separately (as `measure()` does) and price each on its own throughput.

## Common mistakes

- Using peak benchmark throughput at 100% utilisation as the production price: real traffic is bursty.
- Pricing input and output tokens the same.
- Comparing a self-host GPU price with an API price while ignoring engineers, monitoring, redundancy and idle time.
- Using a quantized model's price without checking its quality (step 17 / step 04 perplexity).
- Quoting GPU or API prices from memory: they are volatile; say "example prices" and check the provider.
- Confusing utilisation with MFU.

## Rebuild checklist

1. Write `cost_per_mtok(gpu_per_hour, tokens_per_s, utilisation)`.
2. Check it with `$3.6/h at 1000 tok/s = $1/1M`.
3. Write `request_cost` with uncached, cached and output prices.
4. Time prefill and decode separately with `benchlib` at two batch sizes.
5. Write `est_decode_tps` as `batch / max(memory time, compute time)` with the KV formula.
6. Loop prefix-cache hit rates and utilisations.
7. Write `break_even_mtok_per_month(gpus, price, blended)`.

## Say it aloud

- "Cost per million tokens is GPU dollars per hour divided by tokens per hour, divided again by utilisation, because I pay for idle time."
- "Input and output are priced separately because prefill is compute-bound and fast per token while decode is memory-bound and slow; batching is what makes decode cheap."
- "Quantization helps cost most at small batch where decode is memory-bound; prefix caching cuts input cost by the hit rate; both are measured against a quality check."
- "Self-hosting is a fixed monthly cost, so it wins above a break-even volume, but only if I keep the GPU busy and count the engineers."

## The new problem (cliffhanger)

You now have every piece: formats, quantization, memory, latency, batching, caching, speculation, parallelism, engines and cost. Each was studied in isolation. A hiring manager wants to see them *combined* on one model, with measurements and a recommendation. On to [step 17 — the inference report project](../17_inference_report_project/WALKTHROUGH.md).
