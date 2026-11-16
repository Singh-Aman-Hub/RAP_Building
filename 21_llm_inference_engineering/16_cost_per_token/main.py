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

# TRY IT: edit GPU_PRICE_PER_HOUR to your cloud's real price and re-run [3]-[5]; THREADS=1 or 2 changes the CPU numbers.
# TRY IT: change the 80/20 input/output mix to 50/50 and see how the break-even moves.
# TRY IT: set est_decode_tps(..., ctx=16384) and see why long context at big batch is expensive (KV term).
# BREAK IT: set utilisation to 0 in cost_per_mtok -> ZeroDivisionError; an idle GPU has infinite cost per token.
# BREAK IT: price prefill and decode with the SAME tokens/s (use decode for both): the input price comes out far too high.
