"""
benchlib.py -- small, reusable LLM-inference benchmarking helpers (module 21).

GOAL:  One honest way to time token generation, shared by steps 09, 10, 12, 13, 16, 17.
       Everything runs on CPU (and works on CUDA if you pass a cuda model).

WHAT IT GIVES YOU
    load_causal_lm(name)            load a cached HF causal LM + tokenizer (offline, eval mode)
    timed_generate(model, ids, n)   greedy manual decode loop with a KV cache that records a timestamp
                                    per generated token; prefill is timed separately
    summarize_trial(trial)          TTFT, ITL list, TPOT, E2E latency, output tok/s for ONE run
    run_trials(...)                 warm-up runs (discarded) + repeated timed trials
    aggregate(trials)               mean/std over trials + pooled p50/p95/p99 of ITL and E2E
    peak_rss_mb()                   peak resident memory (macOS reports bytes, Linux kilobytes)
    to_csv(rows, path) / to_markdown(rows)   results table helpers

METRIC DEFINITIONS (names side by side)
    TTFT  (time to first token)        = start -> first token out  = prefill + first sampling step
    ITL   (inter-token latency)        = gap between two consecutive output tokens (a list, one per gap)
    TPOT  (time per output token)      = (E2E - TTFT) / (n_tokens - 1)  = mean ITL (decode only, TTFT excluded)
    E2E   (end-to-end latency)         = start -> last token out
    tok/s (output throughput)          = n_tokens * batch / E2E   (system-wide for the batch)

RUN (self-test):   HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/benchlib.py
NEEDS: torch, transformers (cached sshleifer/tiny-gpt2).  ~5 s on CPU.
"""
import csv
import os
import resource  # built-in (Unix): this process's peak memory
import statistics
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # never try to download anything
import torch


# ---------------------------------------------------------------- memory + clock helpers
def peak_rss_mb() -> float:
    """Peak resident set size of THIS process in MB. ru_maxrss is BYTES on macOS but KILOBYTES on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def sync(device) -> None:
    """GPU kernels are queued asynchronously: wait before reading the clock. No-op on CPU."""
    t = torch.device(device).type if not isinstance(device, torch.device) else device.type
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "mps":
        torch.mps.synchronize()


def percentile(values, q: float) -> float:
    """q in [0, 100]; linear interpolation (same as numpy's default). Works on a plain list."""
    xs = sorted(values)
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100
    lo, hi = int(pos), min(int(pos) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# ---------------------------------------------------------------- model loading
def load_causal_lm(name: str = "distilgpt2", dtype=torch.float32, threads: int | None = None):
    """Return (model, tokenizer) from the local HF cache. Heavy models: call under the global lock (see brief)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if threads:
        torch.set_num_threads(threads)           # thread count changes CPU timings a lot: always report it
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).eval()
    except OSError:
        # Some models (e.g. Qwen/Qwen2.5-0.5B on this laptop) have only config+tokenizer cached, no weight file.
        # Speed/memory depend on SHAPES, not on weight values, so random weights from the config time identically.
        from transformers import AutoConfig
        print(f"[benchlib] no weights cached for {name}: using RANDOM weights from config (fine for timing, useless for text)")
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=dtype).eval()
    return model, tok


# ---------------------------------------------------------------- the timed decode loop
@torch.no_grad()
def timed_generate(model, input_ids: torch.Tensor, new_tokens: int, past_key_values=None):
    """Greedy decode `new_tokens` tokens, recording WHEN each one appeared.

    input_ids: (batch, prompt_len), all prompts the same length (no padding, keeps the loop simple).
    past_key_values: optionally a cache that ALREADY holds a shared prefix (step 12); then input_ids is only the
        NEW part of the prompt. The cache object is extended in place.
    Returns a dict: t0 (start), t_prefill_end, token_times (list of absolute perf_counter stamps, one per
    generated token, the first one is the end of prefill + first argmax), tokens (batch, new_tokens), peak_rss_mb.
    Internally: ONE forward pass over the whole prompt (prefill, fills the KV cache), then one forward pass per
    token feeding only the newest token plus the cache (decode). min_new_tokens is automatic: we never stop at EOS.
    """
    dev = input_ids.device
    sync(dev)
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)   # PREFILL
    nxt = out.logits[:, -1].argmax(-1, keepdim=True)          # (batch, 1) greedy pick
    sync(dev)
    t_pref = time.perf_counter()
    stamps, toks, cache = [t_pref], [nxt], out.past_key_values  # token 1 is available right after prefill
    for _ in range(new_tokens - 1):                           # DECODE loop: one token per forward pass
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        sync(dev)
        stamps.append(time.perf_counter())
        toks.append(nxt)
        cache = out.past_key_values
    return dict(t0=t0, t_prefill_end=t_pref, token_times=stamps, tokens=torch.cat(toks, 1),
                batch=input_ids.shape[0], prompt_len=input_ids.shape[1], peak_rss_mb=peak_rss_mb())


def summarize_trial(trial: dict) -> dict:
    """Turn raw timestamps of ONE run into the standard metrics (seconds unless noted)."""
    ts, t0 = trial["token_times"], trial["t0"]
    n, b = len(ts), trial["batch"]
    itl = [ts[i] - ts[i - 1] for i in range(1, n)]            # inter-token latencies (decode steps only)
    ttft = ts[0] - t0
    e2e = ts[-1] - t0
    return dict(ttft_s=ttft, prefill_s=trial["t_prefill_end"] - t0, itl_s=itl,
                tpot_s=(e2e - ttft) / max(n - 1, 1), e2e_s=e2e,
                tok_per_s=b * n / e2e, decode_tok_per_s=b * (n - 1) / max(e2e - ttft, 1e-12),
                batch=b, prompt_len=trial["prompt_len"], n_tokens=n)


def run_trials(model, input_ids, new_tokens: int = 32, warmup: int = 2, trials: int = 5) -> list[dict]:
    """`warmup` untimed runs (lazy init, allocator, cache warm) then `trials` timed runs -> list of summaries."""
    for _ in range(warmup):
        timed_generate(model, input_ids, min(new_tokens, 8))
    return [summarize_trial(timed_generate(model, input_ids, new_tokens)) for _ in range(trials)]


def aggregate(trials: list[dict]) -> dict:
    """Mean/std across trials for scalar metrics + p50/p95/p99 over ALL ITLs / E2Es pooled together."""
    def ms(key):
        v = [t[key] for t in trials]
        return statistics.fmean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)
    itl_all = [x for t in trials for x in t["itl_s"]]
    e2e_all = [t["e2e_s"] for t in trials]
    agg = dict(trials=len(trials), batch=trials[0]["batch"], prompt_len=trials[0]["prompt_len"], n_tokens=trials[0]["n_tokens"])
    for key in ("ttft_s", "tpot_s", "e2e_s", "tok_per_s", "decode_tok_per_s"):
        agg[key], agg[key + "_std"] = ms(key)
    for q in (50, 95, 99):
        agg[f"itl_p{q}_s"] = percentile(itl_all, q)
        agg[f"e2e_p{q}_s"] = percentile(e2e_all, q)
    agg["peak_rss_mb"] = peak_rss_mb()
    return agg


# ---------------------------------------------------------------- results tables
def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def to_csv(rows: list[dict], path) -> None:
    """Write a list of dicts (same keys) as CSV. newline='' stops blank lines on Windows."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def to_markdown(rows: list[dict], cols: list[str] | None = None) -> str:
    """Render a list of dicts as a GitHub markdown table (paste into notes / README)."""
    cols = cols or list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test with the tiniest cached model. Checks: manual decode == HF generate, metric identities, helpers.
    model, tok = load_causal_lm("sshleifer/tiny-gpt2", threads=2)
    ids = tok(["Hello world, this is a test"] * 2, return_tensors="pt").input_ids
    N = 12
    trial = timed_generate(model, ids, N)
    ref = model.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False, pad_token_id=tok.eos_token_id)[:, ids.shape[1]:]
    assert torch.equal(trial["tokens"], ref), "manual KV-cache decode must match HF generate"
    s = summarize_trial(trial)
    assert len(s["itl_s"]) == N - 1 and s["ttft_s"] <= s["e2e_s"]
    assert abs(s["tpot_s"] - statistics.fmean(s["itl_s"])) < 1e-9, "TPOT == mean ITL"
    agg = aggregate(run_trials(model, ids, N, warmup=1, trials=3))
    assert agg["itl_p50_s"] <= agg["itl_p95_s"] <= agg["itl_p99_s"]
    assert percentile([1, 2, 3, 4, 5], 50) == 3 and percentile([1, 2, 3, 4, 5], 100) == 5
    print(to_markdown([agg], ["batch", "ttft_s", "tpot_s", "itl_p95_s", "tok_per_s", "peak_rss_mb"]))
    print(f"peak RSS {peak_rss_mb():.0f} MB | self-test OK")
