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

# TRY IT: `--model gpt2` (bigger) or `--model sshleifer/tiny-gpt2` (tiny, load-time dominated); compare the recommendation.
# TRY IT: set PRICE_PER_HOUR=3.5 and note that $/1M scales linearly with price, so rankings do not change.
# TRY IT: change `ppl_tolerance` in recommend() to 0.001 and see which configs the quality gate rejects.
# BREAK IT: remove the quality gate in recommend(): the "fastest" config may be one that hurts perplexity.
# BREAK IT: measure RSS in one process for all configs instead of subprocesses: peak RSS only ever grows, so later rows are wrong.
