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
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # find ../benchlib.py
import benchlib

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
ap.add_argument("--new-tokens", type=int, default=32)
ap.add_argument("--prompt-len", type=int, default=16)
ap.add_argument("--threads", type=int, default=4)           # torch CPU threads: ALWAYS report it with a benchmark
ap.add_argument("--trials", type=int, default=5)
args = ap.parse_args()

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

out_file = Path(__file__).resolve().parent / "results.csv"
benchlib.to_csv(rows, out_file)                               # results.csv for your notes
print("\nsaved", out_file)
print("=> throughput rises with batch size while per-user speed falls: the trade-off. TTFT grows with batch (bigger prefill).")
print(f"   coefficient of variation of tok/s (std/mean): {max(r['tok_s_std'] / r['tokens_per_s'] for r in rows):.0%} worst case: report spread, not one number")

# ---- 3. Hygiene check: thread count changes the numbers ---------------------------------------------------------------
if device.type == "cpu":
    print("\nthread-count check (batch 4):")
    for th in (1, 2, 4):
        torch.set_num_threads(th)
        a = benchlib.aggregate(benchlib.run_trials(model, make_ids(4), 16, warmup=1, trials=3))
        print(f"  threads={th}: TPOT {a['tpot_s'] * 1e3:6.1f} ms | tok/s {a['tok_per_s']:6.0f}")
    torch.set_num_threads(args.threads)

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

# TRY IT: python main.py --new-tokens 128 : TPOT stays flat, E2E grows linearly; TTFT does not change.
# TRY IT: python main.py --prompt-len 512 : TTFT jumps (prefill is compute-bound, step 09) while TPOT barely moves.
# TRY IT: python main.py --threads 1 then --threads 8 (more than your cores): compare tok/s and the +-std column.
# BREAK IT: set trials=1 and warmup=0 in the loop: batch 1 (the first measured row) looks slow and the std column is 0 = fake confidence.
# BREAK IT: python main.py --device cuda on this Mac -> AssertionError 'Torch not compiled with CUDA enabled'.
