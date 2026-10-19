"""
GOAL:  Measure the latency-vs-throughput trade-off yourself: generate with batch sizes 1,2,4,8,16 and
       record time per batch, tokens/second, and memory.  Saves results.csv for your notes.

RUN (from this step folder):  python main.py            (add --device mps to try the Apple GPU)
RUN (from repo root):         .venv/bin/python 06_llm_training_and_fast_inference/09_throughput_benchmark/main.py
EXPECTED OUTPUT (one real CPU run on an M1; your numbers will differ and are noisy):
    batch  latency(s)  tokens/s  per-user tok/s  peak RSS(MB)
        1       0.57      70.7            70.7           805
        4       0.81     197.5            49.4           809
       16       0.94     684.1            42.8           809
    => throughput rises with batch size; per-user speed falls: that is the trade-off.
    (peak RSS stays flat here because this tiny model's KV cache is small next to the weights.)
    --device mps is erratic for a model this small: GPUs shine on large models and large batches.
NEEDS: torch, transformers (distilgpt2 cached).  ~30-60 s.
"""

import argparse
import csv
import resource   # built-in (Unix): read this process's peak memory
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
ap.add_argument("--new-tokens", type=int, default=40)
args = ap.parse_args()

device = torch.device(args.device)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2", padding_side="left")   # left padding: step 07
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device).eval()


def sync():
    """GPUs run asynchronously: wait for queued work to finish before reading the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def peak_rss_mb() -> float:
    """Peak resident memory of this process. macOS reports bytes, Linux reports kilobytes."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def run_batch(batch_size: int) -> float:
    prompts = ["Once upon a time in a small village"] * batch_size
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    sync()
    start = time.perf_counter()
    with torch.no_grad():
        # min_new_tokens == max_new_tokens forces every sequence to produce exactly N tokens (fair comparison)
        model.generate(**enc, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
    sync()
    return time.perf_counter() - start


run_batch(1)                                  # warm-up (not recorded)
rows = []
print(f"{'batch':>5} {'latency(s)':>11} {'tokens/s':>9} {'per-user tok/s':>15} {'peak RSS(MB)':>13}")
for bs in (1, 2, 4, 8, 16):
    seconds = run_batch(bs)
    total_tokens = bs * args.new_tokens        # all sequences together
    throughput = total_tokens / seconds        # THROUGHPUT: system-wide tokens per second
    per_user = args.new_tokens / seconds       # what ONE user experiences (inverse of latency)
    rows.append((bs, round(seconds, 2), round(throughput, 1), round(per_user, 1), round(peak_rss_mb())))
    print(f"{bs:>5} {seconds:>11.2f} {throughput:>9.1f} {per_user:>15.1f} {peak_rss_mb():>13.0f}")

out_file = Path(__file__).resolve().parent / "results.csv"
with open(out_file, "w", newline="") as f:      # `with` closes the file even if an error occurs
    writer = csv.writer(f)
    writer.writerow(["batch", "latency_s", "tokens_per_s", "per_user_tokens_per_s", "peak_rss_mb"])
    writer.writerows(rows)
print("\nsaved", out_file)
print("=> throughput rises with batch size; per-user speed falls: that is the trade-off.")

# TRY IT: python main.py --new-tokens 100 : do longer outputs change the shape of the curve?
# TRY IT: python main.py --device mps : is the GPU faster at batch 1? at batch 16? (small models under-use a GPU)
# TRY IT: add 32 to the tuple of batch sizes; where does throughput stop improving or memory jump?
# BREAK IT: python main.py --device cuda on this Mac -> read the 'Torch not compiled with CUDA' error / argparse messages.
