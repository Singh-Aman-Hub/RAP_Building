"""
GOAL:  See, with numbers from YOUR laptop, why prefill is compute-bound and decode is memory-bound:
       (1) measure the machine (numpy matmul GFLOP/s + memory-copy GB/s) -> a roofline,
       (2) compute FLOPs and bytes per token from the parameter count,
       (3) MEASURE prefill time vs prompt length and decode time/token vs batch size on distilgpt2 and Qwen2.5-0.5B.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/09_prefill_decode_roofline/main.py
EXPECTED OUTPUT (shortened; numbers vary by machine):
    machine: peak ~N GFLOP/s (numpy fp32 matmul) | ~M GB/s (copy) | ridge point = N/M FLOP per byte
    reference GPUs (spec sheet): A100 ridge ~153, H100 ridge ~295 FLOP/byte
    [distilgpt2] prefill: ms/token falls then flattens as the prompt grows (matmuls get efficient)
    [distilgpt2] decode:  ms per step ~flat for batch 1..8, then rises; tokens/s per step keeps growing
    bytes/step / measured time = achieved GB/s (compare to the copy bandwidth)
NEEDS: torch, numpy, transformers (cached distilgpt2; Qwen2.5-0.5B has only config cached -> random weights, same timing; no downloads).  ~1.5 min on CPU.
       Qwen can be skipped:  MODELS=distilgpt2 python main.py     (heavy: run through the global lock)
Builds on: 08_hf_inference_batching_kvcache (prefill/decode vocabulary), 03_gpu_memory_calculator (bytes/param, KV formula).
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `import benchlib` works from any cwd
import benchlib

MODELS = os.environ.get("MODELS", "distilgpt2,Qwen/Qwen2.5-0.5B").split(",")
THREADS = int(os.environ.get("THREADS", 4))
torch.set_num_threads(THREADS)


# ---- 1. Measure the machine: compute roof and memory roof ------------------------------------------------------------
def measure_gflops(n=1536, reps=5):
    """Best of `reps` fp32 matmuls: FLOPs = 2*n^3 (n^2 outputs, each n multiplies + n adds)."""
    a, b = torch.randn(n, n), torch.randn(n, n)
    a @ b                                                        # warm-up
    best = min(_time(lambda: a @ b) for _ in range(reps))
    return 2 * n ** 3 / best / 1e9


def measure_gbps(mb=256, reps=5):
    """Copy a big tensor (read mb + write mb). Bigger than any cache, so it measures DRAM, not cache."""
    src = torch.empty(mb * 1024 * 1024 // 4)                    # fp32 elements
    src.fill_(1.0)
    dst = torch.empty_like(src)
    dst.copy_(src)
    best = min(_time(lambda: dst.copy_(src)) for _ in range(reps))
    return 2 * mb * 1024 * 1024 / best / 1e9                    # bytes moved = read + write


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


GFLOPS, GBPS = measure_gflops(), measure_gbps()
RIDGE = GFLOPS / GBPS                                           # FLOP per byte where the two roofs meet
print(f"threads={THREADS} | machine: ~{GFLOPS:.0f} GFLOP/s fp32 matmul | ~{GBPS:.0f} GB/s copy | ridge point = {RIDGE:.1f} FLOP/byte")
print("reference (vendor spec sheets, NOT measured here; dense bf16):")
for name, tflops, tbps in (("A100 80GB", 312, 2.039), ("H100 SXM", 989, 3.35)):
    print(f"  {name:10s} {tflops} TFLOP/s | {tbps} TB/s | ridge ~{tflops / tbps:.0f} FLOP/byte")


def roofline_gflops(intensity):
    """Attainable GFLOP/s = min(compute roof, bandwidth * FLOP-per-byte). This single line IS the roofline model."""
    return min(GFLOPS, GBPS * intensity)


# ---- 2. Per-token FLOPs and bytes from the parameter count -----------------------------------------------------------
def analyse(name):
    model, tok = benchlib.load_causal_lm(name, threads=THREADS)
    n_params = sum(p.numel() for p in model.parameters())      # tied embeddings are counted once by .parameters()
    weight_bytes = n_params * 4                                # fp32
    flops_per_token = 2 * n_params                             # each weight: 1 multiply + 1 add per token
    print(f"\n[{name}] params {n_params / 1e6:.0f}M | weights {weight_bytes / 1e6:.0f} MB fp32 | FLOPs/token ~ 2*params = {flops_per_token / 1e9:.2f} GFLOP")
    print("  arithmetic intensity of ONE decode step by batch size B  =  2*P*B / (4*P)  = B/2 FLOP/byte (KV bytes ignored):")
    for B in (1, 8, 64, 512):
        ai = flops_per_token * B / weight_bytes
        side = "memory-bound" if ai < RIDGE else "compute-bound"
        print(f"    B={B:<4d} intensity {ai:6.1f} -> {side} on this machine (ridge {RIDGE:.0f}); roofline allows {roofline_gflops(ai):.0f} GFLOP/s")
    return model, tok, n_params, weight_bytes, flops_per_token


# ---- 3. Measure prefill time vs prompt length ------------------------------------------------------------------------
def best_of(fn, n=3):
    fn()                                                       # warm-up
    return min(_time(fn) for _ in range(n))


@torch.no_grad()
def prefill_sweep(model, flops_per_token, lengths):
    print(f"  PREFILL (one forward over L prompt tokens, batch 1):")
    print(f"  {'L':>5} {'time ms':>9} {'ms/token':>9} {'GFLOP/s':>9} {'% of matmul peak':>17}")
    for L in lengths:
        ids = torch.randint(100, 5000, (1, L))
        t = best_of(lambda: model(input_ids=ids, use_cache=True))
        g = flops_per_token * L / t / 1e9
        print(f"  {L:>5} {t * 1e3:>9.1f} {t * 1e3 / L:>9.3f} {g:>9.0f} {100 * g / GFLOPS:>16.0f}%")


# ---- 4. Measure decode step time vs batch size -----------------------------------------------------------------------
def decode_sweep(model, weight_bytes, flops_per_token, batches, ctx=16):
    print(f"  DECODE (context {ctx} tokens, TPOT = mean time per generated token, via benchlib.timed_generate):")
    print(f"  {'batch':>5} {'ms/step':>9} {'tokens/s':>9} {'GB/s weights':>13} {'GFLOP/s':>9} {'step x vs B=1':>14}")
    base = None
    for B in batches:
        ids = torch.randint(100, 5000, (B, ctx))
        tr = benchlib.aggregate(benchlib.run_trials(model, ids, new_tokens=9, warmup=1, trials=3))
        step = tr["tpot_s"]
        base = base or step
        print(f"  {B:>5} {step * 1e3:>9.1f} {B / step:>9.0f} {weight_bytes / step / 1e9:>13.1f} {flops_per_token * B / step / 1e9:>9.0f} {step / base:>13.2f}x")


for name in MODELS:
    model, tok, P, WB, FPT = analyse(name)
    big = P > 2e8
    prefill_sweep(model, FPT, (8, 32, 128, 512) if big else (8, 16, 32, 64, 128, 256, 512, 960))
    decode_sweep(model, WB, FPT, (1, 4, 16, 64) if big else (1, 2, 4, 8, 16, 32, 64, 128))
    del model

# ---- 5. KV-cache bytes also move every decode step (grows with context) ---------------------------------------------
# Formula (full derivation: 03_gpu_memory_calculator): kv_bytes_per_token = 2 (K and V) * layers * kv_heads * head_dim * bytes
# Qwen2.5-0.5B: 24 layers, 2 KV heads (GQA), head_dim 64, fp32 -> 2*24*2*64*4 = 24,576 bytes/token = 24 KiB.
kv = 2 * 24 * 2 * 64 * 4
print(f"\nKV bytes/token for Qwen2.5-0.5B fp32 = {kv} B; at context 2048 that is {kv * 2048 / 1e6:.0f} MB read per decode step per sequence")
print("(vs 1976 MB of fp32 weights: tiny here, but for a 7B MHA model at 32k context the KV bytes exceed the weights.)")

# ---- 6. Chunked prefill: why a long prompt must not freeze everyone else's decode (concept, simple arithmetic) -------
# A 8,000-token prompt prefilled in one go occupies the GPU for T_pre. Every running request's next token waits T_pre.
# Splitting it into chunks of C tokens and ALSO running the decode tokens in each iteration caps the stall at ~T_pre*C/8000.
T_PRE_MS, PROMPT, CHUNK = 800.0, 8000, 512                      # illustrative numbers, not measured
print(f"\nchunked prefill (illustrative): unchunked stall {T_PRE_MS:.0f} ms vs chunk of {CHUNK} -> ~{T_PRE_MS * CHUNK / PROMPT:.0f} ms per iteration")

# TRY IT: THREADS=1 python main.py (compute roof drops ~4x; ridge moves; does batch-1 decode time change much?)
# TRY IT: MODELS=distilgpt2 python main.py and compare 'GB/s weights' at batch 1 with the copy GB/s measured at the top.
# BREAK IT: change `ctx=16` in decode_sweep to 900 for distilgpt2: step time rises because attention must read a bigger KV cache.
# BREAK IT: delete the warm-up call in best_of(): the first (smallest) length looks much slower than it really is.
