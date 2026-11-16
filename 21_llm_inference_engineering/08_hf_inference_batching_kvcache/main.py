"""
GOAL:  Make Hugging Face generation fast and understand why: no_grad, dtype, batching with padding,
       and the KV cache (measured, not just described).

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 21_llm_inference_engineering/08_hf_inference_batching_kvcache/main.py
EXPECTED OUTPUT (shortened; times differ per machine):
    parameters: 81.9M | fp32 size 328 MB | bf16 size 164 MB
    --- 1. KV cache ---
    use_cache=True : 100 new tokens in 1.6 s
    use_cache=False: 100 new tokens in 4.2 s   (cache off recomputes everything each step)
    --- 1b. Prefill vs decode (vocabulary; the physics is step 09) ---
    prefill of 121 prompt tokens: 73 ms (0.6 ms/token) | one decode step: 17 ms
    --- 2. Padding side for batches ---
    right padding -> short prompt continues as: 'Hello, my name is\n\n\n\n...'
    left  padding -> short prompt continues as: 'Hello, my name is J.J.K. Rowling.'
    --- 3. Batching ---
    4 prompts one-by-one: 2.1 s | 4 prompts as one batch: 0.8 s
NEEDS: torch, transformers (distilgpt2 cached).  ~30 s.  Uses the CPU on purpose so timings are stable.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
device = torch.device("cpu")     # CPU = no asynchronous GPU queue, so time.perf_counter() is honest

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token          # GPT-2 has no pad token (see 04_llm_fundamentals)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()   # .eval() = inference mode for layers

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params / 1e6:.1f}M | fp32 size {n_params * 4 / 1e6:.0f} MB | bf16 size {n_params * 2 / 1e6:.0f} MB")
# bytes per parameter: fp32 = 4, fp16/bf16 = 2, int8 = 1, int4 = 0.5  -> this is the heart of quantization (step 04).


def timed_generate(prompts, **gen_kwargs):
    """Tokenize `prompts` (list of str), generate, return (decoded_texts, seconds)."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)   # input_ids, attention_mask: shape (n_prompts, longest_prompt)
    start = time.perf_counter()                    # high-resolution clock for benchmarking
    with torch.no_grad():                          # inference never needs gradients: saves memory + time
        out = model.generate(**enc, pad_token_id=tokenizer.eos_token_id, do_sample=False, **gen_kwargs)   # shape: (n_prompts, longest_prompt + new_tokens)
    seconds = time.perf_counter() - start
    return tokenizer.batch_decode(out, skip_special_tokens=True), seconds


# ---- 1. KV cache on vs off ----------------------------------------------------------------------------
# Each new token needs attention over ALL previous tokens' keys (K) and values (V).
# With the cache we store K,V once and only compute the NEW token's.  Without it, every step recomputes
# the whole sequence: cost grows ~quadratically with length.  Price of the cache: GPU/CPU memory (formula in step 03).
print("\n--- 1. KV cache ---")
prompt = ["The future of machine learning is"]
_, t_warm = timed_generate(prompt, max_new_tokens=5)          # warm-up: first call is slower (lazy init)
_, t_on = timed_generate(prompt, max_new_tokens=100, min_new_tokens=100, use_cache=True)
_, t_off = timed_generate(prompt, max_new_tokens=100, min_new_tokens=100, use_cache=False)
print(f"use_cache=True : 100 new tokens in {t_on:.1f} s")
print(f"use_cache=False: 100 new tokens in {t_off:.1f} s   (cache off recomputes everything each step)")

# ---- 1b. Prefill vs decode: only the VOCABULARY here, step 09 (roofline) explains it deeply -----------------
# PREFILL = process the whole prompt in one forward pass and fill the KV cache (many tokens at once).
# DECODE  = then generate ONE token per forward pass, reading the cache (one token at a time).
print("\n--- 1b. Prefill vs decode (vocabulary; the physics is step 09) ---")
long_prompt = tokenizer("The quick brown fox jumps over the lazy dog. " * 12, return_tensors="pt").to(device)
n_prompt = long_prompt["input_ids"].shape[1]
with torch.no_grad():
    t0 = time.perf_counter()
    out = model(**long_prompt, use_cache=True)                    # PREFILL: n_prompt tokens in one pass
    t_prefill = time.perf_counter() - t0
    nxt = out.logits[:, -1:].argmax(-1)                           # greedy next token, shape (1, 1)
    t0 = time.perf_counter()
    model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)   # DECODE: ONE new token, reuse the cache
    t_decode = time.perf_counter() - t0
print(f"prefill of {n_prompt} prompt tokens: {t_prefill * 1000:.0f} ms ({t_prefill / n_prompt * 1000:.1f} ms/token) | one decode step: {t_decode * 1000:.0f} ms")
print("=> prefill handles many tokens per pass (compute-bound on a GPU); each decode step handles one (memory-bound). Step 09 shows why.")

# ---- 2. Padding side matters for decoder-only models ------------------------------------------------------
# In a batch, shorter prompts are padded to equal length.  A decoder-only model CONTINUES from the last
# position, so padding must go on the LEFT; padding on the right makes it continue from pad tokens.
print("\n--- 2. Padding side for batches ---")
mixed = ["Hello, my name is", "In the beginning of the story, a very old wizard walked slowly through the"]
for side in ("right", "left"):
    tokenizer.padding_side = side
    texts, _ = timed_generate(mixed, max_new_tokens=8)
    label = "right padding" if side == "right" else "left  padding"
    print(f"{label} -> short prompt continues as: {texts[0]!r}")
tokenizer.padding_side = "left"      # keep the correct setting from now on

# ---- 3. Batching: one forward pass serves many prompts -------------------------------------------------------
print("\n--- 3. Batching ---")
four = ["The weather today is", "Python is a language that", "My favourite food is", "The capital of France is"]
t_seq = sum(timed_generate([p], max_new_tokens=30, min_new_tokens=30)[1] for p in four)   # sum of 4 separate calls
_, t_batch = timed_generate(four, max_new_tokens=30, min_new_tokens=30)
print(f"4 prompts one-by-one: {t_seq:.1f} s | 4 prompts as one batch: {t_batch:.1f} s")
print("=> same hardware, more tokens per second: THROUGHPUT goes up. Each user still waits roughly the same: LATENCY.")

# ---- 4. dtype: fewer bytes per weight ---------------------------------------------------------------------------
print("\n--- 4. dtype ---")
model_bf16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).eval()   # load as bfloat16 (older versions: torch_dtype=)
print("first weight dtype:", next(model_bf16.parameters()).dtype, "| bytes/param:", next(model_bf16.parameters()).element_size())
# On a GPU, bf16/fp16 is both smaller AND faster. On many CPUs bf16 is not faster - always measure.

# TRY IT: change max_new_tokens to 300 and compare the cache-on / cache-off gap (it widens with length).
# TRY IT: add a 5th, 6th, 8th prompt to `four` and see how batch time grows (sub-linearly at first).
# BREAK IT: set tokenizer.padding_side = "right" before section 3 and compare the generated sentences.
# BREAK IT: remove `with torch.no_grad():` and compare memory use with `/usr/bin/time -l python main.py` (macOS).
