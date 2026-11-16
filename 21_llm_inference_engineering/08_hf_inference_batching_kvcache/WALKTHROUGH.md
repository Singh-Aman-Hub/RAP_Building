# Step 08 — Fast inference with Hugging Face: no_grad, KV cache, batching, dtype

## The story so far
[Step 07](../07_kv_cache_quantization/WALKTHROUGH.md) showed that the KV cache can be shrunk by storing it in fewer bits. Until now we reasoned on paper and with toy tensors. Time to run a real model with the real Hugging Face `generate` and measure what the cache, batching and dtype actually do. Memory arithmetic is in [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md); this step is the stopwatch.

## The pain point
Plain generation is slow in three ways. It tracks gradients it never uses, it recomputes keys and values for the whole sequence at every new token, and it serves one request while the hardware is mostly idle. Batch with the wrong padding side and the output is garbage.

## The spark
Use `torch.no_grad`, keep the KV cache (store each token's K,V so only the new token is computed), and batch several prompts with left padding. Measure each change. Timings vary with machine load; read the ratios, not the seconds.

## Goal
Measure (not just read about) what makes generation fast, avoid the classic padding bug, and meet the words **prefill** and **decode** once, briefly. The physics behind them is [step 09](../09_prefill_decode_roofline/WALKTHROUGH.md).

## The theory (read this before the interview)
```
generation = PREFILL (one pass over the whole prompt, fills the KV cache)
           + DECODE  (one pass per new token, reads the cache, appends one K,V pair)

without cache : step t re-processes all t tokens          -> total work ~ T^2/2 token-passes
with cache    : step t processes 1 token + reads cache    -> total work ~ T token-passes
batching      : B sequences share one read of the weights per step -> throughput up, per-user latency not down
padding       : decoder-only models continue from the LAST position -> pad on the LEFT, pass attention_mask
dtype         : bytes per weight: fp32 4, bf16/fp16 2 -> half the memory (GPU: also faster)
```
- `model.eval()` switches layers such as dropout off; `torch.no_grad()` stops gradient recording; `torch.inference_mode()` is a stricter, slightly faster `no_grad`.
- The cache trades memory for speed: its size is the formula in step 03.
- Prefill and decode are different workloads (many tokens per pass versus one), which is why serving engines treat them differently: [step 09](../09_prefill_decode_roofline/WALKTHROUGH.md) explains why, [step 12](../12_continuous_batching_prefix_caching/WALKTHROUGH.md) shows how engines batch them.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| KV cache (key-value cache) | `past_key_values`, attention cache, `use_cache` | Yes | Stored keys and values of all earlier tokens so each new token only computes its own. |
| Prefill | prompt processing, context encoding, "prompt phase" | Yes | The first pass over all prompt tokens; fills the cache. |
| Decode | generation phase, token-by-token generation, autoregressive step | Yes | One pass per new token, reading the cache. |
| Throughput / latency | tokens per second / time per request | Opposite trade | Throughput is system-wide tokens per second; latency is how long one user waits. Batching raises the first, not the second. |
| Left padding / right padding | `padding_side` | Different | Left puts filler before the prompt so the last position is a real token; right puts it after, which breaks generation in decoder-only models. |
| `no_grad` / `inference_mode` / `eval()` | inference switches | Not the same | `eval()` switches layers such as dropout; `no_grad()` stops gradient recording; `inference_mode()` is a stricter, faster `no_grad`. |
| fp32 / bf16 / fp16 | dtype | Different formats | Bytes per weight: 4 / 2 / 2; bf16 keeps fp32's range with less precision. |

## Run it
```bash
cd 21_llm_inference_engineering/08_hf_inference_batching_kvcache
python main.py       # CPU on purpose; about 30 s on a quiet machine
# from repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/08_hf_inference_batching_kvcache/main.py
```

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–20)

```python
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
```

- **In simple words:** What this step measures and what output to expect.
- **Exactly:** Docstring only. The timing numbers in it come from a quiet machine; yours will differ.
- **Why it is here:** Tells you the shape of a correct result.

### 2. Imports  (`main.py` lines 22–25)

```python
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
```

- **In simple words:** A timer plus PyTorch and the Hugging Face loaders.
- **Exactly:** `time.perf_counter` is the high-resolution clock for benchmarks.
- **Why it is here:** `time.time()` is less precise for short intervals.

### 3. Model name and CPU device  (`main.py` lines 27–28)

```python
MODEL_NAME = "distilgpt2"
device = torch.device("cpu")     # CPU = no asynchronous GPU queue, so time.perf_counter() is honest
```

- **In simple words:** Run on the CPU on purpose.
- **Exactly:** `torch.device("cpu")`. GPU work is queued asynchronously, so naive timing on cuda/mps would be wrong without a synchronize (see [step 10](../10_latency_throughput_benchmark/WALKTHROUGH.md)).
- **Why it is here:** Keeps timings honest.

### 4. Load tokenizer and model for inference  (`main.py` lines 30–32)

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token          # GPT-2 has no pad token (see 04_llm_fundamentals)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device).eval()   # .eval() = inference mode for layers
```

- **In simple words:** Load distilgpt2 and set a pad token.
- **Exactly:** GPT-2 has no pad token, so `pad_token = eos_token`. `.eval()` turns dropout off. Loading itself is covered in [step 01](../01_checkpoint_formats_and_loading/WALKTHROUGH.md).
- **Why it is here:** Without a pad token, batching prompts of different lengths raises an error.

### 5. Count parameters and bytes  (`main.py` lines 34–36)

```python
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params / 1e6:.1f}M | fp32 size {n_params * 4 / 1e6:.0f} MB | bf16 size {n_params * 2 / 1e6:.0f} MB")
# bytes per parameter: fp32 = 4, fp16/bf16 = 2, int8 = 1, int4 = 0.5  -> this is the heart of quantization (step 04).
```

- **In simple words:** Work out how big the model is in memory.
- **Exactly:** `sum(p.numel())` is about 81.9M parameters; x4 bytes = 328 MB (fp32); x2 = 164 MB (bf16). This is the weights term of the [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md) formula.
- **Why it is here:** Bytes per parameter is the core of quantization ([step 04](../04_quantization_math/WALKTHROUGH.md)).

### 6. The timed_generate helper  (`main.py` lines 39–46)

```python
def timed_generate(prompts, **gen_kwargs):
    """Tokenize `prompts` (list of str), generate, return (decoded_texts, seconds)."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)   # input_ids, attention_mask: shape (n_prompts, longest_prompt)
    start = time.perf_counter()                    # high-resolution clock for benchmarking
    with torch.no_grad():                          # inference never needs gradients: saves memory + time
        out = model.generate(**enc, pad_token_id=tokenizer.eos_token_id, do_sample=False, **gen_kwargs)   # shape: (n_prompts, longest_prompt + new_tokens)
    seconds = time.perf_counter() - start
    return tokenizer.batch_decode(out, skip_special_tokens=True), seconds
```

- **In simple words:** A function that tokenizes, generates and returns the texts plus the time taken.
- **Exactly:** `padding=True` pads to the longest prompt; `input_ids` and `attention_mask` have shape `(n_prompts, longest_prompt)`; `generate` with `do_sample=False` is greedy; output shape `(n_prompts, longest + new_tokens)`; extra `**gen_kwargs` (max_new_tokens, use_cache...) pass through.
- **Why it is here:** One function guarantees every experiment is measured the same way.

### 7. Experiment 1: KV cache on vs off  (`main.py` lines 49–59)

```python
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
```

- **In simple words:** Generate 100 tokens with and without the memory of past computation.
- **Exactly:** One warm-up call (5 tokens) first. `min_new_tokens=100` forces exactly 100 tokens. With `use_cache=False` every step re-runs the whole sequence. The timings are in the trace; the ratio matters, not the absolute seconds.
- **Why it is here:** The cache is the reason generation is fast; its price is memory (formula in [step 03](../03_gpu_memory_calculator/WALKTHROUGH.md)).

### 8. Experiment 1b: prefill versus decode (vocabulary only)  (`main.py` lines 61–76)

```python
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
```

- **In simple words:** Time one forward pass over a long prompt, then one pass for a single new token.
- **Exactly:** Prefill = the first pass over all prompt tokens, which fills the KV cache. Decode = every later pass, one new token each, which reads the cache. `model(**prompt, use_cache=True)` returns `logits` and `past_key_values`; the next call passes `past_key_values` back with a single token id. Per token, prefill is much cheaper because it handles many tokens in one pass.
- **Why it is here:** Introduces the two words used in every inference discussion. Why decode is memory-bound and prefill compute-bound is step 09 (roofline), not here.

### 9. Experiment 2: padding side  (`main.py` lines 78–88)

```python
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
```

- **In simple words:** Pad a batch on the right and on the left and see which one still writes sensible text.
- **Exactly:** A decoder-only model continues from the last position, so the last position must be a real token. Right padding makes it continue from pad tokens (plus a transformers warning); left padding gives a sensible continuation. The last line resets to `left`.
- **Why it is here:** The wrong side silently gives junk output with no exception.

### 10. Experiment 3: batching  (`main.py` lines 90–96)

```python
# ---- 3. Batching: one forward pass serves many prompts -------------------------------------------------------
print("\n--- 3. Batching ---")
four = ["The weather today is", "Python is a language that", "My favourite food is", "The capital of France is"]
t_seq = sum(timed_generate([p], max_new_tokens=30, min_new_tokens=30)[1] for p in four)   # sum of 4 separate calls
_, t_batch = timed_generate(four, max_new_tokens=30, min_new_tokens=30)
print(f"4 prompts one-by-one: {t_seq:.1f} s | 4 prompts as one batch: {t_batch:.1f} s")
print("=> same hardware, more tokens per second: THROUGHPUT goes up. Each user still waits roughly the same: LATENCY.")
```

- **In simple words:** Serve four prompts one by one and then all together.
- **Exactly:** `sum(timed_generate([p])[1] for p in four)` adds four separate times. A batch shares one weight read per step across 4 sequences (why: step 09).
- **Why it is here:** Throughput goes up with batching; per-user latency does not improve (measured properly in step 10).

### 11. Experiment 4: load in bf16  (`main.py` lines 98–101)

```python
# ---- 4. dtype: fewer bytes per weight ---------------------------------------------------------------------------
print("\n--- 4. dtype ---")
model_bf16 = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).eval()   # load as bfloat16 (older versions: torch_dtype=)
print("first weight dtype:", next(model_bf16.parameters()).dtype, "| bytes/param:", next(model_bf16.parameters()).element_size())
```

- **In simple words:** Load a copy with half-size numbers.
- **Exactly:** `from_pretrained(..., dtype=torch.bfloat16)`; in transformers v5 the argument is `dtype` (older versions used `torch_dtype`). bf16 weights report `torch.bfloat16`, 2 bytes per parameter.
- **Why it is here:** Halving bytes per weight halves the memory and, on GPU, the time spent reading weights.

### 12. TRY IT / BREAK IT  (`main.py` lines 104–107)

```python
# TRY IT: change max_new_tokens to 300 and compare the cache-on / cache-off gap (it widens with length).
# TRY IT: add a 5th, 6th, 8th prompt to `four` and see how batch time grows (sub-linearly at first).
# BREAK IT: set tokenizer.padding_side = "right" before section 3 and compare the generated sentences.
# BREAK IT: remove `with torch.no_grad():` and compare memory use with `/usr/bin/time -l python main.py` (macOS).
```

- **In simple words:** Experiments in comments.
- **Exactly:** Setting `padding_side = "right"` before section 3 changes generated sentences; removing `no_grad` increases memory use.
- **Why it is here:** Practice isolating one variable at a time.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `def f(prompts, **gen_kwargs)` | `**` collects extra keyword arguments into a dict, forwarded with `**gen_kwargs` | `f(x, max_new_tokens=5)` |
| `for side in ("right", "left"):` | loop over a tuple | |
| `_ , t = f()` | `_` = "I ignore this return value" | |
| `sum(generator)` | add timings of 4 separate calls | |
| `time.perf_counter()` | high-resolution stopwatch; subtract two readings | |
| `**enc` | unpack the tokenizer's dict into keyword arguments `input_ids=..., attention_mask=...` | |
| `out.logits[:, -1:]` | logits of the last position, keeping a sequence dimension of 1 | `(1, 1, vocab)` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `time.perf_counter()` | built-in | stopwatch | – | float seconds |
| `torch.no_grad()` | torch | skip gradient bookkeeping (memory + speed) | – | context |
| `model.generate(**enc, ...)` | transformers | autoregressive decoding loop: forward, pick token, append, reuse cache | `max_new_tokens`, `min_new_tokens`, `do_sample`, `use_cache`, `pad_token_id` | IDs tensor |
| `model(**enc, use_cache=True)` | transformers | one forward pass; returns logits and the cache | `past_key_values` to continue from a cache | `.logits`, `.past_key_values` |
| `tokenizer.padding_side` | transformers | where pad tokens go: `"left"` for decoder-only generation | `"left"/"right"` | – |
| `tokenizer.batch_decode(ids)` | transformers | decode many sequences | `skip_special_tokens` | `list[str]` |
| `from_pretrained(..., dtype=torch.bfloat16)` | transformers | load weights at lower precision | `dtype` (older versions: `torch_dtype`) | model |
| `param.element_size()` | torch | bytes per element | – | int |

## Execution trace (real run)
```
parameters: 81.9M | fp32 size 328 MB | bf16 size 164 MB

--- 1. KV cache ---
use_cache=True : 100 new tokens in 1.6 s
use_cache=False: 100 new tokens in 4.2 s   (cache off recomputes everything each step)

--- 1b. Prefill vs decode (vocabulary; the physics is step 09) ---
prefill of 121 prompt tokens: 73 ms (0.6 ms/token) | one decode step: 17 ms
=> prefill handles many tokens per pass (compute-bound on a GPU); each decode step handles one (memory-bound). Step 09 shows why.

--- 2. Padding side for batches ---
right padding -> short prompt continues as: 'Hello, my name is\n\n\n\n\n\n\n\n'
left  padding -> short prompt continues as: 'Hello, my name is J.J.K. Rowling.'

--- 3. Batching ---
4 prompts one-by-one: 2.1 s | 4 prompts as one batch: 0.8 s
=> same hardware, more tokens per second: THROUGHPUT goes up. Each user still waits roughly the same: LATENCY.

--- 4. dtype ---
first weight dtype: torch.bfloat16 | bytes/param: 2
```
- Parameters 81.9M: 328 MB in fp32, 164 MB in bf16.
- KV cache: 100 new tokens with the cache took 1.6 s, without it 4.2 s (about 2.6x slower; the gap widens with length).
- Prefill versus decode: 121 prompt tokens in one pass took 73 ms (0.6 ms per token); one decode step took 17 ms. Per token, prefill is about 30x cheaper because many tokens share one pass over the weights. Timings vary by machine and load.
- Padding: right padding continued as `'Hello, my name is\n\n\n\n\n\n\n\n'` (junk, plus a transformers warning); left padding gave `'Hello, my name is J.J.K. Rowling.'`.
- Batching: 4 prompts one by one 2.1 s versus one batch 0.8 s.
- dtype: bf16 weights report `torch.bfloat16`, 2 bytes per parameter.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `max_new_tokens=300` | the cache-on/off gap widens | without the cache the cost grows roughly quadratically |
| add more prompts to `four` | batch time grows sub-linearly at first, then linearly | the hardware saturates |
| time bf16 vs fp32 on CPU | often not faster on CPU; on a GPU it nearly always is | CPU lacks fast bf16 paths on many machines; always measure |
| a longer `long_prompt` in section 1b | prefill time grows, one decode step barely changes | prefill touches all prompt tokens; decode touches one (plus reads a bigger cache) |

## Debugging practice
Right padding gives *no traceback*, only odd output and a one-line warning. Habit: read warnings, and compare batch results against single-prompt results (`generate` with batch 1 should match). BREAK IT in the file: set `tokenizer.padding_side = "right"` before section 3.

## Common mistakes
- Right padding for a decoder-only model.
- Timing a first call without a warm-up.
- Timing GPU code without synchronising (this file stays on CPU to avoid it).
- Forgetting `no_grad` in custom generation loops (memory grows with the graph).
- Calling a single-request time "latency and throughput" at once: they are different numbers.

## Rebuild checklist
1. `tokenizer.pad_token = tokenizer.eos_token`, `padding_side="left"`.
2. `model.eval()` + `torch.no_grad()` + `model.generate(**enc, ...)`.
3. Time with `perf_counter`, warm up first.
4. Do one prefill call and one cached decode call by hand.
5. Explain the cache on/off, batching and dtype results.

## Say it aloud
- "I turn off gradient tracking, keep the KV cache so each new token only computes its own keys and values, and batch prompts with left padding for decoder-only models."
- "Generation has two phases: prefill processes the whole prompt in one pass and fills the cache, decode produces one token per pass."
- "The cache trades memory for speed, and batching trades per-user speed for total throughput. I measure each change instead of assuming it helps."

## The new problem (cliffhanger)
Batching helps a lot, and one decode step is far more expensive per token than prefill. Why? Where is the hardware actually spending its time: arithmetic or reading memory? That question has a one-line answer called the roofline: [step 09, prefill, decode and the roofline](../09_prefill_decode_roofline/WALKTHROUGH.md).
