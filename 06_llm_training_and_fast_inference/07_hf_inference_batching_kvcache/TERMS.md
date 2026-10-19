# Terms — Fast Inference with Hugging Face

Folder: `06_llm_training_and_fast_inference/07_hf_inference_batching_kvcache` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [06 README](../README.md)

**Terms in this folder:** [Inference](#inference) · [time.perf_counter](#timeperf_counter) · [Warm-up call](#warm-up-call) · [Bytes per parameter (dtype size)](#bytes-per-parameter-dtype-size) · [**kwargs and `**` unpacking](#kwargs-and--unpacking) · [Underscore `_` as a throwaway name](#underscore-_-as-a-throwaway-name) · [torch.no_grad](#torchno_grad) · [model.generate arguments](#modelgenerate-arguments) · [tokenizer.batch_decode](#tokenizerbatch_decode) · [KV cache](#kv-cache) · [use_cache](#use_cache) · [Padding side (left vs right)](#padding-side-left-vs-right) · [Decoder-only model](#decoder-only-model) · [Batching](#batching) · [Throughput vs latency](#throughput-vs-latency) · [dtype (bfloat16 / fp16 / fp32)](#dtype-bfloat16--fp16--fp32) · [element_size](#element_size) · [Quadratic cost of recomputation](#quadratic-cost-of-recomputation) · [Silent warning bug (right padding)](#silent-warning-bug-right-padding)

---

## Inference  `(concept)`
**1. Definition:** Inference is using a trained model to produce outputs, without updating its weights.
**2. In plain words:** Training teaches the model; inference is the model being used. It has different priorities: speed, memory and cost per request, not learning.
**3. Example to quote:** > "For inference I put the model in `.eval()` mode and disable gradients, because I only need outputs, not learning."
**Seen in:** `main.py`, whole file

## time.perf_counter  `(standard library)`
**1. Definition:** `time.perf_counter()` returns a high-resolution clock value in seconds, suitable for measuring short durations by subtracting two readings.
**2. In plain words:** It's a stopwatch. Take one reading before and one after, then subtract. The script runs on CPU so the timing is honest; GPUs run asynchronously and need synchronising before you stop the clock.
**3. Example to quote:** > "I wrap `generate` between two `perf_counter()` calls: 100 tokens took 1.5 s with the cache and 4.8 s without."
```python
import time
t0 = time.perf_counter(); do_work(); print(time.perf_counter() - t0)
```
**Seen in:** `main.py`, `timed_generate`

## Warm-up call  `(concept)`
**1. Definition:** A warm-up call is a throwaway first run done before measuring, to absorb one-time costs such as lazy initialisation and cache filling.
**2. In plain words:** The first call is often slower for reasons unrelated to what you're testing, so you ignore it and time the later ones.
**3. Example to quote:** > "I generate 5 tokens first as a warm-up and only time the later calls."
**Seen in:** `main.py`, section 1

## Bytes per parameter (dtype size)  `(concept)`
**1. Definition:** Memory for weights equals number of parameters times bytes per parameter: fp32 = 4, fp16/bf16 = 2, int8 = 1, int4 = 0.5.
**2. In plain words:** distilgpt2 has 81.9M parameters, so 328 MB in fp32 and 164 MB in bf16. This arithmetic is the heart of quantization.
**3. Example to quote:** > "81.9 million parameters times 4 bytes is about 328 MB; in bf16 it's 164 MB."
**Seen in:** `main.py`, parameter printout

## **kwargs and `**` unpacking  `(syntax)`
**1. Definition:** In a function definition `**gen_kwargs` collects extra keyword arguments into a dictionary; in a call `**d` expands a dictionary into keyword arguments.
**2. In plain words:** It lets `timed_generate` accept any generation options (`max_new_tokens=100`, `use_cache=False`) and forward them untouched to `model.generate`. `**enc` similarly expands the tokenizer output into `input_ids=..., attention_mask=...`.
**3. Example to quote:** > "`timed_generate(prompt, max_new_tokens=100, use_cache=False)` passes those options straight through via `**gen_kwargs`."
```python
def f(**kw): return kw
f(a=1, b=2)   # {'a': 1, 'b': 2}
```
**Seen in:** `main.py`, `timed_generate`

## Underscore `_` as a throwaway name  `(syntax)`
**1. Definition:** By convention `_` names a value you intend to ignore, e.g. `_, t = f()`.
**2. In plain words:** The function returns two things but only the time matters here; `_` says "I don't need the first one".
**3. Example to quote:** > "`_, t_on = timed_generate(...)` keeps only the seconds and ignores the decoded text."
**Seen in:** `main.py`

## torch.no_grad  `(third-party: torch)`
**1. Definition:** `torch.no_grad()` is a context manager that turns off gradient tracking inside its block.
**2. In plain words:** Inference never needs gradients, so skipping them saves memory and time.
**3. Example to quote:** > "I wrap generation in `with torch.no_grad():` because storing gradient history would waste memory."
**Seen in:** `main.py`, `timed_generate`

## model.generate arguments  `(third-party: transformers)`
**1. Definition:** `model.generate(**enc, ...)` decodes tokens autoregressively; key arguments are `max_new_tokens` (upper limit on new tokens), `min_new_tokens` (lower limit), `do_sample=False` (greedy), `use_cache` (KV cache on/off) and `pad_token_id`.
**2. In plain words:** `min_new_tokens=100` forces exactly 100 tokens so timings compare like for like. `pad_token_id` is set to EOS because GPT-2 has no pad token and `generate` would warn.
**3. Example to quote:** > "I force exactly 100 new tokens with `min_new_tokens=100` so the cache-on and cache-off runs do equal work."
**Seen in:** `main.py`

## tokenizer.batch_decode  `(third-party: transformers)`
**1. Definition:** `tokenizer.batch_decode(ids, skip_special_tokens=True)` converts a batch of token-ID sequences into a list of strings.
**2. In plain words:** The multi-sequence version of `decode`, used because a batch yields several outputs at once.
**3. Example to quote:** > "I call `batch_decode` on the output tensor to get four strings for my four prompts."
**Seen in:** `main.py`, `timed_generate`

## KV cache  `(concept)`
**1. Definition:** The KV cache stores the attention keys (K) and values (V) already computed for earlier tokens so each new token only computes its own K and V.
**2. In plain words:** Without it, every step re-processes the whole sequence. The measured result: 100 tokens in 1.5 s with the cache versus 4.8 s without. The price is memory that grows with sequence length and batch size (see step 08).
**3. Example to quote:** > "With the KV cache on, 100 tokens took 1.5 s; off, 4.8 s, because every step recomputed the entire sequence."
**Seen in:** `main.py`, section 1

## use_cache  `(concept)`
**1. Definition:** `use_cache` is the generation argument that switches the KV cache on or off.
**2. In plain words:** On by default. Turning it off is only useful to demonstrate its benefit or to save memory.
**3. Example to quote:** > "I compare `use_cache=True` with `use_cache=False` on the same 100-token job."
**Seen in:** `main.py`

## Padding side (left vs right)  `(concept)`
**1. Definition:** `tokenizer.padding_side` selects whether pad tokens are added before (`"left"`) or after (`"right"`) the real tokens.
**2. In plain words:** A decoder-only model continues from the last position, so for batched generation padding must go on the left. With right padding the short prompt continued from padding and produced `'Hello, my name is\n\n\n\n...'`; with left padding it said `'Hello, my name is J.J.K. Rowling.'`.
**3. Example to quote:** > "For batched generation with GPT-2 I set `tokenizer.padding_side = 'left'`; right padding made the short prompt continue from pad tokens."
**Seen in:** `main.py`, section 2

## Decoder-only model  `(concept)`
**1. Definition:** A decoder-only transformer generates text left to right, predicting each token from the preceding ones (GPT-2, Llama, Qwen).
**2. In plain words:** There is no separate encoder; the same stack reads the prompt and writes the continuation, which is why where the prompt ends matters for padding.
**3. Example to quote:** > "distilgpt2 is decoder-only, so it continues from the last token position, hence the left padding."
**Seen in:** `main.py`, section 2 comments

## Batching  `(concept)`
**1. Definition:** Batching processes several prompts together in one forward pass.
**2. In plain words:** The hardware does more useful work per pass. Four prompts one by one took 2.8 s; as one batch 0.6 s. Time grows sub-linearly at first, then linearly once the hardware is saturated.
**3. Example to quote:** > "Batching four prompts cut total time from 2.8 s to 0.6 s on the same machine."
**Seen in:** `main.py`, section 3

## Throughput vs latency  `(concept)`
**1. Definition:** Throughput is total work done per unit time (tokens per second across all users); latency is the time one request waits for its answer.
**2. In plain words:** Batching raises throughput, but a single user may wait about as long as before, or longer with big batches. Serving always balances the two.
**3. Example to quote:** > "Bigger batches give me more tokens per second overall, but each individual user doesn't get an answer faster."
**Seen in:** `main.py`, section 3 print

## dtype (bfloat16 / fp16 / fp32)  `(concept)`
**1. Definition:** dtype is the numeric precision of tensors; loading with `dtype=torch.bfloat16` stores weights as 2-byte bfloat16 values instead of 4-byte fp32. Older transformers versions call the argument `torch_dtype`.
**2. In plain words:** Half the bytes, half the memory. On a GPU, 16-bit is usually also faster; on many CPUs it is not, so always measure.
**3. Example to quote:** > "I load with `dtype=torch.bfloat16`; the first weight reports `torch.bfloat16` and 2 bytes per parameter."
**Seen in:** `main.py`, section 4

## element_size  `(third-party: torch)`
**1. Definition:** `tensor.element_size()` returns the number of bytes used by one element of the tensor.
**2. In plain words:** A quick way to confirm the dtype you actually got (2 for bf16, 4 for fp32).
**3. Example to quote:** > "`next(model_bf16.parameters()).element_size()` printed 2, confirming the weights are 16-bit."
**Seen in:** `main.py`, section 4

## Quadratic cost of recomputation  `(concept)`
**1. Definition:** Without a KV cache, generating n tokens recomputes attention over the whole prefix every step, so total work grows roughly with n squared.
**2. In plain words:** That's why the cache-on versus cache-off gap widens with longer outputs (try 300 tokens).
**3. Example to quote:** > "With 300 new tokens the gap between cache-on and cache-off gets even bigger, because recomputation grows roughly quadratically."
**Seen in:** `main.py` comments; `WALKTHROUGH.md`

## Silent warning bug (right padding)  `(concept)`
**1. Definition:** Using the wrong padding side raises no exception, only a one-line warning and odd output.
**2. In plain words:** Read warnings, and compare a batch result against single-prompt results; batch size 1 should match.
**3. Example to quote:** > "There was no traceback, just a warning and strange output, so I compared the batch result with the single-prompt result and found the padding side."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
