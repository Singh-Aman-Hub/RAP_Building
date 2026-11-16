# Step 07 — Fast inference with Hugging Face: no_grad, KV cache, batching, dtype

## The story so far
Chapter 17, Production Speed. The surgical fine-tune of [step 06](../06_qlora_reference/WALKTHROUGH.md) gave you a model that knows your task. Now users arrive, and each of them is waiting.

## The pain point
Plain generation is slow in three ways. It tracks gradients it never uses, it recomputes keys and values for the whole sequence at every new token, and it serves one request while the GPU is mostly idle. Batch with the wrong padding side and the output is garbage.

## The spark
Use `torch.no_grad`, keep the KV cache (store each token's K,V so only the new token is computed), and batch several prompts with left padding. Measure each change: the cache alone cut 100 tokens from 8.5 s to 3.5 s in this step (timings vary with machine load; earlier runs measured 4.8 s versus 1.5 s).

## Goal
Measure (not just read about) what makes generation fast, and avoid the classic padding bug.

## Run it
```bash
cd 06_llm_training_and_fast_inference/07_hf_inference_batching_kvcache
python main.py       # ~30 s, CPU on purpose
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `def f(prompts, **gen_kwargs)` | `**` collects extra keyword arguments into a dict, forwarded with `**gen_kwargs` | `f(x, max_new_tokens=5)` |
| `for side in ("right", "left"):` | loop over a tuple | |
| `_ , t = f()` | `_` = "I ignore this return value" | |
| `sum(generator)` | add timings of 4 separate calls | |
| `time.perf_counter()` | high-resolution stopwatch; subtract two readings | |
| `**enc` | unpack the tokenizer's dict into keyword arguments `input_ids=..., attention_mask=...` | |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `time.perf_counter()` | built-in | stopwatch | – | float seconds |
| `torch.no_grad()` | torch | skip gradient bookkeeping (memory + speed) | – | context |
| `model.generate(**enc, ...)` | transformers | autoregressive decoding | `max_new_tokens`, `min_new_tokens`, `do_sample`, `use_cache`, `pad_token_id` | IDs tensor |
| `tokenizer.padding_side` | transformers | where pad tokens go: `"left"` for decoder-only generation | `"left"/"right"` | – |
| `tokenizer.batch_decode(ids)` | transformers | decode many sequences | `skip_special_tokens` | `list[str]` |
| `from_pretrained(..., dtype=torch.bfloat16)` | transformers | load weights at lower precision | `dtype` (older versions: `torch_dtype`) | model |
| `param.element_size()` | torch | bytes per element | – | int |

## Execution trace (real run)
- `parameters: 81.9M | fp32 size 328 MB | bf16 size 164 MB` — bytes/param x number of params = memory for weights.
- **KV cache:** 100 new tokens: `use_cache=True` **3.5 s**, `use_cache=False` **8.5 s**. Without the cache every step re-processes the whole sequence.
- **Padding side:** with right padding the short prompt continues as `'Hello, my name is\n\n\n\n\n\n\n\n'` (it continued from padding; transformers also prints a warning). With left padding: `'Hello, my name is J.J.K. Rowling.'`.
- **Batching:** 4 prompts one by one **4.3 s** vs as one batch **1.2 s**.
- **dtype:** bf16 weights report `torch.bfloat16`, 2 bytes/param.

## Experiments (change ONE thing)
1. `max_new_tokens=300`: the cache-on/off gap widens (cost grows ~quadratically without cache).
2. Add more prompts to the batch: time grows sub-linearly at first, then linearly once the hardware is saturated.
3. Time bf16 vs fp32 on CPU: often not faster on CPU — on a GPU it nearly always is. Always measure.

## Debugging practice
Right padding gives *no traceback* — only odd output and a one-line warning. Habit: read warnings, and compare batch results against single-prompt results (`generate` with batch 1 should match).

## Rebuild checklist
- [ ] `tokenizer.pad_token = tokenizer.eos_token`, `padding_side="left"`
- [ ] `model.eval()` + `torch.no_grad()` + `model.generate(**enc, ...)`
- [ ] time with `perf_counter`, warm up first
- [ ] explain cache on/off and batching results

## The new problem (cliffhanger)
Batching and the cache help, but how much? Speed has two faces, latency for one user and throughput for the whole system, and they pull against each other. Measure them: [step 09, throughput benchmark](../09_throughput_benchmark/WALKTHROUGH.md).

> **Say it aloud (interview box):** I turn off gradient tracking, keep the KV cache so each new token only computes its own keys and values, and batch prompts with left padding for decoder-only models. The cache trades memory for speed, and batching trades per-user speed for total throughput. I measure each change instead of assuming it helps.
