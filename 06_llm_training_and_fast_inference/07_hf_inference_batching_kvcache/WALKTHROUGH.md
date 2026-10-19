# Step 07 — Fast inference with Hugging Face: no_grad, KV cache, batching, dtype

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
- **KV cache:** 100 new tokens: `use_cache=True` **1.5 s**, `use_cache=False` **4.8 s**. Without the cache every step re-processes the whole sequence.
- **Padding side:** with right padding the short prompt continues as `'Hello, my name is\n\n\n\n\n\n\n\n'` (it continued from padding; transformers also prints a warning). With left padding: `'Hello, my name is J.J.K. Rowling.'`.
- **Batching:** 4 prompts one by one **2.8 s** vs as one batch **0.6 s**.
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

## Say it aloud
"For fast inference I put the model in eval mode, disable gradients, keep the KV cache on so each new token only computes its own keys and values, and batch prompts so the GPU does more useful work per pass. For decoder-only models I pad on the left. I can also load weights in bf16 or int8 to cut memory. Trade-off: the KV cache and batches cost memory and larger batches raise each user's latency."
