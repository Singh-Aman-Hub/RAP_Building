# 05 - LLM Parameters: Training and Inference (syllabus topic 2)

## Why this matters
"What does temperature do?", "What happens if the learning rate is too high?", "Why quantize?" are near-certain interview questions. Strong answers combine a definition, an effect you measured, and a trade-off. This module is the "dials and gauges" of module 04's engine: every parameter is tied to a script you can run, change and break.

## Where this fits
**Builds on**
- [04_llm_fundamentals](../04_llm_fundamentals/) - the **logits** from step 05 (sampling parameters reshape them), the **position table** from step 03 (the context window), the **generation loop** from step 06 (`max_new_tokens`, KV cache), the **training loop** from step 07 (LR/batch/epochs).
- [03_ml_foundations](../03_ml_foundations/) - gradient descent, loss functions, overfitting. Step 06 here re-runs those ideas as parameter sweeps.
- [00_python_foundations](../00_python_foundations/) - dicts, JSON, functions with default/keyword arguments (`**kwargs`), try/except; step 01 uses all of them.

**Leads to**
- [06_llm_training_and_fast_inference](../06_llm_training_and_fast_inference/) - LR/batch/epochs for a real fine-tune; quantization becomes QLoRA / vLLM serving; memory arithmetic and KV-cache become GPU sizing.
- [07_prompt_engineering](../07_prompt_engineering/) - temperature 0 for reproducible prompt tests; max tokens for structured JSON output.
- [08_agent_harness](../08_agent_harness/) - context-window trimming and stop conditions; low temperature for reliable tool calls.
- [09_api_development](../09_api_development/) - these exact parameters become the (validated) fields of a POST request body; `config.json` becomes environment variables/settings.
- [10_ml_system_architecture](../10_ml_system_architecture/) - latency/memory/quality measurements (step 07) feed capacity planning and monitoring.

## Concept map
```
                 +------------------ TRAINING-TIME (fine-tuning) -------------------+
 dataset ---->   | learning rate  batch size  epochs  optimizer(AdamW)  loss(CE)     |  -> better weights
 (module 06)     |   (step size)   (memory,    (passes) (update rule)    (what 'wrong'|     |
                 |                  noise)                                means)      |     |
                 +---------------------------------------------------------------------+     |
                                                                                             v
                 +------------------ MODEL-TIME (what you deploy) --------------------------+
                 | quantization (05): fewer bits per weight -> less memory, faster, small  |
                 |                    accuracy loss         context window (04): hard limit |
                 +----------------------------------------------------------------------+---+
                                                                                        |
                 +------------------ INFERENCE-TIME (per request) ----------------------v--+
 logits -------> | repetition penalty -> temperature -> top-k -> top-p -> softmax -> sample |
 (module 04/05)  | max_new_tokens (length cap)   do_sample False = greedy                   |
                 +----------------------------------------------------------------------+
        ALL of it recorded in config.json (01) and compared with the experiment logger (07)
```

## Study path
| # | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_config_file](01_config_file/) | one validated config for all parameters; parameter glossary | `.venv/bin/python 05_llm_parameters/01_config_file/main.py` | 45 min |
| 2 | [02_sampling_from_scratch](02_sampling_from_scratch/) | temperature, top-k, top-p, repetition penalty implemented on real logits | `.../02_sampling_from_scratch/main.py` | 90 min |
| 3 | [03_hf_generate_compare](03_hf_generate_compare/) | your maths == `generate()`; one-parameter-at-a-time comparisons | `.../03_hf_generate_compare/main.py` | 60 min |
| 4 | [04_max_new_tokens_and_context_window](04_max_new_tokens_and_context_window/) | length cap, context limit, trimming, KV-cache memory | `.../04_max_new_tokens_and_context_window/main.py` | 60 min |
| 5 | [05_quantization](05_quantization/) | int8/int4 by hand, outliers, group-wise scaling | `.../05_quantization/main.py` | 75 min |
| 6 | [06_training_parameters](06_training_parameters/) | LR/batch/epochs/optimizer sweeps + LLM fine-tune LR | `.../06_training_parameters/main.py` | 75 min |
| 7 | [07_experiment_logger](07_experiment_logger/) | measure length, latency, memory, quality into CSV/Markdown | `.../07_experiment_logger/main.py` | 60 min |

Then [exercises.md](exercises.md). Total core study: about 8 hours + exercises.

## Parameter cheat sheet (the thing to memorise)
### Inference-time
| Parameter | Typical values | Raise it to... | Lower it to... | Pitfall |
|---|---|---|---|---|
| `temperature` | 0.0-1.0 (chat ~0.7) | get more varied / creative output | get focused, repeatable output | >1.5 = nonsense; `0` is greedy (HF: use `do_sample=False`) |
| `top_k` | 0 (off), 20-100 | allow rarer tokens | restrict to the few best | ignores how confident the model is |
| `top_p` | 0.8-0.95 | wider candidate set | narrower, safer | usually tune top-p OR temperature, not both wildly |
| `max_new_tokens` | 16-1024+ | allow longer answers (more latency/cost) | cut answers short (may truncate mid-sentence) | prompt+output must fit the context window |
| `repetition_penalty` | 1.0-1.3 | stop loops | allow natural repeats | too high avoids necessary words |
| context window | model-fixed | - | - | overflow = error or silent truncation = forgotten instructions |
| quantization | 16/8/4 bits | (more bits) quality | (fewer bits) memory, speed | 4-bit needs group-wise/outlier handling |

### Training-time
| Parameter | Typical (LLM fine-tune) | Too small | Too large |
|---|---|---|---|
| learning rate | 1e-5 - 5e-4 (LoRA higher: ~1e-4 - 3e-4) | no progress | loss spikes, forgets pretrained knowledge |
| batch size | as large as memory allows (+ gradient accumulation) | noisy | out-of-memory, fewer updates |
| epochs | 1-3 | underfit | overfit |
| optimizer | AdamW | - | Adam keeps 2 extra values per weight (memory!) |
| loss | cross-entropy | - | - |

### Task -> starting settings
| Task | temperature | top_p | max_new_tokens | Note |
|---|---|---|---|---|
| JSON extraction / classification / tool calls | 0 (greedy) | - | small, exact | determinism, easy testing |
| Q&A over facts | 0-0.3 | 0.9 | moderate | reduce hallucination risk |
| Chat | 0.7 | 0.9 | 256-1024 | balance |
| Creative writing / brainstorming | 0.9-1.2 | 0.95 | long | more variety, more errors |

## The ideas in plain English

### Learning rate
How big a step the optimizer takes after each batch. *Analogy:* walking down a foggy hill: tiny steps never arrive, huge steps overshoot the valley. Step 06 showed lr 1e-4 barely learns while lr 10 explodes. Fine-tuning uses small LRs because the model already knows a lot. A **scheduler** (warm-up then decay) is common in practice.

### Batch size
Examples processed per weight update. Larger = smoother gradient + better GPU use but more memory. If memory is short: smaller batch + **gradient accumulation**.

### Epochs
Full passes over the dataset. Watch validation loss; stop when it stops improving (early stopping).

### Optimizers and loss functions
Optimizer = rule that turns gradients into weight changes: SGD (plain), momentum (smooths), Adam/AdamW (per-weight adaptive steps; AdamW = Adam + proper weight decay; default for LLMs). Loss = number to minimise: cross-entropy for next-token prediction/classification, MSE for regression.

### Temperature, top-k, top-p
All reshape the next-token distribution before sampling (step 02). Temperature rescales; top-k and top-p cut off the unlikely tail. Together: "generate the likely, but not always the same".

### Max new tokens and context window
Output cap and input+output ceiling. They drive latency (one forward pass per token), cost (per token) and memory (KV cache).

### Repetition penalty
Lowers the score of tokens already produced; fights loops that greedy/low-temperature decoding creates.

### Quantization
Fewer bits per weight. int8 is almost free; int4 needs care. Enables running bigger models on smaller hardware; costs some accuracy.

### Parameter tuning and trade-offs
Every setting trades **quality vs latency vs memory vs cost vs determinism**. Change one thing at a time, fix the seed, log it (step 07), test on more than one prompt.

## Common mistakes & gotchas
- Changing several parameters at once, then not knowing which caused the effect.
- Comparing sampled outputs without fixing the seed.
- `temperature=0` with `do_sample=True` -> error; use greedy.
- Passing `temperature/top_p` while `do_sample=False` -> silently ignored (HF warns).
- Setting `max_new_tokens` too small -> answers cut mid-sentence / broken JSON.
- Forgetting the context window includes the prompt AND the output.
- Reporting average speed from one run without warm-up.
- Judging quantization by memory only; always re-measure quality.
- Using a high learning rate copied from a from-scratch tutorial for fine-tuning.
- Tuning on the validation set repeatedly until it "looks good" (leaking).

## Self-check
- [ ] I can explain each parameter in `config.json` without looking.
- [ ] I implemented temperature, top-k, top-p, repetition penalty from scratch.
- [ ] I can say which parameter I'd change to make output (a) more deterministic, (b) shorter, (c) less repetitive, (d) cheaper.
- [ ] I can calculate model memory for a dtype and the KV cache size.
- [ ] I can explain why int4 per-tensor failed and group-wise worked.
- [ ] I can recognise too-high LR, too-low LR and overfitting from loss numbers.
- [ ] I produced an experiments table with length, latency, memory and notes.

## Interview questions
1. **What does temperature do?** Divides logits before softmax: <1 sharpens (more deterministic), >1 flattens (more random); 0 limit = greedy. Trade-off: creativity vs reliability.
2. **Top-k vs top-p?** Top-k keeps a fixed number of candidates, top-p keeps the smallest set with cumulative probability >= p, so it adapts to the model's confidence. Often used together with a moderate temperature.
3. **How would you make a model's outputs reproducible?** Greedy decoding (or temperature 0), fixed seed, same model version/hardware; note GPU non-determinism still exists.
4. **What's the context window and what happens when exceeded?** Max tokens model attends to; excess is truncated, summarised, or errors. Keep system prompt + recent turns.
5. **`max_new_tokens` vs context window?** Output cap vs total (prompt + output) capacity.
6. **What is repetition penalty and when is it harmful?** Reduces probability of used tokens; harmful when high because it blocks legitimate repetition (names, code).
7. **Learning rate too high/low? How do you tell?** High: loss spikes/NaN; low: loss barely moves. Fine-tuning LRs are small.
8. **Batch size trade-offs?** Larger: stable gradients, faster per epoch, more memory; smaller: noisier, less memory. Gradient accumulation emulates a bigger batch.
9. **How do you detect overfitting?** Training loss falls while validation loss rises; fix with fewer epochs, more data, regularisation, early stopping.
10. **Why AdamW?** Per-parameter adaptive steps + decoupled weight decay; converges fast and is robust; costs 2 extra states per weight.
11. **What is quantization and its trade-offs?** Fewer bits per weight: less memory/bandwidth, faster; some accuracy loss, outliers need group-wise scaling or calibration.
12. **Memory for a 13B model in fp16 and int4?** ~26 GB and ~6.5 GB for weights, plus KV cache/activations.
13. **How do you choose generation parameters for JSON extraction vs creative writing?** Low/zero temperature and tight max tokens vs higher temperature/top-p and long outputs.
14. **How would you evaluate a parameter change?** Fixed prompts + seeds, one change at a time, log latency/length/memory and a quality metric or human rating, test several prompts.
15. **Why is generation latency proportional to output length?** One forward pass per token; KV cache keeps each pass cheap but doesn't remove the sequential nature.
