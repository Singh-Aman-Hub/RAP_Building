# Step 04 - `max_new_tokens`, the context window, and the KV cache

## Goal
Know what happens at the edge of the context window, how to trim a prompt safely, and how much memory the KV cache needs.

## Run it
```bash
.venv/bin/python 05_llm_parameters/04_max_new_tokens_and_context_window/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `x[:, -budget:]` | negative slice: the LAST `budget` columns | `[1,2,3,4][-2:]` = `[3, 4]` |
| `try: ... except Exception as err:` | catch an error and inspect it | `type(err).__name__`, `str(err)` |
| `str(err)[:110]` | slice a string to its first 110 characters | - |
| `torch.randint(low, high, shape)` | random integers | - |
| function with several default args | `def kv_cache_gb(layers, hidden, tokens, batch=1, bytes_per_number=2)` | `kv_cache_gb(6, 768, 1024)` |
| `len(text.split())` | word count by whitespace | - |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `AutoConfig.from_pretrained(name).n_positions` | transformers | context window of GPT-2 models | - | int |
| `model.generate(..., max_new_tokens=n)` | transformers | cap on NEW tokens | `max_new_tokens` (vs `max_length` = prompt+new) | ids |
| `tok.encode(text)` | transformers | count tokens with `len(...)` | - | list |

## Execution trace
```
context window: 1024
max_new_tokens=  5 -> produced 5 new tokens
max_new_tokens= 50 -> produced 50 new tokens

1030 tokens -> IndexError: index out of range in self
   (the position-embedding table only has 1024 rows - token #1025 has no position vector)

trimmed prompt length: 1004 + 20 new tokens = 1024 <= 1024
generation after trimming works; output length: 1023   (the model may emit end-of-text early)

14 words -> 17 tokens (0.82 words/token)       <- English prose is usually ~0.75 words/token

KV cache size (float16):
  distilgpt2 (6 layers, 768), 1024 tokens, batch 1 : 0.0189 GB
  7B-class model (32 layers, 4096), 4096 tokens, batch 1 : 2.15 GB
  same model, batch 16                                     : 34.36 GB
```
- **Context window** = prompt + output must fit. For GPT-2 it is a hard limit because there are only 1024 learned position vectors (step 03 of module 04). Newer models use RoPE and are trained/extended to much longer windows, but the same budget logic applies.
- **`max_new_tokens` vs `max_length`**: the first limits only the generated part, the second counts the prompt too. Prefer `max_new_tokens`.
- **Longer outputs cost more** time linearly (one forward pass per token) and money (most APIs bill per token).
- **KV cache formula:** `2 x layers x hidden x bytes x tokens x batch`. At 7B scale with batch 16 the cache alone (34 GB) outgrows the 14 GB of weights - the real reason serving throughput is limited by memory (module 06's vLLM "PagedAttention" manages this memory).
- **Trimming = forgetting.** Keeping only the last N tokens loses early instructions. Better strategies: always keep the system prompt, summarise old turns, or retrieve relevant chunks. You will implement trimming for chats in module 08 (`solutions/ex04_chat_trimmer.py` here is a preview).

## Experiments
- Prompt of 1000 tokens + `max_new_tokens=100`: what happens as generation crosses 1024?
- Replace `max_new_tokens=20` with `max_length=30` and compare lengths.
- Compute the KV cache for a 70B-like model (80 layers, 8192 hidden) at 8k tokens.

## Debugging practice
Remove the trim and run `model(long_ids)`. The traceback's deepest frames are inside `transformers`/`torch` (`wpe` embedding lookup). Technique: scan the traceback **bottom-up** for the error message (`IndexError: index out of range in self`), then find the first frame in **your** file, then ask "which of my inputs is too large?" - check `long_ids.shape`. The root cause (input length) is far from where the error surfaced (embedding lookup).

## Rebuild checklist
- [ ] Explain `max_new_tokens` vs context window.
- [ ] Write `fit_to_window`.
- [ ] Compute KV cache memory by hand for given sizes.
- [ ] Name two ways to handle prompts longer than the window.

## Say it aloud
"The context window is the maximum number of tokens the model can attend to, prompt plus output. `max_new_tokens` caps what I generate; if prompt plus output exceeds the window the model fails or must be trimmed, so I keep the most recent tokens and always keep the system prompt. Each token also adds to the KV cache, which grows with context length and batch size and can dominate GPU memory. Trade-off: longer context gives the model more information but increases latency, memory and cost."
