# Step 04 - `max_new_tokens`, the context window, and the KV cache

## The story so far
Sampling now gives good text ([step 03](../03_hf_generate_compare/WALKTHROUGH.md)). The next problem is not what the model says but how much it can say, and how much it can read.

## The pain point
Set `max_new_tokens` too small and answers stop mid-sentence (or leave broken JSON). Feed in a prompt plus answer longer than the model's window and you get an error or silently dropped text. GPT-2 has only 1024 position vectors, so token 1025 has no position at all.

## The spark
Treat the context window as a budget: prompt tokens plus new tokens must fit. Count tokens first, trim the oldest chat turns (keep the system prompt), and cap the output. Also notice what each token costs: the KV cache grows with every token in the window.

## Goal
Know what happens at the edge of the context window, how to trim a prompt safely, and how much memory the KV cache needs.

## Run it
```bash
.venv/bin/python 05_llm_parameters/04_max_new_tokens_and_context_window/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Context window | context length, max sequence length, `n_positions`, `max_position_embeddings`, "n_ctx" | Yes | Max tokens (prompt + output) the model can attend to at once |
| `max_new_tokens` | max output tokens, `max_tokens` (OpenAI), `max_completion_tokens` | Roughly | Cap on tokens GENERATED; provider APIs name it differently |
| `max_length` | total length cap | No | Counts prompt plus output together; older HF parameter |
| Sliding window / truncation | trimming, left-truncation | Related | Keep only the most recent tokens; a different thing from "sliding-window attention" inside a model (Mistral) |
| KV cache | past key values, attention cache | Yes | Stored K and V of every previous token in every layer |
| EOS token | end-of-text, stop token | Yes | The model may stop before the cap |
| Token vs word | tokenisation ratio | No | About 0.75 English words per token (here 0.82 on a sample sentence) |
| Long-context models | 128k / 1M context | Same concept, bigger | Use position schemes such as RoPE with scaling, not GPT-2's learned table of 1,024 rows |

## Code, step by step

### 1. Module docstring: definitions  (`main.py` lines 1–18)

```python
"""
GOAL:   Understand max_new_tokens vs the context window: what happens when the prompt is
        too long, how to trim it, and how much memory the KV cache needs.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/04_max_new_tokens_and_context_window/main.py
EXPECTED OUTPUT (short): generation stops exactly at max_new_tokens; a 1030-token input crashes
        with an index error (limit 1024); trimmed input works; a KV-cache size table.
NEEDS:  torch + transformers (distilgpt2 cached).

Definitions
  context window   : the MAXIMUM number of tokens the model can look at in one go = prompt + output.
                     distilgpt2/gpt2: 1024.  Modern LLMs: 8k - 1M+.
  max_new_tokens   : YOUR cap on how many tokens to GENERATE.  Prompt tokens + max_new_tokens must
                     fit inside the context window.
Why it matters for products: a chat that keeps growing eventually overflows the window; the agent
harness in module 08 (step on context management) must trim or summarise to stay inside it.
"""
```

- **In simple words:** Defines context window (everything the model can see) and `max_new_tokens` (your cap on what it writes).
- **Exactly:** Docstring only. Prompt tokens plus `max_new_tokens` must fit in the context window (1,024 for distilgpt2 and GPT-2; modern LLMs 8k to 1M+).
- **Why it is here:** This simple inequality is the cause of "context length exceeded" errors in production chat apps.

### 2. Load model and read the window size  (`main.py` lines 20–27)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
cfg = AutoConfig.from_pretrained("distilgpt2")
WINDOW = cfg.n_positions          # 1024 - number of position embeddings (module 04 step 03)
print("context window:", WINDOW)
```

- **In simple words:** Get the model and ask its config how long an input it supports.
- **Exactly:** `cfg.n_positions` is `1024`, the number of rows in the learned position-embedding table (module 04 step 03). Real output: `context window: 1024`.
- **Why it is here:** Never hard-code 1024; read it from the config so a different model works.

### 3. max_new_tokens is a hard cap  (`main.py` lines 29–36)

```python
# ---------------------------------------------------------------------------
# 1. max_new_tokens is a hard cap
# ---------------------------------------------------------------------------
ids = tok("Once upon a time", return_tensors="pt")["input_ids"]   # shape: (1, 4)
for n in (5, 20, 50):
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)   # shape: (1, 4 + n)
    print(f"max_new_tokens={n:3d} -> produced {out.shape[1] - ids.shape[1]} new tokens")
# Note: generation can stop EARLIER if the model emits the end-of-text token.
```

- **In simple words:** Ask for 5, 20 and 50 new tokens and count what you get.
- **Exactly:** `ids` is `(1, 4)` for "Once upon a time". `generate(..., max_new_tokens=n)` returns `(1, 4 + n)`. Real output: produced 5, 20 and 50 new tokens. It can be fewer if the model emits the end-of-text token first.
- **Why it is here:** A cap keeps cost and latency predictable.

### 4. What happens when the input is too long  (`main.py` lines 38–46)

```python
# ---------------------------------------------------------------------------
# 2. Overflowing the context window
# ---------------------------------------------------------------------------
long_ids = torch.randint(0, 50000, (1, WINDOW + 6))      # shape: (1, 1030) random token ids
try:
    model(long_ids)
except Exception as err:                                  # catching broad Exception ONLY because this is a demo
    print(f"\n{WINDOW + 6} tokens -> {type(err).__name__}: {str(err)[:110]}")
    print("   (the position-embedding table only has 1024 rows - token #1025 has no position vector)")
```

- **In simple words:** Feed 1,030 random tokens, which is 6 more than the model has position slots for.
- **Exactly:** `torch.randint(0, 50000, (1, 1030))` gives random ids `(1, 1030)`. The forward pass raises `IndexError: index out of range in self`: the position table `wpe` has only 1,024 rows, so position 1024 does not exist. The broad `except Exception` is for demonstration only.
- **Why it is here:** Models with learned absolute positions hard-fail beyond the window; others degrade quality silently. Knowing the real error helps you diagnose it.

### 5. Fix by keeping only the most recent tokens  (`main.py` lines 48–60)

```python
# ---------------------------------------------------------------------------
# 3. Fix: keep the most recent tokens ("sliding window") so prompt + output fit.
# ---------------------------------------------------------------------------
def fit_to_window(input_ids, max_new_tokens, window=WINDOW):
    """Keep only the LAST (window - max_new_tokens) prompt tokens."""
    budget = window - max_new_tokens                      # tokens we may spend on the prompt
    return input_ids[:, -budget:]                         # shape: (1, budget); negative slice = take from the end

trimmed = fit_to_window(long_ids, max_new_tokens=20)
print("\ntrimmed prompt length:", trimmed.shape[1], "+ 20 new tokens =", trimmed.shape[1] + 20, "<= 1024")
out = model.generate(trimmed, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
print("generation after trimming works; output length:", out.shape[1], "(can be shorter: the model may emit end-of-text early)")
print("Trade-off: the dropped tokens are FORGOTTEN - the model cannot use information it can't see.")
```

- **In simple words:** Cut the oldest tokens so the prompt plus the output fits.
- **Exactly:** `budget = window - max_new_tokens = 1004`; `input_ids[:, -budget:]` keeps the last 1,004 columns, shape `(1, 1004)`. 1004 + 20 = 1024 fits. Real output: `output length: 1024`.
- **Why it is here:** A growing chat history must be trimmed or summarised; the cost is that dropped tokens are forgotten.

### 6. How many tokens is my text?  (`main.py` lines 62–67)

```python
# ---------------------------------------------------------------------------
# 4. How many tokens is my text?  Rule of thumb: ~0.75 English words per token.
# ---------------------------------------------------------------------------
sentence = "Large language models read tokens, not words, and every token costs time and memory."
n_words, n_tokens = len(sentence.split()), len(tok.encode(sentence))
print(f"\n{n_words} words -> {n_tokens} tokens ({n_words / n_tokens:.2f} words/token)")
```

- **In simple words:** Compare words and tokens for a sentence.
- **Exactly:** `len(sentence.split())` is 14 words and `len(tok.encode(sentence))` is 17 tokens: 0.82 words per token on this sample. The typical rule of thumb is about 0.75 for English prose.
- **Why it is here:** Prices, limits and speed are in tokens, so you must estimate token counts.

### 7. KV-cache size formula  (`main.py` lines 69–75)

```python
# ---------------------------------------------------------------------------
# 5. KV-cache memory (why long context + big batches exhaust GPU memory; module 06 step on latency)
#    per token we store a Key and a Value vector in EVERY layer:
#    bytes = 2 (K and V) * layers * hidden_size * bytes_per_number * tokens * batch
# ---------------------------------------------------------------------------
def kv_cache_gb(layers, hidden, tokens, batch=1, bytes_per_number=2):
    return 2 * layers * hidden * bytes_per_number * tokens * batch / 1e9
```

- **In simple words:** Each generated token stores a Key and a Value vector in every layer; add up the bytes.
- **Exactly:** `bytes = 2 (K and V) x layers x hidden x bytes_per_number x tokens x batch`. Default 2 bytes (float16).
- **Why it is here:** This is why long contexts and big batches run out of GPU memory even when the weights fit.

### 8. KV-cache examples  (`main.py` lines 77–80)

```python
print("\nKV cache size (float16):")
print(f"  distilgpt2 (6 layers, 768), 1024 tokens, batch 1 : {kv_cache_gb(6, 768, 1024):.4f} GB")
print(f"  7B-class model (32 layers, 4096), 4096 tokens, batch 1 : {kv_cache_gb(32, 4096, 4096):.2f} GB")
print(f"  same model, batch 16                                     : {kv_cache_gb(32, 4096, 4096, 16):.2f} GB")
```

- **In simple words:** Compute the cache for a small and a 7B-class model.
- **Exactly:** Real output: distilgpt2 at 1,024 tokens, batch 1: `0.0189 GB`. A 7B-class model (32 layers, 4,096 hidden) at 4,096 tokens: `2.15 GB`; the same at batch 16: `34.36 GB`. Models with grouped-query attention (GQA) use far less because they store fewer K/V heads.
- **Why it is here:** Explains why serving engines (vLLM, paged attention) focus on cache management.

### 9. Experiments and break-it comments  (`main.py` lines 82–85)

```python
# TRY IT: generate with a prompt of 1000 tokens and max_new_tokens=100 (what happens at position 1024?).
# TRY IT: switch to max_length=30 in generate() - it counts prompt+output together, unlike max_new_tokens.
# BREAK IT: remove the trimming and call generate(long_ids, ...) - read the traceback bottom-up: which layer fails?
# BREAK IT: set max_new_tokens=2000 on a short prompt - what does HF do at the 1024 limit?
```

- **In simple words:** Try a 1,000-token prompt, use `max_length`, or ask for 2,000 new tokens.
- **Exactly:** `max_length` counts prompt plus output. Asking for more than the window allows runs into the position limit.
- **Why it is here:** Practice reading which layer fails (the embedding lookup) in the traceback.

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

## The new problem (cliffhanger)
Output is controlled and the window is respected, but the model itself does not fit: distilgpt2 is small, a 7B model in float32 needs 28 GB. How do we squeeze it into the GPU we have? Next: [step 05, quantization](../05_quantization/WALKTHROUGH.md).

> **Say it aloud (interview box):** The context window is a hard budget that covers the prompt and the output together, and max_new_tokens only caps the output part. I count tokens first, trim the oldest turns while keeping the system prompt, and expect answers cut mid-sentence if the cap is too small. Every token in the window also costs KV-cache memory, so longer context means more memory and slower generation.
