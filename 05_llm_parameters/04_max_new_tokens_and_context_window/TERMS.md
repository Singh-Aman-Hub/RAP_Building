# Terms — max_new_tokens, Context Window and KV Cache

Folder: `05_llm_parameters/04_max_new_tokens_and_context_window` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [05 LLM parameters README](../README.md)

**Terms in this folder:** [Context window](#context-window) · [max_new_tokens](#max_new_tokens) · [max_length vs max_new_tokens](#max_length-vs-max_new_tokens) · [n_positions](#n_positions) · [Early stop at end-of-text](#early-stop-at-end-of-text) · [torch.randint](#torchrandint) · [try / except Exception as err](#try--except-exception-as-err) · [type(err).__name__ and str(err)](#typeerr__name__-and-strerr) · [IndexError at the window limit](#indexerror-at-the-window-limit) · [Sliding window trimming](#sliding-window-trimming) · [Negative slicing](#negative-slicing) · [Token budget](#token-budget) · [Forgetting after trimming](#forgetting-after-trimming) · [Words per token](#words-per-token) · [str.split](#strsplit) · [KV cache memory formula](#kv-cache-memory-formula) · [Function with several default arguments](#function-with-several-default-arguments) · [Batch size and KV cache](#batch-size-and-kv-cache) · [RoPE (rotary position embedding)](#rope-rotary-position-embedding) · [Traceback bottom-up reading](#traceback-bottom-up-reading) · [Context-handling strategies](#context-handling-strategies)

---

## Context window  `(concept)`
**1. Definition:** The maximum number of tokens a model can attend to in one go, counting prompt plus generated output.
**2. In plain words:** It is the model's working memory. For distilgpt2 it is 1024 tokens; modern models go from 8,000 to over a million. A chat that keeps growing eventually overflows it, so applications must trim or summarise (module 08).
**3. Example to quote:** > "distilgpt2's window is 1024: a 1,004-token prompt leaves room for only 20 new tokens."

**Seen in:** `main.py`, docstring and `WINDOW`.

## max_new_tokens  `(third-party: transformers)`
**1. Definition:** The `generate` argument capping how many new tokens may be produced after the prompt.
**2. In plain words:** It is your length limit on the answer. Generation can stop earlier if the model emits end-of-text. Prompt tokens plus `max_new_tokens` must fit in the context window.
**3. Example to quote:** > "With `max_new_tokens=20`, the output grew by exactly 20 tokens."

**Seen in:** `main.py`, section 1.

## max_length vs max_new_tokens  `(concept)`
**1. Definition:** `max_length` counts prompt plus output tokens together; `max_new_tokens` counts only the generated part.
**2. In plain words:** Prefer `max_new_tokens`, because with `max_length` a long prompt silently leaves little room for the answer.
**3. Example to quote:** > "`max_length=30` on a 25-token prompt allows only 5 new tokens, but `max_new_tokens=30` always allows 30."

**Seen in:** `main.py`, TRY IT comment.

## n_positions  `(third-party: transformers)`
**1. Definition:** GPT-2 config field giving the number of position embeddings, hence the context window (1024).
**2. In plain words:** The model learned exactly 1024 position vectors, so token 1025 has no position vector to add.
**3. Example to quote:** > "`AutoConfig.from_pretrained('distilgpt2').n_positions` is 1024."

**Seen in:** `main.py`, `WINDOW`.

## Early stop at end-of-text  `(concept)`
**1. Definition:** Generation ends before `max_new_tokens` if the model outputs the end-of-text token.
**2. In plain words:** The cap is a maximum, not a promise. That is why output lengths printed here can be shorter than expected.
**3. Example to quote:** > "My trimmed run returned 1023 tokens, not 1024, because the model stopped early."

**Seen in:** `main.py`, sections 1 and 3.

## torch.randint  `(third-party: torch)`
**1. Definition:** Creates a tensor of random integers in a range with a given shape.
**2. In plain words:** The demo makes 1,030 random token IDs to simulate a too-long prompt. The IDs are meaningless but they are valid.
**3. Example to quote:** > "`torch.randint(0, 50000, (1, 1030))` builds a fake over-length prompt."

**Seen in:** `main.py`, section 2.

## try / except Exception as err  `(syntax)`
**1. Definition:** Catches any exception and gives it the name `err` so you can inspect it.
**2. In plain words:** Catching the broad `Exception` is acceptable only in a demo. In real code you catch specific error types so you do not hide unrelated bugs.
**3. Example to quote:** > "I wrap the over-length call in `try/except` to print the error instead of crashing the demo."

**Seen in:** `main.py`, section 2.

## type(err).__name__ and str(err)  `(Python built-in)`
**1. Definition:** `type(err).__name__` gives the exception class name as a string; `str(err)` gives its message.
**2. In plain words:** Together they let you print "IndexError: index out of range in self". Slicing `[:110]` keeps the message short.
**3. Example to quote:** > "I print `type(err).__name__` and the first 110 characters of `str(err)`."

**Seen in:** `main.py`, section 2.

## IndexError at the window limit  `(concept)`
**1. Definition:** The error raised when a sequence longer than the position table is fed to GPT-2: the lookup index exceeds the 1024 rows.
**2. In plain words:** The failure surfaces in the embedding lookup, far from the real cause, which is that the input is too long. Check `input.shape`.
**3. Example to quote:** > "1,030 tokens gave `IndexError: index out of range in self`: token 1025 has no position vector."

**Seen in:** `main.py`, section 2.

## Sliding window trimming  `(concept)`
**1. Definition:** Keeping only the most recent tokens so that prompt plus output fits in the context window.
**2. In plain words:** The oldest text is dropped. It is simple, but anything dropped is forgotten, including early instructions.
**3. Example to quote:** > "`fit_to_window` keeps the last `1024 - max_new_tokens` prompt tokens."

**Seen in:** `main.py`, `fit_to_window`.

## Negative slicing  `(syntax)`
**1. Definition:** A slice with a negative start such as `x[:, -budget:]` takes elements counted from the end.
**2. In plain words:** It selects the last `budget` columns, which are the most recent tokens.
**3. Example to quote:** > "`[1,2,3,4][-2:]` is `[3, 4]`; I use the same idea on token IDs."
```python
[1, 2, 3, 4][-2:]  # [3, 4]
```
**Seen in:** `main.py`, `fit_to_window`.

## Token budget  `(concept)`
**1. Definition:** The number of tokens you can spend on the prompt after reserving space for the output: `window - max_new_tokens`.
**2. In plain words:** Think of it like a suitcase of fixed size: the output needs room too. If you reserve 20 for the answer, the prompt gets 1004.
**3. Example to quote:** > "My budget is 1024 minus 20, so 1004 prompt tokens."

**Seen in:** `main.py`, `fit_to_window`.

## Forgetting after trimming  `(concept)`
**1. Definition:** Tokens removed from the prompt can no longer influence the model.
**2. In plain words:** The model cannot use information it cannot see. Better strategies: always keep the system prompt, summarise old turns, or retrieve relevant chunks.
**3. Example to quote:** > "If I trim the start of a chat I may lose the instructions, so I always keep the system prompt."

**Seen in:** `main.py`, section 3.

## Words per token  `(concept)`
**1. Definition:** Rough conversion rate between words and tokens; English prose is about 0.75 words per token.
**2. In plain words:** You can estimate token cost from word count. Code, numbers and other languages use more tokens per word.
**3. Example to quote:** > "My 14-word sentence was 17 tokens, 0.82 words per token."

**Seen in:** `main.py`, section 4.

## str.split  `(Python built-in)`
**1. Definition:** Splits a string into a list of pieces on whitespace (by default).
**2. In plain words:** `len(sentence.split())` is a quick word count.
**3. Example to quote:** > "`len('a b c'.split())` is 3."

**Seen in:** `main.py`, section 4.

## KV cache memory formula  `(concept)`
**1. Definition:** KV cache bytes = 2 (K and V) x layers x hidden size x bytes per number x tokens x batch.
**2. In plain words:** Every token stores a Key and a Value vector in every layer. For a 7B-class model at 4,096 tokens, batch 1 is 2.15 GB, but batch 16 is 34.36 GB, more than the weights. This is why serving is limited by memory.
**3. Example to quote:** > "The cache for a 7B model at batch 16 and 4k tokens is about 34 GB, bigger than its 14 GB of weights."
```python
2 * 32 * 4096 * 2 * 4096 * 16 / 1e9  # about 34.4
```
**Seen in:** `main.py`, `kv_cache_gb`.

## Function with several default arguments  `(syntax)`
**1. Definition:** `def kv_cache_gb(layers, hidden, tokens, batch=1, bytes_per_number=2)` has required parameters and optional ones with defaults.
**2. In plain words:** Common calls stay short (`kv_cache_gb(6, 768, 1024)`) while you can override `batch` or the precision.
**3. Example to quote:** > "I call `kv_cache_gb(32, 4096, 4096, 16)` to get the batch-16 figure."

**Seen in:** `main.py`.

## Batch size and KV cache  `(concept)`
**1. Definition:** KV cache memory grows linearly with batch size, because each sequence in the batch has its own cache.
**2. In plain words:** Serving more users at once multiplies cache memory. Techniques like paged attention (module 06) manage that memory.
**3. Example to quote:** > "Going from batch 1 to batch 16 multiplied the cache from 2.15 GB to 34.36 GB."

**Seen in:** `main.py`, section 5.

## RoPE (rotary position embedding)  `(concept)`
**1. Definition:** A way of encoding token positions by rotating query and key vectors, used by many modern models instead of a learned position table.
**2. In plain words:** It lets models handle longer windows more flexibly than GPT-2's fixed 1024 positions, but the same budget logic applies.
**3. Example to quote:** > "Newer models use RoPE and support far longer contexts, but prompt plus output must still fit."

**Seen in:** `WALKTHROUGH.md`.

## Traceback bottom-up reading  `(concept)`
**1. Definition:** Reading an error from the last line (the message) upward to the first frame in your own file.
**2. In plain words:** The deepest frames are inside the library. Find the message, then your first line of code in the trace, then ask which of your inputs is wrong, here the input length.
**3. Example to quote:** > "The error surfaced in the embedding lookup, but the root cause was that `long_ids.shape` was 1030."

**Seen in:** `WALKTHROUGH.md`, debugging section.

## Context-handling strategies  `(concept)`
**1. Definition:** Ways to cope with text longer than the window: truncate, always keep the system prompt, summarise old turns, or retrieve relevant chunks.
**2. In plain words:** Pure truncation is simplest but forgets; summarising and retrieval keep more useful information. Module 08 implements chat trimming.
**3. Example to quote:** > "For long chats I keep the system prompt and summarise earlier turns."

**Seen in:** `WALKTHROUGH.md`.
