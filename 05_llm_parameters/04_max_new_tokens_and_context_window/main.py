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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
cfg = AutoConfig.from_pretrained("distilgpt2")
WINDOW = cfg.n_positions          # 1024 - number of position embeddings (module 04 step 03)
print("context window:", WINDOW)

# ---------------------------------------------------------------------------
# 1. max_new_tokens is a hard cap
# ---------------------------------------------------------------------------
ids = tok("Once upon a time", return_tensors="pt")["input_ids"]
for n in (5, 20, 50):
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    print(f"max_new_tokens={n:3d} -> produced {out.shape[1] - ids.shape[1]} new tokens")
# Note: generation can stop EARLIER if the model emits the end-of-text token.

# ---------------------------------------------------------------------------
# 2. Overflowing the context window
# ---------------------------------------------------------------------------
long_ids = torch.randint(0, 50000, (1, WINDOW + 6))      # 1030 random token ids
try:
    model(long_ids)
except Exception as err:                                  # catching broad Exception ONLY because this is a demo
    print(f"\n{WINDOW + 6} tokens -> {type(err).__name__}: {str(err)[:110]}")
    print("   (the position-embedding table only has 1024 rows - token #1025 has no position vector)")

# ---------------------------------------------------------------------------
# 3. Fix: keep the most recent tokens ("sliding window") so prompt + output fit.
# ---------------------------------------------------------------------------
def fit_to_window(input_ids, max_new_tokens, window=WINDOW):
    """Keep only the LAST (window - max_new_tokens) prompt tokens."""
    budget = window - max_new_tokens                      # tokens we may spend on the prompt
    return input_ids[:, -budget:]                         # negative slice = take from the end

trimmed = fit_to_window(long_ids, max_new_tokens=20)
print("\ntrimmed prompt length:", trimmed.shape[1], "+ 20 new tokens =", trimmed.shape[1] + 20, "<= 1024")
out = model.generate(trimmed, max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
print("generation after trimming works; output length:", out.shape[1], "(can be shorter: the model may emit end-of-text early)")
print("Trade-off: the dropped tokens are FORGOTTEN - the model cannot use information it can't see.")

# ---------------------------------------------------------------------------
# 4. How many tokens is my text?  Rule of thumb: ~0.75 English words per token.
# ---------------------------------------------------------------------------
sentence = "Large language models read tokens, not words, and every token costs time and memory."
n_words, n_tokens = len(sentence.split()), len(tok.encode(sentence))
print(f"\n{n_words} words -> {n_tokens} tokens ({n_words / n_tokens:.2f} words/token)")

# ---------------------------------------------------------------------------
# 5. KV-cache memory (why long context + big batches exhaust GPU memory; module 06 step on latency)
#    per token we store a Key and a Value vector in EVERY layer:
#    bytes = 2 (K and V) * layers * hidden_size * bytes_per_number * tokens * batch
# ---------------------------------------------------------------------------
def kv_cache_gb(layers, hidden, tokens, batch=1, bytes_per_number=2):
    return 2 * layers * hidden * bytes_per_number * tokens * batch / 1e9

print("\nKV cache size (float16):")
print(f"  distilgpt2 (6 layers, 768), 1024 tokens, batch 1 : {kv_cache_gb(6, 768, 1024):.4f} GB")
print(f"  7B-class model (32 layers, 4096), 4096 tokens, batch 1 : {kv_cache_gb(32, 4096, 4096):.2f} GB")
print(f"  same model, batch 16                                     : {kv_cache_gb(32, 4096, 4096, 16):.2f} GB")

# TRY IT: generate with a prompt of 1000 tokens and max_new_tokens=100 (what happens at position 1024?).
# TRY IT: switch to max_length=30 in generate() - it counts prompt+output together, unlike max_new_tokens.
# BREAK IT: remove the trimming and call generate(long_ids, ...) - read the traceback bottom-up: which layer fails?
# BREAK IT: set max_new_tokens=2000 on a short prompt - what does HF do at the 1024 limit?
