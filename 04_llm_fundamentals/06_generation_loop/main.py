"""
GOAL:   Write the text-generation loop by hand (predict, append, repeat), then show
        model.generate() produces the same thing.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/06_generation_loop/main.py
EXPECTED OUTPUT (short): the hand-written loop and model.generate give IDENTICAL text;
        the KV-cache timing line shows caching is faster.
NEEDS:  torch + transformers (distilgpt2 cached).

LLMs generate ONE token at a time: feed text in, get next-token probabilities, pick one,
append it to the text, repeat.  This is "autoregressive" generation.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()

prompt = "Machine learning is"
ids = tok(prompt, return_tensors="pt")["input_ids"]      # shape (1, prompt_len)

# ---------------------------------------------------------------------------
# 1. The hand-written greedy loop
# ---------------------------------------------------------------------------
def my_generate(input_ids, max_new_tokens=15):
    """Greedy decoding, no cache: recompute the whole sequence every step (simple but slow)."""
    for _ in range(max_new_tokens):                       # '_' = loop variable we don't use
        with torch.no_grad():
            logits = model(input_ids).logits              # (1, seq_len, vocab)
        next_id = logits[0, -1].argmax()                  # best token at the last position (a 0-d tensor)
        # torch.cat joins tensors; next_id.view(1, 1) reshapes it to (1, 1) so it can be glued on the right
        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
        if next_id.item() == tok.eos_token_id:            # stop early if the model says "end of text"
            break
    return input_ids

t0 = time.perf_counter()
mine = my_generate(ids)
t_mine = time.perf_counter() - t0
print("hand-written :", repr(tok.decode(mine[0])))

# ---------------------------------------------------------------------------
# 2. The library way.  do_sample=False means greedy (always argmax) -> should match exactly.
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
lib = model.generate(
    ids,
    max_new_tokens=15,          # how many NEW tokens to add (module 05 step 04)
    do_sample=False,            # greedy decoding
    pad_token_id=tok.eos_token_id,  # silences a warning: GPT-2 has no pad token
)
t_lib = time.perf_counter() - t0
print("model.generate:", repr(tok.decode(lib[0])))
print("identical output:", torch.equal(mine, lib))

# ---------------------------------------------------------------------------
# 3. Why is generate() faster?  KV cache: keys/values of earlier tokens are stored,
#    so each new step only processes the ONE new token instead of the whole text.
# ---------------------------------------------------------------------------
t0 = time.perf_counter()
model.generate(ids, max_new_tokens=60, do_sample=False, use_cache=False, pad_token_id=tok.eos_token_id)
t_nocache = time.perf_counter() - t0
t0 = time.perf_counter()
model.generate(ids, max_new_tokens=60, do_sample=False, use_cache=True, pad_token_id=tok.eos_token_id)
t_cache = time.perf_counter() - t0
print(f"\n60 tokens without KV cache: {t_nocache:.2f}s | with KV cache: {t_cache:.2f}s")
print("(the gap grows with longer outputs; the cache costs GPU memory - see module 06)")

# ---------------------------------------------------------------------------
# 4. Show the loop one token at a time, so you can watch the text grow
# ---------------------------------------------------------------------------
print("\nstep-by-step:")
cur = ids
for step in range(5):
    with torch.no_grad():
        nxt = model(cur).logits[0, -1].argmax().view(1, 1)
    cur = torch.cat([cur, nxt], dim=1)
    print(f"  step {step + 1}: added {tok.decode(nxt[0])!r:10} -> {tok.decode(cur[0])!r}")

# TRY IT: change do_sample=False to True in generate() - outputs now differ run to run (module 05 explains why).
# TRY IT: set max_new_tokens=100 for the hand loop and the cached generate - compare times.
# BREAK IT: remove next_id.view(1, 1) -> torch.cat raises a dimension error; read which shapes it complains about.
# BREAK IT: delete the `ids` argument of generate() -> TypeError / ValueError about missing input_ids.
