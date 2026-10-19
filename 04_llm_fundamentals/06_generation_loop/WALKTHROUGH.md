# Step 06 - The generation loop (by hand, then `generate`)

## Goal
Write autoregressive decoding yourself and show it equals `model.generate`; understand why KV caching is faster.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/06_generation_loop/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `def f(a, max_new_tokens=15):` | default argument value | `f(x)` uses 15 |
| `for _ in range(n):` | `_` = "I don't need the counter" | - |
| `if cond: break` | exit the loop early | - |
| `x.view(1, 1)` | reshape a tensor without copying | `torch.tensor(5).view(1,1).shape` = `(1, 1)` |
| `torch.cat([a, b], dim=1)` | join along columns (append tokens) | `(1,3)+(1,1)` -> `(1,4)` |
| `time.perf_counter()` | high-resolution timer; subtract two readings | - |
| `out[0][ids.shape[1]:]` | slice: skip the first N items | drops the prompt |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `time.perf_counter` | stdlib | stopwatch | - | float seconds |
| `model.generate(ids, ...)` | transformers | full decoding loop with cache, sampling, stopping | `max_new_tokens`, `do_sample`, `temperature`, `top_k`, `top_p`, `repetition_penalty`, `use_cache`, `pad_token_id` | tensor of ids (prompt + new) |
| `tok.eos_token_id` | transformers | the "end of text" id (50256) | - | int |
| `torch.equal(a, b)` | torch | exact tensor equality | - | bool |

## Execution trace
```
hand-written : 'Machine learning is a new way to learn about the world.\n\n\n\n\n\n'
model.generate: 'Machine learning is a new way to learn about the world.\n\n\n\n\n\n'
identical output: True

60 tokens without KV cache: 1.40s | with KV cache: 0.81s

  step 1: added ' a'     -> 'Machine learning is a'
  step 2: added ' new'   -> 'Machine learning is a new'
  step 3: added ' way'   -> 'Machine learning is a new way'
  ...
```
Loop: `ids` -> forward pass -> argmax of last position -> append -> repeat until `max_new_tokens` or end-of-text.

- **Autoregressive** = each token is predicted from all previous tokens, including the model's own earlier outputs.
- **KV cache:** attention needs the Keys/Values of every earlier token. Without a cache they are recomputed every step (cost grows with length); with a cache each step only computes the new token. Trade-off: speed for memory (see module 05 step 04 for the formula, module 06 for vLLM's paged cache).
- The text ends with `\n\n\n...`: greedy decoding of a small model tends to drift into repetition - motivation for sampling and `repetition_penalty` (module 05).
- Each generated token needs a full forward pass: this is why LLM latency is roughly proportional to output length.

## Experiments
- `do_sample=True` in `generate` -> output differs on each run.
- `max_new_tokens=100` for both methods; compare timing. The cache advantage should grow.
- Print `len(tok.encode(...))` of the final text to confirm 15 new tokens were added (prompt has 3).

## Debugging practice
Remove `.view(1, 1)`: `RuntimeError: Tensors must have same number of dimensions: got 2 and 0`. It names the two shapes it can't join - compare with what you expected `next_id` to be (a 0-d tensor vs a `(1, seq)` matrix). Practise: add `print(input_ids.shape, next_id.shape)` right before the failing line (print debugging).

## Rebuild checklist
- [ ] Greedy loop: forward, argmax, append, stop on EOS.
- [ ] Verify against `generate(do_sample=False)`.
- [ ] Explain in one sentence what the KV cache stores and why it helps.

## Say it aloud
"LLMs generate one token at a time: run the model on the text so far, take the last position's logits, choose a token, append it, repeat. I wrote that loop by hand and it matches `model.generate` with greedy decoding. `generate` is faster because of the KV cache, which stores each earlier token's keys and values instead of recomputing them. Trade-off: the cache uses GPU memory that grows with sequence length and batch size."
