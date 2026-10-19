# Step 08 - The generation loop (by hand, then the KV cache)

## The story so far
[Step 07](../07_forward_pass_logits/WALKTHROUGH.md) ended with one forward pass giving one next-token distribution. A sentence needs dozens of tokens, each depending on the one before. So we loop: predict, append, predict again.

## The pain point
The obvious loop feeds the whole text back in at every step. Step t processes t tokens, so N new tokens cost about 1 + 2 + ... + N = N^2 / 2 token-passes. Double the output length and the work roughly quadruples. Most of that work is wasted: the earlier tokens have not changed.

## The spark
Because of the causal mask, an old token's Keys and Values never change once computed. Store them (the **KV cache**) and, at each step, feed in only the newest token. The new token's query reads the stored keys and values. Cost per step becomes roughly constant instead of growing with the text.

## Goal
Write autoregressive decoding yourself and show it equals `model.generate`; understand why KV caching is faster.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/08_generation_loop/main.py
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

seconds to generate N new tokens (greedy; timings vary with machine load):
     N |  no cache |  KV cache | speed-up
    50 |      2.75 |      2.28 |   1.2x
   100 |      7.56 |      3.42 |   2.2x  (no cache x2.7 when N doubled)
   200 |     10.10 |      3.24 |   3.1x  (no cache x1.3 when N doubled)
   400 |     27.31 |      7.77 |   3.5x  (no cache x2.7 when N doubled)

cached loop identical to uncached loop: True
cache for 17 tokens: formula 626,688 bytes | measured 626,688 bytes
distilgpt2: 36,864 bytes per token -> full 1024-token context = 37.7 MB per sequence
7B model, float16: 0.52 MB per token -> 4096 tokens = 2.1 GB PER SEQUENCE (x batch size!)

  step 1: added ' a'     -> 'Machine learning is a'
  step 2: added ' new'   -> 'Machine learning is a new'
  step 3: added ' way'   -> 'Machine learning is a new way'
  step 4: added ' to'    -> 'Machine learning is a new way to'
  step 5: added ' learn' -> 'Machine learning is a new way to learn'
```
Loop: `ids` -> forward pass -> argmax of last position -> append -> repeat until `max_new_tokens` or end-of-text.

- **Autoregressive** = each token is predicted from all previous tokens, including the model's own earlier outputs.
- **Timing is noisy** (this run shared the machine with other jobs, so the 200-token no-cache row looks too fast). The reliable signals: the speed-up grows with N (1.2x, 2.2x, 3.1x, 3.5x), and the no-cache time grows faster than linearly. Run it yourself and expect the same shape, not the same seconds. On a GPU the gap is larger.
- **KV cache (the hero):** attention needs the Keys/Values of every earlier token. Without a cache they are recomputed every step; with a cache each step only computes the new token. The hand-written cached loop gives the identical text.
- **The price is memory:** cache bytes per token = 2 (K and V) x n_layer x d_model x bytes per number. distilgpt2: 2 x 6 x 768 x 4 = 36,864 bytes, and the formula matched the measured tensors exactly (626,688 bytes for 17 tokens). A 7B model in float16 needs 0.52 MB per token, 2.1 GB for one 4096-token sequence, multiplied by the batch size. This is why serving systems such as vLLM page the cache ([06_llm_training_and_fast_inference](../../06_llm_training_and_fast_inference/)).
- The text ends with `\n\n\n...`: greedy decoding of a small model tends to drift into repetition - motivation for sampling and `repetition_penalty` ([05_llm_parameters](../../05_llm_parameters/)).
- Each generated token needs a full forward pass: this is why LLM latency is roughly proportional to output length.

## Experiments
- `do_sample=True` in `generate` -> output differs on each run.
- Add `800` to the N list in section 3; the cache advantage should keep growing.
- Print `len(tok.encode(...))` of the final text to confirm 15 new tokens were added (prompt has 3).

## Debugging practice
Remove `.view(1, 1)`: `RuntimeError: Tensors must have same number of dimensions: got 2 and 0`. It names the two shapes it can't join - compare with what you expected `next_id` to be (a 0-d tensor vs a `(1, seq)` matrix). Practise: add `print(input_ids.shape, next_id.shape)` right before the failing line (print debugging).

## Rebuild checklist
- [ ] Greedy loop: forward, argmax, append, stop on EOS.
- [ ] Verify against `generate(do_sample=False)`.
- [ ] Explain in one sentence what the KV cache stores and why it helps.

## The new problem (cliffhanger)
Our loop always takes the single most likely token (greedy), and the text drifted into `\n\n\n`. Meanwhile we hard-coded everything: 15 or 400 new tokens, float32 weights, `do_sample=False`. And the model itself came from somewhere. Before we turn the knobs, one more chapter: what is the full life cycle of such a model (pretrain, fine-tune, serve)? Next: [09_pretrain_vs_finetune_vs_inference](../09_pretrain_vs_finetune_vs_inference/WALKTHROUGH.md).

> **Say it aloud (interview box):** LLMs generate one token at a time: run the model on the text so far, take the last position's logits, choose a token, append it, repeat. Recomputing the whole history each step costs about N squared work, so the KV cache stores each earlier token's keys and values and only the newest token is processed. The trade-off is memory: the cache needs 2 x layers x d_model x bytes per token, per sequence, which grows with context length and batch size.
