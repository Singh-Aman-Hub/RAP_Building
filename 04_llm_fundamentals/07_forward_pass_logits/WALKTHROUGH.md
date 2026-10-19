# Step 07 - Forward pass, logits, softmax, next token

## The story so far
[Step 06](../06_load_pretrained_model/WALKTHROUGH.md) loaded distilgpt2 and counted its 81.9 M numbers, but we never gave it any text. A model that just sits in memory is not useful.

## The pain point
The stack of blocks from [step 05](../05_transformer_block/WALKTHROUGH.md) outputs one vector per token, with 768 numbers each. Nobody can read that. We need an answer in the form "which word comes next, and how sure are you?".

## The spark
Multiply each output vector by a matrix with one row per vocabulary word (50,257 of them). You get 50,257 raw scores per position, called logits. Softmax (the function you wrote in [step 04](../04_attention_from_scratch/WALKTHROUGH.md)) turns them into probabilities. Compare them with the true next token and you get the loss, the number training will push down.

## Goal
Run one forward pass and convert the raw scores (logits) into a probability for every possible next token.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/07_forward_pass_logits/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `model(**inputs)` | `**` unpacks a dict into keyword arguments | `f(**{"a": 1})` == `f(a=1)` |
| `with torch.no_grad():` | context manager: temporary setting restored on exit | - |
| `logits[0, -1, :]` | batch 0, LAST position, all vocabulary entries | `-1` = last index |
| `zip(a, b)` | walk two sequences together | `list(zip([1,2],[3,4]))` = `[(1,3),(2,4)]` |
| `"%.1f" % x` | old-style string formatting | `"%.1f" % 3.14159` -> `'3.1'` |
| `out.logits` | attribute access on a result object | - |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `tok(prompt, return_tensors="pt")` | transformers | tokenise to PyTorch tensors | `return_tensors` | dict-like |
| `model(input_ids, labels=...)` | transformers | forward pass; computes loss if labels given | `attention_mask`, `labels` | object with `.logits`, `.loss` |
| `torch.softmax(x, dim=-1)` | torch | logits -> probabilities along `dim` | `dim` | tensor |
| `torch.topk(x, k)` | torch | k largest values and their indices | `k` | `.values`, `.indices` |
| `torch.argmax(x)` | torch | index of the largest value | - | 0-d tensor |
| `torch.no_grad()` | torch | skip gradient bookkeeping (saves memory) | - | context manager |

## Execution trace
```
input_ids: tensor([[ 464, 3139,  286, 4881,  318]])  shape: (1, 5)       <- 5 tokens
logits shape (batch, seq_len, vocab): (1, 5, 50257)                       <- a score per token, at EVERY position
last-position logits: min -85.5 max -57.4 (not probabilities yet!)
sum of probabilities: 1.000028133392334                                         <- float rounding

Top 5 guesses after 'The capital of France is':
  ' the'       id=   262  p=0.118
  ' home'      id=  1363  p=0.082
  ' now'       id=   783  p=0.050
  ' a'         id=   257  p=0.049
  ' being'     id=   852  p=0.036

greedy next token: ' the'

cross-entropy loss on this prompt: 5.720  (lower = model finds the text more predictable)
loss by hand: 5.720  (same number: average -log p of the true next tokens)
```
- **Logits are not probabilities**: they can be negative and don't sum to 1. Softmax `exp(x_i) / sum(exp(x))` makes them probabilities.
- **Why only the last position?** Position *i* predicts token *i+1*. For generating, only the final one is new. For training, ALL positions are compared with the true next tokens at once - that's why `labels=input_ids` works (the library shifts by one).
- **Small models are weak:** distilgpt2 does not know "Paris" with confidence; its top guess is " the" with 12%. Compare with the top-1 of ~59% for "The best way to learn programming is" -> " to" ([05_llm_parameters/02_sampling_from_scratch](../../05_llm_parameters/02_sampling_from_scratch/)): confidence varies per context.
- **Loss by hand matches the library:** shift by one (position i predicts token i+1), take `-log p` of each true token, average: 5.720 both ways.
- **Loss = surprise.** 5.72 means on average the model gave the true tokens `e^-5.72` ~ 0.3% probability. Perplexity = `e^loss` (see the module's `solutions/`).

## Experiments
- New prompt `"Once upon a"`; list top 5.
- Print entropy `-(probs * probs.log()).sum()` for `"I like"` vs `"The capital of France is"`: which is more uncertain?
- Run with the `labels` argument removed - `out.loss` is `None`.

## Debugging practice
Use `tok.encode(prompt)` (a list) instead of `return_tensors="pt"` and pass it to `model(...)`: `AttributeError: 'list' object has no attribute 'size'` (or similar). The traceback's last frames are inside the library, but scroll up to the first line that is **your** file: that is where the bad input came from.

## Rebuild checklist
- [ ] Tokenise a prompt, forward pass under `no_grad`, take the last position.
- [ ] Softmax, top-5, decode.
- [ ] Explain why the shape is `(1, seq_len, 50257)`.
- [ ] Compute the loss with `labels` and explain what the number means.

## The new problem (cliffhanger)
One forward pass gives ONE next token. A sentence needs dozens, each depending on the last. So we must loop: predict, append, predict again. But every pass re-reads the whole growing text, so the work piles up. Is there a way to stop redoing it? Next: [08_generation_loop](../08_generation_loop/WALKTHROUGH.md).

> **Say it aloud (interview box):** A forward pass turns token ids into logits of shape batch x sequence x vocabulary, and softmax on the last position gives a probability for each of the 50,257 possible next tokens. Greedy picks the argmax, and the cross-entropy loss measures how surprised the model is by the true next tokens. Training lowers that loss; choosing tokens differently from the same probabilities is the subject of [05_llm_parameters](../../05_llm_parameters/).
