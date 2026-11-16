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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Logits | raw scores, unnormalised log-probabilities, pre-softmax outputs | Yes | Real numbers (any sign) of shape `(vocab,)` per position; softmax turns them into probabilities |
| Softmax | normalised exponential | Yes | `exp(x_i) / sum(exp(x))` |
| Cross-entropy | negative log-likelihood (NLL), log loss | Same for one-hot targets | Average of `-log p(true token)` |
| Perplexity | exp(loss) | Related | A loss of 5.72 is a perplexity of about 305: "as unsure as choosing among ~305 equally likely tokens" |
| Greedy decoding | argmax decoding | Yes | Always choose the highest-probability token (module 05 covers alternatives) |
| `labels=input_ids` | teacher forcing, shifted targets | Yes | The library shifts labels by one so position i is scored against token i+1 |
| LM head | output projection, unembedding, `lm_head` | Yes | Linear layer of shape `(768, 50257)` mapping hidden vectors to logits; tied to `wte` in GPT-2 |
| `torch.no_grad()` vs `model.eval()` | inference setup | No | `eval` turns off dropout; `no_grad` turns off gradient recording |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–13)

```python
"""
GOAL:   Run ONE forward pass and turn the model's raw output (logits) into a
        probability for every possible next token.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/07_forward_pass_logits/main.py
EXPECTED OUTPUT (short): logits shape (1, 5, 50257); top-5 next-token guesses with probabilities
        for "The capital of France is"; probabilities sum to 1.0.
NEEDS:  torch + transformers (distilgpt2 cached; loaded in step 06).

The whole job of a causal LLM:  given tokens so far -> probability of EVERY token being next.
    text --tokenizer--> ids --model--> logits --softmax--> probabilities --choose--> next token
"""
```

- **In simple words:** Says the whole job of a causal LLM: tokens in, probability of every next token out.
- **Exactly:** Docstring only. Pipeline: text, tokenizer, ids, model, logits, softmax, probabilities, choose.
- **Why it is here:** A one-line map you can recite in an interview.

### 2. Imports  (`main.py` lines 15–16)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
```

- **In simple words:** PyTorch and the Hugging Face tokenizer and model classes.
- **Exactly:** `AutoModelForCausalLM` is a GPT-style model with the next-token head.
- **Why it is here:** Same loading tools as step 06.

### 3. Load tokenizer and model, set eval mode  (`main.py` lines 18–19)

```python
tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
```

- **In simple words:** Get distilgpt2 ready for prediction.
- **Exactly:** `.eval()` returns the model itself in evaluation mode (dropout off), so loading and eval fit in one line.
- **Why it is here:** Dropout would add randomness to the logits.

### 4. Tokenise the prompt into tensors  (`main.py` lines 21–23)

```python
prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt")        # dict with input_ids and attention_mask
print("input_ids:", inputs["input_ids"], " shape:", tuple(inputs["input_ids"].shape))   # shape: (batch=1, seq_len=5)
```

- **In simple words:** Turn the sentence into ids the model can read.
- **Exactly:** `tok(prompt, return_tensors="pt")` returns a `BatchEncoding` with `input_ids` and `attention_mask`, both `(1, 5)`. Real ids: `[[464, 3139, 286, 4881, 318]]` for "The", " capital", " of", " France", " is". Shape is `(batch=1, seq_len=5)`.
- **Why it is here:** The model accepts tensors, not strings; the batch axis is required even for one prompt.

### 5. One forward pass without gradients  (`main.py` lines 25–29)

```python
# ---------------------------------------------------------------------------
# 1. Forward pass.  torch.no_grad() = don't record gradients (we are not training) -> less memory.
# ---------------------------------------------------------------------------
with torch.no_grad():                            # 'with' = temporary setting that auto-restores
    out = model(**inputs)                        # ** unpacks the dict into keyword arguments
```

- **In simple words:** Run the model once, without recording what is needed for training.
- **Exactly:** `with torch.no_grad():` disables autograd inside the block, saving memory. `model(**inputs)` unpacks the dict to `model(input_ids=..., attention_mask=...)` and returns a `CausalLMOutputWithPast` object `out`.
- **Why it is here:** Without `no_grad` the output is the same but memory use rises as the graph is stored.

### 6. The logits tensor  (`main.py` lines 31–32)

```python
logits = out.logits                              # shape: (1, 5, 50257) = (batch, seq_len, vocab); raw scores, one per vocabulary token
print("logits shape (batch, seq_len, vocab):", tuple(logits.shape))
```

- **In simple words:** The model's raw answer: one score for every possible next token, at every position.
- **Exactly:** `out.logits` has shape `(1, 5, 50257)` = (batch, seq_len, vocab). That is 5 x 50,257 = 251,285 scores.
- **Why it is here:** Every position makes its own next-token prediction; this is what lets training use all positions at once.

### 7. Take the last position  (`main.py` lines 34–39)

```python
# ---------------------------------------------------------------------------
# 2. We only care about the LAST position: it predicts the token AFTER the prompt.
#    (Every position predicts its own next token; that is how training works in step 09.)
# ---------------------------------------------------------------------------
last_logits = logits[0, -1, :]                   # shape: (50257,) one score per vocabulary token
print("last-position logits: min %.1f max %.1f (not probabilities yet!)" % (last_logits.min(), last_logits.max()))
```

- **In simple words:** Only the final position predicts what comes after our prompt, so keep only that row.
- **Exactly:** `logits[0, -1, :]` picks batch 0, last position, all vocab entries: shape `(50257,)`. Real output: min `-85.5`, max `-57.4`. They are negative and not probabilities.
- **Why it is here:** The other positions predict tokens we already know. Using the wrong position is a classic bug.

### 8. Softmax turns logits into probabilities  (`main.py` lines 41–45)

```python
# ---------------------------------------------------------------------------
# 3. Softmax: logits -> probabilities (positive, sum to 1).  Same function you wrote in numpy in step 04.
# ---------------------------------------------------------------------------
probs = torch.softmax(last_logits, dim=-1)   # shape: (50257,) positive, sums to 1
print("sum of probabilities:", probs.sum().item())
```

- **In simple words:** Convert the scores into positive numbers that add up to 1.
- **Exactly:** `torch.softmax(last_logits, dim=-1)` gives shape `(50257,)`. Real sum `1.000028...`, equal to 1 within float32 rounding. It is the same function you wrote in numpy in step 04.
- **Why it is here:** Without it you cannot say "12% sure"; probabilities are what sampling and the loss need.

### 9. The five most likely next tokens  (`main.py` lines 47–53)

```python
# ---------------------------------------------------------------------------
# 4. Top-5 candidates
# ---------------------------------------------------------------------------
top = torch.topk(probs, k=5)                     # .values shape: (5,), .indices shape: (5,)
print(f"\nTop 5 guesses after {prompt!r}:")
for p, idx in zip(top.values, top.indices):      # zip walks two lists in step
    print(f"  {tok.decode(idx)!r:12} id={idx.item():6d}  p={p.item():.3f}")
```

- **In simple words:** List the model's top guesses and how confident it is in each.
- **Exactly:** `torch.topk(probs, k=5)` returns `.values` and `.indices`, each `(5,)`, sorted descending. `tok.decode(idx)` turns each id back into text. Real output: `' the' id=262 p=0.118`, `' home' 1363 0.082`, `' now' 783 0.050`, `' a' 257 0.049`, `' being' 852 0.036`. Note the leading spaces: tokens carry them.
- **Why it is here:** The top 5 sum to only about 0.33: a small model is unsure, so probability mass is spread over many tokens.

### 10. Greedy choice  (`main.py` lines 55–59)

```python
# ---------------------------------------------------------------------------
# 5. Greedy choice = argmax.  Sampling instead of argmax is module 05's whole topic.
# ---------------------------------------------------------------------------
next_id = torch.argmax(probs).item()          # a plain int: the id of the most likely token
print("\ngreedy next token:", repr(tok.decode(next_id)))
```

- **In simple words:** Pick the single most likely token.
- **Exactly:** `torch.argmax(probs).item()` is a plain int, here `262`; `tok.decode(262)` is `' the'`.
- **Why it is here:** This is the decoding rule of the generation loop in step 08; module 05 replaces it with temperature, top-k and top-p sampling.

### 11. Loss with the library  (`main.py` lines 61–67)

```python
# ---------------------------------------------------------------------------
# 6. Loss = how surprised the model is by the true next token (cross-entropy).
#    Passing labels=input_ids makes the model compute it (labels are shifted internally).
# ---------------------------------------------------------------------------
with torch.no_grad():
    loss = model(**inputs, labels=inputs["input_ids"]).loss      # scalar
print(f"\ncross-entropy loss on this prompt: {loss.item():.3f}  (lower = model finds the text more predictable)")
```

- **In simple words:** Measure how surprised the model is by the actual words of the prompt.
- **Exactly:** Passing `labels=input_ids` makes the model compute the cross-entropy, shifting internally so position i is compared with token i+1; result is a scalar `loss`. Real output: `5.720`. This means that on average the true tokens got probability `e^-5.72` which is about 0.3%.
- **Why it is here:** This number is what training minimises (step 09).

### 12. Loss by hand  (`main.py` lines 69–76)

```python
# 6b. Loss by hand, to see there is no magic: position i predicts token i+1, so drop the last
#     logit row and the first label, then average -log(prob of the true token).
with torch.no_grad():
    all_logits = model(**inputs).logits                          # shape: (1, 5, 50257)
log_probs = torch.log_softmax(all_logits[0, :-1], dim=-1)        # shape: (4, 50257)  positions 0..3
true_next = inputs["input_ids"][0, 1:]                           # shape: (4,)        tokens 1..4
picked = log_probs[torch.arange(len(true_next)), true_next]       # shape: (4,)        log-prob of each true token
print(f"loss by hand: {-picked.mean().item():.3f}  (same number: average -log p of the true next tokens)")
```

- **In simple words:** Recompute the loss ourselves to prove there is no magic.
- **Exactly:** `all_logits` is `(1, 5, 50257)`. `all_logits[0, :-1]` drops the last position (it has no known next token) giving `(4, 50257)`; `log_softmax` gives log-probabilities. `input_ids[0, 1:]` is the 4 true next tokens. `log_probs[torch.arange(4), true_next]` picks one log-probability per row: shape `(4,)`. `-picked.mean()` is the cross-entropy: `5.720`, identical to the library's.
- **Why it is here:** Understanding the shift (position i predicts token i+1) is the single most common source of bugs in custom training code. (Fix noted: the original used a hard-coded `torch.arange(4)`; it is now `len(true_next)` so a longer prompt does not crash.)

### 13. Experiments and break-it comments  (`main.py` lines 78–81)

```python
# TRY IT: change the prompt to "Once upon a" and watch the top-5 change.
# TRY IT: print the entropy -(probs * probs.log()).sum() for a confident prompt vs a vague one ("I like").
# BREAK IT: remove `with torch.no_grad()` - output is the same but memory use rises (graph is stored).
# BREAK IT: pass inputs["input_ids"] without return_tensors="pt" (use tok.encode) -> read the error: lists have no .shape.
```

- **In simple words:** Try other prompts, entropy, and see two common errors.
- **Exactly:** Entropy is `-(probs * probs.log()).sum()`. Passing a plain Python list to the model fails because lists have no `.shape`/`.size`.
- **Why it is here:** Practice reading tracebacks whose last frames are inside a library.

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
