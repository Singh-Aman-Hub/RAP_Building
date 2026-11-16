# Step 02 - Sampling from scratch: temperature, top-k, top-p, repetition penalty

## The story so far
The config in [step 01](../01_config_file/WALKTHROUGH.md) lists `temperature`, `top_k` and `top_p`, but so far every generation used greedy decoding: always take the top token.

## The pain point
Greedy is boring and repetitive. It gives the same answer every time, and often falls into loops such as "the the the" or a repeated sentence. Always choosing the single most probable token is not the same as writing well.

## The spark
The model does not output one token, it outputs a probability for all 50,257 of them. Reshape that distribution before drawing from it: temperature sharpens or flattens it, top-k and top-p cut off the unlikely tail, repetition penalty demotes tokens already used. Then roll a weighted die.

## Goal
Implement the four decoding controls yourself on the real logits of distilgpt2 and see exactly how each reshapes the probability distribution.

## Run it
```bash
.venv/bin/python 05_llm_parameters/02_sampling_from_scratch/main.py
```
Functions live in `sampling.py`; `main.py` demonstrates them.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Temperature | T, softmax temperature, sampling temperature | Yes | Divides logits before softmax; T<1 sharper, T>1 flatter; T=0 is not allowed (use greedy) |
| Top-k | k-sampling, top-K filtering | Yes | Keep only the k most likely tokens |
| Top-p | nucleus sampling | Yes | Keep the smallest set of top tokens whose probabilities sum to at least p |
| Greedy decoding | argmax, `do_sample=False`, temperature 0 (informally) | Yes in effect | Always take the top token; deterministic |
| Beam search | num_beams | No | Keeps several best partial sequences instead of sampling; more deterministic, used for translation |
| Min-p | min_p sampling | Different filter | Keeps tokens with probability at least `min_p` times the top token's probability |
| Repetition penalty | frequency / presence penalty (OpenAI API) | Related, not identical | Hugging Face rescales logits of already-seen tokens; OpenAI-style penalties subtract an amount based on counts |
| Entropy | randomness, uncertainty | Yes | `-sum(p log p)` in nats; 0 means certain, `ln(50257) = 10.8` is uniform over the vocabulary |
| Logits vs probabilities | scores vs softmax output | No | Filters here act on logits by setting them to `-inf`, so softmax gives exactly 0 |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–15)

```python
"""
GOAL:   Implement temperature, top-k, top-p and repetition penalty yourself, apply them to
        the REAL next-token logits of distilgpt2, and watch how the distribution changes.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 05_llm_parameters/02_sampling_from_scratch/main.py
EXPECTED OUTPUT (short): a table of the top tokens under different settings; low temperature
        puts ~all probability on one token; top-k/top-p show how many tokens survive;
        1000 random draws counted per token.
NEEDS:  torch + transformers (distilgpt2 cached from module 04). The functions live in sampling.py.

Builds on: 04_llm_fundamentals/07_forward_pass_logits (logits -> softmax -> next token).
That step always took the argmax (greedy).  Here we pick RANDOMLY according to the probabilities,
and the parameters below decide how adventurous that randomness is.
"""
```

- **In simple words:** Says we implement the sampling knobs ourselves and apply them to real distilgpt2 scores.
- **Exactly:** Docstring only. Builds on step 04.07: that step took the argmax; here we draw randomly according to the probabilities.
- **Why it is here:** Knowing what temperature, top-k and top-p do internally is a favourite interview topic.

### 2. Imports and importing our own sampling module  (`main.py` lines 17–24)

```python
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))     # so `import sampling` works from any cwd
import sampling as S                                # our own module (sampling.py next to this file)
```

- **In simple words:** Load the libraries and our own `sampling.py` that sits in the same folder.
- **Exactly:** `sys.path.insert(0, ...)` makes `import sampling as S` work from any working directory. Module alias `S` is used for `S.apply_temperature`, etc.
- **Why it is here:** Without the path line, running from the repo root raises `ModuleNotFoundError: No module named 'sampling'`.

### 3. Load distilgpt2  (`main.py` lines 26–27)

```python
tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
```

- **In simple words:** Get the tokenizer and the model.
- **Exactly:** Same as module 04; `.eval()` turns dropout off.
- **Why it is here:** Realistic logits are needed so the effect of each knob is believable.

### 4. Get real next-token logits for a prompt  (`main.py` lines 29–32)

```python
prompt = "The best way to learn programming is"
ids = tok(prompt, return_tensors="pt")["input_ids"]   # shape: (1, seq_len=7)
with torch.no_grad():
    logits = model(ids).logits[0, -1]               # raw scores for the next token, shape (50257,)
```

- **In simple words:** Run the prompt through the model once and keep the scores for the next word.
- **Exactly:** `ids` is `(1, 7)` for "The best way to learn programming is". `model(ids).logits` is `(1, 7, 50257)`; `[0, -1]` selects batch 0, last position: `logits` has shape `(50257,)`.
- **Why it is here:** All following experiments transform this single vector.

### 5. Helper to print the top tokens and how many survive  (`main.py` lines 35–42)

```python
def show(title, lg, n=6):
    """Print the top-n tokens after softmax, plus how many tokens still have non-zero probability."""
    probs = torch.softmax(lg, dim=-1)               # shape: (50257,)
    top = torch.topk(probs, n)                      # values, indices: shape (n,)
    alive = int((probs > 0).sum())                  # tokens not filtered out
    entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum())   # randomness in 'nats'
    row = ", ".join(f"{tok.decode(i)!r}:{p:.2f}" for p, i in zip(top.values, top.indices))
    print(f"{title:28} alive={alive:5d} entropy={entropy:4.2f} | {row}")
```

- **In simple words:** Show a distribution compactly: its favourite tokens, how many tokens are still possible, and how random it is.
- **Exactly:** `torch.softmax` gives probabilities `(50257,)`; `topk(probs, 6)` gives values and indices `(6,)`. `alive = (probs > 0).sum()` counts tokens that survived filters. `entropy = -sum(p log p)` over non-zero entries in nats. `{tok.decode(i)!r}` shows the token with quotes so leading spaces are visible.
- **Why it is here:** Alive count and entropy make otherwise invisible changes measurable.

### 6. The raw distribution  (`main.py` lines 45–46)

```python
print(f"prompt: {prompt!r}\n")
show("raw (temperature 1.0)", logits)
```

- **In simple words:** The model's unmodified opinion.
- **Exactly:** Real output: `alive=50257 entropy=2.10`, ' to' 0.59, ' by' 0.15, ' through' 0.05, ' with' 0.04, ' in' 0.01, ' not' 0.01.
- **Why it is here:** This baseline shows how much each knob changes it.

### 7. Temperature sweep  (`main.py` lines 48–52)

```python
# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------
for T in (0.2, 0.7, 1.0, 1.5, 3.0):
    show(f"temperature {T}", S.apply_temperature(logits, T))
```

- **In simple words:** Make the distribution sharper or flatter by dividing the scores.
- **Exactly:** Real output: T=0.2 gives ' to' 1.00, entropy 0.01 and alive=46428 (the rest underflow to exactly 0 in float32). T=0.7 gives ' to' 0.82, entropy 0.73. T=1.0 equals raw. T=1.5 gives ' to' 0.16, entropy 6.84. T=3.0 gives entropy 10.28 (almost uniform; ln 50257 is 10.82).
- **Why it is here:** Low temperature gives safe, repetitive text; high temperature gives creative, then random text.

### 8. Top-k and top-p, alone and combined  (`main.py` lines 54–62)

```python
# ---------------------------------------------------------------------------
# Top-k and top-p
# ---------------------------------------------------------------------------
print()
for k in (1, 5, 50):
    show(f"top_k {k}", S.top_k_filter(logits, k))
for p in (0.3, 0.9, 0.99):
    show(f"top_p {p}", S.top_p_filter(logits, p))
show("top_k 50 + top_p 0.9 + T 0.8", S.top_p_filter(S.top_k_filter(S.apply_temperature(logits, 0.8), 50), 0.9))
```

- **In simple words:** Remove unlikely tokens before choosing, either by count (k) or by cumulative probability (p).
- **Exactly:** Real output: top_k 1 gives alive=1; top_k 5 alive=5; top_k 50 alive=50. top_p 0.3 keeps 1 token (the first already has 0.59 > 0.3); top_p 0.9 keeps 15; top_p 0.99 keeps 1,556. The combined line (T 0.8 then top-k 50 then top-p 0.9) leaves 3 tokens: ' to' 0.82, ' by' 0.14, ' through' 0.04. Order matters: temperature first, then k, then p.
- **Why it is here:** top-k uses a fixed count regardless of context; top-p adapts: few tokens when the model is sure, many when it is unsure.

### 9. Repetition penalty  (`main.py` lines 64–71)

```python
# ---------------------------------------------------------------------------
# Repetition penalty: pretend the model already wrote " to" (the current favourite)
# ---------------------------------------------------------------------------
print()
already = [tok.encode(" to")[0]]       # token ids already present in the output so far
show("no repetition penalty", logits)
show("repetition_penalty 1.5", S.apply_repetition_penalty(logits, already, 1.5))
# GPT-2 logits are mostly NEGATIVE (e.g. -60), so the rule 'negative * penalty' pushes ' to' further down.
```

- **In simple words:** Pretend " to" was already written and see how the penalty pushes it down.
- **Exactly:** `tok.encode(" to")[0]` is the id of ' to'. With penalty 1.5: ' to' drops out of the top 6 and ' by' becomes the favourite (0.35); entropy rises from 2.10 to 3.46. Because GPT-2 logits are mostly negative (e.g. -60), the rule "negative logits are multiplied by the penalty" makes the logit more negative.
- **Why it is here:** Penalising seen tokens stops loops like "the the the", but too high a value damages fluency.

### 10. Draw 1000 samples and count  (`main.py` lines 73–83)

```python
# ---------------------------------------------------------------------------
# Sample 1000 times and count: the counts follow the probabilities
# ---------------------------------------------------------------------------
print("\n1000 draws under two settings (counts of the 4 most frequent tokens):")
for label, kw in [("T=0.3, top_k=50", dict(temperature=0.3, top_k=50)),
                  ("T=1.5, top_p=0.95", dict(temperature=1.5, top_p=0.95))]:
    gen = torch.Generator().manual_seed(0)          # seeded generator -> reproducible randomness
    draws = [S.sample_next_token(logits, generator=gen, **kw) for _ in range(1000)]   # list comprehension
    counts = torch.bincount(torch.tensor(draws)).topk(4)   # shape: (vocab_max_id,) -> top 4 values, indices (4,)
    print(f"  {label:20}", {tok.decode(int(i)): int(c) for c, i in zip(counts.values, counts.indices)},
          "| distinct tokens drawn:", len(set(draws)))
```

- **In simple words:** Sample many times and check the counts match the probabilities.
- **Exactly:** `torch.Generator().manual_seed(0)` gives reproducible draws. Real output: T=0.3 with top_k=50: `' to': 991, ' by': 9`, 2 distinct tokens. T=1.5 with top_p=0.95: ' to' 169, ' by' 58, ' through' 34, ' with' 21 and 501 distinct tokens. `torch.bincount(...)` counts per id; `.topk(4)` returns the 4 largest counts and their ids.
- **Why it is here:** Sampling is random but not arbitrary; the knobs shape the distribution you draw from.

### 11. Generate whole sentences with our own sampler  (`main.py` lines 85–101)

```python
# ---------------------------------------------------------------------------
# A whole sentence generated with our own sampler (loop from module 04 step 08 + sampling)
# ---------------------------------------------------------------------------
def my_sample_generate(prompt_ids, n_tokens=20, **kw):
    cur, gen_ids = prompt_ids, []
    gen = torch.Generator().manual_seed(42)
    for _ in range(n_tokens):
        with torch.no_grad():
            lg = model(cur).logits[0, -1]           # shape: (50257,) next-token scores
        nxt = S.sample_next_token(lg, generated_ids=gen_ids, generator=gen, **kw)
        gen_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)   # shape: (1, seq_len + 1)
    return tok.decode(cur[0])

print("\ngreedy      :", repr(my_sample_generate(ids, greedy=True)))
print("T=0.7 top_p :", repr(my_sample_generate(ids, temperature=0.7, top_p=0.9)))
print("T=2.0 no cap:", repr(my_sample_generate(ids, temperature=2.0)))
```

- **In simple words:** Loop: get scores, sample one token, append, repeat for 20 tokens.
- **Exactly:** `cur` is `(1, seq_len)`, growing by one column each step via `torch.cat`. `gen_ids` records produced ids for the repetition penalty. Real outputs: greedy gives a sensible sentence followed by newlines; T=0.7 top_p=0.9 gives fluent text; T=2.0 with no cap gives gibberish (`'solid fairy tales 903 Request Details...'`).
- **Why it is here:** Shows the practical quality range: too cold is bland, too hot is nonsense.

### 12. Experiments and break-it comments  (`main.py` lines 103–106)

```python
# TRY IT: temperature 0.01 should look like greedy; temperature 10 like random garbage. Check entropy.
# TRY IT: top_p=0.0001 keeps exactly one token (why? read top_p_filter).
# BREAK IT: temperature=0 -> ZeroDivision-like inf/nan. This is why Hugging Face rejects it and uses do_sample=False.
# BREAK IT: pass logits with an extra batch dimension (1, 50257) to top_k_filter and read the error/odd result.
```

- **In simple words:** Try extreme settings and an input with the wrong shape.
- **Exactly:** T=0 divides by zero, giving inf/nan, which is why Hugging Face forbids it. `top_p=0.0001` keeps exactly one token since the first token is always kept. A `(1, 50257)` logits tensor breaks `topk(...).values[-1]` semantics.
- **Why it is here:** Knowing why `temperature=0` is rejected shows you understand the formula.

### 13. sampling.py docstring: the pipeline order  (`sampling.py` lines 1–8)

```python
"""
sampling.py -- the decoding tricks (temperature, top-k, top-p, repetition penalty) written
from scratch on a 1-D tensor of logits.  Imported by main.py here and by step 03 / 07.

Pipeline used by every chat model, in this order:
    logits -> repetition penalty -> / temperature -> top-k filter -> top-p filter -> softmax -> sample
Filters work by setting removed tokens' logits to -inf, so softmax gives them probability exactly 0.
"""
```

- **In simple words:** Lists the order in which the knobs are applied and how filters work.
- **Exactly:** Docstring only. Order: logits, repetition penalty, divide by temperature, top-k, top-p, softmax, sample. Filters set removed logits to `-inf` so softmax gives exactly 0.
- **Why it is here:** Order matters: applying top-p before temperature gives different results.

### 14. Import torch  (`sampling.py` line 10)

```python
import torch
```

- **In simple words:** The only dependency.
- **Exactly:** All functions work on a 1-D `(vocab,)` tensor.
- **Why it is here:** Keeps the module small and testable.

### 15. Repetition penalty  (`sampling.py` lines 13–23)

```python
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """Make tokens that were ALREADY generated less likely (penalty > 1).

    Same rule as Hugging Face: a positive logit is divided by penalty, a negative one multiplied
    (both make the token less attractive).  penalty == 1.0 changes nothing.
    """
    logits = logits.clone()                                  # never mutate the caller's tensor; shape: (vocab,)
    for token_id in set(generated_ids):                      # set(): penalise each distinct token once
        score = logits[token_id]
        logits[token_id] = score / penalty if score > 0 else score * penalty
    return logits
```

- **In simple words:** Make tokens we already wrote less attractive.
- **Exactly:** `logits.clone()` avoids mutating the caller's tensor. `set(generated_ids)` penalises each distinct token once. For each id: if the logit is positive it is divided by the penalty, otherwise multiplied (both lower it). penalty 1.0 changes nothing. Same rule as Hugging Face's `RepetitionPenaltyLogitsProcessor`.
- **Why it is here:** A plain subtraction or multiplication would raise a negative logit instead of lowering it.

### 16. Temperature  (`sampling.py` lines 26–28)

```python
def apply_temperature(logits, temperature=1.0):
    """Divide logits by T.  T<1 widens the gaps (sharper); T>1 shrinks them (flatter)."""
    return logits / temperature                              # shape: (vocab,) -> (vocab,)
```

- **In simple words:** Divide all scores by T.
- **Exactly:** `logits / temperature`, shape `(vocab,) -> (vocab,)`. T<1 widens the gaps between logits so softmax is sharper.
- **Why it is here:** One line, but it controls the randomness of every chat model.

### 17. Top-k filter  (`sampling.py` lines 31–36)

```python
def top_k_filter(logits, k):
    """Keep only the k highest logits; everything else becomes -inf.  k == 0 means 'disabled'."""
    if k <= 0 or k >= logits.numel():
        return logits
    kth_value = torch.topk(logits, k).values[-1]             # shape: () scalar; the k-th largest logit = the cut-off
    return torch.where(logits < kth_value, torch.tensor(float("-inf")), logits)
```

- **In simple words:** Keep only the k best scores; set the rest to minus infinity.
- **Exactly:** If `k <= 0` or `k >= vocab` the logits are returned unchanged. `torch.topk(logits, k).values[-1]` is the k-th largest score (the cut-off). `torch.where(logits < kth, -inf, logits)` masks everything below it.
- **Why it is here:** Cuts off the long tail of silly tokens (a 50,257-token vocabulary has thousands of near-impossible ones).

### 18. Top-p (nucleus) filter  (`sampling.py` lines 39–50)

```python
def top_p_filter(logits, p):
    """Nucleus sampling: keep the smallest set of top tokens whose probabilities sum to >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)   # shape: (vocab,) both
    sorted_probs = torch.softmax(sorted_logits, dim=-1)               # shape: (vocab,)
    cumulative = torch.cumsum(sorted_probs, dim=-1)          # shape: (vocab,) running total of probability
    # Remove a token if the probability mass BEFORE it already reached p (so the first token always stays)
    remove_sorted = (cumulative - sorted_probs) >= p         # shape: (vocab,) bool, in sorted order
    remove = torch.zeros_like(remove_sorted)                 # shape: (vocab,) bool
    remove[sorted_idx] = remove_sorted                       # map back to the original token order
    return logits.masked_fill(remove, float("-inf"))
```

- **In simple words:** Sort tokens by probability, keep adding them until their total reaches p, drop the rest.
- **Exactly:** `torch.sort(descending=True)` gives sorted logits and the original indices. `softmax` of the sorted logits then `cumsum` gives the running total. `(cumulative - sorted_probs) >= p` is "the mass BEFORE this token already reached p", so the first token always stays. `remove[sorted_idx] = remove_sorted` maps the mask back to original token order. `masked_fill(remove, -inf)` applies it.
- **Why it is here:** The set size adapts: confident predictions keep 1 token, uncertain ones keep hundreds.

### 19. The full sampling recipe  (`sampling.py` lines 53–64)

```python
def sample_next_token(logits, generated_ids=(), temperature=1.0, top_k=0, top_p=1.0,
                      repetition_penalty=1.0, greedy=False, generator=None):
    """Full recipe: returns ONE token id (int)."""
    if greedy:                                               # do_sample=False
        return int(torch.argmax(logits))
    logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)
    probs = torch.softmax(logits, dim=-1)                    # shape: (vocab,) sums to 1
    # torch.multinomial draws random indices according to the probabilities
    return int(torch.multinomial(probs, num_samples=1, generator=generator))
```

- **In simple words:** Combine all the steps and return one token id.
- **Exactly:** `greedy=True` returns `int(torch.argmax(logits))`. Otherwise: penalty, temperature, top-k, top-p, `softmax` and `torch.multinomial(probs, 1, generator=generator)`, which draws one index in proportion to `probs`. Returns a Python int.
- **Why it is here:** This is what `model.generate(do_sample=True, ...)` does internally at each step.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `logits / temperature` | tensors divide element-wise by a scalar | `torch.tensor([2., 4.]) / 2` -> `[1., 2.]` |
| `float("-inf")` | negative infinity: `exp(-inf) = 0` so softmax gives probability 0 | - |
| `[... for _ in range(1000)]` | list comprehension: build a list in one line | `[x*x for x in range(3)]` |
| `{k: v for k, v in zip(a, b)}` | dict comprehension | - |
| `set(generated_ids)` | unique items only | `set([1,1,2])` = `{1, 2}` |
| `torch.Generator().manual_seed(0)` | private random stream for reproducibility | - |
| `def f(logits, k)` with default args and `**kw` | flexible parameter passing | `f(**{"temperature": 0.7})` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `torch.sort(x, descending=True)` | torch | sorted values + original indices | `descending` | values, indices |
| `torch.cumsum(x, dim)` | torch | running total | `dim` | tensor |
| `torch.where(cond, a, b)` | torch | element-wise choice | - | tensor |
| `tensor.masked_fill(mask, value)` | torch | replace where mask is True | - | tensor |
| `torch.multinomial(probs, 1)` | torch | draw a random index by probability | `num_samples`, `generator` | index tensor |
| `torch.bincount(t)` | torch | histogram of non-negative ints | - | tensor |

## What each control does
| Control | Rule | Effect | Risk |
|---|---|---|---|
| **temperature T** | `logits / T` before softmax | T<1 peakier (safe, repetitive); T>1 flatter (creative, incoherent); T->0 = greedy | T too high = garbage |
| **top-k** | keep k highest logits | fixed-size candidate list | k is blind to confidence |
| **top-p (nucleus)** | keep smallest set whose probs sum >= p | candidate count adapts to confidence | p near 1 keeps long tail |
| **repetition penalty** | positive logit / r, negative logit * r for already-used tokens | fewer loops | too high = avoids needed words ("the") |

Order used by the library (and `sample_next_token`): repetition penalty -> temperature -> top-k -> top-p -> softmax -> sample.

## Execution trace
Real output (prompt `The best way to learn programming is`):
```
raw (temperature 1.0)        alive=50257 entropy=2.10 | ' to':0.59, ' by':0.15, ' through':0.05, ' with':0.04
temperature 0.2              entropy=0.01 | ' to':1.00         <- almost greedy
temperature 0.7              entropy=0.73 | ' to':0.82, ' by':0.11
temperature 1.5              entropy=6.84 | ' to':0.16, ' by':0.06   <- much flatter
temperature 3.0              entropy=10.28 | ' to':0.00 ...          <- nearly uniform = random tokens

top_k 5                      alive=    5
top_p 0.3                    alive=    1      <- top token alone already has 59% > 30%
top_p 0.9                    alive=   15
top_p 0.99                   alive= 1556
top_k 50 + top_p 0.9 + T 0.8 alive=    3

repetition_penalty 1.5 (' to' already used): ' by':0.35, ' through':0.13 ...   <- ' to' was demoted

1000 draws: T=0.3, top_k=50    {' to': 991, ' by': 9}                distinct tokens drawn: 2
            T=1.5, top_p=0.95  {' to': 169, ' by': 58, ...}          distinct tokens drawn: 501

greedy      : '... is to learn from the experience of the programmer.\n\n\n\n...'
T=0.7 top_p : '... is to follow the same principles as you do. If you are already familiar ...'
T=2.0 no cap: '... is solid fairy tales 903 Request Details Highlights \\(107 Zurich ...'
```
Notes:
- **Entropy** measures randomness: 0 = certain, ~10.8 = uniform over 50257 tokens.
- `alive=46428` at T=0.2: the other tokens' probabilities underflow to exactly 0 in float32 - "alive" counts non-zero probabilities, so it's not a deliberate filter.
- **top_p 0.3 -> 1 token** because the best token alone exceeds 30%. top-p self-adjusts: when the model is unsure it keeps hundreds of candidates, when confident, a handful.
- **Why logits are mostly negative here** (about -60): only differences matter to softmax. That's why the repetition rule multiplies negative logits (to make them *more* negative).

## Experiments
- Temperature 0.01 should behave like greedy; check the entropy.
- `top_p=0.0001` keeps exactly one token. Why?
- Combine `temperature=0.7, top_k=40, top_p=0.9` - the "default chat" recipe - and generate 3 times with different seeds.

## Debugging practice
`temperature=0` gives `inf/nan` (division by zero), then `torch.multinomial` complains `probability tensor contains either inf, nan or element < 0`. Reading the error top-down tells you *where* it surfaced; reasoning backwards tells you *where it started*: logits / 0. Hugging Face prevents this by rejecting `temperature=0`; you use `do_sample=False` instead.

## Rebuild checklist
- [ ] Write temperature and softmax from memory and show T=0.2 vs 2.0.
- [ ] Write top-k (use `torch.topk`, mask the rest with -inf).
- [ ] Write top-p (sort, cumsum, mask).
- [ ] Explain the order of operations.

## The new problem (cliffhanger)
Your own sampler produces varied text, but is it correct? Maybe it only looks right. Prove it against the library: [step 03, compare with Hugging Face generate](../03_hf_generate_compare/WALKTHROUGH.md).

> **Say it aloud (interview box):** The model gives a probability for every token, and greedy always takes the top one, which is repetitive. Temperature divides the logits to sharpen or flatten the distribution, while top-k and top-p cut the unlikely tail, and top-p adapts to how confident the model is. More randomness gives variety but more errors, so I choose low values for extraction and higher ones for brainstorming.
