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
