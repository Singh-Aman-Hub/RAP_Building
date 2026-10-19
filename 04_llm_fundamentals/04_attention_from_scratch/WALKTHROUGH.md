# Step 04 - Attention from scratch (The Great Council of Tokens)

## The story so far
[Step 03](../03_embeddings/WALKTHROUGH.md) ended with tokens that each own one fixed vector plus a position. `bank` has the same numbers in "river bank" and "bank loan", and averaging vectors (step 02) made `driver parks car` equal `car parks driver`. We use made-up vectors here (shape `(4, 8)`) so we can focus on the mechanism; step 03 showed where real ones come from.

## The pain point
A word's meaning depends on its neighbours, but a lookup table knows nothing about them. Averaging the neighbours throws away who matters: in "the river bank was muddy" the word `river` should shape `bank` far more than `the` does. We need a *weighted* mix, where the weights depend on the words.

## The spark
Hold a council. Every token asks a question (**Query**: what am I looking for?), advertises what it has (**Key**: what do I contain?), and carries a payload (**Value**: what do I hand over if chosen?). Each token scores every other token with `Query . Key`, turns the scores into weights with softmax, and rebuilds itself as the weighted sum of Values. That is the "gossip": `bank` listens hard to `river` and barely to `the`. Two details keep it honest. Dividing by `sqrt(d)` stops big dot products from turning softmax into a one-hot spike (printed experiment at the end of the trace). A causal mask hides future tokens, time-travel insurance, so a model trained to predict the next token cannot cheat by reading the answer.

## Goal
Build self-attention, the core operation of every transformer, with ~20 lines of numpy and watch the shapes.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/04_attention_from_scratch/main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `a @ b` | matrix multiplication | `np.ones((2,3)) @ np.ones((3,4))` has shape `(2, 4)` |
| `a.T` | transpose (swap rows/columns) | `np.ones((2,3)).T.shape == (3, 2)` |
| `x[i, j]`, `x[:, 0]` | indexing: row i column j / all rows, column 0 | `x[:, 0]` is the first column |
| `np.where(cond, a, b)` | pick `a` where cond is True else `b` | `np.where(x > 0, x, 0)` |
| `tokens = ["The", ...]` | a Python `list` | `len(tokens)` is 4 |
| `keepdims=True` | keep the reduced axis so shapes still broadcast | `x.max(axis=-1, keepdims=True).shape` is `(4, 1)` |
| `def softmax(a):` + docstring | defining a function; the `"""..."""` string is its documentation | `help(softmax)` prints it |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `numpy` | 3rd-party | fast n-dimensional arrays | - | module |
| `np.random.default_rng(seed)` | numpy | random number generator you can reproduce | `seed` | `Generator` |
| `rng.normal(size=...)` | numpy | random numbers from a bell curve | `size` = shape tuple | `ndarray` |
| `np.sqrt`, `np.exp` | numpy | element-wise maths | array | array |
| `.max(axis=-1)` / `.sum(axis=-1)` | numpy | reduce along the LAST axis (per row) | `axis`, `keepdims` | array |
| `np.triu(m, k=1)` | numpy | keep the upper triangle above the diagonal | `k` = diagonal offset | array |
| `np.round(a, 2)` | numpy | round for display | decimals | array |

## Execution trace (input -> processing -> output)
Real output from a run:
```
x (token vectors) shape: (4, 8)
Q, K, V shapes: (4, 8) (4, 8) (4, 8)
scores shape: (4, 4)

attention weights (row = who is looking, column = who is looked at):
[[0.57 0.   0.43 0.01]
 [1.   0.   0.   0.  ]
 [0.17 0.36 0.46 0.01]
 [0.05 0.35 0.04 0.56]]
row sums: [1. 1. 1. 1.]

output shape: (4, 8) (same as input, so layers can be stacked)

causal attention weights (upper triangle is 0 = cannot see the future):
[[1.   0.   0.   0.  ]
 [1.   0.   0.   0.  ]
 [0.17 0.36 0.47 0.  ]
 [0.05 0.35 0.04 0.56]]

Why /sqrt(d): variance of scores and how peaky softmax gets
    d   var raw  var scaled  max weight raw  max weight scaled
    4       4.2        1.04            0.38               0.25
   64      60.3        0.94            0.75               0.29
 1024    1051.0        1.03            1.00               0.29
-> without scaling the variance is ~d and one key grabs ~all the attention; with it the variance stays ~1.
```
1. `x` = 4 token vectors. 2. Multiply by three weight matrices -> Q (what I look for), K (what I offer), V (what I hand over). 3. `Q @ K.T` = relevance of token j to token i. 4. softmax per row -> weights. 5. `weights @ V` -> each token becomes a blend of the Values it attended to. 6. Causal mask: row 0 can only see itself (`[1, 0, 0, 0]`).

The weights are random here (nothing is trained) so the pattern means nothing - **training adjusts `W_q, W_k, W_v`** so useful patterns emerge. The sqrt(d) table shows why scaling matters: for d = 1024 the raw score variance is about 1000 and a single key takes all the weight (1.00); scaled, the variance stays near 1 and the weights stay spread out (0.29), so gradients can flow. A real transformer block = attention + a small MLP, repeated (6 times in distilgpt2, 12 in gpt2), with several "heads" running attention in parallel.

## Experiments (change ONE thing)
- `seq_len = 6` and add two tokens: shapes should become `(6, 8)` / `(6, 6)`. Nothing else changes.
- Remove `/ np.sqrt(d_head)` or run the sqrt(d) table with d = 4096: the max weight goes to ~1.00 (one key takes everything) and the variance tracks d.
- `scores = scores * 10` before softmax: weights become almost one-hot (very peaky) - the same effect as a low temperature (module 05).
- Change `d_head` to 4 (and the shapes of W_q/W_k/W_v): output becomes `(4, 4)`; real models project back to `d_model` afterwards.

## Debugging practice
Set `W_k = rng.normal(size=(7, 8))`. You get:
`ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 ... (size 7 is different from 8)`.
Read it bottom-up: the last line names the problem (inner dimensions 8 vs 7 disagree), the line above shows your file/line (`K = x @ W_k`). Rule: for `A @ B`, `A.shape[-1]` must equal `B.shape[0]`.

## Rebuild checklist
- [ ] Write `softmax` (subtract the max first - why?).
- [ ] Write `attention(x, W_q, W_k, W_v)` returning output and weights.
- [ ] Add the causal mask with `np.triu`.
- [ ] Assert row sums are 1 and the upper triangle is 0.

## The new problem (cliffhanger)
Attention is one mixing step, and it is only a weighted average: linear in the Values. One round lets `bank` hear from `river`, but understanding a sentence (grammar, then meaning, then the answer) needs many rounds, plus a place for each token to think on its own after listening. We need to add a small neural net after attention and stack the pair many times. Next: [Step 05 - The transformer block](../05_transformer_block/WALKTHROUGH.md).

> **Say it aloud (interview box):** Self-attention lets each token look at the others: I make a Query, Key and Value from each token, score every pair with a dot product divided by sqrt(d) so softmax does not saturate, and take the softmax-weighted sum of the Values. A causal mask sets future scores to minus infinity so the model trained to predict the next token cannot peek ahead. The score matrix is seq_len by seq_len, so cost grows quadratically with context, and one attention step is not enough, so I stack it with an MLP in a transformer block.
