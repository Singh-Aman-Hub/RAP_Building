# Step 01 - Attention from scratch (numpy)

## Goal
Build self-attention, the core operation of every transformer, with ~20 lines of numpy and watch the shapes.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/01_attention_from_scratch/main.py
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
x (token vectors) shape: (4, 8)               <- INPUT: 4 tokens x 8 numbers
Q, K, V shapes: (4, 8) (4, 8) (4, 8)          <- three different "views" of the same tokens
scores shape: (4, 4)                          <- every token vs every token

attention weights (row = who is looking, column = who is looked at):
[[0.57 0.   0.43 0.01]
 [1.   0.   0.   0.  ]
 [0.17 0.36 0.46 0.01]
 [0.05 0.35 0.04 0.56]]
row sums: [1. 1. 1. 1.]                       <- each row is a probability distribution

output shape: (4, 8)                          <- OUTPUT: same shape as input => layers can stack

causal attention weights (upper triangle is 0 = cannot see the future):
[[1.   0.   0.   0.  ]
 [1.   0.   0.   0.  ]
 [0.17 0.36 0.47 0.  ]
 [0.05 0.35 0.04 0.56]]
```
1. `x` = 4 token vectors. 2. Multiply by three weight matrices -> Q (what I look for), K (what I offer), V (what I hand over). 3. `Q @ K.T` = relevance of token j to token i. 4. softmax per row -> weights. 5. `weights @ V` -> each token becomes a blend of the Values it attended to. 6. Causal mask: row 0 can only see itself (`[1, 0, 0, 0]`).

The weights are random here (nothing is trained) so the pattern means nothing - **training adjusts `W_q, W_k, W_v`** so useful patterns emerge. A real transformer block = attention + a small MLP, repeated (6 times in distilgpt2, 12 in gpt2), with several "heads" running attention in parallel.

## Experiments (change ONE thing)
- `seq_len = 6` and add two tokens: shapes should become `(6, 8)` / `(6, 6)`. Nothing else changes.
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

## Say it aloud
"Self-attention lets each token look at the other tokens. I make Query, Key and Value from each token, score every pair with a scaled dot product, softmax the scores into weights, and take a weighted average of the Values. A causal mask hides future tokens so the model can be trained to predict the next one. Trade-off: the score matrix is `seq_len x seq_len`, so cost grows quadratically with context length - that's why long context is expensive."
