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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Self-attention | scaled dot-product attention, intra-attention | Yes (for the same-sequence case) | Each token mixes information from the others, weighted by learned relevance |
| Q / K / V | query, key, value | Three different projections | Q = what I look for, K = what I advertise, V = what I hand over |
| Cross-attention | encoder-decoder attention | No | Queries come from one sequence, keys and values from another |
| Causal mask | look-ahead mask, autoregressive mask, triangular mask | Yes | Hides future positions so token i sees only 0..i |
| Attention weights | attention scores (before softmax), attention map | Not exactly | "Scores" are pre-softmax dot products; "weights" are the post-softmax probabilities |
| `d_model` vs `d_head` | model dim, head dim | Different when there are several heads | `d_head = d_model / n_heads`; here both are 8 for one head |
| `softmax` over the last axis | row-wise softmax | Yes | Each row (one query) becomes a probability distribution over keys |
| Scaled by sqrt(d) | temperature of attention | Related | Scaling keeps score variance near 1 so softmax does not saturate |

## Code, step by step

### 1. Module docstring  (`main.py` lines 1–17)

```python
"""
GOAL:   Build the heart of a transformer (self-attention) from scratch with numpy
        and print every array's shape so you can SEE what the model does.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/04_attention_from_scratch/main.py
EXPECTED OUTPUT (short): shapes like (4, 8), a 4x4 attention-weight table whose rows
        each sum to 1.0, and a causal version where the upper triangle is 0.
NEEDS:  numpy only. No internet.

Big idea
--------
A sentence is a list of token vectors (one row per token).  Attention lets each
token LOOK AT other tokens and mix in their information, weighted by relevance.
A GPT-style LLM repeats "attention + small neural net" many times, then uses the
final vector of the last token to predict the NEXT token.
"""
```

- **In simple words:** States that we build self-attention in numpy and print every shape.
- **Exactly:** Docstring only; no runtime effect.
- **Why it is here:** Gives the goal of seeing the mechanics, not the performance.

### 2. Import numpy and seed a random generator  (`main.py` lines 19–22)

```python
import numpy as np  # numpy = fast arrays + maths (third-party library)

# np.random.seed makes the "random" numbers repeat every run -> reproducible output
rng = np.random.default_rng(seed=0)
```

- **In simple words:** Load numpy and make the "random" numbers repeatable.
- **Exactly:** `np.random.default_rng(seed=0)` creates a `Generator`; every `rng.normal` call advances its state, so the order of calls matters for reproducibility.
- **Why it is here:** Without a seed the printed weights change each run and cannot be discussed.

### 3. Pretend sentence: 4 tokens of 8 numbers each  (`main.py` lines 24–31)

```python
# ---------------------------------------------------------------------------
# 1. A pretend sentence: 4 tokens, each already turned into an 8-number vector.
#    (Step 03 showed where these vectors really come from: embeddings + positions.)
# ---------------------------------------------------------------------------
tokens = ["The", "cat", "sat", "down"]
seq_len, d_model = 4, 8                      # 4 tokens, 8 numbers per token
x = rng.normal(size=(seq_len, d_model))      # shape: (seq_len, d_model) = (4, 8): one row per token
print("x (token vectors) shape:", x.shape)
```

- **In simple words:** Four tokens ("The cat sat down"), each already turned into a vector.
- **Exactly:** `seq_len, d_model = 4, 8`. `x = rng.normal(size=(4, 8))` is `(seq_len, d_model)`: one row per token. Real output `x (token vectors) shape: (4, 8)`. In a real model, x comes from embeddings plus positions (step 03).
- **Why it is here:** The shape `(seq_len, d_model)` is what flows through every layer of a transformer.

### 4. Create Q, K, V by multiplying with three weight matrices  (`main.py` lines 33–46)

```python
# ---------------------------------------------------------------------------
# 2. Learnable weight matrices.  Training (module 03 and step 09) is the process of
#    adjusting these numbers.  Here they are random, so attention is not
#    "smart" yet - but the MECHANICS are identical to GPT.
# ---------------------------------------------------------------------------
d_head = 8
W_q = rng.normal(size=(d_model, d_head))     # shape: (d_model, d_head) = (8, 8); makes a "Query"  : what am I looking for?
W_k = rng.normal(size=(d_model, d_head))     # shape: (8, 8); makes a "Key"    : what do I contain?
W_v = rng.normal(size=(d_model, d_head))     # shape: (8, 8); makes a "Value"  : what do I hand over if chosen?

Q = x @ W_q                                  # shape: (seq_len, d_head) = (4, 8); '@' = matrix multiplication; (4,8)@(8,8) -> (4,8)
K = x @ W_k                                  # shape: (4, 8)
V = x @ W_v                                  # shape: (4, 8)
print("Q, K, V shapes:", Q.shape, K.shape, V.shape)
```

- **In simple words:** Each token vector is turned into three versions: a question it asks, a label it carries, and a payload it gives.
- **Exactly:** `W_q, W_k, W_v` are each `(d_model, d_head) = (8, 8)` random matrices (learned in a real model). `Q = x @ W_q` is `(4,8) @ (8,8) -> (4,8)`; same for K and V. Real output: `Q, K, V shapes: (4, 8) (4, 8) (4, 8)`.
- **Why it is here:** Separating the roles lets the model learn different things for "what to look for" and "what to give". Random weights mean no useful attention yet, but the mechanics are identical to GPT.

### 5. Attention scores: Q dot K, scaled  (`main.py` lines 48–52)

```python
# ---------------------------------------------------------------------------
# 3. Scores: how much should token i look at token j?  Dot product of Query_i and Key_j.
# ---------------------------------------------------------------------------
scores = Q @ K.T / np.sqrt(d_head)           # shape: (seq_len, seq_len) = (4, 4); (4,8)@(8,4) -> (4,4).  /sqrt(d) keeps numbers small & stable
print("scores shape:", scores.shape)
```

- **In simple words:** For every pair of tokens, compute how well one token's question matches another's label.
- **Exactly:** `Q @ K.T` is `(4,8) @ (8,4) -> (4,4)`; entry `[i, j]` is the dot product of the query of token i with the key of token j. Dividing by `np.sqrt(d_head)` = 2.83 keeps the numbers small. Printed `scores shape: (4, 4)`.
- **Why it is here:** The quadratic `(seq, seq)` matrix is why long contexts are expensive: cost grows with the square of length.

### 6. A softmax helper  (`main.py` lines 55–59)

```python
def softmax(a):
    """Turn any numbers into probabilities that are >= 0 and sum to 1 along the last axis."""
    a = a - a.max(axis=-1, keepdims=True)    # shape unchanged; max has shape (rows, 1); subtract max: avoids overflow in exp(); result is unchanged
    e = np.exp(a)                            # same shape as a
    return e / e.sum(axis=-1, keepdims=True)
```

- **In simple words:** Turn each row of scores into probabilities.
- **Exactly:** `a.max(axis=-1, keepdims=True)` is `(rows, 1)`; subtracting it avoids overflow in `exp` without changing the result. `e / e.sum(axis=-1, keepdims=True)` divides each row by its own total.
- **Why it is here:** Without the max subtraction large scores give `inf` and `nan`. Here the same function is reused for the masked scores.

### 7. Attention weights: each row is a distribution  (`main.py` lines 62–65)

```python
weights = softmax(scores)                    # shape: (4, 4); each ROW now sums to 1: "how I split my attention"
print("\nattention weights (row = who is looking, column = who is looked at):")
print(np.round(weights, 2))
print("row sums:", weights.sum(axis=1))
```

- **In simple words:** Row i says how token i splits its attention across all tokens.
- **Exactly:** `weights` is `(4, 4)`, every row sums to 1 (real output `[1. 1. 1. 1.]`). Row 0 ("The") puts 0.57 on itself and 0.43 on "sat"; row 1 ("cat") puts 1.00 on "The". Random weights make these arbitrary.
- **Why it is here:** Rows summing to 1 make the output a proper weighted average.

### 8. Output: weighted average of the Value vectors  (`main.py` lines 67–71)

```python
# ---------------------------------------------------------------------------
# 4. Output: each token becomes a weighted average of all the Value vectors.
# ---------------------------------------------------------------------------
out = weights @ V                            # shape: (seq_len, d_head) = (4, 8); (4,4)@(4,8) -> (4,8): same shape as the input!
print("\noutput shape:", out.shape, "(same as input, so layers can be stacked)")
```

- **In simple words:** Each token becomes a blend of all tokens' payloads, weighted by attention.
- **Exactly:** `weights @ V` is `(4,4) @ (4,8) -> (4,8)`, the same shape as the input. Real output `output shape: (4, 8)`.
- **Why it is here:** Same-shape in and out lets layers be stacked dozens of times.

### 9. Causal mask: hide the future  (`main.py` lines 73–82)

```python
# ---------------------------------------------------------------------------
# 5. CAUSAL mask: GPT-style models predict the next token, so token i may only
#    look at tokens 0..i (not the future).  We hide the future with -infinity
#    BEFORE softmax; exp(-inf) = 0 so those weights become exactly 0.
# ---------------------------------------------------------------------------
mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)   # shape: (4, 4); True above the diagonal = "future"
masked_scores = np.where(mask, -np.inf, scores)                # shape: (4, 4); replace future scores with -inf
causal_weights = softmax(masked_scores)                         # shape: (4, 4)
print("\ncausal attention weights (upper triangle is 0 = cannot see the future):")
print(np.round(causal_weights, 2))
```

- **In simple words:** A model that predicts the next word must not peek at later words, so we block them.
- **Exactly:** `np.triu(np.ones((4,4), dtype=bool), k=1)` is True strictly above the diagonal. `np.where(mask, -np.inf, scores)` replaces those scores; `exp(-inf) = 0`, so after softmax those weights are exactly 0. Real result: row 0 `[1, 0, 0, 0]`, row 1 `[1, 0, 0, 0]`, row 2 `[0.17, 0.36, 0.47, 0]`, row 3 unchanged. The mask is applied BEFORE softmax so rows still sum to 1.
- **Why it is here:** Without it, training would leak the answer (the next token) and the model could not generate.

### 10. Why divide by the square root of d  (`main.py` lines 84–103)

```python
# ---------------------------------------------------------------------------
# 6. WHY divide by sqrt(d)?  A printed experiment.
#    A dot product of two d-dimensional random vectors is a sum of d terms, so its
#    variance grows like d.  Big scores make softmax nearly one-hot, and a one-hot
#    softmax has almost zero gradient -> training stalls.  Dividing by sqrt(d)
#    brings the variance back to ~1.
# ---------------------------------------------------------------------------
print("\nWhy /sqrt(d): variance of scores and how peaky softmax gets")
print(f"{'d':>5} {'var raw':>9} {'var scaled':>11} {'max weight raw':>15} {'max weight scaled':>18}")
for d in [4, 64, 1024]:
    big_q = rng.normal(size=(2000, d))        # shape: (n_samples, d): 2000 random queries
    big_k = rng.normal(size=(2000, d))        # shape: (n_samples, d): 2000 random keys
    raw = (big_q * big_k).sum(axis=-1)        # shape: (2000,): one dot product per pair
    scaled = raw / np.sqrt(d)                 # shape: (2000,)
    row_q = rng.normal(size=(d,))             # shape: (d,): one query
    row_k = rng.normal(size=(10, d))          # shape: (10, d): ten keys to choose from
    peak_raw = softmax(row_k @ row_q).max()                   # scalar: strongest weight among 10 keys, unscaled
    peak_scaled = softmax(row_k @ row_q / np.sqrt(d)).max()   # scalar: same, scaled
    print(f"{d:>5} {raw.var():>9.1f} {scaled.var():>11.2f} {peak_raw:>15.2f} {peak_scaled:>18.2f}")
print("-> without scaling the variance is ~d and one key grabs ~all the attention; with it the variance stays ~1.")
```

- **In simple words:** Run an experiment showing that unscaled scores get huge and make attention too spiky.
- **Exactly:** For random unit-variance vectors a dot product is a sum of `d` terms, so variance is about `d`. Real table: d=4 raw var 4.2 vs scaled 1.04; d=64 60.3 vs 0.94; d=1024 1051.0 vs 1.03. The strongest of 10 attention weights becomes 1.00 for d=1024 unscaled but stays 0.29 scaled. A near one-hot softmax has almost zero gradient.
- **Why it is here:** This justifies the `/ sqrt(d)` in the formula `softmax(QK^T / sqrt(d)) V`; without it training stalls for wide models.

### 11. Cliffhanger and break-it comments  (`main.py` lines 105–111)

```python
# Cliffhanger: this is ONE mixing step.  One layer cannot build deep understanding;
# stack attention + an MLP many times -> the transformer block (../05_transformer_block/).

# TRY IT: change seq_len to 6 (and tokens) - do the shapes still line up?
# TRY IT: multiply `scores` by 10 before softmax - weights become spiky (like low temperature, see module 05).
# BREAK IT: change W_v to shape (d_model, 5): `weights @ V` still works, but the output shape changes. Why?
# BREAK IT: make W_k shape (7, 8): read the ValueError for `x @ W_k` - it names the mismatched dimensions.
```

- **In simple words:** One attention step is not enough; stack them with an MLP to get a transformer block.
- **Exactly:** Comments only. BREAK IT: `W_k` of shape `(7, 8)` raises a `ValueError` naming the mismatched dimensions in `x @ W_k`.
- **Why it is here:** Leads to step 05.

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
