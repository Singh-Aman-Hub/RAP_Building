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

import numpy as np  # numpy = fast arrays + maths (third-party library)

# np.random.seed makes the "random" numbers repeat every run -> reproducible output
rng = np.random.default_rng(seed=0)

# ---------------------------------------------------------------------------
# 1. A pretend sentence: 4 tokens, each already turned into an 8-number vector.
#    (Step 03 showed where these vectors really come from: embeddings + positions.)
# ---------------------------------------------------------------------------
tokens = ["The", "cat", "sat", "down"]
seq_len, d_model = 4, 8                      # 4 tokens, 8 numbers per token
x = rng.normal(size=(seq_len, d_model))      # shape: (seq_len, d_model) = (4, 8): one row per token
print("x (token vectors) shape:", x.shape)

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

# ---------------------------------------------------------------------------
# 3. Scores: how much should token i look at token j?  Dot product of Query_i and Key_j.
# ---------------------------------------------------------------------------
scores = Q @ K.T / np.sqrt(d_head)           # shape: (seq_len, seq_len) = (4, 4); (4,8)@(8,4) -> (4,4).  /sqrt(d) keeps numbers small & stable
print("scores shape:", scores.shape)


def softmax(a):
    """Turn any numbers into probabilities that are >= 0 and sum to 1 along the last axis."""
    a = a - a.max(axis=-1, keepdims=True)    # shape unchanged; max has shape (rows, 1); subtract max: avoids overflow in exp(); result is unchanged
    e = np.exp(a)                            # same shape as a
    return e / e.sum(axis=-1, keepdims=True)


weights = softmax(scores)                    # shape: (4, 4); each ROW now sums to 1: "how I split my attention"
print("\nattention weights (row = who is looking, column = who is looked at):")
print(np.round(weights, 2))
print("row sums:", weights.sum(axis=1))

# ---------------------------------------------------------------------------
# 4. Output: each token becomes a weighted average of all the Value vectors.
# ---------------------------------------------------------------------------
out = weights @ V                            # shape: (seq_len, d_head) = (4, 8); (4,4)@(4,8) -> (4,8): same shape as the input!
print("\noutput shape:", out.shape, "(same as input, so layers can be stacked)")

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

# Cliffhanger: this is ONE mixing step.  One layer cannot build deep understanding;
# stack attention + an MLP many times -> the transformer block (../05_transformer_block/).

# TRY IT: change seq_len to 6 (and tokens) - do the shapes still line up?
# TRY IT: multiply `scores` by 10 before softmax - weights become spiky (like low temperature, see module 05).
# BREAK IT: change W_v to shape (d_model, 5): `weights @ V` still works, but the output shape changes. Why?
# BREAK IT: make W_k shape (7, 8): read the ValueError for `x @ W_k` - it names the mismatched dimensions.
