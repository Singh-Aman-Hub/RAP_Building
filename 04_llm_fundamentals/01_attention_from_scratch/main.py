"""
GOAL:   Build the heart of a transformer (self-attention) from scratch with numpy
        and print every array's shape so you can SEE what the model does.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/01_attention_from_scratch/main.py
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
#    (In step 03 you will see where these vectors really come from: embeddings.)
# ---------------------------------------------------------------------------
tokens = ["The", "cat", "sat", "down"]
seq_len, d_model = 4, 8                      # 4 tokens, 8 numbers per token
x = rng.normal(size=(seq_len, d_model))      # shape (4, 8): one row per token
print("x (token vectors) shape:", x.shape)

# ---------------------------------------------------------------------------
# 2. Learnable weight matrices.  Training (module 03/06) is the process of
#    adjusting these numbers.  Here they are random, so attention is not
#    "smart" yet - but the MECHANICS are identical to GPT.
# ---------------------------------------------------------------------------
d_head = 8
W_q = rng.normal(size=(d_model, d_head))     # makes a "Query"  : what am I looking for?
W_k = rng.normal(size=(d_model, d_head))     # makes a "Key"    : what do I contain?
W_v = rng.normal(size=(d_model, d_head))     # makes a "Value"  : what do I hand over if chosen?

Q = x @ W_q                                  # '@' = matrix multiplication; (4,8)@(8,8) -> (4,8)
K = x @ W_k
V = x @ W_v
print("Q, K, V shapes:", Q.shape, K.shape, V.shape)

# ---------------------------------------------------------------------------
# 3. Scores: how much should token i look at token j?  Dot product of Query_i and Key_j.
# ---------------------------------------------------------------------------
scores = Q @ K.T / np.sqrt(d_head)           # (4,8)@(8,4) -> (4,4).  /sqrt(d) keeps numbers small & stable
print("scores shape:", scores.shape)


def softmax(a):
    """Turn any numbers into probabilities that are >= 0 and sum to 1 along the last axis."""
    a = a - a.max(axis=-1, keepdims=True)    # subtract max: avoids overflow in exp(); result is unchanged
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


weights = softmax(scores)                    # each ROW now sums to 1: "how I split my attention"
print("\nattention weights (row = who is looking, column = who is looked at):")
print(np.round(weights, 2))
print("row sums:", weights.sum(axis=1))

# ---------------------------------------------------------------------------
# 4. Output: each token becomes a weighted average of all the Value vectors.
# ---------------------------------------------------------------------------
out = weights @ V                            # (4,4)@(4,8) -> (4,8): same shape as the input!
print("\noutput shape:", out.shape, "(same as input, so layers can be stacked)")

# ---------------------------------------------------------------------------
# 5. CAUSAL mask: GPT-style models predict the next token, so token i may only
#    look at tokens 0..i (not the future).  We hide the future with -infinity
#    BEFORE softmax; exp(-inf) = 0 so those weights become exactly 0.
# ---------------------------------------------------------------------------
mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)   # True above the diagonal = "future"
masked_scores = np.where(mask, -np.inf, scores)                # replace future scores with -inf
causal_weights = softmax(masked_scores)
print("\ncausal attention weights (upper triangle is 0 = cannot see the future):")
print(np.round(causal_weights, 2))

# TRY IT: change seq_len to 6 (and tokens) - do the shapes still line up?
# TRY IT: multiply `scores` by 10 before softmax - weights become spiky (like low temperature, see module 05).
# BREAK IT: change W_v to shape (d_model, 5): `weights @ V` still works, but the output shape changes. Why?
# BREAK IT: make W_k shape (7, 8): read the ValueError for `x @ W_k` - it names the mismatched dimensions.
