"""Solution ex01: single-head self-attention as a reusable function + tests.
RUN: .venv/bin/python 04_llm_fundamentals/solutions/ex04_attention_function.py"""
import numpy as np


def softmax(a):
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


def attention(x, W_q, W_k, W_v, causal=True):
    """x: (seq, d_model) -> (output (seq, d_head), weights (seq, seq))."""
    Q, K, V = x @ W_q, x @ W_k, x @ W_v
    scores = Q @ K.T / np.sqrt(K.shape[-1])
    if causal:
        future = np.triu(np.ones_like(scores, dtype=bool), k=1)   # True above the diagonal
        scores = np.where(future, -np.inf, scores)
    weights = softmax(scores)
    return weights @ V, weights


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 16))
    Wq, Wk, Wv = (rng.normal(size=(16, 4)) for _ in range(3))   # d_head=4 differs from d_model=16
    out, w = attention(x, Wq, Wk, Wv)
    assert out.shape == (5, 4)
    assert np.allclose(w.sum(axis=1), 1.0)            # rows are probability distributions
    assert np.allclose(np.triu(w, k=1), 0.0)          # causal: nothing above the diagonal
    # Causality test: changing a FUTURE token must not change an earlier token's output
    x2 = x.copy(); x2[4] += 10
    out2, _ = attention(x2, Wq, Wk, Wv)
    assert np.allclose(out[:4], out2[:4]) and not np.allclose(out[4], out2[4])
    print("all attention checks passed; output shape", out.shape)
