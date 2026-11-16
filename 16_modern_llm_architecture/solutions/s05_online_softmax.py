"""
GOAL:  Solutions for step 05: online softmax for one row, tiled attention, block_k sweep, timing against naive
       attention, and the effect of dropping the accumulator rescale.

RUN (from this folder):  python s05_online_softmax.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s05_online_softmax.py
NEEDS: numpy only. About 10 s on CPU (timing loop); timings vary by machine.
"""
import time
import numpy as np


def online_softmax(row, block=4):
    """Return softmax(row) in one pass over blocks keeping only (m, l)."""
    m, l = -np.inf, 0.0
    for i in range(0, len(row), block):
        chunk = row[i:i + block]
        m_new = max(m, chunk.max())
        l = l * np.exp(m - m_new) + np.exp(chunk - m_new).sum()    # rescale the old sum to the new max
        m = m_new
    return np.exp(row - m) / l


r = np.random.default_rng(0).normal(size=37) * 5
ref = np.exp(r - r.max()); ref /= ref.sum()
print("online softmax max error:", np.abs(online_softmax(r) - ref).max())


def naive_attention(Q, K, V):
    S = Q @ K.T / np.sqrt(Q.shape[1])
    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    return P @ V


def flash_attention(Q, K, V, block_k=64, rescale=True):
    N, d = Q.shape
    m, l, acc = np.full(N, -np.inf), np.zeros(N), np.zeros((N, d))
    for j in range(0, K.shape[0], block_k):
        S = Q @ K[j:j + block_k].T / np.sqrt(d)                    # (N, block) scores for this tile only
        m_new = np.maximum(m, S.max(1))
        P = np.exp(S - m_new[:, None])
        scale = np.exp(m - m_new)                                  # shrink previous accumulator to the new max
        l = l * scale + P.sum(1)
        acc = (acc * scale[:, None] if rescale else acc) + P @ V[j:j + block_k]
        m = m_new
    return acc / l[:, None]


rng = np.random.default_rng(1)
N, d = 200, 32
Q, K, V = (rng.normal(size=(N, d)) * 2 for _ in range(3))
ref = naive_attention(Q, K, V)
for bk in (1, 64, 200):
    print(f"block_k={bk:<4} max error {np.abs(flash_attention(Q, K, V, bk) - ref).max():.2e}")
bad = np.abs(flash_attention(Q, K, V, 64, rescale=False) - ref).max()
print(f"WITHOUT rescaling acc: max error {bad:.3f}  (no crash: shapes are identical, only the numbers are wrong)")

for n in (200, 800, 3200):
    Q, K, V = (rng.normal(size=(n, 32)) for _ in range(3))
    t0 = time.perf_counter(); naive_attention(Q, K, V); tn = time.perf_counter() - t0
    t0 = time.perf_counter(); flash_attention(Q, K, V, 128); tf = time.perf_counter() - t0
    print(f"N={n:<5} naive {tn * 1e3:7.1f} ms | tiled {tf * 1e3:7.1f} ms | naive S matrix {n * n * 8 / 1e6:8.1f} MB")
print("On a CPU with numpy the tiled loop is not reliably faster (Python loop, no SRAM/HBM split); the point is O(N) extra memory and, on a GPU, far less HBM traffic.")
