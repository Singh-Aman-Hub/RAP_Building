"""
GOAL:  Understand FlashAttention's maths: the ONLINE softmax lets us process keys/values in tiles without ever
       building the N x N score matrix.  Prove tiled == naive (to float precision), count memory, compare with
       torch's scaled_dot_product_attention (SDPA).

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 16_modern_llm_architecture/05_flash_attention_online_softmax/main.py
EXPECTED OUTPUT (shortened):
    online softmax == softmax: True
    max |tiled - naive| = ...e-16
    N=...  naive score matrix ... MB vs tiled working set ... KB
NEEDS: numpy, torch.  ~5 s.
"""
import time

import numpy as np
import torch

rng = np.random.default_rng(0)


# ---- 1. Naive attention: builds the full (N, N) score matrix ----------------------------------------------------
def naive_attention(Q, K, V, causal=False):
    S = Q @ K.T / np.sqrt(Q.shape[1])                                  # (N, N)  <- the O(N^2) memory
    if causal:
        S = np.where(np.tril(np.ones_like(S, dtype=bool)), S, -np.inf)
    P = np.exp(S - S.max(axis=1, keepdims=True))                       # subtract row max: numerical stability
    P /= P.sum(axis=1, keepdims=True)
    return P @ V                                                       # (N, d)


# ---- 2. Online softmax on ONE row: stream scores in chunks, keep running max m and running denominator l ------------
def online_softmax(scores, chunk=4):
    m, l = -np.inf, 0.0
    for s in np.split(scores, range(chunk, len(scores), chunk)):
        m_new = max(m, s.max())                                        # new running max
        l = l * np.exp(m - m_new) + np.exp(s - m_new).sum()            # rescale old sum to the new max, add new terms
        m = m_new
    return np.exp(scores - m) / l                                      # final normalisation (only needed to compare)


row = rng.normal(size=19) * 5
ref = np.exp(row - row.max()) / np.exp(row - row.max()).sum()
print("online softmax == softmax:", np.allclose(online_softmax(row), ref), "| max diff", np.abs(online_softmax(row) - ref).max())


# ---- 3. FlashAttention forward (tiled): loop over K/V tiles, never materialise (N, N) ---------------------------------
def flash_attention(Q, K, V, block_q=16, block_k=16, causal=False):
    N, d = Q.shape
    O = np.zeros((N, d))
    for i in range(0, N, block_q):                                     # outer loop: tile of queries
        q = Q[i:i + block_q]                                           # (bq, d)  stays "in SRAM"
        m = np.full(len(q), -np.inf)                                   # running row max         (bq,)
        l = np.zeros(len(q))                                           # running softmax denom   (bq,)
        acc = np.zeros((len(q), d))                                    # running unnormalised output (bq, d)
        for j in range(0, N, block_k):                                 # inner loop: tile of keys/values
            if causal and j > i + len(q) - 1:
                break                                                  # whole tile is in the future: skip it (saves ~half the work)
            k, v = K[j:j + block_k], V[j:j + block_k]                  # (bk, d)
            s = q @ k.T / np.sqrt(d)                                   # (bq, bk) small score tile
            if causal:
                qi, kj = np.arange(i, i + len(q))[:, None], np.arange(j, j + len(k))[None, :]
                s = np.where(kj <= qi, s, -np.inf)
            m_new = np.maximum(m, s.max(axis=1))                       # (bq,)
            p = np.exp(s - m_new[:, None])                             # (bq, bk)
            scale = np.exp(m - m_new)                                  # factor that corrects everything accumulated so far
            l = l * scale + p.sum(axis=1)
            acc = acc * scale[:, None] + p @ v                         # rescale old output, add this tile's contribution
            m = m_new
        O[i:i + len(q)] = acc / l[:, None]                             # normalise once at the end
    return O


N, d = 200, 32                                                         # N is not a multiple of 16 on purpose
Q, K, V = (rng.normal(size=(N, d)) for _ in range(3))
for causal in (False, True):
    diff = np.abs(flash_attention(Q, K, V, causal=causal) - naive_attention(Q, K, V, causal=causal)).max()
    print(f"causal={causal!s:<5} max |tiled - naive| = {diff:.2e}")
Q2, K2, V2 = (rng.normal(size=(64, 8)) * 30 for _ in range(3))        # huge scores: exp() would overflow without the max trick
print("huge scores, no NaN:", bool(np.isfinite(flash_attention(Q2, K2, V2)).all()))
for bq, bk in [(1, 1), (7, 13), (200, 200)]:
    print(f"  tile sizes ({bq},{bk}): diff {np.abs(flash_attention(Q, K, V, bq, bk) - naive_attention(Q, K, V)).max():.2e}")

# ---- 4. Memory: what has to live in fast memory -------------------------------------------------------------------------
print("\nN        naive S+P (fp16, one head)   tiled working set (bq=bk=128, d=128)")
for n in (1_024, 8_192, 131_072):
    naive_bytes = 2 * n * n * 2                                        # S and P, 2 bytes each
    tiled_bytes = (128 * 128 + 3 * 128 * 128 + 2 * 128) * 2            # score tile + q,k,v tiles + m,l, fp16
    print(f"{n:<9,}{naive_bytes / 2**20:>12.1f} MB{'':>12}{tiled_bytes / 1024:>10.0f} KB (independent of N)")

# ---- 5. Compare with torch SDPA (which dispatches to a fused flash/memory-efficient kernel on GPU) ----------------------
Qt, Kt, Vt = (torch.randn(1, 4, 512, 64) for _ in range(3))             # (batch, heads, seq, head_dim)
manual = ((Qt @ Kt.transpose(-1, -2) / 8.0).masked_fill(~torch.ones(512, 512, dtype=torch.bool).tril(), float("-inf"))
          .softmax(-1) @ Vt)
sdpa = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt, is_causal=True)
print("\nSDPA == manual causal attention:", torch.allclose(manual, sdpa, atol=1e-5), "| max diff", (manual - sdpa).abs().max().item())
for n in (512, 2048):
    q, k, v = (torch.randn(1, 8, n, 64) for _ in range(3))
    t0 = time.perf_counter()
    for _ in range(3):
        ((q @ k.transpose(-1, -2) / 8).softmax(-1) @ v)
    t_naive = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
    t_sdpa = (time.perf_counter() - t0) / 3
    print(f"CPU N={n}: naive {t_naive * 1e3:.1f} ms | SDPA {t_sdpa * 1e3:.1f} ms  (on a GPU the gap is larger: SDPA avoids HBM round-trips)")
print("note: FlashAttention is EXACT, not an approximation; it is the same maths with a different loop order.")

# TRY IT: change block_k to 1 and 200 in flash_attention; the answer must not change (only speed does).
# BREAK IT: delete the `* scale` rescale in `acc = acc * scale[:, None] + p @ v`. Output silently becomes wrong
#           (the diff jumps from 1e-16 to ~1): the online-softmax correction is the whole trick.
