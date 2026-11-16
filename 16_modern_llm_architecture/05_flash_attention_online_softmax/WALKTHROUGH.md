# Step 05 — FlashAttention and the online softmax

## The story so far
[Step 04](../04_attention_variants_and_kv_cache/WALKTHROUGH.md) shrank the *storage* of K and V. The attention *computation* has its own memory problem. [Module 21, step 08](../../21_llm_inference_engineering/08_hf_inference_batching_kvcache/WALKTHROUGH.md) used `use_cache`; here we look at what happens inside one attention call.

## The pain point
Naive attention builds `S = QK^T` of shape `(N, N)`, applies softmax to get `P`, then computes `PV`. For N = 131,072 and one head in fp16, S alone is 32 GB (S and P together 64 GB, as printed below). Real GPUs have a small, very fast on-chip memory (SRAM, a few hundred KB per streaming multiprocessor) and a large, much slower main memory (HBM). Naive attention writes S and P to HBM and reads them back several times. The arithmetic is cheap; the traffic is the cost. Attention is **memory-bound**, not compute-bound.

## The spark
Never build the N x N matrix. Process Q, K, V in **tiles** that fit in SRAM, and combine the partial softmaxes exactly with the **online softmax** trick: keep a running maximum `m` and running denominator `l` per row, and rescale what was accumulated whenever the maximum grows. The result is bit-for-bit the same attention (up to float rounding) with memory `O(N)` instead of `O(N^2)`, and far less HBM traffic. This is **FlashAttention** (Dao et al. 2022).

## Goal
Implement online softmax for one row, the tiled forward pass of FlashAttention in numpy (causal and non-causal, odd sizes), prove it equals naive attention to ~1e-16, show it survives huge scores, count the memory, and compare with torch's fused `scaled_dot_product_attention` (SDPA).

## The theory (read this before the interview)
**Safe softmax.** `softmax(s)_i = exp(s_i - m) / sum_j exp(s_j - m)` with `m = max(s)`; subtracting the max avoids overflow of `exp`.

**Online softmax.** Process the scores in chunks. After seeing chunk `c` keep
```
m_new = max(m_old, max(chunk))
l_new = l_old * exp(m_old - m_new) + sum(exp(chunk - m_new))
```
When the max changes, the old sum is rescaled by `exp(m_old - m_new)`, because every old term `exp(s - m_old)` becomes `exp(s - m_new) = exp(s - m_old) * exp(m_old - m_new)`. At the end `l` equals the denominator of the full softmax. One pass, O(1) extra memory.

**FlashAttention forward.** For each query tile `Q_i` (rows i..i+Bq) loop over K/V tiles `K_j, V_j`:
```
S_ij = Q_i K_j^T / sqrt(d)                  # small (Bq, Bk) tile, lives in SRAM
m_new = max(m, rowmax(S_ij));  P_ij = exp(S_ij - m_new)
l = l * exp(m - m_new) + rowsum(P_ij)
O_i = O_i * exp(m - m_new) + P_ij V_j       # rescale the accumulated output too
end:  O_i = O_i / l
```
The output accumulator is rescaled with the same factor as `l`: the step people forget (and the BREAK IT in this folder). With a causal mask, tiles entirely in the future are skipped (roughly half the work). The backward pass **recomputes** S tiles instead of storing them (extra FLOPs, far less memory traffic).

**IO-awareness.** The paper's argument is about reads/writes between HBM and SRAM, not FLOPs. Standard attention does `Theta(N d + N^2)` HBM accesses; FlashAttention does `Theta(N^2 d^2 / M)` where `M` is the SRAM size: fewer by a large factor for typical d and M. It is exact, not an approximation (unlike sparse or low-rank attention). Memory is linear in N, which is what allowed training at 32k-128k tokens.

**Versions.** FlashAttention-2 (2023): better parallelism over the sequence dimension and fewer non-matmul FLOPs, about 2x faster than v1. FlashAttention-3 (2024): uses Hopper-GPU features (asynchronous copies, warp specialisation, FP8) to reach much higher utilisation on H100. Newer GPU generations get newer kernels: check the FlashAttention repository for the current version and hardware support (not verified here). **FlashInfer** and vendor kernels (cuDNN attention) cover similar ground for serving.

**PyTorch SDPA.** `torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)` is one API that dispatches to the best backend: FlashAttention kernel, memory-efficient (xFormers-style) kernel, cuDNN, or a plain math fallback on CPU. Hugging Face models call it when `attn_implementation="sdpa"` (the default in recent versions); `"eager"` is the explicit, slow path that exposes attention weights (step 04 uses it); `"flash_attention_2"` uses the external package. Inference engines use **PagedAttention** (module 21, step 11) which is about *where the KV cache lives in memory*, not about the softmax maths; vLLM combines it with FlashAttention-style kernels.

**Limits.** FlashAttention does not reduce FLOPs (still O(N^2 d)); it removes the memory wall. For very long contexts compute still grows quadratically, which motivates sparse/hybrid attention (step 10).

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| FlashAttention | flash attention, FA2, FA3, "fused attention" | Family | exact tiled attention that avoids materialising N x N |
| SDPA | `scaled_dot_product_attention`, `attn_implementation="sdpa"` | No | PyTorch's API that picks a kernel (often FlashAttention) for you |
| PagedAttention | vLLM paging, paged KV cache | No | memory management for the KV cache, not a different softmax |
| Online softmax | streaming softmax, running softmax | Yes | softmax computed in one pass with a running max and sum |
| Tiling | blocking, chunking | Yes | process small blocks that fit in fast memory |
| Memory-bound vs compute-bound | bandwidth-bound vs FLOP-bound | No | time set by bytes moved vs by operations performed |
| HBM vs SRAM | GPU main memory vs on-chip memory | No | large and slow vs small and fast |
| Eager attention | naive attention, "math" backend | Yes | the explicit `softmax(QK^T)V` code |

## Run it
```bash
cd 16_modern_llm_architecture/05_flash_attention_online_softmax
python main.py       # ~5 s; numpy + torch
```

## Code, step by step
### 1. Imports and a seeded random generator  (`main.py` lines 14–19)
```python
import time

import numpy as np
import torch

rng = np.random.default_rng(0)
```
- **In simple words:** numpy for the tile maths, torch for the library comparison, a fixed seed for the same numbers each time.
- **Exactly:** `np.random.default_rng(0)` is numpy's modern generator; all random Q, K, V come from it.
- **Why it is here:** equality tests depend on the same inputs for both implementations.
### 2. Naive attention (the reference)  (`main.py` lines 23–29)
```python
def naive_attention(Q, K, V, causal=False):
    S = Q @ K.T / np.sqrt(Q.shape[1])                                  # (N, N)  <- the O(N^2) memory
    if causal:
        S = np.where(np.tril(np.ones_like(S, dtype=bool)), S, -np.inf)
    P = np.exp(S - S.max(axis=1, keepdims=True))                       # subtract row max: numerical stability
    P /= P.sum(axis=1, keepdims=True)
    return P @ V                                                       # (N, d)
```
- **In simple words:** the textbook formula: all scores at once, softmax, weighted sum of values.
- **Exactly:** `S = Q K^T / sqrt(d)` has shape `(N, N)`; for causal, positions above the diagonal become `-inf`; `S - S.max(axis=1, keepdims=True)` is the safe-softmax shift; `P` rows sum to 1; output `(N, d)`.
- **Why it is here:** the ground truth the tiled version must equal; also the thing whose memory we want to avoid.
### 3. Online softmax on one row  (`main.py` lines 33–39)
```python
def online_softmax(scores, chunk=4):
    m, l = -np.inf, 0.0
    for s in np.split(scores, range(chunk, len(scores), chunk)):
        m_new = max(m, s.max())                                        # new running max
        l = l * np.exp(m - m_new) + np.exp(s - m_new).sum()            # rescale old sum to the new max, add new terms
        m = m_new
    return np.exp(scores - m) / l                                      # final normalisation (only needed to compare)
```
- **In simple words:** read the numbers four at a time, always remembering the biggest so far and a running total, correcting the total whenever a bigger number shows up.
- **Exactly:** `np.split(scores, range(chunk, len(scores), chunk))` makes chunks of 4 (the last is shorter: 19 = 4·4 + 3). Per chunk: `m_new = max(m, s.max())`, `l = l·exp(m - m_new) + sum(exp(s - m_new))`. Initially `m = -inf` so `exp(-inf - m_new) = 0` kills the empty history. The final `exp(scores - m) / l` is only for comparing with the real softmax.
- **Why it is here:** this recurrence is the mathematical heart of FlashAttention.
### 4. Check one row  (`main.py` lines 42–44)
```python
row = rng.normal(size=19) * 5
ref = np.exp(row - row.max()) / np.exp(row - row.max()).sum()
print("online softmax == softmax:", np.allclose(online_softmax(row), ref), "| max diff", np.abs(online_softmax(row) - ref).max())
```
- **Exactly:** a 19-element row of scale 5; `allclose` is True and max difference 0.0.
- **In simple words:** reading in chunks gives exactly the same probabilities.
- **Why it is here:** the first piece of proof.
### 5. Tiled FlashAttention forward pass  (`main.py` lines 48–71)
```python
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
```
- **In simple words:** for each block of queries, walk across the keys block by block, keep running statistics for every query row, and fix up the running answer whenever a larger score appears.
- **Exactly:** outer loop over query tiles of `block_q` rows; `m (bq,)`, `l (bq,)`, `acc (bq, d)` are per-row running statistics. Inner loop over key tiles: `s = q k^T / sqrt(d)` is a `(bq, bk)` tile. Causal handling: tiles fully in the future are skipped with `break`; inside a tile `np.where(kj <= qi, s, -inf)` masks. `scale = exp(m - m_new)` multiplies BOTH `l` and `acc`. At the end `O[i:i+bq] = acc / l[:, None]`. Only `(bq, bk)` scores exist at any time. (The `break` condition `j > i + len(q) - 1` skips tiles whose first key index is beyond the last query row of the tile.)
- **Why it is here:** this loop is what the CUDA kernel does in SRAM; numpy just shows the maths.
### 6. Equality test with an awkward size  (`main.py` lines 74–80)
```python
N, d = 200, 32                                                         # N is not a multiple of 16 on purpose
Q, K, V = (rng.normal(size=(N, d)) for _ in range(3))
for causal in (False, True):
    diff = np.abs(flash_attention(Q, K, V, causal=causal) - naive_attention(Q, K, V, causal=causal)).max()
    print(f"causal={causal!s:<5} max |tiled - naive| = {diff:.2e}")
Q2, K2, V2 = (rng.normal(size=(64, 8)) * 30 for _ in range(3))        # huge scores: exp() would overflow without the max trick
print("huge scores, no NaN:", bool(np.isfinite(flash_attention(Q2, K2, V2)).all()))
```
- **Exactly:** N = 200 is not a multiple of 16, so the last tiles are partial. `max |tiled - naive|` is 5.55e-16 for both non-causal and causal: float64 round-off only. Scores of scale 30 (so `exp` would overflow without the max trick) still give finite output (`True`).
- **In simple words:** the tiled answer is the same answer, not an approximation.
- **Why it is here:** FlashAttention's main selling point is exactness.
### 7. Tile sizes do not matter for the result  (`main.py` lines 81–82)
```python
for bq, bk in [(1, 1), (7, 13), (200, 200)]:
    print(f"  tile sizes ({bq},{bk}): diff {np.abs(flash_attention(Q, K, V, bq, bk) - naive_attention(Q, K, V)).max():.2e}")
```
- **Exactly:** tiles (1,1), (7,13), (200,200) give 8.9e-16, 5.6e-16, 5.6e-16.
- **In simple words:** tile size changes speed, never the answer.
- **Why it is here:** in a real kernel tile size is tuned to SRAM; correctness is independent of it.
### 8. Memory arithmetic  (`main.py` lines 85–89)
```python
print("\nN        naive S+P (fp16, one head)   tiled working set (bq=bk=128, d=128)")
for n in (1_024, 8_192, 131_072):
    naive_bytes = 2 * n * n * 2                                        # S and P, 2 bytes each
    tiled_bytes = (128 * 128 + 3 * 128 * 128 + 2 * 128) * 2            # score tile + q,k,v tiles + m,l, fp16
    print(f"{n:<9,}{naive_bytes / 2**20:>12.1f} MB{'':>12}{tiled_bytes / 1024:>10.0f} KB (independent of N)")
```
- **Exactly:** naive stores S and P: `2 · N² · 2` bytes per head in fp16 = 4 MB at N = 1,024, 256 MB at 8,192, 65,536 MB (64 GB) at 131,072. The tiled working set with 128x128 tiles and d = 128 is `(128·128 + 3·128·128 + 2·128) · 2` bytes = 128 KB, independent of N. (Multiply naive by the number of heads and by batch size for the real figure.)
- **In simple words:** quadratic vs constant working memory.
- **Why it is here:** this is why long-context training is feasible at all.
### 9. SDPA vs hand-written causal attention  (`main.py` lines 92–95)
```python
Qt, Kt, Vt = (torch.randn(1, 4, 512, 64) for _ in range(3))             # (batch, heads, seq, head_dim)
manual = ((Qt @ Kt.transpose(-1, -2) / 8.0).masked_fill(~torch.ones(512, 512, dtype=torch.bool).tril(), float("-inf"))
          .softmax(-1) @ Vt)
sdpa = torch.nn.functional.scaled_dot_product_attention(Qt, Kt, Vt, is_causal=True)
```
- **In simple words:** the library call and our manual formula should agree.
- **Exactly:** tensors `(batch=1, heads=4, seq=512, head_dim=64)`; `is_causal=True` applies the lower-triangular mask internally.
- **Why it is here:** shows the one-line API you will actually use.
### 10. SDPA equality  (`main.py` lines 96–96)
```python
print("\nSDPA == manual causal attention:", torch.allclose(manual, sdpa, atol=1e-5), "| max diff", (manual - sdpa).abs().max().item())
```
- **Exactly:** `allclose(atol=1e-5)` True, max difference about 5e-7 (float32 rounding).
- **In simple words:** same answer from the optimised function.
- **Why it is here:** the same-maths check.
### 11. CPU timing  (`main.py` lines 97–107)
```python
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
```
- **Exactly:** `time.perf_counter()` around 3 repeats. In the recorded run: N=512 naive 13.8 ms vs SDPA 13.7 ms; N=2048 naive 461.8 ms vs SDPA 42.3 ms (timings change with machine load; the ratio grows with N because the naive path allocates and re-reads the 2048 x 2048 matrix). On a GPU the fused kernel's gain is larger because it avoids HBM round-trips.
- **In simple words:** the fused call is faster, and more so for longer sequences.
- **Why it is here:** a measured, not claimed, difference. It is not a rigorous benchmark.
### 12. Reminder  (`main.py` lines 108–108)
```python
print("note: FlashAttention is EXACT, not an approximation; it is the same maths with a different loop order.")
```
- **Exactly:** a printed note.
- **In simple words:** exact, just reordered.
- **Why it is here:** a very common interview trap ("is FlashAttention approximate?" no).

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `np.split(a, indices)` | cut an array at positions | `np.split(x, [4, 8])` |
| `np.where(cond, a, b)` | elementwise choice | causal masking |
| `np.tril(np.ones_like(S, dtype=bool))` | lower-triangular True matrix | causal mask |
| `m[:, None]` | add an axis so a per-row value broadcasts over columns | `(bq,) -> (bq, 1)` |
| `S.max(axis=1, keepdims=True)` | per-row max, keep 2-D shape | stable softmax |
| `for i in range(0, N, B)` | step through in blocks of B | tiling |
| `break` | leave the inner loop | skip future tiles |
| `(rng.normal(size=...) for _ in range(3))` | generator expression unpacked into 3 names | Q, K, V |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `np.exp`, `np.sqrt` | numpy | elementwise | array | array |
| `np.allclose(a, b)` | numpy | equality within tolerance | `atol`, `rtol` | bool |
| `torch.nn.functional.scaled_dot_product_attention` | torch | fused attention | `q,k,v,attn_mask,is_causal,dropout_p,scale` | tensor |
| `time.perf_counter()` | built-in | high-resolution timer | | float seconds |
| `tensor.masked_fill(mask, value)` | torch | fill where mask True | | tensor |

## Execution trace (real run)
```text
online softmax == softmax: True | max diff 0.0
causal=False max |tiled - naive| = 5.55e-16
causal=True  max |tiled - naive| = 5.55e-16
huge scores, no NaN: True
  tile sizes (1,1): diff 8.88e-16
  tile sizes (7,13): diff 5.55e-16
  tile sizes (200,200): diff 5.55e-16

N        naive S+P (fp16, one head)   tiled working set (bq=bk=128, d=128)
1,024             4.0 MB                   128 KB (independent of N)
8,192           256.0 MB                   128 KB (independent of N)
131,072       65536.0 MB                   128 KB (independent of N)

SDPA == manual causal attention: True | max diff 4.76837158203125e-07
CPU N=512: naive 13.8 ms | SDPA 13.7 ms  (on a GPU the gap is larger: SDPA avoids HBM round-trips)
CPU N=2048: naive 461.8 ms | SDPA 42.3 ms  (on a GPU the gap is larger: SDPA avoids HBM round-trips)
note: FlashAttention is EXACT, not an approximation; it is the same maths with a different loop order.
```
Reading it: all differences are at the 1e-16 level (float64), proving exactness. The memory table shows the quadratic term: 4 MB, 256 MB, 64 GB for one head's S and P, against a flat 128 KB tile.

## Experiments (change ONE thing)
Every row below was run on a temporary copy of `main.py` (the real file is untouched). The copies stop before section 5 (the CPU timing loop), so these are all numerical checks, not benchmarks. Only the named line changed.

| Change | What you will see | Why |
|---|---|---|
| add a lopsided tile size `(3, 1000)` to the tile test (measured) | diff 5.55e-16, the same as `(7, 13)` and `(200, 200)`; `(1, 1)` gives 8.88e-16 | Tile size is a speed and memory choice, never an accuracy choice. The result is the same maths in a different loop order. |
| `if causal and j > i + len(q) - 1:` -> `if False:` (measured) | causal diff stays 5.55e-16 | The early `break` is only an optimisation: the `-inf` mask already zeroes future keys, so skipping those tiles saves about half the work without changing the answer. |
| delete the `s = np.where(kj <= qi, s, -np.inf)` mask inside the tile (measured) | `causal=True` diff jumps to 2.52e+00 while `causal=False` stays 5.55e-16 | Skipping whole future tiles is not enough; the tile that straddles the diagonal still needs the per-element mask. |
| `acc = acc + p @ v` (no rescale) (measured) | diff 1.09 (non-causal), 1.25 (causal); tile `(200, 200)` still 5.55e-16 | See drill 1: with a single K tile there is no old running max to correct, so the bug hides. |
| `l = l + p.sum(axis=1)` (no rescale of the denominator) (measured) | diff 0.346 (non-causal), 0.815 (causal); tile `(200, 200)` still exact | The denominator needs the same correction factor as the numerator, otherwise the weights no longer sum to 1. |

## Debugging practice
**BREAK IT drill 1: forget to rescale the old accumulator.** In `flash_attention`:

```diff
-            acc = acc * scale[:, None] + p @ v
+            acc = acc + p @ v
```

Symptom (real run), no exception:

```text
causal=False max |tiled - naive| = 1.09e+00
causal=True  max |tiled - naive| = 1.25e+00
  tile sizes (1,1): diff 1.14e+00
  tile sizes (7,13): diff 1.13e+00
  tile sizes (200,200): diff 5.55e-16
```

Root cause: every tile is computed relative to its own running max `m_new`. When the max grows, everything accumulated before is relative to the old, smaller max and must be multiplied by `exp(m_old - m_new)`. Without it, old and new tiles are added on different scales. The `(200, 200)` tile is a single tile (N is 200), so no rescale is ever needed and that row looks fine: always test several tile sizes, including ones smaller than N.
Fix: restore `acc * scale[:, None]`.

**BREAK IT drill 2: do not subtract the running max.**

```diff
-            m_new = np.maximum(m, s.max(axis=1))                       # (bq,)
+            m_new = np.zeros(len(q))
```

Symptom (real run): the normal-size tests pass (diffs about 5e-16), but the huge-score test fails:

```text
huge scores, no NaN: False
RuntimeWarning: overflow encountered in exp
RuntimeWarning: invalid value encountered in add
RuntimeWarning: invalid value encountered in divide
```

Root cause: scores near 30 * 30 * sqrt(8)-scale are large enough that `exp(s)` overflows float64 to `inf`, and `inf / inf` is `nan`. The max subtraction is what keeps `exp` arguments at most 0. Fix: use the running max. This is also why the file has a "huge scores" test: with ordinary inputs the missing stabilisation is invisible.

## Common mistakes
- Saying FlashAttention is an approximation. It is exact (diffs about 1e-16 here); only the loop order and the memory traffic change.
- Claiming it reduces FLOPs. It does the same arithmetic, even slightly more through the rescaling; the win is fewer reads and writes of the N x N matrix to slow GPU memory (HBM).
- Rescaling `acc` but not `l` (or the other way round). Both need the same `exp(m_old - m_new)` factor.
- Testing only with one tile size or only with N a multiple of the tile: boundary and single-tile cases hide bugs (see drill 1).
- Treating this numpy file as fast. It is a teaching model of the maths; the speed comes from a fused GPU kernel keeping tiles in SRAM.
- Dropping the per-element causal mask because whole future tiles are skipped; the diagonal tile still needs it.

## Rebuild checklist
- [ ] safe softmax with the max subtracted
- [ ] the recurrence `m_new`, `l = l*exp(m-m_new) + sum(...)`
- [ ] the output accumulator rescaled by the same factor, normalised once at the end
- [ ] causal handling: skip future tiles and mask the diagonal tile
- [ ] state clearly: exact, IO-aware, same FLOPs, linear memory

## The new problem (cliffhanger)
So far every token uses every parameter. To grow model capacity (knowledge) without growing the cost per token, the MLP (two thirds of the parameters) can be replaced by many MLPs ("experts"), with a router choosing a few for each token. That is the Mixture-of-Experts layer behind DeepSeek-V3, Qwen3-235B and gpt-oss. Next: [Step 06 — Mixture of Experts](../06_mixture_of_experts/WALKTHROUGH.md).

> **Say it aloud (interview box):** "FlashAttention computes exactly the same attention but in tiles, using an online softmax that keeps a running max and sum per row and rescales the accumulator when the max changes. The N by N matrix is never written to GPU main memory, so memory is linear in sequence length and the kernel is faster because attention is memory-bound. PyTorch's SDPA picks such a kernel for you; PagedAttention is a different thing, a KV-cache memory manager."
