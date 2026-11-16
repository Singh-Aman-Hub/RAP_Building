# Terms — FlashAttention and online softmax

Folder: `16_modern_llm_architecture/05_flash_attention_online_softmax` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [Naive attention](#naive-attention) · [Safe softmax](#safe-softmax) · [Online softmax](#online-softmax) · [Tiling (blocking)](#tiling-blocking) · [FlashAttention](#flashattention) · [IO-awareness](#io-awareness) · [HBM and SRAM](#hbm-and-sram) · [Memory-bound vs compute-bound](#memory-bound-vs-compute-bound) · [Causal tile skipping](#causal-tile-skipping) · [scaled_dot_product_attention (SDPA)](#scaled_dot_product_attention-sdpa) · [attn_implementation](#attn_implementation) · [FlashAttention-2 and -3](#flashattention-2-and--3) · [PagedAttention (contrast)](#pagedattention-contrast)

---

## Naive attention  `(concept)`
**1. Definition:** Compute S = QK^T / sqrt(d) as a full (N, N) matrix, softmax it, then multiply by V.
**2. In plain words:** Write every score on a big sheet.
**3. Example to quote:** > "Needs O(N^2) memory."
**Seen in:** `main.py`, `naive_attention`

## Safe softmax  `(concept)`
**1. Definition:** softmax computed after subtracting the row maximum, to avoid overflow in exp.
**2. In plain words:** Shift the scores so the biggest is 0.
**3. Example to quote:** > "Scores of scale 30 would overflow without it."
**Seen in:** `main.py`, `naive_attention`

## Online softmax  `(concept)`
**1. Definition:** Softmax computed in a single pass over chunks keeping a running max m and sum l, rescaling l by exp(m_old - m_new) when the max changes.
**2. In plain words:** Keep a running total and correct it whenever a bigger number shows up.
**3. Example to quote:** > "It is the recurrence at the heart of FlashAttention."
**Seen in:** `main.py`, `online_softmax`

## Tiling (blocking)  `(concept)`
**1. Definition:** Processing Q, K, V in small blocks that fit in fast on-chip memory.
**2. In plain words:** Work on a page at a time instead of the whole book.
**3. Example to quote:** > "Tile sizes (1,1), (7,13), (200,200) give the same output."
**Seen in:** `main.py`, `flash_attention`

## FlashAttention  `(concept)`
**1. Definition:** An exact, IO-aware attention algorithm that tiles the computation with online softmax and never writes the N x N matrix to GPU main memory.
**2. In plain words:** The same answer with far less memory traffic.
**3. Example to quote:** > "FlashAttention is exact, not an approximation."
**Seen in:** `main.py`, `flash_attention`

## IO-awareness  `(concept)`
**1. Definition:** Designing an algorithm around the cost of moving data between slow HBM and fast SRAM rather than counting FLOPs.
**2. In plain words:** The trip to the warehouse costs more than the arithmetic.
**3. Example to quote:** > "Attention is memory-bound."
**Seen in:** `WALKTHROUGH.md`

## HBM and SRAM  `(concept)`
**1. Definition:** HBM is large, slower GPU main memory; SRAM is small, very fast on-chip memory.
**2. In plain words:** Warehouse versus desk.
**3. Example to quote:** > "Tiles live in SRAM."
**Seen in:** `WALKTHROUGH.md`

## Memory-bound vs compute-bound  `(concept)`
**1. Definition:** Time limited by bytes moved versus operations performed.
**2. In plain words:** Waiting for deliveries versus working fast.
**3. Example to quote:** > "Decoding is memory-bound."
**Seen in:** `WALKTHROUGH.md`

## Causal tile skipping  `(concept)`
**1. Definition:** With a causal mask, key tiles entirely after the query tile are skipped, saving about half the work.
**2. In plain words:** No need to look at the future.
**3. Example to quote:** > "The break in the inner loop."
**Seen in:** `main.py`, `flash_attention`

## scaled_dot_product_attention (SDPA)  `(library)`
**1. Definition:** PyTorch's fused attention function that dispatches to FlashAttention, memory-efficient, cuDNN or a math fallback.
**2. In plain words:** One call, best kernel chosen for you.
**3. Example to quote:** > "SDPA matched our manual attention to 5e-7."
**Seen in:** `main.py`, section 5

## attn_implementation  `(syntax)`
**1. Definition:** Hugging Face argument choosing 'eager', 'sdpa' or 'flash_attention_2'.
**2. In plain words:** Pick the attention code path.
**3. Example to quote:** > "eager returns attention weights; sdpa is fastest by default."
**Seen in:** `WALKTHROUGH.md`

## FlashAttention-2 and -3  `(concept)`
**1. Definition:** Later versions with better work partitioning (v2) and Hopper-specific asynchrony and FP8 (v3).
**2. In plain words:** Faster kernels for the same maths.
**3. Example to quote:** > "Check the repository for current versions."
**Seen in:** `WALKTHROUGH.md`

## PagedAttention (contrast)  `(concept)`
**1. Definition:** vLLM's memory manager that stores the KV cache in fixed-size blocks.
**2. In plain words:** Not a different softmax; a different filing system.
**3. Example to quote:** > "FlashAttention is the kernel; PagedAttention is the memory layout."
**Seen in:** `WALKTHROUGH.md`

