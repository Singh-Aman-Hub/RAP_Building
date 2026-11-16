# Terms — PagedAttention and memory fragmentation

Folder: `06_llm_training_and_fast_inference/11_paged_attention_fragmentation` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [06 README](../README.md)

**Terms in this folder:** [Internal fragmentation](#internal-fragmentation) · [External fragmentation](#external-fragmentation) · [Contiguous pre-allocation](#contiguous-pre-allocation) · [Block (page) size](#block-page-size) · [Block table](#block-table) · [Free list and first-fit](#free-list-and-first-fit) · [Preemption](#preemption) · [Memory utilisation](#memory-utilisation) · [np.searchsorted](#npsearchsorted) · [Ceiling division](#ceiling-division)

---

## Internal fragmentation  `(concept)`
**1. Definition:** Memory that is reserved for a request but not used by it.
**2. In plain words:** A request reserves 2048 token slots and only produces 300 tokens: the other 1748 are held and wasted. In the simulation this was 85% of the reservation.
**3. Example to quote:** > "Reserving the maximum length per request wastes most of the KV cache because real answers are short."
**Seen in:** `main.py`, section A

## External fragmentation  `(concept)`
**1. Definition:** Free memory that is split into holes, each too small for the next request, so the total is free but not usable.
**2. In plain words:** Like a disk full of small gaps: 3000 slots are free in total, but no single gap holds the 1500-slot slab a new request needs.
**3. Example to quote:** > "Slabs of different sizes leave holes behind when requests finish, and the next big slab cannot use them."
**Seen in:** `main.py`, `ContiguousAllocator`, section B

## Contiguous pre-allocation  `(concept)`
**1. Definition:** Giving each request one unbroken region of memory sized for its maximum possible length.
**2. In plain words:** The simple, safe way to allocate a KV cache, and the reason naive servers handle few requests.
**3. Example to quote:** > "Contiguous pre-allocation is simple but it capped us at 8 concurrent requests."
**Seen in:** `main.py`

## Block (page) size  `(concept)`
**1. Definition:** The number of tokens whose K,V vectors live in one physical block (16 in vLLM by default; `BLOCK_SIZE` here).
**2. In plain words:** Small blocks waste less in the last block but mean more bookkeeping; huge blocks bring back internal fragmentation.
**3. Example to quote:** > "I use 16-token blocks, so at most 15 slots per request are wasted."
**Seen in:** `main.py`, `BLOCK_SIZE`

## Block table  `(concept)`
**1. Definition:** A per-request list that maps logical block numbers (position in the sequence) to physical block ids in GPU memory.
**2. In plain words:** The page table of the KV cache. Blocks of one sequence can be anywhere; the attention kernel follows the table to find them.
**3. Example to quote:** > "The block table `[0, 1, 4, 3, 2]` says tokens 32-47 live in physical block 4."
**Seen in:** `main.py`, section C

## Free list and first-fit  `(concept)`
**1. Definition:** A free list records which regions are unused; first-fit gives a request the first hole that is big enough.
**2. In plain words:** The classic allocator for variable-size slabs. It is simple, and it is exactly what produces external fragmentation over time.
**3. Example to quote:** > "First-fit with variable slabs leaves small holes that nobody can use."
**Seen in:** `main.py`, `ContiguousAllocator.alloc`

## Preemption  `(concept)`
**1. Definition:** Pausing or evicting a running request to free its memory, then resuming it later (swap to CPU or recompute).
**2. In plain words:** What a paged server does when it runs out of blocks mid-generation. Here we free the blocks and restart the request.
**3. Example to quote:** > "When blocks ran out the scheduler preempted a request and put it back at the front of the queue."
**Seen in:** `main.py`, `run()`

## Memory utilisation  `(concept)`
**1. Definition:** Tokens that really hold K,V divided by the total token slots in the budget.
**2. In plain words:** How much of the memory is doing work. 14% for reserve-max, 80% for paged in this run.
**3. Example to quote:** > "Paging raised KV-cache utilisation from 14% to 80% on the same GPU."
**Seen in:** `main.py`, section B

## np.searchsorted  `(third-party: numpy)`
**1. Definition:** `np.searchsorted(a, v, side="right")` returns where `v` would be inserted in the sorted array `a`; with `side="right"` that is the count of entries `<= v`.
**2. In plain words:** On a running total of blocks it answers "how many requests fit before the budget runs out".
**3. Example to quote:** > "`searchsorted` on the cumulative block counts gave 53 requests."
**Seen in:** `main.py`, section A

## Ceiling division  `(concept)`
**1. Definition:** `-(-a // b)` rounds `a / b` up to the next integer using only integer maths.
**2. In plain words:** 20 tokens in blocks of 16 need `ceil(20/16)` = 2 blocks.
**3. Example to quote:** > "`ceil_div(70, 16)` is 5 blocks."
**Seen in:** `main.py`, `ceil_div`
