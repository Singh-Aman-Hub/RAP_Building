# Step 11 — PagedAttention and memory fragmentation (simulation)

## The story so far
[Step 10](../10_vllm_reference/WALKTHROUGH.md) told you that vLLM fits many more requests on a GPU because of PagedAttention, but it only said so in words. This step makes you see it: two memory allocators, the same GPU budget, the same 300 requests.

## The pain point
The KV cache costs 0.5 MiB per token for a 7B model (step 08). The server does not know how long an answer will be, so the safe choice is to reserve the maximum length (2048 tokens, 1 GiB) for every request. Real requests average about 300 tokens. The rest of the slab sits empty but reserved, and no other request can use it. This is **internal fragmentation**. When slabs have different sizes and requests finish at different times, the free memory also breaks into holes that are each too small for the next slab: **external fragmentation**. It is the same disaster as a 1990s disk before defragmenting: lots of free space in total, none of it usable as one piece.

## The spark
Operating systems solved this decades ago with virtual memory. Cut memory into small fixed-size pages, give each program a page table, and let its pages sit anywhere. vLLM does the same for the KV cache: blocks of 16 tokens, allocated one at a time as the sequence grows, plus a **block table** that maps "logical block i of this sequence" to "physical block j on the GPU". Waste shrinks to the unused tail of the last block, and any free block is usable by anyone.

## Goal
Simulate both allocators in pure Python and numpy, and print memory utilisation and how many requests fit in the same budget.

## Run it
```bash
.venv/bin/python 06_llm_training_and_fast_inference/11_paged_attention_fragmentation/main.py
```
No GPU, no model, no network. A few seconds.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class X:` with `__init__` | a small object that keeps state (the allocator) | `a = PagedAllocator(8)` |
| `-(-a // b)` | ceiling division with integers | `-(-20 // 16)` = 2 |
| `list.pop()` / `list.append()` | use a list as a stack of free block ids | `free.pop()` takes the last id |
| `dict.setdefault(k, [])` | get the list for k, creating it if missing | block table per request |
| `[(start, length), ...]` | free list: each hole is a (start, length) pair | `[(0, 100)]` |
| `np.cumsum`, `np.searchsorted` | running total, and "where would this value be inserted" | how many requests fit |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `np.random.default_rng(0)` | numpy | seeded random generator (same numbers every run) | seed | Generator |
| `rng.lognormal(mean, sigma, size)` | numpy | skewed lengths: many short, few long | `size` | array `(N_REQ,)` |
| `np.clip(a, lo, hi)` | numpy | limit values to a range | lo, hi | array |
| `np.cumsum(a)` | numpy | running total | - | array same shape |
| `np.searchsorted(a, v, side="right")` | numpy | count of entries <= v in a sorted array | `side` | int |

## The two allocators
| | Contiguous | Paged |
|---|---|---|
| Unit | one slab per request | blocks of 16 tokens |
| When memory is taken | at admission, for the maximum length | one block at a time as tokens appear |
| Bookkeeping | free list of `(start, length)` holes | free stack of block ids + block table per request |
| Waste | internal (unused reservation) + external (holes) | only the unused tail of the last block |

## Execution trace (real run)
```
KV budget 8 GB = 16384 token slots = 1024 blocks of 16
requests: 300 | mean actual length 306 tokens | max 1145 | reserve per request 2048

A) STATIC SNAPSHOT: how many requests fit in the same budget?
  contiguous (reserve 2048):   8 requests | tokens really used  2385 of 16384 reserved -> 85% of the reservation wasted
  paged (block 16):          53 requests | tokens really used 15828 of 16224 allocated -> 2.4% wasted (last-block slack only)
  -> paging fits 6.6x more requests in the same memory

B) DYNAMIC RUN: 300 requests, same budget, same arrival order; one token per running request per step
  allocator                                 mem used  avg running  peak  steps  tokens/step
  contiguous, reserve 2048                     13.8%          7.9     8   6494          7.9
  contiguous, reserve prompt+max_tokens        30.3%         17.2    27   2961         17.3
  paged, block 16                              80.3%         52.5   111   2268         22.6
  external fragmentation (contiguous, variable slabs): in 2519 steps the next request was refused although enough memory was free in total; mean 2113 free slots were stuck in holes too small.
  paged preemptions (ran out of blocks mid-request): 690
  -> same GPU, same requests: paged finishes in 0.35x the steps = 2.9x throughput vs reserve-max

C) BLOCK TABLE: one request, tokens in order, physical blocks are NOT contiguous
  request A has 70 tokens in 5 blocks of 16; block table = [0, 1, 4, 3, 2]
    tokens   0- 15  logical block 0 -> physical block 0
    ...
  The attention kernel reads K,V through this table, so the sequence never needs one contiguous slab.
```
Reading it:
- **A** is the clean comparison: 8 requests versus 53 in 8 GB. The 85% waste is exactly "reserve 2048, use about 300".
- **B** shows the knock-on effect. Reserve-max keeps memory 14% full and runs 8 requests at once; paged keeps it 80% full with 52 on average. The middle row (reserve what the user declared) is better, but external fragmentation appears: holes totalling about 2100 slots were free yet unusable.
- The 690 paged preemptions are honest: this simple scheduler admits greedily, then runs out of blocks mid-run and has to restart some requests. Real vLLM also preempts (swap or recompute) but schedules more carefully. The numbers are one seeded simulation of a toy workload, not a benchmark of vLLM.

## Experiments
- Set `BLOCK_SIZE = 128` and re-run: the last-block waste in section A grows. This is why vLLM uses small blocks.
- Set `KV_BUDGET_GB = 24`: every allocator fits more, the ratio between them stays similar.
- Change `MAX_LEN` to 4096 (a longer-context model): contiguous gets worse, paged barely changes. Long-context models make paging more important.

## Debugging practice
In `PagedAllocator.free`, replace `self.free_blocks.extend(self.block_table.pop(rid))` with `self.block_table.pop(rid)`. The blocks are never returned (a leak), so the pool runs dry, nothing can be admitted, and the run ends with `RuntimeError: simulation did not finish` (confirmed). Section A still prints first because it never frees anything. This is the real production failure of a leaked KV block: throughput decays over hours until the server must be restarted.

## Rebuild checklist
- [ ] Compute KV bytes per token from layers, heads, head dim and dtype.
- [ ] Write a free-stack allocator that hands out block ids and a block table per request.
- [ ] Explain internal versus external fragmentation with one example each.
- [ ] Say why the attention kernel needs the block table.

## The new problem (cliffhanger)
Paging solved the memory side. The model is now fast, cheap to serve and fine-tunable, but a raw model still only completes text: ask "Summarise this email" and it may continue the email instead. We have the engine; we do not yet know how to talk to it. That is Act IV, starting at [07_prompt_engineering](../../07_prompt_engineering/).

> **Say it aloud (interview box):** Reserving the maximum KV-cache length for every request wastes most of the GPU, because real answers are short and the leftover holes are too small to reuse, like a fragmented disk. PagedAttention splits the cache into small fixed blocks and keeps a block table per request that maps logical blocks to any physical block, so waste is only the tail of the last block. In my simulation that fit about 6.6 times more requests in the same budget, which is why vLLM gets much higher throughput.
