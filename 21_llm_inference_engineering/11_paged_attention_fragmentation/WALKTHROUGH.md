# Step 11 — PagedAttention and memory fragmentation (simulation)

## The story so far
[Step 10](../10_latency_throughput_benchmark/WALKTHROUGH.md) measured that batching raises throughput until something runs out, and on a real GPU that something is KV-cache memory. This step shows why a naive server wastes most of that memory, and how PagedAttention (the idea behind vLLM; the engine itself is surveyed in [step 15](../15_serving_engines_landscape/WALKTHROUGH.md)) fixes it: two memory allocators, the same GPU budget, the same 300 requests.

## The pain point
The KV cache costs 0.5 MiB per token for a 7B model ([step 03](../03_gpu_memory_calculator/WALKTHROUGH.md)). The server does not know how long an answer will be, so the safe choice is to reserve the maximum length (2048 tokens, 1 GiB) for every request. Real requests average about 300 tokens. The rest of the slab sits empty but reserved, and no other request can use it. This is **internal fragmentation**. When slabs have different sizes and requests finish at different times, the free memory also breaks into holes that are each too small for the next slab: **external fragmentation**. It is the same disaster as a 1990s disk before defragmenting: lots of free space in total, none of it usable as one piece.

## The spark
Operating systems solved this decades ago with virtual memory. Cut memory into small fixed-size pages, give each program a page table, and let its pages sit anywhere. vLLM does the same for the KV cache: blocks of 16 tokens, allocated one at a time as the sequence grows, plus a **block table** that maps "logical block i of this sequence" to "physical block j on the GPU". Waste shrinks to the unused tail of the last block, and any free block is usable by anyone.

## Goal
Simulate both allocators in pure Python and numpy, and print memory utilisation and how many requests fit in the same budget.

## Run it
```bash
.venv/bin/python 21_llm_inference_engineering/11_paged_attention_fragmentation/main.py
```
No GPU, no model, no network. A few seconds.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Internal / external fragmentation | wasted memory | Different | Internal = unused space inside a reservation (reserving 2048, using 300); external = free space split into holes too small to use. |
| Contiguous allocation / slab | static reservation | Yes | One unbroken run of memory per request. |
| Paged / block-based KV cache | PagedAttention | Yes | Memory cut into fixed blocks (16 tokens) mapped through a block table. |
| Block table / page table | logical-to-physical map | Same idea | Maps a sequence's logical block number to a physical block id. |
| Preemption / eviction / recompute | swap out | Related | Preemption stops a running request, frees its blocks and re-queues it; eviction is the general term. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–15)

```python
"""
GOAL:  See WHY GPU memory fragments when every request reserves its own contiguous KV-cache slab, and how
       PagedAttention (fixed-size blocks + a block table) fixes it.  Pure Python + numpy simulation: no GPU,
       no model, no network.  Same budget, same requests, two allocators.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 21_llm_inference_engineering/11_paged_attention_fragmentation/main.py
EXPECTED OUTPUT (shape of it, numbers come from the seeded run):
    A) static snapshot: how many requests fit in the same KV budget
    B) dynamic run: 300 requests arrive, memory utilisation + concurrency + total steps for each allocator
    C) block table of one request (logical block -> physical block)
NEEDS: numpy only.  Runs in a few seconds.

Builds on: 03_gpu_memory_calculator (KV cache bytes per token) and 10_latency_throughput_benchmark (why KV memory limits batch size).
"""
```

- **In simple words:** What the simulation shows.
- **Exactly:** Docstring. Pure numpy; no GPU. It compares two allocators on the same requests.
- **Why it is here:** Sets expectations before the code.

### 2. Import numpy and seed  (`main.py` lines 17–19)

```python
import numpy as np

rng = np.random.default_rng(0)
```

- **In simple words:** A seeded random generator so the traffic is identical every run.
- **Exactly:** `np.random.default_rng(0)`.
- **Why it is here:** The comparison is only fair if all allocators see identical requests.

### 3. Memory budget  (`main.py` lines 21–30)

```python
# ---------------------------------------------------------------------------------------------------------
# Budget.  Llama-7B style KV cache in fp16: 2 (K and V) * 32 layers * 32 heads * 128 dim * 2 bytes
# = 524,288 bytes = 0.5 MiB for EVERY token of EVERY request (formula from step 03).
# ---------------------------------------------------------------------------------------------------------
BYTES_PER_TOKEN = 2 * 32 * 32 * 128 * 2
KV_BUDGET_GB = 8                                           # GPU memory left for KV cache after the weights
BUDGET_TOKENS = int(KV_BUDGET_GB * 2**30 // BYTES_PER_TOKEN)   # 16384 token slots in total
MAX_LEN = 2048                                             # model's maximum sequence length
BLOCK_SIZE = 16                                            # tokens per page (vLLM default is 16)
N_BLOCKS = BUDGET_TOKENS // BLOCK_SIZE                     # 1024 physical blocks
```

- **In simple words:** Work out how many token slots fit in 8 GB of KV memory.
- **Exactly:** `BYTES_PER_TOKEN = 2*32*32*128*2 = 524,288` (0.5 MiB per token). `BUDGET_TOKENS = 8 * 2**30 // 524288 = 16384`. `MAX_LEN=2048`, `BLOCK_SIZE=16`, `N_BLOCKS = 16384 // 16 = 1024`.
- **Why it is here:** Everything is measured in token slots, so the simulation stays simple.

### 4. Generate the workload  (`main.py` lines 32–42)

```python
# ---------------------------------------------------------------------------------------------------------
# Workload: real chat traffic is mostly short, a few are long.  Lognormal lengths, clipped to [16, MAX_LEN].
# Each request has a prompt length and an output length we only learn at the end (when the model emits EOS).
# ---------------------------------------------------------------------------------------------------------
N_REQ = 300
prompt_len = np.clip(rng.lognormal(mean=4.6, sigma=0.8, size=N_REQ), 8, 1024).astype(int)     # shape: (N_REQ,)
output_len = np.clip(rng.lognormal(mean=4.8, sigma=0.9, size=N_REQ), 4, 1024).astype(int)     # shape: (N_REQ,)
total_len = np.minimum(prompt_len + output_len, MAX_LEN)                                       # shape: (N_REQ,)
# The server does not know output_len in advance.  A simple server reserves for the worst case: MAX_LEN.
print(f"KV budget {KV_BUDGET_GB} GB = {BUDGET_TOKENS} token slots = {N_BLOCKS} blocks of {BLOCK_SIZE}")
print(f"requests: {N_REQ} | mean actual length {total_len.mean():.0f} tokens | max {total_len.max()} | reserve per request {MAX_LEN}")
```

- **In simple words:** Create 300 fake requests; most are short, a few long.
- **Exactly:** Log-normal prompt and output lengths clipped to ranges, each array shape `(300,)`; `total_len = min(prompt+output, 2048)`. Real: mean actual length 306 tokens, max 1145.
- **Why it is here:** Real chat traffic is skewed, which is what makes worst-case reservation wasteful.

### 5. Ceiling division helper  (`main.py` lines 45–46)

```python
def ceil_div(a, b):
    return -(-a // b)
```

- **In simple words:** Divide and round up.
- **Exactly:** `-(-a // b)` equals `ceil(a / b)` and works on numpy arrays.
- **Why it is here:** A request of 17 tokens needs 2 blocks, not 1.

### 6. Static snapshot: contiguous reservation  (`main.py` lines 52–58)

```python
print("\nA) STATIC SNAPSHOT: how many requests fit in the same budget?")

# Contiguous: each request reserves MAX_LEN slots no matter how short it will be.
fit_contiguous = BUDGET_TOKENS // MAX_LEN
used_contiguous = total_len[:fit_contiguous].sum()
print(f"  contiguous (reserve {MAX_LEN}): {fit_contiguous:3d} requests | tokens really used {used_contiguous:5d} of "
      f"{fit_contiguous * MAX_LEN} reserved -> {100 * (1 - used_contiguous / (fit_contiguous * MAX_LEN)):.0f}% of the reservation wasted")
```

- **In simple words:** Count how many requests fit if each reserves the maximum.
- **Exactly:** `16384 // 2048 = 8` requests fit. Real: only 2385 of 16384 reserved slots used, 85% wasted.
- **Why it is here:** Shows internal fragmentation directly.

### 7. Static snapshot: paged allocation  (`main.py` lines 60–68)

```python
# Paged: a request owns ceil(len / BLOCK_SIZE) blocks; only its LAST block can be partly empty.
blocks_needed = ceil_div(total_len, BLOCK_SIZE)            # shape: (N_REQ,)
cum_blocks = np.cumsum(blocks_needed)                      # shape: (N_REQ,)  blocks used if we admit the first i+1 requests
fit_paged = int(np.searchsorted(cum_blocks, N_BLOCKS, side="right"))
used_paged = total_len[:fit_paged].sum()
reserved_paged = int(cum_blocks[fit_paged - 1]) * BLOCK_SIZE
print(f"  paged (block {BLOCK_SIZE}):         {fit_paged:3d} requests | tokens really used {used_paged:5d} of "
      f"{reserved_paged} allocated -> {100 * (1 - used_paged / reserved_paged):.1f}% wasted (last-block slack only)")
print(f"  -> paging fits {fit_paged / fit_contiguous:.1f}x more requests in the same memory")
```

- **In simple words:** Count how many requests fit with block-based allocation.
- **Exactly:** `blocks_needed = ceil(total_len / 16)` shape `(300,)`; `cumsum` gives blocks used if the first i+1 are admitted; `searchsorted(cum_blocks, 1024, side="right")` finds the cut. Real: 53 requests, 2.4% wasted, 6.6x more requests.
- **Why it is here:** Only the last block of each request can be partly empty.

### 8. Declared max_tokens reservation sizes  (`main.py` lines 78–80)

```python
declared_max_new = rng.choice([128, 256, 512, 1024], size=N_REQ)                                # shape: (N_REQ,)
declared_max_new = np.maximum(declared_max_new, output_len)         # the user's cap is never below what is produced
reserve_smart = np.minimum(prompt_len + declared_max_new, MAX_LEN)                               # shape: (N_REQ,)
```

- **In simple words:** A smarter contiguous server that reserves prompt + the user's max_tokens.
- **Exactly:** `declared_max_new` is chosen from 128/256/512/1024 and raised to at least the real output; `reserve_smart` shape `(300,)`.
- **Why it is here:** Gives slabs of different sizes, which is what produces external fragmentation.

### 9. ContiguousAllocator: create and allocate  (`main.py` lines 83–99)

```python
class ContiguousAllocator:
    """First-fit allocator over one long array of token slots.  free_list = sorted [(start, length), ...]."""

    def __init__(self, total):
        self.free_list = [(0, total)]
        self.slabs = {}                                    # request id -> (start, length)

    def alloc(self, rid, size):
        for i, (start, length) in enumerate(self.free_list):
            if length >= size:                             # first hole that is big enough
                self.slabs[rid] = (start, size)
                if length == size:
                    self.free_list.pop(i)
                else:
                    self.free_list[i] = (start + size, length - size)
                return True
        return False
```

- **In simple words:** A first-fit allocator over one long array.
- **Exactly:** `free_list` is a list of `(start, length)` holes. `alloc` takes the first hole at least as big as `size`, records the slab and shrinks or removes the hole; returns False if none is large enough.
- **Why it is here:** This models how a naive server reserves KV memory.

### 10. ContiguousAllocator: free and merge  (`main.py` lines 101–112)

```python
    def free(self, rid):
        start, size = self.slabs.pop(rid)
        self.free_list.append((start, size))
        self.free_list.sort()
        merged = [self.free_list[0]]                       # merge neighbouring holes
        for s, l in self.free_list[1:]:
            ps, pl = merged[-1]
            if ps + pl == s:
                merged[-1] = (ps, pl + l)
            else:
                merged.append((s, l))
        self.free_list = merged
```

- **In simple words:** Give a slab back and join neighbouring holes.
- **Exactly:** Append the hole, sort, then walk once merging holes where `start + length` equals the next start.
- **Why it is here:** Without merging, memory fragments permanently.

### 11. ContiguousAllocator: free space metrics  (`main.py` lines 114–118)

```python
    def free_tokens(self):
        return sum(l for _, l in self.free_list)

    def largest_hole(self):
        return max((l for _, l in self.free_list), default=0)
```

- **In simple words:** How much memory is free, and how big is the biggest hole.
- **Exactly:** `free_tokens()` sums the holes; `largest_hole()` is the max length. External fragmentation = free_tokens large but largest_hole small.
- **Why it is here:** Needed to measure external fragmentation.

### 12. PagedAllocator: setup  (`main.py` lines 121–126)

```python
class PagedAllocator:
    """Physical blocks in a free stack + one block table (list of physical block ids) per request."""

    def __init__(self, n_blocks):
        self.free_blocks = list(range(n_blocks - 1, -1, -1))   # stack of free physical block ids
        self.block_table = {}                                  # request id -> [physical block id, ...]
```

- **In simple words:** A pool of equal blocks and a table for each request.
- **Exactly:** `free_blocks` is a stack of ids `1023..0`, so `pop()` hands out block 0 first; `block_table` maps request id to a list of physical ids.
- **Why it is here:** Equal-size blocks cannot leave awkward holes.

### 13. PagedAllocator: allocate on demand  (`main.py` lines 128–135)

```python
    def alloc_tokens(self, rid, n_tokens):
        need = ceil_div(n_tokens, BLOCK_SIZE) - len(self.block_table.get(rid, []))
        if need > len(self.free_blocks):
            return False
        table = self.block_table.setdefault(rid, [])
        for _ in range(need):
            table.append(self.free_blocks.pop())
        return True
```

- **In simple words:** Give a request just enough extra blocks to hold n tokens.
- **Exactly:** `need = ceil(n / 16) - blocks already owned`; if the pool is too small return False, else pop blocks and append to its table.
- **Why it is here:** Memory is claimed as the answer grows, not reserved up front.

### 14. PagedAllocator: free  (`main.py` lines 137–138)

```python
    def free(self, rid):
        self.free_blocks.extend(self.block_table.pop(rid))
```

- **In simple words:** Return all of a request's blocks.
- **Exactly:** `extend` puts the ids back on the free stack.
- **Why it is here:** BREAK IT shows what happens if this leaks.

### 15. Simulation function: setup  (`main.py` lines 141–148)

```python
def run(policy):
    """Simulate until every request has finished.  Returns metrics."""
    contiguous = policy in ("contiguous_max", "contiguous_declared")
    alloc = ContiguousAllocator(BUDGET_TOKENS) if contiguous else PagedAllocator(N_BLOCKS)
    queue = list(range(N_REQ))                             # waiting request ids, FIFO
    running = {}                                           # rid -> tokens produced so far (prompt counts as present)
    step = 0
    util_samples, conc_samples, ext_samples, preempted = [], [], [], 0
```

- **In simple words:** Prepare one allocator, a queue and counters for the chosen policy.
- **Exactly:** `contiguous` is True for the two slab policies. `queue` holds request ids 0..299 in FIFO order; `running` maps id to tokens produced.
- **Why it is here:** One function runs all three policies so they are compared fairly.

### 16. Admit waiting requests  (`main.py` lines 149–163)

```python
    while queue or running:
        # ---- admit: head of the queue, in order, while it fits ----
        while queue:
            rid = queue[0]
            if policy == "contiguous_max":
                ok = alloc.alloc(rid, MAX_LEN)
            elif policy == "contiguous_declared":
                ok = alloc.alloc(rid, int(reserve_smart[rid]))
            else:
                # paged: only the prompt (+1 token) needs blocks now; the rest is allocated on demand
                ok = alloc.alloc_tokens(rid, int(prompt_len[rid]) + 1)
            if not ok:
                break
            queue.pop(0)
            running[rid] = 0
```

- **In simple words:** Let requests in from the front of the queue while memory allows.
- **Exactly:** Contiguous reserves 2048 or `reserve_smart`; paged reserves only prompt + 1 tokens. If admission fails the loop stops (strict FIFO).
- **Why it is here:** The policy difference starts here.

### 17. One decoding step per running request  (`main.py` lines 164–177)

```python
        # ---- one decoding step: every running request makes one token ----
        for rid in list(running):
            running[rid] += 1
            if not contiguous:
                if not alloc.alloc_tokens(rid, int(prompt_len[rid]) + running[rid] + 1):
                    # out of blocks: preempt this request (free its blocks, redo later)
                    alloc.free(rid)
                    del running[rid]
                    queue.insert(0, rid)
                    preempted += 1
                    continue
            if running[rid] >= output_len[rid]:
                alloc.free(rid)
                del running[rid]
```

- **In simple words:** Every running request makes one token; finished ones leave.
- **Exactly:** Paged grows its allocation by one token each step; if no block is free the request is preempted (blocks freed, put back at the front). A request finishes when `running[rid] >= output_len[rid]`.
- **Why it is here:** Models continuous batching and shows why paging needs a preemption rule.

### 18. Measure utilisation, concurrency, fragmentation  (`main.py` lines 178–186)

```python
        # ---- measure memory utilisation: tokens that really hold KV / total budget ----
        used_tokens = sum(int(prompt_len[r]) + running[r] for r in running)
        util_samples.append(used_tokens / BUDGET_TOKENS)
        conc_samples.append(len(running))
        if contiguous and queue:                           # external fragmentation: free memory the next request cannot use
            need = MAX_LEN if policy == "contiguous_max" else int(reserve_smart[queue[0]])
            if alloc.free_tokens() >= need > alloc.largest_hole():
                ext_samples.append(alloc.free_tokens() - alloc.largest_hole())
        step += 1
```

- **In simple words:** Record how full memory is and how many requests run.
- **Exactly:** `used_tokens` counts only tokens that hold KV; `util = used / 16384`. External fragmentation is recorded when the next request could fit by total free space but not by the largest hole.
- **Why it is here:** These are the numbers in the final table.

### 19. Safety stop and return metrics  (`main.py` lines 187–190)

```python
        if step > 200000:
            raise RuntimeError("simulation did not finish")
    return dict(steps=step, util=float(np.mean(util_samples)), conc=float(np.mean(conc_samples)),
                peak=int(np.max(conc_samples)), preempted=preempted, ext=ext_samples)
```

- **In simple words:** Stop a runaway loop and hand back the results.
- **Exactly:** Raise `RuntimeError` after 200000 steps; return a dict of steps, mean utilisation, mean and peak concurrency, preemptions, fragmentation samples.
- **Why it is here:** A bug (like leaking blocks) would otherwise loop forever.

### 20. Run all three policies and print the table  (`main.py` lines 193–209)

```python
print("\nB) DYNAMIC RUN: 300 requests, same budget, same arrival order; one token per running request per step")
results = {name: run(name) for name in ("contiguous_max", "contiguous_declared", "paged")}
labels = {"contiguous_max": f"contiguous, reserve {MAX_LEN}",
          "contiguous_declared": "contiguous, reserve prompt+max_tokens",
          "paged": f"paged, block {BLOCK_SIZE}"}
print(f"  {'allocator':40s} {'mem used':>9s} {'avg running':>12s} {'peak':>5s} {'steps':>6s} {'tokens/step':>12s}")
total_tokens = int(output_len.sum())
for name, r in results.items():
    print(f"  {labels[name]:40s} {100 * r['util']:8.1f}% {r['conc']:12.1f} {r['peak']:5d} {r['steps']:6d} "
          f"{total_tokens / r['steps']:12.1f}")
r = results["contiguous_declared"]
if r["ext"]:
    print(f"  external fragmentation (contiguous, variable slabs): in {len(r['ext'])} steps the next request was refused although "
          f"enough memory was free in total; mean {np.mean(r['ext']):.0f} free slots were stuck in holes too small.")
print(f"  paged preemptions (ran out of blocks mid-request): {results['paged']['preempted']}")
speedup = results["contiguous_max"]["steps"] / results["paged"]["steps"]
print(f"  -> same GPU, same requests: paged finishes in {1 / speedup:.2f}x the steps = {speedup:.1f}x throughput vs reserve-max")
```

- **In simple words:** Simulate all three and compare.
- **Exactly:** Real: contiguous-max 13.8% memory used, 7.9 running, 6494 steps; contiguous-declared 30.3%, 17.2, 2961 steps (external fragmentation in 2519 steps, 2113 slots stuck); paged 80.3%, 52.5 running, peak 111, 2268 steps, 690 preemptions. Speedup 2.9x versus reserve-max.
- **Why it is here:** This is the end result you quote: same GPU, 2.9x throughput.

### 21. Show a block table  (`main.py` lines 215–226)

```python
print("\nC) BLOCK TABLE: one request, tokens in order, physical blocks are NOT contiguous")
demo = PagedAllocator(12)
demo.alloc_tokens("A", 20)                                 # 2 blocks
demo.alloc_tokens("B", 40)                                 # 3 blocks
demo.alloc_tokens("C", 10)                                 # 1 block
demo.free("B")                                             # B finishes: holes appear in the physical pool
demo.alloc_tokens("A", 70)                                 # A keeps growing and grabs B's old blocks
table_A = np.array(demo.block_table["A"])                  # shape: (n_logical_blocks,) -> physical block ids
print(f"  request A has {70} tokens in {len(table_A)} blocks of {BLOCK_SIZE}; block table = {table_A.tolist()}")
for logical, physical in enumerate(table_A):
    print(f"    tokens {logical * BLOCK_SIZE:3d}-{min(logical * BLOCK_SIZE + BLOCK_SIZE, 70) - 1:3d}  logical block {logical} -> physical block {physical}")
print("  The attention kernel reads K,V through this table, so the sequence never needs one contiguous slab.")
```

- **In simple words:** Make a tiny pool and watch one request's blocks get scattered.
- **Exactly:** A grows to 2 blocks, B takes 3, C takes 1; B finishes; A grows to 70 tokens (5 blocks) and re-uses B's old blocks. Real table: `[0, 1, 4, 3, 2]`; tokens 64-69 live in physical block 2.
- **Why it is here:** Shows the sequence is logically contiguous but physically scattered.

### 22. TRY IT / BREAK IT  (`main.py` lines 228–231)

```python
# TRY IT : set BLOCK_SIZE = 128 and re-run A) - the last-block waste grows; vLLM picks small blocks (16) for this reason.
# TRY IT : set KV_BUDGET_GB = 24 - every allocator fits more, but the ratio between them stays about the same.
# BREAK IT: in PagedAllocator.free, replace extend(...) with a no-op (a leaked block table); the paged run slows down,
#           then stalls with "RuntimeError: simulation did not finish" because blocks are never returned.
```

- **In simple words:** Experiments in comments.
- **Exactly:** Leaking blocks in `free` eventually raises `RuntimeError: simulation did not finish`.
- **Why it is here:** Proves the safety check works.

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
Paging solved the memory side: more requests now fit at once. But a fixed batch of requests that all start and end together still wastes compute: short requests finish and their slots sit idle while the longest one drags on, and new arrivals wait for the whole batch. Who decides, at every single decode step, which requests run? That is the scheduler: [Step 12, continuous batching and prefix caching](../12_continuous_batching_prefix_caching/WALKTHROUGH.md).

> **Say it aloud (interview box):** Reserving the maximum KV-cache length for every request wastes most of the GPU, because real answers are short and the leftover holes are too small to reuse, like a fragmented disk. PagedAttention splits the cache into small fixed blocks and keeps a block table per request that maps logical blocks to any physical block, so waste is only the tail of the last block. In my simulation that fit about 6.6 times more requests in the same budget, which is why vLLM gets much higher throughput.
