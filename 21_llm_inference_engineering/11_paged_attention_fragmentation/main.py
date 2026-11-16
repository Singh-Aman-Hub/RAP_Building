"""
GOAL:  See WHY GPU memory fragments when every request reserves its own contiguous KV-cache slab, and how
       PagedAttention (fixed-size blocks + a block table) fixes it.  Pure Python + numpy simulation: no GPU,
       no model, no network.  Same budget, same requests, two allocators.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 06_llm_training_and_fast_inference/11_paged_attention_fragmentation/main.py
EXPECTED OUTPUT (shape of it, numbers come from the seeded run):
    A) static snapshot: how many requests fit in the same KV budget
    B) dynamic run: 300 requests arrive, memory utilisation + concurrency + total steps for each allocator
    C) block table of one request (logical block -> physical block)
NEEDS: numpy only.  Runs in a few seconds.

Builds on: 08_gpu_memory_calculator (KV cache bytes per token) and 10_vllm_reference (the idea in words).
"""

import numpy as np

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------------------------------------
# Budget.  Llama-7B style KV cache in fp16: 2 (K and V) * 32 layers * 32 heads * 128 dim * 2 bytes
# = 524,288 bytes = 0.5 MiB for EVERY token of EVERY request (formula from step 08).
# ---------------------------------------------------------------------------------------------------------
BYTES_PER_TOKEN = 2 * 32 * 32 * 128 * 2
KV_BUDGET_GB = 8                                           # GPU memory left for KV cache after the weights
BUDGET_TOKENS = int(KV_BUDGET_GB * 2**30 // BYTES_PER_TOKEN)   # 16384 token slots in total
MAX_LEN = 2048                                             # model's maximum sequence length
BLOCK_SIZE = 16                                            # tokens per page (vLLM default is 16)
N_BLOCKS = BUDGET_TOKENS // BLOCK_SIZE                     # 1024 physical blocks

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


def ceil_div(a, b):
    return -(-a // b)


# =========================================================================================================
# A) Static snapshot: admit requests one by one until the next one does not fit.
# =========================================================================================================
print("\nA) STATIC SNAPSHOT: how many requests fit in the same budget?")

# Contiguous: each request reserves MAX_LEN slots no matter how short it will be.
fit_contiguous = BUDGET_TOKENS // MAX_LEN
used_contiguous = total_len[:fit_contiguous].sum()
print(f"  contiguous (reserve {MAX_LEN}): {fit_contiguous:3d} requests | tokens really used {used_contiguous:5d} of "
      f"{fit_contiguous * MAX_LEN} reserved -> {100 * (1 - used_contiguous / (fit_contiguous * MAX_LEN)):.0f}% of the reservation wasted")

# Paged: a request owns ceil(len / BLOCK_SIZE) blocks; only its LAST block can be partly empty.
blocks_needed = ceil_div(total_len, BLOCK_SIZE)            # shape: (N_REQ,)
cum_blocks = np.cumsum(blocks_needed)                      # shape: (N_REQ,)  blocks used if we admit the first i+1 requests
fit_paged = int(np.searchsorted(cum_blocks, N_BLOCKS, side="right"))
used_paged = total_len[:fit_paged].sum()
reserved_paged = int(cum_blocks[fit_paged - 1]) * BLOCK_SIZE
print(f"  paged (block {BLOCK_SIZE}):         {fit_paged:3d} requests | tokens really used {used_paged:5d} of "
      f"{reserved_paged} allocated -> {100 * (1 - used_paged / reserved_paged):.1f}% wasted (last-block slack only)")
print(f"  -> paging fits {fit_paged / fit_contiguous:.1f}x more requests in the same memory")


# =========================================================================================================
# B) Dynamic run.  All requests wait in a queue.  Each step every running request produces one token.
#    A request finishes when it has produced output_len tokens, and gives its memory back.
#    Contiguous server reserves a slab of size MAX_LEN (first-fit in a free list of (start, length) holes).
#    To show EXTERNAL fragmentation too, we also run a "smart" contiguous server that reserves
#    prompt_len + declared_max_new (the user's max_tokens: 128, 256, 512, 1024) -> slabs of varied size.
# =========================================================================================================
declared_max_new = rng.choice([128, 256, 512, 1024], size=N_REQ)                                # shape: (N_REQ,)
declared_max_new = np.maximum(declared_max_new, output_len)         # the user's cap is never below what is produced
reserve_smart = np.minimum(prompt_len + declared_max_new, MAX_LEN)                               # shape: (N_REQ,)


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

    def free_tokens(self):
        return sum(l for _, l in self.free_list)

    def largest_hole(self):
        return max((l for _, l in self.free_list), default=0)


class PagedAllocator:
    """Physical blocks in a free stack + one block table (list of physical block ids) per request."""

    def __init__(self, n_blocks):
        self.free_blocks = list(range(n_blocks - 1, -1, -1))   # stack of free physical block ids
        self.block_table = {}                                  # request id -> [physical block id, ...]

    def alloc_tokens(self, rid, n_tokens):
        need = ceil_div(n_tokens, BLOCK_SIZE) - len(self.block_table.get(rid, []))
        if need > len(self.free_blocks):
            return False
        table = self.block_table.setdefault(rid, [])
        for _ in range(need):
            table.append(self.free_blocks.pop())
        return True

    def free(self, rid):
        self.free_blocks.extend(self.block_table.pop(rid))


def run(policy):
    """Simulate until every request has finished.  Returns metrics."""
    contiguous = policy in ("contiguous_max", "contiguous_declared")
    alloc = ContiguousAllocator(BUDGET_TOKENS) if contiguous else PagedAllocator(N_BLOCKS)
    queue = list(range(N_REQ))                             # waiting request ids, FIFO
    running = {}                                           # rid -> tokens produced so far (prompt counts as present)
    step = 0
    util_samples, conc_samples, ext_samples, preempted = [], [], [], 0
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
        # ---- measure memory utilisation: tokens that really hold KV / total budget ----
        used_tokens = sum(int(prompt_len[r]) + running[r] for r in running)
        util_samples.append(used_tokens / BUDGET_TOKENS)
        conc_samples.append(len(running))
        if contiguous and queue:                           # external fragmentation: free memory the next request cannot use
            need = MAX_LEN if policy == "contiguous_max" else int(reserve_smart[queue[0]])
            if alloc.free_tokens() >= need > alloc.largest_hole():
                ext_samples.append(alloc.free_tokens() - alloc.largest_hole())
        step += 1
        if step > 200000:
            raise RuntimeError("simulation did not finish")
    return dict(steps=step, util=float(np.mean(util_samples)), conc=float(np.mean(conc_samples)),
                peak=int(np.max(conc_samples)), preempted=preempted, ext=ext_samples)


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


# =========================================================================================================
# C) What a block table looks like.  Logical block i of the sequence lives in physical block table[i].
# =========================================================================================================
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

# TRY IT : set BLOCK_SIZE = 128 and re-run A) - the last-block waste grows; vLLM picks small blocks (16) for this reason.
# TRY IT : set KV_BUDGET_GB = 24 - every allocator fits more, but the ratio between them stays about the same.
# BREAK IT: in PagedAllocator.free, replace extend(...) with a no-op (a leaked block table); the paged run slows down,
#           then stalls with "RuntimeError: simulation did not finish" because blocks are never returned.
