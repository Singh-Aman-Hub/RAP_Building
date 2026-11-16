# Step 12 — Continuous batching and prefix caching

## The story so far
[Step 11](../11_paged_attention_fragmentation/WALKTHROUGH.md) fixed the memory side: with paged KV blocks, many requests fit on one GPU at once. [Step 10](../10_latency_throughput_benchmark/WALKTHROUGH.md) showed that throughput climbs with batch size. Both assumed somebody keeps that batch full and moving. That somebody is the **scheduler**, and a naive one throws away most of the gain.

## The pain point
Classic ("static") batching collects N requests, runs them together, and returns when the *last* one finishes. Output lengths vary widely (3 tokens to 400 here), so most slots sit idle waiting for the longest request, and newly arrived requests queue until the whole batch drains. Separately, many requests start with the *same* long prefix (system prompt, few-shot examples, a shared document), and the server prefills those identical tokens again and again.

## The spark
Two scheduling ideas:
1. **Continuous batching** (iteration-level scheduling, in-flight batching): decide *at every decode iteration* who runs. A request that finishes leaves immediately and a waiting one takes its slot in the next iteration.
2. **Prefix caching**: the KV cache of a token depends only on the tokens before it. So if two prompts share their first 480 tokens, the KV for those 480 tokens is identical and can be computed once, stored in blocks keyed by a hash of the tokens, and reused.

## Goal
In a pure-Python discrete-event simulator, run static, dynamic and continuous batching on the same Poisson trace; add chunked prefill; add a chained-hash block cache with LRU eviction; then confirm prefix reuse with a real model by reusing `past_key_values` in distilgpt2.

## The theory (read this before the interview)

**Static batching.**
```
slot 1: [██████████████████████████████████████]  long request
slot 2: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  short request: finished, slot IDLE
slot 3: [██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  short request: IDLE
        └── batch ends only when the longest request ends; new arrivals wait ──┘
```
**Dynamic batching** (Triton-style: `max_batch_size` plus `max_queue_delay`) only changes *when the batch is formed*: wait until it is full or a max wait expires. It trades a little latency for bigger batches, but the batch is still locked until its longest member finishes, so it does not fix the idle-slot problem for autoregressive generation.

**Continuous batching** (Orca, 2022; the basis of vLLM, SGLang, TensorRT-LLM in-flight batching, TGI):
```
iteration:  1  2  3  4  5  6  7  8
slot 1:     A  A  A  A  A  A  A  A
slot 2:     B  B  B  C  C  C  C  C     <- B finished after 3 iterations; C takes the slot at iteration 4
slot 3:     D  D  E  E  E  E  E  F
```
The scheduler loop each iteration: admit waiting requests while slots (and KV blocks) are free; run prefill for new ones; run one decode token for every running sequence; retire finished ones. Variable output lengths no longer waste slots. It works because decode steps are cheap and flat in batch size ([step 09](../09_prefill_decode_roofline/WALKTHROUGH.md)), and paged KV memory ([step 11](../11_paged_attention_fragmentation/WALKTHROUGH.md)) lets sequences come and go at any time.

**Metrics reported by the simulator.** Throughput = output tokens per second. Latency = finish minus arrival (mean, p95). TTFT = first token minus arrival (includes queueing). Slot utilisation = useful decode tokens / (decode iterations x MAX_BATCH) in this toy model.

**Chunked prefill in the scheduler.** A 3,000-token prompt prefilled in one iteration makes that iteration take hundreds of ms and every running request sees one huge inter-token gap. Cap the prompt tokens processed per iteration (a "token budget"); the long prompt finishes over several iterations while decoders keep producing a token each iteration. Cost: that one request's TTFT rises. Gain: bounded ITL for everyone else.

**Prefix caching (automatic prefix caching in vLLM).**
- Cut the prompt into blocks of 16 tokens (same blocks as PagedAttention, step 11). Only *full* blocks are cacheable.
- Key each block by a **chained hash**: `h_j = hash(h_(j-1), tokens of block j)`. Chaining matters: a block's KV depends on everything before it, so the key must too. A hit on block j therefore implies the whole prefix up to j matches.
- On a new request: walk its blocks, count leading hits, skip prefill for those tokens, prefill only the rest. Real systems also hash in extras that change the KV (LoRA adapter id, multimodal inputs, cache salt).
- Blocks have reference counts; blocks not used by any running request go to an **LRU** (least recently used) free list and are evicted when memory is needed. The simulator only models the LRU.
- **Radix tree (SGLang RadixAttention):** store cached prefixes in a tree over tokens (edges are token runs); lookup finds the longest matching prefix at *token* granularity (block hashing only matches multiples of 16), and shared branches are stored once. LRU eviction removes leaves first.
- Wins when many requests share a prefix: chat with a fixed system prompt, few-shot prompts, RAG with a fixed document, agents repeating long histories. No gain on all-unique prompts. Output is *identical* (it is exactly the same KV).

**Why it matters for latency:** prefill cost scales with the number of *uncached* prompt tokens, so TTFT drops roughly in proportion to the hit rate for long prefixes.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Continuous batching | iteration-level scheduling, in-flight batching, rolling batch, Orca-style | Yes | Requests join and leave the batch at every decode step. |
| Static batching | request-level batching, batch-and-wait | Yes | Batch fixed until its longest member ends. |
| Dynamic batching | micro-batching with max delay (Triton) | Related | Chooses when to *form* a batch; still static afterwards. |
| Chunked prefill | split-fuse, prompt chunking | Near-synonyms | Prompt processed in token-budget slices mixed with decode. |
| Prefix caching | automatic prefix caching (APC), prompt caching, KV reuse | Related | Reusing KV blocks of an identical token prefix. Vendor "prompt caching" APIs apply the same idea to billing. |
| RadixAttention | radix tree prefix cache | Specific implementation | SGLang's token-level radix tree over cached prefixes. |
| LRU eviction | least recently used | Yes | Drop the cached block unused for longest. |
| TTFT vs queueing delay | wait time | Different | TTFT = queueing + prefill + first token. |
| Slot / sequence slot | running sequence | Yes | One concurrently running request's KV space. |

## Run it
```bash
cd 21_llm_inference_engineering/12_continuous_batching_prefix_caching
python main.py
# from repo root (part D loads distilgpt2; use the heavy-run lock if other jobs are running):
HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/12_continuous_batching_prefix_caching/main.py
```
About 15 seconds on CPU. Parts A to C are a pure-Python simulation with a toy cost model (decode step = 20 ms + 0.4 ms per sequence, prefill = 0.15 ms per token): the *shapes* are the lesson, not the absolute milliseconds. Part D is real.

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–22)

```python
"""
GOAL:  (A) Simulate three schedulers on the SAME Poisson request trace: static batching, dynamic batching
           (max batch / max wait) and continuous (iteration-level) batching -> throughput, latency, TTFT, slot utilisation.
       (B) Add chunked prefill to the continuous scheduler and watch the worst decode stall shrink.
       (C) Add prefix caching: chained block hashes (block size 16, vLLM style) + LRU eviction -> hit rate, prefill
           tokens saved, TTFT gain.  A tiny token trie shows the radix-tree (SGLang RadixAttention) idea.
       (D) REAL demo: reuse the KV cache (past_key_values) of a shared prefix in distilgpt2 -> same logits, less prefill.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 21_llm_inference_engineering/12_continuous_batching_prefix_caching/main.py
EXPECTED OUTPUT (shortened):
    scheduler            tok/s  mean lat  p95 lat  mean TTFT  slot util
    static                ...
    dynamic               ...
    continuous            ...  (best throughput and TTFT)
    chunked prefill: worst decode stall drops from X ms to Y ms
    prefix cache: hit rate ~..%, prefill tokens saved ..%, mean TTFT ... -> ...
    REAL: logits identical True | full prefill .. ms vs suffix-only .. ms
NEEDS: numpy, torch, transformers (cached distilgpt2, no downloads).  ~15 s on CPU (parts A-C are pure Python/numpy).
       Part D loads a causal LM: run through the global lock.
Builds on: 11_paged_attention_fragmentation (block size 16), 09_prefill_decode_roofline (prefill vs decode cost).
"""
```

- **In simple words:** What the four parts do.
- **Exactly:** Docstring in the required format.
- **Why it is here:** Orientation.

### 2. Imports, knobs and the cost model  (`main.py` lines 23–50)

```python
import copy
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchlib import percentile                                  # shared helper (step 09/10)

SEED = int(os.environ.get("SEED", 0))
RATE = float(os.environ.get("RATE", 2.0))                        # requests per second (Poisson arrivals)
N_REQ = int(os.environ.get("N_REQ", 400))
MAX_BATCH = 16                                                   # number of KV "slots" the GPU can hold
# Toy cost model in milliseconds (shape follows step 09: a decode step is memory-bound, so it is ~flat in batch size;
# prefill is compute-bound, so it costs per prompt token).
DECODE_BASE_MS, DECODE_PER_SEQ_MS, PREFILL_PER_TOKEN_MS = 20.0, 0.4, 0.15


def step_ms(n_decoding, prefill_tokens):
    """Time of ONE scheduler iteration: weights are read once for the whole batch + the prefill tokens computed."""
    base = DECODE_BASE_MS + DECODE_PER_SEQ_MS * n_decoding if (n_decoding or prefill_tokens) else 0.0
    return base + PREFILL_PER_TOKEN_MS * prefill_tokens


```

- **In simple words:** Load libraries and fix a toy cost model.
- **Exactly:** `SEED`, `RATE`, `N_REQ` come from the environment. `MAX_BATCH = 16` slots. `step_ms(n_decoding, prefill_tokens)` = a base decode cost (weights read once, so nearly flat in batch, step 09) + a small per-sequence cost + a per-prompt-token prefill cost.
- **Why it is here:** One function defines time for every scheduler, so the comparison is fair.

### 3. The workload and the report  (`main.py` lines 51–75)

```python
# ---- 1. The workload: Poisson arrivals, varied prompt and output lengths ------------------------------------------------
def make_trace(n=N_REQ, rate=RATE, seed=SEED, shared_prefixes=0):
    """Returns a list of dicts. Poisson process = exponential gaps. Output lengths are heavy-tailed (lognormal): the
    reason static batching hurts. With shared_prefixes>0 each prompt = one of K long system prompts + a unique user part."""
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1000.0 / rate, n))      # ms
    outs = np.clip(rng.lognormal(4.0, 0.9, n).astype(int), 4, 400)
    reqs = []
    sys_prompts = [list(rng.integers(0, 50000, 480)) for _ in range(shared_prefixes)]
    for i in range(n):
        user = list(rng.integers(0, 50000, int(rng.integers(30, 150))))
        prompt = (sys_prompts[int(rng.integers(shared_prefixes))] + user) if shared_prefixes else user + list(rng.integers(0, 50000, 100))
        reqs.append(dict(id=i, arrival=float(arrivals[i]), prompt=prompt, out=int(outs[i])))
    return reqs


def report(name, reqs, t_end, slot_used, slot_total, extra=""):
    lat = [r["finish"] - r["arrival"] for r in reqs]
    ttft = [r["first"] - r["arrival"] for r in reqs]
    toks = sum(r["out"] for r in reqs)
    row = dict(tok_s=toks / (t_end / 1000), mean_lat=np.mean(lat) / 1000, p95=percentile(lat, 95) / 1000,
               ttft=np.mean(ttft) / 1000, util=slot_used / slot_total)
    print(f"{name:<22}{row['tok_s']:>8.0f}{row['mean_lat']:>10.1f}s{row['p95']:>8.1f}s{row['ttft']:>10.2f}s{100 * row['util']:>9.0f}%{extra}")
    return row

```

- **In simple words:** Make Poisson arrivals with varied lengths; print one result row.
- **Exactly:** `rng.exponential` gaps accumulated by `cumsum` give a Poisson process; lognormal output lengths are heavy-tailed (clipped to 4-400). With `shared_prefixes=K`, each prompt = one of K 480-token system prompts + a unique 30-150 token user part. `report` computes tok/s, mean/p95 latency, mean TTFT and slot utilisation.
- **Why it is here:** Same trace for all schedulers.

### 4. Static and dynamic batching  (`main.py` lines 77–107)

```python
# ---- 2. Static and dynamic batching: the batch is formed once and runs until its LONGEST request finishes ------------
def simulate_batch_then_run(trace, max_wait_ms=None):
    """max_wait_ms=None -> static (take whatever is waiting when the GPU frees up, up to MAX_BATCH).
    max_wait_ms=X       -> dynamic batching: wait until the batch is full or X ms passed since the oldest waiting request."""
    reqs = [dict(r) for r in trace]
    t, i, queue, used, total = 0.0, 0, [], 0, 0
    while i < len(reqs) or queue:
        if not queue:
            t = max(t, reqs[i]["arrival"])                       # idle: jump to the next arrival
        while i < len(reqs) and reqs[i]["arrival"] <= t:
            queue.append(reqs[i]); i += 1
        if max_wait_ms is not None:                              # dynamic: hold the door open a little longer
            deadline = queue[0]["arrival"] + max_wait_ms
            while len(queue) < MAX_BATCH and i < len(reqs) and reqs[i]["arrival"] <= deadline:
                t = max(t, reqs[i]["arrival"]); queue.append(reqs[i]); i += 1
            if len(queue) < MAX_BATCH:
                t = max(t, deadline)
        batch, queue = queue[:MAX_BATCH], queue[MAX_BATCH:]
        t += step_ms(0, sum(len(r["prompt"]) for r in batch))   # prefill the whole batch first (all prompts)
        for r in batch:
            r["first"] = t
        steps = max(r["out"] for r in batch)                     # EVERY slot is held until the longest request ends
        for s in range(steps):
            alive = [r for r in batch if r["out"] > s]
            t += step_ms(len(batch), 0)                          # finished slots are still padded/occupied (static)
            for r in alive:
                if r["out"] == s + 1:
                    r["finish"] = t
        used += sum(r["out"] for r in batch); total += steps * MAX_BATCH
    return reqs, t, used, total

```

- **In simple words:** Form a batch, run it until its longest request ends.
- **Exactly:** `max_wait_ms=None` = static (take whatever is waiting, up to 16). With a value = dynamic: wait for a full batch or until the oldest waiting request is `max_wait_ms` old. Prefill is done for the whole batch, then `steps = max(out)` decode steps, each costing `step_ms(len(batch), 0)`: finished slots stay occupied. Utilisation = useful tokens / (steps x MAX_BATCH).
- **Why it is here:** The baseline that wastes slots.

### 5. Block cache with chained hashes and LRU  (`main.py` lines 109–141)

```python
# ---- 3. Continuous (iteration-level) batching, optional chunked prefill, optional prefix cache --------------------------
class BlockCache:
    """Prefix cache: a block is identified by a CHAINED hash = hash(previous block's hash, this block's tokens), so a
    hit on block j means the WHOLE prefix up to j matches (what vLLM's automatic prefix caching does). LRU eviction."""
    def __init__(self, capacity_blocks, block=16):
        self.cap, self.block, self.lru = capacity_blocks, block, OrderedDict()
        self.hits = self.lookups = 0

    def _hashes(self, tokens):
        h, out = 0, []
        for j in range(len(tokens) // self.block):               # ONLY full blocks are cacheable
            h = hash((h, tuple(tokens[j * self.block:(j + 1) * self.block])))
            out.append(h)
        return out

    def match(self, tokens):
        """Number of leading tokens already cached (multiple of block). Touches hits -> most recently used."""
        n = 0
        hs = self._hashes(tokens)
        for h in hs:
            if h not in self.lru:
                break
            self.lru.move_to_end(h); n += 1
        self.hits += n; self.lookups += len(hs)
        return n * self.block

    def insert(self, tokens):
        for h in self._hashes(tokens):
            self.lru[h] = True; self.lru.move_to_end(h)
            while len(self.lru) > self.cap:
                self.lru.popitem(last=False)                     # evict least recently used block


```

- **In simple words:** A prefix cache keyed by chained hash of 16-token blocks.
- **Exactly:** `_hashes`: for each full block, `h = hash((previous h, tokens))`. `match` counts leading blocks present (moving them to the MRU end) and returns the cached token count; `insert` adds all full blocks and pops the oldest when over capacity (`popitem(last=False)`). Hits and lookups are counted for the hit rate. Real vLLM also reference-counts blocks; this models only the LRU.
- **Why it is here:** The mechanism of automatic prefix caching.

### 6. Continuous batching scheduler with chunked prefill and cache  (`main.py` lines 142–185)

```python
def simulate_continuous(trace, chunk_budget=None, cache=None):
    """Every iteration: (1) admit waiting requests into free slots, (2) spend up to `chunk_budget` prompt tokens on
    prefill (None = unlimited, whole prompts at once), (3) every fully-prefilled request decodes one token, (4) finished
    requests free their slot IMMEDIATELY. Returns also the list of decode gaps seen by running requests (ITL)."""
    reqs = [dict(r) for r in trace]
    t, i, waiting, running, used, iters = 0.0, 0, [], [], 0, 0
    saved = prompt_total = 0
    gaps, last_tok = [], {}
    while i < len(reqs) or waiting or running:
        if not waiting and not running:
            t = max(t, reqs[i]["arrival"])
        while i < len(reqs) and reqs[i]["arrival"] <= t:
            waiting.append(reqs[i]); i += 1
        while waiting and len(running) < MAX_BATCH:              # admit
            r = waiting.pop(0)
            cached = cache.match(r["prompt"]) if cache else 0
            r["todo"] = max(len(r["prompt"]) - cached, 1)        # prompt tokens still to prefill
            r["done_tok"] = 0
            saved += len(r["prompt"]) - r["todo"]; prompt_total += len(r["prompt"])
            running.append(r)
        budget = chunk_budget if chunk_budget else 10 ** 9
        pre_tokens, decoders = 0, []
        for r in running:                                        # prefill (possibly chunked)
            if r["todo"] > 0:
                take = min(r["todo"], budget - pre_tokens)
                r["todo"] -= take; pre_tokens += take
        decoders = [r for r in running if r["todo"] == 0 and "first" in r]
        dt = step_ms(len(decoders), pre_tokens)
        t += dt
        for r in running:                                        # a request whose prefill just ended gets its 1st token now
            if r["todo"] == 0 and "first" not in r:
                r["first"] = t; r["done_tok"] = 1; last_tok[r["id"]] = t
                if cache:
                    cache.insert(r["prompt"])
        for r in decoders:
            r["done_tok"] += 1; used += 1
            gaps.append(t - last_tok[r["id"]]); last_tok[r["id"]] = t
        iters += 1
        for r in [r for r in running if r["done_tok"] >= r["out"] and r["todo"] == 0 and "first" in r]:
            r["finish"] = t; running.remove(r)                   # slot freed right now
    stats = dict(saved=saved, prompt_total=prompt_total, itl_max=max(gaps), itl_p95=percentile(gaps, 95))
    return reqs, t, sum(r["out"] for r in reqs), iters * MAX_BATCH, stats


```

- **In simple words:** The iteration-level loop.
- **Exactly:** Each iteration: arrivals join the waiting queue; free slots admit requests (cache lookup reduces `todo`, the prompt tokens still to prefill); up to `chunk_budget` prompt tokens are prefilled; every fully prefilled request decodes one token; the iteration costs `step_ms(decoders, prefill_tokens)`; a request gets its first token when its prefill completes (and its blocks are inserted in the cache); finished requests leave at once. `gaps` records inter-token gaps (ITL) of decoders.
- **Why it is here:** The heart of the step.

### 7. Experiment A: three schedulers at two loads  (`main.py` lines 186–198)

```python
if __name__ == "__main__":
    for rate in (RATE, 2 * RATE):                                # light load, then load above static batching's capacity
        trace = make_trace(rate=rate)
        print(f"A) trace: {len(trace)} requests, Poisson rate {rate}/s, output length mean {np.mean([r['out'] for r in trace]):.0f} "
              f"(min {min(r['out'] for r in trace)}, max {max(r['out'] for r in trace)}), MAX_BATCH={MAX_BATCH}")
        print(f"{'scheduler':<22}{'tok/s':>8}{'mean lat':>11}{'p95 lat':>9}{'mean TTFT':>11}{'slot util':>10}")
        for name, kw in (("static", {}), ("dynamic wait=50ms", dict(max_wait_ms=50)), ("dynamic wait=500ms", dict(max_wait_ms=500))):
            r, t, used, total = simulate_batch_then_run(trace, **kw)
            report(name, r, t, used, total)
        r, t, used, total, st = simulate_continuous(trace)
        report("continuous", r, t, used, total)
        print()

```

- **In simple words:** Print the comparison table at rate and 2x rate.
- **Exactly:** Same `trace` goes to static, dynamic (50 ms and 500 ms wait) and continuous.
- **Why it is here:** Shows equal throughput at low load but very different TTFT, and a throughput gap at high load.

### 8. Experiment B: chunked prefill  (`main.py` lines 199–208)

```python
    # ---- B. chunked prefill: long prompts stall everyone's decode unless they are split ------------------------------
    print("B) chunked prefill (continuous scheduler, long 3000-token prompts mixed in)")
    rng = np.random.default_rng(1)
    long_trace = make_trace(200, 4.0, 1)
    for r in long_trace[::10]:
        r["prompt"] = list(rng.integers(0, 50000, 3000))
    for label, budget in (("no chunking", None), ("chunk budget 256 tok/iter", 256)):
        r, t, used, total, st = simulate_continuous(long_trace, chunk_budget=budget)
        print(f"  {label:<26} worst inter-token gap {st['itl_max']:7.0f} ms | p95 gap {st['itl_p95']:5.1f} ms | mean TTFT {np.mean([x['first'] - x['arrival'] for x in r]) / 1000:.2f}s")

```

- **In simple words:** Insert 3,000-token prompts and compare worst inter-token gap with and without a 256-token budget.
- **Exactly:** Every 10th request gets a 3,000-token prompt. `st['itl_max']` is the largest gap any running request saw.
- **Why it is here:** Quantifies the decode stall.

### 9. Experiment C: prefix cache and the radix idea  (`main.py` lines 209–237)

```python
    # ---- C. prefix caching --------------------------------------------------------------------------------------------
    print("\nC) prefix caching: 400 requests, 4 shared 480-token system prompts + unique 30-150 token user part")
    ptrace = make_trace(400, 3.0, 2, shared_prefixes=4)
    base, t0, *_ = simulate_continuous(ptrace)
    ttft0 = np.mean([x["first"] - x["arrival"] for x in base])
    print(f"  {'no cache':<28} mean TTFT {ttft0:7.0f} ms")
    for cap in (16, 128, 4096):                                  # tiny cache forces LRU eviction; large keeps all 4 prompts (4*30 blocks)
        cache = BlockCache(cap)
        r, t, used, total, st = simulate_continuous(ptrace, cache=cache)
        ttft = np.mean([x["first"] - x["arrival"] for x in r])
        print(f"  cache {cap:>4} blocks (LRU)      hit rate {100 * cache.hits / cache.lookups:4.0f}% | prefill tokens saved {100 * st['saved'] / st['prompt_total']:4.0f}% | mean TTFT {ttft:7.0f} ms ({ttft0 / ttft:.1f}x faster)")

    # radix idea: a trie over TOKENS matches at any token boundary; block hashing only at multiples of 16
    class Trie:
        def __init__(self): self.root = {}
        def insert(self, toks):
            n = self.root
            for x in toks: n = n.setdefault(x, {})
        def match(self, toks):
            n, k = self.root, 0
            for x in toks:
                if x not in n: break
                n, k = n[x], k + 1
            return k
    a = ptrace[0]["prompt"]; b = a[:500] + [7, 7, 7]             # shares exactly 500 tokens
    tr, bc = Trie(), BlockCache(1000)
    tr.insert(a); bc.insert(a)
    print(f"\n  radix/trie match of a prompt sharing 500 tokens: {tr.match(b)} tokens | block-hash match: {bc.match(b)} tokens (floor(500/16)*16)")

```

- **In simple words:** Hit rate, tokens saved and TTFT for three cache sizes; trie vs block-hash match.
- **Exactly:** Capacities 16, 128, 4096 blocks; the working set is 4 prompts x 30 blocks = 120 blocks. The nested `Trie` matches token by token, so a 500-token overlap gives 500 vs 496 for block hashing.
- **Why it is here:** Shows LRU thrash and the granularity difference.

### 10. Experiment D: real prefix reuse with past_key_values  (`main.py` lines 238–267)

```python
    # ---- D. REAL prefix reuse with Hugging Face past_key_values -------------------------------------------------------
    import torch
    import benchlib
    model, tok = benchlib.load_causal_lm("distilgpt2", threads=4)
    prefix = "You are a helpful assistant for a bank. Always answer politely, cite the policy number, and never reveal internal notes. " * 6
    pa, pb = prefix + " Question: how do I reset my password?", prefix + " Question: what is the overdraft fee?"
    ids_a, ids_b = (tok(p, return_tensors="pt").input_ids for p in (pa, pb))
    n_pre = 0
    while n_pre < min(ids_a.shape[1], ids_b.shape[1]) and ids_a[0, n_pre] == ids_b[0, n_pre]:
        n_pre += 1                                               # length of the shared token prefix
    print(f"\nD) REAL: prompts of {ids_a.shape[1]} and {ids_b.shape[1]} tokens share a {n_pre}-token prefix")

    @torch.no_grad()
    def timed(fn, reps=7):
        fn()
        best = 1e9
        for _ in range(reps):
            s = time.perf_counter(); out = fn(); best = min(best, time.perf_counter() - s)
        return out, best * 1000

    with torch.no_grad():
        prefix_out = model(input_ids=ids_a[:, :n_pre], use_cache=True)          # prefill the shared part ONCE
    prefix_cache = prefix_out.past_key_values
    full_logits, t_full = timed(lambda: model(input_ids=ids_b).logits[:, -1])
    def with_cache():
        c = copy.deepcopy(prefix_cache)                          # each request must get its OWN copy (the cache is extended in place)
        return model(input_ids=ids_b[:, n_pre:], past_key_values=c, use_cache=True).logits[:, -1]
    reuse_logits, t_reuse = timed(with_cache)
    print(f"  identical last-token logits: {torch.allclose(full_logits, reuse_logits, atol=1e-4)} (max abs diff {(full_logits - reuse_logits).abs().max():.2e})")
    print(f"  prefill prompt B in full: {t_full:.1f} ms | reuse prefix cache, prefill only {ids_b.shape[1] - n_pre} new tokens: {t_reuse:.1f} ms  ({t_full / t_reuse:.1f}x)")
```

- **In simple words:** Prefill a shared prefix once and reuse its KV cache for a second prompt in distilgpt2.
- **Exactly:** Find the shared token prefix length `n_pre`; `model(ids[:, :n_pre], use_cache=True).past_key_values` is the cache; `copy.deepcopy` gives each request its own copy; `model(ids_b[:, n_pre:], past_key_values=c)` prefills only the new tokens. `torch.allclose` compares the final-position logits with a full prefill; `timed` takes the best of 7 runs after a warm-up.
- **Why it is here:** Proves prefix caching is exact and cheaper on a real model.

### 11. TRY IT / BREAK IT  (`main.py` lines 268–274)

```python

# TRY IT: RATE=6 python main.py : at higher load static batching saturates first (queue grows without bound), continuous degrades gracefully.
# TRY IT: change BlockCache(16) capacities in part C: when the cache is smaller than the working set, hit rate collapses (LRU thrash).
# TRY IT: change MAX_BATCH to 4 or 64 and see how slot utilisation and tok/s respond.
# BREAK IT: in BlockCache._hashes drop the `h` from hash((h, ...)): blocks are no longer chained, so two prompts with the
#           same 2nd block but different 1st block would wrongly "hit" -> wrong KV reused (a real correctness bug).
# BREAK IT: in part D remove copy.deepcopy: the second call finds a cache already extended by the first -> wrong logits / shape growth.
```

- **In simple words:** Experiments in comments.
- **Exactly:** Higher rate; cache size; unchained hash; missing deepcopy.
- **Why it is here:** Practice.

## Syntax explained

| Syntax | Meaning | Tiny example |
|---|---|---|
| `os.environ.get("RATE", 2.0)` | knob with default | `RATE=6 python main.py` |
| `rng.exponential(1000/rate, n)` then `np.cumsum` | Poisson arrival process (exponential gaps, running sum) | arrival times in ms |
| `rng.lognormal(4.0, 0.9, n)` | heavy-tailed output lengths | median about 55, max clipped to 400 |
| `OrderedDict.move_to_end(k)` / `popitem(last=False)` | LRU: touch = move to end; evict = pop the front | |
| `hash((h, tuple(tokens)))` | hash of a tuple; including `h` chains the blocks | |
| `class Trie` nested in `__main__` | tiny helper class defined where it is used | |
| `copy.deepcopy(cache)` | independent copy of the KV cache object | one per request |
| `x[::10]` | every 10th element | long prompts for every 10th request |

## Libraries & functions

| Name | What it does | Internally |
|---|---|---|
| `collections.OrderedDict` | dict that remembers insertion order | doubly linked list plus hash map, so `move_to_end` is O(1) |
| `benchlib.percentile` | linear-interpolation percentile | sorts and interpolates |
| `model(input_ids=..., past_key_values=cache, use_cache=True)` | forward pass that reuses a KV cache | attention uses the cached K,V for earlier positions and computes K,V only for the new tokens, appending them **in place** |
| `torch.allclose(a, b, atol=1e-4)` | numerical equality | elementwise `|a-b| <= atol + rtol*|b|` |

## Execution trace (real run)

Real output (parts A to C are deterministic with `SEED=0`; part D timings vary by machine):

```
A) trace: 400 requests, Poisson rate 2.0/s, output length mean 77 (min 4, max 400), MAX_BATCH=16
scheduler                tok/s   mean lat  p95 lat  mean TTFT slot util
static                     133       5.9s    12.0s      4.04s       21%
dynamic wait=50ms          135       5.7s    11.3s      3.81s       22%
dynamic wait=500ms         133       6.1s    12.0s      4.16s       21%
continuous                 136       1.8s     4.9s      0.06s       20%

A) trace: 400 requests, Poisson rate 4.0/s, output length mean 77 (min 4, max 400), MAX_BATCH=16
scheduler                tok/s   mean lat  p95 lat  mean TTFT slot util
static                     165      36.2s    73.8s     34.13s       29%
dynamic wait=50ms          165      36.2s    73.8s     34.18s       29%
dynamic wait=500ms         162      39.5s    74.3s     37.51s       28%
continuous                 269       2.1s     5.5s      0.07s       42%

B) chunked prefill (continuous scheduler, long 3000-token prompts mixed in)
  no chunking                worst inter-token gap     500 ms | p95 gap  54.1 ms | mean TTFT 0.17s
  chunk budget 256 tok/iter  worst inter-token gap      64 ms | p95 gap  62.4 ms | mean TTFT 0.27s

C) prefix caching: 400 requests, 4 shared 480-token system prompts + unique 30-150 token user part
  no cache                     mean TTFT     196 ms
  cache   16 blocks (LRU)      hit rate    0% | prefill tokens saved    0% | mean TTFT     196 ms (1.0x faster)
  cache  128 blocks (LRU)      hit rate   63% | prefill tokens saved   62% | mean TTFT      76 ms (2.6x faster)
  cache 4096 blocks (LRU)      hit rate   85% | prefill tokens saved   84% | mean TTFT      49 ms (4.0x faster)

  radix/trie match of a prompt sharing 500 tokens: 500 tokens | block-hash match: 496 tokens (floor(500/16)*16)

D) REAL: prompts of 154 and 154 tokens share a 147-token prefix
  identical last-token logits: True (max abs diff 9.16e-05)
  prefill prompt B in full: 56.5 ms | reuse prefix cache, prefill only 7 new tokens: 19.5 ms  (2.9x)
```

Reading the numbers:
- **Light load (2 req/s):** all four schedulers achieve the same throughput (the server is not the bottleneck; output rate equals arrival rate), but static and dynamic batching have 4.0 s mean TTFT against 0.06 s for continuous: new arrivals wait for the running batch to finish. Dynamic batching with a max wait does not help here.
- **Heavy load (4 req/s):** static and dynamic saturate near 165 tok/s and the queue grows (mean latency 36 s, p95 74 s). Continuous keeps up at 269 tok/s with 2.1 s mean latency and 5.5 s p95. Utilisation: 29 percent vs 42 percent.
- **Chunked prefill:** worst inter-token gap falls from 500 ms to 64 ms, paid for with a higher mean TTFT (0.17 s to 0.27 s) because the long prompts are processed more slowly.
- **Prefix cache:** a cache of 16 blocks is smaller than the working set (4 prompts x 30 blocks): LRU thrashes, hit rate 0 percent. 128 blocks holds most prefixes (63 percent hit rate, 2.6x faster TTFT); 4,096 blocks gives 85 percent hits and 4.0x. The hit rate is capped below 100 percent by the first request for each prefix and the unique user text.
- **Radix vs block hash:** for a prompt that shares exactly 500 tokens, the trie matches 500 while block hashing matches 496 (31 full blocks of 16).
- **Real reuse (part D):** logits for the last position agree (max difference at float rounding level) and prefilling only the unshared 7 tokens is about 3x faster than prefilling all 154. At long shared prefixes the gain is larger.

## Experiments (change ONE thing)

| Change | What you will see | Why |
|---|---|---|
| `RATE=6` | static latency explodes first; continuous also saturates later | offered load exceeds capacity |
| `MAX_BATCH = 4` | throughput falls for all; continuous gap to static shrinks | fewer slots, less to waste |
| cache size 16, 128, 4096 | hit rate 0, 63, 85 percent | working set is 4 x 30 blocks |
| `chunk_budget` 64 | smaller worst gap, larger TTFT for long prompts | smaller slice per iteration |
| `shared_prefixes=0` in part C | hit rate near 0 | no shared prefix to reuse |
| remove `copy.deepcopy` in part D | wrong logits or mismatch | the cache is extended in place by the first request |

## Debugging practice
**BREAK IT (simulation):** in `BlockCache._hashes` change `hash((h, tuple(...)))` to `hash(tuple(...))` (drop the chain). Symptom: the hit rate looks *better*, but two prompts whose 2nd block is identical and whose 1st blocks differ now share a key: the wrong KV would be reused, silently producing wrong text in a real system. Fix: always chain the previous block's hash (plus adapter id and similar).

**BREAK IT (real):** drop `copy.deepcopy(prefix_cache)` in part D. The model *extends the cache object in place*, so the second call sees a cache that already holds the first call's suffix: wrong logits (or a shape error in later layers). Fix: copy per request, or use an engine that does copy-on-write on shared blocks (vLLM does, using reference-counted blocks).

## Common mistakes
- Reporting throughput without TTFT: at light load all schedulers have the same throughput and very different latency.
- Thinking dynamic batching equals continuous batching. Dynamic forms batches; continuous re-forms them every token.
- Caching partial blocks, or hashing blocks without chaining.
- Forgetting that prefix caching keys must include everything that changes KV (LoRA adapter, image content, a tenant salt for isolation).
- Assuming prefix caching changes the output. It must not.
- Expecting gains on unique prompts or on tiny prefixes (under one block).

## Rebuild checklist
1. Generate a Poisson trace: exponential gaps, lognormal output lengths.
2. Write the cost model: `step = base + per_seq * n + per_token * prefill_tokens`.
3. Static: take up to B waiting requests, run until the longest finishes.
4. Continuous loop: admit, prefill (with budget), decode one token for everyone, retire finished now.
5. Metrics: tok/s, latency mean and p95, TTFT, utilisation.
6. `BlockCache`: chained hash per full 16-token block, `OrderedDict` LRU, match counts leading hits.
7. Real demo: prefill the shared prefix once, deep-copy its cache, prefill only the suffix, compare logits.

## Say it aloud
- "Static batching holds every slot until the longest request finishes; continuous batching schedules per iteration, so finished requests free their slot immediately and new ones join. Same throughput at low load but far better TTFT, and higher throughput once the load saturates static batching."
- "Chunked prefill puts a token budget on prompt work per iteration so a long prompt cannot freeze everyone's decode; it trades that request's TTFT for bounded inter-token latency."
- "Prefix caching hashes 16-token KV blocks with a chained hash, so a hit implies the whole prefix matches; it skips prefill for shared system prompts and few-shot examples, with identical output."
- "SGLang's RadixAttention stores prefixes in a radix tree for token-level matching and LRU eviction of leaves; vLLM's automatic prefix caching does block-level hashing."
- "If the cache is smaller than the working set of prefixes, LRU thrashes and the hit rate collapses."

## The new problem (cliffhanger)
The scheduler now keeps the GPU full and skips repeated work, but each user still gets one token per forward pass, and decode is memory-bound. Can a single request be made faster without changing its output? [Step 13, speculative decoding](../13_speculative_decoding/WALKTHROUGH.md).
