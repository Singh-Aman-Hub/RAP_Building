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

    # ---- B. chunked prefill: long prompts stall everyone's decode unless they are split ------------------------------
    print("B) chunked prefill (continuous scheduler, long 3000-token prompts mixed in)")
    rng = np.random.default_rng(1)
    long_trace = make_trace(200, 4.0, 1)
    for r in long_trace[::10]:
        r["prompt"] = list(rng.integers(0, 50000, 3000))
    for label, budget in (("no chunking", None), ("chunk budget 256 tok/iter", 256)):
        r, t, used, total, st = simulate_continuous(long_trace, chunk_budget=budget)
        print(f"  {label:<26} worst inter-token gap {st['itl_max']:7.0f} ms | p95 gap {st['itl_p95']:5.1f} ms | mean TTFT {np.mean([x['first'] - x['arrival'] for x in r]) / 1000:.2f}s")

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

# TRY IT: RATE=6 python main.py : at higher load static batching saturates first (queue grows without bound), continuous degrades gracefully.
# TRY IT: change BlockCache(16) capacities in part C: when the cache is smaller than the working set, hit rate collapses (LRU thrash).
# TRY IT: change MAX_BATCH to 4 or 64 and see how slot utilisation and tok/s respond.
# BREAK IT: in BlockCache._hashes drop the `h` from hash((h, ...)): blocks are no longer chained, so two prompts with the
#           same 2nd block but different 1st block would wrongly "hit" -> wrong KV reused (a real correctness bug).
# BREAK IT: in part D remove copy.deepcopy: the second call finds a cache already extended by the first -> wrong logits / shape growth.
