# Terms — Continuous batching and prefix caching

Folder: `21_llm_inference_engineering/12_continuous_batching_prefix_caching` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [21 README](../README.md)

**Terms in this folder:** [Continuous batching (iteration-level scheduling, in-flight batching)](#continuous-batching-iteration-level-scheduling-in-flight-batching) · [Static batching](#static-batching) · [Dynamic batching (max wait, max batch)](#dynamic-batching-max-wait-max-batch) · [Slot utilisation](#slot-utilisation) · [Poisson arrivals](#poisson-arrivals) · [Chunked prefill (token budget)](#chunked-prefill-token-budget) · [Automatic prefix caching (APC, prompt cache)](#automatic-prefix-caching-apc-prompt-cache) · [Chained block hash](#chained-block-hash) · [Radix tree (RadixAttention)](#radix-tree-radixattention) · [LRU eviction](#lru-eviction) · [TTFT (time to first token)](#ttft-time-to-first-token) · [Discrete-event simulation](#discrete-event-simulation) · [OrderedDict move_to_end / popitem](#ordereddict-move_to_end--popitem) · [copy.deepcopy of a KV cache](#copydeepcopy-of-a-kv-cache) · [past_key_values reuse](#past_key_values-reuse)

---

## Continuous batching (iteration-level scheduling, in-flight batching)  `(concept)`
**1. Definition:** Scheduling at every decode iteration: finished requests leave immediately and waiting ones take free slots.
**2. In plain words:** A restaurant seating guests as soon as any table frees, not once the whole room has finished.
**3. Example to quote:** > "Continuous batching cut mean TTFT from 4.0 s to 0.06 s in my simulation."
**Seen in:** `main.py`, `simulate_continuous`

## Static batching  `(concept)`
**1. Definition:** Batch formed once and held until its longest request finishes.
**2. In plain words:** A bus that leaves only when the slowest passenger arrives home.
**3. Example to quote:** > "Static batching wastes slots on short requests."
**Seen in:** `main.py`, `simulate_batch_then_run`

## Dynamic batching (max wait, max batch)  `(concept)`
**1. Definition:** Wait until the batch is full or a timeout expires, then run it statically.
**2. In plain words:** The bus leaves when full or after 5 minutes.
**3. Example to quote:** > "Dynamic batching does not fix idle slots in autoregressive decoding."
**Seen in:** `main.py`, `max_wait_ms`

## Slot utilisation  `(concept)`
**1. Definition:** Useful decode tokens divided by (decode iterations x MAX_BATCH).
**2. In plain words:** How full the batch really is.
**3. Example to quote:** > "Utilisation 29 percent static vs 42 percent continuous at 4 req/s."
**Seen in:** `main.py`, `report`

## Poisson arrivals  `(concept)`
**1. Definition:** Requests arrive randomly with exponential gaps at an average rate.
**2. In plain words:** Customers walking in at random.
**3. Example to quote:** > "`rng.exponential(1000 / rate)` then `cumsum`."
**Seen in:** `main.py`, `make_trace`

## Chunked prefill (token budget)  `(concept)`
**1. Definition:** Limiting prompt tokens processed per scheduler iteration so decoders are not stalled.
**2. In plain words:** Cut the big job into slices.
**3. Example to quote:** > "Worst inter-token gap fell from 500 ms to 64 ms."
**Seen in:** `main.py`, `chunk_budget`

## Automatic prefix caching (APC, prompt cache)  `(concept)`
**1. Definition:** Reusing KV blocks of identical token prefixes across requests, keyed by hash.
**2. In plain words:** Do not redo the shared intro.
**3. Example to quote:** > "Hit rate 85 percent and TTFT 4x lower with a large cache."
**Seen in:** `main.py`, `BlockCache`

## Chained block hash  `(concept)`
**1. Definition:** Hash of (previous block hash, this block's tokens), so a hit means the whole prefix matches.
**2. In plain words:** A fingerprint that includes everything before it.
**3. Example to quote:** > "Dropping the chain would let different prefixes share KV: a correctness bug."
**Seen in:** `main.py`, `BlockCache._hashes`

## Radix tree (RadixAttention)  `(concept)`
**1. Definition:** A tree over tokens storing cached prefixes; lookup finds the longest match at token granularity.
**2. In plain words:** A phone-book tree of prompts.
**3. Example to quote:** > "The trie matched 500 tokens; block hashing matched 496."
**Seen in:** `main.py`, `Trie`

## LRU eviction  `(concept)`
**1. Definition:** Evict the cache entry that was used longest ago.
**2. In plain words:** Throw out the least recently touched item.
**3. Example to quote:** > "A cache smaller than the working set thrashes: hit rate 0 percent."
**Seen in:** `main.py`, `OrderedDict`

## TTFT (time to first token)  `(concept)`
**1. Definition:** Time from arrival to the first output token, including queueing and prefill.
**2. In plain words:** How long until the answer starts.
**3. Example to quote:** > "Prefix caching reduces TTFT through less prefill."
**Seen in:** `main.py`, `report`

## Discrete-event simulation  `(concept)`
**1. Definition:** Advance a clock event by event (arrival, iteration end) instead of in real time.
**2. In plain words:** A fast-forwarded stopwatch.
**3. Example to quote:** > "The scheduler loop jumps to the next arrival when idle."
**Seen in:** `main.py`, simulators

## OrderedDict move_to_end / popitem  `(function)`
**1. Definition:** Move a key to the most recent end; pop from the oldest end.
**2. In plain words:** LRU in two calls.
**3. Example to quote:** > "`self.lru.popitem(last=False)` evicts the oldest."
**Seen in:** `main.py`, `BlockCache`

## copy.deepcopy of a KV cache  `(function)`
**1. Definition:** Make an independent copy of the cache object.
**2. In plain words:** Photocopy before writing on it.
**3. Example to quote:** > "Needed because the model extends the cache in place."
**Seen in:** `main.py`, part D

## past_key_values reuse  `(concept)`
**1. Definition:** Passing a previously computed KV cache so only new tokens are processed.
**2. In plain words:** Resume from a saved state.
**3. Example to quote:** > "Prefill 7 tokens instead of 154 with identical logits."
**Seen in:** `main.py`, part D
