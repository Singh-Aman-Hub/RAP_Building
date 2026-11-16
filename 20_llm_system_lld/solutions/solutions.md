# Module 20 — Solutions

Runnable code for the `(code)` exercises is in `solutions/sol_NN.py`; each section below shows the **real output** of its script and the prose answers. Exercise text: [../exercises.md](../exercises.md).

## Step 01 — LLD foundations
**1.1** `Shouter` is no longer a `StreamingGenerator` (it lacks `stream`); `StreamingShouter` is. Protocol membership is decided by attribute names, so adding a method to a Protocol silently breaks every implementation that lacks it: version your Protocols carefully (or make a new, larger one, which is interface segregation).
**1.2** The `AttributeError` appears at the first call to `.generate`, far from the mistake. An `isinstance(generator, Generator)` check in the constructor moves the failure to construction. (It checks names only, not signatures.)
**1.3** The Protocol fake needs no inheritance; the ABC version refuses instantiation when a method is missing.
**1.4 Explain aloud.** "A Protocol is structural: any object with the right methods fits, no inheritance, ideal for fakes and vendor adapters. An ABC is nominal: you inherit and must implement the abstract methods, and Python refuses to instantiate an incomplete class; I use it when I also ship shared code. `runtime_checkable` lets `isinstance` work but it checks only that the attribute names exist, not their signatures, so I still rely on a type checker and contract tests."

Output of [`sol_01.py`](sol_01.py):
```text
1.1 Shouter is a Generator: True | Shouter is a StreamingGenerator: False | StreamingShouter is: True
1.2 early error: generator must implement generate(system, user); got str
1.3 FakeEmbedder is an Embedder: True
    ABC version refuses early: Can't instantiate abstract class Incomplete without an implementation
```

## Step 02 — Domain model
**2.1** With content-only ids exactly 1 chunk (the new paragraph) is re-embedded; with the position in the id all 21 are. Position-based ids turn a one-paragraph edit into a full re-index.
**2.2** Assignment raises `FrozenInstanceError`. `dataclasses.replace` copies *all* fields including the old `chunk_id`, so the id no longer matches the content; rebuild the chunk with `Chunk.create(...)` so the id is recomputed. (An id that does not match the content is the classic bug of derived fields in frozen objects.)
**2.3** Round trips pass for an answer with citations and for a refusal.
**2.4 Explain aloud.** "Pydantic at trust boundaries - HTTP bodies, LLM JSON, config - because it validates, coerces and gives good errors and a JSON Schema. Inside the system I use frozen, slotted dataclasses: faster, smaller, and the data is already trusted.

Output of [`sol_02.py`](sol_02.py):
```text
2.1 20 paragraphs, 1 inserted at top: content-only ids re-embed 1; position in id re-embed 21
2.2 frozen: cannot assign to field 'text'
    replace keeps the OLD id: True | Chunk.create gives a NEW id: True
2.3 round trips OK; JSON: {"citations": [{"chunk_id": "abc123", "marker": 1, "supported": true}], "context_chunk_ids ...
```

## Step 03 — Ingestion pipeline
**3.1** With the loaded manifest and the same (still populated) store, all 9 documents are skipped and nothing is embedded. With an empty store the manifest still says "everything is indexed", so all 9 are skipped and **the index stays empty**: manifest and vector store must be persisted and versioned together (or the manifest stored inside the store). A cheap safeguard: on startup compare `store.count()` with the manifest's expected chunk count and force a rebuild on mismatch.
**3.2** `ValueError: need 0 <= overlap < size` at construction. Without it `range(0, n, 0)` raises `ValueError: range() arg 3 must not be zero` later, and with overlap larger than size the window would move backwards.
**3.3** stale = {a}, new = {d}, reused = {b, c}. `run()`: version check (clear + rebuild if changed) -> for each doc skip if hash unchanged -> else diff ids, delete stale, embed new in batches, upsert -> update the manifest only on success -> delete documents that vanished -> stamp the version.
**3.4 Explain aloud.** "The manifest is updated only after a document is fully indexed, so a crash leaves it dirty and the next run retries; operations are re-runnable instead of transactional. The index version is a hash of embedder name and dimension, chunker settings and a schema string; if it changes, stored vectors mean something else and the index must be rebuilt, ideally in a new collection behind an alias.

Output of [`sol_03.py`](sol_03.py):
```text
3.1 loaded manifest + SAME store  -> skipped=9 embedded=0
    loaded manifest + EMPTY store -> skipped=9 embedded=0 (manifest and store must be persisted together, else the index silently stays empty)
3.2 ValueError: need 0 <= overlap < size
3.3 stale = ['a'] | new = ['d'] | reused = ['b', 'c']
```

## Step 04 — Query pipeline
**4.1** Stop-at-first packs 0 chunks when the best chunk is too large, so the model sees an empty context; skip-and-continue packs the two smaller ones.
**4.2** Rank 1 in one list and rank 3 in the other: 1/61 + 1/63 = 0.03227, slightly above a chunk at rank 2 in both (2/62 = 0.03226). RRF rewards consistent agreement and top ranks.
**4.3** `30 days` is a wrong fact, yet the verifier says it is supported: most words overlap the cited chunk. Lexical support catches missing/invalid citations and unrelated claims, not contradictions. For faithfulness use an NLI model or an LLM judge on a sample.
**4.4 Explain aloud.** "I refuse before generation, when there are no hits or the best reranked score is under a threshold, which saves cost and removes a hallucination chance. I calibrate the threshold on labelled answerable and unanswerable questions by sweeping it (step 10, exercise 10.1) and look at false accepts (answering the unanswerable) and false refusals (refusing the answerable).

Output of [`sol_04.py`](sol_04.py):
```text
4.1 stop-at-first packs 0 chunk(s); skip-and-continue packs 2 chunk(s)
4.2 rrf scores: {'x': 0.03227, 'y': 0.03226, 'w': 0.01639, 'z': 0.01587} | X by hand = 1/61 + 1/63 = 0.03227 | Y (rank 2 and 2) = 0.03226
4.3 'Refunds are issued within 9 days. [1]'        ok=True
4.3 'Refunds are issued within 30 days. [1]'       ok=True
4.3 'Nothing relevant here. [1]'                   ok=False
    note: '30 days' is WRONG but still 'supported' by word overlap -> lexical support is not faithfulness
```

## Step 05 — Design patterns
**5.1** `RETRIEVERS.register('refunds_doc_only')` plus a config entry `{'retriever': {'type': 'refunds_doc_only'}}`; `factory.py` is untouched. **5.2** Duplicate -> `ValueError: ... already registered`; unknown -> `UnknownPluginError` listing the valid names. **5.3** See `TinyRegistry` in the solution: a dict, a decorator and a helpful `KeyError`.
**5.4 Explain aloud.** "Strategy: the varying algorithm is a separate injected object (`Retriever`). Template Method: the varying step is an overridden method in a subclass (`BaseChunker._groups`). Decorator: wraps an object of the same interface to add behaviour (`CachingEmbedder`) and the wrapping order matters.

Output of [`sol_05.py`](sol_05.py):
```text
5.1 retrievers: ['dense', 'hybrid', 'refunds_doc_only', 'sparse'] | answer: Refunds at Chai Kitchen are issued within 9 days o ...
5.2 retriever 'refunds_doc_only' already registered
     "unknown retriever 'nope'; available: ['dense', 'hybrid', 'refunds_doc_only', 'sparse']"
5.3 tiny registry: 42
```

## Step 06 — Resilience
**6.1** With no jitter all 1000 clients retry together; full jitter spreads them (peak 130 per slot) and waits 1.99 s on average; equal jitter has a higher peak (256) and a longer mean wait (3.00 s). The AWS Architecture Blog reports full jitter as the best simple choice. **6.2** 4 provider calls for a permanent 401 instead of 1: wasted quota and delay. **6.3** The 8-line bucket reproduces `[True, True, True, True, False]` then two more after a second.
**6.4 Explain aloud.** "Rate limiter outermost (admit), then retry (re-try failures), each attempt through the circuit breaker (fails fast when the provider is down) and a per-attempt timeout; a fallback chain wraps the per-provider stacks. Fallback only on transient or open-circuit errors, because a permanent error (bad key, bad request) would just fail again or behave differently on another model.

Output of [`sol_06.py`](sol_06.py):
```text
6.1 peak retries per 0.5 s slot (1000 clients): none=1000 full=130 equal=256
    mean wait: full=1.99s equal=3.00s (equal jitter waits longer on average: it keeps half of the cap)
6.2 provider calls with retry_on=(Exception,): 4 (should be 1)
    with the default retry_on=(TransientError,): 1
6.3 from scratch: [True, True, True, True, False] then at t=1.0: [True, True, False]
```

## Step 07 — Caching
**7.1** `OrderedDict.move_to_end` on get and put, `popitem(last=False)` on overflow. **7.2** Caching wins from the 2nd call for both prefixes: a write costs 1.25x (a 0.25x premium), each later read saves 0.9x, so one reuse already pays for the premium. (If the cache expires between calls - 5 minutes by default - you pay the write again, so low-traffic prompts may never win.) **7.3** `bool(SemanticCache(...))` is `False` when empty (it defines `__len__`), so `if self.semantic:` never uses or fills it.
**7.4 Explain aloud.** "Embedding similarity measures topical closeness, not equivalence, so 'is shipping free' and 'is shipping not free' score 0.93. Mitigate with a high threshold, the same scope (index version and tenant), verification or restricting to FAQ-like questions, and never caching refusals. Versioned keys put the index version inside the key, so a re-index makes old entries unreachable with no scan; TTL and LRU clean them up.

Output of [`sol_07.py`](sol_07.py):
```text
7.1 LRU keys after put a,b / get a / put c: ['a', 'c']
7.2 prefix   1000 tokens: caching wins from call number 2
7.2 prefix  10000 tokens: caching wins from call number 2
    analytic: write premium 0.25*P is repaid by 0.9*P saved on every later call -> from the 2nd call whenever 0.9*P > 0.25*P
7.3 bool(empty SemanticCache) = False -> `if self.semantic:` would skip it; use `is not None`
```

## Step 08 — LLM gateway
**8.1** The third provider answers after two recorded failures; the budget check raises `BudgetExceeded` before any provider is called. **8.2** `KeyError: 'medium'`; wrap it to raise a configuration error listing the valid aliases. **8.3** Routing short requests to the small model cut the illustrative cost by about a third on this traffic mix.
**8.4 Explain aloud.** "Where the provider offers native structured output or strict tool schemas I use them; I still validate with pydantic and give the model its own error for one or two repair attempts, and I track the repair rate. Streaming cannot fall back mid-way because the user has already seen text from the first provider; choose the provider before the first token or buffer the first chunk.

Output of [`sol_08.py`](sol_08.py):
```text
8.1 answered by c | failures: [('a', 'TransientError'), ('b', 'TransientError')]
    budget: budget $0.0020 exhausted (spent $0.0080)
8.2 KeyError: 'medium' -> valid aliases: ['default']
8.3 1000 requests all on large: $5.52; routed by size: $3.76 (32% cheaper)
```

## Step 09 — Observability
**9.1** p50/p95/p99 of 1..100 are 50/95/99 (nearest rank). The cumulative histogram counts samples at or below each bucket bound (Prometheus-style `le`). **9.2** Without `reset(token)` the context variable keeps the first request's span as current, so every later span nests under it and one trace grows without end. **9.3** Copy the context into the worker: `contextvars.copy_context().run(fn)`; the worker's span then has the request span as its parent.
**9.4 Explain aloud.** "Traces show where time went for one request, logs show what happened, metrics show the system trend; they join on trace id. I report p95 because the mean hides the tail. The OpenTelemetry GenAI conventions (`gen_ai.operation.name`, `gen_ai.usage.input_tokens`, ...) are still in Development status as of Nov 2026 and moved to their own repository, so I pin versions and check before depending on a name.

Output of [`sol_09.py`](sol_09.py):
```text
9.1 p50/p95/p99 of 1..100: 50 95 99
    cumulative histogram (le): {25: 5, 50: 6, 100: 8, 500: 9, 2000: 10}
9.2 without reset, request 2 would see parent = request-1 -> every later request nests under request 1
    after reset, parent = None
9.3 plain thread sees span: False | copied context sees it: True | worker span parent is root: True
```

## Step 10 — Interview problems
**10.1** With this offline stack the sweep shows a steep trade-off. Overall `refusal_correct` peaks at threshold 0.2 (0.86) but only because 24 of the 29 questions are answerable: there, only 20% of the unanswerable questions are refused. At 0.4: 79% of answerable questions are answered and 60% of unanswerable ones refused; at 0.6: 54% and 80%. Report the two rates separately, not the pooled accuracy. Before shipping a threshold check: it is computed on a held-out set (not the questions you tuned on), the retriever and reranker are the production ones (scores are not comparable across rerankers), and the product's cost of a false refusal vs a false accept (a support bot may prefer to answer; a legal bot to refuse). **10.2** Validate at the boundary: a pydantic request model (HTTP 422) or an explicit `isinstance` check before the memory store. **10.3** The guard stops the run at step 2 instead of burning all `max_steps`.
**10.4** Use the seven-step frame. Sketch: *entities* `Template(name, version, body, variables)`, `Rendered`, `Experiment(variant weights)`; *interfaces* `TemplateStore` (get by name+version), `Renderer`, `Assigner` (sticky hash(user_id) -> variant); *edge cases* missing variable, unknown version, rollback, weights not summing to 1, injection through variables; *tests* render with all variables, missing variable error, sticky assignment, weight distribution, version pinning.
**10.5 Explain aloud.** "Guards normalise, redact and block injection; a traced, cached wrapper checks the cache with the index version in the key; on a miss the service retrieves hybrid, reranks, refuses if the best score is too low, packs the prompt within the token budget, generates through retry, timeout and fallback, verifies citations and returns an immutable `Answer`; refusals are not cached.

Output of [`sol_10.py`](sol_10.py):
```text
10.1 threshold  answerable-answered  unanswerable-refused  refusal_correct
       0.0         1.00                0.00                0.83
       0.2         1.00                0.20                0.86
       0.3         0.83                0.20                0.72
       0.4         0.79                0.60                0.76
       0.5         0.71                0.60                0.69
       0.6         0.54                0.80                0.59
       0.8         0.25                1.00                0.38
10.2 boundary validation: content must be a non-empty string
10.3 stopped: loop after 2 steps (max_steps was 8)
```
