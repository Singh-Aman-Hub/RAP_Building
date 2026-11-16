# Module 20 — Exercises

Rules: change ONE thing at a time, predict the result before running, write down what happened. Worked answers: [solutions/solutions.md](solutions/solutions.md); runnable code: `solutions/sol_NN.py` (run from anywhere with `HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/solutions/sol_NN.py`). Numbering `N.m` = step N, exercise m. Items marked (code) have a runnable solution; the others are answered in prose.

## Step 01 — LLD foundations
- **1.1 Modify (code):** add a `stream` method to a copy of the `Generator` Protocol. Is `Shouter` still an instance? Is a subclass that has `stream`?
- **1.2 Break it (code):** `Summarizer("not a generator").summarize("x")`. Where does the error appear, and how do you make it appear at construction?
- **1.3 Rebuild (code):** write an `Embedder` Protocol, a fake that satisfies it, and the ABC version; show the instantiation error.
- **1.4 Explain aloud:** Protocol vs ABC in three sentences, including what `runtime_checkable` does and does not check.

## Step 02 — Domain model
- **2.1 Modify (code):** put the position into the chunk id. Insert one paragraph at the top of a 20-paragraph document: how many chunks are re-embedded with and without position in the id?
- **2.2 Break it (code):** assign `chunk.text = ...`. Then use `dataclasses.replace`. Why is the resulting `chunk_id` wrong, and what is the right way to "edit" a chunk?
- **2.3 Rebuild (code):** write `to_dict`, `from_dict`, `to_json`, `from_json` for `Answer` (with nested citations) and a round-trip test, including a refusal.
- **2.4 Explain aloud:** dataclass vs pydantic: which at an HTTP boundary, which inside, and why.

## Step 03 — Ingestion pipeline
- **3.1 Modify (code):** save the `Manifest` to disk, load it in a "new process", and re-run. What happens with the same store and with an empty store? What does that tell you about persisting state?
- **3.2 Break it (code):** `FixedWordChunker(size=40, overlap=40)`. Read the error; what would have happened without the check?
- **3.3 Rebuild (code):** given `existing = {a,b,c}` and `wanted = {b,c,d}` compute stale / new / reused, then write `IngestionPipeline.run()` from memory.
- **3.4 Explain aloud:** why is the manifest updated only after a document succeeds, and what is in the index version?

## Step 04 — Query pipeline
- **4.1 Modify (code):** implement the stop-at-first packer of module 14 next to `PromptBuilder`; find a case where it packs nothing.
- **4.2 Modify (code):** compute RRF scores by hand for a chunk ranked 1st in one list and 3rd in another; verify with `rrf`.
- **4.3 Break it (code):** run `CitationVerifier` on a citation with a wrong number (`30 days` instead of `9`). Is it flagged? What does this say about lexical support?
- **4.4 Explain aloud:** where do you refuse, how do you pick the threshold, and what are false accepts and false refusals?

## Step 05 — Design patterns
- **5.1 Modify (code):** register a new retriever plugin from your own script and select it by name from a config dict, without editing `factory.py`.
- **5.2 Break it (code):** register the same name twice; ask for an unknown name. Read both errors.
- **5.3 Rebuild (code):** write `Registry` (register decorator + create with a helpful error) in about 12 lines.
- **5.4 Explain aloud:** Strategy vs Template Method vs Decorator, each with a ragkit example.

## Step 06 — Resilience
- **6.1 Modify (code):** compare no jitter, full jitter and equal jitter for 1000 clients (peak retries per slot and mean wait).
- **6.2 Break it (code):** `retry_on=(Exception,)` and a 401. How many provider calls are made?
- **6.3 Rebuild (code):** write a token bucket from scratch (8 lines) and reproduce the burst-then-deny pattern.
- **6.4 Explain aloud:** the order of rate limiter, retry, circuit breaker and timeout, and why fallback must not trigger on a permanent error.

## Step 07 — Caching
- **7.1 Rebuild (code):** an LRU cache with `OrderedDict` in about 10 lines.
- **7.2 Modify (code):** after how many calls does provider prompt caching pay off for a 1k and a 10k prefix (write 1.25x, read 0.1x)? Derive it by hand.
- **7.3 Break it (code):** show that an empty `SemanticCache` is falsy and what `if self.semantic:` does to the service.
- **7.4 Explain aloud:** why a semantic cache can return the wrong answer ("shipping free" vs "not free"), four mitigations, and how versioned keys invalidate.

## Step 08 — LLM gateway
- **8.1 Modify (code):** a route with three providers where the first two are down; add a budget and show the call refused before any provider is touched.
- **8.2 Break it (code):** call with an unconfigured model alias. How would you make the error friendlier?
- **8.3 Modify (code):** compute the cost of 1000 requests (80% short, 20% long) all on a large model vs routed by size.
- **8.4 Explain aloud:** structured output: native schema features vs validation and a repair loop; why streaming cannot fall back mid-way.

## Step 09 — Observability
- **9.1 Rebuild (code):** nearest-rank percentile and a cumulative histogram; p50/p95/p99 of 1..100.
- **9.2 Break it (code):** explain (and simulate) what happens to later requests if the span context is never reset.
- **9.3 Modify (code):** make a span created in a worker thread a child of the request's span.
- **9.4 Explain aloud:** the three pillars, why p95 and not the mean, and the status of the OpenTelemetry GenAI conventions.

## Step 10 — Interview problems
- **10.1 Modify (code):** sweep the refusal threshold with the eval harness and print refusal accuracy for answerable and unanswerable questions. Which threshold would you ship, and what do you need to check first?
- **10.2 Break it (code):** `memory.add("s", "user", None)`. Where should validation live?
- **10.3 Modify (code):** make an agent repeat the same tool call; show the loop guard stopping it before `max_steps`.
- **10.4 Rebuild:** apply the seven-step frame to "design a prompt template manager with versioning and A/B testing": requirements, entities, interfaces, class diagram, one sequence diagram, five edge cases, five tests. Time yourself (45 minutes).
- **10.5 Explain aloud:** walk through the RAG service sequence diagram (guards, trace, cache, retrieve, rerank, refuse, prompt, generate with retry, verify) in two minutes.
