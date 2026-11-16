# Module 20 — Interview gist: LLM system low-level design

One-page cram sheet. Full theory and real output: [README](README.md) and the step walkthroughs. All names below are real classes in [`ragkit/`](ragkit/).

## 1. SOLID in one line each
- **S**ingle responsibility: one reason to change (a `Chunker` chunks, an `Embedder` embeds, the pipeline orchestrates).
- **O**pen/closed: add a retriever by registering it, not by editing `build_system` (`Registry`).
- **L**iskov: any `Generator` (extractive, gateway-backed, flaky, cached) works wherever a `Generator` is expected; a subclass must not tighten the contract.
- **I**nterface segregation: small Protocols (`Retriever.retrieve`, `Generator.generate`), not one fat `RagBackend`.
- **D**ependency inversion: `RagService` receives Protocols through its constructor; only `build_system` names concrete classes (composition root).

## 2. Patterns used in ragkit, and where
| Pattern | Where | Why (the reason, not the name) |
|---|---|---|
| Strategy | dense / BM25 / hybrid retrievers, chunkers, rerankers | swap an algorithm behind one interface |
| Registry + Factory | `Registry`, `build_system(config)` | plugins by name from a config dict |
| Adapter | `QdrantStore`, `Module14Embedder`, `SharedClientProvider` | make a foreign API fit our Protocol |
| Decorator | `CachedAnswerer`, `Retrying/Breaker/Timeout/RateLimitedGenerator`, `Traced*`, `GuardedAnswerer` | add a cross-cutting concern without editing the core (order matters) |
| Chain of Responsibility | input guards (`Handler`, `build_chain`) | ordered checks that can stop the request |
| Observer | `EventBus` | emit events; subscribers cannot crash the emitter |
| Template Method | `BaseChunker` | fixed skeleton, subclasses fill the split step |
| Value objects | frozen `Document/Chunk/Answer` | safe to share, hash, cache |
| Dependency injection | every constructor | fakes in tests, adapters in production |

## 3. The interfaces (typing.Protocol), the whole contract surface
`Loader.load() -> Iterator[Document]` · `Parser.parse(doc) -> list[Element]` · `Chunker.chunk(doc, elements) -> list[Chunk]` · `Embedder.embed(texts) -> ndarray (n, dim)` · `VectorStore.upsert/search/delete_*/count` · `Retriever.retrieve(query, k) -> list[SearchHit]` · `Reranker.rerank(query, hits, top_n)` · `Generator.generate(system, user) -> str` · `Answerer.answer(question) -> Answer` · `TokenCounter.count(text) -> int` · gateway `Provider.complete(req)/stream(req)` · `Scorer` (eval) · `EvalHook`.
Protocol = structural (no inheritance, easy fakes; `runtime_checkable` checks names only). ABC = nominal, use when shipping shared behaviour.

## 4. Resilience patterns (step 06)
Classify errors: **transient** (retry) vs **permanent** (never retry, never fall back). Retry with exponential backoff + **full jitter**, honour `retry-after`, cap attempts. **Timeout** per attempt. **Circuit breaker** closed -> open -> half-open. **Token bucket** (burst then steady rate, requests AND tokens per minute). **Fallback** to the next provider only on infrastructure errors. **Idempotency key** so a retried call is not billed or executed twice. Test with a `FakeClock` and a scripted `FlakyGenerator`. Order outside-in: rate limit -> breaker -> retry -> timeout -> call.

## 5. Caching layers (step 07)
1. Exact cache: key = hash(normalised prompt + params + **scope**: index version, model, tenant). 2. Semantic cache: embedding similarity above a high threshold, same scope (risk: "free" vs "not free"). 3. Provider prompt (prefix) caching: byte-identical prefix reuse, a cost/latency discount, not an answer cache. Also embedding cache and retrieval cache. TTL bounds staleness, LRU bounds size, **versioned keys** invalidate without scanning.

## 6. Gateway and streaming (step 08)
Frozen `LLMRequest/LLMResponse/Usage`, model **alias** -> ordered providers + fallback, `CostMeter` with a budget guard (`BudgetExceeded` is permanent), streaming as a generator / async iterator (no fallback mid-stream, TTFT matters), structured output = schema + pydantic validation + bounded **repair loop**, tool schemas.

## 7. Observability (step 09)
Traces (where), logs (what, JSON with `trace_id`), metrics (how much: p50/p95, tokens, cost, cache hit rate, refusal and retry rate). Nested spans via `ContextVar`. OTel GenAI attribute names. **Online eval hooks** (citation valid, slow) and **feedback** joined to the trace id become the next offline eval set.

## 8. Versioning, guardrails, budgets (step 11)
- **Versioning**: prompt = semver + content hash, immutable, **pinned** with the model snapshot in a `ReleaseConfig`; rollout by `sha256(salt + user_id)` bucket; canary owns the lowest buckets (ramp only adds users); percent 0 = rollback; offline eval gate before canary.
- **Guardrails**: in = limits, PII redaction (Luhn for cards), injection heuristics (weak, one layer); out = pydantic schema, citation exists and is supported by a retrieved chunk, confidence floor, PII scrub. Format error -> one repair; ungrounded -> fixed refusal.
- **Budgets**: window = system + history + chunks + question + **answer reserve**; drop oldest history, then lowest-ranked chunks; per-request `max_tokens`; per-tenant token bucket + spend ceiling (reserve worst case, settle after); one ledger row per request.

## 9. How to answer an LLD question in 45 minutes
| Min | Do | Say / draw |
|---|---|---|
| 0-5 | **Clarify**: users, scale, latency, what is in/out of scope, failure tolerance | "I will assume ... ; out of scope: ..." (write 3-5 requirements) |
| 5-10 | **Entities** (nouns) and **flows** (verbs): value objects, ids (content hash or uuid) | list: Document, Chunk, Answer, Request ... |
| 10-20 | **Interfaces first**: 4-8 small Protocols with signatures; what is a seam (model, store, clock, network) | one box per Protocol, arrows = "depends on" |
| 20-30 | **Class diagram + one sequence diagram** (happy path), then the **composition root** | name each pattern with its reason |
| 30-38 | **Failure and edge cases**: timeouts, retries, partial failure, idempotency, duplicates, empty input, concurrency, cost | transient vs permanent, what is logged |
| 38-43 | **Cross-cutting**: caching, observability, versioning, guardrails, budgets, security | decorators around the core |
| 43-45 | **Tests and extension**: fakes, contract tests, one extension ("add a second vector DB") | "closed for modification, open for extension" |
Rules: state trade-offs in one sentence each; code only the core method if asked; do not name a pattern without its reason; end by walking one request through the diagram.

## 10. Quick-fire answers
1. LLD vs HLD? Classes, interfaces and flows inside a service vs services and stores across a system.
2. Why Protocols? Structural contracts make fakes trivial and keep the core independent of vendors.
3. Where do you refuse? Before retrieval (bad input), after retrieval (low score, no paid call), after generation (ungrounded).
4. Retry what? Only transient errors, with full jitter, a cap and an idempotency key.
5. Chunk id? Content hash of (doc, section, text); never the position, so re-indexing is idempotent.
6. Exact vs semantic cache? Exact is safe; semantic saves more but can match a near-miss; scope both by index version.
7. Rate limit an LLM API? Two buckets (requests and tokens per minute) per tenant; reserve then settle.
8. Reliable JSON? Schema + pydantic + one repair; track the repair rate.
9. Safe prompt rollout? Version, hash, pin, offline gate, canary by user-id hash, rollback = percent 0.
10. What do you log per request? trace id, release fingerprint, tokens, cost, latency, outcome, cache hit, flags.

## 11. Confusable pairs
Protocol vs ABC · Adapter vs Decorator (changes vs keeps the interface) · retry vs fallback (same provider vs another) · timeout vs breaker (one call vs a pattern of failures) · exact vs semantic cache vs provider prompt cache · canary vs A/B vs shadow · rate limit vs budget (speed vs total) · refuse vs repair · trace vs log vs metric · dataclass vs pydantic (trusted inside vs untrusted boundary).

Prerequisites if rusty: [05 classes and dataclasses](../00_python_foundations/05_classes_and_dataclasses/WALKTHROUGH.md), [07 decorators and generators](../00_python_foundations/07_modules_decorators_generators/WALKTHROUGH.md). Practice: [exercises.md](exercises.md).
