<!-- vocabulary draft: 20_llm_system_lld -->

### Low-level design (LLD)
<!-- cat: python -->
- **Technical definition:** The design of the classes, interfaces, methods and collaborations inside one service, including patterns, failure handling and tests.
- **Plain English:** The blueprint of one machine, not the map of the factory.
- **Example / analogy:** HLD picks the services; LLD decides that `RagService` receives a `Retriever`, a `Reranker` and a `Generator` and nothing else.
- **Snippet:**
```python
class RagService:
    def __init__(self, retriever, reranker, generator): ...
```
- **Snippet explained:** The service never constructs its collaborators, so tests can pass fakes.
- **Related terms:** [[Dependency injection]], [[SOLID]]

### SOLID
<!-- cat: python -->
- **Technical definition:** Five object-design principles: single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion.
- **Plain English:** Five habits that keep code easy to change.
- **Example / analogy:** A new retriever is added by registering it; `RagService` is not edited (open/closed).
- **Snippet:**
```python
RETRIEVERS.register('dense_top1')(make_retriever)
```
- **Snippet explained:** Registration extends the system without modifying `build_system`.
- **Related terms:** [[Dependency injection]], [[Strategy pattern]], [[Protocol (typing.Protocol)]]

### Protocol (typing.Protocol)
<!-- cat: python -->
- **Technical definition:** A structural interface: any object with the declared methods satisfies it, without inheriting. With `@runtime_checkable`, `isinstance` checks attribute names only, not signatures.
- **Plain English:** A plug shape: if it fits the socket, it works.
- **Example / analogy:** A 3-line test fake and a vendor wrapper both satisfy `Generator`.
- **Snippet:**
```python
@runtime_checkable
class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...
```
- **Snippet explained:** Declares the contract; any class with `generate(system, user)` is accepted.
- **Related terms:** [[Abstract base class (ABC)]], [[Dependency injection]], [[Adapter pattern]]

### Abstract base class (ABC)
<!-- cat: python -->
- **Technical definition:** A class deriving from `abc.ABC` with `@abstractmethod` members; subclasses must implement them or cannot be instantiated.
- **Plain English:** A form with mandatory fields.
- **Example / analogy:** `BaseChunker` fixes the algorithm and forces subclasses to implement `_groups`.
- **Snippet:**
```python
class BaseChunker(ABC):
    @abstractmethod
    def _groups(self, elements): ...
```
- **Snippet explained:** Instantiating a subclass that forgot `_groups` raises TypeError immediately.
- **Related terms:** [[Protocol (typing.Protocol)]], [[Template Method]]

### Frozen dataclass (slots)
<!-- cat: python -->
- **Technical definition:** `@dataclass(frozen=True, slots=True)`: fields cannot be re-assigned, `__hash__` is generated, and fields live in slots (no per-instance `__dict__`). Frozen is shallow.
- **Plain English:** A laminated card: readable, shareable, not editable.
- **Example / analogy:** Chunks, requests and answers are frozen so caches and threads can share them safely.
- **Snippet:**
```python
@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    text: str
```
- **Snippet explained:** Assigning `chunk.text = ...` raises FrozenInstanceError; the object is hashable.
- **Related terms:** [[Value object]], [[Pydantic v2 (BaseModel)]]

### Value object
<!-- cat: python -->
- **Technical definition:** An object defined by its field values, equal when they are equal, normally immutable.
- **Plain English:** A banknote: any 10-rupee note is as good as another.
- **Example / analogy:** Two `Chunk` objects with equal fields are the same chunk.
- **Snippet:**
```python
Chunk('a','d','t') == Chunk('a','d','t')  # True
```
- **Snippet explained:** Equality by value lets chunks be set members and dict keys.
- **Related terms:** [[Frozen dataclass (slots)]], [[Content-hash id]]

### Dependency injection
<!-- cat: python -->
- **Technical definition:** Passing a class the objects it depends on (usually through its constructor) instead of letting it create them.
- **Plain English:** A chef is handed ingredients rather than growing them.
- **Example / analogy:** `Summarizer(generator)` works with a fake LLM in tests and a real API in production.
- **Snippet:**
```python
class Summarizer:
    def __init__(self, generator): self.generator = generator
```
- **Snippet explained:** The dependency is a parameter, so any object with the right method can be passed.
- **Related terms:** [[Composition root]], [[Protocol (typing.Protocol)]]

### Composition root
<!-- cat: python -->
- **Technical definition:** The single place in a program where concrete implementations are chosen and wired together.
- **Plain English:** The electrical panel of a house.
- **Example / analogy:** `build_system(config)` reads names from a config dict and returns a ready `System`.
- **Snippet:**
```python
system = build_system({'store': {'type': 'qdrant'}}, loader)
```
- **Snippet explained:** Changing the store is a one-word config change; no other module names Qdrant.
- **Related terms:** [[Registry and Factory]], [[Dependency injection]]

### Content-hash id
<!-- cat: ml -->
- **Technical definition:** An identifier computed by hashing the content (for chunks: document id, section and text), so identical content always gets an identical id.
- **Plain English:** A fingerprint rather than a ticket number.
- **Example / analogy:** Re-indexing an unchanged document produces the same chunk ids, so nothing is re-embedded.
- **Snippet:**
```python
content_hash(doc_id, section, text)  # 16 hex chars
```
- **Snippet explained:** Joins the parts with a separator and keeps the first 16 hex characters of sha256.
- **Related terms:** [[Idempotent upsert]], [[Index version]]

### Idempotent upsert
<!-- cat: ml -->
- **Technical definition:** A write that creates or overwrites the record with the same id, so repeating it leaves the same state.
- **Plain English:** Pressing an elevator button twice.
- **Example / analogy:** Running the ingestion job twice gives the same index.
- **Snippet:**
```python
store.upsert([(chunk, embedding)])  # same chunk_id overwrites
```
- **Snippet explained:** No duplicates are created for an existing id.
- **Related terms:** [[Content-hash id]], [[Incremental re-indexing (manifest)]]

### Incremental re-indexing (manifest)
<!-- cat: ml -->
- **Technical definition:** Re-indexing only documents whose content hash differs from a recorded manifest, and removing documents that disappeared.
- **Plain English:** Re-reading only the chapters that changed.
- **Example / analogy:** One edited refund section re-embedded 1 chunk instead of 36.
- **Snippet:**
```python
if manifest.docs.get(doc.doc_id) == doc.content_hash: skip
```
- **Snippet explained:** An unchanged document is skipped without parsing or embedding.
- **Related terms:** [[Idempotent upsert]], [[Dead-letter queue]], [[Index version]]

### Dead-letter queue
<!-- cat: api -->
- **Technical definition:** A holding list for items that failed processing, with the stage and error, so they can be inspected and replayed.
- **Plain English:** The 'return to sender' shelf.
- **Example / analogy:** A document the embedder rejects is parked while the other eight are indexed.
- **Snippet:**
```python
DeadLetter(doc_id, source, 'embed', 'RuntimeError: ...')
```
- **Snippet explained:** Records where the failure happened for later repair.
- **Related terms:** [[Incremental re-indexing (manifest)]], [[Idempotent upsert]]

### Index version
<!-- cat: ml -->
- **Technical definition:** A fingerprint of everything that changes what stored vectors mean (embedder, dimension, chunker settings, schema). A change forces a rebuild.
- **Plain English:** A recipe version printed on every jar.
- **Example / analogy:** Switching to a 256-dim embedder changed the version and rebuilt the index.
- **Snippet:**
```python
content_hash(embedder.name, str(embedder.dim), chunker.name, 'schema1', length=8)
```
- **Snippet explained:** A short hash; compared with the manifest's stored version at the start of each run.
- **Related terms:** [[Content-hash id]], [[Exact cache key and scope]]

### Template Method
<!-- cat: python -->
- **Technical definition:** A base class defines a fixed algorithm and calls overridable hook methods that subclasses implement.
- **Plain English:** A form with one blank to fill in.
- **Example / analogy:** `BaseChunker.chunk()` dedupes and assigns ids; subclasses only implement `_groups()`.
- **Snippet:**
```python
def chunk(self, doc, elements):
    for section, text in self._groups(elements): ...
```
- **Snippet explained:** The template calls the hook; each chunker only decides how to group text.
- **Related terms:** [[Abstract base class (ABC)]], [[Strategy pattern]]

### Strategy pattern
<!-- cat: python -->
- **Technical definition:** A family of interchangeable algorithms behind one interface, one of which is injected.
- **Plain English:** Choosing a route in a map app.
- **Example / analogy:** Dense, BM25 and hybrid retrievers all satisfy `Retriever`.
- **Snippet:**
```python
HybridRetriever([DenseRetriever(emb, store), BM25Retriever(store)])
```
- **Snippet explained:** The hybrid retriever is itself a retriever composed of two strategies.
- **Related terms:** [[Registry and Factory]], [[Template Method]]

### Registry and Factory
<!-- cat: python -->
- **Technical definition:** A registry maps names to constructors; a factory builds objects from a name and keyword arguments, so plugins are chosen by config.
- **Plain English:** A menu and a kitchen.
- **Example / analogy:** `STORES.create('qdrant', dim=128)`; an unknown name lists the valid ones.
- **Snippet:**
```python
@STORES.register('memory')
def make(**_): return InMemoryVectorStore()
```
- **Snippet explained:** Registration records the constructor under a name; `create` looks it up.
- **Related terms:** [[Composition root]], [[Strategy pattern]]

### Adapter pattern
<!-- cat: python -->
- **Technical definition:** A class that converts one interface into another that callers expect, without changing the wrapped class.
- **Plain English:** A travel plug adapter.
- **Example / analogy:** `QdrantStore` makes Qdrant's client satisfy `VectorStore`; `Module14Embedder` renames `encode` to `embed`.
- **Snippet:**
```python
class Module14Embedder:
    def embed(self, texts): return self._inner.encode(list(texts))
```
- **Snippet explained:** Translates the method name; the wrapped class is untouched.
- **Related terms:** [[Decorator pattern]], [[Protocol (typing.Protocol)]]

### Decorator pattern
<!-- cat: python -->
- **Technical definition:** A wrapper implementing the same interface as the wrapped object that adds behaviour (cache, retry, timing, tracing) before or after delegating.
- **Plain English:** Gift wrap: still a gift.
- **Example / analogy:** `CachingEmbedder`, `RetryingGenerator`, `TracedGenerator` stack around a `Generator`.
- **Snippet:**
```python
TracedGenerator(RetryingGenerator(real_generator), tracer)
```
- **Snippet explained:** Each layer has the same `generate` method; order decides what each layer sees.
- **Related terms:** [[Adapter pattern]], [[Chain of Responsibility]]

### Chain of Responsibility
<!-- cat: python -->
- **Technical definition:** A sequence of handlers where each processes the request and either passes it on or stops the chain.
- **Plain English:** Airport security lanes.
- **Example / analogy:** Normalize, RedactPII, MaxLength and InjectionGuard run before the model sees a question.
- **Snippet:**
```python
chain = build_chain(Normalize(), RedactPII(), MaxLength(500), InjectionGuard())
```
- **Snippet explained:** Links the handlers in order; any can set `ctx.blocked` to stop.
- **Related terms:** [[Decorator pattern]], [[Observer (event bus)]]

### Observer (event bus)
<!-- cat: python -->
- **Technical definition:** Producers emit named events; subscribers register interest and are called without the producer knowing them. A subscriber error must not break the producer.
- **Plain English:** Subscribing to a newsletter.
- **Example / analogy:** `RagService` emits `retrieved`, `reranked`, `generated`; metrics subscribe.
- **Snippet:**
```python
bus.subscribe('refused', fn); bus.emit('refused', {...})
```
- **Snippet explained:** `emit` calls each subscriber inside try/except.
- **Related terms:** [[Chain of Responsibility]], [[Online eval hook]]

### Reciprocal Rank Fusion (RRF)
<!-- cat: llm -->
- **Technical definition:** Merging ranked lists with score(c) = sum over lists of 1/(k + rank), k = 60.
- **Plain English:** Voting by rank rather than by raw score.
- **Example / analogy:** A chunk first in both dense and BM25 lists scores 2/61 = 0.0328.
- **Snippet:**
```python
fused[id] += 1.0 / (60 + rank)
```
- **Snippet explained:** Adds each list's rank-based contribution per chunk; needs no score calibration.
- **Related terms:** [[Strategy pattern]], [[Token budget (context packing)]]

### Token budget (context packing)
<!-- cat: llm -->
- **Technical definition:** The maximum prompt tokens allotted to retrieved chunks; chunks are added best-first while they fit, and oversized ones are skipped.
- **Plain English:** Packing a suitcase with a weight limit.
- **Example / analogy:** With budget 150 only one chunk fit and four were dropped.
- **Snippet:**
```python
if total + cost <= budget: used.append(h)
else: dropped.append(h)
```
- **Snippet explained:** Skip-and-continue lets smaller chunks fill the remaining room.
- **Related terms:** [[Refusal policy]], [[Citation verification]]

### Refusal policy
<!-- cat: llm -->
- **Technical definition:** A rule that returns 'I don't know' before generation when evidence is missing or the best reranked score is below a threshold.
- **Plain English:** A librarian saying 'we have nothing on that'.
- **Example / analogy:** A cross-encoder score of -10.33 against a gate of -6.5 refuses the London question.
- **Snippet:**
```python
ScoreThresholdPolicy(min_score=-6.5).check(hits)
```
- **Snippet explained:** Returns None to proceed or a reason string to refuse.
- **Related terms:** [[Citation verification]], [[Online eval hook]]

### Citation verification
<!-- cat: llm -->
- **Technical definition:** Checking that every [n] in an answer points to an existing source and that the citing sentence is lexically supported by it; it does not prove faithfulness.
- **Plain English:** Checking the footnote exists and roughly relates.
- **Example / analogy:** 'Refunds take 30 days [1]' still passes because of word overlap.
- **Snippet:**
```python
CitationVerifier(min_overlap=0.3).verify(text, used)
```
- **Snippet explained:** Splits sentences, checks marker range and word overlap with the cited chunk.
- **Related terms:** [[Refusal policy]], [[Eval harness and regression gate]]

### Exponential backoff with jitter
<!-- cat: api -->
- **Technical definition:** Waiting longer after each failed attempt (doubling to a cap) and randomising the delay so clients do not retry together.
- **Plain English:** Knocking again after 1, 2, 4 seconds, each time at a random moment within the interval.
- **Example / analogy:** Full jitter spread 1000 retries over ten slots instead of one.
- **Snippet:**
```python
delay = rng.uniform(0, min(max_delay, base * 2 ** attempt))
```
- **Snippet explained:** The cap doubles per attempt; uniform(0, cap) is the full-jitter delay.
- **Related terms:** [[Circuit breaker]], [[Idempotency key]]

### Circuit breaker
<!-- cat: api -->
- **Technical definition:** A guard that stops calls to a failing dependency after repeated failures (open), then allows one probe (half-open) before closing again.
- **Plain English:** A fuse box.
- **Example / analogy:** After two 503s the breaker rejects calls instantly until 30 seconds pass.
- **Snippet:**
```python
breaker.call(lambda: provider.generate(system, user))
```
- **Snippet explained:** Raises CircuitOpenError while open; success in half-open closes it.
- **Related terms:** [[Exponential backoff with jitter]], [[Token bucket]]

### Token bucket
<!-- cat: api -->
- **Technical definition:** A rate limiter: a bucket of capacity tokens refilled at a fixed rate; each request spends tokens; allows bursts and a steady rate.
- **Plain English:** A jar of tickets that refills slowly.
- **Example / analogy:** capacity 4, rate 2/s: four instant requests, the fifth waits 0.5 s.
- **Snippet:**
```python
tokens = min(capacity, tokens + elapsed * rate)
```
- **Snippet explained:** Refill lazily from timestamps, then spend if enough remain.
- **Related terms:** [[Circuit breaker]], [[Idempotency key]]

### Idempotency key
<!-- cat: api -->
- **Technical definition:** A client-chosen key identifying one logical operation so repeated requests have the effect of one.
- **Plain English:** A cheque number.
- **Example / analogy:** Three retries of a charge executed once.
- **Snippet:**
```python
key = idempotency_key('charge-user-42', 'order-1001')
```
- **Snippet explained:** A deterministic hash of the operation's identity.
- **Related terms:** [[Exponential backoff with jitter]], [[Dead-letter queue]]

### Injectable clock (fake clock)
<!-- cat: python -->
- **Technical definition:** Passing a time source as a parameter so tests control time; the fake's sleep advances time instantly.
- **Plain English:** A remote control for time.
- **Example / analogy:** A 30-second breaker timeout is tested in microseconds.
- **Snippet:**
```python
clock = FakeClock(); clock.advance(30)
```
- **Snippet explained:** Time moves only when the test says so.
- **Related terms:** [[Circuit breaker]], [[Token bucket]]

### Exact cache key and scope
<!-- cat: llm -->
- **Technical definition:** A hash of the normalised prompt, sampling parameters and a scope (index version, model, tenant); only identical requests hit.
- **Plain English:** A locker number made from everything that matters.
- **Example / analogy:** Changing the index version makes every old key unreachable.
- **Snippet:**
```python
cache_key(system, user, {'temperature': 0}, scope='idx-v1')
```
- **Snippet explained:** Canonical JSON with sorted keys, hashed to 32 hex characters.
- **Related terms:** [[TTL and LRU]], [[Semantic cache]]

### TTL and LRU
<!-- cat: python -->
- **Technical definition:** TTL: entries expire after a time limit. LRU: when full, the least recently used entry is evicted.
- **Plain English:** A best-before date and a shelf that drops the longest-unused item.
- **Example / analogy:** A 60 s TTL expired an entry at 61 s; capacity 2 evicted the unused key.
- **Snippet:**
```python
d.move_to_end(k); d.popitem(last=False)
```
- **Snippet explained:** OrderedDict keeps recency order: last = newest, first = oldest.
- **Related terms:** [[Exact cache key and scope]], [[Semantic cache]]

### Semantic cache
<!-- cat: llm -->
- **Technical definition:** A cache returning a stored answer when the new question's embedding is within a similarity threshold of a stored question's, within the same scope.
- **Plain English:** A librarian who recognises the same question in different words.
- **Example / analogy:** 'Is shipping not free' scored 0.93 against 'Is shipping free': a wrong hit.
- **Snippet:**
```python
sc.lookup(question, scope)  # (value, similarity) or None
```
- **Snippet explained:** Compares cosine similarity with every entry in the scope.
- **Related terms:** [[Exact cache key and scope]], [[Prompt caching]]

### Prompt caching
<!-- cat: llm -->
- **Technical definition:** Provider-side reuse of the processed prefix of a long prompt marked with cache_control; reads cost a fraction of normal input tokens.
- **Plain English:** A bookmark in a long document already read.
- **Example / analogy:** A 10k-token prefix over 10 calls cut input cost by about 77%.
- **Snippet:**
```python
{'cache_control': {'type': 'ephemeral'}}
```
- **Snippet explained:** Marks the end of the cacheable prefix; stable content goes first.
- **Related terms:** [[Semantic cache]], [[LLM gateway]]

### LLM gateway
<!-- cat: api -->
- **Technical definition:** A single entry point with unified request/response types that routes a model alias to ordered providers, with fallback, cost metering and a budget guard.
- **Plain English:** A switchboard for model calls.
- **Example / analogy:** Alias `small` maps to two providers; a 503 on the first falls back to the second.
- **Snippet:**
```python
Gateway({'small': [primary, backup]}, CostMeter(prices, budget_usd=0.05))
```
- **Snippet explained:** Routes by alias, falls back on transient errors, records cost per call.
- **Related terms:** [[Structured output repair loop]], [[Tool spec (tool-call schema)]]

### Structured output repair loop
<!-- cat: llm -->
- **Technical definition:** Ask for JSON, extract and validate it with pydantic, and on failure return the validation error to the model for a bounded number of retries.
- **Plain English:** Returning a form with a note about what to fix.
- **Example / analogy:** confidence 7 failed `<= 1`; one repair produced a valid object.
- **Snippet:**
```python
schema.model_validate(extract_json(resp.text))
```
- **Snippet explained:** Parses the first JSON object and validates types and ranges.
- **Related terms:** [[LLM gateway]], [[Pydantic v2 (BaseModel)]]

### Tool spec (tool-call schema)
<!-- cat: llm -->
- **Technical definition:** A description of a callable function (name, description, JSON Schema of arguments) given to the model; rendered per vendor and used to validate returned arguments.
- **Plain English:** A menu of actions with order forms.
- **Example / analogy:** One spec rendered as Anthropic `input_schema` and OpenAI `parameters`.
- **Snippet:**
```python
ToolSpec.from_model('search_docs', 'Search the KB', SearchArgs)
```
- **Snippet explained:** Builds the JSON Schema from a pydantic model.
- **Related terms:** [[LLM gateway]], [[Pydantic v2 (BaseModel)]]

### Pydantic v2 (BaseModel)
<!-- cat: python -->
- **Technical definition:** A library that validates and coerces data from type hints, reports all errors at once and generates JSON Schema.
- **Plain English:** A strict form that corrects '3' to 3 or rejects it.
- **Example / analogy:** ChunkModel(position='3') gives 3; a negative position raises ValidationError.
- **Snippet:**
```python
class AskRequest(BaseModel):
    question: str = Field(min_length=3)
```
- **Snippet explained:** Declares constraints; construction validates them.
- **Related terms:** [[Frozen dataclass (slots)]], [[Tool spec (tool-call schema)]]

### Span and trace
<!-- cat: api -->
- **Technical definition:** A span is a named, timed unit of work with attributes, status and a parent; a trace is the tree of spans for one request.
- **Plain English:** A receipt line per step of an order.
- **Example / analogy:** rag.answer contains retrieve, rerank and chat spans.
- **Snippet:**
```python
with tracer.span('retrieve', k=20) as s: ...
```
- **Snippet explained:** Starts a child of the current span and records its duration on exit.
- **Related terms:** [[Percentile latency (p50, p95)]], [[OpenTelemetry GenAI semantic conventions]]

### Percentile latency (p50, p95)
<!-- cat: ml -->
- **Technical definition:** The latency below which p percent of requests fall; nearest-rank takes sorted[ceil(p/100*n)-1].
- **Plain English:** '95 of 100 users waited at most this long.'
- **Example / analogy:** Mean 63.4 ms hid a p95 of 900 ms.
- **Snippet:**
```python
xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]
```
- **Snippet explained:** Sorts samples and picks the rank for the percentile.
- **Related terms:** [[Span and trace]], [[Online eval hook]]

### OpenTelemetry GenAI semantic conventions
<!-- cat: api -->
- **Technical definition:** Standard gen_ai.* attribute names (operation name, provider, request model, input/output tokens) for LLM spans and metrics; status Development as of Nov 2026 and moved to a separate repository.
- **Plain English:** Common labels so every tool reads your LLM traces.
- **Example / analogy:** A chat span carries gen_ai.usage.input_tokens and gen_ai.usage.output_tokens.
- **Snippet:**
```python
span.attrs['gen_ai.usage.input_tokens'] = 223
```
- **Snippet explained:** Records token usage under the standard key.
- **Related terms:** [[Span and trace]], [[Online eval hook]]

### Online eval hook
<!-- cat: llm -->
- **Technical definition:** A cheap check run on every live answer (citation valid, latency) whose flags become metrics.
- **Plain English:** A smoke detector on every request.
- **Example / analogy:** flag.bad_citation counts answers with no supported citation.
- **Snippet:**
```python
flags.update(hook.evaluate(question, answer, latency_ms))
```
- **Snippet explained:** Merges each hook's boolean flags for counting.
- **Related terms:** [[Eval harness and regression gate]], [[Citation verification]]

### Eval harness and regression gate
<!-- cat: llm -->
- **Technical definition:** Code that runs labelled cases through an answerer with pluggable scorers, aggregates results and lists cases that used to pass and now fail.
- **Plain English:** A test suite for answer quality.
- **Example / analogy:** Raising the refusal threshold regressed three answerable questions and lowered refusal accuracy.
- **Snippet:**
```python
regressions(base_report, new_report, 'contains')
```
- **Snippet explained:** Returns case ids that passed before and fail now.
- **Related terms:** [[Online eval hook]], [[Refusal policy]]

### Prompt versioning (semantic version + content hash)
<!-- cat: llm -->
- **Technical definition:** Giving each prompt text an immutable semantic version and a content hash, so any answer can be traced to the exact prompt that produced it and a change can be rolled back.
- **Plain English:** Prompts are code: edition number plus a fingerprint, never edited in place.
- **Example / analogy:** customer_rag@1.1.0 (hash 27dbd7057edd) replaces 1.0.0; rolling back re-activates 1.0.0 byte for byte.
- **Snippet:**
```python
reg.register('customer_rag', '1.1.0', text)
reg.activate('customer_rag', '1.1.0')
reg.rollback('customer_rag')
```
- **Snippet explained:** Registering a new version stores it immutably; activate moves the live pointer, rollback pops it.
- **Related terms:** [[Model pinning]], [[Canary release and A/B bucketing]], [[Content-hash id]]

### Model pinning
<!-- cat: llm -->
- **Technical definition:** Naming an exact dated model snapshot (and prompt version) in a release instead of a floating alias such as 'latest'.
- **Plain English:** Lock the oven model, not just the recipe.
- **Example / analogy:** A provider can repoint an alias, so behaviour shifts without a deploy on your side; a pinned snapshot is reproducible.
- **Snippet:**
```python
ReleaseConfig('stable','customer_rag','1.0.0','small','mock-small-2026-01-15')
```
- **Snippet explained:** The fingerprint of this object changes if any pinned field changes.
- **Related terms:** [[Prompt versioning (semantic version + content hash)]], [[Canary release and A/B bucketing]]

### Canary release and A/B bucketing
<!-- cat: llm -->
- **Technical definition:** Routing a small, adjustable share of users to a new release by hashing the user id into [0, 1) and comparing with a threshold; the share is raised while metrics are watched, percent 0 is the rollback.
- **Plain English:** A taste test on a few tables before the whole restaurant.
- **Example / analogy:** With the candidate owning the lowest buckets, ramping 10% to 50% only adds users, and a user always sees the same variant.
- **Snippet:**
```python
bucket('user-7', salt)  # deterministic in [0,1)
router.set_percent(50)
```
- **Snippet explained:** Same user and salt give the same bucket, so assignment is sticky and needs no table.
- **Related terms:** [[Prompt versioning (semantic version + content hash)]], [[Eval harness and regression gate]]

### Guardrails
<!-- cat: llm -->
- **Technical definition:** Checks wrapped around an LLM call: limits, PII redaction and injection heuristics on input; schema validation, groundedness and a refusal policy on output.
- **Plain English:** Bumpers on both sides of the lane.
- **Example / analogy:** Input heuristics are bypassable, so they are one layer next to least privilege and output checks.
- **Snippet:**
```python
build_input_guard(max_chars=500, max_tokens=150)
OutputGuard().check(raw, built.used)
```
- **Snippet explained:** Chain of Responsibility on the way in, a validator on the way out.
- **Related terms:** [[PII redaction]], [[Groundedness check]], [[Refusal policy]], [[Chain of Responsibility]]

### PII redaction
<!-- cat: llm -->
- **Technical definition:** Replacing personal data (email, phone, card numbers) with placeholders before the text is logged or sent to a third party; card candidates are confirmed with the Luhn checksum.
- **Plain English:** Black out the numbers before photocopying.
- **Example / analogy:** '4111 1111 1111 1111' becomes [CARD]; an order number that fails Luhn is left alone.
- **Snippet:**
```python
luhn_ok('4111111111111111')  # True
```
- **Snippet explained:** Luhn doubles every second digit from the right; valid sums are divisible by 10.
- **Related terms:** [[Guardrails]], [[Prompt-injection heuristics]]

### Prompt-injection heuristics
<!-- cat: llm -->
- **Technical definition:** Weighted regex patterns that flag or block inputs which try to override instructions, leak the system prompt or spoof prompt delimiters.
- **Plain English:** A bouncer who knows the usual fake IDs.
- **Example / analogy:** Paraphrases and other languages bypass them, so treat as one weak layer.
- **Snippet:**
```python
INJECTION_PATTERNS = [(r'ignore (all )?previous instructions', 1.0)]
```
- **Snippet explained:** A score of 1.0 or more blocks; below that it is only flagged in the context notes.
- **Related terms:** [[Guardrails]], [[PII redaction]]

### Groundedness check
<!-- cat: llm -->
- **Technical definition:** Verifying that every citation marker points to a chunk that was in the prompt and that the cited sentence is supported by that chunk's text.
- **Plain English:** Check the footnote says what the sentence claims.
- **Example / analogy:** A confident, well-formed JSON answer that cites a chunk but claims something the chunk does not say is refused.
- **Snippet:**
```python
CitationVerifier().verify(answer.answer, built.used)
```
- **Snippet explained:** Lexical word overlap here; an NLI model or LLM judge is the production upgrade.
- **Related terms:** [[Citation verification]], [[Guardrails]], [[Refusal policy]]

### Context-window budget
<!-- cat: llm -->
- **Technical definition:** Dividing the model window among system prompt, history, retrieved chunks, the question and a reserved answer: window = system + history + chunks + question + reserve.
- **Plain English:** Pack the return space in the suitcase first.
- **Example / analogy:** The reserve is subtracted first; drop oldest history and lowest-ranked chunks before anything else.
- **Snippet:**
```python
ContextBudget(window=800, answer_reserve=150).plan(system, q, history)
```
- **Snippet explained:** Returns how many history turns fit and the token budget left for PromptBuilder.
- **Related terms:** [[Token budget (context packing)]], [[Tenant budget and cost ledger]]

### Tenant budget and cost ledger
<!-- cat: llm -->
- **Technical definition:** Per-tenant admission control (tokens-per-minute bucket plus a spend ceiling, reserve the worst case then settle) and one ledger row per request with tokens, cost, release and outcome.
- **Plain English:** A bouncer plus a till roll, per customer.
- **Example / analogy:** Rate exhaustion is a retryable 429 with retry-after; budget exhaustion is permanent until the period resets.
- **Snippet:**
```python
gate.admit('acme', est)
gate.settle('acme', est, resp)
```
- **Snippet explained:** Settle refunds the unused reservation and adds the real cost to the tenant meter.
- **Related terms:** [[Token bucket]], [[LLM gateway]], [[Context-window budget]]
