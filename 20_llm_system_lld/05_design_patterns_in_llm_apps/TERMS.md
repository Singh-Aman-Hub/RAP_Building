# Terms — Design patterns in LLM apps: Strategy, Registry/Factory, Adapter, Decorator, Chain, Observer, Template Method

Folder: `20_llm_system_lld/05_design_patterns_in_llm_apps` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Strategy pattern](#strategy-pattern--concept) · [Registry and Factory](#registry-and-factory--concept) · [Composition root](#composition-root--concept) · [Adapter (vector DB adapter)](#adapter-vector-db-adapter--concept) · [Contract test](#contract-test--concept) · [Decorator pattern](#decorator-pattern--concept) · [Python function decorator (functools.wraps)](#python-function-decorator-functoolswraps--syntax) · [Chain of Responsibility](#chain-of-responsibility--concept) · [Observer (event bus, hooks)](#observer-event-bus-hooks--concept) · [Template Method](#template-method--concept) · [Open/closed principle in practice](#openclosed-principle-in-practice--concept) · [Qdrant local mode (':memory:')](#qdrant-local-mode-memory--library) · [UnknownPluginError](#unknownpluginerror--error) · [ValueError: ... already registered](#valueerror--already-registered--error)

---

## Strategy pattern  `(concept)`
**1. Definition:** Define a family of interchangeable algorithms behind one interface and inject the one to use.
**2. In plain words:** Choosing a route in a map app: same goal, different algorithm.
**3. Example to quote:** > "DenseRetriever, BM25Retriever and HybridRetriever are strategies behind Retriever."
**Seen in:** `ragkit/query.py`

## Registry and Factory  `(concept)`
**1. Definition:** A registry maps names to constructors; a factory creates objects from a name and parameters.
**2. In plain words:** A menu and a kitchen.
**3. Example to quote:** > "New plugins register themselves; build_system(config) never changes."
```python
@STORES.register('qdrant')
...
STORES.create('qdrant', dim=128)
```
**Seen in:** `ragkit/patterns.py`

## Composition root  `(concept)`
**1. Definition:** The single place where concrete objects are created and wired together.
**2. In plain words:** The electrician's panel.
**3. Example to quote:** > "build_system is the composition root; no other module names all implementations."
**Seen in:** `ragkit/factory.py`

## Adapter (vector DB adapter)  `(concept)`
**1. Definition:** A wrapper that makes a foreign API implement an interface we own.
**2. In plain words:** A plug adapter for another country's socket.
**3. Example to quote:** > "QdrantStore translates ids, filters and payloads so it can replace InMemoryVectorStore."
**Seen in:** `ragkit/qdrant_store.py`

## Contract test  `(concept)`
**1. Definition:** One test suite run against every implementation of an interface to prove they are substitutable.
**2. In plain words:** The same driving test for every car.
**3. Example to quote:** > "test_store_contract_same_for_every_adapter runs on memory and Qdrant."
```python
@pytest.mark.parametrize('make_store', [InMemoryVectorStore, lambda: QdrantStore(64)])
```
**Seen in:** `tests/test_patterns.py`

## Decorator pattern  `(concept)`
**1. Definition:** A wrapper that implements the same interface as the wrapped object and adds behaviour before or after delegating.
**2. In plain words:** Gift wrap: still a gift, now with a ribbon.
**3. Example to quote:** > "CachingEmbedder, TimedGenerator, later RetryingGenerator and TracedGenerator."
```python
class TimedGenerator:
    def generate(self, s, u):
        t0 = time.perf_counter()
        try: return self.inner.generate(s, u)
        finally: ...
```
**Seen in:** `ragkit/decorators.py`

## Python function decorator (functools.wraps)  `(syntax)`
**1. Definition:** `@decorator` replaces a function with a wrapper; `functools.wraps` copies name and docstring to it.
**2. In plain words:** A wrapper with the original label on it.
**3. Example to quote:** > "The timed decorator uses try/finally so failures are timed."
```python
@functools.wraps(fn)
def wrapper(*a, **kw): ...
```
**Seen in:** `ragkit/decorators.py`

## Chain of Responsibility  `(concept)`
**1. Definition:** A sequence of handlers where each processes the request and either passes it on or stops the chain.
**2. In plain words:** Airport security lanes.
**3. Example to quote:** > "Normalize -> RedactPII -> MaxLength -> InjectionGuard before the model sees a question."
**Seen in:** `ragkit/patterns.py`

## Observer (event bus, hooks)  `(concept)`
**1. Definition:** Producers emit named events; subscribers register interest and are called, without the producer knowing them.
**2. In plain words:** Subscribing to a newsletter.
**3. Example to quote:** > "Isolate subscriber exceptions so observability cannot break the product."
```python
bus.subscribe('refused', fn); bus.emit('refused', {...})
```
**Seen in:** `ragkit/patterns.py`

## Template Method  `(concept)`
**1. Definition:** A base class fixes an algorithm's skeleton and defers some steps to subclasses.
**2. In plain words:** A form with a few blanks.
**3. Example to quote:** > "BaseChunker.chunk() stays fixed; each chunker supplies _groups()."
**Seen in:** `ragkit/chunking.py`

## Open/closed principle in practice  `(concept)`
**1. Definition:** Software entities are open for extension, closed for modification.
**2. In plain words:** Add a room without rebuilding the house.
**3. Example to quote:** > "Registering a retriever extends the system; build_system is not edited."
**Seen in:** `WALKTHROUGH.md`, Experiments

## Qdrant local mode (':memory:')  `(library)`
**1. Definition:** QdrantClient(':memory:') runs the vector DB inside the Python process with no server.
**2. In plain words:** A pocket version of the database.
**3. Example to quote:** > "Good for tests and demos; switch to a URL for production without changing ragkit."
```python
client = QdrantClient(':memory:')
```
**Seen in:** `ragkit/qdrant_store.py`

## UnknownPluginError  `(error)`
**1. Definition:** Raised by Registry.create when a name is not registered; the message lists the available names.
**2. In plain words:** A menu that tells you what you may order.
**3. Example to quote:** > "A helpful error is part of the design."
**Seen in:** `ragkit/patterns.py`

## ValueError: ... already registered  `(error)`
**1. Definition:** Raised when the same plugin name is registered twice.
**2. In plain words:** Two tenants with the same room number.
**3. Example to quote:** > "Fail fast, otherwise import order silently decides which plugin wins."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
