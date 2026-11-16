# Terms — The query pipeline: retrieve, rerank, build the prompt, generate, verify, refuse

Folder: `20_llm_system_lld/04_query_pipeline` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Retriever (dense, sparse, hybrid) as a Strategy](#retriever-dense-sparse-hybrid-as-a-strategy--concept) · [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf--concept) · [BM25](#bm25--concept) · [Reranker (cross-encoder)](#reranker-cross-encoder--concept) · [Token budget (context packing)](#token-budget-context-packing--concept) · [Refusal policy (abstention gate)](#refusal-policy-abstention-gate--concept) · [Citation verification (lexical support)](#citation-verification-lexical-support--concept) · [Adapter pattern](#adapter-pattern--concept) · [Null Object pattern (NoopReranker)](#null-object-pattern-noopreranker--concept) · [Composite (HybridRetriever)](#composite-hybridretriever--concept) · [Lazy import](#lazy-import--syntax) · [False accept / false refusal](#false-accept--false-refusal--concept) · [AttributeError: object has no attribute 'rerank'](#attributeerror-object-has-no-attribute-rerank--error)

---

## Retriever (dense, sparse, hybrid) as a Strategy  `(concept)`
**1. Definition:** A component with `retrieve(query, k)`; dense uses embeddings, sparse uses BM25, hybrid fuses both. Swapping them is the Strategy pattern.
**2. In plain words:** Different search engines behind one search box.
**3. Example to quote:** > "Hybrid is the default because dense misses exact codes and sparse misses paraphrases."
**Seen in:** `ragkit/query.py`

## Reciprocal Rank Fusion (RRF)  `(concept)`
**1. Definition:** Merging ranked lists with score(c) = sum over lists of 1/(k + rank), k=60.
**2. In plain words:** Voting by rank position instead of by raw score.
**3. Example to quote:** > "RRF needs no score calibration between cosine and BM25; a chunk first in two lists scores 2/61 = 0.0328."
```python
fused[id] += 1.0 / (k + rank)
```
**Seen in:** `ragkit/query.py`, rrf

## BM25  `(concept)`
**1. Definition:** A keyword ranking function using term frequency (saturated by k1), inverse document frequency and length normalisation (b).
**2. In plain words:** A smarter word-count that rewards rare words.
**3. Example to quote:** > "BM25 finds exact codes like ck-masala-250 that embeddings blur."
```python
idf * tf * (k1+1) / (tf + k1*(1-b+b*len/avg))
```
**Seen in:** `ragkit/query.py`

## Reranker (cross-encoder)  `(concept)`
**1. Definition:** A second-stage model that scores a (query, passage) pair together and reorders the candidates.
**2. In plain words:** A careful reader checking the shortlist.
**3. Example to quote:** > "Its raw logit is the signal the refusal gate reads."
**Seen in:** `ragkit/adapters.py`

## Token budget (context packing)  `(concept)`
**1. Definition:** The maximum prompt tokens allotted to chunks; chunks are added best-first while they fit.
**2. In plain words:** Packing a suitcase with a weight limit.
**3. Example to quote:** > "Skip-and-continue packing fits smaller chunks after one that is too large."
```python
if total + cost <= budget: used.append(h) else: dropped.append(h)
```
**Seen in:** `ragkit/query.py`, PromptBuilder

## Refusal policy (abstention gate)  `(concept)`
**1. Definition:** A rule that returns a refusal before generation when evidence is missing or too weak.
**2. In plain words:** A librarian saying 'we have nothing on that'.
**3. Example to quote:** > "Calibrate on labelled data; the gift-card question here is a false accept at -6.5."
```python
ScoreThresholdPolicy(min_score=-6.5)
```
**Seen in:** `ragkit/query.py`

## Citation verification (lexical support)  `(concept)`
**1. Definition:** Checking each [n] is in range and that the citing sentence shares enough content words with source n.
**2. In plain words:** Checking that the footnote exists and plausibly relates to the sentence.
**3. Example to quote:** > "It shows plausibility, not faithfulness; use an NLI model or LLM judge for that."
**Seen in:** `ragkit/query.py`

## Adapter pattern  `(concept)`
**1. Definition:** A class that converts one interface into another the caller expects, without changing the wrapped class.
**2. In plain words:** A travel plug adapter.
**3. Example to quote:** > "Module14Embedder renames `encode` to `embed` so MiniLM fits the Embedder Protocol."
```python
class Module14Embedder:
    def embed(self, texts): return self._inner.encode(list(texts))
```
**Seen in:** `ragkit/adapters.py`

## Null Object pattern (NoopReranker)  `(concept)`
**1. Definition:** An object that implements an interface and does nothing, so callers need no `if x is None` branches.
**2. In plain words:** A blank placeholder card.
**3. Example to quote:** > "'No reranking' is still a valid Reranker."
```python
class NoopReranker:
    def rerank(self, q, hits, n): return list(hits)[:n]
```
**Seen in:** `ragkit/query.py`

## Composite (HybridRetriever)  `(concept)`
**1. Definition:** An object that implements an interface by delegating to several objects of the same interface.
**2. In plain words:** A committee that speaks with one voice.
**3. Example to quote:** > "HybridRetriever is a Retriever made of Retrievers."
**Seen in:** `ragkit/query.py`

## Lazy import  `(syntax)`
**1. Definition:** Importing a heavy library inside a function or constructor instead of at module top.
**2. In plain words:** Fetching a tool only when you need it.
**3. Example to quote:** > "`import ragkit` stays instant because torch loads only inside the adapters."
```python
def __init__(self):
    from common import Embedder
```
**Seen in:** `ragkit/adapters.py`

## False accept / false refusal  `(concept)`
**1. Definition:** Answering an unanswerable question (false accept) versus refusing an answerable one (false refusal).
**2. In plain words:** Two kinds of mistake a gate can make.
**3. Example to quote:** > "Threshold moves trade one for the other; measure both on a labelled set."
**Seen in:** `WALKTHROUGH.md`, Execution trace

## AttributeError: object has no attribute 'rerank'  `(error)`
**1. Definition:** Python calls the missing method only when it is used; Protocols are not enforced at runtime.
**2. In plain words:** A plug that fits visually but has no wire inside.
**3. Example to quote:** > "Assert protocol conformance in the composition root or add a contract test."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
