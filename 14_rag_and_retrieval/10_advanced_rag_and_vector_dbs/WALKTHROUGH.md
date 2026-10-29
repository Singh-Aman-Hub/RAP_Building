# Step 10 — Advanced RAG and vector databases

## The story so far
[Step 09](../09_rag_evaluation/WALKTHROUGH.md) measured the pipeline and pointed at specific gaps: London is missed because the question's words ("cousin in London") do not match the document's ("international shipping"), generation picks the wrong sentence of the right chunk, and the "index" is still a NumPy matrix in memory.

## The pain point
The basic pipeline embeds the user's question exactly as typed and searches fixed-size sections. Users write short, vague, colloquial questions; documents are written in policy language. Chunks that are big enough to give context are too blurry to match precisely. Repeated questions pay for the full pipeline every time. And a matrix in RAM has no persistence, metadata filters, updates or replication.

## The spark
Change what you search *with* (rewrites, a hypothetical answer), what you search *over* (small sentences that point to big parents), and what you can *skip* (a semantic cache). Then hand storage and search to a real vector database, chosen for your scale and operations.

## Goal
Run four advanced techniques on questions the basic pipeline handles badly, see what each buys and what it risks, and read the code for FAISS, Chroma, pgvector and Qdrant. `vector_db_reference.py` is **reference only**: those libraries are not installed here, so it is compile-checked, not executed.

## The theory (read this before the interview)
**Query transformations** (change the query before retrieval):
- *Rewriting*: an LLM turns a chatty question into a search query ("cousin in London" -> "international shipping").
- *Multi-query*: generate several rewrites, retrieve for each, fuse the rankings (RRF). Raises recall for vague questions; costs one LLM call plus N searches.
- *Step-back prompting*: also ask a more general question ("What is the shipping policy?") and retrieve for both.
- *Decomposition*: split a multi-part question ("Is the tulsi tea caffeine free and how much is it?") into sub-questions, retrieve for each, answer from the union.

**HyDE (Hypothetical Document Embeddings).** Ask an LLM to write a plausible *answer*, embed that, and search with it. An answer looks like a document; a question does not. The hypothetical may be factually wrong; it is only a search key and must never be shown to users. Risk: a guess in the wrong direction pulls the wrong chunks. Extra cost: one LLM call before retrieval.

**Chunk granularity tricks.**
- *Small-to-big / parent-document retrieval*: index small units (sentences) for sharp matching; return the parent section to the LLM for full context.
- *Sentence-window retrieval*: index sentences; return the matched sentence plus k neighbours on each side.
- *Contextual retrieval / contextual headers*: prepend to each chunk the context it lost when split (document title, section, or an LLM-written one-line summary) before embedding and BM25 indexing. This module's `chunk_for_embedding` already does the simple version: `"title - section: text"`.

**Semantic caching.** Store (question embedding, answer); if a new question's cosine similarity to a cached one exceeds a threshold, return the cached answer and skip retrieval and generation. Risks: paraphrases and different questions can be close (0.74 vs 0.66 here), so a loose threshold returns the *wrong* answer; answers depend on user permissions and personal data, so scope the cache per tenant/permission and never cache personal answers globally; invalidate entries when the source documents change (store which chunk ids each answer used, plus a TTL).

**Metadata filtering and freshness.** Store fields such as `doc_id`, product, region, language, `updated_at`, ACL. Filter during search ("only refunds docs", "only documents this user can read", "only the current policy version"). *Pre-filtering* (filter inside the ANN search) keeps k results; *post-filtering* (filter after top-k) can leave you with too few. Freshness: prefer or require recent versions, delete or tombstone superseded chunks, re-index on document change.

**GraphRAG.** An LLM extracts entities and relationships from all documents into a graph, clusters it into communities and writes community summaries. Good for *global* questions ("What are the main complaint themes across all tickets?") and multi-hop questions that no single chunk answers. Costs: an expensive indexing pass, and the graph must be kept up to date.

**Agentic RAG.** Retrieval becomes a *tool* the model calls when it decides it needs information, possibly several times: search, read, refine the query, search again (iterative retrieval). Related research ideas: *Self-RAG* (a model trained to decide when to retrieve and to critique its own output against the passages) and *Corrective RAG* (a grader judges retrieved documents and, if they are poor, triggers rewriting or another source such as web search). More flexible, harder to bound in latency and cost, and needs the same evaluation (module [08 agent harness](../../08_agent_harness)).

**Long context vs RAG.** Large context windows let you paste whole documents, but you pay per token on every call, latency grows, attention degrades in the middle of long inputs, and the corpus may be far larger than any window or change daily. RAG also gives citations and per-user access control. In practice they combine: retrieve more generously and let a long-context model read it; prompt caching helps when the same large context is reused.

**Vector DB landscape.**
| Tool | What it is | Use it when | Watch out for |
|---|---|---|---|
| FAISS | in-process **library** (Meta), CPU/GPU, Flat, IVF, HNSW, PQ indexes | you want maximum speed in one process, batch jobs, research, you will build storage around it | no server, no metadata store, persistence is a file you manage, updates and deletes are awkward for some index types |
| Chroma | embedded / local database storing text + metadata + vectors | prototypes, notebooks, small apps, fastest start | scaling and operations for large multi-user production |
| pgvector | PostgreSQL extension, HNSW and IVFFlat indexes, distance operators in SQL | you already run Postgres; want joins, transactions, row-level security and existing backups | very large or very high-QPS workloads; benchmark recall and latency with your filters |
| Qdrant / Milvus / Weaviate | dedicated vector database servers | large collections, filtered HNSW search, sharding and replication, built-in hybrid (sparse + dense) | another system to run, monitor and back up |
| Pinecone (and other managed services) | fully hosted vector DB | no ops team, need elastic scale quickly | cost at scale, data residency, vendor lock-in |

Search engines such as Elasticsearch and OpenSearch also offer BM25 plus k-NN in one place, which suits teams that already run them.

**Operational concerns.**
- *Embedding model change = full rebuild.* Vectors from different models (or versions) live in different spaces and cannot be compared. Re-embed every chunk, build a new index alongside the old one, switch traffic after evaluation (blue-green), then delete the old one. Store the model name with each index.
- *Multi-tenancy.* Either one collection/namespace per tenant (strong isolation, many small indexes) or one shared collection filtered by a `tenant_id` payload (efficient, but a missing filter leaks data). Enforce the filter in server code, never in the prompt.
- *Backups and recovery.* Snapshot the vector store **and** the source documents and chunking config, so you can rebuild if the snapshot fails.
- *Cost and memory.* A float32 384-dim vector is 384 × 4 = 1,536 bytes, so 1M chunks is about 1.5 GB before graph links and metadata. Quantisation (int8, PQ) cuts memory at some recall cost. Re-embedding the whole corpus costs compute or API money every time the model changes.

## Run it
```bash
# from the repo root (~5-10 s, offline, cached MiniLM only)
.venv/bin/python 14_rag_and_retrieval/10_advanced_rag_and_vector_dbs/main.py
# reference file: syntax check only (faiss, chromadb, qdrant-client, sentence-transformers are not installed)
.venv/bin/python -m py_compile 14_rag_and_retrieval/10_advanced_rag_and_vector_dbs/vector_db_reference.py
# to actually try it, in a separate environment:
#   pip install faiss-cpu chromadb qdrant-client "psycopg[binary]" pgvector sentence-transformers
```
In `main.py` the rewrites (A) and the hypothetical answer (B) are hand-written; in production an LLM writes them.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `rrf_fuse([r0, r1, r2, r3], top_n=20)` | fuse any number of rankings with Reciprocal Rank Fusion | multi-query |
| `np.where(np.argsort(-s) == i)[0][0] + 1` | 1-based rank of index `i` in a descending sort | HyDE rank |
| `re.split(r"(?<=[.!?])\s+", text)` | split into sentences, keep the punctuation | small-to-big |
| `sentences.append(...); parent.append(ci)` | two parallel lists: child sentence and its parent chunk index | |
| `int(np.argmax(sent_matrix @ q))` | index of the best-matching sentence | |
| `max((... for vec, _ in cache), default=0.0)` | `max` over a generator with a fallback for an empty cache | |
| `def f(q, threshold=0.70)` | default argument | semantic cache |
| `PGVECTOR_SQL = """..."""` | SQL kept as a string (reference file) | `<=>` cosine distance |
| `Filter(must=[FieldCondition(...)])` | Qdrant payload filter (reference file) | `doc_id == "refunds"` |

## Libraries & functions
| Name | Built-in / 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `Embedder().encode` | `common.py` | MiniLM unit vectors | list of texts | `(n, 384)` |
| `BM25`, `rrf_fuse` | `common.py` | sparse search and rank fusion | `k`, `top_n` | list `(index, score)` |
| `faiss.IndexFlatIP(d)` | faiss (reference) | exact inner-product search | dim | index |
| `faiss.IndexHNSWFlat(d, M, metric)` | faiss (reference) | HNSW graph index; `.hnsw.efSearch` is the query-time dial | `M=32` | index |
| `faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)` | faiss (reference) | IVF cells + product quantisation; needs `.train()` | `nlist`, `nprobe` | index |
| `faiss.write_index` | faiss (reference) | persist an index to a file | path | – |
| `chromadb.PersistentClient(path)` | chromadb (reference) | local on-disk database | `path` | client |
| `col.upsert(...)`, `col.query(..., where=...)` | chromadb (reference) | insert/update, search with metadata filter | `n_results`, `where` | dict |
| `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)` | pgvector SQL (reference) | HNSW index in Postgres | `m`, `ef_construction` | – |
| `QdrantClient(":memory:")`, `.query_points(...)` | qdrant-client (reference) | vector DB client, filtered search | `limit`, `query_filter` | result |
| `SentenceTransformer(name).encode` | sentence-transformers (reference) | embeddings in one call | `normalize_embeddings=True` | ndarray |

## Execution trace (real run)
Deterministic numbers; timings vary.
```
A) MULTI-QUERY
   original question -> rank of the gold chunk: 8
   rewrite 'Do you ship internationally?'                       -> rank 1
   rewrite 'Shipping to the United Kingdom or outside India'    -> rank 1
   rewrite 'Delivery to other countries'                        -> rank 1
   all four fused with RRF -> rank 1   (cost: 1 extra LLM call + 3 extra searches)

B) HyDE (dense only)
   embed the question            -> rank 8 (cosine 0.30)
   embed a hypothetical answer   -> rank 1 (cosine 0.79)

C) SMALL-TO-BIG: 80 sentence vectors, each pointing to one of 36 parent sections
   Q: Do I need to send the pouch back to get my money?
   best child sentence : 'How to request a refund: Open the order in your account, choose "Request refund", and upload one photo of the sealed pouch.'
   parent sent to LLM  : refunds_and_returns#How to request a refund (40 words)

D) SEMANTIC CACHE (threshold 0.70)
   'What are your customer support hours please?' cos 0.97 -> HIT: Monday to Saturday, 10am to 6pm IST.
   'What time is customer support open?'          cos 0.74 -> HIT: Monday to Saturday, 10am to 6pm IST.
   'What are your delivery hours?'                cos 0.66 -> MISS: run the full pipeline
```
Reference file: `python -m py_compile vector_db_reference.py` exits 0 (syntax is valid).

What to read from it:
- **A:** the London question fixed by vocabulary alone: rank 8 -> 1. Each rewrite already gets rank 1; fusing keeps the original question in the mix as insurance.
- **B:** cosine with the gold chunk rises from 0.30 to 0.79 because the hypothetical answer uses the document's vocabulary ("deliver within India", "abroad").
- **C:** the best-matching sentence is the "sealed pouch" sentence, not the "You do not need to post the tea back ..." sentence that actually answers. Small-to-big still rescues it: the whole 40-word parent section, which contains the answer, goes to the LLM.
- **D:** a true paraphrase scores 0.74, a different question 0.66. The safety margin is 0.08 of cosine. A threshold of 0.6 would serve support hours as the answer to delivery hours.

## Experiments (change ONE thing)
1. In A, fuse only the three rewrites (drop the original question). Does the rank change? When would dropping the original hurt?
2. In B, use a deliberately wrong hypothetical ("We ship to London in 5 days."). Predict, then check, whether the shipping section still wins: wrong facts in the right vocabulary often still retrieve well.
3. In D, set `threshold=0.6`: "What are your delivery hours?" (cos 0.66) becomes a HIT and returns the support hours, a wrong answer served instantly.
4. In C, return the matched sentence plus one neighbour (sentence window) instead of the parent. Is the "post the tea back" sentence included?
5. Reference file: in a separate environment with the libraries installed, run it and compare FAISS flat scores with Chroma distances (Chroma reports cosine *distance* = 1 − cosine).

## Debugging practice
**BREAK IT, main.py (confirmed in a scratch copy):** in C, append to `sentences` but forget `parent.append(ci)`.
```
C) SMALL-TO-BIG: 80 sentence vectors, each pointing to one of 36 parent sections
   ...
Traceback (most recent call last):
  File ".../main.py", line 91, in <module>
    print(f"   parent sent to LLM  : {ids[parent[best]]} ({len(chunks[parent[best]]['text'].split())} words)")
                                          ~~~~~~^^^^^^
IndexError: list index out of range
```
Here `parent` is empty, so it crashes. If only *some* appends were missed (for example skipping empty sentences in one list but not the other), the lists would drift and the code would silently return the wrong parent. Fix: store the mapping in one structure, e.g. `children.append((sentence, parent_index))`, and assert `len(sentences) == len(parent)`.

**BREAK IT, reference file (confirmed):** the libraries are not installed in this repo's `.venv`.
```
$ .venv/bin/python -c "import faiss"
ModuleNotFoundError: No module named 'faiss'
$ .venv/bin/python 14_rag_and_retrieval/10_advanced_rag_and_vector_dbs/vector_db_reference.py
  File ".../vector_db_reference.py", line 27, in <module>
    from sentence_transformers import SentenceTransformer
ModuleNotFoundError: No module named 'sentence_transformers'
```
Running the file fails at the first missing import, `sentence_transformers`, before reaching FAISS; `chromadb`, `qdrant_client`, `pgvector` and `psycopg` are also missing. That is expected: the file is for reading. `py_compile` only proves the syntax is valid, not that the API calls are right for your installed versions. The common real-world error is a *dimension mismatch*: adding 768-dim vectors to an index created for 384 (FAISS asserts on `d`, pgvector rejects the insert for `vector(384)`), which is what happens if you change the embedding model without rebuilding (expected behaviour, not run here).

## Rebuild checklist
- [ ] multi-query: N rewrites -> retrieve each -> RRF with the original
- [ ] HyDE: embed a hypothetical answer, search with it, never show it
- [ ] small-to-big: index sentences, keep child -> parent mapping in one structure, return the parent
- [ ] semantic cache: (vector, answer, source chunk ids, tenant); threshold from labelled paraphrase/non-paraphrase pairs; invalidate on document change
- [ ] metadata filters (ACL, tenant, freshness) inside the search, not after
- [ ] choose a store: FAISS (library), Chroma (prototype), pgvector (already on Postgres), Qdrant/Milvus/Weaviate/Pinecone (scale)
- [ ] embedding model change -> re-embed, new index, evaluate, switch; back up documents and config as well as vectors

## The new problem (cliffhanger)
That completes the module: from "why RAG" to an evaluated pipeline with advanced retrieval and a real storage choice. Consolidate it with the [module summary](../README.md) and the [interview gist](../INTERVIEW_GIST.md). The next real-world steps: put the pipeline behind an HTTP API with validation, streaming and error handling ([09 API development](../../09_api_development)), and let an agent call retrieval as a tool, decide when to search again and act on the results ([08 agent harness](../../08_agent_harness)). Each new feature should arrive with new questions in the golden set from step 09.

> **Say it aloud (interview box):** When basic RAG misses, I first change the query: multi-query rewriting or HyDE fixed a question here from rank 8 to rank 1, and I index small units like sentences but return the parent section so the model gets full context. I add a semantic cache only with a threshold calibrated on paraphrase pairs and scoped by tenant, because a true paraphrase scored 0.74 and a different question 0.66, and I keep metadata filters for access control and freshness inside the search. For storage I pick FAISS as a library, pgvector if we already run Postgres, or a dedicated database like Qdrant at scale, and I plan for full re-indexing whenever the embedding model changes.
