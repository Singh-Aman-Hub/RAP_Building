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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Multi-query vs query rewriting vs query expansion | "rewrite for search" | Related | Rewriting replaces the query; multi-query asks for several rewrites and fuses all results; expansion adds extra terms to one query |
| HyDE vs query rewriting | hypothetical document embeddings | No | HyDE embeds a made-up ANSWER (looks like a document); rewriting produces another QUESTION |
| Small-to-big vs parent-document retrieval vs sentence-window retrieval | child/parent chunks | Related | All match small units then return a bigger context; sentence-window returns neighbouring sentences, parent-document returns the whole section |
| Semantic cache vs exact cache vs prompt cache | answer reuse | No | Semantic: reuse an answer for a similar question (vector threshold); exact: same string; prompt cache (provider feature) reuses the model's processed prompt prefix |
| FAISS vs Chroma vs pgvector vs Qdrant/Milvus/Weaviate | vector stores | No | Library vs embedded DB vs Postgres extension vs dedicated server |
| HNSW vs IVF | ANN index types | No | Graph vs cluster-based index (step 05) |
| Pre-filter vs in-search filtering vs post-filter | metadata filter | No | Qdrant filters during HNSW search; naive post-filtering can return fewer than k |

## Code, step by step

### 1. Docstring  (`main.py` lines 1–18)
```python
"""
GOAL:  Four "advanced RAG" techniques, each shown on a question the basic pipeline gets wrong or wastes work on:
         A) multi-query rewriting + RRF   (the query's words do not match the document's words)
         B) HyDE: search with a hypothetical ANSWER instead of the question
         C) small-to-big (parent document) retrieval: match small sentences, return the whole section
         D) semantic cache: reuse an answer when a new question means the same as an old one
       The rewrites and the hypothetical answer are hand-written here; in production an LLM writes them.
       Vector databases (FAISS, Chroma, pgvector, Qdrant) are in vector_db_reference.py (read only).

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/10_advanced_rag_and_vector_dbs/main.py
EXPECTED OUTPUT (shortened):
    A) original question rank of 'International shipping': 8 -> multi-query RRF rank: 1
    B) HyDE rank: 1
    C) sentence hit -> parent section returned
    D) cache HIT for paraphrases, MISS for 'delivery hours' -- but the margin is small
NEEDS: torch, transformers (cached MiniLM).  ~10 s.
"""
```
- **In simple words:** Four tricks that fix specific retrieval problems.
- **Exactly:** A) multi-query with RRF; B) HyDE; C) small-to-big; D) semantic cache. Rewrites and the hypothetical answer are hand-written here; in production an LLM writes them. Vector database code is in `vector_db_reference.py`.
- **Why it is here:** Each trick answers one failure of the basic pipeline.

### 2. Imports and path setup  (`main.py` lines 20–27)
```python
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import BM25, Embedder, build_chunks, chunk_for_embedding, rrf_fuse   # noqa: E402
```
- **In simple words:** Load tools and the retrieval helpers.
- **Exactly:** `re`, `numpy`, and `BM25, Embedder, build_chunks, chunk_for_embedding, rrf_fuse` from `common`.
- **Why it is here:** Reuses everything built before.

### 3. Build the index and name the target chunk  (`main.py` lines 29–34)
```python
chunks = build_chunks()
texts = [chunk_for_embedding(c) for c in chunks]
ids = [c["chunk_id"] for c in chunks]
embedder, bm25 = Embedder(), BM25(texts)
matrix = embedder.encode(texts)
GOLD = "shipping_and_delivery#International shipping"
```
- **In simple words:** Embed all chunks, build BM25, and remember which chunk is the correct answer.
- **Exactly:** `matrix` shape `(36, 384)`. `GOLD` = `shipping_and_delivery#International shipping`, the chunk that answers the London question.
- **Why it is here:** Needs a known target to measure ranks.

### 4. hybrid retriever  (`main.py` lines 37–40)
```python
def hybrid(query, k=20):
    s = matrix @ embedder.encode([query])[0]
    dense = [(int(j), float(s[j])) for j in np.argsort(-s)[:k]]
    return rrf_fuse([bm25.search(query, k), dense], top_n=k)
```
- **In simple words:** Search by meaning and keywords and merge.
- **Exactly:** Dense top 20 plus BM25 top 20, RRF.
- **Why it is here:** The baseline all techniques are compared against.

### 5. rank_of  (`main.py` lines 43–45)
```python
def rank_of(ranking, gold=GOLD):
    order = [ids[j] for j, _ in ranking]
    return order.index(gold) + 1 if gold in order else None
```
- **In simple words:** Find the position of the target chunk in a ranking.
- **Exactly:** 1-based rank, or `None` if absent.
- **Why it is here:** One number to compare techniques.

### 6. A: the question and three rewrites  (`main.py` lines 51–56)
```python
question = "Can you send tea to my cousin in London?"
rewrites = [                                                   # what an LLM returns for "rewrite this for search"
    "Do you ship internationally?",
    "Shipping to the United Kingdom or outside India",
    "Delivery to other countries",
]
```
- **In simple words:** A question that shares no words with the answer, plus three alternative phrasings.
- **Exactly:** Question "Can you send tea to my cousin in London?" The chunk says "outside India". The rewrites are what an LLM would produce for "rewrite this for search".
- **Why it is here:** Vocabulary mismatch is the most common retrieval failure.

### 7. A: retrieve with each and fuse  (`main.py` lines 57–62)
```python
print("A) MULTI-QUERY")
print(f"   original question -> rank of the gold chunk: {rank_of(hybrid(question))}")
for r in rewrites:
    print(f"   rewrite {r!r:<52} -> rank {rank_of(hybrid(r))}")
fused = rrf_fuse([hybrid(question)] + [hybrid(r) for r in rewrites], top_n=20)
print(f"   all four fused with RRF -> rank {rank_of(fused)}   (cost: 1 extra LLM call + 3 extra searches)")
```
- **In simple words:** Search with the original and each rewrite, then merge the four lists.
- **Exactly:** Real: original question rank 8; each rewrite rank 1; fused with RRF rank 1. Cost: one extra LLM call and three extra searches.
- **Why it is here:** Extra searches buy recall when the user's wording is off.

### 8. B: the hypothetical answer  (`main.py` lines 67–68)
```python
hypothetical = ("We only deliver within India and do not ship tea abroad. International customers can order "
                "through a partner store.")                   # an LLM's guess; it may be WRONG, that is fine
```
- **In simple words:** A guessed answer, written as a document would be.
- **Exactly:** An LLM would produce this from the question; it can even be wrong.
- **Why it is here:** An answer looks more like a document than a question does.

### 9. B: embed both and compute ranks  (`main.py` lines 69–71)
```python
s_question = matrix @ embedder.encode([question])[0]
s_hyde = matrix @ embedder.encode([hypothetical])[0]
rank = lambda s: int(np.where(np.argsort(-s) == ids.index(GOLD))[0][0]) + 1
```
- **In simple words:** Compare the question vector and the answer vector against all chunks.
- **Exactly:** `s_question` and `s_hyde` are score arrays of shape `(36,)`. `rank` turns scores into the 1-based position of the gold chunk: `np.argsort(-s)` gives row order, `np.where(... == index)[0][0] + 1`.
- **Why it is here:** Shows the effect on one dense search.

### 10. B: results  (`main.py` lines 72–75)
```python
print("\nB) HyDE (dense only)")
print(f"   embed the question            -> rank {rank(s_question)} (cosine {s_question[ids.index(GOLD)]:.2f})")
print(f"   embed a hypothetical answer   -> rank {rank(s_hyde)} (cosine {s_hyde[ids.index(GOLD)]:.2f})")
print("   Risk: if the guess invents facts in the wrong direction it pulls the wrong chunks. Never show the guess to users.")
```
- **In simple words:** Print the two ranks.
- **Exactly:** Real: question rank 8 (cosine 0.30); hypothetical answer rank 1 (cosine 0.79). Risk: a wrong guess can pull wrong chunks; never show the guess to users.
- **Why it is here:** HyDE is cheap but can fail silently.

### 11. C: index sentences, remember the parent  (`main.py` lines 80–85)
```python
sentences, parent = [], []
for ci, c in enumerate(chunks):
    for s in re.split(r"(?<=[.!?])\s+", c["text"]):
        sentences.append(f"{c['section']}: {s}")
        parent.append(ci)                                      # child sentence -> parent chunk index
sent_matrix = embedder.encode(sentences)
```
- **In simple words:** Split every chunk into sentences, embed each, and keep which chunk it came from.
- **Exactly:** `sentences` (80 of them) get the section name as prefix; `parent[i]` is the chunk index of sentence `i`; `sent_matrix` shape `(80, 384)`.
- **Why it is here:** Small units give sharper vectors; the parent gives the model the whole rule.

### 12. C: search sentences, return the parent  (`main.py` lines 86–92)
```python
q = "Do I need to send the pouch back to get my money?"
best = int(np.argmax(sent_matrix @ embedder.encode([q])[0]))
print(f"\nC) SMALL-TO-BIG: {len(sentences)} sentence vectors, each pointing to one of {len(chunks)} parent sections")
print(f"   Q: {q}")
print(f"   best child sentence : {sentences[best]!r}")
print(f"   parent sent to LLM  : {ids[parent[best]]} ({len(chunks[parent[best]]['text'].split())} words)")
print("   Small chunks embed ONE idea (sharp vectors); big chunks give the LLM the surrounding conditions.")
```
- **In simple words:** Match the question to a sentence, then hand over the whole section.
- **Exactly:** `argmax(sent_matrix @ query_vec)`; real best child: "How to request a refund: Open the order in your account, choose "Request refund", and upload one photo of the sealed pouch."; parent returned: `refunds_and_returns#How to request a refund` (40 words).
- **Why it is here:** Without the parent the LLM would miss conditions such as "orders under 500 rupees".

### 13. D: semantic cache function  (`main.py` lines 97–106)
```python
cache = []                                                     # list of (unit vector, answer)


def cached_answer(question, threshold=0.70):
    v = embedder.encode([question])[0]
    for vec, ans in cache:
        sim = float(vec @ v)
        if sim >= threshold:
            return ans, sim
    return None, max((float(vec @ v) for vec, _ in cache), default=0.0)
```
- **In simple words:** Look for a stored question that means the same as the new one.
- **Exactly:** `cache` holds `(unit vector, answer)`. Compare the new vector with each by dot product (cosine); a hit if `>= threshold=0.70`. A miss returns `None` and the best similarity seen.
- **Why it is here:** Saves an LLM call and its latency for repeated questions.

### 14. D: try it on three questions  (`main.py` lines 109–115)
```python
cache.append((embedder.encode(["What are your customer support hours?"])[0], "Monday to Saturday, 10am to 6pm IST."))
print("\nD) SEMANTIC CACHE (threshold 0.70)")
for q in ["What are your customer support hours please?", "What time is customer support open?", "What are your delivery hours?"]:
    ans, sim = cached_answer(q)
    print(f"   {q!r:<46} cos {sim:.2f} -> {'HIT: ' + ans if ans else 'MISS: run the full pipeline'}")
print("   The gap between a true paraphrase and a different question is small (compare the last two cosines).")
print("   Too low a threshold returns the WRONG cached answer; invalidate entries when the documents change.")
```
- **In simple words:** Store one answer, then ask two paraphrases and one different question.
- **Exactly:** Real: "What are your customer support hours please?" cos 0.97 HIT; "What time is customer support open?" cos 0.74 HIT; "What are your delivery hours?" cos 0.66 MISS. The margin between a paraphrase (0.74) and a different question (0.66) is small.
- **Why it is here:** A threshold that is too low returns the wrong cached answer; invalidate entries when documents change.

### 15. Exercises as comments  (`main.py` lines 117–121)
```python
# TRY IT: remove the original question from the RRF list in A (only rewrites). Does the rank change?
# TRY IT: write a deliberately WRONG hypothetical answer ("We ship to London in 5 days.") for HyDE. What rank now?
# TRY IT: in D set threshold=0.6 and ask "What are your delivery hours?" -> it now returns the support hours: a wrong hit.
# BREAK IT: in C, append to `sentences` but forget `parent.append(ci)` -> the lists drift apart and the wrong
#           parent is returned (or IndexError). Keep child -> parent mappings in ONE structure.
```
- **In simple words:** Ideas to try and break.
- **Exactly:** BREAK IT: forgetting `parent.append(ci)` makes the two lists drift apart.
- **Why it is here:** Keep child to parent links in one structure.

### 16. Reference file docstring (read, not run)  (`vector_db_reference.py` lines 1–19)
```python
"""
REFERENCE ONLY -- NOT EXECUTED IN THIS STUDY PACK.

The same "add vectors, search top-k, filter by metadata" job you built by hand in steps 04 and 05,
done by four real tools.  None of them is installed in this repo's .venv; read the code, then try one in a
notebook:   pip install faiss-cpu chromadb qdrant-client psycopg[binary] pgvector sentence-transformers

Check syntax without installing anything:   python -m py_compile vector_db_reference.py

Which one when?
    FAISS     library (not a server). Fastest in-process search, many index types (Flat, IVF, HNSW, PQ).
              You manage persistence, metadata and updates yourself.
    Chroma    embedded/local database. Easiest start: stores text + metadata + vectors, filters with `where`.
    pgvector  extension for PostgreSQL. Vectors live next to your normal tables: one database, SQL joins,
              transactions, backups you already have.  HNSW and IVFFlat indexes.
    Qdrant /  dedicated vector databases (servers). Horizontal scaling, payload filtering during HNSW search,
    Milvus /  replication, snapshots, hybrid (sparse + dense) search built in.  Pinecone is the hosted kind.
    Weaviate
"""
```
- **In simple words:** Introduces four real vector tools and when to use each.
- **Exactly:** This file needs `faiss`, `chromadb`, `psycopg`/`pgvector`, none installed in this repo's `.venv` (`qdrant_client` is installed and used in step 11). Check syntax with `python -m py_compile vector_db_reference.py`. FAISS = library; Chroma = embedded DB; pgvector = Postgres extension; Qdrant/Milvus/Weaviate = dedicated servers; Pinecone = hosted.
- **Why it is here:** Choosing a store is a design decision, so the comparison is here.

### 17. Reference: data and embeddings  (`vector_db_reference.py` lines 21–31)
```python
import numpy as np

texts = ["Refunds are issued within 9 days.", "Shipping is free above 799 rupees.", "Support is closed on Sundays."]
metas = [{"doc_id": "refunds"}, {"doc_id": "shipping"}, {"doc_id": "support"}]

# Embeddings: the same MiniLM model as the rest of the module, via the sentence-transformers library.
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode(texts, normalize_embeddings=True).astype(np.float32)        # (3, 384)
query = model.encode(["How long do refunds take?"], normalize_embeddings=True).astype(np.float32)
```
- **In simple words:** Three tiny texts, their facts, and their vectors made with the sentence-transformers library.
- **Exactly:** `SentenceTransformer(...).encode(texts, normalize_embeddings=True)` returns `(3, 384)` unit vectors, cast to `float32`; the query has shape `(1, 384)`.
- **Why it is here:** Same model as the rest of the module, so results are comparable.

### 18. Reference FAISS: exact flat index  (`vector_db_reference.py` lines 36–41)
```python
import faiss

flat = faiss.IndexFlatIP(384)                    # IP = inner product = cosine on unit vectors (step 04)
flat.add(vectors)
scores, idx = flat.search(query, 2)              # both arrays have shape (n_queries, k)
print("FAISS flat:", [texts[i] for i in idx[0]], scores[0])
```
- **In simple words:** Exact search with FAISS.
- **Exactly:** `IndexFlatIP(384)` stores vectors and compares by inner product (equal to cosine on unit vectors); `search(query, 2)` returns scores and ids both of shape `(n_queries, k)`.
- **Why it is here:** The FAISS version of `FlatIndex` from step 04.

### 19. Reference FAISS: HNSW  (`vector_db_reference.py` lines 43–45)
```python
hnsw = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)   # M = 32 edges per node (step 05 B)
hnsw.hnsw.efSearch = 64                          # ef: the recall/speed dial at query time
hnsw.add(vectors)
```
- **In simple words:** An approximate graph index.
- **Exactly:** `IndexHNSWFlat(384, 32, METRIC_INNER_PRODUCT)` with `M=32` edges per node; `efSearch=64` is the query-time recall versus speed dial.
- **Why it is here:** Step 05 part B in one line.

### 20. Reference FAISS: IVF-PQ shape and persistence  (`vector_db_reference.py` lines 47–52)
```python
# IVF + PQ needs TRAINING data (k-means for cells and codebooks, step 05 A and C). With 3 vectors it cannot
# train, so this is the shape of the code for a real corpus of 100k+ vectors:
#   quantizer = faiss.IndexFlatIP(384)
#   ivfpq = faiss.IndexIVFPQ(quantizer, 384, 1024, 48, 8)   # nlist=1024 cells, 48 sub-vectors, 8 bits each
#   ivfpq.train(big_matrix); ivfpq.add(big_matrix); ivfpq.nprobe = 16
faiss.write_index(flat, "kb.faiss")              # persistence is a file; metadata goes in your own JSON/DB
```
- **In simple words:** Notes for large corpora and saving the index.
- **Exactly:** IVF-PQ needs training data (k-means for cells and codebooks) so it cannot train on 3 vectors; the comments show `nlist=1024`, 48 sub-vectors of 8 bits, `nprobe=16`. `write_index` saves to a file; metadata is your own responsibility.
- **Why it is here:** FAISS gives no metadata or filtering, you manage those.

### 21. Reference Chroma: embedded store with filter  (`vector_db_reference.py` lines 57–63)
```python
import chromadb

client = chromadb.PersistentClient(path="chroma_db")
col = client.get_or_create_collection("kb", metadata={"hnsw:space": "cosine"})
col.upsert(ids=["r1", "s1", "c1"], documents=texts, metadatas=metas, embeddings=vectors.tolist())
res = col.query(query_embeddings=query.tolist(), n_results=2, where={"doc_id": "refunds"})   # pre-filter (step 04)
print("Chroma:", res["documents"][0], res["distances"][0])
```
- **In simple words:** A local database with text, metadata and vectors together.
- **Exactly:** `PersistentClient(path="chroma_db")` stores on disk; `get_or_create_collection(..., metadata={"hnsw:space": "cosine"})`; `upsert` by ids; `query(..., where={"doc_id": "refunds"})` filters by metadata.
- **Why it is here:** Easiest start for a prototype.

### 22. Reference pgvector: SQL  (`vector_db_reference.py` lines 68–88)
```python
PGVECTOR_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE chunks (
    id        bigserial PRIMARY KEY,
    doc_id    text NOT NULL,
    content   text NOT NULL,
    embedding vector(384) NOT NULL,
    updated   timestamptz DEFAULT now()
);
-- HNSW index for cosine distance; m = edges per node, ef_construction = build-time beam width
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
SET hnsw.ef_search = 40;                         -- query-time recall/speed dial

-- top-5 by cosine distance (<=>), filtered by metadata in plain SQL
SELECT doc_id, content, 1 - (embedding <=> %(q)s::vector) AS cosine
FROM chunks
WHERE doc_id = 'refunds'
ORDER BY embedding <=> %(q)s::vector
LIMIT 5;
"""
# Operators: <=> cosine distance, <-> L2 distance, <#> negative inner product.
```
- **In simple words:** Vectors as a column in a normal PostgreSQL table.
- **Exactly:** `vector(384)` column; HNSW index `vector_cosine_ops` with `m = 16`, `ef_construction = 64`; `hnsw.ef_search = 40` at query time; `<=>` is cosine distance, `<->` L2, `<#>` negative inner product; `WHERE` filters in plain SQL; `1 - distance` gives cosine similarity.
- **Why it is here:** One database, joins, transactions and backups you already have.

### 23. Reference Qdrant: in-memory client  (`vector_db_reference.py` lines 93–102)
```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

qdrant = QdrantClient(":memory:")                # production: QdrantClient(url="http://localhost:6333")
qdrant.create_collection("kb", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
qdrant.upsert("kb", points=[PointStruct(id=i, vector=v.tolist(), payload={**m, "text": t})
                            for i, (v, m, t) in enumerate(zip(vectors, metas, texts))])
hits = qdrant.query_points("kb", query=query[0].tolist(), limit=2,
                           query_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value="refunds"))]))
print("Qdrant:", [(h.payload["text"], round(h.score, 3)) for h in hits.points])
```
- **In simple words:** A real vector database used from Python.
- **Exactly:** `QdrantClient(":memory:")` needs no server; `VectorParams(size=384, distance=Distance.COSINE)`; points carry `id`, `vector` and a `payload` (metadata plus text); `query_points(..., query_filter=Filter(must=[FieldCondition(...)]))` filters during the search. Step 11 runs this for real.
- **Why it is here:** The dedicated-server option with filtering inside the index.


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
