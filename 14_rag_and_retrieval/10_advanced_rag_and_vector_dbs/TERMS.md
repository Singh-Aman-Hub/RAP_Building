# Terms — Advanced RAG and Vector Databases

Folder: `14_rag_and_retrieval/10_advanced_rag_and_vector_dbs` · Files covered: `main.py`, `vector_db_reference.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [Query rewriting](#query-rewriting) · [Multi-query retrieval](#multi-query-retrieval) · [Step-back prompting](#step-back-prompting) · [Query decomposition](#query-decomposition) · [HyDE](#hyde) · [Small-to-big (parent-document) retrieval](#small-to-big-parent-document-retrieval) · [Sentence-window retrieval](#sentence-window-retrieval) · [Contextual retrieval](#contextual-retrieval) · [Semantic cache](#semantic-cache) · [Cache invalidation](#cache-invalidation) · [Metadata filtering (pre vs post)](#metadata-filtering-pre-vs-post) · [GraphRAG](#graphrag) · [Agentic RAG](#agentic-rag) · [Long context vs RAG](#long-context-vs-rag) · [FAISS](#faiss) · [Chroma](#chroma) · [pgvector](#pgvector) · [Dedicated vector databases](#dedicated-vector-databases) · [Re-indexing on embedding model change](#re-indexing-on-embedding-model-change) · [Multi-tenancy](#multi-tenancy) · [py_compile](#py_compile)

---

## Query rewriting  `(concept)`
**1. Definition:** Transforming the user's question, usually with an LLM, into a query better suited to the search index.
**2. In plain words:** "Can you send tea to my cousin in London?" becomes "Do you ship internationally?", and the gold chunk moves from rank 8 to rank 1.
**3. Example to quote:** > "Users and documents use different words; rewriting bridges the vocabulary gap before retrieval."
**Seen in:** `main.py`, part A

## Multi-query retrieval  `(concept)`
**1. Definition:** Generating several reformulations of a query, retrieving for each, and merging the result lists, typically with Reciprocal Rank Fusion.
**2. In plain words:** Four searches (original + three rewrites) fused with RRF gave rank 1. Cost: one LLM call plus three extra searches.
**3. Example to quote:** > "I keep the original question in the fusion as insurance against a bad rewrite."
**Seen in:** `main.py`, part A

## Step-back prompting  `(concept)`
**1. Definition:** Asking the model to form a more general, higher-level question first and retrieving for it alongside the original.
**2. In plain words:** For "Can I get CK-MONSOON-250 in March?" also search "When are seasonal blends available?".
**3. Example to quote:** > "Step-back queries fetch the general rule that a very specific question doesn't match."
**Seen in:** `WALKTHROUGH.md`, theory

## Query decomposition  `(concept)`
**1. Definition:** Splitting a complex or multi-hop question into simpler sub-questions that are retrieved and answered separately, then combined.
**2. In plain words:** "Is tulsi caffeine free and how much is it?" becomes two searches, one in allergens and one in the product catalogue.
**3. Example to quote:** > "One embedding can't point at two sections at once, so I decompose compound questions."
**Seen in:** `WALKTHROUGH.md`, theory

## HyDE  `(concept)`
**1. Definition:** Hypothetical Document Embeddings: an LLM writes a plausible answer to the question, and that text, not the question, is embedded and used for search.
**2. In plain words:** Cosine with the gold chunk went from 0.30 (question) to 0.79 (hypothetical answer), rank 8 to 1. The guess may be wrong; it is only a search key.
**3. Example to quote:** > "Answers look like documents and questions don't, so HyDE searches with a fake answer and never shows it to the user."
**Seen in:** `main.py`, part B

## Small-to-big (parent-document) retrieval  `(concept)`
**1. Definition:** Indexing small child units (such as sentences) for matching while returning their larger parent unit (such as the section) as context.
**2. In plain words:** 80 sentence vectors each point to one of 36 sections. The matched sentence was about the sealed-pouch photo, but its 40-word parent also contains "You do not need to post the tea back", which answers the question.
**3. Example to quote:** > "Small chunks give sharp vectors, big chunks give the model enough context; small-to-big gets both."
**Seen in:** `main.py`, part C

## Sentence-window retrieval  `(concept)`
**1. Definition:** Indexing individual sentences and returning each match together with a fixed number of neighbouring sentences.
**2. In plain words:** Like small-to-big, but the "parent" is a window of ±k sentences rather than a predefined section.
**3. Example to quote:** > "Sentence windows help when sections are long and the answer is in the next sentence."
**Seen in:** `WALKTHROUGH.md`, theory

## Contextual retrieval  `(concept)`
**1. Definition:** Prepending chunk-specific context (document title, section heading or an LLM-generated summary of where the chunk sits) to each chunk before embedding and keyword indexing.
**2. In plain words:** `chunk_for_embedding` builds `"Refunds and Returns - Refund window: ..."`, so a chunk that never says "refund" still carries the topic.
**3. Example to quote:** > "Chunking strips context; contextual headers put it back before indexing."
**Seen in:** `common.py` (`chunk_for_embedding`), `WALKTHROUGH.md`

## Semantic cache  `(concept)`
**1. Definition:** A cache keyed by query embeddings that returns a stored answer when a new query's similarity to a cached query exceeds a threshold.
**2. In plain words:** "What time is customer support open?" (cos 0.74) reuses the support-hours answer; "What are your delivery hours?" (0.66) misses. At threshold 0.6 it would be a wrong hit.
**3. Example to quote:** > "A semantic cache saves the whole pipeline, but the gap between paraphrase and different question was only 0.08 cosine."
**Seen in:** `main.py`, part D

## Cache invalidation  `(concept)`
**1. Definition:** Removing or expiring cached entries when the data they were derived from changes.
**2. In plain words:** If the support hours change, the cached "Monday to Saturday, 10am to 6pm IST." becomes a confident wrong answer. Store the source chunk ids with each entry, use a TTL, and scope entries by tenant and permissions.
**3. Example to quote:** > "Every cached answer records which chunks produced it, so a document update evicts it."
**Seen in:** `main.py`, part D; `WALKTHROUGH.md`

## Metadata filtering (pre vs post)  `(concept)`
**1. Definition:** Restricting search results by structured fields; pre-filtering applies the condition inside the vector search, post-filtering applies it to the top-k after search.
**2. In plain words:** Chroma's `where={"doc_id": "refunds"}`, Qdrant's `Filter(must=[...])`, pgvector's SQL `WHERE`. Post-filtering top-5 may leave zero results; pre-filtering still returns k.
**3. Example to quote:** > "ACL, tenant and freshness filters go inside the search, never after it."
**Seen in:** `vector_db_reference.py`

## GraphRAG  `(concept)`
**1. Definition:** A RAG approach that uses an LLM to extract entities and relationships into a knowledge graph, groups them into communities with summaries, and retrieves over the graph and summaries.
**2. In plain words:** Good for "What are the main themes across all documents?" or multi-hop questions no single chunk answers; expensive to build and keep current.
**3. Example to quote:** > "Chunk retrieval answers local questions; GraphRAG targets global, corpus-wide ones."
**Seen in:** `WALKTHROUGH.md`, theory

## Agentic RAG  `(concept)`
**1. Definition:** RAG in which the model treats retrieval as a tool, deciding when and what to search, possibly iterating; related ideas include Self-RAG (retrieve and self-critique) and Corrective RAG (grade retrieved documents and re-query when they are poor).
**2. In plain words:** Instead of one fixed retrieve step, the agent searches, reads, refines and searches again. More capable, harder to bound in cost and latency.
**3. Example to quote:** > "I'd move to agentic retrieval for multi-step questions, with a step limit and the same eval set."
**Seen in:** `WALKTHROUGH.md`, theory

## Long context vs RAG  `(concept)`
**1. Definition:** The design choice between putting large amounts of source text directly into a long context window and retrieving only relevant chunks.
**2. In plain words:** Long context is simple but costs tokens on every call, adds latency and suffers "lost in the middle"; RAG scales to large, changing corpora and supports citations and access control.
**3. Example to quote:** > "Long context and RAG combine: retrieve generously, then let a long-context model read it."
**Seen in:** `WALKTHROUGH.md`, theory

## FAISS  `(third-party: faiss)`
**1. Definition:** Facebook AI Similarity Search, a C++/Python library for similarity search over dense vectors with exact and approximate indexes (Flat, IVF, HNSW, PQ) on CPU or GPU.
**2. In plain words:** A library, not a database: you manage persistence (`faiss.write_index`), metadata and updates yourself. `IndexIVFPQ` must be trained on data first. Not installed here: `import faiss` raises `ModuleNotFoundError`.
**3. Example to quote:** > "FAISS is my choice for fast in-process search or batch jobs; for metadata and multi-user access I want a database."
```python
flat = faiss.IndexFlatIP(384); flat.add(vectors); scores, idx = flat.search(query, 2)
```
**Seen in:** `vector_db_reference.py`, section 1

## Chroma  `(third-party: chromadb)`
**1. Definition:** An open-source embedding database that stores documents, metadata and vectors together, runnable embedded in a Python process or as a server.
**2. In plain words:** The quickest start: `upsert` text plus metadata, `query` with a `where` filter. Its results are cosine *distances* (1 − cosine).
**3. Example to quote:** > "I prototype with Chroma and move to pgvector or a dedicated DB when operations matter."
**Seen in:** `vector_db_reference.py`, section 2

## pgvector  `(third-party: pgvector)`
**1. Definition:** A PostgreSQL extension adding a `vector` column type, distance operators (`<=>` cosine distance, `<->` L2, `<#>` negative inner product) and HNSW/IVFFlat indexes.
**2. In plain words:** Vectors live next to your normal tables: one database, SQL joins and filters, transactions, row-level security, backups you already run. `hnsw.ef_search` is the query-time recall dial.
**3. Example to quote:** > "If the team already runs Postgres, pgvector is the default; I'd benchmark before moving to a dedicated store."
**Seen in:** `vector_db_reference.py`, section 3

## Dedicated vector databases  `(concept)`
**1. Definition:** Server systems built for vector search, such as Qdrant, Milvus and Weaviate (self-hosted or cloud) and Pinecone (managed), offering filtered ANN search, sharding, replication, snapshots and often hybrid sparse + dense search.
**2. In plain words:** The reference file uses Qdrant in memory mode with a payload filter on `doc_id`. Worth it at scale; one more system to operate.
**3. Example to quote:** > "I reach for Qdrant or Milvus when collections and query rates outgrow Postgres, or I need built-in hybrid search."
**Seen in:** `vector_db_reference.py`, section 4

## Re-indexing on embedding model change  `(concept)`
**1. Definition:** Re-embedding the entire corpus and rebuilding the index whenever the embedding model or its version changes, because vectors from different models are not comparable.
**2. In plain words:** MiniLM gives 384 dims; a 768-dim model's vectors won't even fit the index, and same-size vectors from a different model give meaningless scores. Build the new index alongside, evaluate, switch, delete the old.
**3. Example to quote:** > "I store the embedding model name with the index and treat a model change as a blue-green migration."
**Seen in:** `WALKTHROUGH.md`, theory and debugging

## Multi-tenancy  `(concept)`
**1. Definition:** Serving several customers or teams from one system while keeping their data isolated, either via separate collections/namespaces or a shared collection filtered by tenant id.
**2. In plain words:** Separate collections are safer; a shared one is cheaper, but one missing `tenant_id` filter leaks data. Enforce it in server code, not the prompt.
**3. Example to quote:** > "Tenant isolation is enforced at query construction, and the semantic cache is keyed by tenant too."
**Seen in:** `WALKTHROUGH.md`, theory

## py_compile  `(standard library)`
**1. Definition:** `python -m py_compile file.py` compiles a source file to bytecode without running it, reporting syntax errors.
**2. In plain words:** It exits 0 for `vector_db_reference.py`, which proves the syntax is valid but nothing about whether the imports exist or the APIs match your installed versions.
**3. Example to quote:** > "py_compile is a syntax check, not a test; running the file here fails at `import sentence_transformers`."
**Seen in:** `vector_db_reference.py` docstring, `WALKTHROUGH.md`
