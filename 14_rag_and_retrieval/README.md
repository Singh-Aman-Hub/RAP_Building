# Module 14 — RAG and Retrieval

Retrieval-Augmented Generation end to end: loading and chunking documents, embeddings, vector indexes (exact and approximate), BM25 and hybrid search, re-ranking, the full pipeline with citations and refusals, evaluation, and advanced techniques plus real vector databases.

## The story of this module
This is a deep dive into the 20-line demo at the end of Act IV, [07_prompt_engineering/08_minimal_rag](../07_prompt_engineering/08_minimal_rag/WALKTHROUGH.md).
- **Pain point:** a frozen model knows nothing about your private documents or anything after its training cutoff. It still answers confidently, so it hallucinates. Retraining for every new fact is slow and expensive, and pasting every document into every prompt does not fit and costs too much.
- **Spark:** keep the facts outside the model. Index them once. For each question, retrieve only the few relevant pieces, put them in the prompt with source numbers, and make the model answer from them, cite them, or say "I don't know".
- **What you build:** a knowledge base of 9 documents (a tea shop called Chai Kitchen), five chunkers, a bi-encoder, a flat vector index with filters and upserts, IVF / graph / PQ approximate search from scratch, BM25 by hand, hybrid fusion, a cross-encoder re-ranker, a complete `RAGPipeline` class, an evaluation harness, and four advanced techniques.
- **Cliffhanger:** the pipeline answers questions. To serve it, put it behind an API ([09_api_development](../09_api_development/)), or make retrieval a tool an agent decides to call ([08_agent_harness](../08_agent_harness/)).

**Short on time?** Read [INTERVIEW_GIST.md](INTERVIEW_GIST.md) first. It is the whole module's theory on one page, with a quick-fire Q&A.

## 1. Why this matters
"How would you build a chatbot over our documents?" is one of the most common ML-engineering interview questions. Strong answers name the stages (ingest, chunk, embed, index, retrieve, re-rank, generate, evaluate). They also name the trade-off at each stage and how to *measure* it. Every number in this module comes from a real run on a laptop CPU, and the same reasoning applies to millions of documents.

## 2. Where this fits
**Builds on**
- [04_llm_fundamentals/02_vectors_and_similarity](../04_llm_fundamentals/02_vectors_and_similarity/) and [03_embeddings](../04_llm_fundamentals/03_embeddings/): dot product, cosine, embeddings. Step 03 here turns them into sentence embeddings.
- [04_llm_fundamentals/01_tokenizer](../04_llm_fundamentals/01_tokenizer/): tokens, which are the unit of context windows, truncation and cost (steps 01, 02).
- [05_llm_parameters/04_max_new_tokens_and_context_window](../05_llm_parameters/04_max_new_tokens_and_context_window/): why you cannot paste everything in (step 01).
- [06_llm_training_and_lora](../06_llm_training_and_lora/): fine-tuning changes *behaviour*, while RAG adds *knowledge*. PQ in step 05 is quantisation again.
- [07_prompt_engineering](../07_prompt_engineering/): templates, structured output, prompt injection (steps 08, 09).

**Leads to**
- [08_agent_harness](../08_agent_harness/): retrieval as a tool the agent calls (agentic RAG, step 10).
- [09_api_development](../09_api_development/): load the index and models once at startup, then expose `POST /ask`.
- [10_ml_system_architecture](../10_ml_system_architecture/): offline indexing pipeline vs online query pipeline, versioned indexes, prediction logs.
- [13_capstone_project](../13_capstone_project/): a natural extension is "chat with the docs".

## 3. Concept map
```
OFFLINE (indexing, run when documents change)
  raw docs (md/html/pdf) ──load+clean (01)──> text + metadata ──chunk (02)──> chunks
        ──embed with a bi-encoder (03)──> vectors ──index (04 exact / 05 IVF, HNSW, PQ)──> vector store
        ──tokenise──> BM25 inverted index (06)

ONLINE (per question)
  question ──embed──> dense top-20 ─┐
           ──BM25───> sparse top-20 ─┴─RRF fuse (06)──> 20 candidates ──cross-encoder re-rank (07)──>
  best score < threshold? ──yes──> "I don't know"
           │ no
           v
  top-k chunks packed into a token budget ──prompt with numbered sources + rules (08)──> LLM ──> answer [1][2]
           └──> check citations, log, evaluate (09): hit@k, MRR, nDCG, faithfulness, refusals
  improvements (10): multi-query, HyDE, small-to-big, semantic cache, real vector DBs
```

## 4. Study path
Shared helpers used by every step are walked line by line once in [COMMON_WALKTHROUGH.md](COMMON_WALKTHROUGH.md); step 11 (Qdrant fundamentals) has its own [WALKTHROUGH.md](11_qdrant_fundamentals/WALKTHROUGH.md).

| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_documents_and_why_rag](01_documents_and_why_rag/) | why RAG, loading, cleaning, metadata, token arithmetic | `python main.py` | 40 min |
| 2 | [02_chunking_strategies](02_chunking_strategies/) | fixed / overlap / sentence / recursive / heading-aware chunking and their trade-offs | `python main.py` | 50 min |
| 3 | [03_embeddings_bi_encoder](03_embeddings_bi_encoder/) | mean pooling, normalisation, dense vs TF-IDF, truncation, query/passage prefixes | `python main.py` | 60 min |
| 4 | [04_vector_index_and_search](04_vector_index_and_search/) | a flat vector store: top-k, metadata filters, upsert/delete, save/load, scaling | `python main.py` | 50 min |
| 5 | [05_ann_ivf_and_hnsw](05_ann_ivf_and_hnsw/) | approximate search from scratch: IVF (nprobe), graph search (HNSW idea), PQ, recall@10 | `python main.py` | 60 min |
| 6 | [06_bm25_and_hybrid_search](06_bm25_and_hybrid_search/) | BM25 by hand, sparse vs dense by question type, RRF and weighted fusion | `python main.py` | 60 min |
| 7 | [07_reranking_cross_encoder](07_reranking_cross_encoder/) | two-stage retrieval, cross-encoder vs bi-encoder, latency, abstention signal | `python main.py` | 45 min |
| 8 | [08_rag_pipeline_end_to_end](08_rag_pipeline_end_to_end/) | the full pipeline: refusal, token budget, numbered sources, citations, prompt injection | `python main.py` | 75 min |
| 9 | [09_rag_evaluation](09_rag_evaluation/) | hit@k, recall@k, MRR, nDCG, refusal calibration, faithfulness, error analysis, LLM-as-judge | `python main.py` | 60 min |
| 10 | [10_advanced_rag_and_vector_dbs](10_advanced_rag_and_vector_dbs/) | multi-query, HyDE, small-to-big, semantic cache; FAISS / Chroma / pgvector / Qdrant (read only) | `python main.py` | 60 min |

Total is about 9–10 hours including exercises. Each step folder has `main.py` (commented, runnable), `WALKTHROUGH.md` (story, theory, syntax, real output, experiments, debugging drill) and `TERMS.md` (every term: definition, plain words, a sentence to quote). Then do [exercises.md](exercises.md).

Run from the repo root with `.venv/bin/python 14_rag_and_retrieval/<step>/main.py`, or `cd` into the step and run `python main.py`. Everything runs offline on a CPU. It uses the cached models `sentence-transformers/all-MiniLM-L6-v2`, `intfloat/multilingual-e5-small` (step 03) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (steps 07–10). The generator in step 08 is a transparent rule-based **simulator** by default. `LLM_BACKEND=local` or `LLM_BACKEND=anthropic` swaps in a real model (see [shared/llm_client.py](../shared/llm_client.py)).

**Shared files:** [common.py](common.py) holds the finished helpers once a step has taught them (`load_documents`, `split_by_heading`, `Embedder`, `dense_search`, `BM25`, `rrf_fuse`, `CrossEncoderReranker`). The data lives in [data/kb/](data/kb/), and [data/eval_questions.json](data/eval_questions.json) has 29 labelled questions (24 answerable: keyword, paraphrase or exact-code; 5 unanswerable).

## 5. The concepts, one by one

### RAG vs fine-tuning vs long context (step 01)
| | RAG | Fine-tuning (module 06) | Long context |
|---|---|---|---|
| Adds | knowledge (facts, documents) | behaviour, style, format | knowledge, for one request |
| Update a fact | re-index one document (minutes) | retrain (hours, risky) | change the prompt |
| Citations | yes, chunk ids | no | possible |
| Cost per query | small prompt | none extra | pays for the whole context every time |
| Fails when | retrieval misses | facts are rare in training data | context is too large or the model loses the middle |
These are complementary. Use RAG for facts, fine-tuning for how to answer, and long context when the material is small and used once.

### Chunking (step 02)
A vector should hold one idea, and the embedding model only reads so many tokens (all-MiniLM-L6-v2 is truncated at 256 word pieces and was trained on 128). Splitting mid-sentence destroys facts: fixed 200-character chunks cut **20 sentences** and lost one answer. Following the document's structure (headings, then sentences) cut **0**. Overlap repairs boundaries but stores more (10 words of overlap gives **1.25×** the text). Prepending the title and section ("contextual header") helps both retrieval and the LLM.

### Embeddings (step 03)
A **bi-encoder** encodes text once into a fixed-size vector (MiniLM: 384 dims). That is why document vectors can be precomputed. Mean-pool the token vectors, L2-normalise, then use dot product, which equals cosine. Dense vectors match meaning ("Which teas will not keep me awake?" → caffeine chunk, cosine 0.47, with TF-IDF scoring 0.00). They blur exact details ("refunds ... 9 days" vs "shipping ... 9 days" still scores 0.56). Text past `max_length` is silently dropped. Use the same model, and the same query/passage format, at index and query time.

### Vector indexes (steps 04, 05)
Exact search is one matrix–vector product: **4.5 ms** for 100k × 384 on a laptop CPU, and linear in n. Approximate (ANN) indexes trade a little recall for big speed-ups:
- **IVF** clusters the vectors into cells and searches `nprobe` cells. In the run, 1.6% of the vectors scanned gave recall@10 = 1.00.
- **HNSW** walks a navigable small-world graph. Long-range links matter: recall 0.35 → 0.98 at the same `ef`.
- **PQ** compresses vectors 32× (512 → 16 bytes). Recall is 0.36 alone and 0.95 after re-scoring a shortlist with the full vectors.

Metadata filters should be applied *before* ranking (pre-filtering). Post-filtering returned **0** results where pre-filtering returned 3.

### Sparse, dense and hybrid (step 06)
BM25 scores exact term matches. IDF makes rare terms count (`ck-tulsi-100` has idf 3.21 vs `tea` at 1.47), `k1` saturates repeats, and `b` normalises for length. It wins on jargon and codes ("pin codes"). Dense retrieval wins on paraphrases. **Reciprocal Rank Fusion** (`Σ 1/(60 + rank)`) combines rankings without having to calibrate their scores. On this small clean knowledge base, dense (hit@1 0.83) already beats BM25 (0.71), and hybrid ties dense on hit@1 (slightly lower MRR, 0.88 vs 0.90). Hybrid earns its keep on large corpora full of codes and names.

### Re-ranking (step 07)
A **cross-encoder** reads the question and the chunk *together*, so it is more accurate, but it is one forward pass per pair: **~73 ms for 20 candidates**, or about an hour per query for 1M chunks. So retrieve broadly with fast methods (recall), then re-rank a few (precision). Hit@1 went 0.83 → 0.88 and MRR 0.88 → 0.92, but some questions moved *down*. Its scores double as an "is the answer here?" signal.

### The pipeline (step 08)
The pipeline runs retrieve 20 → re-rank → refuse if the best score is below a threshold → pack the top chunks into a token budget → prompt with `<documents>`, numbered sources and three rules (answer only from the documents, cite `[n]`, treat documents as untrusted data) → generate → validate citations. The run shows a correct answer with a citation, a correct refusal (gift cards), a **false refusal** (London: the answer is there, but phrased differently), a **distractor chunk** winning, and a **prompt injection** inside a document that only the guarded system prompt resists.

### Evaluation (step 09)
Evaluate retrieval and generation separately.
- **Retrieval:** hit@k, recall@k, MRR, nDCG. Hybrid + re-rank scored hit@1 0.88, MRR 0.92, nDCG@5 0.93.
- **Refusals:** the unanswerable and answerable score distributions overlap, so every threshold trades false refusals against hallucinations.
- **Generation:** correctness, faithfulness, citation validity, read alongside the individual failures. Most errors here were generation errors (the right chunk was present but the wrong sentence was used).
- **LLM-as-judge** scales grading but must be validated against human labels.
- With 24 questions, one question is worth 0.04, so small differences are noise.

### Advanced RAG and vector databases (step 10)
- **Multi-query rewriting:** the London question's gold chunk moved from rank 8 to rank 1.
- **HyDE** (search with a hypothetical answer): rank 8 → 1, cosine 0.30 → 0.79.
- **Small-to-big:** match a sentence, return its section.
- **Semantic caching:** the margin between a true paraphrase and a different question was only 0.74 vs 0.66.
- **Real tools:** FAISS (library), Chroma (embedded), pgvector (Postgres), Qdrant/Milvus/Weaviate/Pinecone (servers). See `vector_db_reference.py`.

## 6. Common mistakes and gotchas
- Indexing with one embedding model and querying with another, or changing models without re-embedding. Nothing raises an error; scores are just meaningless.
- Forgetting the `query:` / `passage:` prefixes for models trained with them (e5, bge).
- Chunks longer than the embedding model's limit, so the tail is silently truncated.
- Fixed-size chunks that cut sentences and numbers in half.
- Post-filtering metadata after a top-k search (too few results).
- Comparing dot products of unnormalised vectors and calling it cosine.
- Running a cross-encoder over the whole corpus (too slow), or no re-ranking at all when precision matters.
- No refusal path, so the model answers from irrelevant chunks.
- Trusting retrieved text as instructions (indirect prompt injection).
- Judging a change on 5 questions, or by averages without reading the failures.
- Not re-indexing when documents change (stale answers), or keeping duplicates after an update (use upserts with stable ids).

## 7. Self-check
- [ ] I can draw the offline and online pipelines and name the job of each stage.
- [ ] I can explain when to use RAG vs fine-tuning vs long context.
- [ ] I can compare five chunking strategies and choose a chunk size and overlap with reasons.
- [ ] I can explain a bi-encoder, mean pooling, normalisation, and why dot = cosine on unit vectors.
- [ ] I can write a flat top-k search with `argpartition` and a metadata pre-filter from memory.
- [ ] I can explain IVF (nlist, nprobe), HNSW (M, ef, layers) and PQ (sub-vectors, codebooks) and their dials.
- [ ] I can write the BM25 formula and explain idf, k1 and b; and write RRF.
- [ ] I can explain bi-encoder vs cross-encoder and why we retrieve-then-rerank.
- [ ] I can write a RAG prompt with numbered sources, citations, refusal and an injection guard.
- [ ] I can define hit@k, recall@k, MRR, nDCG and faithfulness, and tell a retrieval failure from a generation failure.
- [ ] I can name four advanced techniques and when each one helps.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Embedding model vs bi-encoder vs sentence encoder | "dense retriever", "text embedder" | Same idea | All map text to a fixed vector independently; "bi-encoder" names the two-tower training/architecture |
| Bi-encoder vs cross-encoder | retriever vs reranker | No | Bi-encoder embeds query and document separately (fast, ANN-searchable); cross-encoder reads the pair together (accurate, slow) |
| Vector DB vs vector index vs ANN library | Qdrant, FAISS, HNSW | No | Index = data structure; library (FAISS) = index code; vector DB = index plus storage, filters, CRUD, persistence, server |
| BM25 vs TF-IDF vs keyword/sparse search | lexical retrieval | Related | BM25 is TF-IDF with term-frequency saturation and length normalisation |
| Hybrid search vs RRF | fusion | No | Hybrid = using dense + sparse together; RRF is one way to merge their ranked lists (`1/(k+rank)`) |
| Recall@k vs hit rate vs precision@k vs MRR vs nDCG | retrieval metrics | No | Recall = share of relevant docs found; hit rate = at least one found; precision = share of results relevant; MRR/nDCG reward higher ranks |
| Reranking vs re-scoring vs second-stage retrieval | cross-encoder step | Mostly | All reorder the top-N candidates with a stronger model |
| RAG vs fine-tuning vs long context | "give the model knowledge" | No | RAG retrieves at query time; fine-tuning changes weights; long context pastes everything into the prompt |
| Chunk vs passage vs node | split piece | Mostly | Same unit of text indexed; "node" is LlamaIndex wording |

## 8. Interview questions (short answers in [INTERVIEW_GIST.md](INTERVIEW_GIST.md) §12)
1. Walk me through building a RAG system over 10,000 PDFs.
2. RAG or fine-tuning for a support bot? Why?
3. How do you choose chunk size? What does overlap buy you?
4. Why can't a cross-encoder be used for first-stage retrieval?
5. What is HNSW and which parameters would you tune?
6. When does BM25 beat embeddings? How do you combine them?
7. How do you evaluate a RAG system before launch? After launch?
8. The bot answers confidently from the wrong document. How do you debug it?
9. How do you stop a retrieved document from hijacking the model?
10. Your index has 50M vectors and memory is tight. What do you do?
