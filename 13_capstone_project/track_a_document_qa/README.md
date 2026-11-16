# Track A — Document intelligence: from pixels to a cited answer

**One sentence:** scanned and born-digital documents go in; a question goes in; a short answer comes out with the **page number and bounding box** it came from, behind a FastAPI service you can containerise and evaluate.

Back to the capstone: [13 README](../README.md). Code walkthrough: [WALKTHROUGH.md](WALKTHROUGH.md).

## Why this project
Modules 18 (OCR and parsing), 19 (chunking) and 14 (RAG) are deliberately separate: each is a topic you can be asked about on its own. Interviewers, though, ask "tell me about a system you built". This track is the connective tissue. It adds **no new algorithms**: every stage is imported from the module that teaches it, so you can say "stage 2 is my 19/09 chunker" and open the file.

## What you build (and which module owns each stage)

| Stage | What happens | Reused from (imported, not copied) | New code here |
|---|---|---|---|
| 1. Parse | pixels -> text, boxes, tables; PDF bytes -> route each page: text layer or OCR | [18/07](../../18_ocr_and_document_parsing/07_reading_order_and_tables/WALKTHROUGH.md) XY-cut + template OCR, [18/08](../../18_ocr_and_document_parsing/08_pdf_internals_and_routing/WALKTHROUGH.md) PDF reader + router, `docimg.py`, `dococr.py` | `parse_pdf_bytes` (line -> element + bbox) |
| 2. Handoff | one validated JSON contract between OCR and chunking | [18/10](../../18_ocr_and_document_parsing/10_parsed_document_handoff/WALKTHROUGH.md) `ParsedDocument` pydantic model, `to_parsed_document` | `validate()` call at every entry |
| 3. Chunk | reading order, headers start chunks, tables whole, cross-page table merge, page+bbox kept | [19/09](../../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md) `LayoutChunker`, `common19` | `chunk_document` (to the chunk schema below) |
| 4. Embed + retrieve | MiniLM vectors, brute-force dot product, optional cross-encoder rerank | [14/03](../../14_rag_and_retrieval/03_embeddings_bi_encoder/WALKTHROUGH.md), [14/04](../../14_rag_and_retrieval/04_vector_index_and_search/WALKTHROUGH.md), [14/07](../../14_rag_and_retrieval/07_reranking_cross_encoder/WALKTHROUGH.md) (`Embedder`, `dense_search`, `CrossEncoderReranker`) | `Index` (add / remove document, version) |
| 5. Answer | grounded answer + citations; mock extractive LLM by default, OpenAI-compatible server optional | prompt/citation ideas from [14/08](../../14_rag_and_retrieval/08_rag_pipeline_end_to_end/WALKTHROUGH.md) | `MockLLM`, `OpenAICompatLLM` |
| 6. Operate | exact-match answer cache keyed by index version, p50/p95 metrics, spans | [20/07 caching](../../20_llm_system_lld/07_caching/WALKTHROUGH.md), [20/09 observability](../../20_llm_system_lld/09_observability_and_eval_hooks/WALKTHROUGH.md) (`TTLLRUCache`, `Metrics`, `Tracer` from `ragkit`) | `Engine` |
| 7. Serve | `/health`, `/ingest`, `/ask`, lifespan loads once, Pydantic 422 | [09 FastAPI patterns](../../09_api_development/04_fastapi_basics/WALKTHROUGH.md), [validation](../../09_api_development/05_validation_and_errors/WALKTHROUGH.md), [testing](../../09_api_development/09_testing_and_docs/WALKTHROUGH.md) | `app.py` |
| 8. Ship | Dockerfile + compose file | [11 Docker](../../11_docker/README.md) | `Dockerfile`, `compose.yaml` |

Step scripts in 18 are teaching scripts (they print and write files when imported). `docqa.load_defs` executes only their *definitions* up to a marker line. That is a stopgap: the production move is to lift those functions into a package, which is also a good "what would you refactor" answer.

## Architecture
```
 pixels (docimg pages)      PDF bytes (mixed.pdf, upload)        hand/other parser output (Docling-style JSON)
        |                          |                                      |
        v                          v                                      |
  18/07 XY-cut + OCR        18/08 read_pdf + route(page)                  |
        |                     native / searchable -> text layer           |
        |                     scanned            -> template OCR          |
        v                          v                                      v
        +---------------> parsed_document.json  <------- validate (18/10 pydantic, extra="forbid") -- 422 on violation
                                   |
                                   v
                    19/09 LayoutChunker  (order, headers, whole tables, captions, page+bbox)
                                   |
                                   v   chunks  [{chunk_id, text, embed_text, pages, boxes, ...}]
                    14: MiniLM embed  -->  Index (matrix n x 384, version hash)
                                   |
   question --> embed --> dense top-k --(rerank?: cross-encoder on top 8)--> hits [{chunk + score}]
                                   |
                                   v
                    LLM: MockLLM (extractive)  |  OpenAI-compatible (env DOCQA_LLM_BASE_URL)
                                   |
                                   v
              {answer, grounded, citations:[{doc_id, page, bbox, score, used}], latency_ms, cached}
                                   ^
        20: TTLLRUCache (key = question + params + index version + llm), Metrics p50/p95, Tracer spans
        served by FastAPI: GET /health   POST /ingest   POST /ask
```

## Data contracts between stages
Contracts are the part of this project interviewers care about: each arrow has a schema, and a bad record is rejected at the arrow, not three stages later.

**1. `parsed_document.json`** (OCR/parser -> chunker). Strict pydantic model in [18/10](../../18_ocr_and_document_parsing/10_parsed_document_handoff/main.py); also what [19/09](../../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/sample_parsed_document.json) reads.
```
{ "doc_id": str, "source_file": str, "note": str,
  "pages":    [ {"page": int, "width": float, "height": float} ],              # PDF points
  "elements": [ { "id": str, "type": one of page_header | page_footer | title | section_header | text | list_item
                                   | caption | table | figure | footnote,
                  "text": str,                          # may be "" only for table / figure
                  "page": int, "bbox": [x0, y0, x1, y1],  # PDF points, origin TOP-left, x0<x1, y0<y1, inside the page
                  "reading_order": int,                 # unique per page
                  "level": int            (section_header only, required there, forbidden elsewhere),
                  "table": {"header": [str], "rows": [[str]]}   (table only, rectangular),
                  "figure": {"alt": str}                         (figure only),
                  "continued_from": str | null } ] }
```
OCR confidence is *not* in the schema (unknown keys are rejected); 18/10 writes it to a sidecar `parsed_document.ocr_meta.json`. Here the per-element route (`text layer` vs `OCR`) is kept the same way, outside the contract.

**2. chunk** (chunker -> index). Plain dict, exactly these keys (a test asserts it):
```
{ "chunk_id": "<doc_id>#<n>", "doc_id": str, "kind": "text" | "table" | "figure",
  "text": str,            # what the user / the LLM reads (markdown table or paragraph)
  "embed_text": str,      # what MiniLM embeds: breadcrumb + caption + linearised rows ("Product: Green Tea; Price: 320")
  "section_path": "Title > 2. Availability",
  "pages": [int],         # a merged cross-page table lists every page
  "boxes": [ {"page": int, "bbox": [x0, y0, x1, y1]} ],   # one union box per page: what a PDF viewer highlights
  "element_ids": [str] }  # back-pointers into parsed_document
```

**3. retrieval result / answer** (index -> LLM -> client). Returned by `POST /ask`:
```
{ "question": str, "answer": str, "grounded": bool, "model": "mock-extractive" | "openai-compatible:<name>",
  "citations": [ {"rank": 1, "doc_id": str, "page": int, "pages": [int], "bbox": [x0,y0,x1,y1], "chunk_id": str,
                  "kind": str, "score": float, "used": bool, "snippet": str} ],
  "index_version": str, "cached": bool, "latency_ms": float }
```
`used: true` marks the chunk(s) the answer was taken from; the other citations are the context that was retrieved. `score` is cosine similarity for dense retrieval and a raw logit when the cross-encoder reranked (not comparable across the two).

## Which chunking strategy, and why
[19/10](../../19_chunking_deep_dive/10_chunking_evaluation/WALKTHROUGH.md) scores 19 strategies on 74 questions over 13 *plain-text* documents. Its numbers (dense retrieval): fixed 128 tokens hit@1 0.72 / MRR 0.81; recursive with markdown separators 0.78 / 0.86; structure 100 + breadcrumb 0.82 / 0.88; small-to-big to section 0.91 / 0.93; auto-merging 0.88 / 0.92; propositions 0.62 / 0.74 at 140 LLM calls.
- On raw hit@1 the **parent-child family** (small-to-big, auto-merging) wins. The **structure + breadcrumb** family is the best strategy that needs no extra index structure and no LLM calls.
- This project needs more than text hit@1: every answer must carry **page and bbox**, and tables must survive. Only the element-level [19/09 layout-aware chunker](../../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/main.py) keeps geometry; it is also a structure + breadcrumb chunker (section path in `embed_text`, tables kept whole or split into row groups with the header repeated, rows linearised). So: **chosen = layout-aware (structure + breadcrumb)**; **upgrade path = add small-to-big on top** (embed small pieces, return the whole section's boxes), which is what 19/07 did to reach 0.91.
- Honest caveat: 19/10's table is not a measurement of 19/09 on parsed documents. For that, 19/09's own 7-question test (its trace) gave layout-aware hit@1 6/7 vs 3/7 for naive flatten+window (hit@3 7/7 for both) and page+bbox citations on 6/7 answers vs none; the eval below is this project's own.

## Evaluation (real run, see the WALKTHROUGH trace)
18 questions over 3 documents (`eval_questions.json`): each has gold `(doc, page)` list and an answer string.
- **hit@k:** some top-k chunk is on a gold page **and** contains the answer string. **MRR:** 1/rank of that chunk.
- **citation accuracy:** the chunk the answer was extracted from is on a gold page (the user clicks the citation and lands on the right page).
- **answer accuracy:** the gold string is inside the answer text. With the extractive mock this is a *lower* number than retrieval, on purpose: it shows the retrieval-vs-generation split.
- **latency p50/p95:** per question, end to end, cache bypassed, one process, CPU. Varies by machine.

Run `python run_pipeline.py` and read the table; typical shape: dense hit@3 ~0.94, rerank hit@3 1.00, rerank latency roughly 7-10x dense (a cross-encoder reads 8 passages per question).

## Failure modes per stage
| Stage | What goes wrong | Symptom | Detect / mitigate |
|---|---|---|---|
| Parse (OCR) | glyph confusion (the template OCR misreads the page footer as "Page 1 cif 2"); wrong font size | garbage tokens, answer strings not found | per-element confidence sidecar; footers are dropped anyway; character error rate on a labelled page ([18/06](../../18_ocr_and_document_parsing/06_ocr_evaluation/WALKTHROUGH.md)); use a real engine ([18/09](../../18_ocr_and_document_parsing/09_modern_document_parsers_2026/WALKTHROUGH.md)) |
| Routing | scan wrongly treated as native (or reverse); searchable-scan text layer is wrong | empty or wrong text for a page | `classify` thresholds in 18/08; log the route per element; sample-check pages |
| Reading order | two columns interleaved | sentences that mix two paragraphs | XY-cut gap parameters (18/07); compare WER of order |
| Contract | a parser emits `heading` instead of `section_header`, bbox outside the page, ragged table | 422 at `/ingest` with the field path | the strict pydantic model; never loosen it to "make it pass" |
| Chunking | cross-page table citation points at its first page only; a header-only chunk; two columns merged into one chunk (`c8` in the eval: "support open on weekdays" is buried in a Loyalty+Support chunk) | right chunk is retrieved but cited page is off, or the chunk is diluted and ranks low | gold lists both pages for merged tables; smaller `max_tokens`; small-to-big |
| Duplicates | the same text on several pages/docs (mixed.pdf page 2 and its searchable-scan twin page 3; "Refunds ... nine days" in two documents) | citation lands on the "wrong" duplicate | dedupe by content hash at ingest, or accept all duplicates in the gold set (done here) |
| Retrieval | a question that needs two chunks or an aggregate ("which service grew the most") | right page, wrong sentence | rerank, bigger k, an LLM that reads all rows; this is a generation task |
| Generation | extractive mock picks the wrong sentence on a tie (`m3`, `u5` in the eval); a real LLM may cite a source it did not use or invent | `answer acc` below `hit@k` | `grounded` flag + refusal threshold; citation check hook ([20/09](../../20_llm_system_lld/09_observability_and_eval_hooks/WALKTHROUGH.md)) |
| Cache | stale answers after re-ingest | old answer returned | key includes the index version hash; changing any chunk changes the key |
| Serving | each uvicorn worker loads its own index and models (RAM x workers); index lives in memory, lost on restart | high RSS, empty index after restart | one worker here; persist chunks (artifacts/) or move to a vector DB ([14/11 Qdrant](../../14_rag_and_retrieval/11_qdrant_fundamentals/WALKTHROUGH.md)) |

## Run it
```bash
# from the repo root; HF_HUB_OFFLINE=1 is set for you inside docqa.py (cached models only, no downloads)
.venv/bin/python 13_capstone_project/track_a_document_qa/run_pipeline.py          # ~15 s: the whole pipeline + eval, writes artifacts/
.venv/bin/python -m pytest -q 13_capstone_project/track_a_document_qa/test_track_a.py   # 13 tests, ~20 s
cd 13_capstone_project/track_a_document_qa && ../../.venv/bin/python -m uvicorn app:app --port 8001
curl -s localhost:8001/health
curl -s -X POST localhost:8001/ask -H 'content-type: application/json' -d '{"question":"What is the price of Green Tea?","k":3}'
curl -s -X POST localhost:8001/ask -H 'content-type: application/json' -d '{"question":"Who owns the failover test?","rerank":true}'
```
Ingest your own parsed document: `POST /ingest {"parsed_document": {...}}` (contract above), a PDF `{"pdf_base64": "...", "doc_id": "my_pdf"}` (text-layer PDFs and image-only scans in the style 18/08 generates; real-world PDFs with fonts/encryption need PyMuPDF, see [18/08 pymupdf_reference.py](../../18_ocr_and_document_parsing/08_pdf_internals_and_routing/pymupdf_reference.py)), or `{"builtin": true}`.

Optional real LLM (nothing is paid or required): `DOCQA_LLM_BASE_URL=http://localhost:8000/v1 DOCQA_LLM_MODEL=<name> [DOCQA_LLM_API_KEY=...]` works with any OpenAI-compatible server (vLLM, llama.cpp server, Ollama, a hosted API). The code path is unit-tested with an injected fake transport; it was **not** run against a live endpoint here.

## Docker
Files: [Dockerfile](Dockerfile), [compose.yaml](compose.yaml), [requirements.txt](requirements.txt), [prefetch_models.py](prefetch_models.py). **Not built or run** (no daemon assumed); reviewed by reading. Decisions to be able to defend:
- **Build context = repo root**, because the service imports modules 14/18/19/20 by relative path. The Dockerfile `COPY`s only the files the pipeline imports, at the same relative paths (`docqa.py` computes `REPO` from its own location). Command: `docker build -f 13_capstone_project/track_a_document_qa/Dockerfile -t docqa .`
- **Separate compose file** so the core capstone's `compose.yaml` (api + ui) is untouched: `docker compose -f 13_capstone_project/track_a_document_qa/compose.yaml up --build`.
- Dependencies installed **before** code (layer cache), **CPU-only torch** wheel index (the default wheel drags in CUDA libraries, GBs), the two small models **downloaded at build time** then `HF_HUB_OFFLINE=1` at run time so the container never touches the network.
- Non-root user, `0.0.0.0` bind, a `HEALTHCHECK` on `/health`, **one worker** (every worker would load its own index and models; scale with replicas behind a load balancer and a shared vector store).
- If it fails first time that is a drill: `docker compose logs docqa`, usual suspects are a missing COPY (ImportError), the model prefetch having no network, or the port already in use ([11/05](../../11_docker/05_healthchecks_logs_debugging/WALKTHROUGH.md)).

## How to present this in an interview (5 minutes)
1. **Problem (20 s):** "Answer questions over messy PDFs and scans, and prove where each answer came from."
2. **Draw the architecture above (60 s)** and name the three contracts. Emphasise that parsed_document is validated at the boundary.
3. **One design decision with data (60 s):** chunking. "I chose layout-aware structure+breadcrumb chunking because citations need geometry and tables must stay whole; 19/10 showed parent-child gives higher hit@1 on plain text, which is my upgrade path."
4. **Show the eval table (60 s):** dense vs rerank, hit@k, citation accuracy, p50/p95. "Rerank bought a higher hit@3 for roughly 7-10x the latency; I would enable it behind a flag or for low-confidence queries."
5. **Name your failures (60 s):** pick two from the table (duplicates; extractive picks the wrong sentence while retrieval was right) and say what you would do.
6. **What I would do next (30 s):** pick from the extension ideas.

> **Say it aloud:** I built a document question-answering service where scans go through OCR and page routing into a validated parsed-document contract, a layout-aware chunker keeps tables whole and page-and-box geometry, MiniLM retrieval with an optional cross-encoder rerank feeds a grounded answerer, and every answer returns page and bounding-box citations. I evaluated it on labelled questions with hit-at-k, citation accuracy and p50/p95 latency. The weak spot is generation, not retrieval, so the next step is a real LLM behind the same interface plus a better OCR engine.

## Extension ideas
- **VLM / modern parser as stage 1:** replace template OCR with a vision-language model or layout engine ([18/09 reference engines](../../18_ocr_and_document_parsing/09_modern_document_parsers_2026/WALKTHROUGH.md)); it only has to emit the `parsed_document` contract, nothing downstream changes. Add the OCR confidence sidecar into `/ask` answers as a "low-confidence source" warning.
- **Small-to-big:** retrieve small chunks, return the section's union box ([19/07](../../19_chunking_deep_dive/07_hierarchical_parent_child/WALKTHROUGH.md)).
- **Hybrid search:** BM25 + dense with RRF ([14/06](../../14_rag_and_retrieval/06_bm25_and_hybrid_search/WALKTHROUGH.md)); fixes lexical questions like codes and numbers.
- **Real LLM with citation enforcement:** require `[n]` markers, drop sentences without a marker, add a `CitationHook` ([20/09](../../20_llm_system_lld/09_observability_and_eval_hooks/WALKTHROUGH.md)); stream tokens with SSE ([09/10](../../09_api_development/10_streaming_sse/WALKTHROUGH.md)).
- **Persistence and scale:** Qdrant instead of an in-memory matrix ([14/11](../../14_rag_and_retrieval/11_qdrant_fundamentals/WALKTHROUGH.md)), ingest as a background job, per-tenant `doc_id` scoping, a semantic cache ([20/07](../../20_llm_system_lld/07_caching/WALKTHROUGH.md)).
- **Evaluation depth:** more questions, answer-faithfulness judging ([14/09](../../14_rag_and_retrieval/09_rag_evaluation/WALKTHROUGH.md)), OCR noise sweep (add Gaussian noise to the pages with `docimg.make_report_page(noise=...)` and plot hit@k vs noise).
- **Serving the generator yourself:** the inference side is module 21; see [Track B](../track_b_inference_report/README.md).
