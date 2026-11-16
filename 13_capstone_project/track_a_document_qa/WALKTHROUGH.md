# Track A — Document intelligence: the code, block by block

## The story so far
You have the parts. [18 OCR and parsing](../../18_ocr_and_document_parsing/README.md) turns pixels into a validated `parsed_document.json` ([18/10](../../18_ocr_and_document_parsing/10_parsed_document_handoff/WALKTHROUGH.md)); [19 chunking](../../19_chunking_deep_dive/README.md) cuts such a document into chunks that keep page and box ([19/09](../../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md)); [14 RAG](../../14_rag_and_retrieval/README.md) retrieves and reranks; [20 LLD](../../20_llm_system_lld/README.md) adds cache and metrics; [09](../../09_api_development/README.md) serves it. The core capstone ([WALKTHROUGH](../WALKTHROUGH.md)) wired a *classifier* into an API. This track wires the *document* stack.

## The pain point
Each module's demo runs on its own input and prints its own numbers. Nothing proves the stages fit together: does OCR output pass the chunker's contract, does the page number survive to the answer, does the service answer within a reasonable time, and how often is the cited page right? Until one command runs all of it and scores it, "I built a RAG system" is a claim, not evidence.

## The spark
Treat every arrow between modules as a **data contract**, validate at each arrow, and add the smallest glue (a library file `docqa.py`, a CLI, an API, an eval) that imports the modules instead of copying them. Then the whole system is one run, one test file and one table.

## Goal
Read `run_pipeline.py` (the pipeline as a script) and `app.py` (the pipeline as a service) line by line, know what each stage returns, and be able to rebuild both. `docqa.py` (the glue library) is summarised in a table; its stages are taught in the linked module steps. Architecture, contracts, failure modes and interview framing: [README.md](README.md).

## The theory (read this before the interview)
**Pipeline = functions with contracts.**
```
bytes/pixels --parse--> parsed_document --chunk--> chunks --embed--> index --retrieve--> hits --generate--> answer+citations
              (18)       [contract 1]      (19)   [contract 2]  (14)          (14)               (mock/LLM)   [contract 3]
```
- **Grounding and citation:** the answer must be derivable from the retrieved text, and the response must say *where*: `doc_id`, `page`, `bbox` (PDF points, origin top-left). A citation is only as good as the geometry carried through every stage; one stage that drops `bbox` ends the feature.
- **hit@k** = fraction of questions for which a top-k chunk is on a gold page and contains the answer string. **MRR** = mean of 1/rank of the first such chunk. **citation accuracy** = fraction of answers whose cited chunk is on a gold page. **p50/p95** = 50th/95th percentile of per-question latency (nearest-rank; `Metrics.percentile` in ragkit).
- **Two-stage retrieval:** a bi-encoder (embed query and chunks separately, dot product: fast, approximate) retrieves a pool; a cross-encoder (reads query and passage together: slow, sharper) re-scores the pool. Quality up, latency up roughly by the pool size.
- **Extractive vs generative answer:** the mock LLM only *selects* a sentence by word overlap with the question; it cannot aggregate ("which grew most"). That is deliberate: free, deterministic, always grounded, and it exposes the retrieval-vs-generation gap.
- **Cache key = question + parameters + index version + LLM name.** Re-ingesting changes the index version, so stale answers cannot be served.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| parsed document | structured document, layout JSON, Docling/Unstructured output | Same idea | Here it is one strict schema (18/10), so any parser can plug in |
| chunk | passage, node (LlamaIndex), document (LangChain) | Same | Ours carries `pages` and `boxes` for citations |
| bbox | bounding box, rect, layout box | Same | PDF points, origin top-left, `[x0, y0, x1, y1]` |
| retrieval hit@k | recall@k, top-k accuracy | Close | Ours also requires the answer text to be inside the chunk |
| rerank | cross-encoder re-scoring, second-stage ranking | Same | Reads (query, passage) together |
| grounded answer | faithful, attributed, cited answer | Close | `grounded` here = answer text taken from a retrieved chunk (or `[n]` cited) |
| index version | corpus hash, snapshot id | Same | Used to scope cache keys |
| router (PDF) | page classifier, native-vs-scan detection | Same | 18/08 `route(page)` |
| p95 latency | tail latency, 95th percentile | Same | The number a user feels on a bad day |

## Run it
```bash
cd 13_capstone_project/track_a_document_qa
python run_pipeline.py                         # flags: --k 3  --skip-rerank  --no-save
python -m pytest -q test_track_a.py
```
From the repo root: `HF_HUB_OFFLINE=1 .venv/bin/python 13_capstone_project/track_a_document_qa/run_pipeline.py` (~15 s on CPU; uses cached all-MiniLM-L6-v2 and ms-marco-MiniLM; no downloads, no paid API).
Service: `python -m uvicorn app:app --port 8001` then `curl localhost:8001/health` ([README](README.md#run-it)).

## Code, step by step
The glue library `docqa.py` is referenced by line below; each stage is taught where it lives. Map:

| `docqa.py` piece | What it does | Taught in |
|---|---|---|
| `load_defs`, `step` | execute only the definitions of 18/07, 18/08, 18/10 step scripts (no demo output, no file writes) | this file's docstring |
| `parse_report_pages` | render page pixels, XY-cut + OCR, convert to the contract | [18/07](../../18_ocr_and_document_parsing/07_reading_order_and_tables/WALKTHROUGH.md), [18/10](../../18_ocr_and_document_parsing/10_parsed_document_handoff/WALKTHROUGH.md) |
| `parse_pdf_bytes` | read PDF, route each page (text layer / OCR), line -> element with bbox | [18/08](../../18_ocr_and_document_parsing/08_pdf_internals_and_routing/WALKTHROUGH.md) |
| `validate` | strict pydantic check of the handoff contract | [18/10](../../18_ocr_and_document_parsing/10_parsed_document_handoff/WALKTHROUGH.md) |
| `chunk_document` | 19/09 `LayoutChunker` -> chunk dicts | [19/09](../../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/WALKTHROUGH.md) |
| `Index` | matrix of unit vectors, dense search, optional cross-encoder rerank, version hash | [14/03](../../14_rag_and_retrieval/03_embeddings_bi_encoder/WALKTHROUGH.md), [14/07](../../14_rag_and_retrieval/07_reranking_cross_encoder/WALKTHROUGH.md) |
| `MockLLM`, `OpenAICompatLLM`, `make_llm` | extractive answerer; optional OpenAI-compatible HTTP client (stdlib `urllib`) | [14/08](../../14_rag_and_retrieval/08_rag_pipeline_end_to_end/WALKTHROUGH.md) |
| `Engine` | ingest + ask with `TTLLRUCache`, `Metrics`, `Tracer` | [20/07](../../20_llm_system_lld/07_caching/WALKTHROUGH.md), [20/09](../../20_llm_system_lld/09_observability_and_eval_hooks/WALKTHROUGH.md) |
| `evaluate` | hit@k, MRR, citation and answer accuracy, p50/p95 | [14/09](../../14_rag_and_retrieval/09_rag_evaluation/WALKTHROUGH.md) |

### `run_pipeline.py`

### 1. Imports and flags (`run_pipeline.py` lines 14–27)
```python
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import docqa

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--skip-rerank", action="store_true", help="skip the cross-encoder pass of the eval")
ap.add_argument("--no-save", action="store_true", help="do not write artifacts/")
args = ap.parse_args()
```
- **In simple words:** load the glue library, and let the user choose how many chunks to retrieve and whether to skip the slow reranker pass.
- **Exactly:** `argparse.ArgumentParser().parse_args()` reads `sys.argv`; `--skip-rerank` and `--no-save` are booleans (`store_true`). `import docqa` has side effects on purpose: it sets `HF_HUB_OFFLINE=1` (so `transformers` never contacts the network) and adds the folders of modules 14, 18, 19 and 20 to `sys.path`, so `import common` finds 14's `common.py`.
- **Why it is here:** one switch for the 15-second path (`--skip-rerank`), and offline-by-default so a missing network can never break the run.

### 2. Stage 1 — OCR, routing, parse (`run_pipeline.py` lines 29–38)
```python
# ---- STAGE 1: OCR / parse. Pixels and PDF bytes become validated parsed_document dicts (module 18: steps 07, 08, 10) -------------------------
t0 = time.perf_counter()
corpus = docqa.build_default_corpus()                     # {doc_id: (parsed_document, ocr_meta)}; validation by 18/10's pydantic model happens inside
print(f"STAGE 1  OCR + routing + parse: {len(corpus)} documents in {time.perf_counter() - t0:.1f}s")
for doc_id, (doc, meta) in corpus.items():
    types = dict(Counter(e["type"] for e in doc["elements"]))
    routes = dict(Counter(m["route"] for m in meta.values() if "route" in m))
    print(f"   {doc_id:<20} pages={len(doc['pages'])} elements={len(doc['elements']):<3} types={types}" + (f" page routes={routes}" if routes else ""))
mixed = corpus["mixed_pdf"][0]
print("   mixed.pdf page 2 was an image-only scan; the OCR text we now hold:", [e["text"] for e in mixed["elements"] if e["page"] == 2])
```
- **In simple words:** build the three documents of the corpus and show what kind of elements and which routes came out.
- **Exactly:** `build_default_corpus()` returns `{doc_id: (parsed_document, meta)}`. Document 1 (`chai_kitchen_report`) renders two synthetic pages as pixel arrays (`docimg.make_report_page`), runs 18/07's `parse_page` (recursive XY-cut for reading order, ruling-line table detection, template OCR) and converts pixels to PDF points with 18/10. Document 2 (`mixed_pdf`) is `mixed.pdf` from 18/08: `read_pdf` parses the bytes, `route(page)` picks "text layer" for page 1 and page 3 (searchable scan) and OCR for page 2 (image only). Document 3 (`lumen_q3_report`) is module 19's hand-written 3-page file standing in for a Docling-style parser's output. Every document passes through `validate()` (18/10's pydantic model, `extra="forbid"`). `meta` is a side dictionary (route per element); it is not part of the contract.
- **Why it is here:** the printed `page routes` line is the proof that the scanned page was really OCR'd; the last print shows the text that exists only because OCR ran.

### 3. Stage 2 — chunking (`run_pipeline.py` lines 40–50)
```python
# ---- STAGE 2: chunking. 19/09's LayoutChunker keeps tables whole, binds captions, merges cross-page tables, keeps page+bbox ---------------------
print("\nSTAGE 2  layout-aware chunking (19/09 LayoutChunker, max 140 tokens)")
all_chunks = {}
for doc_id, (doc, _) in corpus.items():
    all_chunks[doc_id] = docqa.chunk_document(doc)
    kinds = dict(Counter(c["kind"] for c in all_chunks[doc_id]))
    print(f"   {doc_id:<20} {len(all_chunks[doc_id])} chunks {kinds}")
sample = all_chunks["lumen_q3_report"][2]
print("   one chunk (the contract between chunking and retrieval):")
print("   " + json.dumps({k: sample[k] for k in ("chunk_id", "kind", "section_path", "pages", "boxes", "element_ids")}))
print("   embed_text (what MiniLM sees): " + repr(sample["embed_text"][:110]))
```
- **In simple words:** cut each document into pieces that respect headers and tables, and show one piece with its geometry.
- **Exactly:** `chunk_document(doc)` sorts elements by `(page, reading_order)`, drops headers and footers, starts a new chunk at each section header, keeps a table whole (or splits into row groups with the header repeated when over 140 tokens), merges a table that continues on the next page, binds captions, and records `pages` and one union `boxes` entry per page. Token counts use the real MiniLM tokenizer. `embed_text` is the breadcrumb plus caption plus rows rewritten as `Column: value` pairs.
- **Why it is here:** this is contract 2. The printed JSON line is exactly what the retriever and the API consume. See the README for why this chunker was chosen after 19/10.

### 4. Stage 3 — index (`run_pipeline.py` lines 52–58)
```python
# ---- STAGE 3: index (embed every chunk with the cached all-MiniLM-L6-v2) + the engine with cache / metrics from module 20 ----------------------
engine = docqa.Engine()
t0 = time.perf_counter()
for doc_id, (doc, meta) in corpus.items():
    info = engine.ingest(doc, meta)
print(f"\nSTAGE 3  index: {len(engine.index.chunks)} chunks, vectors {engine.index.vectors.shape}, version {engine.index.version}, "
      f"built in {time.perf_counter() - t0:.1f}s (first call includes loading MiniLM)")
```
- **In simple words:** create the engine and put every document into it, which embeds all chunks.
- **Exactly:** `Engine()` creates an empty `Index`, a mock LLM (or an OpenAI-compatible client if `DOCQA_LLM_BASE_URL` is set), a `TTLLRUCache(capacity=256, ttl=300 s)`, `Metrics` and a `Tracer`. `engine.ingest(doc)` validates, chunks, embeds with `common19.embed` (MiniLM, mean pooling, L2-normalised, 384 floats per chunk, cached per text) and appends to a `(n_chunks, 384)` float32 matrix. The version hash changes whenever any chunk id or text changes.
- **Why it is here:** the same `ingest` is what `POST /ingest` calls, so the script and the API cannot drift.

### 5. Stage 4 — ask, with citations (`run_pipeline.py` lines 60–69)
```python
# ---- STAGE 4: ask. retrieve -> (rerank) -> mock extractive LLM -> answer + citations ------------------------------------------------------------
print(f"\nSTAGE 4  grounded answers (LLM backend: {engine.llm.name}; set DOCQA_LLM_BASE_URL to use an OpenAI-compatible server)")
for question, rerank in [("What is the price of Green Tea?", False), ("What is the total due on the invoice?", False),
                         ("Who is the CEO of Lumen?", False)]:
    out = engine.ask(question, k=args.k, rerank=rerank)
    print(f"   Q: {question}\n   A: {out['answer']}   (grounded={out['grounded']}, {out['latency_ms']} ms)")
    for c in out["citations"][:2]:
        print(f"      [{c['rank']}] {'USED ' if c['used'] else '     '}{c['doc_id']} p.{c['page']} bbox={c['bbox']} score={c['score']} {c['kind']}")
again = engine.ask("What is the price of Green Tea?", k=args.k)
print(f"   same question again -> cached={again['cached']} in {again['latency_ms']} ms (module 20 TTLLRUCache, key scoped by index version)")
```
- **In simple words:** ask three questions (a table value, a scanned-page fact, and one the documents cannot answer), print the answer and the top citations, then ask the first again to show the cache.
- **Exactly:** `engine.ask` embeds the question, takes the top-k by dot product (`dense_search`, exact brute force), optionally reranks the top 8 with the cross-encoder, calls the LLM, and builds `citations` with `used` flags. The mock refuses (`grounded=False`) when no sentence shares enough question words (overlap below 0.34). The second call returns the cached dict with `cached=True`.
- **Why it is here:** the "CEO" question shows refusal instead of a confident wrong answer; the repeat shows the cache latency (tens of microseconds).

### 6. Stage 5 — evaluation (`run_pipeline.py` lines 71–83)
```python
# ---- STAGE 5: evaluation -----------------------------------------------------------------------------------------------------------------
qs = docqa.load_questions()
print(f"\nSTAGE 5  eval: {len(qs)} questions with gold (doc, page) + answer string; hit@k needs the right page AND the answer text in a top-k chunk")
reports = {"dense": docqa.evaluate(engine, args.k, False)}
if not args.skip_rerank:
    reports["dense+rerank"] = docqa.evaluate(engine, args.k, True)
print(f"   {'setup':<14} {'hit@1':>6} {'hit@'+str(args.k):>6} {'MRR':>6} {'cite acc':>9} {'answer acc':>11} {'p50 ms':>8} {'p95 ms':>8}")
for name, r in reports.items():
    print(f"   {name:<14} {r['hit_at_1']:>6.2f} {r['hit_at_k']:>6.2f} {r['mrr']:>6.2f} {r['citation_acc']:>9.2f} {r['answer_acc']:>11.2f} {r['p50_ms']:>8.1f} {r['p95_ms']:>8.1f}")
worst = reports.get("dense+rerank", reports["dense"])
fails = [r["id"] for r in worst["rows"] if not (r["hit_at_1"] and r["cite_ok"] and r["answer_ok"])]
print("   questions that still miss something:", fails, "(see README 'failure modes' for why)")
print("   engine metrics:", {k: round(v, 1) for k, v in engine.metrics.snapshot().items()})
```
- **In simple words:** score the system on 18 labelled questions, dense only and dense plus rerank, and list what still misses.
- **Exactly:** `evaluate` calls `engine.ask(..., use_cache=False)` per question and times it, then computes hit@1, hit@k, MRR, citation accuracy, answer accuracy and `np.percentile` p50/p95. A question "still misses something" if any of hit@1, citation or answer is false for the better setup.
- **Why it is here:** without a table, the choice between dense and rerank is an opinion.

### 7. Saving the contracts (`run_pipeline.py` lines 85–92)
```python
if not args.no_save:
    out_dir = HERE / "artifacts"
    out_dir.mkdir(exist_ok=True)
    for doc_id, (doc, _) in corpus.items():
        (out_dir / f"parsed_{doc_id}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    (out_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in engine.index.chunks))
    (out_dir / "eval_report.json").write_text(json.dumps(reports, indent=1))
    print(f"\nwrote {out_dir.relative_to(HERE.parent)}/ (parsed_*.json, chunks.jsonl, eval_report.json)")
```
- **In simple words:** write what each stage produced to `artifacts/` so you can open and inspect it.
- **Exactly:** one `parsed_<doc_id>.json` per document (contract 1), `chunks.jsonl` (contract 2, one JSON object per line), `eval_report.json` (per-question rows). `ensure_ascii=False` keeps non-ASCII text readable.
- **Why it is here:** debugging a pipeline means looking at the intermediate files, not re-running.

### 8. Try it / break it (`run_pipeline.py` lines 94–97)
```python
# TRY IT:   run with DOCQA_LLM_BASE_URL unset (mock) then point it at any OpenAI-compatible server (vLLM, llama.cpp, Ollama) and compare answer_acc;
#           change max_tokens in docqa.chunk_document from 140 to 60 and see which table chunks split (19/09 step).
# BREAK IT: edit artifacts/parsed_mixed_pdf.json (bbox [5, 5, 1, 1]) and call docqa.validate(): the handoff contract names the bad field;
#           delete mixed.pdf and watch the pipeline regenerate it by running 18/08 (the self-sufficient step pattern).
```
- **In simple words:** exercises; see the experiments and debugging sections below.
- **Why it is here:** the course format asks every script to end with them.

### `app.py`

### 9. Imports (`app.py` lines 13–22)
```python
import base64
import binascii
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

import docqa
```
- **In simple words:** FastAPI for the web layer, pydantic for request shapes, and our `docqa` module for the work.
- **Exactly:** `HTTPException(status, detail)` turns into a JSON error response; `Field(...)` attaches validation (length, range, regex) that FastAPI enforces before your function runs and returns as HTTP 422 with the failing field path.
- **Why it is here:** the same toolset as [09/05 validation](../../09_api_development/05_validation_and_errors/WALKTHROUGH.md).

### 10. Lifespan: build the engine once (`app.py` lines 25–33)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = docqa.Engine()                       # one engine per worker process: index, cache, metrics live here
    if os.environ.get("DOCQA_AUTOLOAD", "1") == "1":
        app.state.engine.ingest_default()
    yield


app = FastAPI(title="Document QA (Track A)", version="1.0", lifespan=lifespan)
```
- **In simple words:** when the server starts, create one engine (loading the models) and fill it with the built-in documents; keep it for every request.
- **Exactly:** `@asynccontextmanager` turns the generator into a context manager; code before `yield` runs at startup, after it at shutdown. `app.state` is a per-app attribute bag. `DOCQA_AUTOLOAD=0` starts empty. The handlers are plain `def`, so FastAPI runs them in a thread pool and the CPU-bound embedding does not block the event loop.
- **Why it is here:** loading MiniLM per request would cost seconds each time ([09/07 workers](../../09_api_development/07_servers_workers_concurrency/WALKTHROUGH.md): each worker has its own copy, which is why the Dockerfile uses one).

### 11. Request models (`app.py` lines 36–48)
```python
class IngestRequest(BaseModel):
    """Exactly one of three sources. The middle one is the handoff contract: ANY OCR/layout engine can POST its parsed_document here."""
    builtin: bool = False                                   # the default corpus produced by module 18
    parsed_document: Optional[dict] = None                  # validated against 18/10's strict pydantic model
    pdf_base64: Optional[str] = None                        # a PDF file, base64: routed native-vs-OCR by 18/08
    doc_id: Optional[str] = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=3, ge=1, le=10)
    rerank: bool = False                                    # cross-encoder re-scoring of the top-8 (slower, more accurate)
    doc_id: Optional[str] = None                            # restrict to one document
```
- **In simple words:** the shape of what clients may send.
- **Exactly:** `IngestRequest` accepts exactly one of three sources (checked in the handler); the `parsed_document` field is a free `dict` here and is validated by the 18/10 model inside `engine.ingest`, so there is one definition of the contract. `doc_id` has a regex so it is safe in file names and ids. `AskRequest.k` is limited to 1–10.
- **Why it is here:** limits protect latency and memory (a request for k=10000 is rejected before any work).

### 12. `/health` (`app.py` lines 51–56)
```python
@app.get("/health")
def health():
    e = app.state.engine
    return {"status": "ok", "documents": e.index.docs, "n_chunks": len(e.index.chunks), "index_version": e.index.version,
            "llm": e.llm.name, "metrics": e.metrics.snapshot()}

```
- **In simple words:** a cheap endpoint that reports what is loaded and the p50/p95 metrics.
- **Exactly:** returns documents with chunk counts, `index_version`, the LLM name and `Metrics.snapshot()`. The Docker `HEALTHCHECK` calls it.
- **Why it is here:** orchestrators and humans both need one URL that says "ready, and with what".

### 13. `/ingest` (`app.py` lines 58–77)
```python
@app.post("/ingest")
def ingest(req: IngestRequest):
    e = app.state.engine
    sources = [bool(req.builtin), req.parsed_document is not None, req.pdf_base64 is not None]
    if sum(sources) != 1:
        raise HTTPException(422, "send exactly one of: builtin=true, parsed_document, pdf_base64")
    if req.builtin:
        return {"ingested": e.ingest_default()}
    if req.pdf_base64 is not None:
        if not req.doc_id:
            raise HTTPException(422, "doc_id is required with pdf_base64")
        try:
            data = base64.b64decode(req.pdf_base64, validate=True)
            doc, meta = docqa.parse_pdf_bytes(data, req.doc_id)
        except (binascii.Error, ValueError, KeyError, AttributeError, IndexError, StopIteration, TypeError) as exc:   # not base64 / not a PDF our small reader understands
            raise HTTPException(422, f"could not read the PDF: {type(exc).__name__}: {exc}")
        if not doc["elements"]:
            raise HTTPException(422, "no text found (empty page or an unsupported PDF feature)")
        return {"ingested": [_safe_ingest(e, doc, meta)]}
    return {"ingested": [_safe_ingest(e, req.parsed_document, {})]}
```
- **In simple words:** add documents, from the built-in corpus, from a parser's JSON, or from PDF bytes.
- **Exactly:** exactly one source must be set (else 422). For PDFs, base64 is decoded (`validate=True` rejects non-base64) and `docqa.parse_pdf_bytes` reads and routes pages; failures of our small reader (`KeyError`, `StopIteration` ...) become a 422 instead of a 500. An empty result is rejected. Everything funnels into `engine.ingest`.
- **Why it is here:** this is the doorway where any OCR engine can plug in, which is the point of the contract.

### 14. Turning contract violations into 422 (`app.py` lines 80–88)
```python
def _safe_ingest(engine, doc, meta):
    try:
        return engine.ingest(doc, meta)
    except ValidationError as exc:                           # the contract failed: tell the caller which field, as 422
        first = exc.errors()[0]
        raise HTTPException(422, f"parsed_document invalid at {'.'.join(str(x) for x in first['loc'])}: {first['msg']}")
    except Exception as exc:
        raise HTTPException(422, f"parsed_document rejected: {type(exc).__name__}: {exc}")

```
- **In simple words:** if the document breaks the schema, tell the caller which field.
- **Exactly:** `pydantic.ValidationError.errors()` is a list of dicts with `loc` (path) and `msg`; we report the first. A second broad `except` catches downstream failures (for example chunker assumptions) and still answers 422 with the error type.
- **Why it is here:** a 500 says "our bug"; a 422 with a field path says "your data", and the caller can fix it.

### 15. `/ask` (`app.py` lines 90–95)
```python
@app.post("/ask")
def ask(req: AskRequest):
    e = app.state.engine
    if not e.index.chunks:
        raise HTTPException(409, "index is empty: POST /ingest first")
    return e.ask(req.question, k=req.k, rerank=req.rerank, doc_id=req.doc_id)
```
- **In simple words:** answer a question over the current index.
- **Exactly:** 409 (conflict: state not ready) if nothing is ingested; otherwise `engine.ask` returns the answer dict of contract 3. `doc_id` restricts retrieval to one document (the search over-fetches then filters).
- **Why it is here:** the whole system in one call; `latency_ms` and `cached` make behaviour observable from the client side.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `corpus.items()` with `for doc_id, (doc, meta) in ...` | iterate key and unpack a tuple value in one go |
| `f"{x:<20}"`, `f"{x:>6.2f}"` | pad left-aligned to width 20; right-align width 6 with 2 decimals |
| `{k: round(v, 1) for k, v in d.items()}` | dict comprehension |
| `@app.post("/ask")` | decorator registering a function as the handler of that route |
| `req: AskRequest` | FastAPI reads the JSON body and validates it into this model |
| `asynccontextmanager` + `yield` | startup/shutdown pair |
| `app.state.engine` | per-application shared object |
| `with TestClient(app) as c:` (tests) | runs lifespan startup/shutdown around the client |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `argparse.ArgumentParser` | parse CLI flags | reads `sys.argv`, builds a namespace, prints `--help` |
| `docimg.make_report_page(i)` | draw a synthetic page (18) | Pillow `ImageDraw` text and lines into an 800x640 grey array |
| `parse_page(gray)` (18/07) | XY-cut + OCR + tables | binarise (Otsu), cut along empty bands, nearest-template character matching |
| `read_pdf`, `route` (18/08) | parse PDF bytes, choose text layer or OCR | regex over objects, zlib inflate, content-stream operators |
| `ParsedDocument.model_validate` | check the contract | pydantic v2 validation, nested models, custom validators |
| `LayoutChunker(140).chunk(doc)` (19/09) | structure-aware chunks | sort, skip noise, flush at headers, whole tables |
| `common19.embed` | text to unit vectors | MiniLM forward pass, mean pooling, L2 normalise, per-text cache |
| `dense_search(q, M, k)` (14) | top-k by cosine | `M @ q`, `argpartition`, sort the k |
| `CrossEncoderReranker.score` (14/07) | relevance logit per pair | BERT-style model reads `[CLS] query [SEP] passage` |
| `TTLLRUCache` (20/07) | bounded cache | ordered dict, lazy expiry on read, LRU eviction |
| `Metrics.percentile` (20/09) | p50/p95 | nearest-rank on sorted samples |
| `np.percentile` | latency percentiles in `evaluate` | linear interpolation between ranks (slightly different from nearest-rank) |

## Execution trace (real run)
Real output of `HF_HUB_OFFLINE=1 .venv/bin/python 13_capstone_project/track_a_document_qa/run_pipeline.py` (model-loading bars and warnings removed; times vary by machine and by what else is running):
```
STAGE 1  OCR + routing + parse: 3 documents in 2.8s
   chai_kitchen_report  pages=2 elements=12  types={'title': 1, 'section_header': 3, 'text': 4, 'table': 2, 'page_footer': 2}
   mixed_pdf            pages=3 elements=9   types={'title': 1, 'text': 6, 'section_header': 2} page routes={'text layer': 6, 'OCR': 3}
   lumen_q3_report      pages=3 elements=29  types={'page_header': 3, 'title': 1, 'section_header': 5, 'text': 5, 'table': 4, 'caption': 4, 'figure': 1, 'page_footer': 3, 'list_item': 2, 'footnote': 1}
   mixed.pdf page 2 was an image-only scan; the OCR text we now hold: ['Invoice 4821 for Chai Kitchen', 'Total due: 1250 rupees', 'Paid by card on time.']

STAGE 2  layout-aware chunking (19/09 LayoutChunker, max 140 tokens)
   chai_kitchen_report  4 chunks {'text': 2, 'table': 2}
   mixed_pdf            3 chunks {'text': 3}
   lumen_q3_report      10 chunks {'text': 6, 'table': 3, 'figure': 1}
   one chunk (the contract between chunking and retrieval):
   {"chunk_id": "lumen_q3_report#2", "kind": "text", "section_path": "Q3 Infrastructure Report > 2. Availability", "pages": [1], "boxes": [{"page": 1, "bbox": [72, 412, 540, 440]}], "element_ids": ["e7"]}
   embed_text (what MiniLM sees): 'Q3 Infrastructure Report > 2. Availability\nVirginia had the lowest uptime in July because of a network provide'

STAGE 3  index: 17 chunks, vectors (17, 384), version cafb3fcb76, built in 0.4s (first call includes loading MiniLM)

STAGE 4  grounded answers (LLM backend: mock-extractive; set DOCQA_LLM_BASE_URL to use an OpenAI-compatible server)
   Q: What is the price of Green Tea?
   A: Product: Green Tea; Price: 320; Stock: 75   (grounded=True, 5.38 ms)
      [1] USED chai_kitchen_report p.1 bbox=[30.0, 232.5, 556.5, 366.0] score=0.5154 table
      [2]      chai_kitchen_report p.1 bbox=[30.8, 113.2, 562.5, 195.8] score=0.29 text
   Q: What is the total due on the invoice?
   A: Total due: 1250 rupees Paid by card on time.   (grounded=True, 5.01 ms)
      [1] USED mixed_pdf p.2 bbox=[18.8, 45.0, 185.2, 83.2] score=0.6608 text
      [2]      mixed_pdf p.3 bbox=[18.0, 25.5, 191.2, 68.4] score=0.6608 text
   Q: Who is the CEO of Lumen?
   A: I could not find this in the documents.   (grounded=False, 7.61 ms)
      [1]      lumen_q3_report p.2 bbox=[72, 530, 540, 720] score=0.2047 table
      [2]      chai_kitchen_report p.2 bbox=[30.0, 232.5, 556.5, 366.0] score=0.191 table
   same question again -> cached=True in 0.02 ms (module 20 TTLLRUCache, key scoped by index version)

STAGE 5  eval: 18 questions with gold (doc, page) + answer string; hit@k needs the right page AND the answer text in a top-k chunk
   setup           hit@1  hit@3    MRR  cite acc  answer acc   p50 ms   p95 ms
   dense            0.89   0.94   0.92      0.94        0.78      4.8      5.1
   dense+rerank     0.94   1.00   0.97      1.00        0.83     36.4     61.3
   questions that still miss something: ['m3', 'u2', 'u5'] (see README 'failure modes' for why)
   engine metrics: {'ingest.docs': 3, 'cache.miss': 39, 'cache.hit': 1, 'ask.latency_ms.p50': 5.5, 'ask.latency_ms.p95': 41.3}

wrote track_a_document_qa/artifacts/ (parsed_*.json, chunks.jsonl, eval_report.json)
```
Real output of `python -m pytest -v test_track_a.py`:
```
test_track_a.py::test_health_after_autoload PASSED [  7%]
test_track_a.py::test_ask_table_value_with_page_and_bbox_citation PASSED [ 15%]
test_track_a.py::test_scanned_page_is_answerable_after_ocr PASSED [ 23%]
test_track_a.py::test_refuses_when_not_in_documents PASSED [ 30%]
test_track_a.py::test_second_identical_ask_is_cached PASSED [ 38%]
test_track_a.py::test_ask_validation_errors_are_422 PASSED [ 46%]
test_track_a.py::test_ingest_parsed_document_contract PASSED [ 53%]
test_track_a.py::test_ingest_pdf_bytes_native_text PASSED [ 61%]
test_track_a.py::test_ingest_rejects_ambiguous_or_garbage_input PASSED [ 69%]
test_track_a.py::test_ask_on_empty_index_is_409 PASSED [ 76%]
test_track_a.py::test_eval_meets_floor_and_rerank_does_not_hurt PASSED [ 84%]
test_track_a.py::test_openai_compatible_llm_with_fake_transport_and_fallback PASSED [ 92%]
test_track_a.py::test_chunk_schema_and_unit_vectors PASSED [100%]
============================= 13 passed in 17.06s ==============================
```
Real calls against the running service (`uvicorn app:app --port 8011`):
```
$ curl -s localhost:8011/health
{"status":"ok","documents":{"chai_kitchen_report":{"n_elements":12,"n_pages":2,"n_chunks":4},"mixed_pdf":{"n_elements":9,"n_pages":3,"n_chunks":3},"lumen_q3_report":{"n_elements":29,"n_pages":3,"n_chunks":10}},"n_chunks":17,"index_version":"cafb3fcb76","llm":"mock-extractive","metrics":{"ingest.docs":3}}
$ curl -s -X POST localhost:8011/ask -d '{"question":"How many Tulsi Blend tins are in stock?","k":2}'
{"question":"How many Tulsi Blend tins are in stock?","answer":"Product: Tulsi Blend; Price: 290; Stock: 0","grounded":true,"model":"mock-extractive","citations":[{"rank":1,"doc_id":"chai_kitchen_report","page":1,"pages":[1],"bbox":[30.0,232.5,556.5,366.0],"chunk_id":"chai_kitchen_report#1","kind":"table","score":0.4743,"used":true,"snippet":" | Product | Price | Stock | |---|---|---| | Masala Chai | 250 | 120 | | Green Tea | 320 | 75 | | Tulsi Blend | 290 | 0 "},{"rank":2,"doc_id":"chai_kitchen_report","page":1,"p ...
$ curl -s -X POST localhost:8011/ask -d '{"question":"hi"}'
{"detail":[{"type":"string_too_short","loc":["body","question"],"msg":"String should have at least 3 characters","input":"hi","ctx":{"min_length":3}}]}
$ curl ingest bad bbox
{"detail":"parsed_document invalid at elements.0.bbox: Value error, bbox must be [x0, y0, x1, y1] with x0 < x1 and y0 < y1"}
```
- Page 2 of `mixed.pdf` has no text layer; the three OCR'd lines are exactly right, and `route` shows 3 elements via OCR and 6 via text layer. The OCR is not perfect elsewhere: the page footer was read as "Page 1 cif 2" (see README failure modes), harmless because footers are dropped.
- Retrieval vs generation: dense hit@3 is below 1.0 and the cross-encoder reaches 1.00 on these 18 questions; **answer accuracy is lower than hit@k** because the extractive mock sometimes picks the wrong sentence from the right chunk (`m3`, `u5`) or cannot aggregate (`u2`). That gap is what a real LLM would close.
- Rerank costs latency: p50 roughly 7-10x dense (varies per run) (it reads 8 passages per question). Both are small in absolute terms because the corpus is 17 chunks; the ratio is what carries over.
- The "CEO" question is refused (`grounded=False`) rather than answered with the nearest table row.
- Small sample: 18 questions means one question is 5.5 points; do not over-read differences of one question.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `--k 1` | hit@1 equals hit@k; rerank has nothing to reorder | the reranker only helps when the pool (8) is larger than what you return |
| `chunk_document(doc, max_tokens=60)` | the tables split into row groups, more chunks | 19/09's row-group rule repeats the header per group |
| `docqa.MockLLM(min_overlap=0.6)` | more refusals, lower answer accuracy | stricter grounding threshold trades recall for precision |
| set `DOCQA_LLM_BASE_URL` to a local OpenAI-compatible server | `model` changes in the response; answers become sentences with `[n]` | `OpenAICompatLLM` replaces `MockLLM`; retrieval is unchanged |
| change a chunk's text and re-ingest | `index_version` changes; the cache misses | cache keys are scoped by the version hash |
| `make_report_page(i, noise=25)` in `parse_report_pages` | OCR errors rise, some hits vanish | template OCR is brittle to noise; this is why 18/02 preprocessing exists |
| ask with `"doc_id": "mixed_pdf"` | only that document is searched | filter after over-fetching |

## Debugging practice
**BREAK IT 1:** send a document with a bad box.
```bash
curl -s -X POST localhost:8001/ingest -H 'content-type: application/json' \
  -d '{"parsed_document": {"doc_id":"d","source_file":"x","note":"n","pages":[{"page":1,"width":600,"height":480}],"elements":[{"id":"a","type":"text","text":"hi there","page":1,"bbox":[1,2,3],"reading_order":0}]}}'
```
Symptom: HTTP 422, `parsed_document invalid at elements.0.bbox: Value error, bbox must be [x0, y0, x1, y1] with x0 < x1 and y0 < y1`. Fix: emit four numbers in PDF points. (The test `test_ingest_parsed_document_contract` asserts this.)

**BREAK IT 2:** the same fact on two pages. Ask "What is the total due?" and the top citation may be page 2 or page 3 of `mixed_pdf` (identical scores in the trace: 0.6608). Not a bug in the code: the document contains the text twice. Fix: dedupe by content hash at ingest, or accept both pages in the gold set (done here).

**BREAK IT 3:** `ImportError: No module named 'common'` when running from another folder. Cause: something imported `docqa` without its `sys.path` setup, or the Dockerfile forgot a `COPY`. Fix: import `docqa` first; in Docker check the copied paths.

**BREAK IT 4:** answer quality drops after changing the chunker, but `hit@k` did not. Symptom: `answer acc` falls while `hit@k` is flat. Diagnosis: the chunk text changed what the extractive answerer can quote (table rows vs prose). Fix: evaluate retrieval and generation separately, as `evaluate` does.

## Common mistakes
- Copying code from 18/19 instead of importing it; the copies drift and the teaching version stops being the truth.
- Skipping validation "because I wrote both sides"; the first third-party parser breaks it.
- Reporting one latency number from a warm cache (`use_cache=False` in `evaluate` exists for this) or from one request.
- Counting a hit when the right text appears on the wrong page; the user clicks the citation and sees a different page.
- Treating an LLM's `[1]` as proof of grounding; check that the cited chunk actually contains the claim.
- Running several uvicorn workers with an in-memory index (RAM times workers, and each worker has its own index after `/ingest`).

## Rebuild checklist
1. Write `docqa.py` pieces in order: `step()` loaders, `parse_*`, `validate`, `chunk_document`, `Index`, `MockLLM`, `Engine`, `evaluate`.
2. Write `run_pipeline.py`: flags, five stage blocks, artifact dump.
3. Write `eval_questions.json` with gold `(doc, page)` lists and an answer string each.
4. Write `app.py`: lifespan, three endpoints, 422 mapping, 409 on empty index.
5. Write `test_track_a.py`: health, one citation test, scanned-page test, refusal, cache, 422s, contract test, PDF ingest.
6. Write the Dockerfile with the repo root as context and say why.
7. Run the pipeline, then pytest, then curl; copy numbers into your notes.

## Say it aloud
- "Each arrow between the modules is a schema, validated at the boundary, so a bad OCR result fails at ingest with the field name instead of corrupting retrieval."
- "I reused the OCR, chunking and retrieval code by import; the new code is glue, an API, tests and an evaluation."
- "Retrieval was strong on my 18 questions; answer accuracy was lower, because the mock answerer is extractive, so the weak link is generation and I measured that separately."
- "A cross-encoder rerank gave a higher hit rate for roughly eight times the latency, so I made it a request flag."
- "Citations carry page and bounding box from the OCR step through chunking to the response; if any stage drops the box the feature is gone."

## The new problem (cliffhanger)
The generator here is a mock. A real model has to be served: TTFT, inter-token latency, batching, memory and cost decide what you can afford. That is the subject of [Track B](../track_b_inference_report/README.md) and [module 21](../../21_llm_inference_engineering/); when you plug a served model into `DOCQA_LLM_BASE_URL`, that report is how you choose its configuration. For the interview itself, practise with the [mock interviews](../../mock_interviews/).
