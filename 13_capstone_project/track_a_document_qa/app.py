"""
app.py -- Track A as a web service: FastAPI with /health, /ingest and /ask (patterns from 09_api_development: lifespan loads heavy
objects once, Pydantic request/response models, 422 on bad input, plain `def` handlers so CPU work runs in the thread pool).

RUN (dev, from this folder):   uvicorn app:app --port 8001            docs: http://127.0.0.1:8001/docs
RUN (from repo root):          HF_HUB_OFFLINE=1 .venv/bin/python -m uvicorn app:app --app-dir 13_capstone_project/track_a_document_qa --port 8001
TRY:    curl -s localhost:8001/health
        curl -s -X POST localhost:8001/ask -H 'content-type: application/json' -d '{"question":"What is the price of Green Tea?","k":3}'
ENV:    DOCQA_AUTOLOAD=1 (default) ingests the built-in corpus at startup; DOCQA_AUTOLOAD=0 starts empty (POST /ingest {"builtin": true}).
        DOCQA_LLM_BASE_URL / DOCQA_LLM_MODEL / DOCQA_LLM_API_KEY switch from the mock LLM to an OpenAI-compatible server.
NEEDS: fastapi, uvicorn, pydantic + the repo modules 14/18/19/20 on disk (see Dockerfile: the whole repo subset is copied in).
"""
import base64
import binascii
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

import docqa


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = docqa.Engine()                       # one engine per worker process: index, cache, metrics live here
    if os.environ.get("DOCQA_AUTOLOAD", "1") == "1":
        app.state.engine.ingest_default()
    yield


app = FastAPI(title="Document QA (Track A)", version="1.0", lifespan=lifespan)


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


@app.get("/health")
def health():
    e = app.state.engine
    return {"status": "ok", "documents": e.index.docs, "n_chunks": len(e.index.chunks), "index_version": e.index.version,
            "llm": e.llm.name, "metrics": e.metrics.snapshot()}


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


def _safe_ingest(engine, doc, meta):
    try:
        return engine.ingest(doc, meta)
    except ValidationError as exc:                           # the contract failed: tell the caller which field, as 422
        first = exc.errors()[0]
        raise HTTPException(422, f"parsed_document invalid at {'.'.join(str(x) for x in first['loc'])}: {first['msg']}")
    except Exception as exc:
        raise HTTPException(422, f"parsed_document rejected: {type(exc).__name__}: {exc}")


@app.post("/ask")
def ask(req: AskRequest):
    e = app.state.engine
    if not e.index.chunks:
        raise HTTPException(409, "index is empty: POST /ingest first")
    return e.ask(req.question, k=req.k, rerank=req.rerank, doc_id=req.doc_id)
