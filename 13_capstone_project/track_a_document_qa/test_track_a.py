"""
test_track_a.py -- pytest for Track A (FastAPI TestClient: no server process, no network, no paid API).

RUN (from this folder):  python -m pytest -q test_track_a.py
RUN (from repo root):    HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q 13_capstone_project/track_a_document_qa/test_track_a.py
NEEDS: pytest, httpx (TestClient), the cached MiniLM models.  ~25 s on CPU (the module-scoped fixture builds the index once).
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docqa  # noqa: E402
from app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:                               # `with` runs the lifespan: engine created, built-in corpus ingested
        yield c


def test_health_after_autoload(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and set(h["documents"]) == {"chai_kitchen_report", "mixed_pdf", "lumen_q3_report"}
    assert h["n_chunks"] == 17 and h["llm"] == "mock-extractive"


def test_ask_table_value_with_page_and_bbox_citation(client):
    r = client.post("/ask", json={"question": "What is the price of Green Tea?", "k": 3}).json()
    assert "320" in r["answer"] and r["grounded"] is True
    used = [c for c in r["citations"] if c["used"]]
    assert used[0]["doc_id"] == "chai_kitchen_report" and used[0]["page"] == 1
    x0, y0, x1, y1 = used[0]["bbox"]
    assert 0 <= x0 < x1 <= 600 and 0 <= y0 < y1 <= 480       # a real box inside the 600 x 480 pt page


def test_scanned_page_is_answerable_after_ocr(client):
    r = client.post("/ask", json={"question": "What is the total due on the invoice?", "k": 3}).json()
    top = r["citations"][0]
    assert top["doc_id"] == "mixed_pdf" and top["page"] in (2, 3)    # page 2 is image-only: this text exists only because OCR ran
    assert "1250" in r["answer"]


def test_refuses_when_not_in_documents(client):
    r = client.post("/ask", json={"question": "Who is the CEO of Lumen?"}).json()
    assert r["grounded"] is False and "could not find" in r["answer"]


def test_second_identical_ask_is_cached(client):
    body = {"question": "How often is uptime measured?", "k": 2}
    first, second = client.post("/ask", json=body).json(), client.post("/ask", json=body).json()
    assert first["cached"] is False and second["cached"] is True and second["answer"] == first["answer"]


def test_ask_validation_errors_are_422(client):
    assert client.post("/ask", json={"question": "ab"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "k": 99}).status_code == 422


def test_ingest_parsed_document_contract(client):
    doc = {"doc_id": "memo", "source_file": "memo.txt", "note": "hand written", "pages": [{"page": 1, "width": 600, "height": 480}],
           "elements": [{"id": "m1", "type": "text", "text": "The cafeteria opens at eight thirty on Mondays.", "page": 1,
                         "bbox": [50, 50, 400, 80], "reading_order": 0}]}
    ok = client.post("/ingest", json={"parsed_document": doc})
    assert ok.status_code == 200 and ok.json()["ingested"][0]["n_chunks"] == 1
    a = client.post("/ask", json={"question": "When does the cafeteria open?", "doc_id": "memo"}).json()
    assert "eight thirty" in a["answer"] and a["citations"][0]["bbox"] == [50, 50, 400, 80]
    bad = json.loads(json.dumps(doc))
    bad["elements"][0]["bbox"] = [1, 2, 3]
    r = client.post("/ingest", json={"parsed_document": bad})
    assert r.status_code == 422 and "bbox" in r.json()["detail"]


def test_ingest_pdf_bytes_native_text(client):
    s08 = docqa.step("08")
    pdf = s08.Pdf()
    pdf.add_page(600, 480, [(72, 80, 16, "Fire drill report"), (72, 108, 16, "The assembly point is the north car park.")])
    r = client.post("/ingest", json={"pdf_base64": base64.b64encode(pdf.tobytes()).decode(), "doc_id": "drill"})
    assert r.status_code == 200, r.text
    a = client.post("/ask", json={"question": "Where is the assembly point?", "doc_id": "drill"}).json()
    assert "north car park" in a["answer"] and a["citations"][0]["page"] == 1


def test_ingest_rejects_ambiguous_or_garbage_input(client):
    assert client.post("/ingest", json={}).status_code == 422
    assert client.post("/ingest", json={"pdf_base64": "not base64!!", "doc_id": "x"}).status_code == 422
    assert client.post("/ingest", json={"pdf_base64": base64.b64encode(b"hello").decode(), "doc_id": "x"}).status_code == 422


def test_ask_on_empty_index_is_409():
    with TestClient(app) as c:
        for d in list(c.app.state.engine.index.docs):
            c.app.state.engine.index.remove_document(d)
        assert c.post("/ask", json={"question": "anything at all"}).status_code == 409


def test_eval_meets_floor_and_rerank_does_not_hurt():
    eng = docqa.Engine()
    eng.ingest_default()
    dense, rr = docqa.evaluate(eng, 3, False), docqa.evaluate(eng, 3, True)
    assert dense["hit_at_k"] >= 0.85 and dense["citation_acc"] >= 0.85
    assert rr["hit_at_k"] >= dense["hit_at_k"] and rr["mrr"] >= dense["mrr"] - 0.02


def test_openai_compatible_llm_with_fake_transport_and_fallback():
    hits = [dict(doc_id="d", pages=[1], text="Refunds take nine days.", kind="text", section_path="x", embed_text="x", boxes=[], chunk_id="d#0")]
    seen = {}

    def fake(url, payload, headers):
        seen.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "Nine days [1]."}}]}

    llm = docqa.OpenAICompatLLM("http://llm.local/v1/", "m", "sk-test", transport=fake)
    out = llm.answer("How long do refunds take?", hits)
    assert out["answer"] == "Nine days [1]." and out["used"] == [0] and out["grounded"]
    assert seen["url"] == "http://llm.local/v1/chat/completions" and seen["headers"]["Authorization"] == "Bearer sk-test"
    assert "[1] (doc d, page 1)" in seen["payload"]["messages"][1]["content"]

    def boom(*a):
        raise TimeoutError("down")

    out = docqa.OpenAICompatLLM("http://x", "m", transport=boom).answer("How long do refunds take?", hits)
    assert "mock" in out["note"] and out["grounded"]            # degraded to the extractive mock instead of a 500


def test_chunk_schema_and_unit_vectors():
    eng = docqa.Engine()
    eng.ingest_default()
    c = eng.index.chunks[0]
    assert set(c) == {"chunk_id", "doc_id", "kind", "text", "embed_text", "section_path", "pages", "boxes", "element_ids"}
    assert np.allclose(np.linalg.norm(eng.index.vectors, axis=1), 1, atol=1e-4)
