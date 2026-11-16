"""
docqa.py -- the Track A pipeline as a small library. run_pipeline.py (CLI) and app.py (FastAPI) both import it.

    pixels / PDF bytes --OCR+route--> parsed_document --layout chunker--> chunks --embed--> index
        question --retrieve (+ optional rerank)--> top-k chunks --LLM (mock/extractive or OpenAI-compatible)--> answer + citations

Nothing from modules 14 / 18 / 19 / 20 is copied here: it is imported (or, for step files that are scripts and not libraries,
their *definitions* are executed without their demo code, see load_defs). This file only adds the glue and the contracts.
"""
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
M14, M18, M19, M20 = (REPO / n for n in ("14_rag_and_retrieval", "18_ocr_and_document_parsing", "19_chunking_deep_dive", "20_llm_system_lld"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")                       # never download: use the cached MiniLM models only
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for p in (M14, M18, M19, M20):
    sys.path.insert(0, str(p))

import common as c14                                                # noqa: E402  module 14: Embedder, dense_search, CrossEncoderReranker
import common19                                                     # noqa: E402  module 19: embed (cached), count_tokens, load_step
from docimg import make_report_page                                 # noqa: E402  module 18: synthetic report pages (pixels)
from dococr import binarise                                         # noqa: E402  module 18: Otsu binarisation
from ragkit.cache import MISS, TTLLRUCache, cache_key               # noqa: E402  module 20: TTL+LRU cache, scoped keys
from ragkit.ids import content_hash                                 # noqa: E402  module 20: stable content hashes
from ragkit.observability import Metrics, Tracer                    # noqa: E402  module 20: p50/p95 metrics, spans

PX_TO_PT = 0.75


# ------------------------------------------------------------------------------------------------ reuse of step scripts
def load_defs(path, stop_marker):
    """Execute only the DEFINITIONS of a step script (everything before `stop_marker`), skipping its demo code.
    Step files in 18 are teaching scripts, not libraries; this avoids copying them (and avoids re-running their prints/writes)."""
    src = Path(path).read_text(encoding="utf-8")
    if stop_marker not in src:
        raise RuntimeError(f"{path.name}: marker {stop_marker!r} not found; the step file changed, update load_defs() call")
    ns = {"__file__": str(path), "__name__": "step_defs_" + Path(path).parent.name}
    exec(compile(src.split(stop_marker)[0], str(path), "exec"), ns)
    return SimpleNamespace(**ns)


_DEFS = {}


def step(name):
    """Lazy-load the definitions of 18/07 (parse_page), 18/08 (read_pdf, route) and 18/10 (ParsedDocument, to_parsed_document)."""
    if name not in _DEFS:
        spec = {"07": (M18 / "07_reading_order_and_tables" / "main.py", "gray0, _ = make_report_page(0)"),
                "08": (M18 / "08_pdf_internals_and_routing" / "main.py", "# ---- 4. Build the three demo files"),
                "10": (M18 / "10_parsed_document_handoff" / "main.py", "if not STRUCT.exists():")}[name]
        _DEFS[name] = load_defs(*spec)
    return _DEFS[name]


def parsed_model():
    return step("10").ParsedDocument                                # the strict pydantic contract (extra="forbid", bbox checks ...)


# ------------------------------------------------------------------------------------------------ stage 1: OCR / parse -> parsed_document
def parse_report_pages(indices=(0, 1)):
    """Pixels -> parsed_document via 18/07 (XY-cut + template OCR + tables) and 18/10 (px -> pt, element types, schema)."""
    s07, s10 = step("07"), step("10")
    pages = []
    for i in indices:
        gray, _truth = make_report_page(i)
        elements, _ink = s07.parse_page(gray)
        pages.append(dict(page=len(pages) + 1, width=int(gray.shape[1]), height=int(gray.shape[0]), elements=elements))
    doc, meta = s10.to_parsed_document(dict(source="synthetic report", pages=pages))
    return doc, meta


def parse_pdf_bytes(data: bytes, doc_id: str, source_file: str = "upload.pdf"):
    """PDF bytes -> parsed_document. 18/08 reads the file and ROUTES each page: text layer (native / searchable scan) or OCR (scan).
    Each line becomes a `text` element; the first line of page 1 is the title, the first line of later pages a level-1 section header."""
    s08 = step("08")
    elements, meta, sizes = [], {}, []
    for pno, page in enumerate(s08.read_pdf(data), 1):
        kind, how, _lines = s08.route(page)
        W, H = page["width"], page["height"]
        sizes.append(dict(page=pno, width=W, height=H))
        boxes = []                                                   # (text, [x0,y0,x1,y1] in PDF points, top-left origin)
        if kind in ("native", "searchable_scan"):
            for sp in sorted(page["spans"], key=lambda s: -s["y"]):  # PDF y grows upward: larger y = higher on the page
                top = H - sp["y"] - sp["size"]
                boxes.append((sp["text"], [sp["x"], top, sp["x"] + 0.5 * sp["size"] * len(sp["text"]), top + 1.25 * sp["size"]]))
        elif kind == "scanned":
            gray = s08.decode_image(page["images"][0])
            ink = binarise(gray)
            k = W / gray.shape[1]                                    # image pixels -> PDF points for this page
            for ln in s08.OCR.read_region(ink):
                cols = np.flatnonzero(ink[ln["y0"]:ln["y1"]].any(axis=0))
                boxes.append((ln["text"], [cols[0] * k, ln["y0"] * k, (cols[-1] + 1) * k, ln["y1"] * k]))
        for n, (text, bb) in enumerate(boxes):
            if not text.strip():
                continue
            bb = [float(v) for v in (max(0.0, round(bb[0], 1)), max(0.0, round(bb[1], 1)), min(W, round(bb[2], 1)), min(H, round(bb[3], 1)))]
            if bb[2] <= bb[0] or bb[3] <= bb[1]:
                continue
            eid = f"p{pno}e{n}"
            first = n == 0
            if first and pno == 1:
                el = dict(id=eid, type="title", text=text)
            elif first:
                el = dict(id=eid, type="section_header", level=1, text=text)
            else:
                el = dict(id=eid, type="text", text=text)
            elements.append(dict(el, page=pno, bbox=bb, reading_order=n))
            meta[eid] = dict(route=how, page_kind=kind)
    doc = dict(doc_id=doc_id, source_file=source_file, note="Parsed by docqa.parse_pdf_bytes (18/08 routing: text layer or template OCR).",
               pages=sizes, elements=elements)
    return doc, meta


def parse_pdf_file(path, doc_id=None):
    return parse_pdf_bytes(Path(path).read_bytes(), doc_id or Path(path).stem, Path(path).name)


def load_uptime_sample():
    """Hand-written 3-page parsed document that ships with module 19/09: stands in for what Docling / Unstructured would return."""
    p = M19 / "09_layout_aware_chunking_from_parsed_docs" / "sample_parsed_document.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    return doc, {}


def validate(doc):
    """The handoff contract (18/10's pydantic model). Raises pydantic.ValidationError with a readable path on a bad document."""
    parsed_model().model_validate(doc)
    return doc


def build_default_corpus():
    """The three documents of the default corpus, all produced by module 18 (or its stand-in): returns {doc_id: (parsed_doc, meta)}."""
    pdf = M18 / "08_pdf_internals_and_routing" / "mixed.pdf"
    if not pdf.exists():                                             # 18/08 writes it; run that step once if the folder is fresh
        import subprocess
        subprocess.run([sys.executable, str(pdf.parent / "main.py")], check=True, capture_output=True)
    corpus = {}
    for doc, meta in (parse_report_pages(), parse_pdf_file(pdf, "mixed_pdf"), load_uptime_sample()):
        corpus[doc["doc_id"]] = (validate(doc), meta)
    return corpus


# ------------------------------------------------------------------------------------------------ stage 2: chunking
def layout_module():
    """19/09's LayoutChunker. Chosen over the plain-text strategies of 19/10 because it is the only one that keeps page + bbox
    (needed for citations) and it already contains the winners' ideas: section breadcrumb in the embedded text, tables kept whole."""
    return common19.load_step("09_layout_aware_chunking_from_parsed_docs")


def chunk_document(doc, max_tokens=140):
    m19 = layout_module()
    out = []
    for i, c in enumerate(m19.LayoutChunker(max_tokens).chunk(doc)):
        out.append(dict(chunk_id=f"{doc['doc_id']}#{i}", doc_id=doc["doc_id"], kind=c.kind, text=c.text, embed_text=c.embed_text,
                        section_path=c.section_path, pages=c.pages, boxes=c.boxes, element_ids=c.element_ids))
    return out


# ------------------------------------------------------------------------------------------------ stage 3: index + retrieval
class Index:
    """Chunks + their unit vectors (MiniLM, 384-d). Brute-force dot product (module 14/04): fine for thousands of chunks."""

    def __init__(self):
        self.chunks, self.vectors, self.docs = [], np.zeros((0, 384), dtype=np.float32), {}
        self._reranker = None

    @property
    def version(self):                                               # changes whenever the content changes -> cache keys change
        return content_hash("|".join(c["chunk_id"] + c["text"] for c in self.chunks), length=10)

    def add_document(self, doc, max_tokens=140):
        self.remove_document(doc["doc_id"])
        chunks = chunk_document(doc, max_tokens)
        vecs = common19.embed([c["embed_text"] for c in chunks])
        self.chunks += chunks
        self.vectors = np.vstack([self.vectors, vecs]) if len(self.vectors) else vecs
        self.docs[doc["doc_id"]] = dict(n_elements=len(doc["elements"]), n_pages=len(doc["pages"]), n_chunks=len(chunks))
        return chunks

    def remove_document(self, doc_id):
        keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
        self.chunks, self.vectors = [self.chunks[i] for i in keep], self.vectors[keep]
        self.docs.pop(doc_id, None)

    def search(self, question, k=3, rerank=False, pool=8, doc_id=None):
        """Dense top-`pool`, optionally re-scored by the cross-encoder (14/07), return top-k hits (dicts with a `score`)."""
        if not self.chunks:
            return []
        q = common19.embed([question])[0]
        n = pool if rerank else max(k, 1)
        ranked = c14.dense_search(q, self.vectors, max(n, k) * (3 if doc_id else 1))
        if doc_id:
            ranked = [(i, s) for i, s in ranked if self.chunks[i]["doc_id"] == doc_id]
        ranked = ranked[:max(n, k)]
        if rerank and ranked:
            if self._reranker is None:
                self._reranker = c14.CrossEncoderReranker()
            scores = self._reranker.score(question, [self.chunks[i]["embed_text"] for i, _ in ranked])
            ranked = sorted(zip([i for i, _ in ranked], map(float, scores)), key=lambda t: -t[1])
        return [dict(self.chunks[i], score=round(float(s), 4)) for i, s in ranked[:k]]


# ------------------------------------------------------------------------------------------------ stage 4: grounded answer
STOP = set("a an and are as at be by can do does for from how i if in is it its of on or the to was were what when where which who whose "
           "will with this that there their they we you your many much".split())


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def candidates(chunk):
    """Sentences an extractive answer may quote: prose sentences, or one linearised row per table row ("Product: Green Tea; Price: 320")."""
    if chunk["kind"] == "table":
        return [ln for ln in chunk["embed_text"].split("\n") if ": " in ln and "; " in ln]
    heading = chunk["section_path"].split(" > ")[-1]                   # a section header is NOT chunk text (it is the breadcrumb), but may hold the answer
    return [heading] + [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk["text"]) if s.strip()]


class MockLLM:
    """No model, no API: pick the candidate sentence with the highest word overlap with the question. Deterministic, free,
    and it can only quote the retrieved text, so every answer is grounded by construction (it refuses when overlap is too low)."""
    name = "mock-extractive"

    def __init__(self, min_overlap=0.34):
        self.min_overlap = min_overlap

    def answer(self, question, hits):
        q = set(words(question))
        best = (0.0, None, None)
        for rank, h in enumerate(hits):
            for s in candidates(h):
                sw = set(words(s))
                score = len(q & sw) / max(1, len(q)) + 0.001 * (len(hits) - rank)      # tiny tie-break: earlier hit wins
                if score > best[0]:
                    best = (score, rank, s)
        if best[1] is None or best[0] < self.min_overlap:
            return dict(answer="I could not find this in the documents.", used=[], grounded=False)
        return dict(answer=best[2], used=[best[1]], grounded=True)


class OpenAICompatLLM:
    """OPTIONAL real LLM through any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, llama.cpp server, Ollama ...).
    Enabled only when DOCQA_LLM_BASE_URL is set. `transport` is injectable so tests never touch the network. Falls back to the mock on error."""

    def __init__(self, base_url, model, api_key="", transport=None, timeout=30):
        self.base_url, self.model, self.api_key, self.timeout = base_url.rstrip("/"), model, api_key, timeout
        self.transport = transport or self._http
        self.name = f"openai-compatible:{model}"
        self.fallback = MockLLM()

    def _http(self, url, payload, headers):
        import urllib.request
        req = urllib.request.Request(url, json.dumps(payload).encode(), headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:                    # stdlib: no extra dependency
            return json.loads(r.read())

    def prompt(self, question, hits):
        src = "\n\n".join(f"[{i + 1}] (doc {h['doc_id']}, page {h['pages'][0]}) {h['text']}" for i, h in enumerate(hits))
        return [dict(role="system", content="Answer ONLY from the numbered sources. Cite like [1]. If the sources do not contain the answer, say you cannot find it."),
                dict(role="user", content=f"Sources:\n{src}\n\nQuestion: {question}")]

    def answer(self, question, hits):
        try:
            headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})}
            resp = self.transport(f"{self.base_url}/chat/completions", dict(model=self.model, temperature=0, messages=self.prompt(question, hits)), headers)
            text = resp["choices"][0]["message"]["content"].strip()
        except Exception as exc:                                                         # network, auth, bad JSON: degrade, do not 500
            out = self.fallback.answer(question, hits)
            return dict(out, note=f"LLM call failed ({type(exc).__name__}); used mock")
        used = sorted({int(n) - 1 for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
        return dict(answer=text, used=used, grounded=bool(used))


def make_llm():
    base = os.environ.get("DOCQA_LLM_BASE_URL")
    if base:
        return OpenAICompatLLM(base, os.environ.get("DOCQA_LLM_MODEL", "gpt-4o-mini"), os.environ.get("DOCQA_LLM_API_KEY", ""))
    return MockLLM()


# ------------------------------------------------------------------------------------------------ the engine (ingest + ask)
def citation(hit, rank, used):
    b = hit["boxes"][0]
    return dict(rank=rank + 1, doc_id=hit["doc_id"], page=b["page"], pages=hit["pages"], bbox=[round(v, 1) for v in b["bbox"]],
                chunk_id=hit["chunk_id"], kind=hit["kind"], score=hit["score"], used=used, snippet=hit["text"][:120].replace("\n", " "))


class Engine:
    def __init__(self, llm=None, cache_ttl=300.0):
        self.index, self.llm = Index(), llm or make_llm()
        self.cache = TTLLRUCache(capacity=256, ttl=cache_ttl)                            # module 20 step 07: exact-match answer cache
        self.metrics, self.tracer = Metrics(), Tracer()                                  # module 20 step 09

    def ingest(self, doc, meta=None):
        validate(doc)
        t0 = time.perf_counter()
        with self.tracer.span("ingest", doc_id=doc["doc_id"]):
            chunks = self.index.add_document(doc)
        self.metrics.inc("ingest.docs")
        return dict(doc_id=doc["doc_id"], n_elements=len(doc["elements"]), n_chunks=len(chunks), index_version=self.index.version,
                    ms=round((time.perf_counter() - t0) * 1000, 1))

    def ingest_default(self):
        return [self.ingest(d, m) for d, m in build_default_corpus().values()]

    def ask(self, question, k=3, rerank=False, doc_id=None, use_cache=True):
        t0 = time.perf_counter()
        key = cache_key("docqa", question, dict(k=k, rerank=rerank, doc_id=doc_id), scope=self.index.version + self.llm.name)
        hit_cache = self.cache.get(key) if use_cache else MISS             # eval bypasses the cache so latencies are real
        if hit_cache is not MISS:
            self.metrics.inc("cache.hit")
            return dict(hit_cache, cached=True, latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        self.metrics.inc("cache.miss")
        with self.tracer.span("ask", question=question[:60]):
            with self.tracer.span("retrieve", rerank=rerank):
                hits = self.index.search(question, k, rerank, doc_id=doc_id)
            with self.tracer.span("generate", llm=self.llm.name):
                gen = self.llm.answer(question, hits)
        cites = [citation(h, i, i in gen["used"]) for i, h in enumerate(hits)]
        out = dict(question=question, answer=gen["answer"], grounded=gen["grounded"], model=self.llm.name,
                   citations=cites, index_version=self.index.version, **({"note": gen["note"]} if "note" in gen else {}))
        self.cache.put(key, out)
        ms = (time.perf_counter() - t0) * 1000
        self.metrics.observe("ask.latency_ms", ms)
        return dict(out, cached=False, latency_ms=round(ms, 2))


# ------------------------------------------------------------------------------------------------ evaluation
def load_questions():
    return json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))


def evaluate(engine, k=3, rerank=False, questions=None):
    """hit@k: a top-k chunk is on a gold (doc, page) AND contains the gold answer string.
    citation accuracy: the answer's cited chunk (top `used`) is on a gold page. answer match: the gold string is inside the answer."""
    qs = questions or load_questions()
    rows, lat = [], []
    for q in qs:
        gold = {(g["doc_id"], g["page"]) for g in q["gold"]}
        eng_t = time.perf_counter()
        out = engine.ask(q["question"], k=k, rerank=rerank, use_cache=False)
        lat.append((time.perf_counter() - eng_t) * 1000)
        norm = lambda s: " ".join(s.lower().split())
        hit_rank = next((c["rank"] for c in out["citations"] if (c["doc_id"], c["page"]) in gold
                         and norm(q["answer"]) in norm(next(h for h in engine.index.chunks if h["chunk_id"] == c["chunk_id"])["embed_text"])), None)
        used = [c for c in out["citations"] if c["used"]]
        rows.append(dict(id=q["id"], hit_rank=hit_rank, hit_at_1=hit_rank == 1, hit_at_k=hit_rank is not None,
                         cite_ok=bool(used) and (used[0]["doc_id"], used[0]["page"]) in gold,
                         answer_ok=norm(q["answer"]) in norm(out["answer"]), latency_ms=lat[-1]))
    n = len(rows)
    return dict(n=n, k=k, rerank=rerank, hit_at_1=sum(r["hit_at_1"] for r in rows) / n, hit_at_k=sum(r["hit_at_k"] for r in rows) / n,
                mrr=sum(1 / r["hit_rank"] for r in rows if r["hit_rank"]) / n, citation_acc=sum(r["cite_ok"] for r in rows) / n,
                answer_acc=sum(r["answer_ok"] for r in rows) / n, p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)),
                rows=rows)


def pdf_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
