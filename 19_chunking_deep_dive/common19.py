"""
common19.py -- shared helpers for module 19 (chunking deep dive).

What lives here (taught inside the step folders, imported by later steps):
    count_tokens / token_spans   -> taught in step 01 (the REAL MiniLM tokenizer)
    sentence_spans               -> a regex sentence splitter that keeps character offsets
    RawChunk                     -> the light chunk record used by steps 01-10 (step 11 builds the production one)
    load_docs / load_eval        -> 13 documents (9 from module 14 + 4 new messy ones) and 74 labelled questions
    evaluate                     -> the ONE evaluation harness every strategy is scored with (hit@k, MRR, recall@k, tokens)
    load_step                    -> import another step's main.py by path (folder names start with digits)

Relevance is defined by CHARACTER OFFSETS, not by chunk ids: a chunk is relevant to a question if the chunk's
source span fully covers the answer sentence. That is what lets us compare chunkers that cut the text in
totally different places.
"""
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
M14 = REPO / "14_rag_and_retrieval"
sys.path.insert(0, str(M14))
import common as c14   # noqa: E402  (module 14's helpers: Embedder, BM25, rrf_fuse, load_documents)

DATA = HERE / "data"
EMBED_MODEL = c14.EMBED_MODEL


# ------------------------------------------------------------------ tokens (step 01) -------------------
@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    tok.model_max_length = 10**9            # silence the "longer than max length" warning; we measure on purpose
    return tok


@lru_cache(maxsize=200_000)
def count_tokens(text: str) -> int:
    """Number of WordPiece tokens the embedding model sees for `text` (special [CLS]/[SEP] not counted)."""
    return len(get_tokenizer()(text, add_special_tokens=False)["input_ids"])


def token_spans(text: str):
    """[(char_start, char_end), ...] one pair per token: lets us cut by tokens but return exact text slices."""
    enc = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["offset_mapping"]


# ------------------------------------------------------------------ chunk record -----------------------
@dataclass
class RawChunk:
    doc_id: str
    start: int                    # char offset in the ORIGINAL document where the source region starts
    end: int                      # ... and ends (exclusive)
    text: str                     # what was cut out
    embed_text: str = ""          # what gets embedded / BM25-indexed (text + header/context); default = text
    context_text: str = ""        # what is sent to the LLM when retrieved; default = text (parent text for small-to-big)
    spans: list = field(default_factory=list)   # source spans actually contained in context_text
    section_path: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.embed_text = self.embed_text or self.text
        self.context_text = self.context_text or self.text
        self.spans = self.spans or [(self.start, self.end)]


def make_chunk(doc, start, end, **kw):
    """Chunk from a slice of the document; strips whitespace and moves the offsets with it."""
    raw = doc["text"][start:end]
    lead = len(raw) - len(raw.lstrip())
    text = raw.strip()
    return RawChunk(doc["doc_id"], start + lead, start + lead + len(text), text, **kw)


# ------------------------------------------------------------------ sentences --------------------------
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`\"(*|])|\n+")


def sentence_spans(text: str, base=0):
    """[(start, end)] of sentences; boundaries = end punctuation + space + capital, or any newline."""
    spans, pos = [], 0
    for m in list(_SENT.finditer(text)) + [None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            spans.append((base + pos + lead, base + pos + lead + len(seg.strip())))
        if m:
            pos = m.end()
    return spans


# ------------------------------------------------------------------ corpus + eval ----------------------
def load_docs():
    """9 module-14 documents + 4 new messy documents. Each: {doc_id, title, text, corpus}."""
    docs = [dict(d, corpus="kb") for d in c14.load_documents()]
    for p in sorted(DATA.glob("*")):
        if p.suffix in (".md", ".txt"):
            t = p.read_text(encoding="utf-8")
            first = t.splitlines()[0]
            docs.append({"doc_id": p.stem, "title": first.lstrip("# ").strip(), "text": t, "corpus": "new"})
    return docs


def load_eval():
    """All questions in one format: {id, question, doc_id, answers:[exact strings], kind, corpus}."""
    docs = {d["doc_id"]: d["text"] for d in load_docs()}
    out = []
    for q in c14.load_eval_set():
        if not q["gold"]:
            continue                                         # skip the unanswerable ones: no span to find
        doc_id, section = q["gold"][0].split("#")
        text = docs[doc_id]
        s0 = text.index("## " + section)                     # restrict the search to the gold section
        s1 = text.find("\n## ", s0 + 3)
        body = text[s0: s1 if s1 != -1 else len(text)]
        i = body.lower().find(q["answer"].lower())
        out.append({"id": q["id"], "question": q["question"], "doc_id": doc_id,
                    "answers": [body[i:i + len(q["answer"])]], "kind": q["kind"], "corpus": "kb"})
    for q in json.loads((DATA / "eval_questions.json").read_text(encoding="utf-8")):
        out.append(dict(q, corpus="new"))
    return out


def answer_spans(q, docs_by_id):
    text = docs_by_id[q["doc_id"]]["text"]
    return [(text.index(a), text.index(a) + len(a)) for a in q["answers"]]


# ------------------------------------------------------------------ embedding + retrieval --------------
_EMB = None
_CACHE = {}


def get_embedder():
    global _EMB
    if _EMB is None:
        _EMB = c14.Embedder(max_length=256)
    return _EMB


def embed(texts):
    """Embed with a per-process cache keyed by text (the same string is never embedded twice)."""
    need = [t for t in dict.fromkeys(texts) if t not in _CACHE]
    if need:
        for t, v in zip(need, get_embedder().encode(need)):
            _CACHE[t] = v
    return np.vstack([_CACHE[t] for t in texts])


def covers(chunk, doc_id, span):
    return chunk.doc_id == doc_id and any(s <= span[0] and span[1] <= e for s, e in chunk.spans)


def _merge(spans):
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def char_precision(ctxs, answer_spans_, doc_id):
    """Context precision: share of the characters sent to the LLM (top contexts of the right document, source spans only)
    that are the answer evidence. Low = you pay for text that does not help. (A cheap chunk-attribution metric.)"""
    ctx = _merge([sp for c in ctxs if c.doc_id == doc_id for sp in c.spans])
    total = sum(e - s for s, e in ctx) + sum(len(c.context_text) for c in ctxs if c.doc_id != doc_id)
    gold = sum(max(0, min(e, ge) - max(s, gs)) for s, e in ctx for gs, ge in _merge(answer_spans_))
    return gold / total if total else 0.0


def evaluate(chunks, questions=None, docs=None, ks=(1, 3, 5), vectors=None, bm25_texts=None, postprocess=None):
    """Score a list of RawChunk on all questions. Returns {retriever: {metric: value}} plus size numbers.

    hit@k    : 1 if ANY answer sentence is covered by a top-k chunk
    recall@k : share of the answer sentences covered by the union of the top-k chunks
    MRR      : 1 / rank of the first covering chunk
    ctx_tok  : tokens of the top-3 context_text strings (what you would pay to send to the LLM)
    """
    docs = docs or load_docs()
    questions = questions or load_eval()
    by_id = {d["doc_id"]: d for d in docs}
    matrix = vectors if vectors is not None else embed([c.embed_text for c in chunks])   # (n_chunks, 384); late chunking passes its own
    bm25 = c14.BM25(bm25_texts or [c.embed_text for c in chunks])   # contextual BM25 passes its own texts
    qvecs = embed([q["question"] for q in questions])
    rows = {"dense": [], "bm25": [], "hybrid": []}
    per_q = {}
    for q, qv in zip(questions, qvecs):
        spans = answer_spans(q, by_id)
        dense = c14.dense_search(qv, matrix, k=min(20, len(chunks)))
        sparse = bm25.search(q["question"], 20)
        hybrid = c14.rrf_fuse([sparse, dense], top_n=20)
        for name, ranking in (("dense", dense), ("bm25", sparse), ("hybrid", hybrid)):
            order = [j for j, _ in ranking]
            # postprocess (optional) turns the ranked chunk ids into the CONTEXT objects actually sent to the LLM:
            # parent lookup + de-duplication (small-to-big), merging siblings (auto-merging), ...
            ctxs = postprocess(order) if postprocess else [chunks[j] for j in order]
            cov = [[covers(c, q["doc_id"], sp) for sp in spans] for c in ctxs]  # (rank, span)
            first = next((r for r, c in enumerate(cov, 1) if any(c)), None)
            row = {"mrr": 1 / first if first else 0.0, "rank": first,
                   "ctx_tok": sum(count_tokens(c.context_text) for c in ctxs[:3]),
                   "prec3": char_precision(ctxs[:3], spans, q["doc_id"])}
            for k in ks:
                row[f"hit@{k}"] = float(any(any(c) for c in cov[:k]))
                row[f"recall@{k}"] = float(np.mean([any(c[i] for c in cov[:k]) for i in range(len(spans))]))
            rows[name].append(dict(row, kind=q["kind"], corpus=q["corpus"], id=q["id"]))
        per_q[q["id"]] = rows["dense"][-1]["rank"]
    res = {}
    for name, rs in rows.items():
        res[name] = {m: float(np.mean([r[m] for r in rs])) for m in rs[0] if m not in ("kind", "corpus", "id", "rank")}
        res[name]["rows"] = rs
    res["n_chunks"] = len(chunks)
    res["stored_tok"] = sum(count_tokens(c.embed_text) for c in chunks)
    res["mean_tok"] = res["stored_tok"] / max(len(chunks), 1)
    res["per_q_rank"] = per_q
    return res


def table(rows, headers):
    """Tiny fixed-width table printer: rows = list of lists."""
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = lambda r: "  ".join(str(x).ljust(w[i]) if i == 0 else str(x).rjust(w[i]) for i, x in enumerate(r))
    return "\n".join([line(headers), line(["-" * x for x in w])] + [line(r) for r in rows])


def chunk_id(doc_id, start, end, text):
    return hashlib.sha1(f"{doc_id}|{start}|{end}|{text}".encode()).hexdigest()[:12]


def load_step(folder):
    """Import <folder>/main.py by file path (so step 10 can reuse the chunkers written in steps 01-08)."""
    path = HERE / folder / "main.py"
    spec = importlib.util.spec_from_file_location("m_" + folder, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def sparkline(values, lo=None, hi=None):
    bars = "▁▂▃▄▅▆▇█"
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    return "".join(bars[min(7, int((v - lo) / (hi - lo + 1e-9) * 8))] for v in values)
