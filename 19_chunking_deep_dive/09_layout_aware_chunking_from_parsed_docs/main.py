"""
GOAL:  Chunk a PARSED document: a list of layout elements {type, text, page, bbox, reading_order} (what Docling,
       Unstructured or an OCR + layout pipeline hands over), not a string. Rules: sort by reading order, drop page
       headers/footers, start a chunk at every section header, keep tables whole (or row groups with the header
       repeated), merge a table that continues on the next page, bind captions to tables and figures, and keep
       page numbers + bounding boxes so every answer can be cited and highlighted in the PDF.

       Input here is sample_parsed_document.json (hand-written, 3 pages). Module 18 produces a similar file at
       18_ocr_and_document_parsing/10_parsed_document_handoff/parsed_document.json; this step does NOT depend on it.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/main.py
EXPECTED OUTPUT (shortened):
    A) elements by type/page, reading order fixed     B) layout-aware chunks with page + bbox citations
    C) naive flatten-and-window chunks (what goes wrong)    D) retrieval + citation check on 7 questions
NEEDS: torch, transformers (cached MiniLM).  ~10 s.
"""
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common19 import count_tokens, embed, table  # noqa: E402
import common as c14  # noqa: E402  (module 14: dense_search)

NOISE = {"page_header", "page_footer"}


@dataclass
class LChunk:
    kind: str                                   # text | table | figure
    text: str                                   # display text (markdown table / paragraph)
    embed_text: str                             # what is embedded: breadcrumb + caption + linearised rows
    section_path: str
    pages: list
    boxes: list                                 # [{"page": 2, "bbox": [x0, y0, x1, y1]}] union box per page
    element_ids: list
    meta: dict = field(default_factory=dict)

    @property
    def citation(self):
        b = self.boxes[0]
        return f"p.{'-'.join(str(p) for p in sorted(set(self.pages)))} @{[round(v) for v in b['bbox']]}"


def load(path=HERE / "sample_parsed_document.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def union_boxes(elements):
    """One box per page covering all the elements of the chunk (what a PDF viewer highlights)."""
    per_page = {}
    for e in elements:
        x0, y0, x1, y1 = e["bbox"]
        o = per_page.setdefault(e["page"], [x0, y0, x1, y1])
        per_page[e["page"]] = [min(o[0], x0), min(o[1], y0), max(o[2], x1), max(o[3], y1)]
    return [{"page": p, "bbox": b} for p, b in sorted(per_page.items())]


def merge_continued_tables(elements):
    """A table that starts on page N and continues on N+1 arrives as two elements with the same header: merge them."""
    out = []
    for e in elements:
        prev = out[-1] if out else None
        if e["type"] == "table" and prev and prev["type"] == "table" and prev["table"]["header"] == e["table"]["header"] \
                and e["page"] == prev["page"] + 1:
            prev = dict(prev, table=dict(prev["table"], rows=prev["table"]["rows"] + e["table"]["rows"]),
                        pages=sorted(set(prev.get("pages", [prev["page"]]) + [e["page"]])), merged_ids=prev.get("merged_ids", [prev["id"]]) + [e["id"]],
                        bboxes=prev.get("bboxes", [prev]) + [e])
            out[-1] = prev
        else:
            out.append(e)
    return out


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def linearise(header, row):
    """Row -> sentence-like text with the column names repeated: embedders read this better than '| 99.97 | 99.99 |'."""
    return "; ".join(f"{h}: {v}" for h, v in zip(header, row))


class LayoutChunker:
    def __init__(self, max_tokens=140):
        self.max = max_tokens

    def chunk(self, doc):
        els = sorted(doc["elements"], key=lambda e: (e["page"], e["reading_order"]))       # 1. reading order, NOT array order
        footer_pages = {e["page"]: e["text"] for e in els if e["type"] == "page_footer"}
        els = merge_continued_tables([e for e in els if e["type"] not in NOISE])           # 2. drop noise, 3. merge tables
        title, section, buf, chunks = "", [], [], []
        i = 0

        def flush():
            if buf:
                text = " ".join(e["text"] for e in buf)
                path = " > ".join([title] + section)
                chunks.append(LChunk("text", text, f"{path}\n{text}", path, sorted({e["page"] for e in buf}), union_boxes(buf),
                                     [e["id"] for e in buf]))
                buf.clear()

        while i < len(els):
            e = els[i]
            nxt = els[i + 1] if i + 1 < len(els) else None
            if e["type"] == "title":
                title = e["text"]
            elif e["type"] == "section_header":                                            # 4. a header closes the chunk
                flush()
                section = section[:e["level"] - 1] + [e["text"]]
            elif e["type"] == "footnote":                                                  # footnotes stay on their own
                flush()
                buf.append(e)
                flush()
            elif e["type"] in ("text", "list_item"):
                if buf and count_tokens(" ".join(x["text"] for x in buf + [e])) > self.max:
                    flush()
                buf.append(e)
            elif e["type"] == "caption" and nxt and nxt["type"] in ("table", "figure"):    # 5. caption ABOVE its object (tables)
                flush()
                chunks += self.object_chunks(nxt, e, title, section)
                i += 1
            elif e["type"] in ("table", "figure"):
                flush()
                below = nxt if nxt and nxt["type"] == "caption" else None                 # caption BELOW the object (figures)
                chunks += self.object_chunks(e, below, title, section)
                i += 1 if below else 0
            elif e["type"] == "caption":
                buf.append(e)
            i += 1
        flush()
        for c in chunks:
            c.meta["footers"] = [footer_pages.get(p) for p in c.pages]
        return chunks

    def object_chunks(self, obj, caption, title, section):
        path = " > ".join([title] + section)
        cap = caption["text"] if caption else ""
        els = ([caption] if caption else []) + [obj]
        pages = obj.get("pages", [obj["page"]])
        boxes = union_boxes(els + obj.get("bboxes", [])[1:])
        ids = [x["id"] for x in els] + obj.get("merged_ids", [obj["id"]])[1:]
        if obj["type"] == "figure":                                                        # figures: caption + alt text
            text = f"{cap} [figure: {obj['figure']['alt']}]"
            return [LChunk("figure", text, f"{path}\n{text}", path, pages, boxes, ids)]
        header, rows = obj["table"]["header"], obj["table"]["rows"]
        whole = md_table(header, rows)
        if count_tokens(cap + whole) <= self.max:                                           # keep the table whole
            emb = f"{path}\n{cap}\n" + "\n".join(linearise(header, r) for r in rows)
            return [LChunk("table", f"{cap}\n{whole}", emb, path, pages, boxes, ids, {"rows": len(rows)})]
        out, group = [], []                                                                 # too big: row groups, header repeated
        for r in rows + [None]:
            if r is None or (group and count_tokens(cap + md_table(header, group + [r])) > self.max):
                emb = f"{path}\n{cap}\n" + "\n".join(linearise(header, g) for g in group)
                out.append(LChunk("table", f"{cap}\n{md_table(header, group)}", emb, path, pages, boxes, ids, {"rows": len(group)}))
                group = []
            if r is not None:
                group.append(r)
        return out


def naive_chunks(doc, window=60):
    """What a text-only pipeline does: concatenate every element's text (cells flattened) and cut fixed windows."""
    words = []
    for e in doc["elements"]:                                                              # array order, noise included
        if e["type"] == "table":
            words += [c for row in [e["table"]["header"]] + e["table"]["rows"] for c in row]
        else:
            words += e["text"].split()
    return [" ".join(words[i:i + window]) for i in range(0, len(words), window)]


QUESTIONS = [
    ("What was the uptime of Dublin in September?", "e6", "99.93"),
    ("Which service's cost grew the most compared with Q2?", "e13", "+18%"),
    ("Who owns the multi-region failover test and when is it due?", "e21", "30 June"),
    ("What is the due date of the combined dashboard?", "e20", "5 April"),
    ("How much of compute capacity is reserved?", "e15", "60 percent"),
    ("What happened to queue depth during the quarter?", "e8", "20 jobs"),
    ("How often is uptime measured?", "e23", "30 seconds"),
]


def main():
    doc = load()
    els = doc["elements"]
    print("A) parsed document:", doc["source_file"], "| elements by type:", dict(Counter(e["type"] for e in els)))
    print("   array order on page 1 :", [e["id"] for e in els if e["page"] == 1][:7], "(e6/e5 swapped by the parser)")
    srt = sorted(els, key=lambda e: (e["page"], e["reading_order"]))
    print("   reading_order sorted  :", [e["id"] for e in srt if e["page"] == 1][:7])

    chunks = LayoutChunker(140).chunk(doc)
    print(f"\nB) layout-aware chunks (max 140 tokens): {len(chunks)} chunks, kinds {dict(Counter(c.kind for c in chunks))}")
    for c in chunks:
        print(f"   [{c.kind:<6}] {c.citation:<28} {c.section_path.split(' > ', 1)[-1]:<30} {count_tokens(c.text):>3} tok  ids={c.element_ids}")
    t = next(c for c in chunks if c.kind == "table" and "Milestone" in c.text)
    print("   merged cross-page table chunk:\n     " + t.text.replace("\n", "\n     "))
    big = [c for c in LayoutChunker(45).chunk(doc) if c.kind == "table" and "Cost" in c.text]
    print(f"   with max 45 tokens the cost table splits into {len(big)} row groups, header repeated in each:")
    print("     " + big[0].text.replace("\n", "\n     "))

    naive = naive_chunks(doc)
    print(f"\nC) naive flatten + 60-word windows: {len(naive)} chunks; footer noise inside: "
          f"{sum('Confidential' in c for c in naive)} chunks; page numbers known: 0; one of them:")
    print("   ", repr(naive[2][:150]))

    print("\nD) retrieval over the 7 questions (hit = a top-k chunk contains the answer string / element)")
    M_l = embed([c.embed_text for c in chunks])
    M_n = embed(naive)
    qv = embed([q for q, _, _ in QUESTIONS])
    rows = []
    h1 = h3 = n1 = n3 = cited = 0
    for (q, eid, ans), v in zip(QUESTIONS, qv):
        top = [i for i, _ in c14.dense_search(v, M_l, 3)]
        ok = [eid in chunks[i].element_ids for i in top]
        topn = [i for i, _ in c14.dense_search(v, M_n, 3)]
        okn = [ans in naive[i] for i in topn]
        h1 += ok[0]; h3 += any(ok); n1 += okn[0]; n3 += any(okn)
        if ok[0]:
            cited += 1
        rows.append([q[:46], chunks[top[0]].citation if ok[0] else "-", "yes" if ok[0] else "no", "yes" if okn[0] else "no"])
    print(table(rows, ["question", "layout-aware citation", "layout top1", "naive top1"]))
    print(f"   layout-aware: hit@1 {h1}/7, hit@3 {h3}/7, answers with a page+bbox citation {cited}/7 | naive: hit@1 {n1}/7, hit@3 {n3}/7, citations 0/7")


if __name__ == "__main__":
    main()

# TRY IT: LayoutChunker(max_tokens=60): which chunks change? Do the two list items stay with their paragraph?
# TRY IT: delete the merge_continued_tables call; how many 'Planned milestones' chunks do you now get, and is the caption on page 3?
# BREAK IT: delete the "reading_order" key from one element of the JSON -> KeyError: 'reading_order' in the sort key.
#           Real parsers sometimes omit it; fall back to (page, bbox top, bbox left).
