"""
GOAL:  The OCR -> chunking handoff. Convert module 18's structured pages (step 07: pixel boxes, kinds like "heading") into
       parsed_document.json with EXACTLY the schema that 19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs reads
       (PDF-point boxes, element types, reading_order, level / table / figure payloads). Validate it with a strict pydantic model
       (which must also accept module 19's own sample), then prove 19/09's loader and LayoutChunker accept our file.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 18_ocr_and_document_parsing/10_parsed_document_handoff/main.py
EXPECTED OUTPUT (shortened):
    converted 12 OCR elements -> parsed_document.json (2 pages, types {...})
    pydantic: our file VALID | module-19 sample VALID | 4 deliberately broken copies rejected with readable errors
    module 19 loader + LayoutChunker read our file: N chunks, each with page + bbox citation ...
NEEDS: pydantic; for the last check also module 19's imports (cached MiniLM tokenizer, offline, no download).  ~10 s on CPU.
Also writes parsed_document.ocr_meta.json (OCR confidence per element: a SIDECAR, because the shared schema has no confidence field).
"""
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

HERE = Path(__file__).parent
M18 = HERE.parent
M19_STEP = M18.parent / "19_chunking_deep_dive" / "09_layout_aware_chunking_from_parsed_docs"
STRUCT = M18 / "07_reading_order_and_tables" / "structured_pages.json"
PX_TO_PT = 0.75                                           # step 07 pages are 800 x 640 px at 96 dpi = 600 x 480 PDF points (1 pt = 1/72 in)


# ---- 1. The contract: a strict pydantic model of the shared schema --------------------------------------------------------------
ElementType = Literal["page_header", "page_footer", "title", "section_header", "text", "list_item", "caption", "table", "figure", "footnote"]


class TablePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    header: list[str]
    rows: list[list[str]]

    @model_validator(mode="after")
    def rectangular(self):
        bad = [i for i, r in enumerate(self.rows) if len(r) != len(self.header)]
        if bad:
            raise ValueError(f"rows {bad} do not have {len(self.header)} cells like the header")
        return self


class FigurePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alt: str


class Element(BaseModel):
    model_config = ConfigDict(extra="forbid")             # an unknown key is a contract violation, not a feature
    id: str
    type: ElementType
    text: str
    page: int
    bbox: list[float]                                      # [x0, y0, x1, y1] in PDF points, origin TOP-left
    reading_order: int
    level: Optional[int] = None                            # section_header only
    table: Optional[TablePayload] = None                   # table only
    figure: Optional[FigurePayload] = None                 # figure only
    continued_from: Optional[str] = None                   # table continued from an element on the previous page

    @field_validator("bbox")
    @classmethod
    def bbox_ok(cls, v):
        if len(v) != 4 or v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError("bbox must be [x0, y0, x1, y1] with x0 < x1 and y0 < y1")
        return v

    @model_validator(mode="after")
    def payload_matches_type(self):
        if (self.type == "section_header") != (self.level is not None):
            raise ValueError("`level` is required for section_header and forbidden for other types")
        if (self.type == "table") != (self.table is not None):
            raise ValueError("`table` payload is required for (and only for) type=table")
        if (self.type == "figure") != (self.figure is not None):
            raise ValueError("`figure` payload is required for (and only for) type=figure")
        if self.type not in ("table", "figure") and not self.text.strip():
            raise ValueError("text must not be empty for a text-like element")
        return self


class PageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int
    width: float
    height: float


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    source_file: str
    note: str
    pages: list[PageInfo]
    elements: list[Element]

    @model_validator(mode="after")
    def cross_checks(self):
        ids = [e.id for e in self.elements]
        dup = [i for i, n in Counter(ids).items() if n > 1]
        if dup:
            raise ValueError(f"duplicate element ids {dup}")
        size = {p.page: (p.width, p.height) for p in self.pages}
        for e in self.elements:
            if e.page not in size:
                raise ValueError(f"{e.id}: page {e.page} is not listed in `pages`")
            if e.bbox[2] > size[e.page][0] + 1 or e.bbox[3] > size[e.page][1] + 1:
                raise ValueError(f"{e.id}: bbox {e.bbox} lies outside the {size[e.page]} page")
        for p in size:
            ro = [e.reading_order for e in self.elements if e.page == p]
            if len(ro) != len(set(ro)):
                raise ValueError(f"page {p}: reading_order values must be unique")
        return self


# ---- 2. Convert step 07's output into that contract ---------------------------------------------------------------------------------
def to_parsed_document(struct):
    elements, meta, seen_title, seen_h1 = [], {}, False, False
    for page in struct["pages"]:
        p, n_body = page["page"], 0
        for e in page["elements"]:
            x0, y0, x1, y1 = (round(v * PX_TO_PT, 1) for v in e["bbox"])             # pixels -> PDF points
            eid = f"p{p}e{e['order']}"
            kind, base = e["kind"], dict(id=eid, page=p, bbox=[x0, y0, x1, y1])
            if kind == "title" and not seen_title:                                   # heuristic: only the first big line is THE title
                el = dict(base, type="title", text=e["text"], reading_order=n_body)
                seen_title = True
            elif kind in ("title", "heading"):
                level = 2 if (kind == "heading" and seen_h1) else 1                  # later big lines = top sections; headings under one = level 2
                seen_h1 = seen_h1 or level == 1
                el = dict(base, type="section_header", level=level, text=e["text"], reading_order=n_body)
            elif kind == "paragraph":
                el = dict(base, type="text", text=e["text"], reading_order=n_body)
            elif kind == "table":
                header, rows = e["rows"][0], e["rows"][1:]
                el = dict(base, type="table", text="", table=dict(header=header, rows=rows), reading_order=n_body)
            elif kind == "footer":
                el = dict(base, type="page_footer", text=e["text"], reading_order=99)   # the sample uses 99 so footers sort last
            else:
                raise ValueError(f"unmapped kind {kind!r}")
            if kind != "footer":
                n_body += 1
            elements.append(el)
            meta[eid] = dict(confidence=e["confidence"], bbox_px=e["bbox"], ocr_kind=kind)
    doc = dict(doc_id="chai_kitchen_report", source_file="synthetic_report (sample_page1.png, sample_page2.png)",
               note="Synthetic. Produced by module 18 (steps 03-07): template OCR + XY-cut reading order + ruling-line tables. "
                    "bbox = [x0, y0, x1, y1] in PDF points (pixels x 0.75), origin top-left. Text is raw OCR output, errors included.",
               pages=[dict(page=pg["page"], width=pg["width"] * PX_TO_PT, height=pg["height"] * PX_TO_PT) for pg in struct["pages"]],
               elements=elements)
    return doc, meta


if not STRUCT.exists():                                   # make the step self-sufficient: regenerate step 07's output
    subprocess.run([sys.executable, str(M18 / "07_reading_order_and_tables" / "main.py")], check=True, capture_output=True)
doc, meta = to_parsed_document(json.loads(STRUCT.read_text()))
(HERE / "parsed_document.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
(HERE / "parsed_document.ocr_meta.json").write_text(json.dumps(meta, indent=1))
print(f"converted {len(doc['elements'])} OCR elements -> parsed_document.json ({len(doc['pages'])} pages, "
      f"types {dict(Counter(e['type'] for e in doc['elements']))})")
for e in doc["elements"]:
    shown = e["text"] or " / ".join(e["table"]["header"]) + f"  (+{len(e['table']['rows'])} rows)"
    print(f"   {e['id']:<6} p{e['page']} ro={e['reading_order']:<2} {e['type']:<15} {e['bbox']}  {shown[:46]!r}")

# ---- 3. Validate: ours, module 19's own sample, and four broken copies --------------------------------------------------------------------
reloaded = json.loads((HERE / "parsed_document.json").read_text())
ParsedDocument.model_validate(reloaded)
sample19 = json.loads((M19_STEP / "sample_parsed_document.json").read_text())
ParsedDocument.model_validate(sample19)
print("\npydantic: our parsed_document.json VALID | module-19 sample_parsed_document.json VALID (same model, so same contract)")

key_sets = lambda d: {t: set().union(*[set(e) for e in d["elements"] if e["type"] == t]) for t in {e["type"] for e in d["elements"]}}
ours, theirs = key_sets(reloaded), key_sets(sample19)
print("keys we emit per type are a subset of keys the sample uses:", all(ours[t] <= theirs[t] for t in ours if t in theirs),
      "| top-level keys equal:", set(reloaded) == set(sample19))

broken = {
    "bbox with 3 numbers": lambda d: d["elements"][0].update(bbox=[1, 2, 3]),
    "unknown element type": lambda d: d["elements"][1].update(type="heading"),
    "section_header without level": lambda d: next(e for e in d["elements"] if e["type"] == "section_header").pop("level"),
    "ragged table row": lambda d: next(e for e in d["elements"] if e["type"] == "table")["table"]["rows"][0].pop(),
}
for name, mutate in broken.items():
    bad = json.loads(json.dumps(reloaded))
    mutate(bad)
    try:
        ParsedDocument.model_validate(bad)
        print(f"   NOT rejected (bug): {name}")
    except ValidationError as err:
        first = err.errors()[0]
        print(f"   rejected: {name:<30} -> {'.'.join(str(x) for x in first['loc'])}: {first['msg'][:70]}")

# ---- 4. Can module 19/09 really consume it? Import its main.py by path and call ITS loader and chunker ---------------------------------------
try:
    spec = importlib.util.spec_from_file_location("layout_chunking_19_09", M19_STEP / "main.py")
    m19 = importlib.util.module_from_spec(spec)
    sys.modules["layout_chunking_19_09"] = m19             # (dataclasses need the module registered while it executes)
    spec.loader.exec_module(m19)                           # its main() is behind `if __name__ == "__main__"`, so only definitions run
    loaded = m19.load(HERE / "parsed_document.json")       # exactly the call 19/09 makes for its own sample
    chunks = m19.LayoutChunker(140).chunk(loaded)
    print(f"\nmodule 19 loader + LayoutChunker read our file: {len(chunks)} chunks, kinds {dict(Counter(c.kind for c in chunks))}")
    for c in chunks:
        print(f"   [{c.kind:<5}] {c.citation:<22} {c.section_path.split(' > ', 1)[-1][:28]:<28} {c.text[:48]!r}")
    table_chunk = next(c for c in chunks if c.kind == "table")
    print("   first table chunk, as an embedder will see it:\n     " + table_chunk.embed_text.replace("\n", "\n     ")[:260])
    print("   footers kept as metadata, not as chunk text:", [c.meta["footers"] for c in chunks][:2])
except Exception as e:                                     # e.g. module 19's heavy imports unavailable: say so, do not fake a pass
    print(f"\n[could not run module 19's chunker here: {type(e).__name__}: {e}]  The pydantic check above still passed.")

# TRY IT:   hand-edit parsed_document.json (move a bbox x1 to 900) and re-validate: the page cross-check rejects it; add an `author` key to an
#           element (extra="forbid" rejects it); in `to_parsed_document` map footers to reading_order=n_body and see module 19 treat them as body.
# BREAK IT: emit "heading" instead of "section_header" (the OCR parser says heading, the contract says section_header: a classic handoff bug)
#           and watch pydantic name the bad value; or write reading_order as "first": without validation, module 19's sorted() would raise TypeError.
