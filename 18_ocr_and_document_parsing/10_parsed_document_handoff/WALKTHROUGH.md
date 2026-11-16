# Step 10 — The parsed-document handoff: OCR output as a contract

## The story so far
[Step 09](../09_modern_document_parsers_2026/WALKTHROUGH.md) mapped the tools you could put in the parsing slot. Whatever you choose (our template OCR, Docling, a VLM), its output must leave the module in a **shape the next stage can trust**. The next stage, in this course, is [module 19 step 09](../../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/main.py): a chunker that consumes a list of layout elements, not a string.

## The pain point
Handoffs between stages are where pipelines rot. The OCR side says `heading`, the chunker side expects `section_header`; boxes are in pixels on one side and PDF points on the other; `reading_order` becomes a string; a table arrives as flattened text so its header row can no longer be repeated. None of these crash the OCR stage; they surface three stages later as a wrong citation or an unanswerable question.

## The spark
Write the contract down as code: a strict pydantic model of exactly the JSON that the consumer reads. Convert at the border (pixels to points, `kind` to `type`, rows to `header` + `rows`), validate immediately, and prove the consumer accepts the file by running the consumer's own loader and chunker on it. If the same model also accepts the consumer's own sample, you did not invent a new format by accident.

## Goal
Convert step 07's `structured_pages.json` into `parsed_document.json` with exactly the schema module 19/09 reads, validate it with a strict pydantic model (which also accepts module 19's sample), show four broken variants being rejected with readable errors, and feed the file to module 19's `load` and `LayoutChunker` to get chunks with page and bounding-box citations. This closes the OCR to chunking handoff of the document-intelligence track.

## The theory (read this before the interview)
**The schema (taken from module 19's `sample_parsed_document.json`, unchanged).**
```json
{ "doc_id": "...", "source_file": "...", "note": "...",
  "pages":    [ {"page": 1, "width": 600.0, "height": 480.0} ],
  "elements": [ {"id": "p1e1", "type": "section_header", "level": 1, "text": "Refund policy",
                 "page": 1, "bbox": [32.2, 79.5, 153.8, 98.2], "reading_order": 1},
                {"id": "p1e4", "type": "table", "text": "", "page": 1, "bbox": [...], "reading_order": 4,
                 "table": {"header": ["Product", "Price", "Stock"], "rows": [["Masala Chai", "250", "120"]]}} ] }
```
Element `type` is one of `page_header, page_footer, title, section_header, text, list_item, caption, table, figure, footnote`: the DocLayNet-style label set most layout models use. `level` exists only on `section_header`; `table` only on `table`;
`figure.alt` only on `figure`; `continued_from` links a table to its first part on the previous page. `bbox = [x0, y0, x1, y1]` in **PDF points, origin top-left** (the sample's convention, so a viewer can highlight it).
`reading_order` is an integer per page (the consumer sorts by `(page, reading_order)`, not by array order); the sample gives footers 99 so they sort last.

**Units.** Step 07 works in pixels of an 800 x 640 image rendered at 96 dpi. PDF points are 1/72 inch, so a pixel is 0.75 pt and the page is 600 x 480 pt. A real PDF's boxes from a layout model run on a rasterised page must be scaled by `72 / dpi` (and from bottom-left origin to top-left if
they come from the PDF text layer: [step 08](../08_pdf_internals_and_routing/WALKTHROUGH.md)). A unit mix-up produces citations that highlight the wrong place by a constant factor: easy to miss in review, obvious on a plot.

**Mapping from the OCR side to the contract.**
| OCR output (step 07) | Contract | Rule used here |
|---|---|---|
| `title` (34 px) | `title` | only the first one in the document; later ones are top-level sections |
| `heading` (26 px) | `section_header`, `level` | level 1 unless a level-1 header already exists, then level 2 (heuristic) |
| `paragraph` | `text` | as is |
| `table` (`rows`) | `table` with `header` = first row, `rows` = the rest, `text = ""` | the consumer renders Markdown itself and repeats the header across row groups |
| `footer` | `page_footer`, `reading_order = 99` | consumer drops it from chunk text and keeps it as metadata |
All heuristics are visible in about 15 lines and are the part you replace when the parser changes (a layout model gives the labels directly).

**Why a strict model.** `extra="forbid"` turns an unknown key into an error, `Literal[...]` turns `heading` into an error that lists the allowed values, validators encode cross-field rules (level iff section_header; table payload iff table; rectangular rows; bbox inside the page; unique ids; unique `reading_order` per page). Pydantic v2 parses the JSON into typed objects and raises one `ValidationError` listing every violation with its location path (`elements.4.table`).
Validate **where data crosses a boundary** (parser to chunker, API to service), not deep inside the consumer. The same idea as a JSON Schema or a protobuf; pydantic is the Python-native way and is what FastAPI ([module 09](../../09_api_development/README.md)) uses for request bodies.

**Provenance is the point.** `page` and `bbox` on every element are what let a RAG answer say "p.1 at [30, 232, 556, 366]" and let a viewer highlight it ([module 14](../../14_rag_and_retrieval/README.md), [module 13 capstone](../../13_capstone_project/README.md)). Lose them at the handoff and the answer cannot be verified.

**What is intentionally NOT in the schema: confidence.** The shared contract has no confidence field, and extra keys would be rejected. So OCR confidence travels in a **sidecar** `parsed_document.ocr_meta.json` keyed by element id (confidence, pixel bbox, original kind). A production pipeline uses it to route low-confidence blocks to a second engine or a human; the footer misread of step 07 is exactly the case (confidence 0.79).

**Errors are carried, not hidden.** The footer text stays `Page 1 cif 2` in the JSON. Silent correction would make downstream metrics look better than the pipeline is; a visible low-confidence block can be fixed deliberately (a rule for "Page N of M" footers, a second engine, a human).

**Real-world equivalents.** Docling's `DoclingDocument` (typed items such as texts, tables and pictures with provenance: page number and bounding box, plus reading order from the body tree), Unstructured's elements (`Title`, `NarrativeText`, `Table`, ... with `page_number` and coordinates in metadata) and the JSON that layout parsers emit all carry the same idea. Writing the adapter from your parser to your chunker's contract is a normal 50-line job.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Parsed document | structured document, document AST, layout elements, DoclingDocument | near | A list of typed elements with page, box and order, not a string |
| Element | block, node, item, layout element | near | One title, paragraph, table, footer ... with provenance |
| Provenance | citation info, `prov`, source location, page + bbox | near | Where in the source an element came from |
| `bbox` | bounding box, box, rectangle, `[x0, y0, x1, y1]` | yes | Here PDF points, origin top-left; YOLO-style `[x, y, w, h]` and `[x0, y1, x1, y0]` bottom-left exist: always check |
| Schema / contract | data contract, interface, JSON Schema, pydantic model | near | The agreed shape of the data at a boundary |
| Sidecar file | metadata file next to the data | yes | Extra information (confidence) kept out of the strict contract |
| `reading_order` | read order, `order`, sequence index | yes | Integer sort key within a page |
| `section_header` | heading, `h1/h2`, SectionHeaderItem | near | DocLayNet-style label; the OCR side's `heading` |

## Run it
```bash
# from this folder (Hugging Face offline: the chunker only loads the cached MiniLM tokenizer)
HF_HUB_OFFLINE=1 python main.py
# from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 18_ocr_and_document_parsing/10_parsed_document_handoff/main.py
```
A few seconds on a CPU laptop when nothing else is running (most of it is importing module 19's dependencies; timing varies by machine). Needs pydantic and, for the last check, module 19's imports. It regenerates step 07's `structured_pages.json` if missing, and writes `parsed_document.json` and `parsed_document.ocr_meta.json` here. If module 19 cannot be imported, the script says so instead of pretending the check passed.

## Code, step by step

### 1. Imports and paths  (`main.py` lines 16–30)
```python
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
```
- **In simple words:** Standard library, pydantic, and the locations of step 07's output and module 19's step folder.
- **Exactly:** `typing.Literal` restricts a field to listed strings; `importlib.util` loads a module from a file path (module folders start with digits, so a normal `import` cannot name them). `PX_TO_PT = 0.75` is the pixel-to-point factor at 96 dpi.
- **Why it is here:** Everything the contract needs is a path and a constant; the schema itself lives in the next block.

### 2. The allowed element types  (`main.py` lines 33–34)
```python
# ---- 1. The contract: a strict pydantic model of the shared schema --------------------------------------------------------------
ElementType = Literal["page_header", "page_footer", "title", "section_header", "text", "list_item", "caption", "table", "figure", "footnote"]
```
- **In simple words:** A `Literal` of the ten labels module 19's chunker understands.
- **Exactly:** A `Literal[...]` type alias; pydantic checks membership at validation time and its error message lists the allowed values.
- **Why it is here:** Catches the `heading` versus `section_header` class of bug at the border.

### 3. Table and figure payloads  (`main.py` lines 37–52)
```python
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
```
- **In simple words:** `TablePayload` has a header and rows and checks every row has as many cells as the header; `FigurePayload` has alt text.
- **Exactly:** `ConfigDict(extra="forbid")` makes unknown keys an error. `@model_validator(mode="after")` runs after field parsing and receives the built object; raising `ValueError` becomes a pydantic validation error with the path.
- **Why it is here:** A ragged row would later break `md_table` or the header repetition when module 19 splits a long table into row groups.

### 4. Element: one layout element with cross-field rules  (`main.py` lines 55–86)
```python
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
```
- **In simple words:** Fields shared by all elements, optional payloads, a bbox validator and a rule that the payload must match the type.
- **Exactly:** `Optional[X] = None` means the key may be absent. `@field_validator("bbox")` checks four numbers with positive extent. The model validator enforces: `level` iff `section_header`; `table` iff `table`; `figure` iff `figure`; non-empty text for text-like types.
- **Why it is here:** Encodes the unwritten rules of the consumer in one place, so a violation is reported at the handoff with the element's path.

### 5. PageInfo  (`main.py` lines 88–93)
```python
class PageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page: int
    width: float
    height: float
```
- **In simple words:** Page number and size in points.
- **Exactly:** Plain strict model.
- **Why it is here:** The size is needed to check that boxes lie inside the page and to scale boxes in a viewer.

### 6. ParsedDocument: the whole file and its cross-checks  (`main.py` lines 95–119)
```python
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
```
- **In simple words:** Top-level keys plus rules over several elements: unique ids, every page listed, boxes inside their page, unique reading order per page.
- **Exactly:** `collections.Counter` finds duplicate ids. The bbox check allows 1 pt of slack for rounding. Errors in nested models are reported as `elements.4.table` style paths.
- **Why it is here:** Per-element validation cannot see duplicates or out-of-page boxes; the document-level validator can.

### 7. to_parsed_document: the adapter  (`main.py` lines 123–156)
```python
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
```
- **In simple words:** Walk step 07's pages and elements and emit contract elements: scale boxes, rename kinds, split tables into header and rows, give footers `reading_order` 99, and record OCR metadata in a separate dict.
- **Exactly:** `round(v * PX_TO_PT, 1)` converts pixel boxes to points. Ids are `p{page}e{order}`. `reading_order` counts body elements per page from 0; the footer does not consume a number. `meta` maps element id to `{confidence, bbox_px, ocr_kind}`. Returns `(doc dict, meta dict)`.
- **Why it is here:** This block is the only part that knows both vocabularies; replace it when you replace the parser.

### 8. Run the conversion and write both files  (`main.py` lines 159–168)
```python
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
```
- **In simple words:** Regenerate step 07's output if missing, convert, write `parsed_document.json` and the sidecar, print every element.
- **Exactly:** `subprocess.run([sys.executable, ...])` runs step 07 in a child Python with the same interpreter. `json.dumps(..., ensure_ascii=False)` keeps non-ASCII text readable.
- **Why it is here:** Making the step self-sufficient means it works from a clean checkout.

### 9. Validate ours, validate module 19's sample, compare key sets  (`main.py` lines 171–180)
```python
reloaded = json.loads((HERE / "parsed_document.json").read_text())
ParsedDocument.model_validate(reloaded)
sample19 = json.loads((M19_STEP / "sample_parsed_document.json").read_text())
ParsedDocument.model_validate(sample19)
print("\npydantic: our parsed_document.json VALID | module-19 sample_parsed_document.json VALID (same model, so same contract)")

key_sets = lambda d: {t: set().union(*[set(e) for e in d["elements"] if e["type"] == t]) for t in {e["type"] for e in d["elements"]}}
ours, theirs = key_sets(reloaded), key_sets(sample19)
print("keys we emit per type are a subset of keys the sample uses:", all(ours[t] <= theirs[t] for t in ours if t in theirs),
      "| top-level keys equal:", set(reloaded) == set(sample19))
```
- **In simple words:** The same model must accept both files; the keys we emit per type must be a subset of the keys the sample uses.
- **Exactly:** `model_validate(dict)` parses and validates; it returns a model and raises `ValidationError` on failure. `key_sets` unions the keys of all elements of each type.
- **Why it is here:** If a model written from our output rejected the consumer's sample (or the reverse), the contract would be ours alone, not shared.

### 10. Four deliberately broken copies  (`main.py` lines 182–196)
```python
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
```
- **In simple words:** Mutate a deep copy of the file in four ways and show each is rejected with a readable location.
- **Exactly:** `json.loads(json.dumps(x))` is a deep copy. `err.errors()` is a list of dicts with `loc` (path tuple), `msg` and `type`.
- **Why it is here:** A validator you never saw reject anything is a validator you do not trust.

### 11. Run module 19's own loader and chunker on our file  (`main.py` lines 199–213)
```python
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
```
- **In simple words:** Import `19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/main.py` by path, call its `load` and `LayoutChunker(140).chunk`, print chunks with their citations.
- **Exactly:** `spec_from_file_location` + `module_from_spec` + `exec_module` run the file's top level only (its `main()` is behind `if __name__ == "__main__"`). The module is put in `sys.modules` first because its dataclass needs it. `load(path)` is `json.loads` of the file. Each chunk has `kind`, `text`, `embed_text`, `section_path`, `pages`, `boxes`, `citation`.
- **Why it is here:** The only convincing test of a contract is the consumer. It is wrapped in `try` so a missing dependency prints an honest message.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `Literal["a", "b"]` | type whose only allowed values are the listed ones |
| `class M(BaseModel): x: int` | pydantic model: annotations are the schema; instances are validated objects |
| `Optional[T] = None` | field that may be missing or null |
| `@field_validator("f")` / `@model_validator(mode="after")` | per-field check / check on the whole built model |
| `ConfigDict(extra="forbid")` | reject unknown keys |
| `dict(base, type="title", text=...)` | copy `base` and add or override keys |
| `lambda d: ...` stored in a dict | named mutation functions used for the negative tests |
| `try ... except Exception as e` | guard that reports a failed optional check instead of crashing |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `BaseModel.model_validate(data)` | dict to validated typed object; raises `ValidationError` | pydantic-core (Rust) walks the schema, coerces lax types (`"3"` to `3`) and collects all errors |
| `ValidationError.errors()` | list of `{loc, msg, type, ...}` | structured form of the message |
| `importlib.util.spec_from_file_location` | build an import spec from a path | lets you import a module whose folder name is not a valid identifier |
| `json.dumps(obj, indent=1)` / `json.loads` | serialise / parse JSON | needs plain Python types |
| `m19.load`, `m19.LayoutChunker` | module 19's loader and chunker | sort by `(page, reading_order)`, drop headers/footers, one chunk per section, tables whole or in row groups |

## Execution trace (real run)
```
converted 12 OCR elements -> parsed_document.json (2 pages, types {'title': 1, 'section_header': 3, 'text': 4, 'table': 2, 'page_footer': 2})
   p1e0   p1 ro=0  title           [31.5, 23.2, 370.5, 48.0]  'Chai Kitchen Quarterly Report'
   p1e1   p1 ro=1  section_header  [32.2, 79.5, 153.8, 98.2]  'Refund policy'
   p1e2   p1 ro=2  text            [30.8, 113.2, 297.0, 195.8]  'Refunds are issued within nine days of deliver'
   p1e3   p1 ro=3  text            [315.8, 113.2, 562.5, 195.8]  'Shipping inside India takes three to five work'
   p1e4   p1 ro=4  table           [30.0, 232.5, 556.5, 366.0]  'Product / Price / Stock  (+3 rows)'
   p1e5   p1 ro=99 page_footer     [271.5, 441.0, 331.5, 452.2]  'Page 1 cif 2'
   p2e0   p2 ro=0  section_header  [32.2, 23.2, 333.8, 48.0]  'Loyalty and Support Notes'
   p2e1   p2 ro=1  section_header  [32.2, 80.2, 154.5, 98.2]  'Loyalty points'
   p2e2   p2 ro=2  text            [30.8, 113.2, 279.8, 192.8]  'Members earn one point for every ten rupees sp'
   p2e3   p2 ro=3  text            [315.8, 113.2, 546.8, 195.8]  'Support is open from nine to six on weekdays. '
   p2e4   p2 ro=4  table           [30.0, 232.5, 556.5, 366.0]  'Plan / Points / Fee  (+3 rows)'
   p2e5   p2 ro=99 page_footer     [271.5, 441.0, 331.5, 452.2]  'Page 2 cif 2'

pydantic: our parsed_document.json VALID | module-19 sample_parsed_document.json VALID (same model, so same contract)
keys we emit per type are a subset of keys the sample uses: True | top-level keys equal: True
   rejected: bbox with 3 numbers            -> elements.0.bbox: Value error, bbox must be [x0, y0, x1, y1] with x0 < x1 and y0 < y1
   rejected: unknown element type           -> elements.1.type: Input should be 'page_header', 'page_footer', 'title', 'section_header
   rejected: section_header without level   -> elements.1: Value error, `level` is required for section_header and forbidden for 
   rejected: ragged table row               -> elements.4.table: Value error, rows [0] do not have 3 cells like the header

module 19 loader + LayoutChunker read our file: 4 chunks, kinds {'text': 2, 'table': 2}
   [text ] p.1 @[31, 113, 562, 196] Refund policy                'Refunds are issued within nine days of delivery.'
   [table] p.1 @[30, 232, 556, 366] Refund policy                '\n| Product | Price | Stock |\n|---|---|---|\n| Mas'
   [text ] p.2 @[31, 113, 547, 196] Loyalty and Support Notes >  'Members earn one point for every ten rupees spen'
   [table] p.2 @[30, 232, 556, 366] Loyalty and Support Notes >  '\n| Plan | Points | Fee |\n|---|---|---|\n| Basic |'
   first table chunk, as an embedder will see it:
     Chai Kitchen Quarterly Report > Refund policy
     
     Product: Masala Chai; Price: 250; Stock: 120
     Product: Green Tea; Price: 320; Stock: 75
     Product: Tulsi Blend; Price: 290; Stock: 0
   footers kept as metadata, not as chunk text: [['Page 1 cif 2'], ['Page 1 cif 2']]
```
- 12 OCR elements become 12 contract elements: 1 title, 3 section headers (page 2's big line is a top-level section, its 26 px heading is level 2), 4 text blocks, 2 tables, 2 footers (reading order 99).
- Boxes are now in points: the page-1 title `[31.5, 23.2, 370.5, 48.0]` is step 07's `[42, 31, 494, 64]` times 0.75.
- The same pydantic model accepts our file and module 19's hand-written sample, and every broken variant is rejected with a path (`elements.4.table`) and a human-readable message.
- Module 19's loader and chunker run unchanged on our file and produce 4 chunks (2 text, 2 table) with `p.1 @[x0, y0, x1, y1]` citations; the table chunk is linearised as `Product: Masala Chai; Price: 250; Stock: 120`.
- The OCR error is still there: the footer metadata says `Page 1 cif 2`. The contract carries the error; the sidecar carries the confidence that predicted it.
- Module 19's chunker prints no caption for our tables (we have none), so a table chunk begins with a blank line: real parsers give captions, and the adapter would map them to `caption` elements.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| Set `PX_TO_PT = 1.0` | Boxes stay in pixels and pages become 800 x 640: still valid, but citations are no longer in PDF points | The contract cannot know your unit; document it in `note` and check it against the viewer |
| Hand-edit one bbox `x1` to 900 and validate | `elements.N: ... lies outside the (600.0, 480.0) page` | Document-level cross-check |
| Add `"author": "x"` to an element | `Extra inputs are not permitted` | `extra="forbid"` |
| Give footers `reading_order = n_body` | Module 19 still drops them (it filters by type), but they now sit between body elements in the sorted order | Reading-order numbers only matter among body types; 99 is a convention |
| Emit `heading` for headings | `Input should be 'page_header', ... 'section_header' ...` | `Literal` type |
| Remove `level` from a section header | `level is required for section_header` | Cross-field validator |

## Debugging practice
**BREAK IT:** in `to_parsed_document`, write `reading_order=str(n_body)` (a string) and skip the pydantic check, then call module 19's chunker.

**Symptom:** `TypeError: '<' not supported between instances of 'str' and 'int'` inside `sorted(doc["elements"], key=lambda e: (e["page"], e["reading_order"]))`, because the footer's `reading_order` is the int 99 and the others are strings (or, if all were strings, `"10"` would sort before `"2"`: a silent mis-ordering). With the validator in place, pydantic's lax mode coerces `"3"` to `3` but rejects `"first"` at the border: `elements.0.reading_order: Input should be a valid integer, unable to parse string as an integer`.

**Fix:** type the field as `int`, validate every file you hand over (`ParsedDocument.model_validate`), and consider `strict=True` on the model if you do not want coercion.

## Common mistakes
- Passing the OCR parser's native vocabulary (`heading`, `kind`, pixel boxes) to a consumer that expects another, and discovering it as bad citations.
- Putting confidence or debug fields into the shared file: either extend the contract deliberately (and version it) or use a sidecar.
- Dropping `page` and `bbox` because "the chunker only needs text": then no answer can be cited or highlighted.
- Flattening tables to text at the handoff. Keep `header` and `rows`: row-group chunking repeats the header (module 19, step 09).
- "Fixing" OCR errors silently in the adapter. Fix them deliberately and log what you changed.
- Mixing coordinate origins: PDF text-layer boxes are bottom-left; image-space boxes are top-left; the contract says top-left points.

## Rebuild checklist
1. Copy the key set of module 19's sample JSON (read its file, do not guess) and write the `Literal` of element types.
2. Write `TablePayload`, `FigurePayload`, `Element` (with bbox and payload validators), `PageInfo`, `ParsedDocument` (ids, pages, boxes, reading order).
3. Write `to_parsed_document`: pixels to points, kind mapping, table split, footer order 99, sidecar dict.
4. Write the files; validate our file and module 19's sample with the same model.
5. Add four negative tests (bbox, type, level, ragged row).
6. Import module 19/09 by path and run `load` and `LayoutChunker(140).chunk` on your file; print citations.

## Say it aloud
- "The handoff between OCR and chunking is a contract: typed elements with page, bbox in PDF points, reading order, and structured tables, not a string."
- "I validate at the border with a strict pydantic model that also accepts the consumer's own sample, so I know I match their schema rather than inventing one."
- "I carry OCR confidence in a sidecar keyed by element id, because the shared schema forbids extra fields, and I use it to route low-confidence blocks to a second engine."
- "Provenance (page and box) is what lets the RAG layer cite and highlight; losing it at the handoff makes answers unverifiable."
- "Tables keep their header and rows so the chunker can split long tables and repeat the header.

## The new problem (cliffhanger)
We have a validated `parsed_document.json`. What a chunker does with it (keep sections together, keep tables whole, merge a table across pages, bind captions) is [module 19, step 09](../../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/main.py); how retrieval over those chunks works is [module 14](../../14_rag_and_retrieval/README.md); and how a document-QA system is assembled end to end is the [capstone](../../13_capstone_project/README.md). The module README ([18 README](../README.md)) ties the ten steps together.
