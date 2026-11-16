# Step 09 — Layout-aware chunking from parsed documents

## The story so far
[Step 08](../08_propositions_and_agentic_chunking/WALKTHROUGH.md) put an LLM inside the chunker, but every chunker so far still started from a plain string. A PDF is not a string. Back to the [module README](../README.md).

**Where the input comes from.** The input of this step is the **parsed-document JSON**: a list of layout elements with type, text, page, bounding box and reading order. In the full pipeline this JSON is produced by [module 18, step 10 (parsed document hand-off)](../../18_ocr_and_document_parsing/10_parsed_document_handoff/WALKTHROUGH.md), which converts OCR and layout results into **exactly the schema of this step's sample file** [sample_parsed_document.json](sample_parsed_document.json). This step deliberately *does not depend on* module 18 to run: it uses the hand-written 3-page sample, so you can study chunking without an OCR run. The schema is the contract between the two modules; do not change it on one side only.

## The pain point
Flatten a PDF to text and you lose exactly what layout encodes:
- **page headers and footers** ("Confidential - page 2") repeat on every page and pollute chunks;
- **reading order**: parsers may emit a caption after its table, or columns interleaved, so array order is not reading order;
- **tables** become a stream of cell words with no column names ("99.93 99.96 99.94"), and a table that **continues on the next page** is two fragments, the second with no caption;
- **figures** have a caption but no text;
- **citations**: after flattening you can no longer say *which page and where on the page* an answer came from, so you cannot highlight it.

## The spark
Chunk the **elements**, not a string. Sort by `(page, reading_order)`, drop headers and footers (keep them as metadata), start a chunk at every section header, keep a table whole (or as row groups with the header repeated), merge a table that continues over a page break, bind each caption to its table or figure, and carry `page` and `bbox` through to the chunk so every answer has a citation that a PDF viewer can highlight.

## Goal
1. Read a parsed document and see that array order is not reading order (A).
2. Build layout-aware chunks with page + bbox citations; see the cross-page table merged and a big table split into row groups with the header repeated (B).
3. Build the "naive" version a text-only pipeline would, and see what it loses (C).
4. Score retrieval on 7 questions and count answers that come with a page + bbox citation (D).

## The theory (read this before the interview)
**The parsed-document schema** (what Docling, Unstructured, layout-model pipelines and the module-18 hand-off all reduce to):
```
{ "doc_id", "source_file", "pages": [{"page", "width", "height"}],
  "elements": [
    { "id", "type", "text", "page", "bbox": [x0, y0, x1, y1], "reading_order",
      "level"  (section_header only),
      "table":  {"header": [...], "rows": [[...], ...]}      (table only),
      "figure": {"alt": "..."}                                (figure only),
      "continued_from": "<id>"                                (optional hint) } ] }
type in: title, section_header, text, list_item, table, figure, caption, footnote, page_header, page_footer
bbox: PDF points, origin top-left; reading_order: integer within a page
```
**The rules and why each exists:**
```
1 sort by (page, reading_order)       arrays are not reading order
2 drop page_header / page_footer       noise; keep the footer text as metadata
3 merge continued tables               same header + next page  -> one table
4 section_header closes a chunk        chunks never straddle a heading; path = Title > Section > Subsection
5 caption binds to its object          caption ABOVE a table, BELOW a figure
6 table whole if it fits, else row groups with the header repeated
7 table embed text = linearised rows   "Region: Dublin; July: 99.95; ..." (embedders read words better than | pipes |)
8 footnotes stay alone                 they qualify a statement, they are not part of the paragraph
9 chunk keeps pages + union bbox per page + element ids          -> citation, highlight, traceability
```
**Citation chain.** `page` and `bbox` start on each element. A chunk made of several elements keeps `element_ids` and one **union box per page**. A merged table keeps the boxes of both pages. At answer time: top chunk, then `chunk.boxes`, then highlight in the PDF.

**Where the parsing quality comes from.** If the parser misses a table or mislabels a heading, no chunker fixes it: the garbage is chunked faithfully. Evaluate the parser separately (module 18).

**Status.** Layout-aware chunking on parsed elements is *current practice* for PDFs and slides (Docling's `HybridChunker`, Unstructured `chunk_by_title`, LlamaParse/Textract outputs). Newer *vision-language* parsers and page-image retrievers (ColPali-style) go around text chunking entirely for visually rich pages; that is *emerging*.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| layout-aware chunking | element-based chunking, structure-preserving chunking, document-AI chunking, Docling hierarchical / hybrid chunker, Unstructured `chunk_by_title` | near | Same idea; tools differ in merging rules |
| parsed document | document AST, DoclingDocument, Unstructured elements, layout JSON | near | A list or tree of typed elements with geometry |
| bbox | bounding box, `[x0, y0, x1, y1]`, polygon, coordinates | near | Rectangle on a page in PDF points (or pixels, check the origin) |
| reading order | logical order, flow, `reading_order` index | yes | The order a human reads the elements |
| cross-page table merge | continued table, table stitching, multi-page table | yes | Join fragments with the same header on adjacent pages |
| linearised table | table-to-text, row serialisation, "key: value" rows | near | Each row as `col: value; ...` text for embedding |
| caption binding | caption association, figure-caption linking | yes | Attach the caption to its table or figure |
| citation / provenance | grounding, source attribution, highlight | near | Page + box back to the source |

## Run it
```bash
cd 19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs
python main.py
# from the repo root:
HF_HUB_OFFLINE=1 .venv/bin/python 19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/main.py
```
About 10 s of CPU time. Needs `torch` + `transformers` and the cached MiniLM; no downloads.

## Code, step by step

### 1. Header docstring, imports, the noise types  (`main.py` lines 1–31)

```python
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
```

- **In simple words:** the contract of the file, the helpers, and the set of element types that are page furniture.
- **Exactly:** `HERE` is this step's folder so the sample JSON is found whatever the working directory. `common19` provides `count_tokens`, `embed`, `table`; `common as c14` is module 14's helper file, used here only for `dense_search`. `NOISE = {"page_header", "page_footer"}`.
- **Why it is here:** the docstring states the dependency honestly: the module-18 file has the same schema, but this step runs without it.

### 2. `LChunk`: a chunk that remembers its place on the page  (`main.py` lines 34–48)

```python
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
```

- **In simple words:** a chunk with kind, display text, the text to embed, a section path, pages, boxes and the ids of its source elements.
- **Exactly:** `text` is what is *shown* (a Markdown table, a paragraph), `embed_text` is what is *embedded* (breadcrumb + caption + linearised rows). `boxes` is a list of `{page, bbox}`, one union box per page. The `citation` property formats `p.2-3 @[x0, y0, x1, y1]` from the first box and the pages.
- **Why it is here:** the same separation of "searched" and "shown" as `RawChunk` in [step 01](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py), plus geometry. Step 11's production `Chunk` has `page` and `bbox` fields for the same reason.

### 3. `load` and `union_boxes`  (`main.py` lines 51–62)

```python
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
```

- **In simple words:** read the JSON; compute the rectangle that covers a group of elements on each page.
- **Exactly:** `json.loads(Path(path).read_text())` returns plain dicts and lists. `union_boxes` keeps, per page, the minimum of the x0/y0 values and the maximum of the x1/y1 values, and returns `[{"page": p, "bbox": [...]}]` sorted by page.
- **Why it is here:** a chunk of three paragraphs needs one highlight rectangle, not three; a chunk spanning two pages needs one per page.

### 4. `merge_continued_tables`  (`main.py` lines 65–78)

```python
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
```

- **In simple words:** if a table is directly followed by another table with the same header on the next page, join their rows into one table.
- **Exactly:** the test is `e.type == "table"`, the previous element is a table, `prev["table"]["header"] == e["table"]["header"]` and `e["page"] == prev["page"] + 1`. The merged element keeps `rows` from both, `pages` (all pages), `merged_ids` (both element ids) and `bboxes` (both source elements, so boxes can be unioned per page later). `dict(prev, table=dict(...), ...)` copies a dict with replaced keys.
- **Why it is here:** the header-equality + adjacent-page rule is the usual heuristic. The sample also carries a `continued_from` hint on `e21`; a parser that provides it makes the rule exact, but the heuristic does not need it. It runs *after* noise removal, so the page footer between the two tables is already gone.

### 5. `md_table` and `linearise`: two views of a table  (`main.py` lines 81–87)

```python
def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows])


def linearise(header, row):
    """Row -> sentence-like text with the column names repeated: embedders read this better than '| 99.97 | 99.99 |'."""
    return "; ".join(f"{h}: {v}" for h, v in zip(header, row))
```

- **In simple words:** `md_table` makes a Markdown table for display; `linearise` turns one row into `Column: value; Column: value`.
- **Exactly:** `"|" + "---|" * len(header)` builds the separator row. `zip(header, row)` pairs each cell with its column name.
- **Why it is here:** display and search want different forms. A row such as `| Dublin | 99.95 | 99.96 | 99.93 |` embeds poorly; `Region: Dublin; July: 99.95; August: 99.96; September: 99.93` embeds well, because the column names ride along with every value.

### 6. `LayoutChunker`, part 1: set-up, noise, merge, `flush`  (`main.py` lines 90–107)

```python
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
```

- **In simple words:** sort elements by reading order, remember the footers per page, drop headers and footers, merge continued tables, and define how a pending buffer becomes a text chunk.
- **Exactly:** `sorted(..., key=lambda e: (e["page"], e["reading_order"]))` is rule 1. `footer_pages` maps page to footer text for metadata. `buf` collects consecutive text elements. `flush()` (nested function) joins the buffered texts with spaces, builds the section path from `title` and `section`, and appends an `LChunk` whose `embed_text` is `path + "\n" + text`, `pages` the sorted set of source pages, `boxes` the union boxes, and `element_ids` the source ids; then it empties the buffer with `buf.clear()`.
- **Why it is here:** the setup is rules 1, 2, 3 in four lines; `flush` is where citation data is attached.

### 7. `LayoutChunker`, part 2: the element loop  (`main.py` lines 108–140)

```python

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
```

- **In simple words:** walk the elements once and apply the rule that matches each type.
- **Exactly:** `title` sets the document title; `section_header` flushes and updates `section[:level-1] + [text]` (a level-2 header keeps its level-1 parent); `footnote` flushes, then makes its own one-element chunk; `text`/`list_item` are added to the buffer unless the token total would exceed `max` (then flush first), so a list item stays with its paragraph. A `caption` followed by a table or figure flushes and builds the object chunk with the caption (caption **above**); a bare `table` or `figure` checks whether the *next* element is a caption (caption **below**, the figure case) and consumes it with `i += 1`; a stray caption goes to the buffer. At the end `flush()` and attach the footer text of each chunk's pages to `meta["footers"]`.
- **Why it is here:** this is the whole chunker. Notice what it never does: it never cuts in the middle of an element, so a paragraph, a list item, a table row or a caption is atomic.

### 8. `object_chunks`: tables and figures  (`main.py` lines 142–165)

```python
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
```

- **In simple words:** turn a table or figure (with its caption) into chunks: a figure becomes caption plus alt text; a table that fits stays whole; a table that does not fit becomes row groups with the header repeated.
- **Exactly:** `els` is `[caption, obj]` (caption optional); `boxes` unions them and any extra merged elements; `ids` lists all source ids including merged ones. For tables, `whole = md_table(header, rows)` is displayed if `tokens(cap + whole) <= max`; its `embed_text` is `path + caption + linearised rows`. Otherwise rows are added to `group` until the next row would exceed `max`, and each group becomes its own chunk with `md_table(header, group)`: the header repeats because it is part of `md_table`. The loop uses a sentinel `rows + [None]` to flush the last group.
- **Why it is here:** this is the "table with its header and its caption, always" guarantee, and the figure case shows how a text-less object becomes searchable (caption + alt text).

### 9. `naive_chunks`: what a text-only pipeline does  (`main.py` lines 168–176)

```python
def naive_chunks(doc, window=60):
    """What a text-only pipeline does: concatenate every element's text (cells flattened) and cut fixed windows."""
    words = []
    for e in doc["elements"]:                                                              # array order, noise included
        if e["type"] == "table":
            words += [c for row in [e["table"]["header"]] + e["table"]["rows"] for c in row]
        else:
            words += e["text"].split()
    return [" ".join(words[i:i + window]) for i in range(0, len(words), window)]
```

- **In simple words:** concatenate the words of every element in array order (cells flattened, footers included) and cut fixed 60-word windows.
- **Exactly:** table cells are appended as separate words; there is no reading-order sort, no noise removal, no geometry. The result is a list of plain strings.
- **Why it is here:** the control: it is what you get from `pdftotext` followed by a fixed-size splitter.

### 10. The seven test questions  (`main.py` lines 179–187)

```python
QUESTIONS = [
    ("What was the uptime of Dublin in September?", "e6", "99.93"),
    ("Which service's cost grew the most compared with Q2?", "e13", "+18%"),
    ("Who owns the multi-region failover test and when is it due?", "e21", "30 June"),
    ("What is the due date of the combined dashboard?", "e20", "5 April"),
    ("How much of compute capacity is reserved?", "e15", "60 percent"),
    ("What happened to queue depth during the quarter?", "e8", "20 jobs"),
    ("How often is uptime measured?", "e23", "30 seconds"),
]
```

- **In simple words:** seven questions, each with the element id and the answer string that must be found.
- **Exactly:** each tuple is `(question, element_id, answer_string)`: layout-aware retrieval is correct if a top chunk contains the element id; naive retrieval is correct if the answer string appears in a top chunk. The questions cover a table cell, a computed comparison, a cross-page table row, the first rows of that table, a text fact, a figure, and a footnote.
- **Why it is here:** a tiny, fully inspectable evaluation: with seven questions every miss can be read by hand. This is not a benchmark.

### 11. Section A and B: elements and layout-aware chunks  (`main.py` lines 190–206)

```python
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
```

- **In simple words:** show element counts per type, the parser's array order against the sorted order on page 1, then every chunk with its citation, and the merged and split tables.
- **Exactly:** `c.citation` prints page(s) and the first box; `c.section_path.split(" > ", 1)[-1]` drops the document title from the path for display. `next(c for c in chunks if c.kind == "table" and "Milestone" in c.text)` finds the cross-page table. A second `LayoutChunker(45)` shows what a tiny budget does to the cost table.
- **Why it is here:** every rule is visible in the output: the swapped `e6`/`e5`, the merged table with rows from two pages, and the repeated header.

### 12. Section C: the naive result  (`main.py` lines 208–211)

```python
    naive = naive_chunks(doc)
    print(f"\nC) naive flatten + 60-word windows: {len(naive)} chunks; footer noise inside: "
          f"{sum('Confidential' in c for c in naive)} chunks; page numbers known: 0; one of them:")
    print("   ", repr(naive[2][:150]))
```

- **In simple words:** run the naive chunker and count how many chunks contain footer text.
- **Exactly:** `sum('Confidential' in c for c in naive)` counts chunks containing the footer word; citations are zero by construction.
- **Why it is here:** concrete damage numbers, as in step 03.

### 13. Section D: retrieval and citation check  (`main.py` lines 213–229)

```python
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
```

- **In simple words:** embed layout-aware `embed_text` and naive chunks, and for each question see whether the top-1 and top-3 results contain the answer, and whether we can print a page + bbox citation.
- **Exactly:** `c14.dense_search(v, M, 3)` returns the top-3 `(index, score)` pairs by dot product. `ok` tests `eid in chunks[i].element_ids`; `okn` tests `ans in naive[i]`. `cited` counts answers whose top-1 chunk is correct and therefore comes with a citation. The table prints the citation for each correct layout-aware answer.
- **Why it is here:** the citation column is the point: even when a naive chunk contains the answer string, it cannot say where it came from.

### 14. Entry point and the TRY IT / BREAK IT lines  (`main.py` lines 232–238)

```python
if __name__ == "__main__":
    main()

# TRY IT: LayoutChunker(max_tokens=60): which chunks change? Do the two list items stay with their paragraph?
# TRY IT: delete the merge_continued_tables call; how many 'Planned milestones' chunks do you now get, and is the caption on page 3?
# BREAK IT: delete the "reading_order" key from one element of the JSON -> KeyError: 'reading_order' in the sort key.
#           Real parsers sometimes omit it; fall back to (page, bbox top, bbox left).
```

- **In simple words:** guard plus exercises.
- **Exactly:** deleting the `reading_order` key from one element raises `KeyError: 'reading_order'` inside the sort key.
- **Why it is here:** real parsers sometimes omit fields; production code must have a fallback order (page, bbox top, bbox left).

## Syntax explained
| Syntax | Meaning |
|---|---|
| `sorted(els, key=lambda e: (e["page"], e["reading_order"]))` | Sort by a tuple key: page first, then order inside the page |
| `{e["page"]: e["text"] for e in els if ...}` | Dict comprehension |
| `dict(prev, table=dict(prev["table"], rows=...))` | Copy a dict, replacing selected keys |
| `@property def citation(self)` | A method used like an attribute |
| `def flush(): ... buf.clear()` (nested) | Inner function that closes over `buf` and `chunks` |
| `for r in rows + [None]:` | Sentinel: one extra iteration to flush the last group |
| `i += 1 if below else 0` | Conditional expression as an increment |
| `ok = [eid in chunks[i].element_ids for i in top]` | List of booleans, then `ok[0]`, `any(ok)` |
| `f"p.{'-'.join(str(p) for p in sorted(set(self.pages)))}"` | Join the distinct page numbers with dashes |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `json.loads(text)` | JSON string to dicts/lists | Standard-library parser |
| `Path.read_text(encoding="utf-8")` | File content as a string | Opens, reads, closes |
| `c14.dense_search(qvec, matrix, k)` | Top-`k` rows by dot product | `argsort` of `matrix @ qvec` (module 14) |
| `embed(texts)` | Cached MiniLM vectors | See the [step 01 tour](../01_tokens_not_characters/WALKTHROUGH.md#the-shared-helpers-once-a-tour-of-common19py) |
| `Counter` | Count element types | `dict` with default 0 |
| `count_tokens` | Tokens of a string | Cached tokenizer |

## Execution trace (real run)
Real output of `python main.py` (progress bar removed; timings vary by machine).
```
A) parsed document: lumen_q3_report.pdf | elements by type: {'page_header': 3, 'title': 1, 'section_header': 5, 'text': 5, 'table': 4, 'caption': 4, 'figure': 1, 'page_footer': 3, 'list_item': 2, 'footnote': 1}
   array order on page 1 : ['p1h', 'e1', 'e2', 'e3', 'e4', 'e6', 'e5'] (e6/e5 swapped by the parser)
   reading_order sorted  : ['p1h', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6']

B) layout-aware chunks (max 140 tokens): 10 chunks, kinds {'text': 6, 'table': 3, 'figure': 1}
   [text  ] p.1 @[72, 156, 540, 214]     1. Summary                      52 tok  ids=['e3']
   [table ] p.1 @[72, 270, 540, 400]     2. Availability                 82 tok  ids=['e5', 'e6']
   [text  ] p.1 @[72, 412, 540, 440]     2. Availability                 19 tok  ids=['e7']
   [figure] p.1 @[72, 452, 540, 662]     2. Availability                 30 tok  ids=['e9', 'e8']
   [text  ] p.2 @[72, 96, 540, 144]      3. Costs                        31 tok  ids=['e11']
   [table ] p.2 @[72, 156, 540, 330]     3. Costs                        84 tok  ids=['e12', 'e13']
   [text  ] p.2 @[72, 374, 540, 466]     3. Costs > 3.1 Reserved capacity  40 tok  ids=['e15', 'e16', 'e17']
   [table ] p.2-3 @[72, 530, 540, 720]   4. Roadmap                      84 tok  ids=['e19', 'e20', 'e21']
   [text  ] p.3 @[72, 216, 540, 264]     4. Roadmap                      23 tok  ids=['e22']
   [text  ] p.3 @[72, 700, 420, 716]     4. Roadmap                      13 tok  ids=['e23']
   merged cross-page table chunk:
     Table 3: Planned milestones
     | Milestone | Owner | Due |
     |---|---|---|
     | Combined dashboard | Priya | 5 April |
     | Async quota job | Tomas | 11 April |
     | Multi-region failover test | Mira | 30 June |
     | Storage reservation | Finance | 1 September |
     | SAML self-service setup | Priya | 15 October |
   with max 45 tokens the cost table splits into 4 row groups, header repeated in each:
     Table 2: Cost per service in thousands of euros
     | Service | Q2 | Q3 | Change |
     |---|---|---|---|
     | Compute | 412 | 431 | +5% |

C) naive flatten + 60-word windows: 5 chunks; footer noise inside: 3 chunks; page numbers known: 0; one of them:
    'Q2. Most of the growth came from storage. The table below lists cost per service in thousands of euros. Table 2: Cost per service in thousands of euro'

D) retrieval over the 7 questions (hit = a top-k chunk contains the answer string / element)
question                                             layout-aware citation  layout top1  naive top1
----------------------------------------------  --------------------------  -----------  ----------
What was the uptime of Dublin in September?       p.1 @[72, 270, 540, 400]          yes          no
Which service's cost grew the most compared wi                           -           no         yes
Who owns the multi-region failover test and wh  p.2-3 @[72, 530, 540, 720]          yes          no
What is the due date of the combined dashboard  p.2-3 @[72, 530, 540, 720]          yes          no
How much of compute capacity is reserved?         p.2 @[72, 374, 540, 466]          yes         yes
What happened to queue depth during the quarte    p.1 @[72, 452, 540, 662]          yes          no
How often is uptime measured?                     p.3 @[72, 700, 420, 716]          yes         yes
   layout-aware: hit@1 6/7, hit@3 7/7, answers with a page+bbox citation 6/7 | naive: hit@1 3/7, hit@3 7/7, citations 0/7
```

How to read it:
- **A:** 29 elements. On page 1 the parser emitted the table `e6` before its caption `e5`; sorting by `reading_order` restores `e5, e6`.
- **B:** 10 chunks: 6 text, 3 table, 1 figure. Page headers and footers are gone. Each table chunk starts with its caption (`ids=['e5', 'e6']`), the figure chunk with its caption `e9` plus the alt text of `e8`, and the 3 list/text elements of "3.1 Reserved capacity" stay together (`e15, e16, e17`). The footnote is its own chunk. The milestone table is **one** chunk across pages 2 and 3 (`p.2-3`, ids `e19, e20, e21`), with the page-3 rows under the page-2 caption and one union box per page. With a 45-token budget the cost table splits into 4 row groups, each with the header row.
- **C:** the naive flatten-and-window version produces 5 chunks, **all 3 page footers** end up inside chunks, and none can say which page it came from.
- **D:** layout-aware hit@1 is 6/7 and hit@3 7/7, and 6 answers come with a page + box citation; naive has hit@1 3/7, hit@3 7/7 and 0 citations. The one layout-aware miss is "Which service's cost grew the most compared with Q2?": it needs *comparing* rows of the cost table, and the embedding of a whole table does not rank first for that question while the naive chunk happens to contain the sentence "Most of the growth came from storage". Seven questions is a demonstration, not a benchmark.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `LayoutChunker(60)` | 16 chunks instead of 10: the tables split into row groups (about three row groups each for the uptime, cost and milestone tables), each with the header and the same citation box | `max` applies to every chunk type; list items stay with their paragraph until the budget is hit |
| Remove the `merge_continued_tables` call | 11 chunks instead of 10: the milestone table is two chunks, `p.2` with its caption and `p.3` as a headerless-looking fragment with no caption (real run) | The merge is what restores the table; without it the page-3 rows lose their caption |
| Remove the `sorted(...)` on reading order | In this sample the output is **unchanged** (10 chunks): the swapped `e6`/`e5` still bind, because a table followed by a caption takes the "caption below" branch | A real check that array order only hurts when it separates things the rules look at together; with interleaved columns it would break. Keep the sort |
| Keep page footers (delete the `NOISE` filter) | "Confidential - page N" appears inside chunks | Footer text is not content |
| Replace `max_tokens=140` by 400 | Fewer, larger text chunks; tables stay whole | Budget decides splitting, structure decides where |

## Debugging practice
**BREAK IT:** delete the `"reading_order"` key from one element of the JSON.

```
KeyError: 'reading_order'
```
raised inside the sort key `lambda e: (e["page"], e["reading_order"])`. Real parsers omit reading order sometimes, or give it only per column. **Fix:** a fallback key `(e["page"], e.get("reading_order", e["bbox"][1]), e["bbox"][0])` (page, then box top, then box left), and a validation step *before* chunking that rejects or repairs elements with a clear error. Module 18's hand-off validates this schema with a strict model for exactly this reason.

## Common mistakes
- Trusting array order as reading order.
- Embedding table cells as a bag of words with no header.
- Cutting a table in the middle with no repeated header.
- Treating the page footer as content (it repeats on every page and matches many queries).
- Merging two tables just because they are adjacent: require the same header (or a parser hint).
- Losing `page` and `bbox` while merging; keep per-page union boxes and element ids.
- Putting a figure caption into a *text* paragraph: the figure then has no searchable text; bind it.
- Believing the chunker fixed bad parsing. Evaluate the parser separately.
- Changing the element schema on one side (module 18 or module 19) and not the other.

## Rebuild checklist
1. Load the JSON; sort by `(page, reading_order)`.
2. Drop `page_header` / `page_footer` (keep footers as metadata).
3. `merge_continued_tables`: same header, next page; keep `pages`, `merged_ids`, `bboxes`.
4. `union_boxes` per page.
5. The loop: title, section path by level, text buffer with a token cap, footnote alone, caption above tables and below figures.
6. `object_chunks`: figure = caption + alt; table whole or row groups with the header repeated; `embed_text` linearised.
7. `LChunk.citation` from pages and the first box.
8. The naive control and the 7-question comparison with a citation column.

## Say it aloud
"For parsed PDFs I chunk the layout elements, not a string: sort by page and reading order, drop page headers and footers, start a chunk at every section header, keep tables whole or split by rows with the header repeated, merge a table that continues on the next page, and bind captions to their tables and figures. Tables are embedded as linearised rows, with the column name in front of each value, and displayed as Markdown. Each chunk keeps its pages, a union bounding box per page and its element ids, so every answer can be cited and highlighted in the original PDF. The input is the parsed-document JSON that the OCR and parsing module hands over; if the parser got the structure wrong, the chunker cannot fix it."

## The new problem (cliffhanger)
We now have a dozen ways to chunk, each with a story and a cost. Which one should *you* use? Every table so far came from a different script with a different metric. The next step puts every strategy of steps 01 to 08 on **one** scoreboard with hit@k, MRR, context tokens, ingestion time and noise estimates: [step 10, chunking evaluation](../10_chunking_evaluation/WALKTHROUGH.md).
