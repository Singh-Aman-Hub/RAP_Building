# Step 07 — Reading order and tables

## The story so far
[Step 06](../06_ocr_evaluation/WALKTHROUGH.md) gave us the ruler: CER, WER, and the warning that a page with every character right can still be useless if the columns are read in the wrong order or a table loses its grid. [Step 03](../03_layout_segmentation/WALKTHROUGH.md) cut a page into lines with projection profiles, and [step 04](../04_character_recognition/WALKTHROUGH.md) plus the helper `dococr.py` turned a line into text. Now we assemble the whole page.

## The pain point
A real page is not a list of lines. It has a title, a heading, **two columns**, a ruled table and a footer. Sort every text line by its y coordinate (the obvious thing) and the left and right columns interleave, so "Refunds are issued within nine days" is followed by "Shipping inside India takes" instead of by its own next line. For a RAG system that is worse than a typo: a chunk made of two half-sentences from different topics. And a table flattened into text loses which number belongs to which column.

## The spark
Use the white space. **Recursive XY-cut**: find the widest empty horizontal band and cut there (top part is read first); if there is none, find the widest empty vertical band and cut there (left part first); recurse until nothing can be cut. The leaves are blocks and the recursion order IS the reading order. Tables are the opposite case: instead of cutting at white space, look for the **ink that is not text**, long horizontal and vertical ruling lines, and read each grid cell separately.

## Goal
Implement ruling-line detection, recursive XY-cut, a grid-based table reader with Markdown output and a typed block list (title, heading, paragraph, table, footer). Compare it with the naive "sort lines by y" order using WER, save `structured_pages.json` for [step 10](../10_parsed_document_handoff/WALKTHROUGH.md), and be honest about what still goes wrong (a real OCR misread, a threshold that fails on one page).

## The theory (read this before the interview)
**Reading order.** The sequence in which a human reads the blocks of a page. It is not stored in a scanned image at all; a layout parser must infer it.
Born-digital PDFs store text in the order the producer drew it, which is often (not always) the reading order; two-column Word documents usually are, text boxes in
slide-like layouts usually are not.

**Naive y-sort.** Sort all lines by (y, x). Correct only for single-column pages. With two columns, lines that sit at the same height from different
columns land next to each other.

**Recursive XY-cut (Nagy and Seth, 1984).** Project the ink onto an axis and look for empty bands.
```
xy_cut(region):
    trim region to its ink
    if there is an empty HORIZONTAL band >= min_gap_y:  return xy_cut(top) + xy_cut(bottom)
    elif there is an empty VERTICAL band   >= min_gap_x: return xy_cut(left) + xy_cut(right)
    else: return [region]            # a leaf = one block (a paragraph, a title, a table)
```
Why horizontal first: in a Western page headings and full-width elements separate vertical sections; columns only exist **inside** a section. The recursion
produces: title, heading, [left column, right column], table, footer. Complexity: each level scans the region once, so `O(depth x pixels)`.
Weakness: it needs clean white gutters. A figure that spans two columns, a pull quote, a skewed scan (step 02) or noise inside the gutter can make
every band "non-empty" and the cut fails. Modern parsers replace it with a learned layout detector (boxes + classes) plus a learned reading-order model
(for example LayoutReader, or the order a VLM writes), but XY-cut remains the baseline you can explain on a whiteboard.

**The min-gap thresholds are the model.** `min_gap_y=14` px separates paragraphs, `min_gap_x=20` px separates columns. Word gaps inside a line are about 5 px
at this size, so any value between a word gap and the real gutter works. On page 1 the gutter is only **25 px** (left column ends at x=396, right starts at x=421):
the first draft of this step used 30 and silently merged the two columns on that page. Lesson: thresholds must be calibrated on the narrowest real gutter, and a
parser should log which cut it made so a human can see it.

**Table structure from ruling lines.** A table with visible borders has long horizontal and vertical lines. A row of pixels whose ink covers more than ~80% of
the table width is a horizontal rule; the same on columns gives vertical rules. The N horizontal rules define N-1 rows, the vertical ones define the columns, and
each rectangle is a **cell** that can be OCR'd on its own (inset a few pixels so the border is not read as "|" or "l"). Tables **without** borders (common in
reports) need a different method: cluster text boxes into columns by x alignment. Learned models (TableFormer in Docling, SLANet in PaddleOCR) predict the grid
as a sequence of HTML-like tags and handle merged cells (rowspan/colspan), which a ruling-line scan cannot.

**Typed blocks.** A block is more useful when it has a type. Here the type comes from the **font size** of its first line (34 title, 26 heading, 21 paragraph, 16 footer):
a heuristic that works only on this synthetic page family. Real systems classify with a layout model (title, section-header, text, list-item, table, figure,
caption, footnote, page-header, page-footer: the label set DocLayNet made popular).

**What the OCR still gets wrong.** The template OCR reads the footer "Page 1 of 2" as "Page 1 cif 2". That is a real recognition error, not a bug in the
reading order: at 16 px the letter `o` loses a pixel of its ring after binarisation, splits into a `c`-shaped piece and a sliver, and the template matcher reads
those as `c` and `i`; the `f` then follows. A fix would be to merge touching fragments inside `segment_chars` ([step 03](../03_layout_segmentation/WALKTHROUGH.md)),
but that changes `dococr.py` for every other step, and any tuning that fits one font size can break another. We leave it visible and carry the OCR confidence (0.79 for that
footer, the lowest on the page) into step 10, which is exactly the signal a production pipeline uses to route a block to a second opinion.

**Ground truth and the two WER numbers.** The page generator ([docimg.py](../docimg.py)) returns the elements in the correct reading order, so we can score the pipeline's
text order with WER from [step 06](../06_ocr_evaluation/WALKTHROUGH.md): XY-cut gives WER 0.014 and 0.016 on the two pages (one wrong word each, the "cif"), while a naive y-sort of the same
recognised lines scores 0.554 and 0.538: all characters are right, the order is wrong.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| XY-cut | recursive XY-cut, X-Y tree, projection-profile segmentation | yes | Cut on the widest empty band, horizontal first, then vertical, recursively |
| Reading order | logical order, text flow, read order | near | Which block comes after which; in a scan it must be inferred |
| Block | region, zone, layout element, text box | near | One leaf of the cut: paragraph, title, table, footer |
| Ruling lines | table borders, grid lines, rules | yes | Long thin ink lines; also used as decoration, so lone ones are erased |
| Table structure recognition (TSR) | table parsing, table-to-HTML, grid recovery | near | Recover rows, columns, spans, then read cell text; table detection only finds the box |
| Gutter | column gap, white channel | yes | The empty vertical band between two columns |
| Layout analysis | page segmentation, document layout analysis (DLA) | near | Boxes + classes; reading order is a separate step on top |

## Run it
```bash
# from this folder
python main.py
# from the repo root
.venv/bin/python 18_ocr_and_document_parsing/07_reading_order_and_tables/main.py
```
About 7 seconds on a CPU laptop (it builds the templates for four font sizes and reads two pages; timing varies by machine). Needs only numpy, scipy and Pillow. It writes `sample_page1.png`, `sample_page2.png` and `structured_pages.json` next to the script.

## Code, step by step

### 1. Imports, the shared OCR object and the size-to-type heuristic  (`main.py` lines 19–28)
```python
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docimg import make_report_page, save_png   # noqa: E402
from dococr import binarise, runs, TemplateOCR  # noqa: E402

HERE = Path(__file__).parent
OCR = TemplateOCR()                               # templates for font sizes 16, 21, 26, 34 (the page uses all four)
KIND_BY_SIZE = {34: "title", 26: "heading", 21: "paragraph", 16: "footer"}   # font size -> element type (a heuristic)
```
- **In simple words:** Load helpers from the module (`docimg.make_report_page`, `dococr.binarise/runs/TemplateOCR`), build the template OCR once and say which font size means which block type.
- **Exactly:** `sys.path.insert(0, parents[1])` adds the module folder so `import docimg` works from this subfolder. `TemplateOCR()` renders one template per character per size (16, 21, 26, 34): that is the slow part, so it is done once at import. `KIND_BY_SIZE` is a plain dict.
- **Why it is here:** Everything below reuses these objects, and the dict is the only place that says what a "title" is.

### 2. Ruling lines: erase decoration, keep table grids  (`main.py` lines 31–53)
```python
# ---- 1. Separate ruling lines (decoration or table grid) from text -----------------------------------------------
def find_rules(ink, frac=0.5):
    """Rows whose ink spans more than `frac` of the page width are horizontal rules. Returns their (y0, y1) runs."""
    return runs(ink.sum(axis=1) > frac * ink.shape[1])


def blank_lone_rules(ink):
    """A single long thin line (e.g. under the title) is decoration: erase it. Three or more nearby lines = table: keep."""
    rules = find_rules(ink)
    out = ink.copy()
    groups, cur = [], []
    for r in rules:                                    # group rules that are less than 80 px apart
        if cur and r[0] - cur[-1][1] > 80:
            groups.append(cur)
            cur = []
        cur.append(r)
    if cur:
        groups.append(cur)
    for g in groups:
        if len(g) < 3:                                 # fewer than 3 horizontal lines cannot form a table
            for y0, y1 in g:
                out[y0:y1] = False
    return out
```
- **In simple words:** Find rows where more than half the page width is ink. A single such line (the rule under the title) is decoration and is erased; three or more close together are a table and stay.
- **Exactly:** `ink.sum(axis=1)` counts ink pixels per row (a 1-D array of length H); `runs(mask)` ([step 03](../03_layout_segmentation/WALKTHROUGH.md)) turns True stretches into `(start, end)` pairs. Rules are grouped when closer than 80 px; groups with fewer than 3 lines are blanked with `out[y0:y1] = False`.
- **Why it is here:** Without this the title underline would become its own block (the BREAK IT). The `< 3` rule encodes: a table needs at least a header line, a separator and a bottom line.

### 3. widest_gap: the empty band the cut will use  (`main.py` lines 56–66)
```python
# ---- 2. Recursive XY-cut: split on the widest empty horizontal band, else the widest empty vertical band ----------------
def widest_gap(nonempty, min_gap):
    """Widest run of False strictly between True values, if at least min_gap long. Returns (start, end) or None."""
    idx = np.flatnonzero(nonempty)
    if idx.size < 2:
        return None
    gaps = [(idx[i + 1] - idx[i] - 1, idx[i] + 1, idx[i + 1]) for i in range(len(idx) - 1) if idx[i + 1] - idx[i] - 1 >= min_gap]
    if not gaps:
        return None
    _, a, b = max(gaps)
    return int(a), int(b)
```
- **In simple words:** Given a 1-D True/False array (does this row or column contain ink?), return the widest run of False between two True values, if it is long enough.
- **Exactly:** `np.flatnonzero` gives indices of True. The list comprehension takes consecutive index pairs, the gap is `idx[i+1] - idx[i] - 1`, kept only if `>= min_gap`. `max(gaps)` compares tuples so the widest wins; returns `(start, end)` of the empty band or `None`.
- **Why it is here:** It is the single primitive for both axes: rows for horizontal cuts, columns for vertical cuts. Gaps at the very edges of the region do not count (we trim first).

### 4. xy_cut: the recursion that produces the reading order  (`main.py` lines 69–92)
```python
def xy_cut(ink, box, min_gap_y=14, min_gap_x=20, depth=0, log=None):
    """box = (x0, y0, x1, y1). Returns leaf boxes in reading order. log collects a printable tree."""
    x0, y0, x1, y1 = box
    sub = ink[y0:y1, x0:x1]
    rows, cols = np.flatnonzero(sub.any(axis=1)), np.flatnonzero(sub.any(axis=0))
    if rows.size == 0:
        return []
    x0, x1, y0, y1 = (int(v) for v in (x0 + cols[0], x0 + cols[-1] + 1, y0 + rows[0], y0 + rows[-1] + 1))   # trim to the ink (plain ints for JSON)
    sub = ink[y0:y1, x0:x1]
    gy = widest_gap(sub.any(axis=1), min_gap_y)         # empty horizontal band? -> top part first, then bottom
    if gy:
        if log is not None:
            log.append("  " * depth + f"H-cut at y={y0 + gy[0]}..{y0 + gy[1]} (gap {gy[1] - gy[0]}px) of box {(x0, y0, x1, y1)}")
        return (xy_cut(ink, (x0, y0, x1, y0 + gy[0]), min_gap_y, min_gap_x, depth + 1, log) +
                xy_cut(ink, (x0, y0 + gy[1], x1, y1), min_gap_y, min_gap_x, depth + 1, log))
    gx = widest_gap(sub.any(axis=0), min_gap_x)         # no horizontal band: empty vertical band? -> left part first
    if gx:
        if log is not None:
            log.append("  " * depth + f"V-cut at x={x0 + gx[0]}..{x0 + gx[1]} (gap {gx[1] - gx[0]}px) of box {(x0, y0, x1, y1)}")
        return (xy_cut(ink, (x0, y0, x0 + gx[0], y1), min_gap_y, min_gap_x, depth + 1, log) +
                xy_cut(ink, (x0 + gx[1], y0, x1, y1), min_gap_y, min_gap_x, depth + 1, log))
    if log is not None:
        log.append("  " * depth + f"LEAF {(x0, y0, x1, y1)}")
    return [(x0, y0, x1, y1)]
```
- **In simple words:** Trim the region to its ink, try a horizontal cut, else a vertical cut, else return the region as a block. Concatenating the results of the top/left part first gives reading order for free.
- **Exactly:** `box = (x0, y0, x1, y1)`. `sub.any(axis=1)` is a bool vector per row (ink in this row?). Coordinates are converted with `int(...)` because numpy integers are not JSON serialisable (step 07 writes JSON). The optional `log` list receives one indented line per cut, which is the printed tree. Returns a list of leaf boxes.
- **Why it is here:** The order of the two recursive calls (top before bottom, left before right) is the whole reading-order policy; for right-to-left scripts you would swap the vertical case.

### 5. read_table: grid from ruling lines, one OCR call per cell  (`main.py` lines 95–116)
```python
# ---- 3. Table structure from ruling lines ----------------------------------------------------------------------------------
def read_table(ink, box):
    """Find the grid from long horizontal and vertical lines, then OCR each cell. Returns (rows, mean confidence)."""
    x0, y0, x1, y1 = box
    t = ink[y0:y1, x0:x1]
    hy = [(a + b) // 2 for a, b in runs(t.sum(axis=1) > 0.8 * t.shape[1])]       # y centre of each horizontal rule
    vx = [(a + b) // 2 for a, b in runs(t.sum(axis=0) > 0.8 * t.shape[0])]       # x centre of each vertical rule
    rows, confs = [], []
    for r in range(len(hy) - 1):
        row = []
        for c in range(len(vx) - 1):
            cell = t[hy[r] + 4: hy[r + 1] - 3, vx[c] + 4: vx[c + 1] - 3]           # inset: do not read the border lines
            lines = OCR.read_region(cell)
            row.append(" ".join(l["text"] for l in lines))
            confs += [l["conf"] for l in lines]
        rows.append(row)
    return rows, float(np.mean(confs)), (len(hy) - 1, len(vx) - 1)


def to_markdown(rows):
    head, body = rows[0], rows[1:]
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in body])
```
- **In simple words:** Find horizontal and vertical lines inside the table box, take the rectangles between them as cells, shrink each cell by a few pixels and OCR it. `to_markdown` prints the rows as a Markdown table.
- **Exactly:** `t.sum(axis=1) > 0.8 * t.shape[1]` marks rows covered by ink for at least 80% of the width; each rule is a run of 2 pixels (lines are drawn 2 px wide) so `(a + b) // 2` is its centre. `len(hy) - 1` rows by `len(vx) - 1` columns. The `+4` and `-3` insets keep the border out of the cell. Returns `(rows, mean confidence, (n_rows, n_cols))`; `rows` is a list of lists of strings.
- **Why it is here:** The Markdown form is what a chunker ([module 19](../../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/main.py)) and an LLM can read; the list-of-lists form is what step 10 stores.

### 6. parse_page: binarise, cut, and read each block by its type  (`main.py` lines 119–136)
```python
# ---- 4. Parse one page --------------------------------------------------------------------------------------------------------
def parse_page(gray, log=None):
    ink = blank_lone_rules(binarise(gray))
    leaves = xy_cut(ink, (0, 0, ink.shape[1], ink.shape[0]), log=log)
    elements = []
    for order, (x0, y0, x1, y1) in enumerate(leaves):
        t = ink[y0:y1, x0:x1]
        is_table = len(runs(t.sum(axis=1) > 0.8 * t.shape[1])) >= 3                # three or more full-width rules
        if is_table:
            rows, conf, shape = read_table(ink, (x0, y0, x1, y1))
            elements.append(dict(order=order, kind="table", text=to_markdown(rows), rows=rows, grid=shape,
                                 bbox=[x0, y0, x1, y1], confidence=round(conf, 3)))
        else:
            lines = OCR.read_region(t)
            kind = KIND_BY_SIZE.get(lines[0]["size"], "paragraph")
            elements.append(dict(order=order, kind=kind, text=" ".join(l["text"] for l in lines), bbox=[x0, y0, x1, y1],
                                 confidence=round(float(np.mean([l["conf"] for l in lines])), 3)))
    return elements, ink
```
- **In simple words:** Run the whole page pipeline: Otsu binarisation, erase lone rules, XY-cut, then for each leaf decide table or text and read it.
- **Exactly:** A leaf is a table when it contains three or more rows that are at least 80% ink. Otherwise `OCR.read_region` returns one dict per text line (`text`, `conf`, `size`, `y0`, `y1`). The block type comes from the font size of its first line via `KIND_BY_SIZE`. Each element is a dict: `order`, `kind`, `text`, `bbox`, `confidence` (mean of the line confidences).
- **Why it is here:** This is the contract output of the step: a list of typed, ordered, boxed blocks per page; step 10 converts exactly this into the schema module 19 consumes.

### 7. lev and wer: scoring the order  (`main.py` lines 139–151)
```python
# ---- 5. Compare orders: naive "sort every line by y then x" against XY-cut ---------------------------------------------------------
def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    return lev(ref.split(), hyp.split()) / len(ref.split())
```
- **In simple words:** A compact Levenshtein distance over any sequence, and WER on top of it.
- **Exactly:** Same dynamic programme as [step 06](../06_ocr_evaluation/WALKTHROUGH.md) but keeping only the previous row (`O(m)` memory). Applied to `ref.split()` it counts word-level edits; the result is divided by the number of reference words.
- **Why it is here:** To measure reading order we compare the recognised text sequence with the true sequence: one swapped column makes WER huge while CER of each line is zero.

### 8. Why min_gap_x is 20, not 30: the page-1 gutter is 25 px  (`main.py` lines 154–161)
```python
# ---- 6. Why min_gap_x is 20, not 30: the page-1 column gutter is only 25 px (left column ends x=396, right starts x=421) ------------
gray0, _ = make_report_page(0)
ink0 = blank_lone_rules(binarise(gray0))
for mgx in (30, 20):
    leaves = xy_cut(ink0, (0, 0, ink0.shape[1], ink0.shape[0]), min_gap_x=mgx)
    body = [b for b in leaves if 140 < b[1] < 300]
    print(f"min_gap_x={mgx}: {len(body)} body block(s) between the heading and the table -> {'columns SPLIT' if len(body) == 2 else 'columns MERGED'}")
print()
```
- **In simple words:** Run the same cut with two thresholds on page 1 and count how many body blocks come out.
- **Exactly:** `blank_lone_rules` + `xy_cut(..., min_gap_x=mgx)` with the value overridden; blocks whose top edge is between y=140 and y=300 are the body area between the heading and the table. Two blocks mean the columns were split, one means merged.
- **Why it is here:** It documents a real failure found while writing this step (threshold 30 merged the columns on page 1) and makes it reproducible; a parser that hides such a choice is untrustworthy.

### 9. Main loop: render a page, parse it, print blocks in reading order  (`main.py` lines 163–177)
```python
pages_out = []
for idx in range(2):
    gray, truth = make_report_page(idx)
    save_png(gray, HERE / f"sample_page{idx + 1}.png")
    log = []
    elements, ink = parse_page(gray, log)
    if idx == 0:
        print("XY-cut tree for page 1:")
        print("\n".join("   " + l for l in log))
    print(f"\npage {idx + 1}: blocks in reading order:")
    for e in elements:
        print(f"   {e['order']}  {e['kind']:<9} conf {e['confidence']:.2f}  bbox {e['bbox']}  {e['text'].splitlines()[0][:50]!r}")
    gold_text = " ".join(t["text"] if t["kind"] != "table" else " ".join(c for r in t["text"] for c in r) for t in truth
                         if t["kind"] != "footer") + " " + [t["text"] for t in truth if t["kind"] == "footer"][0]
    # gold order = title, heading, left column, right column, table, footer; truth list is already in that order (see docimg)
```
- **In simple words:** For both synthetic pages: render, save the PNG, parse, print the cut tree (page 1) and the typed blocks.
- **Exactly:** `make_report_page(idx)` returns `(gray uint8 (640, 800), truth)`; `truth` is the list of true elements in the correct order. The f-string prints order, kind, confidence and bounding box.
- **Why it is here:** The printout is the sanity check a human does first: are the blocks in the order I would read them?

### 10. Score the order against the truth, and show naive vs XY-cut  (`main.py` lines 178–196)
```python
    xy_text = " ".join(" ".join(c for r in e["rows"] for c in r) if e["kind"] == "table" else e["text"] for e in elements)
    # naive: take every text line (no table), sort by y then x, ignore columns
    flat = []
    for e in elements:
        if e["kind"] == "table":
            continue
        for l in OCR.read_region(ink[e["bbox"][1]:e["bbox"][3], e["bbox"][0]:e["bbox"][2]]):
            flat.append((e["bbox"][1] + l["y0"] // 6 * 6, e["bbox"][0], l["text"]))     # y bucketed to 6 px so same-row lines tie
    naive_text = " ".join(t for _, _, t in sorted(flat))
    table_rows = [e for e in elements if e["kind"] == "table"][0]["rows"]
    gold_table = [t for t in truth if t["kind"] == "table"][0]["text"]
    cells_ok = sum(a == b for ra, rb in zip(table_rows, gold_table) for a, b in zip(ra, rb))
    print(f"   page-{idx + 1} WER vs ground truth: XY-cut order {wer(gold_text, xy_text):.3f}  |  naive y-sort of lines (text only, table excluded) "
          f"{wer(' '.join(t['text'] for t in truth if t['kind'] not in ('table',)), naive_text):.3f}  |  table cells exact {cells_ok}/{sum(len(r) for r in gold_table)}")
    pages_out.append(dict(page=idx + 1, width=int(gray.shape[1]), height=int(gray.shape[0]), elements=elements))
    if idx == 0:
        print("\nMarkdown table from page 1:\n" + [e for e in elements if e["kind"] == "table"][0]["text"])
        print("\nnaive y-sort text (first 110 chars):", naive_text[:110])
        print("XY-cut text     (first 110 chars):", xy_text[:110])
```
- **In simple words:** Build the three texts (gold order, XY-cut order, naive y-sorted lines) and print WER and table cell accuracy.
- **Exactly:** `gold_text` joins the true elements in order (table cells flattened, footer last). The naive version OCRs every line of every non-table block, buckets y to 6 px so two lines at the same height tie, and sorts by `(y, x)`: columns interleave. `cells_ok` counts table cells that are exactly equal to the truth.
- **Why it is here:** This is the experiment that proves the cut is worth writing: same characters, order WER 0.014 versus 0.554.

### 11. Save the structured pages  (`main.py` lines 198–199)
```python
(HERE / "structured_pages.json").write_text(json.dumps(dict(source="synthetic report", pages=pages_out), indent=1))
print("\nsaved structured_pages.json:", sum(len(p["elements"]) for p in pages_out), "elements on", len(pages_out), "pages")
```
- **In simple words:** Write all elements of both pages to `structured_pages.json` and print a summary.
- **Exactly:** `json.dumps(..., indent=1)`; all values are plain Python types (that is why `xy_cut` converts to `int`). The file has `source`, then `pages: [{page, width, height, elements: [...]}]`.
- **Why it is here:** [Step 10](../10_parsed_document_handoff/WALKTHROUGH.md) starts from this file.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `ink[y0:y1, x0:x1]` | 2-D slice: rows y0..y1-1, columns x0..x1-1 (a view, no copy) |
| `a.any(axis=1)` / `a.sum(axis=0)` | reduce over columns (one value per row) / over rows (one value per column) |
| `return f(a) + f(b)` on lists | list concatenation: top part's blocks then bottom part's blocks |
| `_, a, b = max(gaps)` | tuple unpacking; tuples compare element by element, so the largest gap wins |
| `(int(v) for v in (...))` unpacked into 4 names | generator expression used to convert numpy ints to plain ints |
| `"  " * depth + f"..."` | indentation by recursion depth for the printed tree |
| `sorted(flat)` on `(y, x, text)` tuples | sorts by y first, then x, then text |
| `{34: "title", ...}.get(size, "paragraph")` | dict lookup with a default value |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `numpy.flatnonzero` | indices of True/non-zero entries | scans the flat array once |
| `numpy.ndarray.any(axis)` | True if any element along the axis is True | logical OR reduction |
| `dococr.binarise` | bool array, True = ink | Otsu threshold ([step 01](../01_images_as_arrays/WALKTHROUGH.md)) then `gray <= t` |
| `dococr.runs` | (start, end) of True stretches | first/last index scan of `np.flatnonzero` |
| `TemplateOCR.read_region` | text lines of a region with confidence and font size | row projection into lines, connected components into glyphs, L2 nearest template, best of four sizes |
| `docimg.make_report_page` | synthetic page + true elements | draws with Pillow `ImageDraw`, records each element's box |
| `json.dumps` | serialise dict to JSON text | needs plain types: numpy scalars raise `TypeError` |

## Execution trace (real run)
```
min_gap_x=30: 1 body block(s) between the heading and the table -> columns MERGED
min_gap_x=20: 2 body block(s) between the heading and the table -> columns SPLIT

XY-cut tree for page 1:
   H-cut at y=488..588 (gap 100px) of box (40, 31, 750, 603)
     H-cut at y=261..310 (gap 49px) of box (40, 31, 750, 488)
       H-cut at y=64..106 (gap 42px) of box (41, 31, 750, 261)
         LEAF (42, 31, 494, 64)
         H-cut at y=131..151 (gap 20px) of box (41, 106, 750, 261)
           LEAF (43, 106, 205, 131)
           V-cut at x=396..421 (gap 25px) of box (41, 151, 750, 261)
             LEAF (41, 151, 396, 261)
             LEAF (421, 151, 750, 261)
       LEAF (40, 310, 742, 488)
     LEAF (362, 588, 442, 603)

page 1: blocks in reading order:
   0  title     conf 0.96  bbox [42, 31, 494, 64]  'Chai Kitchen Quarterly Report'
   1  heading   conf 0.90  bbox [43, 106, 205, 131]  'Refund policy'
   2  paragraph conf 0.91  bbox [41, 151, 396, 261]  'Refunds are issued within nine days of delivery. T'
   3  paragraph conf 0.90  bbox [421, 151, 750, 261]  'Shipping inside India takes three to five working '
   4  table     conf 0.91  bbox [40, 310, 742, 488]  '| Product | Price | Stock |'
   5  footer    conf 0.79  bbox [362, 588, 442, 603]  'Page 1 cif 2'
   page-1 WER vs ground truth: XY-cut order 0.014  |  naive y-sort of lines (text only, table excluded) 0.554  |  table cells exact 12/12

Markdown table from page 1:
| Product | Price | Stock |
|---|---|---|
| Masala Chai | 250 | 120 |
| Green Tea | 320 | 75 |
| Tulsi Blend | 290 | 0 |

naive y-sort text (first 110 chars): Chai Kitchen Quarterly Report Refund policy Refunds are issued within nine days Shipping inside India takes of
XY-cut text     (first 110 chars): Chai Kitchen Quarterly Report Refund policy Refunds are issued within nine days of delivery. The money is retu

page 2: blocks in reading order:
   0  title     conf 0.97  bbox [43, 31, 445, 64]  'Loyalty and Support Notes'
   1  heading   conf 0.93  bbox [43, 107, 206, 131]  'Loyalty points'
   2  paragraph conf 0.92  bbox [41, 151, 373, 257]  'Members earn one point for every ten rupees spent.'
   3  paragraph conf 0.92  bbox [421, 151, 729, 261]  'Support is open from nine to six on weekdays. Repl'
   4  table     conf 0.91  bbox [40, 310, 742, 488]  '| Plan | Points | Fee |'
   5  footer    conf 0.78  bbox [362, 588, 442, 603]  'Page 2 cif 2'
   page-2 WER vs ground truth: XY-cut order 0.016  |  naive y-sort of lines (text only, table excluded) 0.538  |  table cells exact 12/12

saved structured_pages.json: 12 elements on 2 pages
```
- The cut tree on page 1 reads top to bottom: three H-cuts peel off footer, table and title, then a **V-cut at x=396..421** (gap 25 px) separates the two columns, left first.
- With `min_gap_x=30` the same page yields 1 body block (columns merged); with 20 it yields 2. This is the failure the demo block reproduces.
- XY-cut order WER is 0.014 (page 1) and 0.016 (page 2): in both cases exactly one wrong word, the footer "of" read as "cif". The naive y-sort scores 0.554 and 0.538 with the very same recognised text.
- Both tables are read perfectly (12/12 cells) and come out as a Markdown table; confidence is lowest for the footer (0.79 and 0.78): the OCR itself tells you where to look.
- Timing and exact confidences vary slightly by machine only through floating point; the structure is deterministic.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| `min_gap_x=30` for the whole script | Page 1 body becomes one paragraph block; XY-cut WER on page 1 jumps from 0.014 to about 0.44 | Gutter (25 px) is narrower than the threshold: columns never cut |
| `min_gap_y=6` | Every text line becomes its own block | Line gaps are about 8 px, so each one is now a cut |
| `min_gap_y=40` | Heading and paragraph merge into one block (the 20 px band is no longer cut); the title still separates (42 px) | The heading-to-paragraph band is only 20 px |
| Raise `frac` in `find_rules` to 0.95 | `find_rules` returns `[]`: the title underline is no longer erased and becomes a junk block | The underline spans 90% of the page width and each table rule 88%, both below 95% |
| Call `OCR.read_region` on the footer with `TemplateOCR(sizes=(16,))` | Same "cif" | The misread is in the glyph split, not in size selection |
| Draw the table without the outer border | Fewer than 3 horizontal rules: read as text, cells run together | Ruling-line detection has nothing to latch onto |

## Debugging practice
**BREAK IT:** delete the call `blank_lone_rules(...)` in `parse_page` (use `ink = binarise(gray)`).

**Symptom (real run):** the XY-cut gets an extra leaf `(40, 78, 761, 79)`, the 1-pixel-high title rule. `read_region` reads it as the text `'.'` with confidence 0.003 and, because its "font size" falls back to 16, `parse_page` labels it a **footer** block. The page now has a junk footer in the middle, right after the title. Nothing crashes: the failure is silent, which is why it is dangerous, and in step 10 the pydantic check would only catch it if the text were empty.

**Fix:** keep `blank_lone_rules` before the cut. In general, strip decoration before layout analysis and **validate every block** (non-empty text, sensible box) before it leaves the parser; step 10 does the second part with pydantic.

## Common mistakes
- Cutting vertically first. A wide heading spanning both columns makes the vertical band appear empty only after the heading is separated: always try horizontal cuts first.
- Choosing thresholds on one page. Calibrate on the narrowest gutter and the tightest line spacing of your corpus, and log the cuts.
- Treating a sorted-by-y list as reading order on a multi-column page, then blaming the OCR for "garbled" text.
- Reading cell text including the border pixels: the `|` and `-` become characters like `l`, `1`, `_` inside cells (the inset in `read_table` prevents it).
- Believing XY-cut handles skew, noise or a full-width figure inside a column: de-skew (step 02) first and expect to need a learned layout model on real documents.
- Reporting 100% for the table because the **text** matches. Check the grid too (step 06's TEDS-like score), and merged cells need a real structure model.

## Rebuild checklist
1. Import numpy, Pillow helpers and the shared OCR object; write `KIND_BY_SIZE`.
2. `find_rules` (row ink fraction), then `blank_lone_rules` (group rules, blank groups of fewer than 3).
3. `widest_gap(nonempty, min_gap)` over a boolean vector.
4. `xy_cut`: trim to ink, horizontal cut, vertical cut, leaf; pass `log` and `depth`; return top/left first.
5. `read_table`: rule centres from 80%-coverage rows and columns, inset cells, `OCR.read_region`, return rows and mean confidence; `to_markdown`.
6. `parse_page`: binarise, erase lone rules, cut, classify table by three full-width rules, read the rest and type it by size.
7. `lev` and `wer`; build gold, XY-cut and naive texts and print the three numbers.
8. Dump `structured_pages.json` with plain `int`s.

## Say it aloud
- "Reading order is not stored in a scan; I infer it with a recursive XY-cut: cut the widest empty horizontal band, else the widest empty vertical band, and the recursion order is the reading order."
- "The thresholds are the model: on this page the gutter was 25 px and a threshold of 30 silently merged the columns, so I log every cut."
- "A naive sort by y had the same characters but WER 0.55; the cut brought it to 0.014, which is the difference between usable and unusable chunks for RAG."
- "Bordered tables come from ruling lines, one OCR call per cell; borderless or merged-cell tables need a learned structure model such as TableFormer."
- "The footer misread is a real glyph-segmentation error; I keep the per-block confidence so downstream code can route low-confidence blocks to a second engine.

## The new problem (cliffhanger)
We now have typed, ordered, boxed blocks, but only for a rendered picture. Most business documents arrive as **PDF files**, and a PDF may hold real text, only a scanned image, or both on different pages. Running OCR on a born-digital PDF throws away free, exact text; skipping OCR on a scan returns nothing. [Step 08](../08_pdf_internals_and_routing/WALKTHROUGH.md) opens the PDF format by hand and builds the router that decides, page by page.
