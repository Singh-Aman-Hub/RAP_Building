"""
GOAL:  Turn a two-column page with a title, a ruled table and a footer into ordered, typed blocks:
       recursive XY-cut for reading order, ruling-line detection for the table grid, template OCR for the text,
       and a Markdown table. Compare with the naive "sort lines by y" order. Saves structured_pages.json for step 10.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/07_reading_order_and_tables/main.py
EXPECTED OUTPUT (shortened):
    XY-cut tree ...   blocks in reading order: title, heading, paragraph, paragraph, table, footer
    naive y-sort WER ...   XY-cut WER ...
    | Product | Price | Stock |  ...
    saved structured_pages.json
NEEDS: numpy, scipy, Pillow.  A few seconds.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docimg import make_report_page, save_png   # noqa: E402
from dococr import binarise, runs, TemplateOCR  # noqa: E402

HERE = Path(__file__).parent
OCR = TemplateOCR()                               # templates for font sizes 16, 21, 26, 34 (the page uses all four)
KIND_BY_SIZE = {34: "title", 26: "heading", 21: "paragraph", 16: "footer"}   # font size -> element type (a heuristic)


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


# ---- 6. Why min_gap_x is 20, not 30: the page-1 column gutter is only 25 px (left column ends x=396, right starts x=421) ------------
gray0, _ = make_report_page(0)
ink0 = blank_lone_rules(binarise(gray0))
for mgx in (30, 20):
    leaves = xy_cut(ink0, (0, 0, ink0.shape[1], ink0.shape[0]), min_gap_x=mgx)
    body = [b for b in leaves if 140 < b[1] < 300]
    print(f"min_gap_x={mgx}: {len(body)} body block(s) between the heading and the table -> {'columns SPLIT' if len(body) == 2 else 'columns MERGED'}")
print()

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

(HERE / "structured_pages.json").write_text(json.dumps(dict(source="synthetic report", pages=pages_out), indent=1))
print("\nsaved structured_pages.json:", sum(len(p["elements"]) for p in pages_out), "elements on", len(pages_out), "pages")

# TRY IT:   set min_gap_y to 6 (every line becomes a block) or to 40 (heading merges with paragraph); change min_gap_x to 30 (columns merge).
# BREAK IT: remove blank_lone_rules and watch the title rule become its own block; or draw a table without its outer border and the
#           ruling-line detector finds fewer than 3 lines, so it is read as text.
