# Terms — Reading order and tables

Folder: `18_ocr_and_document_parsing/07_reading_order_and_tables` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [Reading order](#reading-order) · [Recursive XY-cut (X-Y tree, projection-profile segmentation)](#recursive-xy-cut-x-y-tree-projection-profile-segmentation) · [Gutter](#gutter) · [Block (region, zone, layout element)](#block-region-zone-layout-element) · [Ruling lines (table borders, grid lines)](#ruling-lines-table-borders-grid-lines) · [Table structure recognition (TSR)](#table-structure-recognition-tsr) · [Markdown table](#markdown-table) · [widest_gap](#widest_gap) · [min_gap_y / min_gap_x (cut thresholds)](#min_gap_y--min_gap_x-cut-thresholds) · [Naive y-sort](#naive-y-sort) · [Block confidence](#block-confidence) · [Glyph over-segmentation (the 'of' to 'cif' error)](#glyph-over-segmentation-the-of-to-cif-error) · [Decoration vs structure (lone rule erasure)](#decoration-vs-structure-lone-rule-erasure)

---

## Reading order  `(concept)`
**1. Definition:** The sequence in which a human reads the blocks of a page; absent from a scanned image, so it must be inferred.
**2. In plain words:** Which block comes after which. Get it wrong and a column's second paragraph is glued to the other column's first.
**3. Example to quote:** > "A naive y-sort had the right characters but WER 0.55; XY-cut order brought it to 0.014."
**Seen in:** `main.py`, `xy_cut`, `parse_page`

## Recursive XY-cut (X-Y tree, projection-profile segmentation)  `(concept)`
**1. Definition:** A top-down page segmentation: cut at the widest empty horizontal band, else the widest empty vertical band, recurse until nothing can be cut; leaves are blocks in reading order.
**2. In plain words:** Use the white space between things to split a page into smaller and smaller pieces, top part and left part first.
**3. Example to quote:** > "XY-cut is the whiteboard baseline for reading order; learned layout detectors replace it on messy pages."
```python
gy = widest_gap(sub.any(axis=1), min_gap_y)
```
**Seen in:** `main.py`, `xy_cut`

## Gutter  `(concept)`
**1. Definition:** The empty vertical band between two text columns.
**2. In plain words:** The white channel between columns. Here it is 25 px wide, narrower than the first threshold (30) we tried.
**3. Example to quote:** > "Calibrate min_gap_x on the narrowest real gutter, not on a typical one."
**Seen in:** `main.py`, lines 154-161 (the min_gap_x demo)

## Block (region, zone, layout element)  `(concept)`
**1. Definition:** One leaf of the layout analysis: a title, paragraph, table, figure or footer, with a bounding box and a type.
**2. In plain words:** A rectangle of the page that means one thing.
**3. Example to quote:** > "I typed each block by font size here; production systems classify with a layout model."
**Seen in:** `main.py`, `parse_page`

## Ruling lines (table borders, grid lines)  `(concept)`
**1. Definition:** Long thin ink lines: horizontal and vertical rules that draw a table grid, or a decorative underline.
**2. In plain words:** The lines you can see in a bordered table; they tell you where the cells are.
**3. Example to quote:** > "Three or more close full-width rules mean a table; a lone one is decoration and I erase it."
```python
runs(t.sum(axis=1) > 0.8 * t.shape[1])
```
**Seen in:** `main.py`, `find_rules`, `read_table`

## Table structure recognition (TSR)  `(concept)`
**1. Definition:** Recovering the row/column grid (and spans) of a table, then reading cell text; different from table detection, which only finds the table's box.
**2. In plain words:** Not only 'there is a table here' but 'row 2, column 3 says 120'.
**3. Example to quote:** > "Ruling lines handle simple bordered tables; merged cells and borderless tables need a learned model such as TableFormer."
**Seen in:** `main.py`, `read_table`

## Markdown table  `(syntax)`
**1. Definition:** A table written as pipe-separated rows with a `---` separator row after the header.
**2. In plain words:** A text format of a table that both people and LLMs read well.
**3. Example to quote:** > "I emit the Markdown for the LLM and keep rows/cells in JSON for citations."
```python
to_markdown([["Product", "Price"], ["Tea", "250"]])
```
**Seen in:** `main.py`, `to_markdown`

## widest_gap  `(function)`
**1. Definition:** Returns the widest run of empty entries between two non-empty entries of a 1-D boolean array if at least min_gap long, else None.
**2. In plain words:** Finds where to cut.
**3. Example to quote:** > "It is the one primitive used for horizontal and vertical cuts."
```python
gaps = [(idx[i+1] - idx[i] - 1, idx[i] + 1, idx[i+1]) for i in range(len(idx) - 1) if ...]
```
**Seen in:** `main.py`, `widest_gap`

## min_gap_y / min_gap_x (cut thresholds)  `(concept)`
**1. Definition:** Minimum empty-band widths (pixels) that count as a separation between blocks vertically (paragraphs) and horizontally (columns).
**2. In plain words:** How big a white space must be before we say 'new block'.
**3. Example to quote:** > "The thresholds are the model: log every cut so a human can audit them."
```python
xy_cut(ink, box, min_gap_y=14, min_gap_x=20)
```
**Seen in:** `main.py`, `xy_cut`

## Naive y-sort  `(concept)`
**1. Definition:** Ordering all text lines by (y, x), ignoring columns.
**2. In plain words:** Reads straight across the page like a ruler, which scrambles two-column text.
**3. Example to quote:** > "Correct only for single-column pages."
**Seen in:** `main.py`, section 5 comparison

## Block confidence  `(concept)`
**1. Definition:** A per-block score, here the mean of line confidences (1 - best distance / second-best distance).
**2. In plain words:** How sure the OCR was; the footer's 0.79 flagged the one misread.
**3. Example to quote:** > "Carry confidence downstream so low-confidence blocks can be re-read by another engine."
**Seen in:** `main.py`, `parse_page`

## Glyph over-segmentation (the 'of' to 'cif' error)  `(error)`
**1. Definition:** A character split into several connected components (a ring with a one-pixel gap), each read as a different letter.
**2. In plain words:** The letter o broke into a c-shaped piece and a sliver, read as 'c' and 'i'.
**3. Example to quote:** > "The footer 'Page 1 of 2' came out as 'Page 1 cif 2': a real recognition error, left visible on purpose."
**Seen in:** `main.py` output; `dococr.segment_chars`

## Decoration vs structure (lone rule erasure)  `(concept)`
**1. Definition:** A single long line is styling; three or more close lines are structure. The first is erased before layout analysis.
**2. In plain words:** A title underline is not a table.
**3. Example to quote:** > "Without it the underline becomes a junk 'footer' block with confidence 0.003."
```python
if len(g) < 3: out[y0:y1] = False
```
**Seen in:** `main.py`, `blank_lone_rules`
