# Terms — Layout segmentation

Folder: `18_ocr_and_document_parsing/03_layout_segmentation` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [Row and column projection profile](#row-and-column-projection-profile) · [Run (maximal run of True)](#run-maximal-run-of-true) · [Word-gap threshold](#word-gap-threshold) · [Connected components (labelling)](#connected-components-labelling) · [Multi-piece glyphs (i, j, ;, :, !)](#multi-piece-glyphs-i-j) · [Touching and broken characters](#touching-and-broken-characters) · [Bounding box (bbox)](#bounding-box-bbox) · [Page segmentation and layout analysis](#page-segmentation-and-layout-analysis) · [np.diff and np.flatnonzero](#npdiff-and-npflatnonzero) · [Reading order of characters](#reading-order-of-characters) · [Silent logic error](#silent-logic-error)

---

## Row and column projection profile  `(concept)`
**1. Definition:** Sums of ink pixels along each row (ink.sum(axis=1)) or column (ink.sum(axis=0)).
**2. In plain words:** A bar chart of "how much ink" at every height or horizontal position.
**3. Example to quote:** > "Lines are runs of non-zero rows; word gaps are runs of zero columns wider than a threshold."
```python
row_profile = ink.sum(axis=1)
lines = find_runs(row_profile > 0)
```
**Seen in:** `main.py`, sections 3-4

## Run (maximal run of True)  `(concept)`
**1. Definition:** A maximal consecutive stretch of True values in a 1-D boolean array, returned as (start, end_exclusive).
**2. In plain words:** One text line or one word seen in a profile.
**3. Example to quote:** > "I write a `find_runs(mask, min_gap)` helper once and reuse it for lines, words and tables."
**Seen in:** `main.py`, `find_runs`

## Word-gap threshold  `(concept)`
**1. Definition:** The gap width (in pixels) above which two ink groups are different words.
**2. In plain words:** Gaps between letters were 1-5 px and between words 6-11 px in this font; we used 0.25 x line height = 5.
**3. Example to quote:** > "The threshold depends on font size, so derive it from the line height or by clustering gap widths, never hard-code it."
**Seen in:** `main.py`, section 4

## Connected components (labelling)  `(concept)`
**1. Definition:** Grouping touching foreground pixels into numbered blobs; 4-connectivity uses edge neighbours, 8-connectivity also diagonals.
**2. In plain words:** Each separate squiggle of ink gets its own number; 176 on the test page.
**3. Example to quote:** > "scipy.ndimage.label returns (labels, count); find_objects gives the boxes."
```python
labels, n = ndimage.label(ink)
boxes = ndimage.find_objects(labels)
```
**Seen in:** `main.py`, section 5

## Multi-piece glyphs (i, j, ;, :, !)  `(concept)`
**1. Definition:** Characters made of several unconnected components, such as the dot and stem of an i.
**2. In plain words:** 16 such glyphs make 176 components out of 160 characters.
**3. Example to quote:** > "I merge components that overlap horizontally before classifying."
**Seen in:** `main.py`, section 5

## Touching and broken characters  `(concept)`
**1. Definition:** Touching: two letters form one blob (rn, tt). Broken: one letter splits into several blobs (thin strokes, noise).
**2. In plain words:** At size 14 the page produced 278 components instead of 176: broken glyphs.
**3. Example to quote:** > "These are the failure cases that make character segmentation unreliable and motivate CTC line recognition."
**Seen in:** `WALKTHROUGH.md`, experiments

## Bounding box (bbox)  `(concept)`
**1. Definition:** The smallest axis-aligned rectangle around a region, usually (x0, y0, x1, y1) with x1, y1 exclusive in array slices.
**2. In plain words:** A box around a letter, word or line.
**3. Example to quote:** > "Pillow boxes are (x0, y0, x1, y1); numpy slices are [y0:y1, x0:x1]: mixing them up is a classic bug."
**Seen in:** `main.py`, `component_boxes`

## Page segmentation and layout analysis  `(concept)`
**1. Definition:** Page segmentation cuts a page into text lines/blocks; layout analysis also labels regions (title, table, figure).
**2. In plain words:** Where is the text, and what kind of region is each part?
**3. Example to quote:** > "Tesseract's --psm is page segmentation; Docling and PaddleOCR use learned layout detectors."
**Seen in:** `WALKTHROUGH.md`, theory

## np.diff and np.flatnonzero  `(syntax)`
**1. Definition:** np.diff(a) gives differences of neighbours; np.flatnonzero(a) gives indices of non-zero entries.
**2. In plain words:** Turn a yes/no row into the positions of "yes", then measure gaps between them.
**3. Example to quote:** > "np.diff(idx) - 1 = number of empty positions between two filled ones."
**Seen in:** `main.py`, `find_runs`

## Reading order of characters  `(concept)`
**1. Definition:** Sorting boxes by line (y) and then by x within the line.
**2. In plain words:** Left to right, top to bottom, valid only for a single-column left-to-right page.
**3. Example to quote:** > "Multi-column pages need XY-cut (step 07)."
**Seen in:** `main.py`, section 6

## Silent logic error  `(error)`
**1. Definition:** A bug that raises no exception but gives wrong output, such as 159 words instead of 31.
**2. In plain words:** No traceback to read; you only find it by comparing with known truth.
**3. Example to quote:** > "I print counts against expectations and view the drawn boxes."
**Seen in:** BREAK IT in `main.py`
