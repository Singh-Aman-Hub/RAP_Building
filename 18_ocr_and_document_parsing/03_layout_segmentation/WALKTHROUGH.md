# Step 03 — Layout segmentation: lines, words, characters

## The story so far
[Step 02](../02_preprocessing/WALKTHROUGH.md) produced a clean, straight, binary page and showed that a skewed page makes line finding collapse (1 "line" instead of 4). Now we use the straight page.

## The pain point
The page is still one big array. A recogniser reads one small piece at a time: a character, a word or a line. Someone has to cut the page into pieces, and the cuts must follow the text, not a fixed grid.

## The spark
Text is regular. Ink is dense where letters are and empty in the gaps between lines, words and letters. Adding up the ink along rows gives one profile (lines show up as runs of non-zero rows); adding along columns inside a line gives another (gaps between words are wider than gaps between letters). Individual characters are the connected blobs of ink.

## Goal
Find lines with a horizontal projection profile, words with a vertical profile and a gap threshold, characters with connected-component labelling (`scipy.ndimage.label`), compare every count with the known truth, and draw the boxes.

## The theory (read this before the interview)
**Projection profile segmentation.** For a binary page `ink[H, W]`:
```
row_profile = ink.sum(axis=1)        # H values: ink pixels in each row
col_profile = ink.sum(axis=0)        # W values: ink pixels in each column
lines = maximal runs of rows where row_profile > 0
words = within a line: runs of columns where col_profile > 0, bridging gaps up to a threshold
```
The word-gap threshold separates two populations: gaps between letters (1 to 5 px here) and gaps between words (6 to 11 px). Good rules: a fraction of the line height (about 0.25 here), Otsu on the list of gap widths, or a 2-cluster k-means. Whichever you pick, some pairs fall in the overlap (here one "gift jars" gap is as wide as a letter gap).

**Connected components.** `ndimage.label(mask)` gives every group of touching ink pixels an integer id. Connectivity: 4-connected (default, neighbours up/down/left/right) or 8-connected (also diagonals; `structure=np.ones((3, 3))`). `ndimage.find_objects(labels)` returns one bounding slice per component. Most letters are one component. Exceptions:
| Case | Components | Effect on OCR |
|---|---|---|
| `i`, `j`, `!`, `;`, `:`, `%`, `=` | 2 or more pieces | merge pieces that overlap horizontally |
| Touching letters (`rn`, `ff`, `tt` at low DPI or bold fonts) | fewer components than letters | one blob needs to be split |
| Broken letters (thin strokes, noise) | more components than letters | pieces need to be joined |
| Cursive / handwriting / Arabic / Devanagari | letters always touch | character segmentation is impossible; this is why step 05 recognises whole lines |

**Why this breaks down.** Projection profiles assume (1) straight lines (step 02), (2) one column of text, (3) no tables or images mixed in. Two columns need recursive cuts (step 07), and character cutting fails on connected scripts and touching letters. Classic OCR engines used segmentation followed by a classifier. Modern recognisers (step 05) read whole lines with CTC or an attention decoder and need no character cut. Modern layout analysis (YOLO / DETR style detectors such as DocLayout-YOLO, PP-DocLayout, Docling's layout model) replaces the profile heuristics for complex pages. The projection profile is still the right tool for clean, single-column scans and as the inner loop of XY-cut.

**Bounding boxes.** Two conventions cause bugs: `(x0, y0, x1, y1)` corner format (Pillow, most OCR output) and numpy slices `[y0:y1, x0:x1]` (row first). `y1` and `x1` are exclusive in slices.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Layout analysis | document layout analysis (DLA), page segmentation, page layout detection | near | DLA classifies regions (title, table, figure); page segmentation (Tesseract `--psm`) mostly cuts lines and blocks |
| Text line detection | line segmentation | yes | Output is boxes or runs of rows |
| Text detection | scene text detection (EAST, DB), "detector" in PP-OCR | near | Finds text boxes at any angle in photos; our profile method needs a straight page |
| Connected component analysis | blob analysis, CCL | yes | Group touching foreground pixels |
| Word segmentation | tokenisation of a line into words | near | By gap width, not by language |
| Bounding box | bbox, ROI | yes | Order of coordinates differs between libraries |
| Character segmentation | glyph segmentation, cutting | yes | Needed by classic OCR, avoided by CRNN/CTC and VLM OCR |

## Run it
```bash
cd 18_ocr_and_document_parsing/03_layout_segmentation
python main.py       # instant; writes sample_page.png and sample_boxes.png
```

## Code, step by step

### 1. Imports and the test text  (`main.py` lines 13–31)
```python
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docimg import render_lines, save_png     # noqa: E402
from dococr import binarise, otsu_threshold   # noqa: E402

HERE = Path(__file__).parent
LINES = ["Chai Kitchen ships tea worldwide.",
         "Refunds take nine days; call 080-4000-1234.",
         "Jaipur gift jars: fifty pieces in stock!",
         "",                                          # a blank line = paragraph break
         "Loyalty members earn 1 point per Rs 10.",
         "Opened tins cannot be returned."]
text_lines = [t for t in LINES if t]                   # lines that really contain ink
```
- **In simple words:** Load tools, then define six lines of text, one of them blank like a paragraph break.
- **Exactly:** `render_lines` and `dococr.binarise` are the helpers from steps 01 and 02. `text_lines` drops the empty string, so it holds the 5 lines that contain ink. The text includes `;`, `:`, `!`, `i`, `j` (multi-piece glyphs), digits and hyphens.
- **Why it is here:** The truth (5 lines, 31 words, 160 non-space characters) is known and every later number is compared with it.

### 2. Render and binarise  (`main.py` lines 33–36)
```python
# ---- 1. Render and binarise ---------------------------------------------------------------------------
page, _ = render_lines(LINES, size=24, pad=24, line_gap=1.55)
ink = binarise(page)                                    # bool (H, W), True = ink
print("page", page.shape, "ink fraction %.3f" % ink.mean())
```
- **In simple words:** Draw the page and turn it into ink / paper.
- **Exactly:** `render_lines(..., size=24, line_gap=1.55)` returns `uint8[270, 527]`; `binarise` returns a bool mask. 7.0% of the pixels are ink.
- **Why it is here:** All following code runs on the boolean mask.

### 3. A helper that finds runs of True  (`main.py` lines 39–48)
```python
# ---- 2. Find runs of True in a 1-D profile (the one helper everything below uses) ----------------------
def find_runs(mask, min_gap=0):
    """Return [(start, end_exclusive)] of runs of True; gaps of <= min_gap False values do not split a run."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) - 1 > min_gap)   # np.diff - 1 = number of False values between two Trues
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]] + 1
    return list(zip(starts.tolist(), ends.tolist()))
```
- **In simple words:** Given a row of yes/no values, return where each stretch of "yes" starts and ends, optionally ignoring small holes.
- **Exactly:** `np.flatnonzero(mask)` gives indices of True. `np.diff(idx) - 1` is the number of False values between neighbours. A new run starts where that exceeds `min_gap`. `np.r_[...]` concatenates starts and ends; ends are made exclusive by `+ 1`.
- **Why it is here:** Lines and words are the same operation on different profiles; one helper does both. Bug magnet: off-by-one on the gap count.

### 4. Find lines with the row profile  (`main.py` lines 51–54)
```python
# ---- 3. Lines: horizontal projection profile = ink pixels per row --------------------------------------
row_profile = ink.sum(axis=1)                           # (H,)
lines = find_runs(row_profile > 0)
print(f"lines found {len(lines)} (truth {len(text_lines)});  line heights {[b - a for a, b in lines]}")
```
- **In simple words:** Add up the ink in every row. Rows with ink belong to a line; empty rows separate lines.
- **Exactly:** `ink.sum(axis=1)` has shape `(270,)`. `find_runs(row_profile > 0)` returns 5 runs with heights 23, 23, 23, 22, 22 px. The blank line in the input produced a gap, not a sixth line.
- **Why it is here:** Lines are the unit for recognition in step 05 and the first cut of XY-cut in step 07.

### 5. Measure the gaps and choose a word threshold  (`main.py` lines 56–67)
```python
# ---- 4. Words: vertical projection profile of each line; wide gaps separate words -----------------------
all_gaps = []
for y0, y1 in lines:
    col = ink[y0:y1].sum(axis=0) > 0                    # (W,) True where this line has ink in the column
    runs_x = find_runs(col)                             # runs of ink columns = letters or touching letters
    all_gaps += [runs_x[i + 1][0] - runs_x[i][1] for i in range(len(runs_x) - 1)]
all_gaps = np.array(all_gaps)
gap_thr = otsu_threshold(np.clip(all_gaps, 0, 255).astype(np.uint8))   # Otsu again: two groups of gaps
line_h = int(np.median([b - a for a, b in lines]))
word_thr = int(0.25 * line_h)                           # rule of thumb: a word space is wider than a quarter of the line height
print("gap widths (px): min", all_gaps.min(), "median", int(np.median(all_gaps)), "max", all_gaps.max(),
      "| Otsu on gaps says", gap_thr, "| quarter-of-line-height rule says", word_thr)
```
- **In simple words:** Look at every gap between inked column groups. Small gaps are between letters, bigger ones between words. Pick the dividing line.
- **Exactly:** For each line, `ink[y0:y1].sum(axis=0) > 0` is `(527,)` bool. `find_runs(col)` with no bridging returns letter-sized runs. `all_gaps` holds the widths in pixels: min 1, median 3, max 11. `otsu_threshold` reused on the gap widths says 4; the rule `0.25 * line height` says 5. We use the rule because a threshold of 4 splits wide letter gaps.
- **Why it is here:** The gap threshold is the single dial of word segmentation, and its value must come from the data (font size), not a magic number.

### 6. Cut the lines into words  (`main.py` lines 69–75)
```python
words = []                                              # (x0, y0, x1, y1, line_index)
for li, (y0, y1) in enumerate(lines):
    col = ink[y0:y1].sum(axis=0) > 0
    for x0, x1 in find_runs(col, min_gap=word_thr):     # bridge gaps up to the threshold: letters join into words
        words.append((x0, y0, x1, y1, li))
n_true_words = sum(len(t.split()) for t in text_lines)
print(f"words found {len(words)} (truth {n_true_words})")
```
- **In simple words:** Within each line, join letters whose gaps are small; whatever is left are words.
- **Exactly:** `find_runs(col, min_gap=word_thr)` bridges gaps up to 5 px. Result: 30 words against a truth of 31. In line 3, "gift jars" has a gap of 5 px, equal to the widest letter gap, so the two words merged.
- **Why it is here:** It shows that a profile method is a heuristic with a fuzzy edge, and it motivates recognisers that decide boundaries from content.

### 7. Characters = connected components  (`main.py` lines 77–84)
```python
# ---- 5. Characters: connected component labelling ------------------------------------------------------
labels, n_cc = ndimage.label(ink)                       # 4-connected by default; labels: int array (H, W), 0 = background
boxes = ndimage.find_objects(labels)                    # one (row_slice, col_slice) per component
n_true_chars = sum(len(t.replace(" ", "")) for t in text_lines)
print(f"connected components {n_cc} vs non-space characters {n_true_chars}")
# i, j, ; : ! ? are two or three pieces each; touching letters would be fewer. Count the multi-piece glyphs:
multi = sum(sum(t.count(c) for c in "ij;:!?=%") for t in text_lines)
print(f"  glyphs made of 2 pieces (i j ; : !): {multi}  -> expected components if nothing touches: {n_true_chars + multi}")
```
- **In simple words:** Give every group of touching ink pixels its own number. Each group is roughly one letter.
- **Exactly:** `ndimage.label(ink)` returns `labels (270, 527) int32` and `n_cc = 176`. There are 160 non-space characters, and 16 of them are made of two pieces (`i`, `j`, `;`, `:`, `!`), so the expected count is 176. Measured: 176. No two letters touched at 24 px.
- **Why it is here:** It proves that components are the right unit for clean print, and tells you exactly where the "one component = one letter" assumption breaks.

### 8. Boxes of the first line in reading order  (`main.py` lines 86–94)
```python
# ---- 6. Per-line component boxes sorted into reading order ------------------------------------------------
def component_boxes(y0, y1):
    lab, _ = ndimage.label(ink[y0:y1])
    return sorted(((s[1].start, s[0].start + y0, s[1].stop, s[0].stop + y0) for s in ndimage.find_objects(lab)))


first_line = component_boxes(*lines[0])
print("first line:", len(first_line), "components; first three boxes (x0, y0, x1, y1):", first_line[:3])
print("  width x height of those:", [(b[2] - b[0], b[3] - b[1]) for b in first_line[:3]])
```
- **In simple words:** Get the boxes of the letters in the first line, left to right.
- **Exactly:** Label only `ink[y0:y1]`, then convert each `(row_slice, col_slice)` to `(x0, y0, x1, y1)`; add `y0` back to translate to page coordinates. The first line has 33 components. The first box is `(25, 32, 39, 48)`, 14 px wide and 16 tall (the "C").
- **Why it is here:** Sorting by x inside a line is the reading order of the characters. This is the input a template matcher (step 04) needs.

### 9. Draw lines, words and characters  (`main.py` lines 96–107)
```python
# ---- 7. Draw: lines red, words green, characters blue ------------------------------------------------------
vis = Image.fromarray(page).convert("RGB")
d = ImageDraw.Draw(vis)
for y0, y1 in lines:
    d.rectangle([0, y0 - 1, page.shape[1] - 1, y1], outline=(230, 0, 0))
for x0, y0, x1, y1, _ in words:
    d.rectangle([x0 - 1, y0, x1, y1 - 1], outline=(0, 160, 0))
for sl in boxes:
    d.rectangle([sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1], outline=(0, 0, 255))
vis.save(HERE / "sample_boxes.png")
save_png(page, HERE / "sample_page.png")
print("boxes saved to sample_boxes.png")
```
- **In simple words:** Make a picture with red boxes around lines, green around words, blue around characters.
- **Exactly:** `ImageDraw.rectangle([x0, y0, x1, y1], outline=colour)`; `convert("RGB")` is needed to draw coloured boxes on a gray image. Open `sample_boxes.png` to see the result.
- **Why it is here:** Looking at the picture catches errors (like the merged "gift jars") that counts hide.

### 10. TRY IT and BREAK IT notes  (`main.py` lines 109–110)
```python
# TRY IT:   render with size=14 (a small font) and watch word and character counts drift from the truth.
# BREAK IT: use find_runs(col, min_gap=0) for words (every letter gap splits) or min_gap=30 (words merge).
```
- **In simple words:** Two experiments.
- **Exactly:** Comments only.
- **Why it is here:** Practice.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `np.r_[a, b]` | concatenate 1-D arrays | `np.r_[0, [3, 5]]` -> `[0 3 5]` |
| `ink[y0:y1]` | rows y0 to y1-1 | one line of the page |
| `for i, (a, b) in enumerate(pairs)` | index plus tuple unpacking | line loop |
| `sl[1].start`, `sl[1].stop` | bounds of a `slice` object | box x range |
| `sorted(generator)` | sort tuples by first element, then the next | boxes by x |
| `f"{x:.3f}"` / `"%.3f" % x` | float formatting | the ink fraction |
| `zip(starts, ends)` | pair up two lists | runs |

## Libraries & functions
| Name | Library | What it does | Important args | Returns |
|---|---|---|---|---|
| `ndimage.label(mask, structure=None)` | scipy | label connected True regions | `structure` sets 4 or 8 connectivity | `(labels, count)` |
| `ndimage.find_objects(labels)` | scipy | bounding slice of each label | | list of `(slice, slice)` |
| `np.diff(a)` | numpy | difference of neighbours | | array one shorter |
| `np.flatnonzero(a)` | numpy | indices of non-zero | | int array |
| `ImageDraw.rectangle(xy, outline)` | Pillow | draw a box | `[x0, y0, x1, y1]` | None |
| `otsu_threshold(arr)` | `dococr.py` | step 01's Otsu on any `uint8` array | | int |

## Execution trace (real run)
```
page (270, 527) ink fraction 0.070
lines found 5 (truth 5);  line heights [23, 23, 23, 22, 22]
gap widths (px): min 1 median 3 max 11 | Otsu on gaps says 4 | quarter-of-line-height rule says 5
words found 30 (truth 31)
connected components 176 vs non-space characters 160
  glyphs made of 2 pieces (i j ; : !): 16  -> expected components if nothing touches: 176
first line: 33 components; first three boxes (x0, y0, x1, y1): [(25, 32, 39, 48), (43, 30, 54, 48), (57, 35, 67, 48)]
  width x height of those: [(14, 16), (11, 18), (10, 13)]
boxes saved to sample_boxes.png
```
- **lines 5 (truth 5):** the blank input line left no ink, so it makes no run.
- **gaps:** the biggest letter gap and the smallest word gap overlap (4 to 5 px), hence the rule-based threshold of 5.
- **words 30 vs 31:** one merge ("gift jars"). Per-line counts: 5/5, 6/6, 6/7, 8/8, 5/5.
- **components 176 vs 160 + 16:** an exact match, so on clean 24 px text every letter is its own component.
- **first line:** 33 components. The first box is the capital C at x 25 to 39.

## Experiments (change ONE thing)
1. `size=14` instead of 24: words 31 of 31 (the smaller gaps happened to separate cleanly), but components jump to 278 against an expected 176: thin strokes break into pieces after thresholding. Character segmentation degrades long before word segmentation does.
2. `min_gap=0` for words: 159 "words". Every letter gap splits.
3. `min_gap=30` for words: 5 "words" (one per line). Everything merged.
4. `line_gap=1.0` (tight line spacing): still 5 lines here, because the font has no overlapping ascenders and descenders at this size. Tight proportional text with descenders touching ascenders would merge lines; that is when you need XY-cut with a smoothed profile or a layout model.

## Debugging practice
BREAK IT: in the word loop use `min_gap=0`:
```
gap widths (px): min 1 median 3 max 11 | Otsu on gaps says 4 | quarter-of-line-height rule says 5
words found 159 (truth 31)
```
No traceback: this is a silent logic error, and the most dangerous kind. The only protection is a check against known truth (31 words) or a plot. Print counts, compare with an expected value, and look at `sample_boxes.png` before you trust a segmenter.

## Rebuild checklist
- [ ] `find_runs` with a gap tolerance, and the off-by-one in `np.diff(idx) - 1`
- [ ] lines from the row profile, words from the column profile of one line
- [ ] a word-gap threshold taken from the data
- [ ] `ndimage.label` + `find_objects`, and merging multi-piece glyphs
- [ ] the conditions under which each cut fails (touching, broken, connected scripts, columns)

## The new problem (cliffhanger)
We have a box around every character. What letter is in each box? And what happens with `0` and `O`, or `l`, `I` and `1`, which look nearly the same? Next: [step 04, character recognition](../04_character_recognition/WALKTHROUGH.md).

> **Say it aloud (interview box):** Classic layout segmentation uses projection profiles: sum the ink along rows to find lines, along columns inside a line to find words using a gap threshold taken from the line height, and connected components for characters. It needs straight text, one column and clean separation. It fails on touching or broken letters and on connected scripts, which is why modern OCR recognises a whole line with CTC or a VLM and uses a learned layout detector for complex pages.
