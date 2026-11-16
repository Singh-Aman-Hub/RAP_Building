"""
GOAL:  Cut a clean page into lines, words and characters with projection profiles and connected components,
       and draw the bounding boxes. This is the "segmentation" half of classic OCR.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/03_layout_segmentation/main.py
EXPECTED OUTPUT (shortened):
    lines found 6 (truth 6)   words found ... (truth ...)   connected components ... vs non-space characters ...
    gap widths ... Otsu on gaps says ... | quarter-of-line-height rule says ...
    boxes saved to sample_boxes.png
NEEDS: numpy, scipy, Pillow.  Instant.
"""
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

# ---- 1. Render and binarise ---------------------------------------------------------------------------
page, _ = render_lines(LINES, size=24, pad=24, line_gap=1.55)
ink = binarise(page)                                    # bool (H, W), True = ink
print("page", page.shape, "ink fraction %.3f" % ink.mean())


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


# ---- 3. Lines: horizontal projection profile = ink pixels per row --------------------------------------
row_profile = ink.sum(axis=1)                           # (H,)
lines = find_runs(row_profile > 0)
print(f"lines found {len(lines)} (truth {len(text_lines)});  line heights {[b - a for a, b in lines]}")

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

words = []                                              # (x0, y0, x1, y1, line_index)
for li, (y0, y1) in enumerate(lines):
    col = ink[y0:y1].sum(axis=0) > 0
    for x0, x1 in find_runs(col, min_gap=word_thr):     # bridge gaps up to the threshold: letters join into words
        words.append((x0, y0, x1, y1, li))
n_true_words = sum(len(t.split()) for t in text_lines)
print(f"words found {len(words)} (truth {n_true_words})")

# ---- 5. Characters: connected component labelling ------------------------------------------------------
labels, n_cc = ndimage.label(ink)                       # 4-connected by default; labels: int array (H, W), 0 = background
boxes = ndimage.find_objects(labels)                    # one (row_slice, col_slice) per component
n_true_chars = sum(len(t.replace(" ", "")) for t in text_lines)
print(f"connected components {n_cc} vs non-space characters {n_true_chars}")
# i, j, ; : ! ? are two or three pieces each; touching letters would be fewer. Count the multi-piece glyphs:
multi = sum(sum(t.count(c) for c in "ij;:!?=%") for t in text_lines)
print(f"  glyphs made of 2 pieces (i j ; : !): {multi}  -> expected components if nothing touches: {n_true_chars + multi}")

# ---- 6. Per-line component boxes sorted into reading order ------------------------------------------------
def component_boxes(y0, y1):
    lab, _ = ndimage.label(ink[y0:y1])
    return sorted(((s[1].start, s[0].start + y0, s[1].stop, s[0].stop + y0) for s in ndimage.find_objects(lab)))


first_line = component_boxes(*lines[0])
print("first line:", len(first_line), "components; first three boxes (x0, y0, x1, y1):", first_line[:3])
print("  width x height of those:", [(b[2] - b[0], b[3] - b[1]) for b in first_line[:3]])

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

# TRY IT:   render with size=14 (a small font) and watch word and character counts drift from the truth.
# BREAK IT: use find_runs(col, min_gap=0) for words (every letter gap splits) or min_gap=30 (words merge).
