# Step 02 — Preprocessing: noise, morphology, skew, resolution

## The story so far
[Step 01](../01_images_as_arrays/WALKTHROUGH.md) turned pixels into a binary ink mask and showed that thresholding breaks when the page is dim or shadowed. It ended on a warning: real scans are also noisy, rotated and low-resolution.

## The pain point
Dust and sensor noise create thousands of tiny fake "letters". A page scanned 4 degrees crooked makes two text lines overlap in every row, so a line finder reports one giant line. A 75 DPI photo has letters only 6 pixels tall, and neighbouring letters melt together.

## The spark
Clean the image with small neighbourhood operations. The **median** of a 3x3 window ignores an extreme pixel; **morphology** (shrink, then grow) removes specks and fills pinholes; the **projection profile** of a straight page has sharp peaks, so try rotating until the peaks are sharpest.

## Goal
Implement a median filter and four morphology operators from scratch, check them against `scipy.ndimage`, find a 4 degree skew by brute-force search, and measure what lower resolution does to the glyphs.

## The theory (read this before the interview)
**Noise types.**
| Noise | Looks like | Typical source | Best simple fix |
|---|---|---|---|
| Gaussian | every pixel jittered a little | sensor, low light | Gaussian / bilateral / non-local-means blur, or just a good threshold |
| Salt-and-pepper (impulse) | isolated pure black / white pixels | dust, bad fax, dead pixels | **median filter** |
| Bleed-through, stains | large smooth patches | old paper | adaptive threshold, learned cleaners |

A mean (box) filter averages the outlier into its neighbours, so a black speck becomes a gray smudge. A median filter throws the outlier away. It also keeps edges sharp, because the median of a 3x3 window across an edge is one of the two sides, not a blend.

**Morphology** works on binary images with a small window (structuring element). Using a k x k box:
```
erode(b)  : pixel stays ink only if ALL pixels in the window are ink  (shapes shrink, specks vanish)
dilate(b) : pixel becomes ink if ANY pixel in the window is ink       (shapes grow, gaps fill)
open  = dilate(erode(b))   removes specks smaller than the window, keeps big shapes about the same size
close = erode(dilate(b))   fills holes and gaps smaller than the window, but also merges touching shapes
```
Danger: letters at 26 px have strokes only 2 to 3 px wide. A 3x3 opening erases strokes that thin. Use morphology after the median filter, or with a window smaller than the stroke width.

**Skew.** A scanned page rotated by theta makes baselines slope. Detection by **projection profile**: for each candidate angle, rotate the ink mask and compute the row sums. When lines are horizontal, rows alternate between "full of ink" and "empty", so the variance of the row sums is highest. Pick the angle with maximum variance. Alternatives: Hough transform on text lines, minimum-area rectangle of the ink pixels, and learned orientation models. Search range is usually +/-10 degrees; 90/180/270 degree rotations need a separate orientation classifier (modern pipelines such as PaddleOCR and docling include one).

**Resolution.** Classic OCR is tuned for about 300 DPI (x-height roughly 20 px). Below about 150 DPI, strokes shrink to 1 px, letters touch (`rn` looks like `m`), and recognition accuracy falls steeply. Upscaling a low-resolution scan does not add information, though it can help engines that expect larger glyphs. VLM OCR resizes pages to a fixed number of visual tokens, so it has an effective resolution limit as well (long pages are tiled or downscaled).

**Modern (2026) practice.** Deep recognisers are far more tolerant of noise and moderate skew. The remaining checklist before OCR: fix page orientation (0/90/180/270), render PDFs at 200-300 DPI, do not over-binarise colour scans, and run adaptive thresholding only for classic engines. Over-processing hurts: a median filter on an already clean 300 DPI scan only blurs thin strokes.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Median filter | rank filter (rank = middle), "despeckle" | near | Despeckle may be a different algorithm (connected component size filter) with the same goal |
| Erosion / dilation | min filter / max filter on gray images | yes (binary case) | On binary data, min = AND, max = OR over the window |
| Opening / closing | morphological open / close | yes | Open = erode then dilate; close = dilate then erode |
| Deskew | skew correction, de-rotation, straightening | yes | Fixes small angles; orientation fixing handles 90/180/270 |
| Skew vs warp / dewarp | page curvature correction, "unwarping" | no | Skew is a rigid rotation; warp is a curved page (book spine) that needs a non-linear model |
| Denoise vs despeckle vs blur | smoothing | family | Blur averages (hurts edges), median removes outliers, despeckle removes small blobs |
| DPI vs pixel size | resolution, x-height in pixels | near | What OCR cares about is pixels per letter, not the DPI label |

## Run it
```bash
cd 18_ocr_and_document_parsing/02_preprocessing
python main.py       # under 5 seconds; writes sample_*.png
```

## Code, step by step

### 1. Imports, seed, the test text  (`main.py` lines 15–30)
```python
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docimg import render_lines, add_gaussian, add_salt_pepper, save_png   # noqa: E402
from dococr import otsu_threshold, runs                                     # noqa: E402

HERE = Path(__file__).parent
rng = np.random.default_rng(1)
LINES = ["Refunds are issued within nine days", "of delivery. Opened tea tins cannot", "be returned. Call 080 4000 1234 today.",
         "Total due: $1,250.00 (incl. tax)"]
```
- **In simple words:** Load tools, fix the random seed, and define four lines of text (with digits and punctuation) to degrade.
- **Exactly:** `sliding_window_view` returns a read-only view that exposes every k x k window of an array without copying. `dococr` supplies `otsu_threshold` (step 01) and `runs` (step 03). `default_rng(1)` seeds the noise.
- **Why it is here:** The text contains `$1,250.00` and `(incl. tax)` because punctuation and digits are what hurts first.

### 2. A clean page, its ink mask, and a noisy copy  (`main.py` lines 32–38)
```python
# ---- 1. A clean page, its ink mask, and a noisy copy ------------------------------------------------------
clean, _ = render_lines(LINES, size=26, pad=30)             # (H, W) uint8, black text on white
clean_ink = clean < 128                                      # ground-truth ink mask
noisy = add_gaussian(add_salt_pepper(clean, 0.06, rng), 25, rng)   # 6% salt & pepper, then gaussian sigma 25
save_png(clean, HERE / "sample_clean.png")
save_png(noisy, HERE / "sample_noisy.png")
print("page", clean.shape, "ink pixels in truth:", int(clean_ink.sum()))
```
- **In simple words:** Draw the page, remember where the ink really is, then wreck a copy with specks and jitter.
- **Exactly:** `render_lines` returns `uint8[216, 520]`; `clean < 128` gives 8,471 ink pixels. `add_salt_pepper(clean, 0.06, rng)` flips 6% of the pixels to 0 or 255, then `add_gaussian(..., 25, rng)` adds noise with standard deviation 25.
- **Why it is here:** A clean ground truth is what lets us give every step below a number.

### 3. Score helper: Otsu then F1  (`main.py` lines 41–46)
```python
def ink_f1(gray):
    """Binarise with Otsu and score against the clean ink mask."""
    pred = gray <= otsu_threshold(gray)
    tp = (pred & clean_ink).sum()
    p, r = tp / max(pred.sum(), 1), tp / clean_ink.sum()
    return 2 * p * r / (p + r)
```
- **In simple words:** Binarise a gray image and measure how much of the true ink it recovers.
- **Exactly:** `pred = gray <= otsu_threshold(gray)`, then pixel F1 against `clean_ink`.
- **Why it is here:** We compare three filters on the same footing.

### 4. Median filter from scratch  (`main.py` lines 50–53)
```python
def median3(img):
    padded = np.pad(img, 1, mode="symmetric")                # copy edge pixels outwards so borders have neighbours
    windows = sliding_window_view(padded, (3, 3))            # (H, W, 3, 3): a view, no copy
    return np.median(windows, axis=(2, 3)).astype(np.uint8)  # median over each 3x3 -> (H, W)
```
- **In simple words:** For every pixel, look at its 3x3 neighbourhood and take the middle value.
- **Exactly:** `np.pad(img, 1, mode="symmetric")` makes the array `(218, 522)`. `sliding_window_view(padded, (3, 3))` has shape `(216, 520, 3, 3)`. `np.median(..., axis=(2, 3))` collapses the last two axes to `(216, 520)`. `symmetric` padding matches scipy's default `reflect` border mode.
- **Why it is here:** It is 4 lines, and it removes impulse noise without blurring stroke edges.

### 5. Compare raw, box blur and median  (`main.py` lines 56–62)
```python
med = median3(noisy)
box = ndimage.uniform_filter(noisy.astype(np.float32), 3).astype(np.uint8)   # 3x3 mean blur for comparison
print("noise: F1 of ink after Otsu:  raw %.3f   box blur %.3f   median %.3f"
      % (ink_f1(noisy), ink_f1(box), ink_f1(med)))
print("median from scratch == scipy.ndimage.median_filter:",
      bool(np.array_equal(med, ndimage.median_filter(noisy, size=3))))
save_png(med, HERE / "sample_median.png")
```
- **In simple words:** Run three versions through the same score and also check our code against scipy's.
- **Exactly:** The F1 values are in the trace. `np.array_equal(med, ndimage.median_filter(noisy, size=3))` is `True`: the from-scratch version is exactly scipy's.
- **Why it is here:** "Equal to the library" is the cheapest correctness test for code you wrote yourself.

### 6. Erode, dilate, open, close  (`main.py` lines 66–79)
```python
def erode(b, k=3):
    """A pixel stays ink only if ALL k x k neighbours are ink (shrinks shapes, deletes specks)."""
    w = sliding_window_view(np.pad(b, k // 2, constant_values=False), (k, k))
    return w.all(axis=(2, 3))


def dilate(b, k=3):
    """A pixel becomes ink if ANY neighbour is ink (grows shapes, fills small gaps)."""
    w = sliding_window_view(np.pad(b, k // 2, constant_values=False), (k, k))
    return w.any(axis=(2, 3))


opening = lambda b, k=3: dilate(erode(b, k), k)     # erode then dilate: removes specks, keeps big shapes
closing = lambda b, k=3: erode(dilate(b, k), k)     # dilate then erode: fills pinholes and tiny gaps
```
- **In simple words:** Erode shrinks ink and kills specks. Dilate grows ink and fills gaps. Open and close are the two combinations.
- **Exactly:** The window view has shape `(H, W, k, k)`; `.all(axis=(2, 3))` is "every neighbour is ink" (erosion) and `.any(...)` is "at least one is" (dilation). Padding with `False` means the outside of the page counts as paper. `opening` and `closing` are lambdas composing the two.
- **Why it is here:** These are the building blocks of every classic cleanup and of line-finding tricks (a wide horizontal dilation joins letters into text lines).

### 7. Verify erosion and compare cleanup recipes  (`main.py` lines 81–90)
```python
print("erosion == scipy.ndimage.binary_erosion:",
      bool(np.array_equal(erode(clean_ink), ndimage.binary_erosion(clean_ink, np.ones((3, 3)), border_value=0))))
_, n_truth = ndimage.label(clean_ink)                  # how many separate blobs a perfect page has
raw_bin = noisy <= otsu_threshold(noisy)               # binarise the noisy page WITHOUT median filtering
for name, b in [("noisy, no cleaning", raw_bin), ("opening(3x3)", opening(raw_bin)),
                ("median then Otsu", med <= otsu_threshold(med)),
                ("median, Otsu, closing(3x3)", closing(med <= otsu_threshold(med)))]:
    _, n = ndimage.label(b)
    f = (b & clean_ink).sum() * 2 / (b.sum() + clean_ink.sum())      # Dice = F1 on pixel sets
    print(f"  {name:<28} components {n:5d} (truth {n_truth})   ink F1 {f:.3f}")
```
- **In simple words:** Check erosion against scipy, then try four cleanup recipes and count the blobs that remain.
- **Exactly:** `ndimage.label` counts connected components: a perfect page has 128. `Dice` = `2|A and B| / (|A| + |B|)`, identical to pixel F1. The output is in the trace: opening the *noisy* binary page destroys the thin strokes (F1 0.284), the median then Otsu is the best recipe (0.892), and closing afterwards merges touching letters (95 blobs instead of 128).
- **Why it is here:** It shows that the order and the choice of operations matter, and that "more cleaning" is not always better.

### 8. Rotate the page 4 degrees  (`main.py` lines 92–94)
```python
# ---- 4. Skew: rotate the page 4 degrees, then recover the angle from projection profiles ---------------------
skewed = np.asarray(Image.fromarray(clean).rotate(4.0, resample=Image.BICUBIC, expand=True, fillcolor=255))
save_png(skewed, HERE / "sample_skewed.png")
```
- **In simple words:** Tilt the page the way a crooked scanner would.
- **Exactly:** `Image.rotate(4.0, expand=True, fillcolor=255)` rotates counter-clockwise by 4 degrees, enlarges the canvas so nothing is cut, and fills new corners with white.
- **Why it is here:** We know the true angle, so we can check the detector.

### 9. Skew score and brute-force search  (`main.py` lines 97–109)
```python
def rotate_ink(ink, angle):
    return np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).rotate(angle, resample=Image.BILINEAR)) > 127


def profile_score(ink, angle):
    """Variance of the row sums. Aligned text lines give sharp peaks and empty gaps: high variance."""
    return float(rotate_ink(ink, angle).sum(axis=1).astype(np.float64).var())


sk_ink = skewed < 128
angles = np.arange(-10, 10.01, 0.25)                          # brute-force search, 81 candidate angles
scores = np.array([profile_score(sk_ink, a) for a in angles])
best = float(angles[scores.argmax()])
```
- **In simple words:** Try every angle from -10 to +10 degrees; the best one is where the text rows line up into sharp bands.
- **Exactly:** `rotate_ink` rotates a 0/255 version of the mask. `profile_score` is `var(row_sums)`. `angles` has 81 candidates (0.25 degree steps). `best = angles[scores.argmax()]`.
- **Why it is here:** This is the classic deskew algorithm. Cost: 81 rotations of a small image. For big pages, downscale before searching, or search coarse then fine.

### 10. Apply the correction and count text lines  (`main.py` lines 110–116)
```python
print(f"skew search: rotate by {best:+.2f} deg to straighten (we skewed it by +4.00, so -4.00 is right); "
      f"profile variance at best {scores.max():.0f} vs at 0 deg {profile_score(sk_ink, 0):.0f}")
deskewed = rotate_ink(sk_ink, best)
save_png((~deskewed * 255).astype(np.uint8), HERE / "sample_deskewed.png")
n_lines = lambda ink: len(runs(ink.sum(axis=1) > 0))          # text lines = runs of non-empty rows
print("text lines found by row projection: skewed", n_lines(sk_ink), "-> deskewed", n_lines(deskewed),
      "(truth", len(LINES), ")")
```
- **In simple words:** Rotate back and check that we find the four lines again.
- **Exactly:** The best angle is -4.00. The score at 0 degrees is 945, at the best angle 3460. `n_lines` counts runs of non-empty rows: the skewed page gives 1 "line" (all rows have ink), the deskewed page gives 4, the truth.
- **Why it is here:** It connects deskew to the next step: line segmentation depends on straight lines.

### 11. Resolution sweep  (`main.py` lines 118–128)
```python
# ---- 5. Resolution: the same page at lower and lower DPI ---------------------------------------------------------
print("resolution (300 dpi = scale 1.0):")
big, _ = render_lines(LINES, size=40, pad=30)                  # a "high-res scan" of the page
_, n_big = ndimage.label(big < 128)
for scale in [1.0, 0.75, 0.5, 0.35, 0.25]:
    h, w = big.shape
    small = np.asarray(Image.fromarray(big).resize((int(w * scale), int(h * scale)), Image.BILINEAR))
    ink = small <= otsu_threshold(small)
    lab, n = ndimage.label(ink)
    heights = [s[0].stop - s[0].start for s in ndimage.find_objects(lab)]
    print(f"  scale {scale:4.2f} (~{int(300 * scale):3d} dpi): glyph blobs {n:3d} (full-res {n_big}), median blob height {int(np.median(heights)):2d} px")
```
- **In simple words:** Shrink the same page and see when letters start to merge.
- **Exactly:** The page is rendered with font size 40 ("300 DPI"), then resized by scale 1.0 to 0.25 with bilinear interpolation. After Otsu, the number of connected components is compared with the full-resolution count 128. Median blob height shrinks 23, 16, 11, 8, 6 px; blobs drop to 126, 120 and finally 97 as neighbouring letters fuse.
- **Why it is here:** Components of touching letters are what the next steps cannot separate, so you should know the resolution where it begins.

### 12. TRY IT and BREAK IT notes  (`main.py` lines 130–132)
```python
# TRY IT:   change the salt & pepper amount to 0.2 and compare box blur with median; then try median3 with a 5x5 window.
# BREAK IT: in `erode`, drop the np.pad (use sliding_window_view(b, (k, k)) directly): the output shrinks to (H-2, W-2)
#           and the F1 line fails with 'ValueError: operands could not be broadcast together'.
```
- **In simple words:** Two experiments for you.
- **Exactly:** Comments only.
- **Why it is here:** Practice.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `sliding_window_view(a, (3, 3))` | view with shape `a.shape - 2` + `(3, 3)`; each entry is a window | `(218, 522) -> (216, 520, 3, 3)` |
| `.all(axis=(2, 3))` / `.any(axis=(2, 3))` | reduce over the window axes | erosion / dilation |
| `lambda b, k=3: ...` | small anonymous function with a default argument | `opening`, `closing` |
| `np.pad(a, n, mode="symmetric")` | add a border of n pixels by mirroring | median borders |
| `ndimage.label(b)` | label 4-connected blobs; returns `(labels, count)` | `_, n = ...` |
| `"%.3f" % x` | old-style format | the noise line |
| `[... for s in ndimage.find_objects(lab)]` | list comprehension over bounding slices | blob heights |

## Libraries & functions
| Name | Library | What it does | Important args | Returns |
|---|---|---|---|---|
| `np.median(a, axis=(2, 3))` | numpy | median over several axes | `axis` tuple | array |
| `ndimage.median_filter(a, size=3)` | scipy | reference median filter | `size`, `mode` (default `reflect`) | array |
| `ndimage.binary_erosion(b, structure, border_value)` | scipy | reference erosion | `structure=np.ones((3,3))` | bool array |
| `ndimage.uniform_filter(a, 3)` | scipy | box mean | `size` | float array |
| `ndimage.find_objects(lab)` | scipy | bounding slices per label | labelled array | list of `(slice, slice)` |
| `Image.rotate(angle, resample, expand, fillcolor)` | Pillow | rotate counter-clockwise by degrees | `expand=True` keeps corners | `Image` |
| `Image.resize((w, h), resample)` | Pillow | scale; size is `(width, height)` | `Image.BILINEAR` | `Image` |

## Execution trace (real run)
```
page (216, 520) ink pixels in truth: 8471
noise: F1 of ink after Otsu:  raw 0.801   box blur 0.770   median 0.892
median from scratch == scipy.ndimage.median_filter: True
erosion == scipy.ndimage.binary_erosion: True
  noisy, no cleaning           components  2821 (truth 128)   ink F1 0.801
  opening(3x3)                 components   134 (truth 128)   ink F1 0.284
  median then Otsu             components   133 (truth 128)   ink F1 0.892
  median, Otsu, closing(3x3)   components    95 (truth 128)   ink F1 0.870
skew search: rotate by -4.00 deg to straighten (we skewed it by +4.00, so -4.00 is right); profile variance at best 3460 vs at 0 deg 945
text lines found by row projection: skewed 1 -> deskewed 4 (truth 4 )
resolution (300 dpi = scale 1.0):
  scale 1.00 (~300 dpi): glyph blobs 128 (full-res 128), median blob height 23 px
  scale 0.75 (~225 dpi): glyph blobs 127 (full-res 128), median blob height 16 px
  scale 0.50 (~150 dpi): glyph blobs 126 (full-res 128), median blob height 11 px
  scale 0.35 (~105 dpi): glyph blobs 120 (full-res 128), median blob height  8 px
  scale 0.25 (~ 75 dpi): glyph blobs  97 (full-res 128), median blob height  6 px
```
- **noise line:** raw Otsu F1 0.801, box blur 0.770 (worse than doing nothing: speck pixels spread into smudges), median 0.892.
- **equal to scipy:** both `True`, so the from-scratch median and erosion are exact.
- **components:** the raw noisy binary page has 2,821 blobs where truth has 128: every speck is a fake "letter". Median gives 133. Opening the raw binary image gives 134 blobs but F1 0.284: it removed specks and most of the thin strokes. Closing after the median reaches 95: neighbouring letters were merged.
- **skew:** -4.00 degrees recovered. Row projection counts 1 line before and 4 after.
- **resolution:** blobs 128, 127, 126, 120, 97 as the scale goes 1.0 to 0.25 (median blob height 23 down to 6 px).

## Experiments (change ONE thing)
1. Salt-and-pepper 0.06 -> 0.2: raw F1 0.561, box blur 0.603, median 0.848. The heavier the impulse noise, the larger the median's lead.
2. Search range `np.arange(-3, 3.01, 0.25)` (does not contain the true -4): the search returns -3.00 and still finds 4 lines. The best angle is clamped to the range edge, so a result at the edge of the range means "widen the search".
3. Skew the page by 4.13 degrees instead of 4.0: the search returns -4.25 (grid step 0.25), and the lines are still found. For OCR a quarter degree is fine.
4. Drop the `np.pad` in `erode` and read the debugging section.

## Debugging practice
BREAK IT: in `erode`, replace the window line with `w = sliding_window_view(b, (k, k))`. Without padding the output is smaller, `(214, 518)`, and the first place that combines it with a full-size array fails:
```
  File ".../main.py", line 89, in <module>
    f = (b & clean_ink).sum() * 2 / (b.sum() + clean_ink.sum())
         ~~^~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (214,518) (216,520)
```
Read it bottom-up: two arrays that should be the same size are not. Print `.shape` of both and trace the smaller one back to its source. The root cause was in a different function than the one that raised. A "valid" convolution-style window shrinks the output by `k - 1`; always pad.

## Rebuild checklist
- [ ] median filter with `sliding_window_view` and `np.median`
- [ ] erode / dilate / open / close and what each does to specks and holes
- [ ] skew search: rotate, row-sum variance, argmax; say its search range and step
- [ ] check each from-scratch function against scipy on the same input
- [ ] name the resolution where letters start to touch

## The new problem (cliffhanger)
The page is now clean and straight, but it is still one big array of ink. We need to find the lines, the words and the individual characters before any recogniser can read them. Next: [step 03, layout segmentation](../03_layout_segmentation/WALKTHROUGH.md).

> **Say it aloud (interview box):** I match the filter to the noise: a median filter for salt-and-pepper because it removes outliers without blurring edges, Gaussian or bilateral for sensor noise. Morphological opening removes specks and closing fills gaps, but a 3x3 window can erase thin strokes, so I check on real samples. I detect skew by rotating the ink mask over a range of angles and picking the angle that maximises the variance of the row sums. I render at 200 to 300 DPI, since below about 150 DPI letters start to merge.
