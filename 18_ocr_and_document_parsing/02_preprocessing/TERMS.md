# Terms — Preprocessing

Folder: `18_ocr_and_document_parsing/02_preprocessing` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [Median filter](#median-filter) · [Salt-and-pepper (impulse) noise](#salt-and-pepper-impulse-noise) · [Gaussian noise](#gaussian-noise) · [Structuring element and sliding window](#structuring-element-and-sliding-window) · [Erosion and dilation](#erosion-and-dilation) · [Opening and closing](#opening-and-closing) · [Skew (and deskew)](#skew-and-deskew) · [Projection profile](#projection-profile) · [Page orientation vs skew](#page-orientation-vs-skew) · [Resolution and DPI effects](#resolution-and-dpi-effects) · [Border padding (symmetric, constant)](#border-padding-symmetric-constant) · [Dice coefficient](#dice-coefficient) · [ValueError: operands could not be broadcast together](#valueerror-operands-could-not-be-broadcast-together)

---

## Median filter  `(concept)`
**1. Definition:** Replaces each pixel by the median of its k x k neighbourhood.
**2. In plain words:** Ignores one wildly wrong pixel; keeps edges sharp. Raised ink F1 from 0.801 to 0.892 on a salt-and-pepper page.
**3. Example to quote:** > "A mean filter smears a speck into its neighbours; a median removes it."
```python
windows = sliding_window_view(np.pad(img, 1, mode="symmetric"), (3, 3))
med = np.median(windows, axis=(2, 3))
```
**Seen in:** `main.py`, `median3`

## Salt-and-pepper (impulse) noise  `(concept)`
**1. Definition:** Noise where random pixels become pure black or pure white.
**2. In plain words:** Dust, bad fax lines, dead sensor pixels.
**3. Example to quote:** > "Impulse noise breaks Otsu and creates 2,821 fake connected components on a page with 128 real ones."
**Seen in:** `main.py`, section 1

## Gaussian noise  `(concept)`
**1. Definition:** Additive noise drawn from a normal distribution, independent for each pixel.
**2. In plain words:** Every pixel is a bit off, like film grain.
**3. Example to quote:** > "Thresholding survives mild Gaussian noise but not sigma around 40 on a dim page."
**Seen in:** `main.py`, `add_gaussian`

## Structuring element and sliding window  `(concept)`
**1. Definition:** The small shape (here a 3x3 square) that defines each pixel's neighbourhood for filtering and morphology.
**2. In plain words:** The little stencil you slide over the image.
**3. Example to quote:** > "`sliding_window_view` gives all windows as a zero-copy view of shape (H, W, k, k)."
**Seen in:** `main.py`, `erode`, `dilate`

## Erosion and dilation  `(concept)`
**1. Definition:** Erosion keeps a pixel only if all window pixels are ink; dilation sets a pixel if any window pixel is ink.
**2. In plain words:** Shrink and grow.
**3. Example to quote:** > "Dilating horizontally is a trick for joining letters into words and lines."
**Seen in:** `main.py`, section 3

## Opening and closing  `(concept)`
**1. Definition:** Opening = erosion then dilation (removes small specks); closing = dilation then erosion (fills small holes and gaps).
**2. In plain words:** Open removes dust. Close seals pinholes but can glue neighbouring letters together.
**3. Example to quote:** > "A 3x3 opening on 2-3 px strokes erased most of the text (F1 0.284), so window size must stay below stroke width."
**Seen in:** `main.py`, section 3

## Skew (and deskew)  `(concept)`
**1. Definition:** Skew is a small rigid rotation of the page; deskewing rotates it back by the estimated angle.
**2. In plain words:** A crooked scan. Straighten it before finding lines.
**3. Example to quote:** > "On a page skewed by 4 degrees row projection found 1 line; after deskew it found 4."
**Seen in:** `main.py`, section 4

## Projection profile  `(concept)`
**1. Definition:** The sum of ink pixels along each row (horizontal profile) or column (vertical profile).
**2. In plain words:** A histogram of ink by row: text lines are peaks, line gaps are zeros.
**3. Example to quote:** > "Variance of the row profile is maximal when the text is horizontal, so it can drive deskew search."
```python
row_sums = ink.sum(axis=1)
score = row_sums.var()
```
**Seen in:** `main.py`, `profile_score`

## Page orientation vs skew  `(concept)`
**1. Definition:** Orientation = 0/90/180/270 degree rotation (needs a classifier); skew = small angle (projection or Hough search).
**2. In plain words:** Upside down vs slightly crooked.
**3. Example to quote:** > "Row-variance search cannot tell a 180 degree rotated page from an upright one; use an orientation model."
**Seen in:** `WALKTHROUGH.md`, theory

## Resolution and DPI effects  `(concept)`
**1. Definition:** Lower resolution means fewer pixels per glyph; below roughly 150 DPI strokes thin out and letters touch.
**2. In plain words:** Median blob height fell 23, 16, 11, 8, 6 px as scale dropped; blobs fell from 128 to 97.
**3. Example to quote:** > "Upscaling a bad scan adds no information; re-scan or render the PDF at 200-300 DPI instead."
**Seen in:** `main.py`, section 5

## Border padding (symmetric, constant)  `(syntax)`
**1. Definition:** np.pad modes: `symmetric` mirrors edge pixels, `constant_values=False` pads with paper.
**2. In plain words:** Windows near the page edge need neighbours; padding supplies them and keeps the output size.
**3. Example to quote:** > "Dropping the pad shrinks the array to (H-2, W-2) and later broadcasts fail."
**Seen in:** `main.py`, `median3`, `erode`

## Dice coefficient  `(concept)`
**1. Definition:** 2|A and B| / (|A| + |B|), equal to F1 on pixel sets.
**2. In plain words:** Overlap score between predicted ink and true ink.
**3. Example to quote:** > "Dice and F1 are the same formula for binary masks."
**Seen in:** `main.py`, section 3

## ValueError: operands could not be broadcast together  `(error)`
**1. Definition:** numpy cannot combine arrays whose shapes are not equal or broadcast-compatible.
**2. In plain words:** Here (214, 518) from an unpadded window and (216, 520) from the page.
**3. Example to quote:** > "I print the shapes of both operands and trace the smaller one back to where it shrank."
**Seen in:** BREAK IT in `main.py`
