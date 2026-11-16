# Terms — Images as arrays

Folder: `18_ocr_and_document_parsing/01_images_as_arrays` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [Pixel](#pixel) · [Image array and shape (H, W, C)](#image-array-and-shape-h-w-c) · [uint8 and dtype](#uint8-and-dtype) · [Grayscale (luma, ITU-R BT.601)](#grayscale-luma-itu-r-bt601) · [Broadcasting](#broadcasting) · [Histogram](#histogram) · [Binarisation (thresholding)](#binarisation-thresholding) · [Otsu's method](#otsus-method) · [Adaptive (local) threshold](#adaptive-local-threshold) · [Precision, recall, F1 (pixel level)](#precision-recall-f1-pixel-level) · [DPI (dots per inch)](#dpi-dots-per-inch) · [np.errstate and NaN-safe argmax](#nperrstate-and-nan-safe-argmax) · [TypeError: Cannot cast array data from dtype('float64') to dtype('int64')](#typeerror-cannot-cast-array-data-from-dtypefloat64-to-dtypeint64)

---

## Pixel  `(concept)`
**1. Definition:** The smallest addressable element of a raster image; its value is one number (gray) or a tuple (R, G, B).
**2. In plain words:** One dot of the picture. A 520 x 160 page has 83,200 of them.
**3. Example to quote:** > "A pixel is just a number between 0 and 255 in a uint8 image; OCR is arithmetic on a grid of those numbers."
**Seen in:** `main.py`, sections 1 and 4

## Image array and shape (H, W, C)  `(concept)`
**1. Definition:** A numpy array holding an image: `[rows, columns]` for gray, `[rows, columns, channels]` for colour, indexed `(y, x)`.
**2. In plain words:** The page as a spreadsheet; row first, column second. Pillow sizes are `(width, height)` and numpy shapes are `(height, width)`.
**3. Example to quote:** > "Mixing up (x, y) and (row, column) is the number one bug in image code."
```python
rgb = np.asarray(img)   # (160, 520, 3)
```
**Seen in:** `main.py`, section 1

## uint8 and dtype  `(syntax)`
**1. Definition:** Unsigned 8-bit integer, values 0 to 255, the standard dtype of image data; arithmetic that leaves 0-255 silently wraps around.
**2. In plain words:** Cheap to store, but `np.uint8(200) + np.uint8(100)` gives 44, not 300. Cast to int or float before doing maths.
**3. Example to quote:** > "I convert to float for arithmetic and back to uint8 with clipping at the end."
**Seen in:** `main.py`, sections 2, 3

## Grayscale (luma, ITU-R BT.601)  `(concept)`
**1. Definition:** A single-channel image computed as Y = 0.299 R + 0.587 G + 0.114 B; Pillow's `convert("L")` uses it.
**2. In plain words:** Colour to brightness the way the eye sees it; green counts most.
**3. Example to quote:** > "The naive average of R, G, B is up to 17 levels different from luma on the test page."
```python
gray = (0.299*R + 0.587*G + 0.114*B).round().astype(np.uint8)
```
**Seen in:** `main.py`, section 2

## Broadcasting  `(syntax)`
**1. Definition:** numpy's rule that arrays of different shapes are stretched along axes of length 1 so element-wise operations work.
**2. In plain words:** The `(1, 520)` brightness ramp is multiplied against every row of the `(160, 520)` page.
**3. Example to quote:** > "Broadcasting lets me apply a per-column gradient without a Python loop."
```python
shadow = np.linspace(1.0, 0.45, W)[None, :]   # (1, W)
page = gray * shadow                 # (H, W)
```
**Seen in:** `main.py`, section 3

## Histogram  `(concept)`
**1. Definition:** A count of pixels per gray level: `hist[v]` = number of pixels equal to v.
**2. In plain words:** A bar chart of brightness. Text pages show a small dark peak (ink) and a big bright peak (paper).
**3. Example to quote:** > "I look at the histogram first; if it is not two-peaked, a global threshold will struggle."
```python
hist = np.bincount(img.ravel(), minlength=256)
```
**Seen in:** `main.py`, section 4

## Binarisation (thresholding)  `(concept)`
**1. Definition:** Converting a gray image to a two-valued image by comparing each pixel with a threshold: ink = gray <= t.
**2. In plain words:** Every pixel becomes ink or paper.
**3. Example to quote:** > "Binarisation decides what is ink before recognition starts, so its mistakes cannot be repaired later."
**Seen in:** `main.py`, section 6

## Otsu's method  `(concept)`
**1. Definition:** An automatic global threshold: pick t maximising the between-class variance w0 * w1 * (mu0 - mu1)^2 of the histogram split at t.
**2. In plain words:** Try every cut; keep the one that separates dark and bright pixels best. It gave 69 on the dim page and 140 on the clean page.
**3. Example to quote:** > "Otsu assumes a two-peaked histogram and uniform lighting; under a shadow gradient it did worse than a fixed 128 here."
```python
sigma_b = (mu_total*w0 - m)**2 / (w0*(1-w0)); t = np.nanargmax(sigma_b)
```
**Seen in:** `main.py`, section 5

## Adaptive (local) threshold  `(concept)`
**1. Definition:** A threshold computed for each pixel from its neighbourhood, for example the local mean minus an offset (Sauvola and Niblack also use the local standard deviation).
**2. In plain words:** Ink means "darker than the paper next to it", not "darker than 128".
**3. Example to quote:** > "Adaptive thresholding fixed the shadowed page (F1 0.949 against Otsu's 0.194)."
```python
ink = page < ndimage.uniform_filter(page, size=31) - 25
```
**Seen in:** `main.py`, section 6

## Precision, recall, F1 (pixel level)  `(concept)`
**1. Definition:** Precision = tp / predicted positives, recall = tp / true positives, F1 = 2PR / (P + R).
**2. In plain words:** Did you find the ink (recall) without calling paper ink (precision)?
**3. Example to quote:** > "A threshold that marks everything as ink has recall 1.0 and a terrible precision; F1 exposes it."
**Seen in:** `main.py`, `f1()`

## DPI (dots per inch)  `(concept)`
**1. Definition:** Scan resolution: pixels per inch of paper. Classic OCR engines work best at roughly 300 DPI.
**2. In plain words:** More dots, more detail per letter.
**3. Example to quote:** > "Below about 150 DPI, small letters merge and OCR accuracy falls (measured in step 02)."
**Seen in:** `WALKTHROUGH.md`, theory

## np.errstate and NaN-safe argmax  `(syntax)`
**1. Definition:** `np.errstate(divide="ignore", invalid="ignore")` silences floating point warnings; `np.nanargmax` returns the index of the largest non-NaN value.
**2. In plain words:** At the ends of the histogram the formula is 0/0 = NaN; this skips them.
**3. Example to quote:** > "I guard expected NaNs explicitly rather than hiding warnings globally."
**Seen in:** `main.py`, `otsu()`

## TypeError: Cannot cast array data from dtype('float64') to dtype('int64')  `(error)`
**1. Definition:** Raised by `np.bincount` when it is given a float array.
**2. In plain words:** bincount counts integer values, so a 0-1 float image is the wrong input.
**3. Example to quote:** > "I keep image dtypes consistent: uint8 0-255 everywhere."
**Seen in:** BREAK IT in `main.py`
