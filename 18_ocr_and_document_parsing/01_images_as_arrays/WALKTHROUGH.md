# Step 01 — Images as arrays: pixels, grayscale, histograms, Otsu

## The story so far
This is the first step of the module. It builds on [14_rag_and_retrieval/01_documents_and_why_rag](../../14_rag_and_retrieval/01_documents_and_why_rag/WALKTHROUGH.md), where every document was already clean text. Real companies also have scanned invoices, photographed receipts and image-only PDFs. A RAG system cannot index them until the pixels become text.

## The pain point
A scanner gives you a grid of numbers, not letters. The same page can be bright, dim, shadowed or noisy, and the program has to decide which pixels are ink and which are paper before anything else can happen. A fixed rule such as "darker than 128 is ink" works on one scan and fails on the next.

## The spark
Look at the histogram. Ink pixels and paper pixels form two groups of gray levels. Pick the threshold that separates the two groups best (Otsu, 1979). If lighting changes across the page, pick a threshold per neighbourhood (adaptive thresholding).

## Goal
Treat an image as a numpy array, convert RGB to gray with the real formula, build a histogram, implement Otsu's method from scratch, and compare fixed, Otsu and local-mean thresholds on a dim page and a shadowed page using a known ground truth.

## The theory (read this before the interview)
**An image is an array.** A grayscale image is `uint8[H, W]` (rows first, then columns), values 0 (black) to 255 (white). A colour image is `uint8[H, W, 3]` (R, G, B). Pillow loads it, numpy computes on it: `np.asarray(img)`. Coordinates are `(row, column)` = `(y, x)`. This is the most common bug source in OCR code, because bounding boxes in libraries are usually `(x0, y0, x1, y1)`.

**Grayscale conversion.** Humans see green brighter than blue, so the standard weighted sum (ITU-R BT.601 "luma") is:
```
Y = 0.299 R + 0.587 G + 0.114 B
```
Pillow's `convert("L")` uses this. A plain average `(R+G+B)/3` is different by up to 17 levels in this step's colour page, and it makes yellow paper look darker than it should.

**Resolution (DPI).** Dots per inch. OCR engines were tuned for about 300 DPI scans, where lower-case letters are around 20 px tall. Step 02 shows what lower DPI does.

**Histogram.** `hist[v]` = number of pixels with gray level v. A page of text has two peaks: a big one for paper and a small one for ink.

**Binarisation (thresholding).** Output is a boolean image: `ink = gray <= t`.
| Method | Rule | Works when | Fails when |
|---|---|---|---|
| Fixed | `gray < 128` | scans are consistently exposed | dim or bright scans |
| Otsu (global) | `t` that maximises between-class variance | one lighting level over the page, two clear peaks | shadows, gradients, stains, coloured background |
| Adaptive (local mean / Gaussian / Sauvola / Niblack) | compare each pixel with its neighbourhood statistics | uneven lighting | large solid dark regions, very noisy backgrounds |
| Learned (U-Net, or a VLM that never binarises) | the network decides | hard degraded documents | needs training data or a GPU |

**Otsu's method.** For each candidate threshold `t`, split the histogram into class 0 (levels <= t) and class 1. With weights `w0, w1` and class means `mu0, mu1`:
```
sigma_b^2(t) = w0 * w1 * (mu0 - mu1)^2          choose t = argmax sigma_b^2(t)
```
Maximising between-class variance is the same as minimising within-class variance. Using cumulative sums, all 256 thresholds are computed in one pass: `w0 = cumsum(p)`, `m = cumsum(p * level)`, `sigma_b^2 = (mu_total * w0 - m)^2 / (w0 * (1 - w0))`.

**Modern (2026) practice.** Modern VLM-based OCR (step 09) takes the grayscale or colour page directly and mostly skips binarisation. Classic engines (Tesseract 5) still binarise internally and benefit from good preprocessing. For classic pipelines, a safe default is: grayscale, denoise, Otsu or adaptive threshold, deskew.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Binarisation | thresholding, black-and-white conversion, "bitonal" | yes | Output is two-valued (ink / paper) |
| Otsu threshold | Otsu's method, automatic global threshold | yes | Picks `t` from the histogram instead of using 128 |
| Adaptive threshold | local threshold, Sauvola, Niblack, "adaptiveThreshold" in OpenCV | family | One threshold per neighbourhood, not per page |
| Grayscale | luma, "L" mode in Pillow | near | Luma is the weighted-sum definition; "average gray" is a different, worse one |
| Pixel intensity | gray level, brightness | yes | 0-255 for uint8 |
| DPI | PPI, scan resolution | near | DPI is printer/scanner jargon, PPI is screen jargon, OCR docs use DPI |

## Run it
```bash
cd 18_ocr_and_document_parsing/01_images_as_arrays
python main.py       # instant; writes sample_*.png next to main.py
```
Sample images written by the run: `sample_gray.png`, `sample_dim_noisy.png`, `sample_shadow_noisy.png`, and the binarised results `sample_dim_otsu.png`, `sample_dim_adaptive.png`, `sample_shadow_otsu.png`, `sample_shadow_adaptive.png`.

## Code, step by step

### 1. Imports, shared helpers, seed  (`main.py` lines 14–25)
```python
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docimg import get_font, add_gaussian, save_png   # noqa: E402

HERE = Path(__file__).parent
rng = np.random.default_rng(0)                          # seed: the "noise" is the same on every run
```
- **In simple words:** Load the tools. The seed makes the "random" noise identical every time, so your output matches this walkthrough.
- **Exactly:** `sys.path.insert` lets Python import `docimg.py` from the module folder (`parents[1]`). `np.random.default_rng(0)` creates a seeded generator; `HERE` is the folder that receives the sample PNGs. `from scipy import ndimage` is used for the local mean filter.
- **Why it is here:** Without the seed the F1 numbers change from run to run and you cannot compare experiments.

### 2. Render a colour page and look at its array  (`main.py` lines 27–35)
```python
# ---- 1. Render a colour page: dark-blue ink on yellowish paper --------------------------------------
paper, ink_rgb = (245, 232, 170), (20, 30, 110)
img = Image.new("RGB", (520, 160), paper)
d = ImageDraw.Draw(img)
d.text((20, 20), "Invoice 4081  Total 1,250.00", fill=ink_rgb, font=get_font(30))
d.text((20, 80), "Paid by card on 12 March", fill=ink_rgb, font=get_font(30))
rgb = np.asarray(img)                                   # (H, W, 3) uint8: rows, columns, R/G/B
print("colour page: shape", rgb.shape, "dtype", rgb.dtype, "min/max", rgb.min(), rgb.max())
print("pixel (0,0) =", rgb[0, 0], "(paper);  darkest values per channel =", rgb.reshape(-1, 3).min(axis=0), "(ink)")
```
- **In simple words:** Draw two lines of text in dark blue on yellowish paper, then turn the picture into a grid of numbers.
- **Exactly:** `Image.new("RGB", (520, 160), paper)` makes a page 520 wide and 160 high (width first in Pillow). `np.asarray(img)` gives `rgb.shape == (160, 520, 3)`, `dtype == uint8`. Pixel (0, 0) is `[245 232 170]`, the paper; the darkest value per channel is `[20 30 110]`, the ink colour.
- **Why it is here:** Everything later is arithmetic on this array. The shape order (rows, columns, channels) explains most indexing bugs.

### 3. Convert to gray with the luma formula  (`main.py` lines 37–42)
```python
# ---- 2. Grayscale: weighted sum of channels (ITU-R BT.601 luma, what Pillow's convert('L') uses) -----
gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().astype(np.uint8)  # (H, W)
pil_gray = np.asarray(img.convert("L"))
mean_gray = rgb.mean(axis=2).round().astype(np.uint8)   # the naive alternative: plain average
print("gray formula max abs diff vs Pillow convert('L'):", int(np.abs(gray.astype(int) - pil_gray).max()))
print("naive mean(R,G,B) differs from luma by up to", int(np.abs(mean_gray.astype(int) - gray).max()), "levels")
```
- **In simple words:** Mix the red, green and blue values with weights that match how our eyes see brightness.
- **Exactly:** `rgb[..., 0]` is the red plane `(160, 520)`. The result is `float64`, rounded and cast to `uint8`. The check against Pillow's `convert("L")` prints a max absolute difference of `0`: it is the same formula. The naive mean differs by up to 17 levels.
- **Why it is here:** Gray conversion is the first step of every classic OCR pipeline. A bad conversion shifts the ink/paper contrast before thresholding sees it.

### 4. Make a dim page and a shadowed page  (`main.py` lines 44–52)
```python
# ---- 3. Make two hard versions: a uniformly DIM scan, and a page with a SHADOW gradient; both noisy ----
H, W = gray.shape
dim = add_gaussian((gray * 0.5).astype(np.uint8), 8, rng)                       # whole page at 50% brightness
shadow = np.linspace(1.0, 0.45, W)[None, :]             # left bright, right dark: (1, W) broadcasts to (H, W)
shaded = add_gaussian((gray * shadow).astype(np.uint8), 8, rng)
truth_ink = gray < 120                                  # ground truth: ink pixels of the clean page
save_png(gray, HERE / "sample_gray.png")
save_png(dim, HERE / "sample_dim_noisy.png")
save_png(shaded, HERE / "sample_shadow_noisy.png")
```
- **In simple words:** Ruin the page on purpose in two realistic ways: scan it too dark, and light it unevenly. Add camera noise to both.
- **Exactly:** `gray * 0.5` halves every level. `np.linspace(1.0, 0.45, W)[None, :]` has shape `(1, 520)`; broadcasting multiplies each column by its own factor, so the right edge is 45% as bright as the left. `add_gaussian(..., 8, rng)` adds noise with standard deviation 8 and clips to 0-255. `truth_ink = gray < 120` is the ground truth taken from the clean page.
- **Why it is here:** You can only measure a binariser if you know the right answer, and synthetic pages give you that for free.

### 5. Build the histogram  (`main.py` lines 54–58)
```python
# ---- 4. Histogram: how many pixels have each of the 256 gray levels -----------------------------------
hist = np.bincount(dim.ravel(), minlength=256)          # (256,) counts
print("histogram total =", hist.sum(), "= H*W =", H * W)
lo, hi = hist[:90].argmax(), 90 + hist[90:].argmax()
print("dim page: dark peak near level", lo, "(ink), bright peak near level", hi, "(paper)")
```
- **In simple words:** Count how many pixels there are at each brightness.
- **Exactly:** `np.bincount(dim.ravel(), minlength=256)` returns 256 counts and needs non-negative integers. The sum is 83,200 = 160 x 520. The dark peak is near level 17 and the paper peak near 113: after halving, the paper is at about 113, below the fixed threshold of 128.
- **Why it is here:** This is where you see why a fixed 128 fails: the whole paper peak sits underneath it.

### 6. Otsu's method in 8 lines  (`main.py` lines 61–71)
```python
# ---- 5. Otsu's method from scratch ---------------------------------------------------------------------
def otsu(gray_img):
    """Choose threshold t maximising between-class variance  sigma_b^2(t) = w0*w1*(mu0 - mu1)^2."""
    h = np.bincount(gray_img.ravel(), minlength=256).astype(np.float64)
    p = h / h.sum()                                      # probability of each level
    w0 = np.cumsum(p)                                    # weight of class "dark" (levels <= t)
    m = np.cumsum(p * np.arange(256))                    # cumulative mean up to t
    mu_total = m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):  # w0 = 0 or 1 at the ends -> 0/0
        sigma_b = (mu_total * w0 - m) ** 2 / (w0 * (1 - w0))
    return int(np.nanargmax(sigma_b))
```
- **In simple words:** Try every possible cut point between "dark" and "light" and keep the one where the two groups are most different from each other.
- **Exactly:** `p` = probability of each level. `w0 = cumsum(p)` is the weight of the dark class for each candidate `t`. `m = cumsum(p * arange(256))` is its cumulative mean, `mu_total = m[-1]`. The expression `(mu_total*w0 - m)**2 / (w0*(1-w0))` equals `w0*w1*(mu0-mu1)^2`. `np.errstate` silences the 0/0 at the ends and `np.nanargmax` skips the resulting NaNs. On the dim page it returns 69, between the peaks at 17 and 113.
- **Why it is here:** It is the standard automatic threshold and a favourite interview "implement it" question. All 256 candidates are scored with vector operations, no Python loop.

### 7. An F1 score for ink pixels  (`main.py` lines 74–77)
```python
def f1(pred_ink, true_ink):
    tp = (pred_ink & true_ink).sum()
    prec, rec = tp / max(pred_ink.sum(), 1), tp / max(true_ink.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec
```
- **In simple words:** Score a binarisation against the truth: how much of the real ink did you find, and how much paper did you wrongly call ink?
- **Exactly:** `tp` counts pixels that are ink in both. Precision = tp / predicted ink, recall = tp / true ink, F1 = harmonic mean.
- **Why it is here:** One number per method makes the comparison fair. Recall alone would reward calling everything ink.

### 8. Compare fixed, Otsu and adaptive on both pages  (`main.py` lines 80–92)
```python
# ---- 6. Compare on both pages: fixed 128, Otsu (global), local mean (adaptive) -----------------------
for page_name, page in [("DIM page", dim), ("SHADOW page", shaded)]:
    t_otsu = otsu(page)
    local_mean = ndimage.uniform_filter(page.astype(np.float32), size=31)        # mean of a 31x31 window around each pixel
    methods = [("fixed t=128", page < 128), (f"Otsu t={t_otsu}", page <= t_otsu),
               ("local mean (31x31, -25)", page < local_mean - 25)]               # ink = clearly darker than its neighbourhood
    print(page_name)
    for name, b in methods:
        f, p, r = f1(b, truth_ink)
        print(f"  {name:<26} F1 {f:.3f}  precision {p:.3f}  recall {r:.3f}")
    tag = page_name.split()[0].lower()
    save_png((~methods[1][1] * 255).astype(np.uint8), HERE / f"sample_{tag}_otsu.png")
    save_png((~methods[2][1] * 255).astype(np.uint8), HERE / f"sample_{tag}_adaptive.png")
```
- **In simple words:** Run the three methods on both bad pages and print how well each did.
- **Exactly:** `ndimage.uniform_filter(page, size=31)` gives the mean of the 31x31 window around every pixel. `page < local_mean - 25` means "at least 25 levels darker than the neighbourhood". `~methods[1][1] * 255` turns a boolean ink mask into a black-on-white PNG. See the execution trace for the numbers.
- **Why it is here:** It shows each method's failure mode with numbers instead of opinions.

### 9. Otsu on the clean page  (`main.py` lines 94–95)
```python
# ---- 7. Otsu on the CLEAN page: the easy case (a clearly two-peaked histogram) ------------------------
print("Otsu threshold on the clean gray page:", otsu(gray))
```
- **In simple words:** The easy case, for comparison.
- **Exactly:** The clean gray page has threshold 140. Anything between the two peaks works there.
- **Why it is here:** It shows that Otsu is not magic: it needs a histogram with two peaks.

### 10. TRY IT and BREAK IT notes  (`main.py` lines 97–98)
```python
# TRY IT:   change the shadow end value 0.45 to 0.8 (mild shadow) and watch fixed t=128 catch up with Otsu.
# BREAK IT: call otsu(gray / 255.0) (a float image) and read the np.bincount TypeError; bincount needs non-negative ints.
```
- **In simple words:** Two things to try yourself.
- **Exactly:** They are comments only and do not run.
- **Why it is here:** Changing one thing and predicting the result is how you learn.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `rgb[..., 0]` | `...` means "all leading axes"; picks the red plane | shape `(160, 520)` |
| `a[None, :]` | add a new axis of length 1 | `(520,)` -> `(1, 520)`, broadcasts over rows |
| `x.astype(np.uint8)` | cast; float values are truncated, not rounded | `3.9 -> 3` |
| `x.ravel()` | flatten to 1-D | `(160,520) -> (83200,)` |
| `~mask` | boolean NOT | ink -> paper |
| `with np.errstate(...)` | silence floating point warnings in the block | 0/0 in Otsu |
| `f"{name:<26}"` | left-align in 26 columns | table output |

## Libraries & functions
| Name | Library | What it does | Important args | Returns |
|---|---|---|---|---|
| `Image.new(mode, size, color)` | Pillow | blank image | size is `(width, height)` | `Image` |
| `ImageDraw.Draw(img).text(xy, s, fill, font)` | Pillow | draw text | `xy` is the top-left `(x, y)` | None |
| `ImageFont.load_default(size=30)` | Pillow | bundled scalable font, works offline | `size` in pixels | font |
| `np.asarray(img)` | numpy | view of the pixels (read-only for Pillow images) | | `ndarray` |
| `img.convert("L")` | Pillow | luma gray | | `Image` |
| `np.bincount` | numpy | histogram of non-negative ints | `minlength` | `int64[256]` |
| `ndimage.uniform_filter` | scipy | box mean over a window | `size` | float array |
| `np.nanargmax` | numpy | index of the max, ignoring NaN | | int |

## Execution trace (real run)
```
colour page: shape (160, 520, 3) dtype uint8 min/max 20 245
pixel (0,0) = [245 232 170] (paper);  darkest values per channel = [ 20  30 110] (ink)
gray formula max abs diff vs Pillow convert('L'): 0
naive mean(R,G,B) differs from luma by up to 17 levels
histogram total = 83200 = H*W = 83200
dim page: dark peak near level 17 (ink), bright peak near level 113 (paper)
DIM page
  fixed t=128                F1 0.098  precision 0.051  recall 1.000
  Otsu t=69                  F1 0.976  precision 0.954  recall 0.999
  local mean (31x31, -25)    F1 0.968  precision 0.938  recall 0.999
SHADOW page
  fixed t=128                F1 0.323  precision 0.193  recall 1.000
  Otsu t=154                 F1 0.194  precision 0.107  recall 1.000
  local mean (31x31, -25)    F1 0.949  precision 0.903  recall 1.000
Otsu threshold on the clean gray page: 140
```
- Shape `(160, 520, 3)` is rows x columns x channels. Pixel (0,0) is the paper colour.
- Diff `0` against Pillow proves the formula is the one Pillow uses. The naive mean is off by up to 17 levels.
- On the **DIM** page the paper peak (113) is under 128, so the fixed threshold calls nearly everything ink: precision 0.051, F1 0.098. Otsu finds t=69 and reaches F1 0.976.
- On the **SHADOW** page the right side is too dark for any single threshold. Fixed gives F1 0.323, Otsu 0.194 (it can even be worse than the fixed rule), while the local-mean threshold gets 0.949. Global thresholds fail when the lighting varies across the page.
- Clean page: Otsu 140.

## Experiments (change ONE thing)
1. Shadow end `0.45` -> `0.8` (mild shadow): fixed F1 0.976, Otsu 0.978, local mean 0.943. With mild shadows the simple global methods win; adaptive costs a bit of precision.
2. Noise sigma 8 -> 40 on the dim page: the paper and ink peaks smear into each other and all three methods collapse (fixed 0.139, Otsu 0.189, local mean 0.270). Thresholding cannot fix heavy noise; denoise first (step 02).
3. Replace the Otsu line in section 7 with `otsu(gray / 255.0)`: see the debugging section.

## Debugging practice
BREAK IT: pass a float image to `otsu`:
```
  File ".../main.py", line 64, in otsu
    h = np.bincount(gray_img.ravel(), minlength=256).astype(np.float64)
TypeError: Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'
```
Read it bottom-up: `np.bincount` only accepts integer arrays, and `gray / 255.0` produced floats in 0-1. Convert with `(x * 255).astype(np.uint8)` before histogramming, and keep one convention for image dtypes (uint8 0-255) across a pipeline.

## Rebuild checklist
- [ ] load an image to `uint8[H, W, 3]` and say what each axis is
- [ ] gray = 0.299 R + 0.587 G + 0.114 B
- [ ] histogram with `np.bincount(..., minlength=256)`
- [ ] Otsu: probabilities, cumulative weight and mean, `argmax` of between-class variance
- [ ] adaptive threshold with a local mean and an offset
- [ ] score a binarisation with F1 against a known mask

## The new problem (cliffhanger)
Binarisation assumes the picture is clean. Real scans also have speckles, blur, broken strokes, touching letters and a page that is rotated by a few degrees. A skewed page breaks line segmentation completely. Next: [step 02, preprocessing](../02_preprocessing/WALKTHROUGH.md).

> **Say it aloud (interview box):** An image is a uint8 array of rows, columns and channels. I convert to gray with the luma weights, then binarise. A fixed threshold fails when exposure changes, so I use Otsu, which picks the threshold that maximises between-class variance from the histogram. When lighting varies across the page, one global threshold cannot work, so I use an adaptive local threshold. With modern VLM OCR I usually skip binarisation, but for Tesseract-style engines it still matters.
