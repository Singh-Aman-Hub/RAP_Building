"""
GOAL:  Clean a degraded page before recognition: remove noise (median filter), repair strokes (morphology),
       find and fix page skew (projection-profile search), and see what low resolution does to letters.
       Median and morphology are implemented from scratch and checked against scipy.ndimage.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/02_preprocessing/main.py
EXPECTED OUTPUT (shortened):
    noise: ... F1 of ink after Otsu: raw ... box blur ... median ...
    median from scratch == scipy: True      erosion == scipy: True
    skew search: rotate by -4.00 deg ...  lines found: skewed ... -> deskewed ...
    dpi table: scale ... components ...
NEEDS: numpy, scipy, Pillow.  Under 5 seconds.
"""
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

# ---- 1. A clean page, its ink mask, and a noisy copy ------------------------------------------------------
clean, _ = render_lines(LINES, size=26, pad=30)             # (H, W) uint8, black text on white
clean_ink = clean < 128                                      # ground-truth ink mask
noisy = add_gaussian(add_salt_pepper(clean, 0.06, rng), 25, rng)   # 6% salt & pepper, then gaussian sigma 25
save_png(clean, HERE / "sample_clean.png")
save_png(noisy, HERE / "sample_noisy.png")
print("page", clean.shape, "ink pixels in truth:", int(clean_ink.sum()))


def ink_f1(gray):
    """Binarise with Otsu and score against the clean ink mask."""
    pred = gray <= otsu_threshold(gray)
    tp = (pred & clean_ink).sum()
    p, r = tp / max(pred.sum(), 1), tp / clean_ink.sum()
    return 2 * p * r / (p + r)


# ---- 2. Median filter from scratch: replace each pixel with the median of its 3x3 neighbourhood ----------
def median3(img):
    padded = np.pad(img, 1, mode="symmetric")                # copy edge pixels outwards so borders have neighbours
    windows = sliding_window_view(padded, (3, 3))            # (H, W, 3, 3): a view, no copy
    return np.median(windows, axis=(2, 3)).astype(np.uint8)  # median over each 3x3 -> (H, W)


med = median3(noisy)
box = ndimage.uniform_filter(noisy.astype(np.float32), 3).astype(np.uint8)   # 3x3 mean blur for comparison
print("noise: F1 of ink after Otsu:  raw %.3f   box blur %.3f   median %.3f"
      % (ink_f1(noisy), ink_f1(box), ink_f1(med)))
print("median from scratch == scipy.ndimage.median_filter:",
      bool(np.array_equal(med, ndimage.median_filter(noisy, size=3))))
save_png(med, HERE / "sample_median.png")


# ---- 3. Morphology from scratch on a binary image ------------------------------------------------------------
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

# ---- 4. Skew: rotate the page 4 degrees, then recover the angle from projection profiles ---------------------
skewed = np.asarray(Image.fromarray(clean).rotate(4.0, resample=Image.BICUBIC, expand=True, fillcolor=255))
save_png(skewed, HERE / "sample_skewed.png")


def rotate_ink(ink, angle):
    return np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).rotate(angle, resample=Image.BILINEAR)) > 127


def profile_score(ink, angle):
    """Variance of the row sums. Aligned text lines give sharp peaks and empty gaps: high variance."""
    return float(rotate_ink(ink, angle).sum(axis=1).astype(np.float64).var())


sk_ink = skewed < 128
angles = np.arange(-10, 10.01, 0.25)                          # brute-force search, 81 candidate angles
scores = np.array([profile_score(sk_ink, a) for a in angles])
best = float(angles[scores.argmax()])
print(f"skew search: rotate by {best:+.2f} deg to straighten (we skewed it by +4.00, so -4.00 is right); "
      f"profile variance at best {scores.max():.0f} vs at 0 deg {profile_score(sk_ink, 0):.0f}")
deskewed = rotate_ink(sk_ink, best)
save_png((~deskewed * 255).astype(np.uint8), HERE / "sample_deskewed.png")
n_lines = lambda ink: len(runs(ink.sum(axis=1) > 0))          # text lines = runs of non-empty rows
print("text lines found by row projection: skewed", n_lines(sk_ink), "-> deskewed", n_lines(deskewed),
      "(truth", len(LINES), ")")

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

# TRY IT:   change the salt & pepper amount to 0.2 and compare box blur with median; then try median3 with a 5x5 window.
# BREAK IT: in `erode`, drop the np.pad (use sliding_window_view(b, (k, k)) directly): the output shrinks to (H-2, W-2)
#           and the F1 line fails with 'ValueError: operands could not be broadcast together'.
