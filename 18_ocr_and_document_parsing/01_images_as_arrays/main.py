"""
GOAL:  Understand an image as a numpy array, convert colour to grayscale with the real formula, read a
       histogram, and binarise a badly lit, noisy page with a fixed threshold vs Otsu's method (from scratch).

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/01_images_as_arrays/main.py
EXPECTED OUTPUT (shortened):
    colour page: shape (160, 520, 3) dtype uint8 ...
    gray formula max abs diff vs Pillow convert('L'): 1
    DIM page: fixed t=128 F1 ...  Otsu ...  local mean ...
    SHADOW page: ...
NEEDS: numpy, scipy, Pillow.  Instant.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docimg import get_font, add_gaussian, save_png   # noqa: E402

HERE = Path(__file__).parent
rng = np.random.default_rng(0)                          # seed: the "noise" is the same on every run

# ---- 1. Render a colour page: dark-blue ink on yellowish paper --------------------------------------
paper, ink_rgb = (245, 232, 170), (20, 30, 110)
img = Image.new("RGB", (520, 160), paper)
d = ImageDraw.Draw(img)
d.text((20, 20), "Invoice 4081  Total 1,250.00", fill=ink_rgb, font=get_font(30))
d.text((20, 80), "Paid by card on 12 March", fill=ink_rgb, font=get_font(30))
rgb = np.asarray(img)                                   # (H, W, 3) uint8: rows, columns, R/G/B
print("colour page: shape", rgb.shape, "dtype", rgb.dtype, "min/max", rgb.min(), rgb.max())
print("pixel (0,0) =", rgb[0, 0], "(paper);  darkest values per channel =", rgb.reshape(-1, 3).min(axis=0), "(ink)")

# ---- 2. Grayscale: weighted sum of channels (ITU-R BT.601 luma, what Pillow's convert('L') uses) -----
gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().astype(np.uint8)  # (H, W)
pil_gray = np.asarray(img.convert("L"))
mean_gray = rgb.mean(axis=2).round().astype(np.uint8)   # the naive alternative: plain average
print("gray formula max abs diff vs Pillow convert('L'):", int(np.abs(gray.astype(int) - pil_gray).max()))
print("naive mean(R,G,B) differs from luma by up to", int(np.abs(mean_gray.astype(int) - gray).max()), "levels")

# ---- 3. Make two hard versions: a uniformly DIM scan, and a page with a SHADOW gradient; both noisy ----
H, W = gray.shape
dim = add_gaussian((gray * 0.5).astype(np.uint8), 8, rng)                       # whole page at 50% brightness
shadow = np.linspace(1.0, 0.45, W)[None, :]             # left bright, right dark: (1, W) broadcasts to (H, W)
shaded = add_gaussian((gray * shadow).astype(np.uint8), 8, rng)
truth_ink = gray < 120                                  # ground truth: ink pixels of the clean page
save_png(gray, HERE / "sample_gray.png")
save_png(dim, HERE / "sample_dim_noisy.png")
save_png(shaded, HERE / "sample_shadow_noisy.png")

# ---- 4. Histogram: how many pixels have each of the 256 gray levels -----------------------------------
hist = np.bincount(dim.ravel(), minlength=256)          # (256,) counts
print("histogram total =", hist.sum(), "= H*W =", H * W)
lo, hi = hist[:90].argmax(), 90 + hist[90:].argmax()
print("dim page: dark peak near level", lo, "(ink), bright peak near level", hi, "(paper)")


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


def f1(pred_ink, true_ink):
    tp = (pred_ink & true_ink).sum()
    prec, rec = tp / max(pred_ink.sum(), 1), tp / max(true_ink.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9), prec, rec


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

# ---- 7. Otsu on the CLEAN page: the easy case (a clearly two-peaked histogram) ------------------------
print("Otsu threshold on the clean gray page:", otsu(gray))

# TRY IT:   change the shadow end value 0.45 to 0.8 (mild shadow) and watch fixed t=128 catch up with Otsu.
# BREAK IT: call otsu(gray / 255.0) (a float image) and read the np.bincount TypeError; bincount needs non-negative ints.
