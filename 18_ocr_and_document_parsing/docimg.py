"""
Shared helpers for module 18: synthetic page rendering (Pillow only) and noise.
Everything is generated from code, so no dataset download is needed and every run is reproducible.

Fonts: ImageFont.load_default(size=N) ships INSIDE Pillow (a FreeType font), so it works offline on any
machine. Optional macOS fonts are used only where a step asks for extra fonts (step 04, 05).
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SYSTEM_FONTS = [                       # optional extras; missing files are skipped silently
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Verdana.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
]


def get_font(size, path=None):
    """Return a PIL font. path=None -> Pillow's bundled default font (always available)."""
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass                                          # fall back below
    try:
        return ImageFont.load_default(size=size)          # Pillow >= 10.1: scalable bundled font
    except TypeError:                                     # very old Pillow: fixed-size bitmap font
        return ImageFont.load_default()


def available_system_fonts():
    return [p for p in _SYSTEM_FONTS if Path(p).exists()]


def render_lines(lines, size=28, pad=20, width=None, line_gap=1.5, font=None):
    """Render text lines black-on-white. Returns (gray uint8 array (H, W), list of (x, y, text))."""
    font = font or get_font(size)
    step = int(size * line_gap)                           # vertical distance between baselines
    w = width or int(max(font.getlength(t) for t in lines)) + 2 * pad
    h = step * len(lines) + 2 * pad
    img = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(img)
    pos = []
    for i, t in enumerate(lines):
        xy = (pad, pad + i * step)
        d.text(xy, t, fill=0, font=font)
        pos.append((xy[0], xy[1], t))
    return np.asarray(img), pos


def to_uint8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def add_gaussian(img, sigma, rng):
    """Add N(0, sigma^2) noise to every pixel (sensor / scanner noise)."""
    return to_uint8(img.astype(np.float32) + rng.normal(0, sigma, img.shape))


def add_salt_pepper(img, amount, rng):
    """Flip a fraction `amount` of pixels to pure black or pure white (dust, dead pixels, bad fax)."""
    out = img.copy()
    r = rng.random(img.shape)
    out[r < amount / 2] = 0
    out[(r >= amount / 2) & (r < amount)] = 255
    return out


def save_png(arr, path):
    Image.fromarray(arr).save(path)


# --------------------------------------------------------------------------------------------------
# A synthetic "report" page used by steps 07 and 10: title, two text columns, a ruled table, a footer.
# --------------------------------------------------------------------------------------------------
PAGES = [
    dict(
        title="Chai Kitchen Quarterly Report",
        left_head="Refund policy",
        left=["Refunds are issued within nine days",
              "of delivery. The money is returned",
              "to the original payment method.",
              "Opened tea tins cannot be returned."],
        right=["Shipping inside India takes",
               "three to five working days. Orders",
               "above 499 rupees ship free. We",
               "do not ship to a PO box."],
        table=[["Product", "Price", "Stock"],
               ["Masala Chai", "250", "120"],
               ["Green Tea", "320", "75"],
               ["Tulsi Blend", "290", "0"]],
        footer="Page 1 of 2"),
    dict(
        title="Loyalty and Support Notes",
        left_head="Loyalty points",
        left=["Members earn one point for every",
              "ten rupees spent. One hundred",
              "points are worth ten rupees off",
              "a later order."],
        right=["Support is open from nine to six",
               "on weekdays. Replies to email",
               "arrive within one working day.",
               "Phone support is not offered."],
        table=[["Plan", "Points", "Fee"],
               ["Basic", "1", "0"],
               ["Silver", "2", "499"],
               ["Gold", "3", "999"]],
        footer="Page 2 of 2"),
]


def make_report_page(idx, noise=0, seed=0):
    """Render PAGES[idx] to an 800x640 grayscale page. Returns (img, truth) where truth lists the
    elements in the CORRECT reading order with their true boxes (x0, y0, x1, y1)."""
    spec = PAGES[idx]
    W, H = 800, 640
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    f_title, f_head, f_body, f_small = get_font(34), get_font(26), get_font(21), get_font(16)
    truth = []

    def put(xy, text, font, kind):
        d.text(xy, text, fill=0, font=font)
        b = d.textbbox(xy, text, font=font)
        truth.append(dict(kind=kind, text=text, bbox=list(b)))
        return b

    put((40, 24), spec["title"], f_title, "title")
    d.line([(40, 78), (W - 40, 78)], fill=0, width=1)          # a rule under the title (ink, not text)
    put((40, 100), spec["left_head"], f_head, "heading")
    for i, t in enumerate(spec["left"]):
        put((40, 146 + i * 30), t, f_body, "paragraph")
    for i, t in enumerate(spec["right"]):
        put((420, 146 + i * 30), t, f_body, "paragraph")
    # table with ruling lines: 4 rows x 3 columns
    tx, ty, cw, rh = 40, 310, [300, 200, 200], 44
    rows = spec["table"]
    x_edges = [tx]
    for c in cw:
        x_edges.append(x_edges[-1] + c)
    y_edges = [ty + r * rh for r in range(len(rows) + 1)]
    for y in y_edges:
        d.line([(x_edges[0], y), (x_edges[-1], y)], fill=0, width=2)
    for x in x_edges:
        d.line([(x, y_edges[0]), (x, y_edges[-1])], fill=0, width=2)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            d.text((x_edges[c] + 12, y_edges[r] + 10), cell, fill=0, font=f_body)
    truth.append(dict(kind="table", text=rows, bbox=[x_edges[0], y_edges[0], x_edges[-1], y_edges[-1]]))
    put((W // 2 - 40, H - 56), spec["footer"], f_small, "footer")
    arr = np.asarray(img)
    if noise:
        arr = add_gaussian(arr, noise, np.random.default_rng(seed))
    return arr, truth
