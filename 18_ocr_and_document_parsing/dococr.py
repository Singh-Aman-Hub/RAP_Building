"""
Finished helpers for module 18, assembled from what steps 01-04 teach (like 14_rag_and_retrieval/common.py):
Otsu binarisation, projection-profile segmentation, and a nearest-template character recogniser.
Needs: numpy, scipy, Pillow.   It recognises the bundled Pillow font at known sizes: it is a teaching OCR, not a product.
"""
import string

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from docimg import get_font

CHARS = [c for c in string.ascii_letters + string.digits + ".,:;!?-$%/()'&"]


def otsu_threshold(gray):
    """Pick the threshold t that maximises between-class variance of the 256-bin histogram."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)                       # P(class 0) for each candidate t
    mu = np.cumsum(p * np.arange(256))         # cumulative mean
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.nanargmax(sigma_b))


def binarise(gray):
    """Boolean array, True = ink (dark). Pixels <= Otsu threshold are ink."""
    return gray <= otsu_threshold(gray)


def runs(mask1d, min_gap=0):
    """Start/end (end exclusive) of runs of True in a 1-D bool array. Gaps shorter than min_gap are bridged."""
    idx = np.flatnonzero(mask1d)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev - 1 > min_gap:
            out.append((int(start), int(prev) + 1))
            start = i
        prev = i
    out.append((int(start), int(prev) + 1))
    return out


def segment_lines(ink):
    """Row projection profile -> list of (y0, y1) text lines."""
    return runs(ink.sum(axis=1) > 0)


def segment_chars(line_ink):
    """Connected components of a line, merging parts stacked on top of each other (the dot of i, j, : ;)."""
    lab, n = ndimage.label(line_ink)                       # 4-connected labelling (default structure)
    boxes = []
    for sl in ndimage.find_objects(lab):
        boxes.append([sl[1].start, sl[1].stop, sl[0].start, sl[0].stop])  # x0, x1, y0, y1
    boxes.sort()
    merged = []
    for b in boxes:
        if merged:
            a = merged[-1]
            overlap = min(a[1], b[1]) - max(a[0], b[0])
            if overlap > 0.6 * min(a[1] - a[0], b[1] - b[0]):  # x-ranges (almost) coincide: same glyph
                merged[-1] = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
                continue
        merged.append(b)
    return merged


class TemplateOCR:
    """One clean template per (font size, character), aligned on the baseline; nearest neighbour (L2)."""

    def __init__(self, sizes=(16, 21, 26, 34), chars=CHARS):
        self.sizes, self.chars = list(sizes), list(chars)
        self.tpl = {}                                      # size -> (labels, list of bool arrays)
        self.frame = {}                                    # size -> (ascent, descent)
        for s in self.sizes:
            font = get_font(s)
            asc, desc = int(s * 1.1), int(s * 0.35)
            self.frame[s] = (asc, desc)
            labels, arrs = [], []
            for c in self.chars:
                im = Image.new("L", (3 * s, asc + desc), 255)
                ImageDraw.Draw(im).text((s, asc), c, fill=0, font=font, anchor="ls")  # anchor ls: baseline at y=asc
                a = np.asarray(im) <= 127
                cols = np.flatnonzero(a.any(axis=0))
                labels.append(c)
                arrs.append(a[:, cols[0]:cols[-1] + 1])
            self.tpl[s] = (labels, arrs)

    def _classify(self, comp, size):
        labels, arrs = self.tpl[size]
        def dist(t, shifts):
            w = max(comp.shape[1], t.shape[1]) + 2          # 1 px of slack on each side for horizontal shifts
            b = np.zeros((comp.shape[0], w)); b[:, 1:1 + t.shape[1]] = t
            best = np.inf
            for dy, dx in shifts:
                a = np.zeros((comp.shape[0], w)); a[:, 1 + dx:1 + dx + comp.shape[1]] = np.roll(comp, dy, axis=0)
                best = min(best, ((a - b) ** 2).sum() + 0.5 * (abs(dy) + abs(dx)))   # small penalty so exact alignment wins ties
            return best

        d = np.array([dist(t, [(0, 0)]) for t in arrs])      # pass 1: exact alignment against every template
        for i in np.argsort(d)[:8]:                          # pass 2: tolerate +-1 px shifts, only for the 8 best candidates
            d[i] = dist(arrs[i], [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        order = np.argsort(d)
        best = order[0]
        second = next(j for j in order[1:] if labels[j] != labels[best])
        return labels[best], d[best], d[second]

    def read_line(self, line_ink, space_frac=0.2):
        """Return (text, mean confidence, size, char boxes) for the best-fitting font size."""
        boxes = segment_chars(line_ink)
        if not boxes:
            return "", 0.0, None, []
        base = int(np.median([b[3] for b in boxes]))      # baseline = where most glyphs end
        best = None
        for s in self.sizes:
            asc, desc = self.frame[s]
            text, tot, confs, prev = "", 0.0, [], None
            for x0, x1, y0, y1 in boxes:
                comp = np.zeros((asc + desc, x1 - x0), bool)
                sub = line_ink[:, x0:x1]
                for r in range(sub.shape[0]):             # paste rows at their offset from the baseline
                    rr = r - base + asc
                    if 0 <= rr < asc + desc:
                        comp[rr] = sub[r]
                lab, d1, d2 = self._classify(comp, s)
                if prev is not None and x0 - prev > space_frac * s:
                    text += " "
                text += lab
                tot += d1
                confs.append(max(0.0, 1 - d1 / max(d2, 1e-9)))
                prev = x1
            if best is None or tot < best[0]:
                best = (tot, text, float(np.mean(confs)), s)
        return best[1], best[2], best[3], boxes

    def read_region(self, ink):
        """Segment a text region into lines and read each. Returns list of dicts."""
        out = []
        for y0, y1 in segment_lines(ink):
            text, conf, s, boxes = self.read_line(ink[y0:y1])
            out.append(dict(y0=y0, y1=y1, text=text, conf=conf, size=s))
        return out
