"""
GOAL:  Open the black box called "PDF". Build a born-digital PDF and a scanned PDF by hand (bytes, zlib, Pillow), parse them with
       plain Python (regex + zlib), detect per page whether a text layer exists, and ROUTE: native text extraction vs OCR (module 18's
       template OCR). This is what PyMuPDF / pypdf do internally, minus fonts, encryption and the thousand edge cases.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/08_pdf_internals_and_routing/main.py
EXPECTED OUTPUT (shortened):
    first bytes of born_digital.pdf ...   (header, objects, a content stream with BT /F1 16 Tf ... Tj ET)
    page 1: native (text layer, 3 lines) ...   page 2: scanned -> OCR ...   page 3: searchable_scan (invisible text over an image)
    routed text per page, CER of OCR page vs truth
NEEDS: numpy, scipy, Pillow (no pymupdf / pypdf).  Writes born_digital.pdf, scanned.pdf, mixed.pdf here.  ~2 s on CPU.
"""
import io
import re
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docimg import render_lines                      # noqa: E402
from dococr import binarise, TemplateOCR             # noqa: E402

HERE = Path(__file__).parent
PX_TO_PT = 0.75                                      # a 96-dpi pixel is 0.75 PDF points (1 point = 1/72 inch)


# ---- 1. A minimal PDF WRITER: objects, a cross-reference table, a trailer --------------------------------------------------
def esc(s):
    """Escape a text string for a PDF literal string (...): backslash and parentheses."""
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Pdf:
    """Numbered objects 1..N. Object 1 = Catalog, 2 = Pages, 3 = the one font (Helvetica, a built-in 'standard 14' font)."""

    def __init__(self):
        self.objs = [b"", b"", b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
        self.kids = []

    def add(self, body):                             # returns the new object number
        self.objs.append(body)
        return len(self.objs)

    def stream(self, data, extra="", compress=False):
        if compress:
            data, extra = zlib.compress(data), extra + " /Filter /FlateDecode"
        return self.add(f"<< /Length {len(data)} {extra} >>\nstream\n".encode() + data + b"\nendstream")

    def add_page(self, w, h, text=(), image=None, invisible=False, compress=False):
        """text = [(x, y_from_top, size, string)] in points; image = gray uint8 array drawn over the whole page."""
        ops, res_x = [], ""
        if image is not None:
            im = self.stream(image.tobytes(), f"/Type /XObject /Subtype /Image /Width {image.shape[1]} /Height {image.shape[0]} "
                                              f"/ColorSpace /DeviceGray /BitsPerComponent 8", compress=True)
            ops.append(f"q {w} 0 0 {h} 0 0 cm /Im0 Do Q")          # cm = scale the 1x1 unit square to the page, Do = paint the image
            res_x = f"/XObject << /Im0 {im} 0 R >>"
        if invisible:
            ops.append("3 Tr")                                       # text render mode 3 = invisible (the OCR'd layer of a searchable scan)
        for x, y, size, s in text:
            ops.append(f"BT /F1 {size} Tf {x} {h - y} Td ({esc(s)}) Tj ET")   # PDF y axis points UP: flip from top-left coordinates
        content = self.stream("\n".join(ops).encode("latin-1"), compress=compress)
        page = self.add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] /Contents {content} 0 R "
                        f"/Resources << /Font << /F1 3 0 R >> {res_x} >> >>".encode())
        self.kids.append(page)

    def tobytes(self):
        self.objs[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
        self.objs[1] = f"<< /Type /Pages /Count {len(self.kids)} /Kids [{' '.join(f'{k} 0 R' for k in self.kids)}] >>".encode()
        out, offsets = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), []
        for i, body in enumerate(self.objs, 1):
            offsets.append(len(out))                                 # byte offset where "N 0 obj" starts: the xref table stores these
            out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(self.objs) + 1}\n0000000000 65535 f \n".encode()
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += f"trailer\n<< /Size {len(self.objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        return bytes(out)


# ---- 2. A minimal PDF READER: objects, page tree, content-stream operators ---------------------------------------------------
def read_objects(data):
    """{number: (dict_text, stream_bytes or None)}. Uses /Length to slice binary streams, so bytes that look like 'endobj' are safe."""
    objs, pos = {}, 0
    head = re.compile(rb"(\d+) 0 obj")
    while (m := head.search(data, pos)):
        num, i = int(m[1]), m.end()
        s_at, e_at = data.find(b"stream", i), data.find(b"endobj", i)
        if s_at != -1 and (e_at == -1 or s_at < e_at):               # this object carries a stream
            d = data[i:s_at].decode("latin-1")
            start = s_at + len(b"stream") + (2 if data[s_at + 6:s_at + 8] == b"\r\n" else 1)
            length = int(re.search(r"/Length (\d+)", d)[1])
            stream = data[start:start + length]
            if "/FlateDecode" in d:
                stream = zlib.decompress(stream)                     # the "uncompress" step every PDF library performs
            objs[num] = (d, stream)
            pos = start + length
        else:
            objs[num] = (data[i:e_at].decode("latin-1"), None)
            pos = e_at + 6
    return objs


def page_list(objs, data):
    """Trailer /Root -> Catalog -> Pages -> Kids (recursively): the page object numbers in order."""
    roots = re.findall(rb"/Root (\d+) 0 R", data)              # the LAST trailer wins (incremental updates append new ones)
    root = int(roots[-1]) if roots else next(n for n, (d, _) in objs.items() if "/Type /Catalog" in d)   # damaged file: scan, like real "repair" modes
    pages_ref = int(re.search(r"/Pages (\d+) 0 R", objs[root][0])[1])

    def walk(n):
        d = objs[n][0]
        if "/Kids" in d:
            return [p for k in re.findall(r"(\d+) 0 R", re.search(r"/Kids \[(.*?)\]", d, re.S)[1]) for p in walk(int(k))]
        return [n]
    return walk(pages_ref)


TOKEN = re.compile(rb"\((?:\\.|[^\\)])*\)|\[.*?\]|/\w+|[-+]?\d*\.?\d+|[A-Za-z'\"*]+", re.S)


def unescape(b):
    return re.sub(rb"\\(.)", rb"\1", b[1:-1]).decode("latin-1")


def run_content(stream):
    """Interpret just enough operators: text (BT Tf Td Tm Tr Tj TJ ET) and images (q cm Do Q). Returns (spans, images)."""
    spans, images, stack, st = [], [], [], dict(x=0.0, y=0.0, size=12.0, mode=0, ctm=[1, 0, 0, 1, 0, 0])
    operands = []
    for tok in TOKEN.findall(stream):
        if re.fullmatch(rb"[A-Za-z'\"*]+", tok) and not tok.startswith(b"/"):
            op = tok.decode()
            if op == "q":
                stack.append(dict(st, ctm=list(st["ctm"])))
            elif op == "Q" and stack:
                st = stack.pop()
            elif op == "cm":
                st["ctm"] = [float(v) for v in operands[-6:]]        # (simplification: we assume one cm per q block, no concatenation)
            elif op == "Do":
                images.append(dict(name=operands[-1].decode().lstrip("/"), w=st["ctm"][0], h=st["ctm"][3]))
            elif op == "BT":
                st["x"], st["y"] = 0.0, 0.0
            elif op == "Tf":
                st["size"] = float(operands[-1])
            elif op == "Tr":
                st["mode"] = int(float(operands[-1]))
            elif op == "Td":
                st["x"] += float(operands[-2]); st["y"] += float(operands[-1])
            elif op == "Tm":
                st["x"], st["y"] = float(operands[-2]), float(operands[-1])      # e, f of the text matrix = position
            elif op in ("Tj", "TJ"):
                a = operands[-1]
                if op == "Tj":
                    text = unescape(a)
                else:                                                # TJ array: strings with kerning numbers in between
                    text = ""
                    for t in re.findall(rb"\((?:\\.|[^\\)])*\)|[-+]?\d*\.?\d+", a):
                        text += unescape(t) if t.startswith(b"(") else (" " if float(t) < -200 else "")
                spans.append(dict(x=st["x"], y=st["y"], size=st["size"], text=text, mode=st["mode"]))
            operands = []
        else:
            operands.append(tok)
    return spans, images


def read_pdf(data):
    """Parse bytes -> list of page dicts: size, text spans, images (with decoded pixel data when we can)."""
    objs = read_objects(data)
    pages = []
    for n in page_list(objs, data):
        d = objs[n][0]
        w, h = (float(v) for v in re.search(r"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", d).groups())
        contents = [int(v) for v in re.findall(r"(\d+) 0 R", re.search(r"/Contents\s*(\[.*?\]|\d+ 0 R)", d, re.S)[1])]
        spans, placed = [], []
        for c in contents:
            s, i = run_content(objs[c][1])
            spans += s
            placed += i
        xobj = dict(re.findall(r"/(\w+)\s+(\d+) 0 R", (re.search(r"/XObject\s*<<(.*?)>>", d, re.S) or [0, ""])[1]))
        images = []
        for p in placed:
            idict, istream = objs[int(xobj[p["name"]])]
            images.append(dict(p, filter=(re.search(r"/Filter /(\w+)", idict) or [0, "none"])[1], dict=idict, data=istream))
        pages.append(dict(obj=n, width=w, height=h, spans=spans, images=images))
    return pages


# ---- 3. Detect and ROUTE ----------------------------------------------------------------------------------------------------
OCR = TemplateOCR(sizes=(21,))


def classify(page, min_chars=20, min_cover=0.5):
    visible = sum(len(s["text"]) for s in page["spans"] if s["mode"] != 3)
    hidden = sum(len(s["text"]) for s in page["spans"] if s["mode"] == 3)
    cover = max([im["w"] * im["h"] for im in page["images"]], default=0) / (page["width"] * page["height"])
    if visible >= min_chars:
        return "native", visible, hidden, cover
    if hidden >= min_chars and cover >= min_cover:
        return "searchable_scan", visible, hidden, cover       # image + an invisible OCR layer someone else produced
    if cover >= min_cover:
        return "scanned", visible, hidden, cover
    return "empty_or_vector", visible, hidden, cover


def native_text(page):
    """Group spans into lines by baseline y (PDF y grows upward, so sort by -y), then left to right."""
    rows = {}
    for s in page["spans"]:
        rows.setdefault(round(s["y"]), []).append(s)
    return [" ".join(s["text"] for s in sorted(r, key=lambda s: s["x"])) for _, r in sorted(rows.items(), reverse=True)]


def decode_image(im):
    """Return a gray uint8 array from an image XObject. Flate was already undone by read_objects; DCT is a JPEG file."""
    if im["filter"] == "DCTDecode":
        return np.asarray(Image.open(io.BytesIO(im["data"])).convert("L"))
    w, h = (int(v) for v in re.search(r"/Width (\d+) /Height (\d+)", im["dict"]).groups())
    return np.frombuffer(im["data"], np.uint8).reshape(h, w)


def route(page):
    kind, vis, hid, cover = classify(page)
    if kind in ("native", "searchable_scan"):
        return kind, "text layer", native_text(page)
    if kind == "scanned":
        gray = decode_image(page["images"][0])
        return kind, "OCR", [l["text"] for l in OCR.read_region(binarise(gray))]
    return kind, "nothing to read", []


def cer(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(ref)


# ---- 4. Build the three demo files ------------------------------------------------------------------------------------------------
LINES_NATIVE = ["Chai Kitchen Quarterly Report", "Refunds are issued within nine days of delivery.", "Orders above 499 rupees ship free."]
LINES_SCAN = ["Invoice 4821 for Chai Kitchen", "Total due: 1250 rupees", "Paid by card on time."]
W, H = 600, 480                                              # points (an 800 x 640 px page at 96 dpi)

scan_gray, _ = render_lines(LINES_SCAN, size=21, pad=24, width=int(W / PX_TO_PT))
scan_gray = np.pad(scan_gray, ((0, int(H / PX_TO_PT) - scan_gray.shape[0]), (0, 0)), constant_values=255)   # white-fill to the page height

born = Pdf()                                                 # (a) born-digital: only text operators, nothing to OCR
born.add_page(W, H, [(72, 80 + 28 * i, 16, t) for i, t in enumerate(LINES_NATIVE)])
(HERE / "born_digital.pdf").write_bytes(born.tobytes())

Image.fromarray(scan_gray).save(HERE / "scanned.pdf", resolution=96)      # (b) "scanner output": Pillow writes an image-only PDF

mixed = Pdf()                                                # (c) a realistic bundle: page 1 native, page 2 scan, page 3 searchable scan
mixed.add_page(W, H, [(72, 80 + 28 * i, 16, t) for i, t in enumerate(LINES_NATIVE)], compress=True)
mixed.add_page(W, H, image=scan_gray)
mixed.add_page(W, H, [(18, 18 + 23.25 * i, 15.75, t) for i, t in enumerate(LINES_SCAN)], image=scan_gray, invisible=True)
(HERE / "mixed.pdf").write_bytes(mixed.tobytes())

# ---- 5. Look at the bytes, then parse and route ---------------------------------------------------------------------------------------
raw = (HERE / "born_digital.pdf").read_bytes()
print(f"born_digital.pdf = {len(raw)} bytes. Start of the file:")
print("   " + raw[:raw.index(b"2 0 obj")].decode("latin-1").replace("\n", "\n   "))
print("   ...")
print("   " + raw[raw.index(b"4 0 obj"):raw.index(b"\nendstream")].decode("latin-1").replace("\n", "\n   "))
print("   ...  ends with:", raw[-48:].decode("latin-1").replace("\n", " | "))

pil = read_pdf((HERE / "scanned.pdf").read_bytes())[0]
print(f"\nPillow's scanned.pdf: {len(pil['images'])} image, {len(pil['spans'])} text spans, image filter /{pil['images'][0]['filter']}, "
      f"page {pil['width']:.0f}x{pil['height']:.0f} pt  -> {classify(pil)[0]}")

_, how, lines = route(pil)
print(f"   routed via {how} (the JPEG is decoded by Pillow, then binarised and read): {lines}")

print("\nmixed.pdf, page by page:")
pages = read_pdf((HERE / "mixed.pdf").read_bytes())
for i, p in enumerate(pages, 1):
    kind, how, lines = route(p)
    kind, vis, hid, cover = classify(p)
    print(f"   page {i}: {kind:<16} visible chars {vis:>3}, invisible chars {hid:>3}, image cover {cover:4.0%}  -> route: {how}")
    for l in lines:
        print(f"           {l!r}")

ocr_lines = route(pages[1])[2]
print(f"\nOCR page vs truth: CER {cer(' '.join(LINES_SCAN), ' '.join(ocr_lines)):.3f}   |   native page vs truth: CER "
      f"{cer(' '.join(LINES_NATIVE), ' '.join(route(pages[0])[2])):.3f}")
print("page-2 pixels as stored in the PDF:", decode_image(pages[1]["images"][0]).shape, "-> same array we drew; PDF stored it Flate-compressed")

# TRY IT:   add a 4th page with no text and no image (Pdf().add_page(W, H)) and see "empty_or_vector"; set min_cover to 1.1 and watch the scans
#           fall out of the OCR route; change the OCR sizes to (16,) and watch the scanned page degrade (CER about 0.85).
# BREAK IT: cut born_digital.pdf at "xref": our reader still works (it never reads the xref table: it scans for "N 0 obj" and finds the Catalog
#           itself, which is what real viewers' "repair" mode does). Then change "/Length 206" to "/Length 100": the reader slices the stream
#           short and silently returns only ONE of the three lines, the worst kind of failure.
