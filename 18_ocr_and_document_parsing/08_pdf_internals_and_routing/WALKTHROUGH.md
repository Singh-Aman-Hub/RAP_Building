# Step 08 — PDF internals and routing: text layer or OCR?

## The story so far
[Step 07](../07_reading_order_and_tables/WALKTHROUGH.md) turned a rendered page image into typed, ordered, boxed blocks. But nobody hands you a PNG: business documents arrive as PDF files. A PDF is not a picture and not a document in the Word sense; it is a program that paints a page.

## The pain point
Run OCR on every PDF and you throw away exact text that was sitting in the file for free (and pay seconds per page for an approximation). Skip OCR and a scanned PDF returns an empty string, silently, which a RAG pipeline indexes as "no content". Both mistakes are common, and both are invisible until someone asks a question the index cannot answer. Worse, a third kind exists: a **searchable scan**, an image with someone else's invisible OCR text on top. Its text layer exists but may be wrong.

## The spark
Open the file. A PDF is a set of numbered objects; a page object points to a **content stream** (drawing operators like `BT ... (Hello) Tj ET` for text, `Do` for an image). Decode the content streams, count the visible characters and measure how much of the page an image covers. Visible text means extract it; a full-page image with no text means rasterise-and-OCR; invisible text over an image means trust but verify. That is a per-page **router**, and it is 30 lines of Python. Libraries (PyMuPDF, pypdf, pdfminer) do the same decoding with far more coverage.

## Goal
Write a minimal PDF writer (objects, xref, trailer, Flate streams, an image XObject, invisible text) and a minimal PDF reader (objects by `/Length`, page tree, content-stream operators), build `born_digital.pdf`, `scanned.pdf` (Pillow) and a 3-page `mixed.pdf`, classify each page and route it: native extraction or OCR with `dococr`.

## The theory (read this before the interview)
**What a PDF file is.** A page description language (a descendant of PostScript): it says *where to paint which glyphs and images*, not "this is a paragraph" or "this is a table". Structure
(PDF/UA "tagged PDF") exists only if the producer added it. The file is:
```
%PDF-1.4                                    header
1 0 obj  << /Type /Catalog /Pages 2 0 R >>  endobj          object 1 = the Catalog (root)
2 0 obj  << /Type /Pages /Count 1 /Kids [5 0 R] >> endobj    the page tree
5 0 obj  << /Type /Page /MediaBox [0 0 600 480] /Contents 4 0 R /Resources << /Font ... /XObject ... >> >> endobj
4 0 obj  << /Length 206 >> stream  BT /F1 16 Tf 72 400 Td (Hello) Tj ET  endstream endobj   content stream
xref  0 6   0000000000 65535 f / 0000000015 00000 n / ...   byte offset of every object
trailer << /Size 6 /Root 1 0 R >>  startxref 576  %%EOF      where to start reading (from the END of the file)
```
Object types: numbers, names (`/Type`), strings `(...)`, arrays `[...]`, dictionaries `<< ... >>`, streams (a dictionary + raw bytes) and indirect references `N 0 R`. A reader goes
`trailer /Root -> Catalog /Pages -> /Kids -> /Page -> /Contents`. The **xref table** gives each object's byte offset so it can be read without scanning; if it is damaged, readers scan for
`N G obj` ("repair mode"). Modern files may pack objects into compressed *object streams* with a cross-reference *stream*, which is why a regex-based parser like ours is only a teaching tool.

**Coordinates.** Units are points (1/72 inch); the origin is the **bottom-left** and y grows **upwards**. Image pixels have the origin top-left, so every PDF-to-image conversion flips y: `y_top = page_height - y_pdf`.
Pixels to points: 96-dpi pixel x 0.75.

**Content stream operators (text).** `BT`/`ET` begin/end text; `/F1 16 Tf` select font and size; `x y Td` move; `(text) Tj` show a string; `[(A) -250 (V)] TJ` show with kerning adjustments (a large
negative number is often a word space); `Tr` render mode (0 fill = visible, **3 = invisible**, used by OCR software to put searchable text under a scan). Images: `q w 0 0 h x y cm /Im0 Do Q` scales the unit square to
w x h points and paints XObject `Im0`.

**Why extracted text can be garbage even when a text layer exists.** The stream holds *glyph codes*, not Unicode. Mapping code to character needs the font's encoding or a `/ToUnicode` CMap. Subset fonts with custom encodings and no ToUnicode (some
old exports, some Indic and Arabic PDFs) yield `(cid:123)` or random symbols; ligatures (`fi`) arrive as one glyph; text drawn as vector outlines has **no** text layer at all; and text order is drawing order, so columns, footnotes
and headers may interleave. Always look for garbage: a high fraction of `(cid:`, U+FFFD, control characters, or non-dictionary words means "treat this page as a scan".

**Image PDFs.** A scan is one image XObject per page, compressed with a **filter**: `FlateDecode` (zlib, lossless), `DCTDecode` (JPEG, lossy: Pillow's `save("x.pdf")` uses it for greyscale images, as the trace shows), `CCITTFaxDecode`/`JBIG2Decode` (bilevel fax-style, tiny files), `JPXDecode` (JPEG 2000). For OCR it
is better to **extract the embedded image** (native resolution, no re-sampling) than to render the page; render (PyMuPDF `get_pixmap(dpi=200..300)`, poppler `pdftoppm -r 300`) only when the page mixes vectors, text and images.

**The four page types and the router.**
| Page type | What the file contains | Route |
|---|---|---|
| native (born-digital) | visible text operators | extract the text layer (+ boxes); no OCR |
| scanned | full-page image, no text | extract/render the image, then OCR (this module) |
| searchable scan | full-page image + invisible text (`Tr 3`) | use the layer, but sample-check it against your OCR: it was someone else's engine |
| empty / vector | neither | flag it: charts drawn as paths, forms with outlined text, blank pages |
Heuristic thresholds: at least ~20 visible characters, image area at least 50% of the page. Route **per page** (real bundles mix types), keep the route in the output metadata and log pages sent to OCR because they cost 100-1000x more.

**What PyMuPDF and pypdf do internally.** They implement exactly the loop in this step: parse the cross-reference data (or rebuild it), decrypt if needed, walk the page tree, decode streams through filters, interpret operators with a graphics state (including full matrix concatenation, clipping, fonts and encodings) and expose results:
`page.get_text("words")` (PyMuPDF: words with boxes), `page.extract_text()` (pypdf), `page.get_images()` and, for PyMuPDF, a renderer (`get_pixmap`) because it is built on MuPDF, a C rendering engine. pdfminer.six and pdfplumber are pure-Python alternatives with strong layout/table-from-lines helpers. [`pymupdf_reference.py`](pymupdf_reference.py) shows the calls and skips cleanly when the packages are missing (they are not installed here).
Licensing note: PyMuPDF/MuPDF is AGPL-3.0 (or commercial); pypdf is BSD-3: check this before shipping inside a product.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Born-digital PDF | native PDF, digital-native, text PDF, vector PDF | yes | Created by software (Word, LaTeX, a browser): has a real text layer |
| Scanned PDF | image PDF, image-only PDF, raster PDF | yes | One image per page, no text: OCR is mandatory |
| Searchable PDF | OCR'd PDF, PDF with text layer, sandwich PDF | near | Scan image plus invisible text from an OCR engine; quality = that engine's quality |
| Text layer | embedded text, text objects, selectable text | yes | The `BT ... Tj ET` operators with positions |
| Content stream | page contents, drawing instructions | yes | The bytes of operators that paint one page; usually Flate-compressed |
| XObject (image) | image object, `Do` operand | near | A reusable object (image or form) painted by the `Do` operator |
| Filter | stream encoding, decode filter, `/FlateDecode` | yes | Compression applied to stream bytes: Flate, DCT (JPEG), CCITT, JBIG2, JPX |
| xref table | cross-reference table/stream, object index | near | Byte offsets of objects; can be rebuilt by scanning |
| ToUnicode CMap | glyph-to-Unicode map | yes | Without it, copied text can be `(cid:..)` garbage |
| Rasterise / render | page to pixmap, pdf2image, `pdftoppm` | near | Draw the page to pixels (needed to OCR vector or mixed pages) |

## Run it
```bash
# from this folder
python main.py
python pymupdf_reference.py      # optional: skips cleanly, PyMuPDF / pypdf are not installed here
# from the repo root
.venv/bin/python 18_ocr_and_document_parsing/08_pdf_internals_and_routing/main.py
```
About 2 seconds on a CPU laptop (timing varies by machine). Needs numpy, scipy, Pillow; no PDF library. Writes `born_digital.pdf`, `scanned.pdf`, `mixed.pdf` here (you can open them in any PDF viewer).

## Code, step by step

### 1. Imports and the two module helpers  (`main.py` lines 14–28)
```python
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
```
- **In simple words:** Standard library tools for bytes, regex and zlib; numpy and Pillow for pixels; `docimg.render_lines` to draw the scan; `dococr` for the OCR route.
- **Exactly:** `zlib` is Python's binding to the Flate compression every PDF uses; `io.BytesIO` wraps bytes as a file for Pillow. `PX_TO_PT = 0.75` converts 96-dpi pixels to PDF points.
- **Why it is here:** The point of the step is that nothing PDF-specific is imported.

### 2. esc: escaping text for a PDF string  (`main.py` lines 31–34)
```python
# ---- 1. A minimal PDF WRITER: objects, a cross-reference table, a trailer --------------------------------------------------
def esc(s):
    """Escape a text string for a PDF literal string (...): backslash and parentheses."""
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
```
- **In simple words:** In a PDF literal string `( ... )` backslash and parentheses must be escaped.
- **Exactly:** Three `str.replace` calls; the backslash is escaped first so the new backslashes are not doubled.
- **Why it is here:** Otherwise a text line containing `)` would end the string early and corrupt the content stream.

### 3. The writer: object list, add, stream  (`main.py` lines 37–51)
```python
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
```
- **In simple words:** `Pdf` keeps numbered objects (1 Catalog, 2 Pages, 3 the font). `stream` builds `<< /Length n ... >> stream ... endstream`, optionally Flate-compressed.
- **Exactly:** `zlib.compress(data)` returns a zlib-wrapped deflate stream, which is what `/FlateDecode` expects. `/Length` must be the **compressed** byte count. `add` returns the new object number (its index plus 1).
- **Why it is here:** Everything in a PDF is an indirect object; this is the smallest wrapper that lets us create them.

### 4. add_page: a page object, a content stream and optionally an image  (`main.py` lines 53–68)
```python
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
```
- **In simple words:** Turn `(x, y_from_top, size, text)` tuples into `BT ... Tj ET` operators, optionally paint a full-page image first and mark the text invisible with `3 Tr`.
- **Exactly:** `h - y` flips the y axis (PDF origin is bottom-left). The image is added as an XObject with width, height, `DeviceGray` and 8 bits per component, Flate-compressed; `q w 0 0 h 0 0 cm /Im0 Do Q` scales it to the page. The page dictionary references the content stream (`/Contents`), the font and the XObject (`/Resources`).
- **Why it is here:** This produces all three page kinds: text only, image only, image + invisible text.

### 5. tobytes: header, objects, xref table, trailer  (`main.py` lines 70–82)
```python
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
```
- **In simple words:** Fill in the Catalog and Pages objects, write every object recording its byte offset, then the xref table and the trailer pointing at the Catalog.
- **Exactly:** `offsets` is the byte position of each `N 0 obj`. The xref line format is 10-digit offset, 5-digit generation, `n` (in use); object 0 is the head of the free list. `startxref` stores the byte offset of the table: a reader jumps to the end of the file, reads that number and finds everything.
- **Why it is here:** Shows why a PDF can be read from the end and why editors can append changes without rewriting the file.

### 6. read_objects: find every object, slice streams by /Length, inflate Flate  (`main.py` lines 86–105)
```python
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
```
- **In simple words:** Scan for `N 0 obj`; if the object has a stream, read its dictionary text, take exactly `/Length` bytes after `stream` and, if `/FlateDecode`, decompress.
- **Exactly:** A walrus loop (`while (m := head.search(data, pos))`) advances through the bytes. Slicing by `/Length` (not by searching `endstream`) is essential: binary data may contain the characters `endobj`. Returns `{number: (dictionary text, stream bytes or None)}`. `zlib.decompress` reverses `zlib.compress`.
- **Why it is here:** This is the core of every PDF library: object parsing and filter decoding. It also never uses the xref table, which is what repair mode does.

### 7. page_list: walk Catalog, Pages and Kids  (`main.py` lines 108–119)
```python
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
```
- **In simple words:** Find the root from the trailer (or scan for the Catalog in a damaged file), follow `/Pages` and recurse through `/Kids` to get page objects in order.
- **Exactly:** `re.findall(rb"/Root ...")` takes the last match (incremental updates append trailers). The nested `walk` function returns `[n]` for a leaf page or the concatenated lists of its kids. Real files can nest `/Pages` several levels and inherit `/MediaBox` and `/Resources` from parents; we ignore inheritance.
- **Why it is here:** Page order comes from the tree, not from object numbers.

### 8. The tokenizer: strings, arrays, names, numbers, operators  (`main.py` lines 122–126)
```python
TOKEN = re.compile(rb"\((?:\\.|[^\\)])*\)|\[.*?\]|/\w+|[-+]?\d*\.?\d+|[A-Za-z'\"*]+", re.S)


def unescape(b):
    return re.sub(rb"\\(.)", rb"\1", b[1:-1]).decode("latin-1")
```
- **In simple words:** One regex that splits a content stream into tokens; `unescape` removes the backslash escapes of a string.
- **Exactly:** Alternatives, in order: literal string with escapes `\((?:\\.|[^\\)])*\)`, array `\[.*?\]`, name `/\w+`, number, operator letters. `findall` returns a list of `bytes`. `rb"..."` is a raw bytes pattern, needed because streams are bytes.
- **Why it is here:** A content stream is postfix: operands first, then the operator. The tokenizer separates them; the next block interprets.

### 9. run_content: interpret text and image operators  (`main.py` lines 129–166)
```python
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
```
- **In simple words:** Walk the tokens, collecting operands until an operator letter appears, then act: `Tf` size, `Td`/`Tm` position, `Tr` render mode, `Tj`/`TJ` emit a span, `cm`/`Do` record an image placement.
- **Exactly:** `st` is a minimal graphics state (position, size, render mode, matrix); `q`/`Q` push and pop it. `operands[-2]`, `operands[-1]` are the last numbers before `Td`. A `TJ` array interleaves strings and kerning numbers; a number below -200 is treated as a word space. Output: `spans` (x, y, size, text, mode) and `images` (name, drawn width, height).
- **Why it is here:** This is a toy of what PyMuPDF's text extractor does; real ones also handle font encodings, text matrices, clipping and Form XObjects.

### 10. read_pdf: bytes in, pages out  (`main.py` lines 169–188)
```python
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
```
- **In simple words:** For every page: read the MediaBox, run its content streams, find the image XObjects it painted and their filters.
- **Exactly:** Regexes are tolerant to both our own formatting (`/MediaBox [0 0 600 480]`) and Pillow's (`[ 0 0 600.0 480.0 ]`). `/Contents` can be one reference or an array. The XObject names (`/Im0`, `/image`) map to object numbers through `/Resources /XObject`. Returns a list of dicts: `width`, `height`, `spans`, `images` (each with `filter`, `dict`, `data`).
- **Why it is here:** The reader works on a file written by a different producer (Pillow), which is the real test that it is not just inverting our own writer.

### 11. The OCR engine for the OCR route  (`main.py` line 192)
```python
OCR = TemplateOCR(sizes=(21,))
```
- **In simple words:** Template OCR for font size 21 only, enough for the demo scan and faster to build.
- **Exactly:** `TemplateOCR(sizes=(21,))` renders templates for one size ([step 04](../04_character_recognition/WALKTHROUGH.md)).
- **Why it is here:** The router must have something to route to; it reuses module 18's recogniser instead of a new one.

### 12. classify: the per-page decision  (`main.py` lines 195–205)
```python
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
```
- **In simple words:** Count visible and invisible characters and the fraction of the page covered by the largest image; return one of four labels.
- **Exactly:** `mode != 3` selects visible text. `cover = image area / page area` from the drawn size (the `cm` matrix), not from pixel counts. Rules are checked in order: enough visible text wins, then invisible text over an image, then a covering image, else empty or vector.
- **Why it is here:** Thresholds are heuristics; add a garbage-text check (`(cid:` ratio) before trusting `native` in production.

### 13. native_text: spans back into lines  (`main.py` lines 208–213)
```python
def native_text(page):
    """Group spans into lines by baseline y (PDF y grows upward, so sort by -y), then left to right."""
    rows = {}
    for s in page["spans"]:
        rows.setdefault(round(s["y"]), []).append(s)
    return [" ".join(s["text"] for s in sorted(r, key=lambda s: s["x"])) for _, r in sorted(rows.items(), reverse=True)]
```
- **In simple words:** Group spans by baseline y, then sort left to right inside a line and top to bottom across lines.
- **Exactly:** PDF y grows upward, so `sorted(..., reverse=True)` puts the top line first. `round(s["y"])` merges spans on the same baseline. Returns a list of strings.
- **Why it is here:** Drawing order is not reading order; sorting by position repairs simple cases, and for columns you would reuse [step 07](../07_reading_order_and_tables/WALKTHROUGH.md)'s XY-cut on the span boxes.

### 14. decode_image: from an image XObject to a pixel array  (`main.py` lines 216–221)
```python
def decode_image(im):
    """Return a gray uint8 array from an image XObject. Flate was already undone by read_objects; DCT is a JPEG file."""
    if im["filter"] == "DCTDecode":
        return np.asarray(Image.open(io.BytesIO(im["data"])).convert("L"))
    w, h = (int(v) for v in re.search(r"/Width (\d+) /Height (\d+)", im["dict"]).groups())
    return np.frombuffer(im["data"], np.uint8).reshape(h, w)
```
- **In simple words:** A JPEG (`DCTDecode`) is opened by Pillow; a Flate image was already inflated and is reshaped from raw bytes.
- **Exactly:** `Image.open(BytesIO(bytes)).convert("L")` decodes the JPEG to greyscale; `np.frombuffer(data, uint8).reshape(h, w)` interprets 8-bit gray rows. Width and height come from the image dictionary.
- **Why it is here:** OCR needs pixels; this shows that extracting the embedded image is lossless for Flate and keeps the scanner's own JPEG artefacts for DCT.

### 15. route: native text or OCR  (`main.py` lines 224–231)
```python
def route(page):
    kind, vis, hid, cover = classify(page)
    if kind in ("native", "searchable_scan"):
        return kind, "text layer", native_text(page)
    if kind == "scanned":
        gray = decode_image(page["images"][0])
        return kind, "OCR", [l["text"] for l in OCR.read_region(binarise(gray))]
    return kind, "nothing to read", []
```
- **In simple words:** Call `classify`, then extract the text layer for native and searchable pages, or decode the image and run the OCR for scanned pages.
- **Exactly:** `binarise` is Otsu ([step 01](../01_images_as_arrays/WALKTHROUGH.md)); `OCR.read_region` returns line dicts. The function returns `(kind, how, lines)` so the caller can log the route.
- **Why it is here:** This is the whole idea of the step in eight lines.

### 16. cer: a small edit-distance for the check  (`main.py` lines 234–241)
```python
def cer(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(ref)
```
- **In simple words:** Character error rate of the OCR output against the known text, same recurrence as [step 06](../06_ocr_evaluation/WALKTHROUGH.md).
- **Exactly:** Rolling-row Levenshtein; dividing by `len(ref)`.
- **Why it is here:** To prove the two routes both return the right text on the demo pages.

### 17. Build the three demo PDFs  (`main.py` lines 245–262)
```python
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
```
- **In simple words:** A born-digital file, a Pillow-written scan, and a 3-page bundle: native (Flate-compressed content), scanned, searchable scan.
- **Exactly:** `render_lines` returns `(gray array, positions)`; `np.pad` extends the image with white to the page height. `Image.save("x.pdf", resolution=96)` makes Pillow write a PDF with one `/DCTDecode` image. In the third page the invisible text is positioned over the image lines (`3 Tr` + `BT ... Tj ET`).
- **Why it is here:** Real bundles mix page types; only a per-page router handles them.

### 18. Look at the raw bytes  (`main.py` lines 265–270)
```python
raw = (HERE / "born_digital.pdf").read_bytes()
print(f"born_digital.pdf = {len(raw)} bytes. Start of the file:")
print("   " + raw[:raw.index(b"2 0 obj")].decode("latin-1").replace("\n", "\n   "))
print("   ...")
print("   " + raw[raw.index(b"4 0 obj"):raw.index(b"\nendstream")].decode("latin-1").replace("\n", "\n   "))
print("   ...  ends with:", raw[-48:].decode("latin-1").replace("\n", " | "))
```
- **In simple words:** Print the start of the file, one content stream and the end.
- **Exactly:** The slices use `bytes.index` to find the first object boundaries; `decode("latin-1")` maps every byte to a character so it never fails.
- **Why it is here:** Seeing `BT /F1 16 Tf 72 400 Td (Chai Kitchen Quarterly Report) Tj ET` in a plain text file demystifies PDF.

### 19. Parse, classify, route, check  (`main.py` lines 272–291)
```python
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
```
- **In simple words:** Read Pillow's file and the mixed bundle, print the class and route of every page and the extracted text, then compare with the truth.
- **Exactly:** Pillow's PDF has different object numbering and spacing than ours (the page is object 2, the catalog is object 4), and the reader still works because it follows the trailer.
- **Why it is here:** Proof that the router works across producers and that the OCR route's text equals the truth (CER 0.000) on this clean scan.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `rb"..."` | raw bytes literal (regex on bytes; backslashes not interpreted) |
| `while (m := head.search(data, pos)):` | walrus assignment: search and test in one expression |
| `data[start:start + length]` | slice bytes by an exact length |
| `f"{off:010d}"` | format an int zero-padded to 10 digits (xref entries) |
| `b"..." + bytes` / `bytearray +=` | bytes concatenation; bytearray is mutable |
| `re.search(...)[1]` | group 1 of a match (Python 3.6+ match objects are subscriptable) |
| `np.frombuffer(b, np.uint8).reshape(h, w)` | interpret bytes as an array without copying |
| `next(gen)` / `max(..., default=0)` | first item of a generator / max with a fallback for empty input |

## Libraries & functions
| Function | What it does | Internally |
|---|---|---|
| `zlib.compress` / `zlib.decompress` | DEFLATE with a zlib header (PDF `/FlateDecode`) | LZ77 + Huffman coding |
| `PIL.Image.save("x.pdf")` | writes a one-page PDF around the image | chooses the filter by mode: JPEG (`DCTDecode`) for greyscale/RGB, CCITT for 1-bit |
| `PIL.Image.open(BytesIO(...))` | decode an image from bytes | detects the format from the magic bytes (JPEG `FFD8`) |
| `re.findall` / `re.search` | pattern matching on bytes or str | backtracking regex engine |
| `dococr.TemplateOCR.read_region` | OCR of a binarised page region | [steps 03-04](../04_character_recognition/WALKTHROUGH.md) |
| PyMuPDF `page.get_text`, pypdf `extract_text` (not run) | text layer extraction | this step's loop plus fonts, encodings and a renderer |

## Execution trace (real run)
```
born_digital.pdf = 759 bytes. Start of the file:
   %PDF-1.4
   %âãÏÓ
   1 0 obj
   << /Type /Catalog /Pages 2 0 R >>
   endobj
   
   ...
   4 0 obj
   << /Length 206  >>
   stream
   BT /F1 16 Tf 72 400 Td (Chai Kitchen Quarterly Report) Tj ET
   BT /F1 16 Tf 72 372 Td (Refunds are issued within nine days of delivery.) Tj ET
   BT /F1 16 Tf 72 344 Td (Orders above 499 rupees ship free.) Tj ET
   ...  ends with: r | << /Size 6 /Root 1 0 R >> | startxref | 576 | %%EOF | 

Pillow's scanned.pdf: 1 image, 0 text spans, image filter /DCTDecode, page 600x480 pt  -> scanned
   routed via OCR (the JPEG is decoded by Pillow, then binarised and read): ['Invoice 4821 for Chai Kitchen', 'Total due: 1250 rupees', 'Paid by card on time.']

mixed.pdf, page by page:
   page 1: native           visible chars 111, invisible chars   0, image cover   0%  -> route: text layer
           'Chai Kitchen Quarterly Report'
           'Refunds are issued within nine days of delivery.'
           'Orders above 499 rupees ship free.'
   page 2: scanned          visible chars   0, invisible chars   0, image cover 100%  -> route: OCR
           'Invoice 4821 for Chai Kitchen'
           'Total due: 1250 rupees'
           'Paid by card on time.'
   page 3: searchable_scan  visible chars   0, invisible chars  72, image cover 100%  -> route: text layer
           'Invoice 4821 for Chai Kitchen'
           'Total due: 1250 rupees'
           'Paid by card on time.'

OCR page vs truth: CER 0.000   |   native page vs truth: CER 0.000
page-2 pixels as stored in the PDF: (640, 800) -> same array we drew; PDF stored it Flate-compressed
```
- The raw file is plain text for everything but the stream bytes: object numbers, `/Length 206` for a 3-line content stream, and the trailer pointing `/Root 1 0 R`.
- Pillow's `scanned.pdf` stores the page as one **`/DCTDecode`** (JPEG) image with zero text spans: classified `scanned`, routed to OCR, and the JPEG-decoded pixels still read perfectly.
- Page 3 has 0 visible and 72 invisible characters over a 100% image: `searchable_scan`. The router uses its layer, and the text is the OCR'd one we planted (which is right here; in practice, sample-check it).
- Both routes give CER 0.000 on these clean synthetic pages; the cost difference (pixels, binarise, template matching versus a regex over bytes) is the reason to route at all.
- Timing varies by machine; nothing here is slow.

## Experiments (change ONE thing)
| Change | What you will see | Why |
|---|---|---|
| Add `Pdf().add_page(W, H)` as page 4 of the bundle | `empty_or_vector`, route "nothing to read" | No text, no image: charts drawn as paths land here in real files |
| `classify(page, min_cover=1.1)` | scans become `empty_or_vector` | The cover can never reach 110%: the OCR route never fires |
| `TemplateOCR(sizes=(16,))` | CER about 0.85 on the scan | The page was drawn at 21 px: templates of the wrong size |
| Cut `born_digital.pdf` at `xref` | Reader still returns 3 lines | It never uses the xref table; it finds the Catalog by scanning |
| Change `/Length 206` to `/Length 100` | Only the first line comes back, no error | The stream is sliced short; always validate against `endstream` |
| `compress=True` on every page | Smaller file; the raw bytes no longer show the text | Content streams are Flate-compressed in almost every real PDF |

## Debugging practice
**BREAK IT:** in `Pdf.stream`, build the dictionary with `len(data) + 5` instead of `len(data)` for `/Length`.

**Symptom:** `read_objects` slices five bytes too many: for a Flate stream, `zlib.decompress` raises `zlib.error: Error -3 while decompressing data: incorrect header check` or silently returns trailing garbage for a plain stream; the page then has fewer or garbled spans, and `classify` may call a native page `empty_or_vector`. In a real viewer the file shows a "damaged, repairing" dialog, because viewers recompute lengths by finding `endstream`.

**Fix:** write the exact byte length of the **stored** (post-compression) data, and when reading, validate that `data[start + length:]` begins with `endstream` (allowing one end-of-line), falling back to searching for `endstream` if not.

## Common mistakes
- Treating "has a text layer" as "has good text": check for `(cid:` / U+FFFD / mojibake and for the invisible-layer case.
- OCRing born-digital pages (slow and lossy) or not OCRing scans (empty index): route per page, not per file.
- Rendering a scanned page at an arbitrary dpi instead of extracting the embedded image at its native resolution.
- Forgetting the y flip: PDF boxes are bottom-left origin, image boxes are top-left. Citations drawn upside down are this bug.
- Assuming reading order equals drawing order in a text layer. It often does, but not for columns, text boxes or footnotes.
- Writing your own PDF parser for production. This step is for understanding: use a maintained library (mind the licence), which handles encryption, object streams, fonts and malformed files.

## Rebuild checklist
1. Write `esc`, then `Pdf` with `add` (object list), `stream` (`/Length`, optional `/FlateDecode`), `add_page` (text operators, optional image XObject, optional `3 Tr`) and `tobytes` (offsets, xref, trailer).
2. Write `read_objects` (find `N 0 obj`, slice by `/Length`, inflate Flate) and `page_list` (trailer `/Root`, `/Pages`, `/Kids`).
3. Write the tokenizer regex and `run_content` handling `q Q cm Do BT Tf Td Tm Tr Tj TJ`.
4. Write `read_pdf`: MediaBox, contents, XObject map, filter names.
5. Write `classify` (visible chars, hidden chars, image cover), `native_text`, `decode_image`, `route`.
6. Build `born_digital.pdf`, a Pillow `scanned.pdf` and a 3-page `mixed.pdf`; print class, route, text and CER for each page.

## Say it aloud
- "A PDF is a page description language: a tree of objects whose content streams paint glyphs and images at coordinates; paragraphs and tables are not stored, they are inferred."
- "I route per page: enough visible text means extract the text layer, a full-page image with no text means OCR, and invisible text over an image is a searchable scan whose layer I verify because it is somebody else's OCR."
- "PyMuPDF and pypdf do the same loop I wrote: find objects, walk the page tree, inflate streams, interpret operators, map glyphs to Unicode; PyMuPDF adds a renderer, and note it is AGPL."
- "Failure modes I check for: text layers full of `(cid:123)`, pages that are vector drawings, and the y-axis flip when I draw citation boxes."
- "OCR is the expensive fallback, so I log which pages used it and why.

## The new problem (cliffhanger)
The router needs something to send hard pages to. In 2026 the choice is wide: pipeline toolkits (Docling, MinerU, marker), engines (Tesseract, PaddleOCR), end-to-end vision-language models (PaddleOCR-VL, olmOCR-2, DeepSeek-OCR, dots.ocr, GLM-OCR) and cloud APIs. Which one for which document, on which hardware, and how do you know? [Step 09](../09_modern_document_parsers_2026/WALKTHROUGH.md) maps the landscape and builds a decision helper.
