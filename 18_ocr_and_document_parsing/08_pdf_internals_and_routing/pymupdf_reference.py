"""
OPTIONAL REFERENCE -- runs only if PyMuPDF (import name `fitz` / `pymupdf`) or pypdf is installed. Neither is installed in this
course's .venv and this script NEVER installs or downloads anything: without them it prints a skip message and exits 0.
The code below was written against the documented APIs and is NOT verified in this repo (the libraries are absent).

What the libraries do internally (the same steps as main.py, in C / optimised Python):
  * pypdf  (pure Python): reads the xref table (or rebuilds it by scanning), walks the page tree, decodes each content stream
    (FlateDecode, LZW, ASCII85 ...), runs the text operators and maps font codes to Unicode via the font's /ToUnicode CMap or
    encoding. page.extract_text() returns the concatenated result.
  * PyMuPDF (MuPDF, C library): the same, plus a full renderer. page.get_text("words") returns (x0, y0, x1, y1, word, block, line,
    word_no) with real bounding boxes; page.get_images() lists image XObjects; page.get_pixmap() rasterises a page (useful to OCR it).

RUN (from this step folder):  python pymupdf_reference.py
"""
from pathlib import Path

HERE = Path(__file__).parent
FILE = HERE / "mixed.pdf"                      # created by main.py: run that first


def with_pymupdf(path):
    import fitz                                # PyMuPDF; newer releases also expose `import pymupdf`
    doc = fitz.open(path)
    for i, page in enumerate(doc, 1):
        text = page.get_text("text").strip()                  # text-layer extraction; "" for a pure scan
        words = page.get_text("words")                         # [(x0, y0, x1, y1, word, block, line, word_no), ...]: boxes for citations
        print(f"page {i}: {len(text)} chars, {len(words)} words, {len(page.get_images())} images")
        if len(text) < 20 and page.get_images():
            pix = page.get_pixmap(dpi=200)                     # rasterise -> feed to an OCR engine; here we only report the size
            print(f"   scanned-looking page: would OCR a {pix.width}x{pix.height} pixmap")


def with_pypdf(path):
    from pypdf import PdfReader
    for i, page in enumerate(PdfReader(path).pages, 1):
        text = (page.extract_text() or "").strip()
        print(f"page {i}: {len(text)} chars, {len(page.images)} images")


if not FILE.exists():
    print("mixed.pdf not found: run main.py first. Skipping.")
else:
    for name, fn in (("pymupdf", with_pymupdf), ("pypdf", with_pypdf)):
        try:
            fn(str(FILE))
            break
        except ImportError:
            print(f"[skip] {name} is not installed (this is expected in the course .venv; nothing will be installed).")
