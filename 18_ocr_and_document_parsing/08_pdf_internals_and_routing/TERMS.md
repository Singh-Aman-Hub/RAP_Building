# Terms — PDF internals and routing

Folder: `18_ocr_and_document_parsing/08_pdf_internals_and_routing` · Files covered: `main.py`, `pymupdf_reference.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [PDF (page description language)](#pdf-page-description-language) · [Born-digital (native) PDF](#born-digital-native-pdf) · [Scanned (image-only) PDF](#scanned-image-only-pdf) · [Searchable scan (sandwich PDF)](#searchable-scan-sandwich-pdf) · [Indirect object, xref table, trailer](#indirect-object-xref-table-trailer) · [Content stream](#content-stream) · [Text render mode (Tr)](#text-render-mode-tr) · [FlateDecode / DCTDecode (filters)](#flatedecode--dctdecode-filters) · [XObject (image) and the Do operator](#xobject-image-and-the-do-operator) · [PDF coordinate system](#pdf-coordinate-system) · [ToUnicode CMap and (cid:N) garbage](#tounicode-cmap-and-cidn-garbage) · [Page router (native vs OCR)](#page-router-native-vs-ocr) · [PyMuPDF and pypdf (what they do)](#pymupdf-and-pypdf-what-they-do) · [Rasterise (render) a page](#rasterise-render-a-page) · [/Length slicing error](#length-slicing-error)

---

## PDF (page description language)  `(concept)`
**1. Definition:** A file format that describes where to paint glyphs, lines and images on pages; it does not store paragraphs or tables unless tagged.
**2. In plain words:** A program that draws the page, not a document with structure.
**3. Example to quote:** > "A PDF is a page description language: structure is inferred, not stored."
**Seen in:** `main.py`, `Pdf`

## Born-digital (native) PDF  `(concept)`
**1. Definition:** A PDF produced by software, with a real text layer.
**2. In plain words:** Text you can select and copy; extract it, do not OCR it.
**3. Example to quote:** > "Born-digital pages go through text extraction; OCR would be slower and less accurate."
**Seen in:** `main.py`, `classify`

## Scanned (image-only) PDF  `(concept)`
**1. Definition:** A PDF whose pages are full-page images with no text.
**2. In plain words:** A photo of a page wrapped in PDF.
**3. Example to quote:** > "A scanned PDF returns an empty string from a text extractor, which a RAG index silently stores as no content."
**Seen in:** `main.py`, `scanned.pdf`

## Searchable scan (sandwich PDF)  `(concept)`
**1. Definition:** A scan image with an invisible OCR text layer (render mode 3) on top.
**2. In plain words:** Selectable text, but from someone else's OCR.
**3. Example to quote:** > "I use the layer but sample-check it against my own OCR."
```python
ops.append("3 Tr")
```
**Seen in:** `main.py`, page 3 of `mixed.pdf`

## Indirect object, xref table, trailer  `(concept)`
**1. Definition:** `N G obj ... endobj` objects addressed as `N G R`; the xref table stores byte offsets; the trailer names the root Catalog and `startxref` locates the table.
**2. In plain words:** Numbered boxes plus an index at the end of the file.
**3. Example to quote:** > "Readers start at the end of the file; if the xref is damaged they rebuild it by scanning."
```python
out += f"{off:010d} 00000 n \n".encode()
```
**Seen in:** `main.py`, `Pdf.tobytes`, `read_objects`

## Content stream  `(concept)`
**1. Definition:** The operator stream that paints one page (text, paths, images), usually Flate-compressed.
**2. In plain words:** The drawing instructions for a page.
**3. Example to quote:** > "BT /F1 16 Tf 72 400 Td (Hello) Tj ET is how text is stored: a position and a string, no paragraph."
```python
BT /F1 16 Tf 72 400 Td (Hello) Tj ET
```
**Seen in:** `main.py`, `run_content`

## Text render mode (Tr)  `(concept)`
**1. Definition:** Operator selecting how text is painted; mode 3 is invisible.
**2. In plain words:** Text that exists but is not drawn.
**3. Example to quote:** > "OCR software uses Tr 3 to put searchable text under a scan."
**Seen in:** `main.py`, `classify`

## FlateDecode / DCTDecode (filters)  `(concept)`
**1. Definition:** Stream compressions: Flate is zlib (lossless), DCT is JPEG (lossy); others are CCITT, JBIG2, JPX.
**2. In plain words:** How the bytes of a stream are packed.
**3. Example to quote:** > "Pillow saved the greyscale scan as DCTDecode: the JPEG artefacts are in the PDF already."
```python
stream = zlib.decompress(stream)
```
**Seen in:** `main.py`, `read_objects`, `decode_image`

## XObject (image) and the Do operator  `(concept)`
**1. Definition:** A reusable object (image or form) painted with `Do`; a `cm` matrix sets its size and position.
**2. In plain words:** A picture placed on the page.
**3. Example to quote:** > "Image cover is measured from the drawn size in the cm matrix, not from pixels."
```python
q w 0 0 h 0 0 cm /Im0 Do Q
```
**Seen in:** `main.py`, `add_page`, `run_content`

## PDF coordinate system  `(concept)`
**1. Definition:** Units are points (1/72 in), origin bottom-left, y up; images use top-left origin.
**2. In plain words:** Flip y when you draw boxes on an image.
**3. Example to quote:** > "A constant upside-down citation box is the y-flip bug."
```python
y_top = page_height - y_pdf
```
**Seen in:** `main.py`, `add_page` (`h - y`)

## ToUnicode CMap and (cid:N) garbage  `(error)`
**1. Definition:** The glyph-code to Unicode map of a font; without it extraction yields `(cid:123)` or random symbols.
**2. In plain words:** The PDF has text but the codes do not map to letters.
**3. Example to quote:** > "I treat a page full of (cid: as a scan and OCR it."
**Seen in:** Step walkthrough, theory section

## Page router (native vs OCR)  `(concept)`
**1. Definition:** Per-page decision using visible-character count and image coverage.
**2. In plain words:** Use free text when present; OCR only when needed.
**3. Example to quote:** > "Route per page, log the route, and OCR is the expensive fallback."
```python
kind, how, lines = route(page)
```
**Seen in:** `main.py`, `classify`, `route`

## PyMuPDF and pypdf (what they do)  `(concept)`
**1. Definition:** Libraries that implement object parsing, filter decoding, operator interpretation and font mapping; PyMuPDF also renders pages (MuPDF, AGPL), pypdf is pure Python (BSD).
**2. In plain words:** The grown-up versions of this step's 100 lines.
**3. Example to quote:** > "PyMuPDF get_text('words') gives words with boxes; get_pixmap renders a page for OCR."
```python
page.get_text("words")
```
**Seen in:** `pymupdf_reference.py` (not run here)

## Rasterise (render) a page  `(concept)`
**1. Definition:** Draw a PDF page to pixels at a chosen dpi (PyMuPDF get_pixmap, poppler pdftoppm).
**2. In plain words:** Turn a vector page into an image you can OCR.
**3. Example to quote:** > "For scans extract the embedded image instead; render only mixed pages."
**Seen in:** Theory section

## /Length slicing error  `(error)`
**1. Definition:** A wrong stream length makes a reader slice the wrong bytes: zlib 'incorrect header check' or silently truncated content.
**2. In plain words:** The size label of the stream lies.
**3. Example to quote:** > "Changing /Length 206 to 100 returned one of three lines with no error."
**Seen in:** `main.py`, `read_objects`; BREAK IT
