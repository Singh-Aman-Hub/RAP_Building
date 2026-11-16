# Terms — Layout-Aware Chunking

Folder: `19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [19 README](../README.md)

**Terms in this folder:** [Parsed element](#parsed-element) · [Reading order](#reading-order) · [Bounding box (bbox)](#bounding-box-bbox) · [Page header / footer noise](#page-header--footer-noise) · [Caption binding](#caption-binding) · [Cross-page table merge](#cross-page-table-merge) · [Row linearisation](#row-linearisation) · [Docling HybridChunker / Unstructured chunk_by_title](#docling-hybridchunker--unstructured-chunk_by_title) · [Citation](#citation)

---

## Parsed element  `(concept)`
**1. Definition:** {id, type, text, page, bbox, reading_order, ...} produced by a parser or OCR + layout model.
**2. In plain words:** A labelled box on a page.
**3. Example to quote:** > "Parsed element: A labelled box on a page."
**Seen in:** sample_parsed_document.json

## Reading order  `(concept)`
**1. Definition:** The order a human reads the elements (columns, captions). Array order from a parser may differ.
**2. In plain words:** Sort by (page, reading_order).
**3. Example to quote:** > "Reading order: Sort by (page, reading_order)."
**Seen in:** main.py, LayoutChunker.chunk

## Bounding box (bbox)  `(concept)`
**1. Definition:** [x0, y0, x1, y1] in page points, origin top-left here.
**2. In plain words:** Where the text is on the page.
**3. Example to quote:** > "Bounding box: Where the text is on the page."
**Seen in:** main.py, union_boxes

## Page header / footer noise  `(concept)`
**1. Definition:** Repeated lines ('Confidential - page 2') that pollute chunks.
**2. In plain words:** Drop them, keep the page number as metadata.
**3. Example to quote:** > "Page header / footer noise: Drop them, keep the page number as metadata."
**Seen in:** main.py, NOISE

## Caption binding  `(concept)`
**1. Definition:** Attach a caption to the table (above) or figure (below) it describes.
**2. In plain words:** A figure without its caption is mute.
**3. Example to quote:** > "Caption binding: A figure without its caption is mute."
**Seen in:** main.py, chunk

## Cross-page table merge  `(concept)`
**1. Definition:** Two table elements with the same header on consecutive pages become one table.
**2. In plain words:** One table, one chunk (or one set of row groups).
**3. Example to quote:** > "Cross-page table merge: One table, one chunk (or one set of row groups)."
**Seen in:** main.py, merge_continued_tables

## Row linearisation  `(concept)`
**1. Definition:** Row -> 'Column: value; Column: value' text for embedding.
**2. In plain words:** Embedders read sentences better than pipes.
**3. Example to quote:** > "Row linearisation: Embedders read sentences better than pipes."
**Seen in:** main.py, linearise

## Docling HybridChunker / Unstructured chunk_by_title  `(library)`
**1. Definition:** Docling: token-aware chunking over the document hierarchy with merge_peers; Unstructured: a Title starts a chunk, with max_characters, new_after_n_chars, combine_text_under_n_chars.
**2. In plain words:** Ready-made versions of this step.
**3. Example to quote:** > "Both consume parsed elements, not raw text."
**Seen in:** WALKTHROUGH, theory

## Citation  `(concept)`
**1. Definition:** page + bbox of the answering chunk, e.g. p.2-3 @[72, 530, 540, 720].
**2. In plain words:** What you show the user to verify the answer.
**3. Example to quote:** > "Citation: What you show the user to verify the answer."
**Seen in:** main.py, LChunk.citation
