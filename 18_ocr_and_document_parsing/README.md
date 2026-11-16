# Module 18 — OCR and Document Parsing

How a scanned or digital document becomes clean, ordered, citable text and tables: pixels, binarisation, layout, character and word recognition (CRNN + CTC), evaluation, reading order and tables, PDF internals and routing, the 2026 landscape of parsers, and the handoff to chunking. Everything runs on a laptop CPU with synthetic pages generated from code (no datasets, no downloads, no OCR engine installed).

## The story of this module
- **Pain point:** a RAG system is only as good as the text you feed it. A scanned invoice returns an empty string, a two-column paper comes out interleaved, a table loses its columns, and a "99% accurate" OCR hides the one wrong digit in the total. Most "the LLM got it wrong" bugs in document QA are parsing bugs.
- **Spark:** document parsing is a pipeline of small, checkable stages. Treat an image as an array, clean it, cut it into lines and characters, recognise them, score the result with metrics that fit the task, recover the layout (reading order, tables), route each PDF page to the text layer or to OCR, pick a modern parser on purpose, and hand the result to the next stage as a validated contract.
- **What you build:** Otsu thresholding, a median filter and morphology, skew detection, projection-profile segmentation, a template matcher and a small CNN, CTC from scratch plus a tiny CRNN, edit distance / CER / WER / field accuracy / TEDS-like table scores, recursive XY-cut with ruling-line tables, a PDF writer and reader with a per-page router, a decision helper for the 2026 landscape, and a pydantic-validated `parsed_document.json` that module 19's chunker reads unchanged.
- **Cliffhanger:** you now have typed elements with pages and boxes. How to cut them into chunks that keep sections and tables together is [19_chunking_deep_dive](../19_chunking_deep_dive/README.md), step 09.

**Short on time?** Read [INTERVIEW_GIST.md](INTERVIEW_GIST.md) first.

## 1. Why this matters
Interviews for document-AI, RAG and ML-engineering roles ask: "How would you extract text and tables from scanned PDFs?", "How do you evaluate OCR?", "Why does my RAG answer the wrong page?". Strong answers separate **born-digital** from **scanned** pages, name the stages (preprocess, layout, recognise, reading order, tables), pick metrics (CER/WER, field accuracy, TEDS, reading order), know the trade-off between classic engines, pipelines, VLM OCR and cloud APIs, and say what they would verify (a gold set of their own pages, not a leaderboard).

## 2. Where this fits
**Builds on**
- [03_ml_foundations](../03_ml_foundations/README.md): numpy arrays, the PyTorch training loop ([step 04](../03_ml_foundations/04_pytorch_training_loop/)) and cross-entropy ([step 06](../03_ml_foundations/06_softmax_cross_entropy/)), used by the CNN and CRNN here.
- [15_neural_networks/08_cnn_convolutions](../15_neural_networks/08_cnn_convolutions/): convolutions and pooling, the feature extractor inside the CRNN of step 05.
- [14_rag_and_retrieval/01_documents_and_why_rag](../14_rag_and_retrieval/01_documents_and_why_rag/WALKTHROUGH.md): why documents have to become text in the first place.

**Leads to**
- [19_chunking_deep_dive](../19_chunking_deep_dive/README.md): [step 09, layout-aware chunking from parsed docs](../19_chunking_deep_dive/09_layout_aware_chunking_from_parsed_docs/main.py) consumes the `parsed_document.json` shape that step 10 here produces.
- [14_rag_and_retrieval](../14_rag_and_retrieval/README.md): retrieval over the chunks, with page and bounding-box citations.
- [13_capstone_project](../13_capstone_project/README.md): the document-QA track (ingest, parse, chunk, retrieve, answer with citations).
- Practice: [AI_ML_Debugging_Practice/04_pdf_text_extraction](../AI_ML_Debugging_Practice/04_pdf_text_extraction/question.md): a buggy PDF-extraction stage (wrong page numbers, footer text leaking into the body, words glued together, a scanned page that needs OCR). Do it after step 08: the page that "has no text layer" is exactly step 08's `scanned` class.

## 3. Concept map
```
 PDF / photo / scan
      |  08  per-page router: text layer (born-digital) | image-only -> OCR | image + invisible text -> verify
      v
 01 pixels -> gray -> histogram -> Otsu / local threshold          (binary ink mask)
 02 median filter, morphology, deskew, resolution                    (clean, straight, big enough)
 03 lines (row profile) -> words (gaps) -> characters (components)   (boxes)
 04 template matching -> CNN (+ augmentation)                        (character classifier, confusion pairs)
 05 CRNN + CTC: whole-word reading without cutting letters           (text with blanks collapsed)
      |
      v
 07 reading order (XY-cut) + tables (ruling lines) + typed blocks    (structured pages)
 06 evaluation: edit distance, CER/WER, normalisation, field accuracy, TEDS-like  (how good, honestly)
 09 landscape: engines | pipeline toolkits | VLM OCR | cloud APIs  -> pick by constraints
      |
      v
 10 parsed_document.json (typed elements, page, bbox in points, reading_order)  --pydantic-->  19/09 layout-aware chunker -> 14 RAG -> 13 capstone
```

## 4. Prerequisites
Python and numpy at the level of [00](../00_python_foundations/README.md) and [03](../03_ml_foundations/README.md); the PyTorch loop from 03 for steps 04-05; nothing else. No OCR package is needed: the module builds its own teaching OCR in `dococr.py` (it recognises the bundled Pillow font at known sizes: a teaching OCR, not a product).

## 5. Study path
Each step folder has `main.py`, `WALKTHROUGH.md` (line-by-line explanation with a real execution trace) and `TERMS.md`. Timings are for a CPU laptop and vary by machine.

| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_images_as_arrays](01_images_as_arrays/) | RGB to gray (luma), histograms, Otsu from scratch, fixed vs Otsu vs local threshold on dim and shadowed pages | `python main.py` | 40 min |
| 2 | [02_preprocessing](02_preprocessing/) | median filter, erosion/dilation/opening/closing, brute-force skew search, resolution vs glyph size | `python main.py` | 45 min |
| 3 | [03_layout_segmentation](03_layout_segmentation/) | projection profiles for lines and words, connected components for characters, multi-piece glyphs | `python main.py` | 45 min |
| 4 | [04_character_recognition](04_character_recognition/) | 1-NN templates, a small CNN, augmentation, confusion pairs (`0/O`, `1/l`, `rn/m`) | `python main.py` (about 20 s) | 60 min |
| 5 | [05_crnn_and_ctc](05_crnn_and_ctc/) | CTC collapse, CTC loss vs brute force, greedy and prefix-beam decoding, a tiny CRNN | `python main.py` (about 1 min) | 75 min |
| 6 | [06_ocr_evaluation](06_ocr_evaluation/) | Levenshtein, CER/WER, normalisation, reading-order damage, field accuracy, TEDS-like | `python main.py` | 50 min |
| 7 | [07_reading_order_and_tables](07_reading_order_and_tables/) | recursive XY-cut, ruling-line tables, typed blocks, the 25 px gutter and the "cif" misread | `python main.py` | 60 min |
| 8 | [08_pdf_internals_and_routing](08_pdf_internals_and_routing/) | PDF objects/xref/content streams by hand, text layer vs scan vs searchable scan, per-page routing | `python main.py` | 60 min |
| 9 | [09_modern_document_parsers_2026](09_modern_document_parsers_2026/) | engines vs pipelines vs VLM OCR vs cloud, what fits in 8 GB, a decision helper, benchmark caveats | `python main.py` | 45 min |
| 10 | [10_parsed_document_handoff](10_parsed_document_handoff/) | the OCR to chunking contract, pydantic validation, running module 19's chunker on our file | `HF_HUB_OFFLINE=1 python main.py` | 50 min |

Shared helpers (read them, they are short): [docimg.py](docimg.py) (synthetic pages, noise, the report page) and [dococr.py](dococr.py) (Otsu, projection profiles, component merging, template OCR). Then do [exercises.md](exercises.md) and check [solutions/solutions.md](solutions/solutions.md).

## 6. The concepts, one by one
- **Image as array.** A gray page is a `uint8` array (H, W); colour adds a channel. Gray = 0.299 R + 0.587 G + 0.114 B (luma), not the plain mean ([step 01](01_images_as_arrays/WALKTHROUGH.md)).
- **Binarisation.** Otsu chooses the threshold that maximises between-class variance: perfect on evenly lit pages, wrong on shadows, where a local (neighbourhood-mean) threshold wins.
- **Cleaning.** Median filter for salt-and-pepper noise (preserves edges, unlike a box blur); opening removes specks but erodes thin strokes; deskew by maximising the variance of the row profile; glyphs below roughly 10 px tall break recognition ([step 02](02_preprocessing/WALKTHROUGH.md)).
- **Segmentation.** Row sums give lines, column sums plus a gap threshold give words, connected components give characters; `i`, `j`, `:`, `;` and `!` are two components and touching letters (`rn`) are one ([step 03](03_layout_segmentation/WALKTHROUGH.md)).
- **Character classification.** Template matching works on the training font and degrades on a new font plus rotation plus noise; a small CNN with augmentation is much more robust; the confusable pairs are confusable in pixel space too ([step 04](04_character_recognition/WALKTHROUGH.md)).
- **CRNN + CTC.** Convolutions make a feature sequence, a bidirectional GRU adds context, CTC sums over all alignments (with a blank symbol and a collapse rule) so no character segmentation is needed; decoding is greedy or prefix beam search ([step 05](05_crnn_and_ctc/WALKTHROUGH.md)).
- **Evaluation.** CER/WER are edit distances over characters/words; normalisation rules change the numbers more than the model does; reading order and tables need their own metrics; fields need exact or numeric matching ([step 06](06_ocr_evaluation/WALKTHROUGH.md)).
- **Layout.** XY-cut uses white space for reading order; ruling lines give table grids; thresholds are the model and must be calibrated on the narrowest gutter ([step 07](07_reading_order_and_tables/WALKTHROUGH.md)).
- **PDF.** A page description language: objects, a page tree, content streams of operators. Text layer, image-only, or image plus invisible text; route per page ([step 08](08_pdf_internals_and_routing/WALKTHROUGH.md)).
- **Landscape.** Engines (Tesseract, PaddleOCR), pipeline toolkits (Docling, MinerU, marker), VLM OCR (PaddleOCR-VL, olmOCR-2, DeepSeek-OCR, dots.ocr, GLM-OCR) and cloud APIs: choose by hard constraints first, then skills; benchmark numbers are leads to verify ([step 09](09_modern_document_parsers_2026/WALKTHROUGH.md)).
- **Handoff.** Typed elements with page, bbox (PDF points, top-left), reading order, structured tables; validated at the border; confidence in a sidecar ([step 10](10_parsed_document_handoff/WALKTHROUGH.md)).

## 7. Same thing, different names
| Term you will hear | Also called | Exactly the same? | Difference in one line |
|---|---|---|---|
| Binarisation | thresholding, ink mask | yes | Gray to black/white; global (Otsu) or local (adaptive) |
| Layout analysis | page segmentation, document layout analysis (DLA) | near | Boxes and classes; reading order is a separate step |
| CER / WER | character / word error rate, normalised edit distance | near | Same formula over characters or words; both can exceed 1 |
| Born-digital PDF | native, text PDF, vector PDF | yes | Has a text layer; no OCR needed |
| Scanned PDF | image PDF, raster PDF | yes | Page is an image; OCR mandatory |
| Searchable PDF | sandwich PDF, OCR'd PDF | near | Image plus invisible OCR text, quality = someone else's OCR |
| VLM OCR | OCR-free / end-to-end document parsing | near | A vision-language model writes the page as Markdown |
| Parsed document | layout elements, DoclingDocument, structured document | near | Typed elements with page/bbox/order instead of a string |
| bbox | bounding box, `[x0, y0, x1, y1]` | yes | Units and origin differ per tool: always check |
| CTC | connectionist temporal classification | yes | Loss that sums over alignments with a blank and a collapse rule |

## 8. Common mistakes
- Running OCR on every PDF page (slow, lossy) or on none (scans become empty strings): route per page.
- Reporting "accuracy" without saying CER or WER, normalisation, micro/macro averaging, or the document set.
- Trusting a fuzzy field match for amounts and IDs (`230.00` versus `2300.00` passes a 15% CER threshold).
- Sorting lines by y on a multi-column page and blaming OCR for garbled text.
- Quoting leaderboard numbers (versions, runners and document mixes differ) instead of scoring your own 50 pages.
- Using a generating VLM on numbers-critical text without a cross-check.
- Flattening tables to text, or dropping page and bbox, at the handoff.
- Mixing coordinate systems (pixels vs points, bottom-left vs top-left origin).

## 9. Interview Q&A
**1. Born-digital vs scanned PDF: how do you handle each?** Born-digital has a text layer: extract it (with boxes) and skip OCR. Scanned has only images: extract or render the page image, preprocess, OCR. Route per page because bundles mix both; a searchable scan (image + invisible text) uses the layer but should be sample-checked.

**2. Explain Otsu's method.** Choose the threshold t that maximises the between-class variance of the two pixel populations (below and above t); equivalently minimise the within-class variance. It assumes a roughly bimodal histogram, so it works on evenly lit pages and fails on shadows, where a local threshold is better.

**3. Why do OCR engines use CTC?** Handwriting and printed text have no per-character alignment labels and letters touch. CTC adds a blank symbol and sums the probability of all frame-level paths that collapse (merge repeats, drop blanks) to the target string, so a CRNN trains on whole lines with only the transcript.

**4. CER vs WER?** CER = (S + D + I) / reference characters; WER is the same over words. One wrong letter makes a whole word wrong, so WER is larger (here 3.4% CER was 15.4% WER). Both can exceed 100%. State the normalisation and whether the average is micro or macro.

**5. How do you evaluate tables and reading order?** Text-only CER is blind to structure (two merged columns scored CER 0.000). Use TEDS (tree-edit-distance on the table's HTML tree) or cell-level metrics for tables, and a reading-order edit distance or order-sensitive WER; swapped columns gave CER 0.79 with every character present.

**6. How does XY-cut work and when does it fail?** Recursively cut at the widest empty horizontal band, else the widest empty vertical band; leaves in recursion order are the reading order. It fails without clean gutters (full-width figures, skew, noise) and its thresholds must be calibrated: a 30 px threshold merged a 25 px gutter in step 07.

**7. What is inside a PDF and how do libraries extract text?** Numbered objects, a page tree, content streams of drawing operators (`BT ... (text) Tj ET`), fonts needing a ToUnicode map, images as XObjects with filters (Flate, JPEG). PyMuPDF and pypdf parse objects, inflate streams, interpret operators and map glyph codes to Unicode.

**8. Why can a text layer still give garbage?** Glyph codes without a ToUnicode CMap yield `(cid:123)`, ligatures and custom encodings break mapping, text may be outlines, and drawing order is not reading order. Detect with a garbage-ratio check and fall back to OCR.

**9. Classic OCR, pipeline toolkit, VLM OCR or cloud API?** Engines (Tesseract, PaddleOCR) for fast CPU text with confidences; pipeline toolkits (Docling, MinerU, marker) for structure; VLM OCR (PaddleOCR-VL, olmOCR-2, dots.ocr...) for complex layouts on GPUs, with hallucination risk; cloud for forms and small volume when data may leave. Hard constraints (privacy, hardware) first, then skills, then a gold-set test.

**10. What are the risks of VLM OCR?** It generates rather than copies: omissions, fabrications, repetition loops, "corrected" typos, no calibrated confidence, and token costs on dense pages. Mitigate with element-level crops, cross-checks of numbers with a classic engine, and field-level validation.

**11. Does a 3B VLM run on an 8 GB laptop?** Weights alone are 3 x 10^9 x 2 bytes = 5.6 GiB in bf16: not with the OS and activations; 4-bit is about 1.4 GiB so it fits, but CPU generation of a Markdown page is minutes. Use a GPU server for volume.

**12. What do you hand to the chunker?** Typed elements (title, section_header, text, table, ...) with page, bbox in PDF points (top-left), reading_order and structured tables (header + rows), validated with a schema. Page and bbox make citations and highlights possible; tables kept structured let the chunker repeat headers.

**13. How do you know your pipeline is good?** A gold set of about 50 to a few hundred of your own pages, scored by CER/WER, field accuracy (exact for amounts), table TEDS and reading order; look at the worst ten pages; and report an interval (45/50 correct is 90% with a 95% interval of about 79 to 96%).

## 10. Self-check
1. Convert an 800 x 640 px (96 dpi) box to points, and flip it to a bottom-left origin.
2. Why does Otsu fail on the shadow page while local mean works?
3. Write the CTC collapse rule and say how many length-4 paths collapse to `ab`.
4. Why is WER larger than CER, and when can CER exceed 1?
5. Why did `min_gap_x=30` merge the columns in step 07, and how would you choose the threshold properly?
6. List the four page types the router distinguishes and the route for each.
7. Name one failure mode of each tool family (engine, pipeline, VLM, cloud).
8. Which fields does `parsed_document.json` carry for every element, and why does confidence live in a sidecar?

## 11. Practice and reference
[exercises.md](exercises.md) · [solutions/solutions.md](solutions/solutions.md) · [INTERVIEW_GIST.md](INTERVIEW_GIST.md) · [AI_ML_Debugging_Practice/04_pdf_text_extraction](../AI_ML_Debugging_Practice/04_pdf_text_extraction/question.md) · vocabulary draft [18_ocr_and_document_parsing](../vocabulary/_drafts/18_ocr_and_document_parsing.md)

Optional, never run in this course (guarded, skip cleanly when packages are absent): [08 `pymupdf_reference.py`](08_pdf_internals_and_routing/pymupdf_reference.py), [09 `reference_engines.py`](09_modern_document_parsers_2026/reference_engines.py).
