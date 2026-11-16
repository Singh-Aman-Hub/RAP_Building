<!-- vocabulary draft: 18_ocr_and_document_parsing -->

### OCR (optical character recognition)
<!-- cat: ml -->
- **Technical definition:** Converting images of text (scans, photos, screenshots) into machine-readable text, usually with positions and confidences.
- **Plain English:** Teaching the computer to read a picture of a page.
- **Example / analogy:** A scanned invoice is OCR'd into text and boxes before any field extraction.
- **Snippet:**
```python
text = ocr.read_region(binarise(gray))
```
- **Snippet explained:** Binarises the page and returns recognised lines.
- **Related terms:** [[Binarisation (thresholding)]], [[Layout analysis (page segmentation)]], [[CER and WER]]

### Binarisation (thresholding)
<!-- cat: ml -->
- **Technical definition:** Turning a gray image into a black-and-white ink mask by comparing each pixel with a global or local threshold.
- **Plain English:** Deciding which pixels are ink and which are paper.
- **Example / analogy:** Otsu picks one global threshold; adaptive methods pick one per neighbourhood.
- **Snippet:**
```python
ink = gray <= otsu_threshold(gray)
```
- **Snippet explained:** True marks ink pixels.
- **Related terms:** [[Otsu's method]], [[Adaptive (local) threshold]], [[OCR (optical character recognition)]]

### Otsu's method
<!-- cat: ml -->
- **Technical definition:** Chooses the global threshold that maximises the between-class variance of the histogram of a gray image.
- **Plain English:** Finds the cut that best separates dark ink from bright paper.
- **Example / analogy:** Great on evenly lit pages, wrong on a shadowed page (F1 0.194 in the module run).
- **Snippet:**
```python
t = int(np.nanargmax((mu_t*omega - mu)**2 / (omega*(1-omega))))
```
- **Snippet explained:** Evaluates every threshold from the cumulative histogram and takes the best.
- **Related terms:** [[Binarisation (thresholding)]], [[Adaptive (local) threshold]]

### Adaptive (local) threshold
<!-- cat: ml -->
- **Technical definition:** A threshold computed per pixel from its neighbourhood (for example mean minus an offset).
- **Plain English:** Compare each pixel with its neighbours, so shadows cancel.
- **Example / analogy:** Local mean (31x31) scored F1 0.949 on the shadowed page where Otsu scored 0.194.
- **Snippet:**
```python
ink = gray < local_mean - 25
```
- **Snippet explained:** Pixels darker than their neighbourhood by 25 levels count as ink.
- **Related terms:** [[Otsu's method]], [[Binarisation (thresholding)]]

### Median filter
<!-- cat: ml -->
- **Technical definition:** Replaces each pixel by the median of its neighbourhood; removes salt-and-pepper noise while preserving edges.
- **Plain English:** Throws away the odd one out.
- **Example / analogy:** Median then Otsu gave ink F1 0.892 versus 0.770 for a box blur.
- **Snippet:**
```python
scipy.ndimage.median_filter(img, size=3)
```
- **Snippet explained:** Slides a 3x3 window and takes the median.
- **Related terms:** [[Morphological operations (erosion, dilation, opening, closing)]], [[Binarisation (thresholding)]]

### Morphological operations (erosion, dilation, opening, closing)
<!-- cat: ml -->
- **Technical definition:** Binary image operators that shrink (erosion) or grow (dilation) shapes; opening = erode then dilate, closing = dilate then erode.
- **Plain English:** Shaving off specks and filling tiny gaps in strokes.
- **Example / analogy:** Opening removes dust but erodes thin strokes, so ink F1 fell to 0.284 on thin text.
- **Snippet:**
```python
scipy.ndimage.binary_opening(ink, np.ones((3, 3)))
```
- **Snippet explained:** Removes ink blobs smaller than 3x3.
- **Related terms:** [[Median filter]], [[Connected components]]

### Deskew (skew correction)
<!-- cat: ml -->
- **Technical definition:** Estimating the rotation of a scanned page and rotating it back; commonly by maximising the variance of the row projection profile.
- **Plain English:** Straightening a crooked scan.
- **Example / analogy:** A 4 degree skew made row projection find 1 line instead of 4.
- **Snippet:**
```python
best = max(angles, key=lambda a: profile_variance(rotate(img, a)))
```
- **Snippet explained:** Tries angles and keeps the one with the sharpest row profile.
- **Related terms:** [[Projection profile]], [[Layout analysis (page segmentation)]]

### Projection profile
<!-- cat: ml -->
- **Technical definition:** Sum of ink pixels along rows or columns; gaps in the profile separate lines (rows) and words or columns (columns).
- **Plain English:** A histogram of ink along one axis.
- **Example / analogy:** Row profile gives text lines; column profile gives words and column gutters.
- **Snippet:**
```python
lines = runs(ink.sum(axis=1) > 0)
```
- **Snippet explained:** Rows with any ink form line runs.
- **Related terms:** [[Layout analysis (page segmentation)]], [[Recursive XY-cut]], [[Connected components]]

### Connected components
<!-- cat: ml -->
- **Technical definition:** Labelling groups of touching foreground pixels; each group is one blob, typically a character.
- **Plain English:** Finding separate pieces of ink.
- **Example / analogy:** An i is two components; rn touching is one.
- **Snippet:**
```python
lab, n = scipy.ndimage.label(line_ink)
```
- **Snippet explained:** Returns a label image and the number of components.
- **Related terms:** [[Projection profile]], [[CTC (connectionist temporal classification)]]

### Layout analysis (page segmentation)
<!-- cat: ml -->
- **Technical definition:** Detecting and classifying page regions (title, text, table, figure, header, footer) and ordering them.
- **Plain English:** Understanding the page before reading it.
- **Example / analogy:** Modern parsers use learned layout models with DocLayNet-style labels.
- **Snippet:**
```python
elements = [dict(kind=k, bbox=b) for k, b in blocks]
```
- **Snippet explained:** A list of typed boxes per page.
- **Related terms:** [[Recursive XY-cut]], [[Reading order]], [[Parsed document (layout elements)]]

### Recursive XY-cut
<!-- cat: ml -->
- **Technical definition:** Top-down page segmentation: cut at the widest empty horizontal band, else the widest empty vertical band, recursively; leaves in recursion order give reading order.
- **Plain English:** Use white space to split a page, top and left first.
- **Example / analogy:** Order WER 0.014 with XY-cut versus 0.554 for a naive y-sort on the same text.
- **Snippet:**
```python
gy = widest_gap(sub.any(axis=1), min_gap_y)
```
- **Snippet explained:** Finds the widest empty row band to cut at.
- **Related terms:** [[Reading order]], [[Projection profile]]

### Reading order
<!-- cat: ml -->
- **Technical definition:** The sequence in which the blocks of a page are read; must be inferred for scans.
- **Plain English:** Which block comes next.
- **Example / analogy:** Swapped columns gave CER 0.79 though every character was present.
- **Snippet:**
```python
els = sorted(els, key=lambda e: (e['page'], e['reading_order']))
```
- **Snippet explained:** Consumers sort by page then reading_order, not array order.
- **Related terms:** [[Recursive XY-cut]], [[Parsed document (layout elements)]]

### Table structure recognition
<!-- cat: ml -->
- **Technical definition:** Recovering rows, columns and spans of a table, then reading the cells; distinct from table detection.
- **Plain English:** Not just where the table is, but what is in row 2 column 3.
- **Example / analogy:** Ruling lines handle bordered tables; models such as TableFormer handle merged cells.
- **Snippet:**
```python
rows = [[ocr(cell) for cell in row] for row in grid]
```
- **Snippet explained:** One OCR call per cell of the detected grid.
- **Related terms:** [[TEDS (tree-edit-distance-based similarity)]], [[Layout analysis (page segmentation)]]

### CNN character classifier
<!-- cat: ml -->
- **Technical definition:** A small convolutional network mapping a cropped character image to one of the character classes.
- **Plain English:** Learns what an a looks like despite font and noise.
- **Example / analogy:** With augmentation a small CNN reached 0.782 on an unseen font plus rotation plus noise.
- **Snippet:**
```python
logits = cnn(x)  # (N, 62)
```
- **Snippet explained:** One score per character class.
- **Related terms:** [[CRNN (convolutional recurrent neural network)]], [[Data augmentation]]

### Data augmentation
<!-- cat: ml -->
- **Technical definition:** Training on randomly altered copies of the data (rotation, noise, scale) to improve robustness.
- **Plain English:** Practising on messy versions.
- **Example / analogy:** Augmentation took held-out accuracy from 0.468 to 0.782.
- **Snippet:**
```python
x = rotate(x, rng.uniform(-8, 8)) + noise
```
- **Snippet explained:** Each epoch sees different distortions.
- **Related terms:** [[CNN character classifier]]

### CRNN (convolutional recurrent neural network)
<!-- cat: ml -->
- **Technical definition:** A CNN feature extractor, a recurrent (BiGRU/BiLSTM) sequence model and a CTC output layer that reads a whole text line.
- **Plain English:** Looks at the line as a sequence of vertical slices.
- **Example / analogy:** A 133k-parameter CRNN reached CER 0.122 on an unseen font.
- **Snippet:**
```python
feats = cnn(img); out, _ = gru(feats)
```
- **Snippet explained:** Convolutions then bidirectional GRU produce per-frame class scores.
- **Related terms:** [[CTC (connectionist temporal classification)]], [[CNN character classifier]]

### CTC (connectionist temporal classification)
<!-- cat: ml -->
- **Technical definition:** A loss and decoding scheme that sums over all frame-level alignments that collapse (merge repeats, drop blanks) to the target text.
- **Plain English:** Lets the network read without being told where each letter is.
- **Example / analogy:** There are 15 valid alignments of ab in 4 frames.
- **Snippet:**
```python
loss = nn.CTCLoss(blank=0)(log_probs, targets, in_len, tgt_len)
```
- **Snippet explained:** Computes -log of the summed path probabilities.
- **Related terms:** [[CRNN (convolutional recurrent neural network)]], [[Prefix beam search]], [[Blank symbol (CTC)]]

### Blank symbol (CTC)
<!-- cat: ml -->
- **Technical definition:** The extra output class in CTC meaning no character here; it separates repeated letters and absorbs padding frames.
- **Plain English:** A pause between letters.
- **Example / analogy:** a-a collapses to aa but aa collapses to a.
- **Snippet:**
```python
BLANK = 0
```
- **Snippet explained:** Index 0 is the blank by convention.
- **Related terms:** [[CTC (connectionist temporal classification)]]

### Prefix beam search
<!-- cat: ml -->
- **Technical definition:** CTC decoding that keeps the best prefixes and sums the probability of all paths reaching each prefix.
- **Plain English:** A smarter decoder than taking the best frame each time.
- **Example / analogy:** It found a with probability 0.64 where greedy gave an empty string.
- **Snippet:**
```python
beams = prefix_beam_search(log_probs, width=5)
```
- **Snippet explained:** Returns the best collapsed strings.
- **Related terms:** [[CTC (connectionist temporal classification)]]

### CER and WER
<!-- cat: ml -->
- **Technical definition:** Character and word error rate: (substitutions + deletions + insertions) divided by reference length in characters or words.
- **Plain English:** How many fixes the text needs, per character or per word.
- **Example / analogy:** CER 3.4% was WER 15.4% in the module run.
- **Snippet:**
```python
cer = lev(ref, hyp) / len(ref)
```
- **Snippet explained:** Edit distance divided by reference length.
- **Related terms:** [[Levenshtein distance]], [[Field-level accuracy]]

### Levenshtein distance
<!-- cat: ml -->
- **Technical definition:** Minimum number of substitutions, deletions and insertions turning one sequence into another; O(nm) dynamic programme.
- **Plain English:** The edit count between two strings.
- **Example / analogy:** kitten to sitting is 3.
- **Snippet:**
```python
d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1] + (a[i-1] != b[j-1]))
```
- **Snippet explained:** Classic edit-distance recurrence.
- **Related terms:** [[CER and WER]]

### Field-level accuracy
<!-- cat: ml -->
- **Technical definition:** Fraction of extracted key-value fields (date, total, ID) that equal the gold values, under exact, normalised or fuzzy matching.
- **Plain English:** Did we get the numbers the business needs?
- **Example / analogy:** Fuzzy matching passed 2300.00 for 230.00; use exact for amounts.
- **Snippet:**
```python
ok = got['total'] == gold['total']
```
- **Snippet explained:** Exact string match for an amount.
- **Related terms:** [[CER and WER]], [[TEDS (tree-edit-distance-based similarity)]]

### TEDS (tree-edit-distance-based similarity)
<!-- cat: ml -->
- **Technical definition:** Table similarity: 1 minus the tree edit distance between the predicted and gold table trees, normalised by the larger tree.
- **Plain English:** A score that notices structure, not just text.
- **Example / analogy:** Merged columns scored text CER 0.000 but TEDS-like 0.645.
- **Snippet:**
```python
teds = 1 - tree_edit_distance(pred, gold) / max(size(pred), size(gold))
```
- **Snippet explained:** Normalised tree edit distance.
- **Related terms:** [[Table structure recognition]], [[CER and WER]]

### Born-digital PDF
<!-- cat: api -->
- **Technical definition:** A PDF produced by software that contains a real text layer.
- **Plain English:** Text you can select and copy.
- **Example / analogy:** Extract it directly; OCR would be slower and less accurate.
- **Snippet:**
```python
text = page.get_text()  # PyMuPDF
```
- **Snippet explained:** Reads the text layer.
- **Related terms:** [[Scanned PDF]], [[Searchable PDF (sandwich PDF)]], [[PDF content stream]]

### Scanned PDF
<!-- cat: api -->
- **Technical definition:** A PDF whose pages are images with no text layer.
- **Plain English:** A photo of a page in a PDF wrapper.
- **Example / analogy:** A text extractor returns an empty string, which a RAG index stores as no content.
- **Snippet:**
```python
if len(text) < 20 and page.get_images(): ocr(page)
```
- **Snippet explained:** Falls back to OCR when there is no text.
- **Related terms:** [[Born-digital PDF]], [[Searchable PDF (sandwich PDF)]]

### Searchable PDF (sandwich PDF)
<!-- cat: api -->
- **Technical definition:** A scan image with an invisible OCR text layer (text render mode 3) on top.
- **Plain English:** Looks like a scan, copies like text.
- **Example / analogy:** The layer is someone else's OCR: use it but sample-check it.
- **Snippet:**
```python
3 Tr  % invisible text render mode
```
- **Snippet explained:** PDF operator that makes text invisible.
- **Related terms:** [[Scanned PDF]], [[Born-digital PDF]]

### PDF content stream
<!-- cat: api -->
- **Technical definition:** The operator stream that paints one PDF page, for example BT /F1 12 Tf x y Td (text) Tj ET for text.
- **Plain English:** The drawing instructions of a page.
- **Example / analogy:** Text is a position and a string, not a paragraph.
- **Snippet:**
```python
BT /F1 16 Tf 72 400 Td (Hello) Tj ET
```
- **Snippet explained:** Begin text, set font, move, show string, end text.
- **Related terms:** [[Born-digital PDF]], [[PDF coordinate system]]

### PDF coordinate system
<!-- cat: api -->
- **Technical definition:** Units of 1/72 inch with the origin at the bottom-left and y growing upwards; images use top-left.
- **Plain English:** Flip y when mixing PDF and image coordinates.
- **Example / analogy:** A pixel at 96 dpi is 0.75 points.
- **Snippet:**
```python
y_top = page_height - y_pdf
```
- **Snippet explained:** Converts a PDF y to a top-left y.
- **Related terms:** [[PDF content stream]], [[Bounding box (bbox)]]

### Page router (native vs OCR)
<!-- cat: api -->
- **Technical definition:** Per-page decision using visible text and image coverage: extract the text layer or run OCR.
- **Plain English:** Use free text when it exists.
- **Example / analogy:** Mixed bundles need per-page routing, not per-file.
- **Snippet:**
```python
kind, how, lines = route(page)
```
- **Snippet explained:** Returns the page class, the route and the text.
- **Related terms:** [[Born-digital PDF]], [[Scanned PDF]], [[OCR (optical character recognition)]]

### VLM OCR (end-to-end document parsing)
<!-- cat: llm -->
- **Technical definition:** A vision-language model that reads a page image and generates Markdown/HTML/LaTeX: PaddleOCR-VL, olmOCR-2, DeepSeek-OCR, dots.ocr, GLM-OCR.
- **Plain English:** A model that writes the page instead of copying it.
- **Example / analogy:** Great layout handling, but it can hallucinate or skip text and has no calibrated confidence.
- **Snippet:**
```python
resp = client.chat(model=ocr_vlm, image=page_png, prompt='to Markdown')
```
- **Snippet explained:** Sends a page image to a served VLM.
- **Related terms:** [[OCR (optical character recognition)]], [[Document parsing pipeline (Docling, MinerU, marker)]]

### Document parsing pipeline (Docling, MinerU, marker)
<!-- cat: llm -->
- **Technical definition:** Toolkits chaining layout detection, reading order, table and formula models and OCR into Markdown or JSON.
- **Plain English:** A toolbox that returns structure, not just text.
- **Example / analogy:** Docling uses the PDF text layer when present and adds a table model; MinerU and marker prefer a GPU.
- **Snippet:**
```python
DocumentConverter().convert(path).document.export_to_markdown()
```
- **Snippet explained:** Docling's documented one-liner (not run in this course).
- **Related terms:** [[VLM OCR (end-to-end document parsing)]], [[Parsed document (layout elements)]]

### Parsed document (layout elements)
<!-- cat: llm -->
- **Technical definition:** A document represented as typed elements (title, section_header, text, table, ...) with page, bounding box and reading order.
- **Plain English:** The page as labelled boxes in reading order.
- **Example / analogy:** It is the OCR to chunking contract: module 19 chunks it, keeps tables whole and cites page and bbox.
- **Snippet:**
```python
doc['elements'] = sorted(els, key=lambda e: (e['page'], e['reading_order']))
```
- **Snippet explained:** Order elements the way a human reads.
- **Related terms:** [[Bounding box (bbox)]], [[Reading order]], [[Data contract (pydantic validation)]]

### Bounding box (bbox)
<!-- cat: ml -->
- **Technical definition:** A rectangle [x0, y0, x1, y1] locating an element; the unit and origin vary by tool.
- **Plain English:** Where on the page.
- **Example / analogy:** Our contract: PDF points, origin top-left.
- **Snippet:**
```python
bbox = [round(v * 0.75, 1) for v in box_px]
```
- **Snippet explained:** Pixels to points at 96 dpi.
- **Related terms:** [[PDF coordinate system]], [[Parsed document (layout elements)]]

### Data contract (pydantic validation)
<!-- cat: python -->
- **Technical definition:** A strict typed schema (pydantic BaseModel) validated where data crosses a pipeline boundary.
- **Plain English:** A schema both sides sign, checked on arrival.
- **Example / analogy:** One model accepted both our parsed_document.json and module 19's sample.
- **Snippet:**
```python
ParsedDocument.model_validate(json.loads(text))
```
- **Snippet explained:** Parses, coerces and raises ValidationError with the error path.
- **Related terms:** [[Parsed document (layout elements)]], [[Sidecar metadata file]]

### Sidecar metadata file
<!-- cat: python -->
- **Technical definition:** A companion file next to the data that holds fields the strict schema forbids, such as OCR confidence.
- **Plain English:** Notes that must not break the contract.
- **Example / analogy:** Confidence travels in parsed_document.ocr_meta.json keyed by element id.
- **Snippet:**
```python
meta[element_id] = {'confidence': 0.79}
```
- **Snippet explained:** Maps element ids to OCR confidence.
- **Related terms:** [[Data contract (pydantic validation)]]

### OmniDocBench and olmOCR-Bench
<!-- cat: llm -->
- **Technical definition:** Document parsing benchmarks: OmniDocBench scores text, formula, table and reading-order metrics; olmOCR-Bench uses pass/fail unit tests.
- **Plain English:** Scoreboards with versions: verify before quoting.
- **Example / analogy:** A leaderboard score is a lead, your own 50-page gold set is the evidence.
- **Snippet:**
```python
score = passed_tests / total_tests
```
- **Snippet explained:** olmOCR-Bench style pass rate.
- **Related terms:** [[TEDS (tree-edit-distance-based similarity)]], [[CER and WER]]
