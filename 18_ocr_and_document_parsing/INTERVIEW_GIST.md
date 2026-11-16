# Module 18 — Interview gist (cram sheet)

Full explanations: [README](README.md). Numbers below are from real runs of this module (CPU, synthetic pages); your machine's timings differ.

## Key formulas
```
gray      = 0.299 R + 0.587 G + 0.114 B                       (luma; plain mean differs by up to 17 levels)
Otsu      t* = argmax_t  omega(t) (1 - omega(t)) (mu_0(t) - mu_1(t))^2     (between-class variance)
CER       = (S + D + I) / N_chars_ref        WER = (S + D + I) / N_words_ref        (can exceed 1)
Levenshtein d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1] + (a_i != b_j))
CTC       P(y|x) = sum over alignments pi with collapse(pi) = y  of  prod_t p(pi_t | x_t)     collapse = merge repeats, drop blanks
TEDS      = 1 - TreeEditDistance(T_pred, T_gold) / max(|T_pred|, |T_gold|)
pixel -> point (96 dpi) : pt = px * 0.75 ;  PDF y-flip : y_top = page_h - y_pdf
weights   : params x bytes (bf16 2, int8 1, 4-bit 0.5)   e.g. 3B bf16 = 5.6 GiB
```

## Numbers to remember (this module's runs)
- Dim page: fixed threshold 128 F1 0.098, Otsu F1 0.976. Shadow page: Otsu F1 0.194, local mean F1 0.949.
- Salt-and-pepper noise: raw F1 0.801, box blur 0.770, median 0.892. Skew +4 degrees: row-profile finds 1 line; after deskew, 4.
- Glyphs below about 8-11 px tall start to break (75 dpi: 97 of 128 glyph blobs found).
- Templates on an unseen font + rotation + noise: 0.36-0.57; CNN without augmentation 0.468, with augmentation 0.782.
- CTC: 15 valid alignments of `ab` in 4 frames; brute-force loss 1.2996 = `nn.CTCLoss`. Greedy decodes `''` where prefix beam finds `a`.
- CRNN on an unseen font: exact match 0.673 (greedy), CER 0.122; prefix beam 0.687 / 0.117.
- CER 3.4% was WER 15.4%. Swapped columns: CER 0.79 with all characters present. Merged table columns: text CER 0.000, TEDS-like 0.645.
- XY-cut order WER 0.014 versus naive y-sort 0.554 on the same recognised text. Gutter 25 px beat threshold 30.
- PDF: content stream `BT /F1 16 Tf 72 400 Td (text) Tj ET`; invisible text is render mode 3.

## 10 one-line answers
1. **Born-digital vs scanned?** Text layer means extract; image-only means OCR; image plus invisible text means verify the layer.
2. **Why Otsu?** It picks the threshold maximising between-class variance; fine for even light, wrong with shadows (use a local threshold).
3. **Why CTC?** No alignment labels needed: sum over all frame paths that collapse to the transcript, with a blank symbol.
4. **CER vs WER?** Same edit-distance formula over characters or words; WER is larger; both can exceed 1; say the normalisation.
5. **How to score a table?** TEDS or cell-level; text-only CER misses merged columns.
6. **Reading order?** Recursive XY-cut or a learned order; sort-by-y fails on columns.
7. **Why is the footer misread?** Glyph over-segmentation: the `o` split into pieces; carry confidence downstream.
8. **PyMuPDF/pypdf inside?** Parse objects, inflate streams, interpret operators, map glyphs to Unicode; PyMuPDF also renders (AGPL).
9. **Which parser?** Hard constraints first (privacy, hardware), then skills; verify benchmark versions; test on 50 of your pages.
10. **What to hand to the chunker?** Typed elements with page, bbox (points, top-left), reading order, structured tables; confidence in a sidecar.

## Confusable pairs
| Pair | Distinction |
|---|---|
| Otsu vs adaptive/local threshold | global vs per-neighbourhood threshold |
| Median vs box blur | median keeps edges and kills outliers; box blur smears both |
| Erosion/opening vs dilation/closing | shrink (remove specks) vs grow (fill gaps); opening = erode then dilate |
| Template matching vs CNN | exact pixel similarity vs learned invariance |
| Character classifier vs CRNN+CTC | needs perfect cuts vs reads a whole line |
| CER vs WER | per character vs per word |
| Text detection vs recognition | where is the text vs what does it say |
| Table detection vs structure recognition | the table's box vs its rows/columns/spans |
| Born-digital vs searchable scan | real text vs someone's invisible OCR text |
| Pixels vs points; top-left vs bottom-left origin | 0.75 pt/px at 96 dpi; flip y |
| Engine vs pipeline toolkit vs VLM OCR vs cloud API | text only / + layout, tables / one model writes the page / managed service |
| Hard filter vs weight | cannot be outvoted vs can be |
| Leaderboard score vs gold-set score | their documents vs yours |

## Failure modes to name
- Empty string from a scanned PDF indexed as "no content"; `(cid:N)` garbage from a missing ToUnicode map.
- Columns interleaved; thresholds that merge a narrow gutter; skewed scans collapsing line finding.
- VLM OCR hallucination, omission and repetition; no confidence scores.
- Fuzzy field matching hiding `230.00` vs `2300.00`; normalisation hiding real errors (`0` vs `O`).
- Cloud: documents leave the network; GPU models loaded on an 8 GB machine swap instead of run.
- Handoff: pixel boxes where points are expected, `heading` vs `section_header`, dropped page/bbox, flattened tables.
