# Module 18 — Solutions

Runnable code for most answers is in this folder (`python sNN_*.py`, each verified by running it). Back to [exercises](../exercises.md) · [README](../README.md).

## 1. Coordinates — [s01_coordinates.py](s01_coordinates.py)
(a) 96-dpi pixels times 0.75 give points: `[31.5, 23.25, 370.5, 48.0]`. (b) With a bottom-left origin the y values flip: `y_low = 480 - 48 = 432`, `y_high = 480 - 23.25 = 456.75`, so `[31.5, 432.0, 370.5, 456.75]`. (c) Module 19's sample (and our `parsed_document.json`) use **top-left points**, as in (a). Drawing the (b) box on an image puts every box vertically mirrored (title at the bottom), the classic y-flip symptom.

## 2. CER and WER — [s02_cer_wer_by_hand.py](s02_cer_wer_by_hand.py)
Reference has 19 characters, one deletion (`quick` to `quik`): CER = 1/19 = 0.0526. Four words, one wrong: WER = 0.25. For `1250` versus `125`: CER 0.067 (1 of 15 characters) but the amount is wrong by a factor of ten. CER alone hides it; **field-level accuracy with exact numeric equality** ([step 06](../06_ocr_evaluation/WALKTHROUGH.md)) catches it, and fuzzy matching with a CER threshold would let it through.

## 3. Otsu on the shadow page (no code; numbers from step 01)
Otsu assumes a roughly two-peaked global histogram (ink versus paper). A shadow adds a third population (paper in shadow) that is darker than the lit paper and often darker than Otsu's chosen threshold (154): the whole shadowed area is classified as ink. Recall is 1.000 (all true ink is below 154) but precision is 0.107 (about nine of ten "ink" pixels are shadowed paper). A **local** threshold compares each pixel with the mean of its 31 x 31 neighbourhood (minus an offset), so paper in shadow is compared with other shadowed paper and stays background: F1 0.949. Otsu is still right for evenly lit pages (the dim page: F1 0.976), it is cheap, and it is the right default when lighting is controlled (flatbed scans).

## 4. CTC — [s04_ctc_alignments.py](s04_ctc_alignments.py)
(a) Merge repeats first, then remove blanks: `aa-bb` -> `ab`, `a-a` -> `aa`, `aaa` -> `a`, `-a-a-` -> `aa`. (b) 15 of the 81 paths collapse to `ab` (matches step 05). (c) Three frames (`a-a`): a blank must separate two identical letters, otherwise the repeat-merge rule fuses them.

## 5. XY-cut gap — [s05_widest_gap_threshold.py](s05_widest_gap_threshold.py)
With a 25 px gutter, `min_gap=30` returns `None` (columns stay merged) and `min_gap=20` returns the band `(240, 265)`. With `min_gap=3` every word gap (4-5 px) becomes a cut: the page shatters into one block per word. The threshold must lie between the largest within-line gap and the narrowest real gutter, so measure both on your corpus.

## 6. Router with a garbage check — [s06_router_with_garbage_check.py](s06_router_with_garbage_check.py)
`garbage_ratio` counts characters inside `(cid:N)` markers plus U+FFFD and control characters; above 20% the page is labelled `garbled_text_layer -> OCR`. The five test pages give `native`, `garbled...`, `scanned`, `searchable_scan`, `empty_or_vector`. Production versions also check dictionary-word ratio and unusual-character ratios.

## 7. Contract — [s07_contract_validator.py](s07_contract_validator.py)
A `model_validator(mode="after")` on the document compares every element's bbox with its page size. The pixel box `[42, 31, 794, 64]` is rejected (x1 = 794 > 600 points), and `heading` is rejected by the `Literal` (the error lists the allowed values). Validate at the border so the failure names the element.

## 8. Gold set size — [s08_how_many_pages.py](s08_how_many_pages.py)
45/50 gives a Wilson 95% interval of about [0.786, 0.957], width 0.17. For +-3 points at 90% you need about 384 pages. So two engines that differ by 2 points on 50 pages are statistically indistinguishable; compare them on the **same** pages (paired differences, or the worst-10 pages by eye) and use more pages for close calls.

## 9. Logged repair — [s09_footer_repair.py](s09_footer_repair.py)
The rule only fires for `kind == "footer"`, confidence below 0.85, a matching pattern `Page N <word> M` with N equal to the page number, and a text different from the correct form; both footers are repaired (`Page 1 of 2`, `Page 2 of 2`) and each change is printed. Silently fixing in the parser would make the OCR look better than it is, hide a systematic glyph-segmentation bug, and risk "repairing" real text; a logged, narrow rule (or a second engine for low-confidence blocks) keeps the error visible and the fix auditable.
