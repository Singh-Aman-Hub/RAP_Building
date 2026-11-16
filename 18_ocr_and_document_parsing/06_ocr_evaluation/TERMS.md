# Terms — OCR evaluation

Folder: `18_ocr_and_document_parsing/06_ocr_evaluation` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [18 README](../README.md)

**Terms in this folder:** [Levenshtein (edit) distance](#levenshtein-edit-distance) · [Character error rate (CER)](#character-error-rate-cer) · [Word error rate (WER)](#word-error-rate-wer) · [Substitutions, deletions, insertions (S, D, I)](#substitutions-deletions-insertions-s-d-i) · [Text normalisation before scoring](#text-normalisation-before-scoring) · [NFKC normalisation](#nfkc-normalisation) · [Reading-order error](#reading-order-error) · [Field-level accuracy (key-value extraction)](#field-level-accuracy-key-value-extraction) · [TEDS (tree-edit-distance-based similarity)](#teds-tree-edit-distance-based-similarity) · [Selkow top-down tree distance](#selkow-top-down-tree-distance) · [OmniDocBench and olmOCR-Bench](#omnidocbench-and-olmocr-bench) · [TypeError: 'int' object is not iterable](#typeerror-int-object-is-not-iterable)

---

## Levenshtein (edit) distance  `(concept)`
**1. Definition:** The minimum number of single-item substitutions, deletions and insertions that turn the reference into the hypothesis, computed by dynamic programming in O(n*m).
**2. In plain words:** How many fixes it takes to correct the text: kitten to sitting is 3.
**3. Example to quote:** > "I can write the 2-D table with the recurrence min(delete, insert, match-or-substitute) from memory and backtrace S, D, I."
```python
d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + (ref[i-1] != hyp[j-1]))
```
**Seen in:** `main.py`, `align`

## Character error rate (CER)  `(concept)`
**1. Definition:** (S + D + I) divided by the number of reference characters; it can exceed 1.
**2. In plain words:** Wrong letters per letter. 49 injected edits in 618 characters gave CER 0.071.
**3. Example to quote:** > "Report whether it is micro (pooled) or macro (per-line mean) averaged."
**Seen in:** `main.py`, `cer`

## Word error rate (WER)  `(concept)`
**1. Definition:** The same formula over whitespace-split word tokens.
**2. In plain words:** One wrong letter ruins a word, so WER was about 4.5 times CER here (0.034 vs 0.154).
**3. Example to quote:** > "3% CER can mean 15% WER."
**Seen in:** `main.py`, `wer`

## Substitutions, deletions, insertions (S, D, I)  `(concept)`
**1. Definition:** The three edit operation counts recovered by backtracing the edit-distance table.
**2. In plain words:** Confused, dropped and invented items.
**3. Example to quote:** > "A high D means the model skips text (common for VLMs dropping headers); a high I means invented text."
**Seen in:** `main.py`, `align`

## Text normalisation before scoring  `(concept)`
**1. Definition:** Rules applied to both reference and hypothesis before comparing: Unicode form, case, whitespace, hyphenation.
**2. In plain words:** The same OCR output scored CER 0.138 with no rules and 0.000 with all of them.
**3. Example to quote:** > "I fix and publish the normalisation rules before comparing systems."
**Seen in:** `main.py`, section 3

## NFKC normalisation  `(syntax)`
**1. Definition:** `unicodedata.normalize("NFKC", s)` replaces compatibility characters (ligatures such as the fi ligature, full-width forms, non-breaking spaces) by their plain equivalents.
**2. In plain words:** Makes the ligature and the two letters compare equal.
**3. Example to quote:** > "NFKC is lossy by design; do not use it on data where the distinction matters."
**Seen in:** `main.py`, `norm_nfkc`

## Reading-order error  `(concept)`
**1. Definition:** Correct characters in the wrong sequence, which flat CER/WER punishes heavily (swapped columns CER 0.792).
**2. In plain words:** Right words, wrong story.
**3. Example to quote:** > "Bag-of-words metrics cannot see it; order-aware metrics can."
**Seen in:** `main.py`, section 4

## Field-level accuracy (key-value extraction)  `(concept)`
**1. Definition:** The fraction of gold fields whose extracted value matches under a stated rule (exact, normalised, fuzzy), reported per field and per document.
**2. In plain words:** Did we get the invoice number, date, total and vendor right?
**3. Example to quote:** > "Fuzzy matching accepted 230.00 for 2300.00; for amounts and IDs I use exact or numeric equality."
**Seen in:** `main.py`, section 5

## TEDS (tree-edit-distance-based similarity)  `(concept)`
**1. Definition:** 1 - TreeEditDistance(pred, gold) / max(nodes), over HTML-like table trees, with text cost by normalised edit distance (PubTabNet metric, APTED in the official code).
**2. In plain words:** One score for "is the grid right and are the cells right".
**3. Example to quote:** > "Merged columns had text-only CER 0.000 but TEDS-like 0.645."
**Seen in:** `main.py`, `teds`

## Selkow top-down tree distance  `(concept)`
**1. Definition:** An ordered tree edit distance that aligns children lists level by level and never moves a subtree to a different level.
**2. In plain words:** Rows match rows, cells match cells; simple and exact for tables.
**3. Example to quote:** > "It is not APTED, so scores differ slightly from official TEDS."
**Seen in:** `main.py`, `tree_dist`

## OmniDocBench and olmOCR-Bench  `(concept)`
**1. Definition:** Public document-parsing benchmarks: OmniDocBench scores text, formula, table and reading order; olmOCR-Bench uses pass/fail unit tests.
**2. In plain words:** Page-level benchmarks; always quote the version (v1.5, v1.6).
**3. Example to quote:** > "Scores across benchmark versions are not comparable."
**Seen in:** `WALKTHROUGH.md`, theory

## TypeError: 'int' object is not iterable  `(error)`
**1. Definition:** Raised when `sum()` receives a single integer instead of the (S, D, I) tuple.
**2. In plain words:** A function's return type changed but a caller was not updated.
**3. Example to quote:** > "I grep all callers when changing a return type."
**Seen in:** BREAK IT in `WALKTHROUGH.md`
