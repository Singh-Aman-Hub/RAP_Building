# Module 18 — Exercises

Graded easy to hard. Try each on paper or in a scratch file before opening [solutions/solutions.md](solutions/solutions.md). Runnable solutions are in [solutions/](solutions/). Everything runs on CPU in seconds.

| # | Level | Topic | Needs |
|---|---|---|---|
| 1 | easy | pixels, PDF points, the y-flip | paper or Python |
| 2 | easy | CER versus WER by hand | paper or Python |
| 3 | easy | why Otsu failed on the shadowed page | [step 01 trace](01_images_as_arrays/WALKTHROUGH.md) |
| 4 | medium | CTC: collapse rule and counting alignments | Python |
| 5 | medium | XY-cut: widest gap and a threshold that is too big | numpy |
| 6 | medium | page router with a garbage-text check | Python |
| 7 | medium | a validated handoff contract (pydantic) | pydantic |
| 8 | hard | how many gold pages do I need? | Python (math) |
| 9 | hard | a deliberate, logged repair of the "cif" footer | step 07 output |

## 1. Coordinates (easy)
Step 07's title block is `[42, 31, 494, 64]` in pixels (top-left origin) on an 800 x 640 px page rendered at 96 dpi.
(a) Give the box in PDF points with a top-left origin. (b) Give it in PDF user space (bottom-left origin) for a page of height 480 pt. (c) Which of the two does module 19's sample JSON use, and what visible symptom do you get if you draw the (b) version on an image?

## 2. CER and WER by hand (easy)
Reference `the quick brown fox`, hypothesis `the quik brown fox`. Compute CER and WER. Then for `pay 1250 rupees` versus `pay 125 rupees`: why is a small CER dangerous for fields, and which metric from [step 06](06_ocr_evaluation/WALKTHROUGH.md) catches it?

## 3. Why Otsu failed on the shadow page (easy)
In [step 01](01_images_as_arrays/WALKTHROUGH.md) the shadow page gave: fixed threshold 128 F1 0.323, Otsu (t=154) F1 0.194 with recall 1.000 and precision 0.107, local mean F1 0.949. Explain in words why Otsu got recall 1.0 but terrible precision, and what the local method changes. When would you still use Otsu?

## 4. CTC (medium)
(a) Write `collapse(path)` (merge repeats, then drop blanks) and check `aa-bb`, `a-a`, `aaa`, `-a-a-`. (b) By brute force over `{a, b, blank}`, how many length-4 paths collapse to `ab`? Compare with step 05's "15". (c) What is the minimum number of frames needed to output `aa`, and why?

## 5. XY-cut gap (medium)
Implement `widest_gap(nonempty, min_gap)` and build a column-ink vector with a 25-pixel gutter. Show that `min_gap=30` finds nothing and `min_gap=20` finds the gutter. What breaks if you set `min_gap` to 3?

## 6. Router with a garbage check (medium)
Extend step 08's `classify` so a page whose visible text is mostly `(cid:N)` markers is routed to OCR instead of being called native. Test on five synthetic pages: good text, cid garbage, scan, searchable scan, blank.

## 7. A validated contract (medium)
Write a small pydantic model of an element (`type` restricted, `bbox`, `page`, `reading_order`) and a document model that rejects a bbox outside its page. Show that it rejects (a) a pixel box on a point page and (b) the type `heading`.

## 8. How many gold pages? (hard)
You score 45 of 50 pages correct (90%). Compute the 95% Wilson interval. How wide is it? How many pages would you need for a +-3 point interval at 90% accuracy? What does that say about comparing two OCR engines that differ by 2 points on 50 pages?

## 9. A logged repair (hard)
Step 07's footer reads `Page 1 cif 2` (confidence 0.79). Write a rule that repairs footers of the form `Page N <word> M` only when the block is a footer, confidence is below 0.85 and N equals the page number, logs every change, and never touches body text. Then argue in two sentences why silently fixing it in the parser would be worse.
