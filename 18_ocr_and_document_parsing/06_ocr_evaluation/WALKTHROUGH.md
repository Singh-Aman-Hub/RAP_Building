# Step 06 — OCR evaluation: edit distance, CER, WER, fields, tables

## The story so far
[Step 05](../05_crnn_and_ctc/WALKTHROUGH.md) trained a CRNN and reported "exact match 0.673, CER 0.122" using a quick edit-distance function. Is that good? And how do you compare it with Tesseract, a VLM, or last month's model?

## The pain point
"95% accurate" means nothing until you say what was counted. Characters or words? Before or after lowercasing? With or without the reading order? A page can have every character right and still be useless to a RAG system, because two columns were read in the wrong order or a table lost its columns. And a CER of 3% can hide the one wrong digit in an invoice total.

## The spark
Use a small family of metrics, each answering one question: edit distance (how far is the text?), CER and WER (how often is it wrong?), normalisation rules written down before scoring, field-level accuracy (are the values the business cares about right?), and a structure-aware table score (is the grid right?).

## Goal
Implement Levenshtein distance with a backtrace, CER and WER; show how normalisation changes the numbers; show that reading-order errors destroy CER and WER while keeping every character right; compute field-level accuracy under three matching rules; and implement a TEDS-like table similarity.

## The theory (read this before the interview)
**Levenshtein (edit) distance.** The minimum number of single-item edits (substitute, delete, insert) turning the reference into the hypothesis. Dynamic programme:
```
d[i][0] = i        d[0][j] = j
d[i][j] = min( d[i-1][j] + 1,                      # delete ref[i]
               d[i][j-1] + 1,                      # insert hyp[j]
               d[i-1][j-1] + (ref[i] != hyp[j]) )  # match or substitute
```
Time and memory `O(n m)`; memory can be reduced to one row if you only need the distance. The backtrace from `d[n][m]` counts substitutions S, deletions D, insertions I.

**CER and WER.**
```
CER = (S + D + I) / N_chars_in_reference        WER = (S + D + I) / N_words_in_reference    (same formula over word tokens)
```
Properties people forget: both can exceed 100% (many insertions); a single wrong letter makes the whole word wrong, so WER is much larger than CER (CER 3.4% was WER 15.4% in this step's run); both depend on how the text is tokenised. Report which one, over which unit (page, line, whole corpus), and whether the total is a micro average (pool all errors, divide by all reference characters) or a macro average (mean of per-line scores). Micro is standard.

**Normalisation pitfalls.** Write the rules before you score, apply them to both sides, and publish them.
| Difference | Treat as error? | Notes |
|---|---|---|
| Case, extra whitespace, line breaks | usually not (for search) | but case matters for names and codes |
| Unicode forms: ligature `ﬁ` vs `fi`, full-width digits, non-breaking space, curly vs straight quotes | normalise with NFKC | `unicodedata.normalize("NFKC", s)` |
| End-of-line hyphenation `inter-\nnational` | de-hyphenate | benchmark-specific |
| `0` vs `O`, `1` vs `l` | **yes**: these are exactly the errors OCR makes | never normalise them away |
| Reading order | yes, for downstream RAG | flat CER makes it look huge; see next |
| Header, footer, page numbers | decide: some benchmarks drop them | VLMs often drop them, classic engines keep them |

**Reading order.** Text with the same characters in a different order scores a large CER/WER (0.79 for swapped columns here), and a bag-of-words metric would call it perfect. For RAG you need both: an order-sensitive text metric for chunk quality, and a layout/element metric (step 07).

**Field-level accuracy (key-value extraction, invoices, forms).** Compare extracted fields with a gold record. Three rules, from strict to lenient: exact string match; match after field-specific normalisation (dates to ISO, remove thousands separators); fuzzy match (CER under a threshold). Report per field and per document (a document is correct only if every required field is correct). **Fuzzy match can hide dangerous errors**: `230.00` against `2300.00` is only 14% CER and would pass a 15% threshold, yet it is a wrong amount. Use exact or numeric equality for amounts, IDs and dates.

**Table metrics.**
- **TEDS** (Tree-Edit-Distance-based Similarity, from PubTabNet, Zhong et al. 2020): convert both tables to HTML trees and compute `TEDS = 1 - TreeEditDistance(T_pred, T_gold) / max(|T_pred|, |T_gold|)`. A cell with text costs the normalised edit distance of its text when relabelled; inserting or deleting a node costs 1. The official implementation uses the APTED algorithm; our version is the simpler top-down (Selkow) alignment: rows are matched with rows, cells with cells, and a subtree is never moved to another level. Variants: TEDS-S (structure only, ignore text). 
- **Text-only metrics are blind to structure.** The table with two columns merged has text-only CER 0.000 (all the characters are there) and TEDS-like 0.645.
- **Cell-level** adjacency / F1 metrics (GriTS, "cell adjacency relations") are an alternative that is less sensitive to tree shape.

**Document-level benchmarks (2026).** OmniDocBench (CVPR 2025, OpenDataLab) evaluates full-page parsing: text edit distance, formula CDM, table TEDS and reading-order edit distance, across nine document types, in English and Chinese. olmOCR-Bench (Ai2) uses pass/fail unit tests ("this phrase must appear", "these two lines must be in this order"), which is less tolerant to normalisation choices. Check the benchmark version (v1.5, v1.6 ...) whenever someone quotes a score: numbers across versions are not comparable (step 09).

**What to track in practice.** A small gold set drawn from your real documents (even 50 pages), per-document-type CER/WER, field accuracy for the fields you extract, table TEDS if tables matter, and a latency/cost number. Then look at the worst 10 pages by eye.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| CER | character error rate, "normalised edit distance" | near | Edit distance divided by reference length; can exceed 1 |
| WER | word error rate | yes (same formula, word tokens) | Much larger than CER for the same errors |
| Levenshtein distance | edit distance | yes | Unit costs for substitute, delete, insert (Damerau adds transposition) |
| Accuracy (1 - CER) | character accuracy | near | "99% accurate" is CER 1%: ask which one |
| Exact match | EM, string accuracy, sequence accuracy | yes | Whole string right or wrong; used for line and field metrics |
| Field accuracy | key-value F1, "KIE F1", extraction accuracy | near | Per-field correctness after a chosen normalisation |
| TEDS | table edit distance similarity, TEDS-S (structure) | near | Tree similarity of the HTML table; 1.0 is perfect |
| Reading-order edit distance | ROED | near | Edit distance over the sequence of blocks, not characters |

## Run it
```bash
cd 18_ocr_and_document_parsing/06_ocr_evaluation
python main.py       # instant, standard library only
```

## Code, step by step

### 1. Imports  (`main.py` lines 15–17)
```python
import random
import re
import unicodedata
```
- **In simple words:** Three standard-library modules: random numbers, regular expressions, Unicode tools.
- **Exactly:** No numpy: the metrics are plain Python so you can paste them into any codebase. `random.Random(seed)` is used for reproducible corruption.
- **Why it is here:** An evaluation harness should have no heavy dependencies.

### 2. Edit distance with a backtrace  (`main.py` lines 20–47)
```python
# ---- 1. Levenshtein distance with a backtrace, from scratch -------------------------------------------------
def align(ref, hyp):
    """Dynamic programme over a (len(ref)+1) x (len(hyp)+1) table. Returns (substitutions, deletions, insertions).
    ref and hyp can be strings (characters) or lists (words)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]            # d[i][j] = edit distance between ref[:i] and hyp[:j]
    for i in range(n + 1):
        d[i][0] = i                                       # delete all i reference items
    for j in range(m + 1):
        d[0][j] = j                                       # insert all j hypothesis items
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1   # match is free, substitution costs 1
            d[i][j] = min(d[i - 1][j] + 1,                # deletion
                          d[i][j - 1] + 1,                # insertion
                          d[i - 1][j - 1] + cost)         # match / substitution
    i, j, S, D, I = n, m, 0, 0, 0                         # walk back from the corner to count the operations
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return int(S), D, I
```
- **In simple words:** Fill a table where each cell says "how many edits to turn the first i reference items into the first j hypothesis items", then walk back to count the kinds of edits.
- **Exactly:** `d` is `(n+1) x (m+1)`. The first row and column are the cost of deleting or inserting everything. The inner `min(...)` is the recurrence from the theory section. The backtrace starts at `(n, m)` and prefers diagonal moves (match or substitution), then deletion, then insertion, adding to `S`, `D`, `I`. `align` accepts strings (characters) or lists (words) because it only indexes and compares. `kitten -> sitting` gives `(S, D, I) = (2, 0, 1)`: distance 3.
- **Why it is here:** Writing this from memory is a standard interview exercise; and the S/D/I split tells you whether the model drops, invents or confuses characters.

### 3. The metrics  (`main.py` lines 50–62)
```python
def lev(a, b):
    return sum(align(a, b))


def cer(ref, hyp):
    """Character error rate = (S + D + I) / number of reference characters."""
    return lev(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    """Word error rate: the same formula over whitespace-split tokens."""
    r, h = ref.split(), hyp.split()
    return lev(r, h) / max(len(r), 1)
```
- **In simple words:** Distance divided by the length of the correct text, once over letters, once over words.
- **Exactly:** `lev` sums the three counts. `cer` divides by `len(ref)`; `wer` splits both strings on whitespace with `str.split()` (which also collapses runs of spaces) and divides by the number of reference words. `max(..., 1)` avoids division by zero on an empty reference.
- **Why it is here:** The metrics themselves are three lines; the choices around them (tokenisation, averaging) are where scores differ.

### 4. First check  (`main.py` lines 65–65)
```python
print("edit distance kitten -> sitting =", lev("kitten", "sitting"), "(S, D, I) =", align("kitten", "sitting"))
```
- **In simple words:** Test on a known example.
- **Exactly:** Prints `3` and `(2, 0, 1)`.
- **Why it is here:** Never trust an evaluation function you did not test on a case with a known answer.

### 5. A reference paragraph and plausible OCR confusions  (`main.py` lines 67–71)
```python
# ---- 2. CER versus WER on a paragraph with realistic OCR confusions -----------------------------------------------
TEXT = ("Refunds are issued within nine days of delivery. The money is returned to the original payment method. "
        "Opened tea tins cannot be returned. Orders above 499 rupees ship free inside India. Loyalty members earn "
        "one point per ten rupees. Please contact the modern support team on 080 4000 1234 for any questions. ") * 2
CONFUSIONS = [("rn", "m"), ("m", "rn"), ("l", "1"), ("0", "O"), ("1", "l"), ("e", "c"), ("i", "j"), ("h", "b")]
```
- **In simple words:** A 618-character paragraph (two copies of a 3-sentence text) and a list of mistakes OCR really makes: `rn` for `m`, `l` for `1`, `0` for `O`.
- **Exactly:** `TEXT` has 618 characters and 104 words. `CONFUSIONS` are `(wrong -> right)` pairs reused from step 04's analysis.
- **Why it is here:** The corruption is realistic, not random characters, so the WER/CER relationship is realistic.

### 6. Corrupt text at a chosen rate  (`main.py` lines 74–85)
```python
def corrupt(text, rate, seed):
    """Replace a fraction `rate` of the characters using visually plausible substitutions (and some drops)."""
    r = random.Random(seed)
    out = text
    n_edits = int(rate * len(text))
    for _ in range(n_edits):
        a, b = r.choice(CONFUSIONS)
        spots = [m.start() for m in re.finditer(re.escape(a), out)]
        if spots:
            k = r.choice(spots)
            out = out[:k] + b + out[k + len(a):]
    return out
```
- **In simple words:** Make `rate * length` random substitutions, always at a place where the pattern exists.
- **Exactly:** `re.finditer(re.escape(a), out)` finds all positions of the pattern; `r.choice(spots)` picks one; slicing replaces it. Seeded by `random.Random(seed)`.
- **Why it is here:** It simulates an OCR system with a known error rate, so we can see how CER and WER respond.

### 7. CER versus WER  (`main.py` lines 88–91)
```python
print(f"reference has {len(TEXT)} characters and {len(TEXT.split())} words")
for rate in [0.01, 0.03, 0.08]:
    hyp = corrupt(TEXT, rate, seed=3)
    print(f"  injected edits {int(rate * len(TEXT)):3d}:  CER {cer(TEXT, hyp):.3f}   WER {wer(TEXT, hyp):.3f}")
```
- **In simple words:** Print both metrics for 1%, 3% and 8% injected errors.
- **Exactly:** With 6 edits CER 0.013 and WER 0.058; with 18 edits CER 0.034 and WER 0.154; with 49 edits CER 0.071 and WER 0.317. WER is roughly 4.5 times CER here, because each wrong letter ruins a whole word of about 5 letters plus a space.
- **Why it is here:** It is the number to remember: "3% CER can mean 15% WER".

### 8. A reference and a hypothesis with formatting differences  (`main.py` lines 94–95)
```python
ref = "The ﬁnal price was “$1,250.00” (inter-national)\nsee page 4"          # ligature, curly quotes, hyphen, newline
hyp = "the final  price was \"$1,250.00\" (international) see page 4"       # lower-cased, double space, ASCII quotes
```
- **In simple words:** The reference has a ligature `ﬁ`, curly quotes, a hyphenated line break; the OCR output has plain letters, straight quotes, lower case and a double space.
- **Exactly:** After correct normalisation the two are the same text.
- **Why it is here:** Real OCR output differs from ground truth in exactly these ways.

### 9. Four ways to normalise, four scores  (`main.py` lines 98–116)
```python
def norm_whitespace(s):
    return " ".join(s.split())


def norm_nfkc(s):
    return unicodedata.normalize("NFKC", s)                                  # ligature -> 'fi', full-width -> ASCII


def norm_all(s):
    s = norm_nfkc(s).lower().replace("“", '"').replace("”", '"')
    s = re.sub(r"-\s*\n\s*|(?<=\w)-(?=\w)", "", s)                             # de-hyphenate "inter-national"
    return norm_whitespace(s)


print("normalisation (same OCR output, different pre-processing before scoring):")
for name, f in [("none", lambda s: s), ("whitespace only", norm_whitespace), ("NFKC + whitespace", lambda s: norm_whitespace(norm_nfkc(s))),
                ("everything + lowercase", norm_all)]:
    print(f"   {name:<24} CER {cer(f(ref), f(hyp)):.3f}   WER {wer(f(ref), f(hyp)):.3f}")
print("   (case-folding hides a real capitalisation error; over-normalising 0/O would hide the very errors OCR makes)")
```
- **In simple words:** Score the same pair with no clean-up, with whitespace clean-up, with Unicode clean-up, and with everything.
- **Exactly:** `unicodedata.normalize("NFKC", s)` turns the ligature into `fi`. `norm_all` also lowercases, straightens quotes and removes hyphens inside words. Result: CER 0.138 (none), 0.103, 0.068, 0.000; WER 0.444, 0.444, 0.333, 0.000.
- **Why it is here:** The same OCR output scores 0.138 or 0.000 depending on the rules. Compare systems only with identical normalisation.

### 10. Reading order errors  (`main.py` lines 119–126)
```python
left = ["Refunds are issued", "within nine days", "of delivery."]            # column 1, one list item per line
right = ["Shipping inside", "India takes three", "to five days."]           # column 2
truth = " ".join(left + right)                                              # correct order: all of column 1, then column 2
swapped = " ".join(right + left)                                            # columns in the wrong order
interleaved = " ".join(x for pair in zip(left, right) for x in pair)        # line 1 of col 1, line 1 of col 2, ... (naive y-sort)
print("reading order (all characters present):")
for name, h in [("correct", truth), ("columns swapped", swapped), ("lines interleaved", interleaved)]:
    print(f"   {name:<18} CER {cer(truth, h):.3f}   WER {wer(truth, h):.3f}   same words: {sorted(h.split()) == sorted(truth.split())}")
```
- **In simple words:** Same words, different order: columns swapped, or lines interleaved the way a naive top-to-bottom sort would.
- **Exactly:** `interleaved` zips line 1 of column 1 with line 1 of column 2, and so on. Swapped columns: CER 0.792, WER 1.000. Interleaved: CER 0.469, WER 0.500. In both cases `sorted(words)` is the same as the truth.
- **Why it is here:** It is the metric-level proof that reading order matters, and a preview of step 07.

### 11. Gold and predicted invoice fields  (`main.py` lines 129–137)
```python
GOLD = [{"invoice_no": "INV-4081", "date": "2026-03-12", "total": "1250.00", "vendor": "Chai Kitchen"},
        {"invoice_no": "INV-4082", "date": "2026-03-13", "total": "89.50", "vendor": "Masala Mart"},
        {"invoice_no": "INV-4083", "date": "2026-03-14", "total": "2300.00", "vendor": "Green Leaf Co"},
        {"invoice_no": "INV-4084", "date": "2026-03-15", "total": "15.00", "vendor": "Tulsi Traders"}]
PRED = [{"invoice_no": "INV-4O81", "date": "12/03/2026", "total": "1,250.00", "vendor": "Chai Kitchen"},
        {"invoice_no": "INV-4082", "date": "2026-03-13", "total": "89.50", "vendor": "Masala Mart"},
        {"invoice_no": "INV-4083", "date": "2026-03-14", "total": "230.00", "vendor": "Green Leaf Co."},
        {"invoice_no": "INV-4084", "date": "2026-03-15", "total": "15.00", "vendor": "Tulsi Traders"}]

```
- **In simple words:** Four invoices with four fields each, and an extraction result with realistic mistakes.
- **Exactly:** Errors: `INV-4O81` (letter O for zero), date written `12/03/2026` (correct date, different format), `1,250.00` (comma), `230.00` instead of `2300.00` (a lost digit) and `Green Leaf Co.` (trailing dot).
- **Why it is here:** It mixes harmless format differences with a harmful value error.

### 12. Field-specific normalisation  (`main.py` lines 139–146)
```python
def norm_field(key, v):
    if key == "total":
        return v.replace(",", "")                                                  # 1,250.00 == 1250.00
    if key == "date":
        m = re.fullmatch(r"(\d\d)/(\d\d)/(\d{4})", v)
        return f"{m[3]}-{m[2]}-{m[1]}" if m else v                                 # dd/mm/yyyy -> ISO
    return v.rstrip(".")

```
- **In simple words:** Teach the evaluator that `1,250.00` equals `1250.00` and that `12/03/2026` is `2026-03-12`.
- **Exactly:** `re.fullmatch(r"(\d\d)/(\d\d)/(\d{4})", v)` captures day, month, year; `m[3]` is the third group. Vendor names lose a trailing dot.
- **Why it is here:** Field normalisation is part of the metric definition, not an afterthought.

### 13. Three matching rules, per field and per document  (`main.py` lines 148–162)
```python
fields = list(GOLD[0])
for scheme in ["exact", "normalised", "fuzzy (CER <= 0.15)"]:
    row = []
    for k in fields:
        ok = 0
        for g, p in zip(GOLD, PRED):
            a, b = g[k], p[k]
            if scheme != "exact":
                b = norm_field(k, b)
            ok += (a == b) if scheme != "fuzzy (CER <= 0.15)" else (cer(a, b) <= 0.15)
        row.append(ok / len(GOLD))
    docs_ok = sum(all((g[k] == norm_field(k, p[k])) if scheme == "normalised" else (g[k] == p[k]) if scheme == "exact"
                      else cer(g[k], norm_field(k, p[k])) <= 0.15 for k in fields) for g, p in zip(GOLD, PRED))
    print(f"field accuracy, {scheme:<20}", {k: round(v, 2) for k, v in zip(fields, row)}, f"| whole documents correct {docs_ok}/{len(GOLD)}")

```
- **In simple words:** Score every field exactly, after normalisation, and fuzzily, and count the documents where all fields are right.
- **Exactly:** Exact: invoice 0.75, date 0.75, total 0.50, vendor 0.75, 2 of 4 documents. Normalised: date and vendor 1.0, invoice 0.75, total 0.75; still 2 of 4. Fuzzy (CER <= 0.15): everything 1.0, 4 of 4 documents, yet it accepts `230.00` for `2300.00` and `INV-4O81` for `INV-4081`.
- **Why it is here:** It shows the lenient metric scoring a wrong amount as correct.

### 14. A tree edit distance for tables  (`main.py` lines 165–195)
```python
def tree(table):
    """('table', [('tr', [('td', text), ...]), ...]) with one tuple per node."""
    return ("table", [("tr", [("td", c) for c in row]) for row in table])


def size(t):
    return 1 + (sum(size(c) for c in t[1]) if isinstance(t[1], list) else 0)


def tree_dist(a, b):
    """Top-down ordered tree edit distance (Selkow): relabel the roots, then align the child sequences.
    Leaf cells cost the normalised edit distance of their text. Deleting or inserting a subtree costs its size."""
    if not isinstance(a[1], list) or not isinstance(b[1], list):                       # two cells
        if isinstance(a[1], list) or isinstance(b[1], list):
            return max(size(a), size(b))
        return lev(a[1], b[1]) / max(len(a[1]), len(b[1]), 1)
    ca, cb = a[1], b[1]
    d = [[0.0] * (len(cb) + 1) for _ in range(len(ca) + 1)]
    for i in range(1, len(ca) + 1):
        d[i][0] = d[i - 1][0] + size(ca[i - 1])
    for j in range(1, len(cb) + 1):
        d[0][j] = d[0][j - 1] + size(cb[j - 1])
    for i in range(1, len(ca) + 1):
        for j in range(1, len(cb) + 1):
            d[i][j] = min(d[i - 1][j] + size(ca[i - 1]), d[i][j - 1] + size(cb[j - 1]),
                          d[i - 1][j - 1] + tree_dist(ca[i - 1], cb[j - 1]))
    return d[-1][-1] + (0 if a[0] == b[0] else 1)


def teds(gold, pred):
    a, b = tree(gold), tree(pred)
```
- **In simple words:** Treat a table as a tree (table, rows, cells). The distance is how many cells, rows and characters you must change to turn one tree into the other.
- **Exactly:** `tree` builds `("table", [("tr", [("td", text), ...]), ...])`. `size` counts nodes. `tree_dist` is the Selkow top-down distance: two cells cost the normalised edit distance of their texts; for two row-or-table nodes it aligns the child lists with the same edit-distance table as step 1, where deleting or inserting a child costs its subtree size and substituting costs the distance of the two children; add 1 if the labels differ. `teds = 1 - distance / max(size_gold, size_pred)`, floored at 0. This is not APTED, so values can differ from the official TEDS slightly, but the behaviour is the same.
- **Why it is here:** It turns "is the grid right?" into one number that also gives partial credit.

### 15. Score five table variants  (`main.py` lines 196–210)
```python
    return max(0.0, 1 - tree_dist(a, b) / max(size(a), size(b)))                       # 1.0 = identical, floored at 0


G = [["Product", "Price", "Stock"], ["Masala Chai", "250", "120"], ["Green Tea", "320", "75"]]
cases = {
    "perfect": G,
    "one cell misread (250 -> 2S0)": [G[0], ["Masala Chai", "2S0", "120"], G[2]],
    "last row missing": G[:2],
    "two columns merged": [["Product", "Price Stock"], ["Masala Chai", "250 120"], ["Green Tea", "320 75"]],
    "flattened to one row": [[c for row in G for c in row]],
}
print("TEDS-like similarity to the gold table (1.0 = identical):")
for name, t in cases.items():
    flat_cer = cer(" ".join(c for r in G for c in r), " ".join(c for r in t for c in r))
    print(f"   {name:<32} TEDS-like {teds(G, t):.3f}   text-only CER {flat_cer:.3f}")
```
- **In simple words:** Break a 3x3 table in five different ways and compare TEDS-like with a flat text score.
- **Exactly:** Perfect 1.000. One cell misread (`250 -> 2S0`) 0.974 (CER 0.018). Last row missing 0.692 (CER 0.304). Two columns merged 0.645 while text-only CER is 0.000. Everything flattened into a single row 0.000, CER 0.000.
- **Why it is here:** Text metrics miss the two failures RAG suffers from most: lost columns and lost rows.

### 16. TRY IT and BREAK IT notes  (`main.py` lines 212–214)
```python
# TRY IT:   add a "docs correct" count to the CER/WER section by scoring whole lines; or change the fuzzy threshold to 0.05.
# BREAK IT: in `align`, return only d[n][m] but call it as `S, D, I = align(...)`: "TypeError: cannot unpack non-iterable int";
#           or compute WER by splitting on single spaces ("a  b".split(" ")) and watch empty tokens inflate the error.
```
- **In simple words:** Experiments for you.
- **Exactly:** Comments only.
- **Why it is here:** Practice.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `[[0] * (m + 1) for _ in range(n + 1)]` | a 2-D table (never `[[0]*m]*n`, which repeats one row object) | `d` |
| `S += a != b` | `True` adds 1 | substitution count |
| `m[3]` | the 3rd regex group of a match object | `norm_field` |
| `(?<=\w)-(?=\w)` | regex: a hyphen between two word characters | de-hyphenation |
| `{k: round(v, 2) for k, v in zip(...)}` | dict comprehension | per-field scores |
| `f"{name:<32}"` | pad to 32 columns | tables |
| `isinstance(x, list)` | type check; leaf cells hold strings, inner nodes lists | `tree_dist` |
| `max(0.0, x)` | floor at zero | `teds` |

## Libraries & functions
| Name | Library | What it does | Important args | Returns |
|---|---|---|---|---|
| `unicodedata.normalize(form, s)` | stdlib | Unicode normalisation | `"NFKC"` folds compatibility characters (ligatures, full-width) | `str` |
| `re.sub(pattern, repl, s)` | stdlib | regex replace | | `str` |
| `re.fullmatch(pattern, s)` | stdlib | match the whole string | | match or `None` |
| `re.finditer(p, s)` | stdlib | iterate matches | `m.start()` | iterator |
| `str.split()` vs `str.split(" ")` | stdlib | whitespace-run split vs single-space split | | `list[str]` |
| `random.Random(seed).choice(seq)` | stdlib | seeded random pick | | item |
| Real tools | `jiwer`, `rapidfuzz`, `editdistance`; `table-recognition-metric` (TEDS) | production metric libraries | | not installed here |

## Execution trace (real run)
```
edit distance kitten -> sitting = 3 (S, D, I) = (2, 0, 1)
reference has 618 characters and 104 words
  injected edits   6:  CER 0.013   WER 0.058
  injected edits  18:  CER 0.034   WER 0.154
  injected edits  49:  CER 0.071   WER 0.317
normalisation (same OCR output, different pre-processing before scoring):
   none                     CER 0.138   WER 0.444
   whitespace only          CER 0.103   WER 0.444
   NFKC + whitespace        CER 0.068   WER 0.333
   everything + lowercase   CER 0.000   WER 0.000
   (case-folding hides a real capitalisation error; over-normalising 0/O would hide the very errors OCR makes)
reading order (all characters present):
   correct            CER 0.000   WER 0.000   same words: True
   columns swapped    CER 0.792   WER 1.000   same words: True
   lines interleaved  CER 0.469   WER 0.500   same words: True
field accuracy, exact                {'invoice_no': 0.75, 'date': 0.75, 'total': 0.5, 'vendor': 0.75} | whole documents correct 2/4
field accuracy, normalised           {'invoice_no': 0.75, 'date': 1.0, 'total': 0.75, 'vendor': 1.0} | whole documents correct 2/4
field accuracy, fuzzy (CER <= 0.15)  {'invoice_no': 1.0, 'date': 1.0, 'total': 1.0, 'vendor': 1.0} | whole documents correct 4/4
TEDS-like similarity to the gold table (1.0 = identical):
   perfect                          TEDS-like 1.000   text-only CER 0.000
   one cell misread (250 -> 2S0)    TEDS-like 0.974   text-only CER 0.018
   last row missing                 TEDS-like 0.692   text-only CER 0.304
   two columns merged               TEDS-like 0.645   text-only CER 0.000
   flattened to one row             TEDS-like 0.000   text-only CER 0.000
```
- **kitten:** 3 edits: two substitutions and one insertion.
- **CER vs WER:** 6 edits -> CER 1.3%, WER 5.8%; 49 edits -> CER 7.1%, WER 31.7%.
- **Normalisation:** one OCR output, CER from 0.138 down to 0.000 just by changing the evaluation rules.
- **Reading order:** the same characters, CER 0.792 for swapped columns.
- **Fields:** the lenient fuzzy rule says 4/4 documents are fine; exact says 2/4. The truth: 2/4 documents have a real error (`INV-4O81`, `230.00`), as normalised matching reports.
- **Tables:** missing row 0.692, merged columns 0.645 (CER 0.000), flattened 0.000.

## Experiments (change ONE thing)
1. Fuzzy threshold 0.15 -> 0.05: the fuzzy rule now rejects the lost digit (`230.00`) and `INV-4O81`: invoice 0.75, total 0.75, 2 of 4 documents. A tight threshold is what protects against dangerous near-misses; for amounts, use equality.
2. WER by splitting on single spaces (`ref.split(" ")`) instead of `split()`: the normalisation row "none" jumps from WER 0.444 to 0.750 because the double space produces an empty token and the newline stays glued to a word. Tokenisation is part of the metric.
3. Not run, for you: replace the `corrupt` function with random insertions and deletions and compare CER and WER growth.

## Debugging practice
BREAK IT: make `align` return only the distance (`return d[n][m]`) while `lev` still does `sum(align(a, b))`:
```
  File ".../main.py", line 51, in lev
    return sum(align(a, b))
           ^^^^^^^^^^^^^^^^
TypeError: 'int' object is not iterable
```
Read it: `sum(...)` was handed a single integer, not the `(S, D, I)` tuple. Return types are part of a function's contract: when you change one, search all callers (`grep -n "align("`).

## Rebuild checklist
- [ ] Levenshtein with a 2-D table and a backtrace into S, D, I
- [ ] CER, WER and the reason WER is larger
- [ ] NFKC, whitespace, case, hyphenation: say which you apply before scoring
- [ ] why reading order needs its own metric
- [ ] field accuracy with exact / normalised / fuzzy, and why fuzzy is dangerous for amounts
- [ ] TEDS: tree edit distance over rows and cells, 1 - distance / max size

## The new problem (cliffhanger)
We can now say how wrong a transcript is. But a real page has two columns, a title, a table and a footer, and reading it top to bottom, left to right produces the interleaved text that scored CER 0.469 above. The page needs a reading order, and the table needs a grid. Next: [step 07, reading order and tables](../07_reading_order_and_tables/WALKTHROUGH.md).

> **Say it aloud (interview box):** CER and WER are edit distance divided by the reference length over characters or words. The distance is a dynamic programme with substitution, deletion and insertion cost 1. WER is several times CER, since one wrong letter ruins a word. I fix and publish the normalisation rules first (Unicode NFKC, whitespace, case) because they move the score as much as the model does. For documents I add field-level accuracy with exact matching on amounts and IDs, an order-aware metric for reading order, and TEDS for tables, since text-only metrics cannot see lost columns.
