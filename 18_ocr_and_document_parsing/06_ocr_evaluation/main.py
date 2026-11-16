"""
GOAL:  Measure OCR properly: Levenshtein distance from scratch, CER and WER, why normalisation changes the score,
       field-level accuracy for key-value extraction, and a table metric in the spirit of TEDS.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 18_ocr_and_document_parsing/06_ocr_evaluation/main.py
EXPECTED OUTPUT (shortened):
    edit distance kitten -> sitting = 3 ...
    CER ... WER ...   (CER 3% can be WER 15%)
    normalisation table ...   reading-order error ...
    field accuracy: exact ... normalised ... fuzzy ...
    TEDS-like: perfect 1.00 ... flattened ...
NEEDS: Python standard library only.  Instant.
"""
import random
import re
import unicodedata


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


def lev(a, b):
    return sum(align(a, b))


def cer(ref, hyp):
    """Character error rate = (S + D + I) / number of reference characters."""
    return lev(ref, hyp) / max(len(ref), 1)


def wer(ref, hyp):
    """Word error rate: the same formula over whitespace-split tokens."""
    r, h = ref.split(), hyp.split()
    return lev(r, h) / max(len(r), 1)


print("edit distance kitten -> sitting =", lev("kitten", "sitting"), "(S, D, I) =", align("kitten", "sitting"))

# ---- 2. CER versus WER on a paragraph with realistic OCR confusions -----------------------------------------------
TEXT = ("Refunds are issued within nine days of delivery. The money is returned to the original payment method. "
        "Opened tea tins cannot be returned. Orders above 499 rupees ship free inside India. Loyalty members earn "
        "one point per ten rupees. Please contact the modern support team on 080 4000 1234 for any questions. ") * 2
CONFUSIONS = [("rn", "m"), ("m", "rn"), ("l", "1"), ("0", "O"), ("1", "l"), ("e", "c"), ("i", "j"), ("h", "b")]


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


print(f"reference has {len(TEXT)} characters and {len(TEXT.split())} words")
for rate in [0.01, 0.03, 0.08]:
    hyp = corrupt(TEXT, rate, seed=3)
    print(f"  injected edits {int(rate * len(TEXT)):3d}:  CER {cer(TEXT, hyp):.3f}   WER {wer(TEXT, hyp):.3f}")

# ---- 3. Normalisation pitfalls: the same OCR output scored three ways ---------------------------------------------------
ref = "The ﬁnal price was “$1,250.00” (inter-national)\nsee page 4"          # ligature, curly quotes, hyphen, newline
hyp = "the final  price was \"$1,250.00\" (international) see page 4"       # lower-cased, double space, ASCII quotes


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

# ---- 4. Reading order: every character right, score terrible ----------------------------------------------------------------
left = ["Refunds are issued", "within nine days", "of delivery."]            # column 1, one list item per line
right = ["Shipping inside", "India takes three", "to five days."]           # column 2
truth = " ".join(left + right)                                              # correct order: all of column 1, then column 2
swapped = " ".join(right + left)                                            # columns in the wrong order
interleaved = " ".join(x for pair in zip(left, right) for x in pair)        # line 1 of col 1, line 1 of col 2, ... (naive y-sort)
print("reading order (all characters present):")
for name, h in [("correct", truth), ("columns swapped", swapped), ("lines interleaved", interleaved)]:
    print(f"   {name:<18} CER {cer(truth, h):.3f}   WER {wer(truth, h):.3f}   same words: {sorted(h.split()) == sorted(truth.split())}")

# ---- 5. Field-level accuracy for key-value extraction -------------------------------------------------------------------------
GOLD = [{"invoice_no": "INV-4081", "date": "2026-03-12", "total": "1250.00", "vendor": "Chai Kitchen"},
        {"invoice_no": "INV-4082", "date": "2026-03-13", "total": "89.50", "vendor": "Masala Mart"},
        {"invoice_no": "INV-4083", "date": "2026-03-14", "total": "2300.00", "vendor": "Green Leaf Co"},
        {"invoice_no": "INV-4084", "date": "2026-03-15", "total": "15.00", "vendor": "Tulsi Traders"}]
PRED = [{"invoice_no": "INV-4O81", "date": "12/03/2026", "total": "1,250.00", "vendor": "Chai Kitchen"},
        {"invoice_no": "INV-4082", "date": "2026-03-13", "total": "89.50", "vendor": "Masala Mart"},
        {"invoice_no": "INV-4083", "date": "2026-03-14", "total": "230.00", "vendor": "Green Leaf Co."},
        {"invoice_no": "INV-4084", "date": "2026-03-15", "total": "15.00", "vendor": "Tulsi Traders"}]


def norm_field(key, v):
    if key == "total":
        return v.replace(",", "")                                                  # 1,250.00 == 1250.00
    if key == "date":
        m = re.fullmatch(r"(\d\d)/(\d\d)/(\d{4})", v)
        return f"{m[3]}-{m[2]}-{m[1]}" if m else v                                 # dd/mm/yyyy -> ISO
    return v.rstrip(".")


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


# ---- 6. A table metric in the spirit of TEDS (tree-edit-distance based similarity) ---------------------------------------------------------
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

# TRY IT:   add a "docs correct" count to the CER/WER section by scoring whole lines; or change the fuzzy threshold to 0.05.
# BREAK IT: in `align`, return only d[n][m] but call it as `S, D, I = align(...)`: "TypeError: cannot unpack non-iterable int";
#           or compute WER by splitting on single spaces ("a  b".split(" ")) and watch empty tokens inflate the error.
