"""Exercise 6: extend the step-08 page classifier with a garbage-text check, standalone and testable.
RUN: python s06_router_with_garbage_check.py"""
import re


def garbage_ratio(text):
    """Fraction of the characters that sit inside '(cid:N)' markers or are U+FFFD / control characters."""
    if not text:
        return 0.0
    bad = sum(len(m) for m in re.findall(r"\(cid:\d+\)", text)) + sum(c == "�" or (ord(c) < 32 and c not in "\n\t") for c in text)
    return bad / len(text)


def classify(page, min_chars=20, min_cover=0.5, max_garbage=0.2):
    text = " ".join(s["text"] for s in page["spans"] if s["mode"] != 3)
    hidden = sum(len(s["text"]) for s in page["spans"] if s["mode"] == 3)
    cover = page.get("image_cover", 0.0)
    if len(text) >= min_chars:
        return "native" if garbage_ratio(text) <= max_garbage else "garbled_text_layer -> OCR"
    if hidden >= min_chars and cover >= min_cover:
        return "searchable_scan"
    if cover >= min_cover:
        return "scanned"
    return "empty_or_vector"


good = dict(spans=[dict(text="Refunds are issued within nine days.", mode=0)], image_cover=0.0)
cid = dict(spans=[dict(text="(cid:12)(cid:45)(cid:7)(cid:99) (cid:3)(cid:8)(cid:21)(cid:64) ok", mode=0)], image_cover=0.0)
scan = dict(spans=[], image_cover=1.0)
sandwich = dict(spans=[dict(text="x" * 40, mode=3)], image_cover=1.0)
for name, p in [("good", good), ("cid garbage", cid), ("scan", scan), ("searchable scan", sandwich), ("blank", dict(spans=[], image_cover=0.0))]:
    print(f"{name:<16} -> {classify(p)}")
assert classify(good) == "native" and classify(cid).startswith("garbled") and classify(scan) == "scanned" and classify(sandwich) == "searchable_scan"
print("ok")
