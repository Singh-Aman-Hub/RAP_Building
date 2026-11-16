"""Exercise 9: a deliberate, logged, rule-based repair of low-confidence page footers in step 07's output.
RUN: python s09_footer_repair.py   (needs 07_reading_order_and_tables/structured_pages.json: run that step first)"""
import json
import re
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "07_reading_order_and_tables" / "structured_pages.json"
pages = json.loads(path.read_text())["pages"]
FOOTER = re.compile(r"^Page (\d+) \S+ (\d+)$")                  # 'Page 1 cif 2'
changes = []
for pg in pages:
    for e in pg["elements"]:
        m = FOOTER.match(e["text"])
        if e["kind"] == "footer" and e["confidence"] < 0.85 and m and e["text"] != f"Page {m[1]} of {m[2]}" and int(m[1]) == pg["page"]:
            new = f"Page {m[1]} of {m[2]}"
            changes.append((pg["page"], e["text"], new, e["confidence"]))
            e["text"] = new
for c in changes:
    print("repaired page %d: %r -> %r (confidence %.2f)" % c)
assert [c[2] for c in changes] == ["Page 1 of 2", "Page 2 of 2"]
print("log every repair; never apply it to body text, where 'cif' could be a real word. ok")
