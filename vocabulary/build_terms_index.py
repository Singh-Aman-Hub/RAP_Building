"""
GOAL : Check every step folder has a TERMS.md (definition / plain words / example for each term)
       and write TERMS_INDEX.md at the repo root listing them all with term counts.
RUN  : python vocabulary/build_terms_index.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP = re.compile(r"^\d\d_")                       # step/module folders start with two digits and underscore
rows, problems, total = [], [], 0

for module in sorted(p for p in ROOT.iterdir() if p.is_dir() and STEP.match(p.name)):
    rows.append(f"\n## {module.name}\n\n| Folder | Terms | File |\n|---|--:|---|")
    folders = sorted(p for p in module.iterdir() if p.is_dir() and STEP.match(p.name))
    if module.name.startswith("13_"):
        folders = [module]                          # capstone has one TERMS.md at module level
    for folder in folders:
        f = folder / "TERMS.md"
        if not f.exists():
            problems.append(f"missing {f.relative_to(ROOT)}")
            rows.append(f"| {folder.name} | 0 | (missing) |")
            continue
        t = f.read_text(encoding="utf-8")
        heads = len(re.findall(r"^## ", t, flags=re.M))
        d, p_, e = (t.count(x) for x in ("**1. Definition:**", "**2. In plain words:**", "**3. Example to quote:**"))
        if not (heads == d == p_ == e):
            problems.append(f"{f.relative_to(ROOT)}: headings={heads} def={d} plain={p_} example={e}")
        total += heads
        rows.append(f"| {folder.name} | {heads} | [{f.relative_to(ROOT)}]({f.relative_to(ROOT)}) |")

extra = [p for p in ROOT.rglob("TERMS.md") if p.parent.name in ("shared", "diagrams", "common", "broken_examples")]
if extra:
    rows.append("\n## Other folders\n\n| Folder | Terms | File |\n|---|--:|---|")
    for f in sorted(extra):
        t = f.read_text(encoding="utf-8"); n = len(re.findall(r"^## ", t, flags=re.M)); total += n
        rows.append(f"| {f.parent.relative_to(ROOT)} | {n} | [{f.relative_to(ROOT)}]({f.relative_to(ROOT)}) |")

head = ["# TERMS Index", "",
        f"{total} explained terms across every step folder. Each `TERMS.md` gives, per term: 1. definition, 2. plain-words paragraph, 3. an example to quote.",
        "The merged dictionary (alphabetical, cross-linked) is in [vocabulary/index.md](vocabulary/index.md)."]
(ROOT / "TERMS_INDEX.md").write_text("\n".join(head + rows) + "\n", encoding="utf-8")
print(f"{total} terms.", "Problems:" if problems else "All good.")
for p in problems: print("  ", p)
