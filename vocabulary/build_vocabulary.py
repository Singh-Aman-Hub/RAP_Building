"""
GOAL   : Merge the per-module term drafts in vocabulary/_drafts/ into the four searchable
         dictionary files + index.md.

RUN    : python vocabulary/build_vocabulary.py          (from the repo root, any cwd works)

INPUT  : vocabulary/_drafts/*.md   -- entries shaped like:
             ### Term name
             <!-- cat: python|ml|llm|api -->
             - **Technical definition:** ...
OUTPUT : vocabulary/python_syntax.md, ml_terms.md, llm_terms.md, api_and_deployment_terms.md, index.md

You can also add your own terms: write them in any draft file (or a new one, e.g. _drafts/mine.md)
using the same format and re-run this script.  Existing outputs are overwritten, so always edit the drafts.
"""

import re
from collections import defaultdict
from pathlib import Path

VOCAB_DIR = Path(__file__).resolve().parent          # .../vocabulary
DRAFT_DIR = VOCAB_DIR / "_drafts"

# category tag -> (output file name, title shown at the top of the file)
CATEGORIES = {
    "python": ("python_syntax.md", "Python Syntax, Standard Library, Debugging and DSA Terms"),
    "ml": ("ml_terms.md", "Machine Learning Terms"),
    "llm": ("llm_terms.md", "LLM, Prompting and Agent Terms"),
    "api": ("api_and_deployment_terms.md", "API, Docker, Architecture and Deployment Terms"),
}

ENTRY_START = re.compile(r"^### (.+?)\s*$")                 # a line like "### Temperature (sampling)"
CAT_TAG = re.compile(r"<!--\s*cat:\s*(\w+)\s*-->")          # the category comment


def slugify(term: str) -> str:
    """GitHub-style anchor: lowercase, drop punctuation, spaces -> hyphens."""
    slug = re.sub(r"[^\w\s-]", "", term.lower())
    return re.sub(r"\s+", "-", slug.strip())


def parse_drafts() -> list[dict]:
    """Return a list of {term, cat, body, source} dicts from every draft file."""
    entries = []
    for draft in sorted(DRAFT_DIR.glob("*.md")):
        current = None
        in_code = False                                      # '### ' inside a code block is not a heading
        for line in draft.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("```"):
                in_code = not in_code
            match = None if in_code else ENTRY_START.match(line)
            if match:
                current = {"term": match.group(1), "cat": None, "lines": [], "source": draft.stem}
                entries.append(current)
            elif current is not None:
                tag = CAT_TAG.search(line)
                if tag and current["cat"] is None and not in_code:
                    current["cat"] = tag.group(1).lower()
                else:
                    current["lines"].append(line)
    for e in entries:
        e["body"] = "\n".join(e["lines"]).strip()
    return entries


def main() -> None:
    entries = parse_drafts()
    by_cat = defaultdict(list)
    seen = {}
    problems = []
    for e in entries:
        if e["cat"] not in CATEGORIES:
            problems.append(f"{e['source']}: '{e['term']}' has missing/unknown category {e['cat']!r} -> put in python")
            e["cat"] = "python"
        key = e["term"].lower()
        if key in seen:                                      # duplicate term names: keep the first one
            problems.append(f"duplicate term '{e['term']}' ({seen[key]} and {e['source']}) -> kept the first")
            continue
        seen[key] = e["source"]
        by_cat[e["cat"]].append(e)

    # term -> link target, used to turn [[Wiki links]] into clickable markdown links
    target = {t["term"].lower(): (CATEGORIES[c][0], t["term"]) for c, ts in by_cat.items() for t in ts}

    def linkify(body: str) -> str:
        def repl(m):
            name = m.group(1)
            if name.lower() in target:
                fname, real = target[name.lower()]
                return f"[{name}]({fname}#{slugify(real)})"
            return name                                      # unknown term: leave plain text
        return re.sub(r"\[\[(.+?)\]\]", repl, body)

    index_lines = ["# Vocabulary Index", "",
                   f"{len(seen)} terms. Use your editor's search (Cmd+F / Cmd+Shift+F) for a term, "
                   "or click a link below.", ""]
    for cat, (fname, title) in CATEGORIES.items():
        terms = sorted(by_cat[cat], key=lambda x: x["term"].lower())
        out = [f"# {title}", "", f"{len(terms)} terms, alphabetical. Back to [index](index.md).", ""]
        # mini table of contents at the top of each file
        out += [f"- [{t['term']}](#{slugify(t['term'])})" for t in terms] + ["", "---", ""]
        for t in terms:
            out += [f"### {t['term']}", f"*From module: {t['source']}*", "", linkify(t["body"]), "", "---", ""]
        (VOCAB_DIR / fname).write_text("\n".join(out), encoding="utf-8")

        index_lines += [f"## {title}  ({len(terms)})", ""]
        index_lines += [f"- [{t['term']}]({fname}#{slugify(t['term'])})" for t in terms] + [""]

    (VOCAB_DIR / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    # [[wiki links]] check: warn about links that do not point to a known term
    known = set(seen)
    for e in entries:
        for link in re.findall(r"\[\[(.+?)\]\]", re.sub(r"```.*?```", "", e["body"], flags=re.S)):
            if link.lower() not in known:
                problems.append(f"{e['source']}: '{e['term']}' links to unknown term [[{link}]]")

    print(f"Wrote {len(seen)} terms.")
    for p in problems:
        print("  WARN:", p)


if __name__ == "__main__":
    main()
