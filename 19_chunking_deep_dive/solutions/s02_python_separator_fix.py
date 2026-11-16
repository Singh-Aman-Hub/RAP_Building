"""Step 02 exercise: the "\\n\\tdef " separator misses space-indented methods. Fix: use a regex that accepts any indentation.
RUN: HF_HUB_OFFLINE=1 ../../.venv/bin/python s02_python_separator_fix.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common19 import load_step  # noqa: E402

s2 = load_step("02_recursive_splitter_rebuilt")
code = ("class Client:\n    def __init__(self, key):\n        self.key = key\n\n    def get(self, path):\n        return self.key + path\n\n"
        "    def post(self, path, body):\n        return self.key + path + body\n")
FIXED = [r"\nclass ", r"\n[ \t]*def ", "\n\n", "\n", " ", ""]       # any indentation before def
for name, seps in (("original PYTHON", s2.PYTHON), ("fixed", FIXED)):
    pieces = s2.RecursiveSplitter(22, 0, separators=seps, strip_whitespace=False).split_text(code)
    print(f"{name}: {len(pieces)} pieces; starts: {[p.lstrip()[:12] for p in pieces]}")
