"""Solution for exercise 07: extend the test set and add a v4 that changes exactly ONE thing vs v3.
RUN: ../../.venv/bin/python ex07_add_cases.py
Idea: import the pieces of 07_prompt_testing by running its code is awkward (folder name starts with a digit),
so we use importlib to load the file by path, then reuse TESTS / PROMPTS / parse_label.
"""
import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "07_prompt_testing" / "main.py"
spec = importlib.util.spec_from_file_location("prompt_testing", path)
mod = importlib.util.module_from_spec(spec)
with redirect_stdout(io.StringIO()):          # silence the table printed at import time
    spec.loader.exec_module(mod)

mod.TESTS.append(("ambiguous", "Great screen, terrible battery.", "neutral"))   # mixed sentiment
mod.TESTS.append(("ambiguous", "Muy bueno, me encanta.", "positive"))           # other language
mod.TESTS.append(("normal", "Excellent, I am so happy.", "positive"))

system, user = mod.PROMPTS["v3_full"]
mod.PROMPTS["v4_no_example"] = (system.split("Text: Oh great")[0], user)         # ONE change: drop the few-shot example

for name in ("v3_full", "v4_no_example"):
    rows = mod.run_version(name)
    print(f"{name:15s} {sum(r['correct'] for r in rows)}/{len(rows)}")
    for r in rows:
        if not r["correct"]:
            print("   fails:", r["category"], repr(r["text"]), "->", r["got"])
