"""
GOAL:  Answer "WHICH Python am I running, and WHICH packages can it see?" -- the first question for any import error.

RUN:   cd 01_debugging_and_environment/03_import_and_environment && python main.py
       Crashing versions (EXPECTED):  python broken_import.py 1|2|3

EXPECTED OUTPUT: interpreter path, version, package versions, first sys.path entries.

NEEDS: nothing (standard library only), no internet.
"""

import sys
import platform
import importlib.metadata as md          # read installed-package versions
import importlib.util                    # check if a module can be found without importing it

print("Python executable :", sys.executable)        # THE interpreter running this script
print("Python version    :", platform.python_version())
print("In a virtualenv?  :", sys.prefix != sys.base_prefix)  # True inside venv/.venv
print("sys.prefix        :", sys.prefix)

print("\nsys.path (where `import` searches, in order):")
for p in sys.path[:4]:
    print("   ", p)                         # first entry is this script's folder -> can shadow packages!

print("\nInstalled versions:")
for pkg in ["numpy", "torch", "transformers", "fastapi", "pydantic", "not-installed-pkg"]:
    try:
        print(f"   {pkg:18s} {md.version(pkg)}")   # md.version raises if the package is missing
    except md.PackageNotFoundError:
        print(f"   {pkg:18s} NOT INSTALLED in this interpreter")

# find_spec returns None if Python cannot locate a module. Safer than try/except around import for checks.
for mod in ["json", "numpy", "numpyy"]:
    spec = importlib.util.find_spec(mod)
    print(f"can import {mod!r}? {spec is not None}", "->", getattr(spec, "origin", None))

# Version-conflict check: compare against a requirement you expect.
def version_tuple(v): return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())

need = (1, 20)
have = version_tuple(md.version("numpy"))
print("\nnumpy >= 1.20 ?", have >= need, f"(have {have})")

# TRY IT: run `python3 main.py` vs `.venv/bin/python main.py` and compare the 'Python executable' lines.
# BREAK IT: add a file named json.py in this folder, run again, then delete it (module shadowing).
