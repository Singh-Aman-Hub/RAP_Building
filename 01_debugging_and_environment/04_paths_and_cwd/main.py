"""
GOAL:  Understand the working directory (cwd) and why relative paths break depending on WHERE you run.

RUN:   cd 01_debugging_and_environment/04_paths_and_cwd && python main.py
       then from repo root:  python 01_debugging_and_environment/04_paths_and_cwd/main.py
       Crashing version (EXPECTED when run from the wrong folder):  python broken_paths.py

EXPECTED OUTPUT: cwd, script folder, and the same file read two ways.

NEEDS: nothing (standard library only), no internet.
"""

import os
from pathlib import Path

print("cwd (where you ran the command):", os.getcwd())
print("script file                      :", Path(__file__).resolve())
print("script folder                    :", Path(__file__).resolve().parent)

# Fragile: relative to the CWD, so it only works if cwd == this folder.
relative = Path("data.txt")
print("\nrelative path exists from here?  :", relative.exists())

# Robust: anchored to the script's own location -> works from anywhere.
anchored = Path(__file__).resolve().parent / "data.txt"
print("anchored path exists?            :", anchored.exists())
print("content:", anchored.read_text().strip())

# Debug helper: print absolute path before opening. Often the path is not what you think.
print("\nabsolute form of relative path   :", relative.resolve())

# TRY IT: run from repo root and compare the 'relative path exists' line.
# BREAK IT: change "data.txt" to "Data.txt" -> macOS may still find it (case-insensitive) but Linux/Docker will NOT.
