"""Solution for broken_paths.py: anchor the path to this script's location.
Verify from BOTH folders:
  cd solutions && python fix_broken_paths.py
  cd ../.. && python 01_debugging_and_environment/solutions/fix_broken_paths.py"""
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "04_paths_and_cwd" / "data.txt"
print(DATA.read_text())
