"""
EXPECTED TO CRASH when run from any folder except this one.
  works : cd 01_debugging_and_environment/04_paths_and_cwd && python broken_paths.py
  fails : cd <repo root> && python 01_debugging_and_environment/04_paths_and_cwd/broken_paths.py
Fix it by anchoring the path to __file__ (see main.py), then verify from BOTH folders.
"""
with open("data.txt") as f:
    print(f.read())
