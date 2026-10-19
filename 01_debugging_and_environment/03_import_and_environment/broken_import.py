"""
EXPECTED TO CRASH.  Choose a case:   python broken_import.py 1    (or 2, 3)
 1: package name typo / not installed         -> ModuleNotFoundError
 2: module exists but the NAME is wrong       -> ImportError (cannot import name ...)
 3: wrong attribute on a real module          -> AttributeError
For each: read the last line, then decide which check from main.py would have found the cause.
"""
import sys

case = sys.argv[1] if len(sys.argv) > 1 else "1"

if case == "1":
    import numpyy                      # typo of numpy
elif case == "2":
    from json import load_string       # json has loads, not load_string
elif case == "3":
    import math
    print(math.squareroot(9))          # real function is math.sqrt
else:
    print("choose 1, 2 or 3")
