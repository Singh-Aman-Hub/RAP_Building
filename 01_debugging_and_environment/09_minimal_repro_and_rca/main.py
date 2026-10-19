"""
GOAL:  Shrink a bug to a minimal reproducible example (MRE) and prove the root cause with asserts.

RUN:   cd 01_debugging_and_environment/09_minimal_repro_and_rca && python main.py
       (first run `python buggy_pipeline.py` and look at the suspicious second result)

EXPECTED OUTPUT: the shrunken repro, the diagnosis, and a verified fix.

NEEDS: nothing (standard library only).
"""

# ---- Step 1: the MINIMAL reproduction ------------------------------------------------------
# From 40 lines we keep only what is needed to still see the bug:
#   * a function with a mutable default argument
#   * two calls
def add_item(x, bucket=[]):          # default list is created ONCE when `def` runs
    bucket.append(x)
    return bucket


first = add_item("a")
second = add_item("b")
print("repro  :", first, second)      # expected ['a'] ['b'];  actual ['a','b'] ['a','b'] (same list!)

# ---- Step 2: prove the hypothesis, don't just believe it -----------------------------------
print("same object? ", first is second)                 # True -> hypothesis confirmed
print("stored on the function itself:", add_item.__defaults__)   # the shared default lives here

# ---- Step 3: fix at the ROOT CAUSE (not by patching the symptom) -----------------------------
def add_item_fixed(x, bucket=None):
    if bucket is None:
        bucket = []                  # fresh list on every call
    bucket.append(x)
    return bucket


# ---- Step 4: VERIFY the fix with an assert that would have failed before -----------------------
assert add_item_fixed("a") == ["a"]
assert add_item_fixed("b") == ["b"], "state leaked between calls"
print("verified: fixed version passes both asserts")

# ---- Techniques for shrinking in real life ----------------------------------------------------
print("""
Shrinking checklist
  1. Make it fail every time (fix random seeds, same input).
  2. Delete code/inputs in halves ("bisect"); keep deleting while the bug persists.
  3. Replace real data / models / network by 2-3 hard-coded values.
  4. Remove third-party libraries until you know if it's yours or theirs.
  5. Write the expected vs actual output in the last two lines.
""")

# TRY IT: apply the same fix to buggy_pipeline.py (the `results={}` default), re-run it, check the second call.
# BREAK IT: remove `if bucket is None` branch from add_item_fixed and watch the second assert fail.
