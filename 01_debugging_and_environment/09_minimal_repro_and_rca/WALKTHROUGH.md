# Walkthrough 09 — Minimal reproducible example and root-cause analysis

## Goal
Turn "something is wrong somewhere" into a 6-line repro, a proven root cause and a verified fix, recorded in a journal.

## Run it
```bash
cd 01_debugging_and_environment/09_minimal_repro_and_rca
python buggy_pipeline.py     # runs with NO crash but a wrong second result
python main.py               # the shrink + proof + verified fix
```
Then fill in [../DEBUG_JOURNAL_TEMPLATE.md](../DEBUG_JOURNAL_TEMPLATE.md) (a finished example is in [../solutions/journal_example.md](../solutions/journal_example.md)).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| MRE | minimal reproducible example, minimal repro, SSCCE, reprex | Same | The smallest code that still shows the bug. |
| root cause vs symptom | why vs what you see | No | The shared default list is the cause; the wrong accuracy is the symptom. |
| RCA | root cause analysis, post-mortem, five whys | Roughly | RCA is the method; a post-mortem is the written report. |
| `assert` vs `if ... raise` | quick check vs explicit error | Nearly | `assert` can be disabled with `python -O`; do not use it for user-input validation. |
| regression test | test that stays after a fix | Same | Prevents the same bug from returning. |
| bisect (debugging) | binary search through code or history | Same idea | Halve the suspect area each step; `git bisect` does it over commits. |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–10)

```python
"""
GOAL:  Shrink a bug to a minimal reproducible example (MRE) and prove the root cause with asserts.

RUN:   cd 01_debugging_and_environment/09_minimal_repro_and_rca && python main.py
       (first run `python buggy_pipeline.py` and look at the suspicious second result)

EXPECTED OUTPUT: the shrunken repro, the diagnosis, and a verified fix.

NEEDS: nothing (standard library only).
"""
```

- **In simple words:** Label for the file: shrink a bug and prove its cause.
- **Exactly:** Docstring; nothing runs. It tells you to run `buggy_pipeline.py` first.
- **Why it is here:** Sets the workflow: see the silent bug, then shrink it.

### 2. Step 1: the minimal reproduction function  (`main.py` lines 12–18)

```python
# ---- Step 1: the MINIMAL reproduction ------------------------------------------------------
# From 40 lines we keep only what is needed to still see the bug:
#   * a function with a mutable default argument
#   * two calls
def add_item(x, bucket=[]):          # default list is created ONCE when `def` runs
    bucket.append(x)
    return bucket
```

- **In simple words:** Keep only the part that still shows the bug: a function whose default list is shared.
- **Exactly:** `bucket=[]` is evaluated ONCE when `def` runs, not per call. `append` mutates that single list.
- **Why it is here:** A 40-line problem reduced to 3 lines is easier to reason about and to show to others.

### 3. Two calls and the symptom  (`main.py` lines 21–23)

```python
first = add_item("a")
second = add_item("b")
print("repro  :", first, second)      # expected ['a'] ['b'];  actual ['a','b'] ['a','b'] (same list!)
```

- **In simple words:** Call it twice and show the surprise.
- **Exactly:** Expected `['a'] ['b']`; actual `['a', 'b'] ['a', 'b']`: both names point to the same list. Real output: `repro  : ['a', 'b'] ['a', 'b']`.
- **Why it is here:** Writing expected vs actual is the core of a bug report.

### 4. Step 2: prove the hypothesis  (`main.py` lines 25–27)

```python
# ---- Step 2: prove the hypothesis, don't just believe it -----------------------------------
print("same object? ", first is second)                 # True -> hypothesis confirmed
print("stored on the function itself:", add_item.__defaults__)   # the shared default lives here
```

- **In simple words:** Do not just believe your theory; check it.
- **Exactly:** `first is second` -> `True` (identity: same object). `add_item.__defaults__` is the tuple of default values stored on the function: `(['a', 'b'],)`, the shared list itself.
- **Why it is here:** Confirms the root cause with evidence rather than a guess.

### 5. Step 3: fix at the root cause  (`main.py` lines 29–34)

```python
# ---- Step 3: fix at the ROOT CAUSE (not by patching the symptom) -----------------------------
def add_item_fixed(x, bucket=None):
    if bucket is None:
        bucket = []                  # fresh list on every call
    bucket.append(x)
    return bucket
```

- **In simple words:** Use `None` as a placeholder and create a new list inside the function.
- **Exactly:** `if bucket is None: bucket = []` runs on every call, so every call gets its own list.
- **Why it is here:** Fix the cause, not the symptom (e.g. clearing the list afterwards would be a patch).

### 6. Step 4: verify with asserts  (`main.py` lines 37–40)

```python
# ---- Step 4: VERIFY the fix with an assert that would have failed before -----------------------
assert add_item_fixed("a") == ["a"]
assert add_item_fixed("b") == ["b"], "state leaked between calls"
print("verified: fixed version passes both asserts")
```

- **In simple words:** Executable proof that the bug is gone.
- **Exactly:** `assert expr` raises `AssertionError` if `expr` is false. `add_item_fixed("a") == ["a"]` and `add_item_fixed("b") == ["b"]` both hold (the second one has a message `"state leaked between calls"`). Output: `verified: fixed version passes both asserts`.
- **Why it is here:** Such a check would have failed before the fix and becomes a regression test.

### 7. Shrinking checklist  (`main.py` lines 42–50)

```python
# ---- Techniques for shrinking in real life ----------------------------------------------------
print("""
Shrinking checklist
  1. Make it fail every time (fix random seeds, same input).
  2. Delete code/inputs in halves ("bisect"); keep deleting while the bug persists.
  3. Replace real data / models / network by 2-3 hard-coded values.
  4. Remove third-party libraries until you know if it's yours or theirs.
  5. Write the expected vs actual output in the last two lines.
""")
```

- **In simple words:** Five practical techniques for reducing real bugs.
- **Exactly:** A triple-quoted multi-line string printed verbatim: make it deterministic, bisect, hard-code the data, strip libraries, write expected vs actual.
- **Why it is here:** The same recipe works for model, API and pipeline bugs.

### 8. TRY IT and BREAK IT comments  (`main.py` lines 52–53)

```python
# TRY IT: apply the same fix to buggy_pipeline.py (the `results={}` default), re-run it, check the second call.
# BREAK IT: remove `if bucket is None` branch from add_item_fixed and watch the second assert fail.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** Apply the same fix to `results={}` in `buggy_pipeline.py`; removing the `if bucket is None` branch makes the second assert fail.
- **Why it is here:** Applies the lesson to a larger program.

### 9. buggy_pipeline.py: docstring  (`buggy_pipeline.py` lines 1–7)

```python
"""
A SILENT bug: no crash, but the numbers are wrong.  Practise the full workflow from DEBUG_JOURNAL_TEMPLATE.md:
  observe -> reproduce -> shrink to a minimal example -> hypothesise -> test -> find root cause -> fix -> verify.

RUN: python buggy_pipeline.py        (runs fine, prints a suspicious result)
Task: "accuracy per class" for a tiny sentiment classifier. Class 'neg' should NOT be 100%.
"""
```

- **In simple words:** Describes a SILENT bug (no crash, wrong numbers) and the workflow to follow.
- **Exactly:** Docstring: observe, reproduce, shrink, hypothesise, test, root cause, fix, verify.
- **Why it is here:** Silent bugs are harder than crashes: there is no traceback to read.

### 10. buggy_pipeline: the data  (`buggy_pipeline.py` lines 9–13)

```python
# (text, true_label)
DATA = [
    ("great film", "pos"), ("loved it", "pos"), ("awful plot", "neg"),
    ("boring and slow", "neg"), ("fantastic acting", "pos"), ("terrible", "neg"),
]
```

- **In simple words:** Six labelled review snippets.
- **Exactly:** `DATA` is a list of `(text, true_label)` tuples: 3 `pos` and 3 `neg` (`awful plot`, `boring and slow`, `terrible`).
- **Why it is here:** The tiny dataset makes manual checking possible.

### 11. buggy_pipeline: a deliberately weak classifier  (`buggy_pipeline.py` lines 16–18)

```python
def predict(text):
    # deliberately weak rule: only knows two negative words
    return "neg" if ("awful" in text or "boring" in text) else "pos"
```

- **In simple words:** Says "neg" only if the text contains `awful` or `boring`.
- **Exactly:** `"terrible"` contains neither word, so it is predicted `pos` (wrong). The classifier knows only two negative words.
- **Why it is here:** Explains why neg accuracy on the first call is 2/3 and not 1.0.

### 12. buggy_pipeline: preprocess  (`buggy_pipeline.py` lines 21–25)

```python
def preprocess(rows):
    cleaned = []
    for text, label in rows:
        cleaned.append((text.lower().strip(), label))
    return cleaned
```

- **In simple words:** Lower-case and strip each text.
- **Exactly:** Builds a new list of `(text.lower().strip(), label)` tuples; creates `cleaned` fresh every call, so this function is innocent.
- **Why it is here:** Shows that not every function in the pipeline is guilty.

### 13. buggy_pipeline: accuracy_per_class (the bug is in the signature)  (`buggy_pipeline.py` lines 28–33)

```python
def accuracy_per_class(rows, labels=["pos", "neg"], results={}):     # <- look closely at this signature
    for label in labels:
        results.setdefault(label, [])
    for text, true_label in rows:
        results[true_label].append(predict(text) == true_label)
    return {k: sum(v) / len(v) for k, v in results.items()}
```

- **In simple words:** Counts correct predictions for each class and returns the fraction.
- **Exactly:** `results={}` is a mutable default shared across calls (same bug as the repro). `setdefault(label, [])` keeps existing lists. Each row appends `predict(text) == true_label` (a bool). `sum(v) / len(v)` is the mean of True/False values. First call: pos 3/3 = 1.0, neg 2/3 = 0.667.
- **Why it is here:** The root cause is one line: the default dict persists between calls, so history leaks into the next result.

### 14. buggy_pipeline: first call  (`buggy_pipeline.py` lines 36–37)

```python
train_acc = accuracy_per_class(preprocess(DATA))
print("accuracy (first call) :", train_acc)
```

- **In simple words:** Compute accuracy on the main data.
- **Exactly:** Real output: `accuracy (first call) : {'pos': 1.0, 'neg': 0.6666666666666666}`.
- **Why it is here:** Looks plausible, so the bug hides.

### 15. buggy_pipeline: second call exposes the leak  (`buggy_pipeline.py` lines 39–43)

```python
# Second call with different data: only 'neg' examples that the model gets right
extra = [("awful", "neg"), ("boring", "neg")]
print("accuracy (second call):", accuracy_per_class(preprocess(extra)))
print("expected second call  : only 'neg': 1.0 (the new data has no 'pos' rows)")
print("actual shows 'pos' and a changed 'neg' -> results from the FIRST call leaked into the second. Why?")
```

- **In simple words:** Run on only two negatives and compare with what you expect.
- **Exactly:** Real output: `accuracy (second call): {'pos': 1.0, 'neg': 0.8}`. Expected only `'neg': 1.0`. `'pos'` appears and `neg` is 4/5 because the second call appended to the lists from the first (`[T,T,F] + [T,T]`). Fix: `results=None` then `results = {}` inside; see `main.py` for the pattern.
- **Why it is here:** This is the full lesson: a silent wrong number, traced to a shared mutable default.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `def f(x, bucket=[])` | Default evaluated **once** at definition time (the bug) |
| `a is b` | Same object in memory? (vs `==` same value) |
| `assert cond, "msg"` | Raise `AssertionError` with msg if cond is False |
| `f.__defaults__` | Tuple holding a function's default values (shows the shared list) |

## Libraries & functions
Standard library / built-ins only: `setdefault`, `sum`, `enumerate`, `assert`.

## Execution trace (real output)
`buggy_pipeline.py`:
```
accuracy (first call) : {'pos': 1.0, 'neg': 0.6666666666666666}
accuracy (second call): {'pos': 1.0, 'neg': 0.8}
```
Second call should only reflect the two new `neg` rows (`neg` = 1.0, no `pos`), yet `pos` appears and `neg` is 0.8: results from the first call leaked in. The function signature `results={}` is a mutable default.
`main.py`:
```
repro  : ['a', 'b'] ['a', 'b']
same object?  True
stored on the function itself: (['a', 'b'],)
verified: fixed version passes both asserts
```

## The method (5 questions)
1. **Observe**: what exactly did I expect vs what did I get? (write both)
2. **Reproduce**: does it happen every time? with the smallest input?
3. **Shrink**: delete half the code/data; does it still fail? repeat.
4. **Hypothesise & test**: one guess at a time; an experiment that can prove it wrong (`print(a is b)`).
5. **Fix the cause, then verify**: re-run the original repro **and** add an assert/test that would have caught it.

## Experiments
1. Fix `buggy_pipeline.py` using `results=None`; confirm the second call now prints only `neg`.
2. Add a `tests` section: `assert accuracy_per_class([("awful","neg")]) == {"pos": ..., ...}` — decide what `pos` should be with no rows (ZeroDivisionError! decide behaviour).
3. Use `git bisect`-style thinking on a file with 3 functions: comment out two and see if the bug stays.

## Debugging practice
Use the template on `buggy_pipeline.py`. A good journal entry names the **root cause** ("mutable default argument shared between calls"), not the symptom ("wrong numbers").

## Rebuild checklist
- [ ] Explain MRE properties (small, self-contained, runnable, shows expected vs actual)
- [ ] Prove a hypothesis with a print/assert, not a hunch
- [ ] Write a verification step for every fix

## Say it aloud
"I reproduce the bug reliably, shrink it to the smallest example that still fails, form one hypothesis at a time and test it, then fix the root cause and verify with the original repro plus a new test. Here the root cause was a mutable default argument shared across calls."
Trade-off: shrinking takes time up front but is almost always faster than debugging inside the full system, and it produces a ready-made regression test.
