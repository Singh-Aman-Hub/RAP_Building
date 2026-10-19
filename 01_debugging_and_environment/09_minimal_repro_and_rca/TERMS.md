# Terms — Minimal Reproducible Example and Root-Cause Analysis

Folder: `01_debugging_and_environment/09_minimal_repro_and_rca` · Files covered: `main.py`, `buggy_pipeline.py`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [Minimal reproducible example (MRE)](#minimal-reproducible-example-mre) · [Root-cause analysis (RCA)](#root-cause-analysis-rca) · [Silent bug](#silent-bug) · [Mutable default argument](#mutable-default-argument) · [The None-default fix](#the-none-default-fix) · [is vs ==](#is-vs-) · [func.__defaults__](#func__defaults__) · [assert for verification](#assert-for-verification) · [Hypothesis testing](#hypothesis-testing) · [Bisecting (shrinking by halves)](#bisecting-shrinking-by-halves) · [Verify the fix](#verify-the-fix) · [Debug journal](#debug-journal) · [dict.setdefault](#dictsetdefault) · [Accuracy per class](#accuracy-per-class) · [Dict comprehension over items](#dict-comprehension-over-items) · [Expected vs actual](#expected-vs-actual) · [Regression test](#regression-test) · [Triple-quoted string](#triple-quoted-string)

---

## Minimal reproducible example (MRE)  `(concept)`
**1. Definition:** A minimal reproducible example is the smallest self-contained piece of code (with hard-coded data) that still shows the bug when run, together with the expected and actual results.
**2. In plain words:** From a 40-line pipeline the bug shrinks to a function with a default list and two calls. A good MRE is small, complete, runnable and states expected vs actual output. It is also the best way to ask others for help and becomes a ready-made test.
**3. Example to quote:**
> "I shrank the pipeline to six lines: one function with `bucket=[]` and two calls; it still printed `['a', 'b'] ['a', 'b']`."
```python
def add_item(x, bucket=[]):
    bucket.append(x); return bucket
```
**Seen in:** `main.py`, Step 1

## Root-cause analysis (RCA)  `(concept)`
**1. Definition:** Root-cause analysis is the process of tracing a failure back to the underlying cause rather than treating its visible symptom.
**2. In plain words:** "Wrong numbers" is the symptom; "mutable default argument shared between calls" is the cause. Fixing the cause means the bug can't come back in another form. Write the cause in your notes.
**3. Example to quote:**
> "The symptom was an unexpected 'pos' score in the second call; the root cause was `results={}` in the function signature."
```python
# symptom: wrong accuracy   cause: shared default dict
```
**Seen in:** `buggy_pipeline.py`; `WALKTHROUGH.md`

## Silent bug  `(concept)`
**1. Definition:** A silent bug produces wrong results without raising any exception.
**2. In plain words:** These are harder than crashes, because nothing tells you something is wrong. You notice only if you compare against what you expected. `buggy_pipeline.py` runs fine yet the second result is off.
**3. Example to quote:**
> "`buggy_pipeline.py` never crashed, but the second call returned a 'pos' key that shouldn't exist, so I compared against my expectation."
```python
print("expected:", {"neg": 1.0})
```
**Seen in:** `buggy_pipeline.py`

## Mutable default argument  `(concept)`
**1. Definition:** A default value such as `[]` or `{}` is created once when the function is defined and shared by all calls that rely on the default.
**2. In plain words:** The first call changes the shared dict, and the second call sees those changes. In the pipeline, `results={}` carried first-call results into the second call.
**3. Example to quote:**
> "`def accuracy_per_class(rows, labels=[...], results={})` kept results from call one, so call two was contaminated."
```python
def f(x, bucket=[]):
    bucket.append(x); return bucket
```
**Seen in:** `main.py`, `add_item`; `buggy_pipeline.py`, `accuracy_per_class`

## The None-default fix  `(concept)`
**1. Definition:** The standard fix is to default to `None` and create a fresh object inside the function: `if bucket is None: bucket = []`.
**2. In plain words:** Every call that doesn't pass a bucket gets its own new list, so state doesn't leak.
**3. Example to quote:**
> "I changed `bucket=[]` to `bucket=None` and created the list inside, so the second call returned `['b']`."
```python
def add_item_fixed(x, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(x)
    return bucket
```
**Seen in:** `main.py`, Step 3

## is vs ==  `(syntax)`
**1. Definition:** `a is b` tests whether two names refer to the exact same object; `a == b` tests whether their values are equal.
**2. In plain words:** `first is second` being `True` proves both calls returned the very same list, confirming the shared-default hypothesis. Two different lists with equal contents are `==` but not `is`.
**3. Example to quote:**
> "`first is second` printed `True`, proving both variables point to one list object."
```python
print([1] == [1], [1] is [1])   # True False
```
**Seen in:** `main.py`, Step 2

## func.__defaults__  `(Python built-in)`
**1. Definition:** `function.__defaults__` is a tuple holding the function's default parameter values.
**2. In plain words:** It lets you look at the shared default directly: `(['a', 'b'],)` shows the list that accumulated items across calls. It is evidence rather than guesswork.
**3. Example to quote:**
> "`add_item.__defaults__` printed `(['a', 'b'],)`, showing the shared list living on the function itself."
```python
print(add_item.__defaults__)
```
**Seen in:** `main.py`, Step 2

## assert for verification  `(syntax)`
**1. Definition:** `assert condition, "message"` raises `AssertionError` with the message when the condition is false; here it checks the fix.
**2. In plain words:** The asserts fail on the buggy version and pass on the fixed one, so they prove the fix and guard against regressions.
**3. Example to quote:**
> "`assert add_item_fixed('b') == ['b'], 'state leaked between calls'` would have failed before my fix."
```python
assert add_item_fixed("a") == ["a"]
```
**Seen in:** `main.py`, Step 4

## Hypothesis testing  `(concept)`
**1. Definition:** Hypothesis testing in debugging means forming one specific guess about the cause and running an experiment that can prove it wrong.
**2. In plain words:** "Maybe both calls share one list" becomes `print(first is second)`. Change one thing at a time, so you know what mattered.
**3. Example to quote:**
> "My hypothesis was a shared default list; `first is second` returning `True` confirmed it."
```python
print(first is second)
```
**Seen in:** `main.py`, Step 2; `WALKTHROUGH.md`, the method

## Bisecting (shrinking by halves)  `(concept)`
**1. Definition:** Bisecting is repeatedly removing half of the code or data and checking whether the bug persists, to isolate its location.
**2. In plain words:** Comment out half the functions; if the bug stays, it's in the remaining half. Repeat until only the faulty piece remains. The checklist also says to fix random seeds and replace real data or models with 2 or 3 hard-coded values.
**3. Example to quote:**
> "I commented out two of three functions to see whether the bug remained, halving the suspect code each time."
```python
# keep deleting while the bug persists
```
**Seen in:** `main.py`, shrinking checklist; `WALKTHROUGH.md`, experiment 3

## Verify the fix  `(concept)`
**1. Definition:** Verifying a fix means re-running the original reproduction and adding a test that would have failed before the fix.
**2. In plain words:** A fix isn't done until you've shown the original failure no longer occurs. Otherwise you only hope it's fixed.
**3. Example to quote:**
> "After the fix I re-ran the repro and added asserts for both calls, so I know it works and won't regress."
```python
assert add_item_fixed("b") == ["b"]
```
**Seen in:** `main.py`, Step 4

## Debug journal  `(concept)`
**1. Definition:** A debug journal is a written record of a bug's symptom, expected vs actual behaviour, hypotheses, root cause, fix and verification.
**2. In plain words:** Writing it forces clarity and builds your debugging memory. A good entry names the cause ("mutable default argument shared between calls"), not the symptom ("wrong numbers"). The template is `DEBUG_JOURNAL_TEMPLATE.md`.
**3. Example to quote:**
> "I fill in the journal template with the root cause and how I verified the fix, so I can reuse the lesson."
```text
Symptom / Root cause / Fix / Verification
```
**Seen in:** `WALKTHROUGH.md`, debugging practice

## dict.setdefault  `(Python built-in)`
**1. Definition:** `d.setdefault(key, default)` returns `d[key]` if present, otherwise inserts `key: default` and returns it.
**2. In plain words:** It's used in `buggy_pipeline.py` to make sure each label has a list before appending results. Combined with the shared `results` dict, entries persist across calls.
**3. Example to quote:**
> "`results.setdefault(label, [])` creates an empty list for a label the first time it's seen."
```python
d = {}
d.setdefault("pos", []).append(True)
```
**Seen in:** `buggy_pipeline.py`, `accuracy_per_class`

## Accuracy per class  `(concept)`
**1. Definition:** Accuracy per class is, for each label, the fraction of that label's examples the model predicted correctly.
**2. In plain words:** Overall accuracy can hide that one class is poor. Here, `neg` is 0.667 because the toy rule only knows two negative words. It's computed as `sum(v) / len(v)` where `v` is a list of True/False values.
**3. Example to quote:**
> "The first call showed `{'pos': 1.0, 'neg': 0.667}`: the weak rule misses the negative review 'terrible'."
```python
print(sum([True, False, True]) / 3)   # 0.666...
```
**Seen in:** `buggy_pipeline.py`

## Dict comprehension over items  `(syntax)`
**1. Definition:** `{k: sum(v) / len(v) for k, v in results.items()}` builds a new dict by computing a value from each key's list.
**2. In plain words:** It turns lists of booleans into ratios in one line. `sum()` counts the Trues because `True` is 1.
**3. Example to quote:**
> "I used a dict comprehension over `results.items()` to turn each label's hits and misses into an accuracy."
```python
{k: sum(v) / len(v) for k, v in {"a": [True, False]}.items()}
```
**Seen in:** `buggy_pipeline.py`, return line

## Expected vs actual  `(concept)`
**1. Definition:** Stating the expected output next to the actual output is the first step of any bug report.
**2. In plain words:** You can only spot a silent bug if you know what you expected. The scripts print both: expected `['a'] ['b']`, actual `['a','b'] ['a','b']`.
**3. Example to quote:**
> "I wrote 'expected: only neg = 1.0; actual: pos appears and neg is 0.8', which pointed me to leaked state."
```python
print("expected:", ["a"], "actual:", ["a", "b"])
```
**Seen in:** `main.py`; `buggy_pipeline.py`

## Regression test  `(concept)`
**1. Definition:** A regression test is a test added after a bug fix that fails if the bug ever returns.
**2. In plain words:** The MRE you built while shrinking the bug is already most of the test. Keeping it means the same mistake can't sneak back.
**3. Example to quote:**
> "I kept my two asserts as a regression test so the shared-default bug can't return unnoticed."
```python
assert add_item_fixed("a") == ["a"]
```
**Seen in:** `WALKTHROUGH.md`, trade-off

## Triple-quoted string  `(syntax)`
**1. Definition:** Text wrapped in `"""..."""` is a string that can span multiple lines.
**2. In plain words:** `main.py` uses it to print the multi-line shrinking checklist in one `print` call. The same syntax is used for docstrings.
**3. Example to quote:**
> "I printed the checklist with one triple-quoted string instead of many `print` lines."
```python
print("""line one
line two""")
```
**Seen in:** `main.py`, shrinking checklist
