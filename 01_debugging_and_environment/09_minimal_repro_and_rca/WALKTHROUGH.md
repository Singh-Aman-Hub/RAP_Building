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
