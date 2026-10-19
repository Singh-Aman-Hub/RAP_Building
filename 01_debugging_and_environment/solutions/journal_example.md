# Debug journal — example (filled in)

**Date:** 2026-10-19  **Project:** 01_debugging/09_minimal_repro_and_rca

## 1. Symptom (what I saw)
Second call of `accuracy_per_class` printed `{'pos': 1.0, 'neg': 0.8}`.

## 2. Expected vs actual
- Expected: `{'neg': 1.0}` (new data only has `neg` rows, both predicted correctly)
- Actual: includes `pos` and `neg` = 0.8

## 3. Reproduction
- Command: `python buggy_pipeline.py` — happens every run, no randomness.
- Minimal repro (6 lines): `add_item("a")`, `add_item("b")` with default `bucket=[]` -> `['a','b'] ['a','b']`.

## 4. Environment
Python 3.12.13, `.venv`, macOS; standard library only.

## 5. Hypotheses (one at a time)
| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | `predict()` is wrong for new rows | call it directly on the 2 rows | correct -> rejected |
| 2 | State is kept between calls | `print(accuracy_per_class.__defaults__)` | shows dict with first-call data -> **confirmed** |

## 6. Root cause
`results={}` is a mutable default argument; Python builds it once at function definition, so all calls share one dict.

## 7. Fix
Default `None`, create `results = {}` inside the function.

## 8. Verification
- Original repro now prints `{'neg': 1.0}`.
- Added `assert second == {"neg": 1.0}` in `solutions/fix_buggy_pipeline.py`.

## 9. Prevention / lesson
Never use `[]`/`{}` as defaults. Linters (ruff B006) flag it.
