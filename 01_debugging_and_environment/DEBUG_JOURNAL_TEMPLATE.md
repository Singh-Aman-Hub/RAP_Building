# Debug journal — template

Copy this file for every bug you investigate (`journal_YYYY-MM-DD_short-name.md`). A finished example: [solutions/journal_example.md](solutions/journal_example.md).
Rule: **do not ask an AI for the fix until sections 1-5 are filled in.** If you do ask, paste your minimal repro (section 3), not the whole project.

**Date:**  **Project / file:**

## 1. Symptom (what I saw)
(Exact error text or wrong output. Copy the LAST line of the traceback.)

## 2. Expected vs actual
- Expected:
- Actual:

## 3. Reproduction
- Command to trigger it (from which folder?):
- Happens every time? (yes / sometimes -> seed? timing? input?)
- Minimal reproducible example (smallest code + data that still fails):
```python

```

## 4. Environment
Python version / interpreter path (`sys.executable`) / venv active? / key package versions / OS / Docker?

## 5. Hypotheses (one at a time)
| # | Hypothesis | Test that could prove it WRONG | Result |
|---|---|---|---|
| 1 | | | |

## 6. Root cause
(One sentence naming the CAUSE, not the symptom.  "Mutable default argument shared across calls", not "numbers were wrong".)

## 7. Fix
(What changed and why this addresses the root cause.)

## 8. Verification
- [ ] Original repro no longer fails
- [ ] A new assert/test covers this case
- [ ] Nothing else broke (re-ran related scripts)

## 9. Prevention / lesson
(Check, linter rule, validation, logging that would catch this sooner.)
