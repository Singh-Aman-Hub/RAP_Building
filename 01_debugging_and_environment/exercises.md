# Exercises — Module 01 (Debugging & Environment)

House rule for this module: **no AI for the fix.** Use the docs (`help()`, library docs), the traceback, the [CHEATSHEET](CHEATSHEET.md) and the [journal template](DEBUG_JOURNAL_TEMPLATE.md). Worked answers are in [solutions/](solutions/) — open them only after you have a verified fix.

| Step | Modify it | Break it / drill | Rebuild from memory | Explain aloud |
|---|---|---|---|---|
| 01 traceback | Add a 4th nested call; list the frames in order | `broken_traceback.py`: write (a) type (b) raising line (c) calling line (d) root cause -> [solution](solutions/fix_broken_traceback.py) | Code that raises `KeyError` 3 frames deep and prints `traceback.format_exc()` | The 4-step reading recipe |
| 02 syntax/runtime | Add `[].pop()` to the catalogue | `broken_syntax.py` (3 errors) -> [solution](solutions/fix_broken_syntax.py); `broken_runtime.py` -> [solution](solutions/fix_broken_runtime.py) | List 10 errors with one cause each | Why does a SyntaxError print nothing first? |
| 03 imports/env | Print the version of `requests` and `pytest` | `broken_import.py 1/2/3`; create `json.py` to cause shadowing | Create a venv, install a package, prove it from inside Python | `python` vs `python -m pip` mismatch |
| 04 paths | Run `main.py` from 3 different folders | `broken_paths.py` from repo root -> [solution](solutions/fix_broken_paths.py) (verify from both folders) | Anchored-path read of any file | cwd vs `__file__` |
| 05 types | Add a `stop: list[str]` field and a `max_length` constraint | `broken_types.py` (3 bugs) -> [solution](solutions/fix_broken_types.py) | A Pydantic model with 3 constrained fields | Coerce vs reject; why validate at the boundary |
| 06 http | Add a `/created` route returning 201; print it | `broken_http.py` -> [solution](solutions/fix_broken_http.py) | A client that handles Timeout / ConnectionError / HTTPError separately | 4xx vs 5xx; retry policy |
| 07 env vars | Add a `LOG_LEVEL` setting with allowed values | `broken_env.py` -> [solution](solutions/fix_broken_env.py) | `get_int`/`get_required` helpers | Why env vars for secrets and ports |
| 08 logging | Add a `FileHandler` with a different level (`WARNING`) | Call `basicConfig` twice and note what is ignored | Logging setup + `logger.exception` | print vs logging; when `breakpoint()` |
| 09 MRE/RCA | Apply the fix to `buggy_pipeline.py` -> [solution](solutions/fix_buggy_pipeline.py) | Reintroduce the bug; show the assert catches it | Fill in the journal for it -> [example](solutions/journal_example.md) | What makes a good MRE? |
| 10 docker errors | Add 2 new regexes to the lookup | After [module 11](../11_docker/): break your Dockerfile each way | Table: earlier step -> Docker form | Why `localhost` in a container differs |

## Syllabus "practical examples" checklist (topic 9)
- [ ] Diagnose an import error without AI (03: `broken_import.py 1`)
- [ ] Read a traceback and locate the relevant line (01)
- [ ] Check the interpreter and installed package versions (03 `main.py`)
- [ ] Reproduce a bug with the smallest possible example (09)
- [ ] Write down the root cause and how the fix was verified (journal)

## Mock debugging interview (30 min, timed, no AI)
1. Pick a file from [`../mock_interviews/`](../mock_interviews/) if present, else run `broken_*.py` you have *not* solved yet.
2. Say aloud: what you see, what you check first, your hypothesis, your test. Narrating is the skill being assessed.
3. Fix, verify, and write 3 sentences: root cause, fix, verification.

## Capstone drill
Create a 20-line script with **three** planted bugs of different kinds (wrong path, wrong type, missing env var). Swap with a friend (or yourself next day) and fix them with a journal entry for each.
