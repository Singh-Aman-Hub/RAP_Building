# Module 01 — Debugging & Environment (syllabus topic 9)

## Why this matters
Interviewers will ask you to debug code **without ChatGPT/Claude**. They watch whether you read the traceback, check the environment, isolate the failure and verify a fix — not whether you magically know the answer. You will also use these skills constantly in every later module (an import error in [04](../04_llm_fundamentals/), a 422 in [09](../09_api_development/), a failing container in [11](../11_docker/)).

## Where this fits
**Builds on:** [00 Python foundations](../00_python_foundations/) — exceptions (06), `if __name__` (07), pathlib (06), type hints (03).

**Leads to (concrete reuse):**
| Skill here | Reused in |
|---|---|
| Traceback reading (01-02) | everything; Docker logs ARE tracebacks ([11](../11_docker/)) |
| venv + versions (03) | installing torch/transformers ([04](../04_llm_fundamentals/)); `requirements.txt` in [11](../11_docker/) |
| Paths/cwd (04) | saving/loading models ([03](../03_ml_foundations/), [10](../10_ml_system_architecture/)); Docker `COPY`/`WORKDIR` |
| Type validation (05) | Pydantic request models in [09](../09_api_development/); tool-argument validation in [08](../08_agent_harness/) |
| HTTP errors & timeouts (06) | LLM API calls + retries ([08](../08_agent_harness/)); Streamlit calling the API ([12](../12_streamlit_gradio/)) |
| Env vars (07) | API keys ([08](../08_agent_harness/)), Docker `-e`/Compose ([11](../11_docker/)), `LLM_BACKEND` ([shared/llm_client.py](../shared/llm_client.py)) |
| Logging (08) | model-serving monitoring ([10](../10_ml_system_architecture/)) |
| MRE + RCA (09) | prompt regressions ([07](../07_prompt_engineering/)), the capstone ([13](../13_capstone_project/)) |

## Concept map
```
            ┌──────────── the debugging loop ────────────┐
            │ observe -> reproduce -> shrink -> hypothesis│
            │          -> test -> fix cause -> verify     │
            └────────────────────┬────────────────────────┘
                                 │ each stage uses a toolbox:
 reading errors ── 01 traceback ─ 02 syntax/runtime types
 environment ───── 03 interpreter/venv/versions ─ 04 paths/cwd ─ 07 env vars/config
 data ──────────── 05 types/validation
 network ───────── 06 HTTP status, timeouts, connection errors
 observability ─── 08 logging / print / breakpoint / assert
 process ───────── 09 MRE + root cause + journal
 containers ────── 10 docker logs/exec/inspect (taught fully in module 11)
```

## Study path
| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_reading_tracebacks](01_reading_tracebacks/) | bottom-up reading, frames | `python main.py` + `broken_traceback.py` | 40 min |
| 2 | [02_syntax_and_runtime_errors](02_syntax_and_runtime_errors/) | the 10 common exceptions, SyntaxError | `main.py`, `broken_syntax.py`, `broken_runtime.py` | 45 min |
| 3 | [03_import_and_environment](03_import_and_environment/) | which Python? venv, versions, shadowing | `main.py`, `broken_import.py 1/2/3` | 1 h |
| 4 | [04_paths_and_cwd](04_paths_and_cwd/) | cwd vs script folder | run from two folders | 30 min |
| 5 | [05_types_and_validation](05_types_and_validation/) | isinstance, Pydantic | `main.py`, `broken_types.py` | 45 min |
| 6 | [06_http_and_network](06_http_and_network/) | status codes, timeout vs connection error | `main.py`, `broken_http.py` | 1 h |
| 7 | [07_env_vars_and_config](07_env_vars_and_config/) | os.environ, .env | `PORT=abc python main.py` | 40 min |
| 8 | [08_logging_vs_print](08_logging_vs_print/) | logging levels, `breakpoint()` | `LOG_LEVEL=DEBUG python main.py` | 45 min |
| 9 | [09_minimal_repro_and_rca](09_minimal_repro_and_rca/) | MRE, hypotheses, journal | `buggy_pipeline.py`, `main.py` | 1 h |
| 10 | [10_docker_errors](10_docker_errors/) | container debugging map (re-read after [11](../11_docker/)) | `python main.py` | 30 min |

Total about 7 h, then [exercises.md](exercises.md). Extras: [CHEATSHEET.md](CHEATSHEET.md), [DEBUG_JOURNAL_TEMPLATE.md](DEBUG_JOURNAL_TEMPLATE.md), [solutions/](solutions/).

**Convention:** files named `broken_*.py` / `buggy_*.py` are **expected to crash or misbehave** — they are drills, not bugs in the pack. Do not ask an AI to fix them; use the journal.

## Concepts in plain English (syllabus bullets)
- **Tracebacks:** the call stack at the moment of failure. Read bottom-up. (01)
- **Syntax vs runtime errors:** found at compile time (nothing runs) vs while running. (02)
- **Import/dependency errors:** almost always "wrong interpreter", "not installed", or "name shadowed". (03)
- **Virtual environments:** a private folder of packages per project so versions don't clash. `python -m venv .venv`. (03)
- **Version conflicts:** two libs need incompatible versions of a third. Diagnose with `pip check`; fix by pinning in a fresh venv. (03)
- **File paths/working directory:** relative paths depend on where you launched; anchor with `Path(__file__)`. (04)
- **Type errors & validation:** check at the boundary; Pydantic automates it. (05)
- **API/HTTP and network/timeout errors:** 4xx you, 5xx server, timeout = slow, connection error = unreachable. (06)
- **Env vars/config:** settings outside code; always strings. (07)
- **Docker build/runtime errors:** logs first, then compare container vs laptop. (10 + [11](../11_docker/))
- **Logging & print debugging:** `print` for a quick peek, `logging` for anything long-lived, `breakpoint()` to inspect live. (08)
- **Minimal reproducible example:** the smallest program that still shows the bug. (09)
- **Root-cause analysis & systematic testing:** one hypothesis at a time, prove it, fix the cause, add a test. (09)

## Common mistakes
- Fixing before reading the last line. - Changing several things at once. - Installing with `pip` for one Python and running another.
- `except: pass` hiding the real error. - Forgetting `timeout=` on network calls. - Assuming the cwd is the script's folder. - Printing secrets.

## Self-check
- [ ] I can state the 4-step reading recipe for a traceback
- [ ] I can tell SyntaxError from runtime errors and name 10 error types with causes
- [ ] I can print the interpreter path and package version from Python, and fix an import error with `python -m pip`
- [ ] I can explain a venv and a version conflict, and run `pip check`
- [ ] I can fix a path bug and verify from two directories
- [ ] I can validate input with `isinstance` and a Pydantic model
- [ ] I can distinguish 404 / 422 / 500 / Timeout / ConnectionError
- [ ] I can read config from env vars with defaults and clear failures
- [ ] I can configure `logging` and use `breakpoint()`
- [ ] I can shrink a bug to an MRE and complete a debug journal
- [ ] I can name the first 3 commands for a failing container

## Interview questions
1. **You get `ModuleNotFoundError` — what do you do?** Read the module name (typo?), print `sys.executable`, run `python -m pip show NAME` with that same interpreter, check the venv is active and that no local file shadows the name. Fix: `python -m pip install NAME` (and add to `requirements.txt`).
2. **How do you read a traceback?** Bottom line first (type + message), frame above it (raising line), then climb to see which caller supplied the bad value.
3. **`FileNotFoundError` but the file exists?** Relative path vs cwd. Print `os.getcwd()` and the absolute path; anchor to `Path(__file__)`. On Linux/Docker check case and `COPY`.
4. **What is a virtual environment and why use it?** Isolated interpreter + packages per project to avoid version conflicts; reproducible with `requirements.txt`. Limitation: only isolates Python packages, not system libraries or OS — Docker does that.
5. **401 vs 403 vs 404 vs 422 vs 500 vs 503?** Not authenticated / not allowed / no such route / body failed validation / server bug / overloaded or unavailable.
6. **Timeout vs connection error?** Timeout: connected but no response in time. ConnectionError: could not connect (down, wrong host/port).
7. **Why always set a timeout on HTTP calls?** The default waits forever, so one hung server freezes your app.
8. **print vs logging?** logging has levels, timestamps, handlers and can be configured without editing code; use it in servers/containers; `print` only for quick exploration.
9. **What is an MRE and why make one?** The smallest self-contained code showing the bug; it isolates the cause, often reveals it, and becomes a regression test.
10. **How do you prove a fix works?** Re-run the original repro, add an assert/test that would have failed before, and re-run related checks.
11. **Works on my machine, fails in Docker — why?** Different Python/dependencies, missing files (`COPY`), env vars not passed, server bound to 127.0.0.1, case-sensitive paths, different CPU architecture.
12. **Why not wrap everything in `try/except Exception: pass`?** It hides the traceback and bugs; catch specific exceptions, log with `logger.exception`, re-raise if you can't handle it.
