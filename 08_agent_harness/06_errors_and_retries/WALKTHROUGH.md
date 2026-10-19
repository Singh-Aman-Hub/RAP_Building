# Step 06 — Error handling and retries

## The story so far
In [step 05](../05_memory/) the agent has tools and memory, and every one of them can fail.

## The pain point
A flaky API times out, the model wraps JSON in prose, a tool raises `ZeroDivisionError`. A single uncaught exception kills the run, and a naive retry loop can spin forever.

## The spark
Classify failures. Retry transient ones (network) with capped exponential backoff; repair bad JSON by telling the model what was wrong; turn tool exceptions into observations; and always cap attempts and steps.

## Goal
Survive flaky networks, malformed model output and failing tools without crashing or looping forever.

## Run it
```bash
cd 08_agent_harness/06_errors_and_retries && ../../.venv/bin/python main.py
```
Real output (sleep times vary because of random jitter):
```
A flaky network (2 timeouts, then OK)
  Q: What is 6 x 7?
      retry 1/3 after TimeoutError; sleeping 0.05s
      retry 2/3 after TimeoutError; sleeping 0.11s
      tool calculator -> 42
  A: Result: 42

B invalid JSON once
  Q: What is 6 x 7?
      bad JSON: 'Sure! I will calculate that for you right away.' -> asking the model to repair (repair 1/2)
      tool calculator -> 42
  A: Result: 42

C tool error (divide by zero)
  Q: What is 5 / 0?
      tool calculator -> ERROR: ZeroDivisionError: division by zero
  A: Sorry, the tool failed: ERROR: ZeroDivisionError: division by zero

D permanent outage
  Q: What is 6 x 7?
      retry 1/3 after TimeoutError; sleeping 0.10s
      retry 2/3 after TimeoutError; sleeping 0.12s
      retry 3/3 after TimeoutError; sleeping 0.20s
  A: FAILED: model unavailable after retries (TimeoutError). Try again later.
```

(Sleep values contain random jitter, so they differ slightly on every run.)

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class FlakyLLM:` with `__init__` and `__call__` | a class whose instances can be called like functions | `class A: def __call__(self): ...; A()()` |
| `self.calls += 1` | instance attribute updated on every call | – |
| `RETRYABLE = (TimeoutError, ConnectionError)` | tuple of exception classes, usable in `except` | `except (A, B):` |
| `raise` (alone) | re-raise the exception being handled | – |
| `2 ** (attempt - 1)` | exponent: 1, 2, 4, 8... | `2 ** 3` -> 8 |
| `lambda: generate(...)` | zero-argument function wrapping a call to retry | – |
| `continue` | skip to the next loop iteration | – |
| `type(err).__name__` | the exception's class name as text | `type(ValueError()).__name__` |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `time.sleep(s)` | standard library | pause `s` seconds | None |
| `random.uniform(a, b)` | standard library | random float (the **jitter**) | float |
| `TimeoutError`, `ConnectionError` | built-in exceptions | transient network failures | – |
| `extract_json` | shared | raises `ValueError` if no JSON | dict |

## Execution trace
Three layers: (1) `call_with_retry` wraps the model call — waits ~0.05, 0.10, 0.20 s (doubling + jitter) and retries only `RETRYABLE` errors; (2) if no JSON, append the bad reply and a repair request, up to `max_repairs`; (3) `run_tool` converts any tool exception into `ERROR: ...` text. Scenario D shows a clean failure message instead of a traceback.

## Experiments
- `base_delay=0.5` — time the run; backoff protects the server but costs the user time.
- `max_repairs=0` -> scenario B fails immediately.

## Debugging practice
Delete the `except RETRYABLE` in `run_agent` and run D: the `TimeoutError` traceback shows the exact line that raised it — read bottom-up. Decide which layer *should* have caught it.

## Rebuild checklist
- [ ] retry with backoff + jitter and retryable-only. [ ] JSON repair with a cap. [ ] errors as observations.

## The new problem (cliffhanger)
The agent now survives failures, but it will also happily run `delete_file("/")` or read `/etc/passwd` if the model asks. Resilience is not safety. Next: [07_safety_and_permissions](../07_safety_and_permissions/).

> **Say it aloud (interview box):** I separate transient failures, which I retry with capped exponential backoff, from model-output failures, which I repair by feeding the error back, and from tool failures, which I return as observations. Every retry has a hard cap so the agent cannot loop forever or burn money. The goal is that a failure becomes information for the model, not a crash.
