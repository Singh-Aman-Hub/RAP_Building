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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Exponential backoff | retry with increasing delay, backoff | Same | Wait 1x, 2x, 4x the base delay between attempts. |
| Jitter | random delay, randomised backoff | Same | A small random extra wait so many clients do not retry together. |
| Transient vs permanent error | retryable vs non-retryable, 5xx/429 vs 4xx | Opposites | Transient may fix itself (timeout, rate limit); permanent will not (bad request, bug). |
| JSON repair | reprompt, self-correction, retry-with-feedback | Near-same | Tell the model its output was invalid and ask again. |
| Failure injection | chaos testing, fault injection, test double | Near-same | Deliberately causing failures so defences can be tested. |

## Code, step by step

### 1. Imports and path setup  (`main.py` lines 9–18)

```python
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json, MockLLM
from fake_llm import fake_reply
```

- **In simple words:** Load randomness, time, and helpers.
- **Exactly:** `random.uniform` for jitter, `time.sleep` to wait, `fake_reply` is the scripted brain function itself (not wrapped in a MockLLM yet).
- **Why it is here:** Retries need timing and randomness.

### 2. A flaky fake model  (`main.py` lines 23–31)

```python
class FlakyLLM:
    """Raises TimeoutError for the first `fail_times` calls, then behaves."""
    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0
    def __call__(self, messages, system=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("simulated: request timed out")
        return fake_reply(messages, system)
```

- **In simple words:** A pretend model that times out a few times and then works.
- **Exactly:** A class whose `__call__(messages, system)` makes instances usable as a mock rule. Counts calls; raises `TimeoutError("simulated: request timed out")` while `calls <= fail_times`, then returns `fake_reply(...)`.
- **Why it is here:** You cannot test retry logic without failures on demand.

### 3. A model that answers with chatty text once  (`main.py` lines 33–49)

```python
class BadJsonOnceLLM:
    """First reply is chatty text with no JSON; afterwards normal."""
    def __init__(self):
        self.calls = 0
    def __call__(self, messages, system=None):
        self.calls += 1
        if self.calls == 1:
            return "Sure! I will calculate that for you right away."
        # A well-behaved model reads the repair request and answers properly. Our fake brain cannot "read"
        # it, so we hide the repair exchange (bad reply + repair request) and let it answer the question.
        cleaned = []
        for i, m in enumerate(messages):
            repair_next = i + 1 < len(messages) and "not valid JSON" in messages[i + 1]["content"]
            if repair_next or "not valid JSON" in m["content"]:
                continue
            cleaned.append(m)
        return fake_reply(cleaned, system)
```

- **In simple words:** The first reply has no JSON; later replies are proper.
- **Exactly:** Call 1 returns `Sure! I will calculate that for you right away.` After that it removes the repair exchange from the messages (the bad reply and the "not valid JSON" request) because the scripted brain cannot read them, then calls `fake_reply(cleaned, system)`.
- **Why it is here:** Lets the repair path be exercised deterministically.

### 4. Wrap any callable as a mock model  (`main.py` lines 51–52)

```python
def as_mock(callable_brain):
    return MockLLM(rules=[(r"", callable_brain)])
```

- **In simple words:** Adapter so a function can be passed to `generate`.
- **Exactly:** `MockLLM(rules=[(r"", callable_brain)])`: empty regex matches everything.
- **Why it is here:** `generate` expects a `MockLLM`.

### 5. Which errors are retryable  (`main.py` lines 57–57)

```python
RETRYABLE = (TimeoutError, ConnectionError)        # network-ish problems that may go away on their own
```

- **In simple words:** Only network-style errors are worth retrying.
- **Exactly:** Tuple `(TimeoutError, ConnectionError)`, usable directly in `except`.
- **Why it is here:** Retrying bugs or bad requests wastes time and money.

### 6. Retry with exponential backoff and jitter  (`main.py` lines 59–71)

```python
def call_with_retry(call, max_attempts=4, base_delay=0.05):
    """call: a zero-argument function. Waits base*2^(attempt-1) (+ random jitter) between attempts."""
    for attempt in range(1, max_attempts + 1):
        try:
            return call()
        except RETRYABLE as err:
            if attempt == max_attempts:
                raise                               # out of attempts: re-raise the SAME error
            delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, base_delay)   # 0.05, 0.10, 0.20 ...
            print(f"      retry {attempt}/{max_attempts - 1} after {type(err).__name__}; sleeping {delay:.2f}s")
            time.sleep(delay)
        # any OTHER exception type (ValueError, KeyError...) is a bug or a bad request: it propagates at once.
        # Retrying a permanent error just wastes time and money.
```

- **In simple words:** Try the call; on a network error wait longer each time; give up after the last attempt.
- **Exactly:** `call()` is a zero-argument function. Delay is `base_delay * 2 ** (attempt-1) + random.uniform(0, base_delay)`: about 0.05, 0.10, 0.20 s plus up to 0.05 jitter. Real run: sleeps of 0.08 s and 0.15 s. On the last attempt a bare `raise` re-raises the same error. Other exception types are not caught and propagate immediately.
- **Why it is here:** Backoff gives the server time to recover; jitter avoids a thundering herd; the cap prevents infinite waiting.

### 7. Calculator tool and registry  (`main.py` lines 76–81)

```python
def calculator(a, op, b):
    operations = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
                  "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(operations[op](a, b))               # b == 0 with "/" raises ZeroDivisionError

TOOLS = {"calculator": calculator}
```

- **In simple words:** The same calculator as before.
- **Exactly:** `b == 0` with `/` raises `ZeroDivisionError`; unknown op raises `KeyError`.
- **Why it is here:** Gives scenario C an error to handle.

### 8. Run a tool without ever raising  (`main.py` lines 83–89)

```python
def run_tool(name, args):
    try:
        return TOOLS[name](**args)
    except KeyError as err:
        return f"ERROR: unknown tool or operator {err}"
    except Exception as err:
        return f"ERROR: {type(err).__name__}: {err}"
```

- **In simple words:** Turn tool errors into text.
- **Exactly:** `KeyError` -> `ERROR: unknown tool or operator ...`; any other exception -> `ERROR: <TypeName>: <message>`. Real: `ERROR: ZeroDivisionError: division by zero`.
- **Why it is here:** A raised exception would end the agent; text lets the model react.

### 9. Start the resilient agent  (`main.py` lines 94–96)

```python
def run_agent(question, brain, max_steps=6, max_repairs=2):
    mock = as_mock(brain)
    messages, repairs = [{"role": "user", "content": question}], 0
```

- **In simple words:** Set up messages and counters.
- **Exactly:** `mock = as_mock(brain)`; `messages` begins with the question; `repairs = 0`. Limits: `max_steps=6`, `max_repairs=2`.
- **Why it is here:** Two independent budgets prevent endless loops.

### 10. Layer 1: model call with retries  (`main.py` lines 97–101)

```python
    for step in range(1, max_steps + 1):
        try:
            reply = call_with_retry(lambda: generate(messages, mock=mock))   # lambda = tiny function with no args
        except RETRYABLE as err:
            return f"FAILED: model unavailable after retries ({type(err).__name__}). Try again later."
```

- **In simple words:** Call the model through the retry helper; fail cleanly if it stays down.
- **Exactly:** `lambda: generate(messages, mock=mock)` wraps the call as a zero-arg function. If retries run out, the `TimeoutError` is caught and a `FAILED: ...` string returned (scenario D).
- **Why it is here:** Without the `except`, a permanent outage shows a raw traceback.

### 11. Layer 2: repair invalid JSON  (`main.py` lines 102–112)

```python
        try:
            decision = extract_json(reply)
        except ValueError:                                    # layer 2: JSON repair
            repairs += 1
            if repairs > max_repairs:
                return "FAILED: model keeps returning invalid JSON"
            print(f"      bad JSON: {reply!r} -> asking the model to repair (repair {repairs}/{max_repairs})")
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "Observation: ERROR: your reply was not valid JSON. "
                                                     'Reply with ONLY {"tool":..,"args":..} or {"final":..}.'}]
            continue
```

- **In simple words:** If the reply is not JSON, tell the model and ask again.
- **Exactly:** `extract_json` raises `ValueError` on no JSON. Count a repair; above `max_repairs` give up. Otherwise append the bad reply and an Observation error message, then `continue` to the next step. Real: `bad JSON: 'Sure! I will ...' -> asking the model to repair (repair 1/2)`.
- **Why it is here:** Real models sometimes add chatter around JSON; one polite re-ask usually fixes it.

### 12. Layer 3: run the tool, feed back the result  (`main.py` lines 113–119)

```python
        messages.append({"role": "assistant", "content": reply})
        if "final" in decision:
            return decision["final"]
        result = run_tool(decision["tool"], decision["args"])  # layer 3: never raises
        print(f"      tool {decision['tool']} -> {result}")
        messages.append({"role": "user", "content": f"Observation: {result}"})
    return "STOPPED: max_steps"
```

- **In simple words:** Normal loop: stop on final, otherwise run the tool and return the observation.
- **Exactly:** `run_tool` never raises. Scenario C: result `ERROR: ZeroDivisionError: division by zero`, and the fake brain answers `Sorry, the tool failed: ...`. Falls through to `STOPPED: max_steps` if never final.
- **Why it is here:** Completes the loop while keeping every failure as data.

### 13. Run four scenarios  (`main.py` lines 121–130)

```python
if __name__ == "__main__":
    scenarios = [
        ("A flaky network (2 timeouts, then OK)", "What is 6 x 7?", FlakyLLM(fail_times=2)),
        ("B invalid JSON once",                   "What is 6 x 7?", BadJsonOnceLLM()),
        ("C tool error (divide by zero)",         "What is 5 / 0?", fake_reply),
        ("D permanent outage",                    "What is 6 x 7?", FlakyLLM(fail_times=99)),
    ]
    for title, question, brain in scenarios:
        print(f"\n{title}\n  Q: {question}")
        print("  A:", run_agent(question, brain))
```

- **In simple words:** Flaky network, bad JSON, tool error, permanent outage.
- **Exactly:** Real: A recovers after 2 retries -> `Result: 42`; B repaired once -> `Result: 42`; C -> failure text from tool error; D -> 3 retries then `FAILED: model unavailable after retries (TimeoutError). Try again later.`
- **Why it is here:** Each scenario proves one defence.

### 14. Rules of thumb and experiments  (`main.py` lines 132–135)

```python
# Rules of thumb: retry only TRANSIENT errors (timeouts, 429 rate limit, 5xx); never retry 4xx bugs (400/401/404).
# Always cap attempts AND total steps. Add JITTER so many clients do not retry at the same instant.
# TRY IT: raise base_delay to 0.5 and time the run. TRY IT: set max_repairs=0.
# BREAK IT: remove the `except RETRYABLE` in run_agent and run scenario D -> read the TimeoutError traceback.
```

- **In simple words:** Advice in comments.
- **Exactly:** Comments only: retry only 429/5xx/timeouts, never 400/401/404; cap attempts and steps; add jitter.
- **Why it is here:** Interview-ready summary of retry policy.

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
