# Terms — Step 06: Error handling and retries

Folder: `08_agent_harness/06_errors_and_retries` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Transient vs permanent error](#transient-vs-permanent-error) · [Retry](#retry) · [Exponential backoff](#exponential-backoff) · [Jitter](#jitter) · [call_with_retry](#call_with_retry) · [RETRYABLE (tuple of exception classes)](#retryable-tuple-of-exception-classes) · [TimeoutError and ConnectionError](#timeouterror-and-connectionerror) · [Re-raise (raise alone)](#re-raise-raise-alone) · [time.sleep](#timesleep) · [random.uniform](#randomuniform) · [Exponent operator (**)](#exponent-operator) · [Failure injection](#failure-injection) · [Class with __init__ and __call__](#class-with-__init__-and-__call__) · [Instance attribute update (self.calls += 1)](#instance-attribute-update-selfcalls--1) · [JSON repair](#json-repair) · [extract_json and ValueError](#extract_json-and-valueerror) · [Tool errors as observations](#tool-errors-as-observations) · [Layered defences](#layered-defences) · [type(err).__name__](#typeerr__name__) · [Lambda as a zero-argument wrapper](#lambda-as-a-zero-argument-wrapper) · [continue](#continue) · [Step and repair caps](#step-and-repair-caps) · [MockLLM with a callable rule (as_mock)](#mockllm-with-a-callable-rule-as_mock) · [fake_reply](#fake_reply) · [if __name__ == "__main__"](#if-__name__--__main__)

---

## Transient vs permanent error
*Type: concept*

**1. Definition:** A transient error (timeout, rate limit 429, server 5xx) may disappear if you try again; a permanent error (bad request 400, unauthorised 401, not found 404, a code bug) will not.
**2. In plain words:** Retry only the first kind. Retrying a permanent error wastes time and money and hides the bug.
**3. Example to quote:**
> "A timeout is worth retrying; a 400 means my request is wrong, so I fix it instead of retrying."

**Seen in:** `main.py`, comments in `call_with_retry` and the closing rules of thumb.

## Retry
*Type: concept*

**1. Definition:** Repeating a failed operation a limited number of times.
**2. In plain words:** Networks are flaky, so one failure should not end the agent. Always cap attempts so a permanent outage cannot loop forever.
**3. Example to quote:**
> "The flaky network scenario fails twice and succeeds on the third attempt, and the agent answers 42."

**Seen in:** `main.py`, `call_with_retry`.

## Exponential backoff
*Type: concept*

**1. Definition:** Increasing the wait between retries geometrically, here `base_delay * 2 ** (attempt - 1)`: 0.05, 0.10, 0.20 seconds.
**2. In plain words:** Waiting longer each time gives an overloaded server room to recover instead of hammering it. The trade-off is longer waits for the user.
**3. Example to quote:**
> "After the first failure it sleeps 0.05 s, after the second about 0.10 s, doubling each time."

**Seen in:** `main.py`, `call_with_retry`.

## Jitter
*Type: concept*

**1. Definition:** A small random amount added to each backoff delay, here `random.uniform(0, base_delay)`.
**2. In plain words:** If many clients fail at once and retry on the same schedule they all hit the server at the same moment. Randomness spreads them out. It is also why printed sleep times vary between runs.
**3. Example to quote:**
> "The sleeps printed 0.05 and 0.11 seconds because jitter adds a random bit to the doubling."

**Seen in:** `main.py`, `call_with_retry`.

## call_with_retry
*Type: project function*

**1. Definition:** A function taking a zero-argument callable, calling it up to `max_attempts` times, catching only `RETRYABLE` errors, and re-raising the last one if attempts run out.
**2. In plain words:** It wraps any risky call without changing it. Other exception types pass straight through.
**3. Example to quote:**
> "`call_with_retry(lambda: generate(messages, mock=mock))` retries the model call on timeouts only."

**Seen in:** `main.py`, `call_with_retry`.

## RETRYABLE (tuple of exception classes)
*Type: Python syntax / concept*

**1. Definition:** `RETRYABLE = (TimeoutError, ConnectionError)` is a tuple that can be used directly in `except RETRYABLE:` to catch any of those types.
**2. In plain words:** It defines in one place which failures count as transient.
**3. Example to quote:**
> "A `ValueError` is not in `RETRYABLE`, so it propagates immediately instead of being retried."

**Seen in:** `main.py`, `RETRYABLE`.

## TimeoutError and ConnectionError
*Type: Python built-in exceptions*

**1. Definition:** `TimeoutError` is raised when an operation takes too long; `ConnectionError` when a network connection fails.
**2. In plain words:** They are typical transient network problems, which is why they are the retryable set.
**3. Example to quote:**
> "The flaky fake model raises `TimeoutError('simulated: request timed out')` for its first calls."

**Seen in:** `main.py`, `FlakyLLM`, `RETRYABLE`.

## Re-raise (`raise` alone)
*Type: Python syntax*

**1. Definition:** A bare `raise` inside an `except` block re-raises the exception being handled with its original traceback.
**2. In plain words:** When attempts run out, the caller should see the real error, not a made-up one.
**3. Example to quote:**
> "On the last attempt `call_with_retry` re-raises the same `TimeoutError`, which `run_agent` turns into a clean failure message."

**Seen in:** `main.py`, `call_with_retry`.

## time.sleep
*Type: Python standard library*

**1. Definition:** `time.sleep(seconds)` pauses the program for that many seconds.
**2. In plain words:** It implements the backoff wait. In an async or multi-user server it would block, so real services use non-blocking waits.
**3. Example to quote:**
> "`time.sleep(delay)` waits before the next attempt."

**Seen in:** `main.py`, `call_with_retry`.

## random.uniform
*Type: Python standard library*

**1. Definition:** `random.uniform(a, b)` returns a random float between `a` and `b`.
**2. In plain words:** It supplies the jitter.
**3. Example to quote:**
> "`random.uniform(0, base_delay)` adds up to 0.05 s of randomness."

**Seen in:** `main.py`, `call_with_retry`.

## Exponent operator (`**`)
*Type: Python syntax*

**1. Definition:** `a ** b` raises `a` to the power `b`.
**2. In plain words:** `2 ** (attempt - 1)` gives 1, 2, 4, 8 for attempts 1, 2, 3, 4, the doubling factor.
**3. Example to quote:**
> "`2 ** 3` is 8, so the fourth attempt would wait eight times the base delay."

```python
2 ** 3   # 8
```

**Seen in:** `main.py`, `call_with_retry`.

## Failure injection
*Type: concept*

**1. Definition:** Deliberately making a dependency fail (timeouts, bad output) to test that your error handling works.
**2. In plain words:** You cannot wait for the real network to fail on cue, so you build fake models that misbehave on purpose.
**3. Example to quote:**
> "`FlakyLLM(fail_times=2)` times out twice then works, proving my retry code recovers."

**Seen in:** `main.py`, `FlakyLLM`, `BadJsonOnceLLM`.

## Class with `__init__` and `__call__`
*Type: Python syntax*

**1. Definition:** A class whose `__init__` sets up instance attributes and whose `__call__` method lets an instance be called like a function.
**2. In plain words:** The fake model needs to remember how many calls it has seen, so it stores a counter on the object while still being usable wherever a function is expected.
**3. Example to quote:**
> "`FlakyLLM` counts calls in `self.calls` and raises `TimeoutError` while the count is at most `fail_times`."

```python
class A:
    def __call__(self): return 1
A()()   # 1
```

**Seen in:** `main.py`, `FlakyLLM`, `BadJsonOnceLLM`.

## Instance attribute update (`self.calls += 1`)
*Type: Python syntax*

**1. Definition:** Increasing an attribute stored on the object, so state persists between calls.
**2. In plain words:** It is the memory of the fake model: how many times it has been asked.
**3. Example to quote:**
> "After each call `self.calls` goes up, so the first two calls fail and the third succeeds."

**Seen in:** `main.py`, `FlakyLLM.__call__`.

## JSON repair
*Type: concept*

**1. Definition:** When the model's reply is not valid JSON, appending the bad reply and an error message to the conversation and asking the model to answer again in the correct format, up to a cap.
**2. In plain words:** A real model can read the complaint and fix itself. The cap (`max_repairs`) prevents endless loops.
**3. Example to quote:**
> "The first reply was 'Sure! I will calculate that', so I asked for ONLY JSON and the second reply gave the calculator call."

**Seen in:** `main.py`, `run_agent`, `BadJsonOnceLLM`.

## extract_json and ValueError
*Type: project function / Python built-in exception*

**1. Definition:** `extract_json(text)` raises `ValueError` when no JSON object is in the text, and `run_agent` catches it to start a repair.
**2. In plain words:** The exception is the signal that the model broke the protocol.
**3. Example to quote:**
> "`except ValueError:` is where the repair counter goes up and the repair request is appended."

**Seen in:** `main.py`, `run_agent`.

## Tool errors as observations
*Type: concept*

**1. Definition:** Catching exceptions from a tool and returning them as `ERROR: ...` text for the model instead of letting the loop crash.
**2. In plain words:** The model can tell the user why something failed or try another approach. In scenario C dividing by zero produces 'Sorry, the tool failed'.
**3. Example to quote:**
> "`5 / 0` returned `ERROR: ZeroDivisionError: division by zero` as an observation, and the final reply explained the failure."

**Seen in:** `main.py`, `run_tool`.

## Layered defences
*Type: concept*

**1. Definition:** Three independent protections: retries for transient errors (layer 1), JSON repair for malformed output (layer 2), and tool errors as data (layer 3).
**2. In plain words:** Each layer handles a different failure and none handles everything, so you combine them with step limits.
**3. Example to quote:**
> "Retries handle timeouts, repair handles bad JSON, and `run_tool` handles tool exceptions."

**Seen in:** `main.py`, section comments; `WALKTHROUGH.md` execution trace.

## type(err).__name__
*Type: Python built-in*

**1. Definition:** `type(obj)` returns an object's class and `.__name__` is the class name as a string.
**2. In plain words:** It lets error messages say `ZeroDivisionError` rather than a long repr.
**3. Example to quote:**
> "`f'ERROR: {type(err).__name__}: {err}'` printed `ERROR: ZeroDivisionError: division by zero`."

```python
type(ValueError()).__name__   # 'ValueError'
```

**Seen in:** `main.py`, `run_tool`, `run_agent`.

## Lambda as a zero-argument wrapper
*Type: Python syntax*

**1. Definition:** `lambda: expression` defines a tiny function with no parameters that evaluates the expression when called.
**2. In plain words:** `call_with_retry` needs something it can call repeatedly, so the model call is wrapped in a lambda.
**3. Example to quote:**
> "`lambda: generate(messages, mock=mock)` delays the model call until the retry helper decides to run it."

**Seen in:** `main.py`, `run_agent`.

## continue
*Type: Python syntax*

**1. Definition:** `continue` skips the rest of the loop body and starts the next iteration.
**2. In plain words:** After a repair request is appended, the loop jumps straight to asking the model again.
**3. Example to quote:**
> "After appending the repair message I `continue` so the loop calls the model again without running the tool code."

**Seen in:** `main.py`, `run_agent`.

## Step and repair caps
*Type: concept*

**1. Definition:** Hard limits on total steps (`max_steps`) and on repairs (`max_repairs`).
**2. In plain words:** Without caps, a confused model plus retries could run indefinitely and cost real money.
**3. Example to quote:**
> "With `max_repairs=0` the invalid-JSON scenario fails immediately with 'model keeps returning invalid JSON'."

**Seen in:** `main.py`, `run_agent` signature; TRY IT comment.

## MockLLM with a callable rule (`as_mock`)
*Type: project helper*

**1. Definition:** `as_mock(brain)` wraps any callable in `MockLLM(rules=[(r"", brain)])` so `generate` can call it.
**2. In plain words:** It lets the flaky and bad-JSON fakes plug into the same `generate` function used elsewhere.
**3. Example to quote:**
> "`mock = as_mock(brain)` makes my failure-injecting class look like a normal model to `generate`."

**Seen in:** `main.py`, `as_mock`.

## fake_reply
*Type: project helper*

**1. Definition:** The scripted brain function from `fake_llm.py` that answers in the JSON tool protocol; the flaky classes call it when they behave.
**2. In plain words:** It provides the "correct" behaviour underneath, so only the failures differ between scenarios.
**3. Example to quote:**
> "After two timeouts `FlakyLLM` delegates to `fake_reply` and the calculator call comes back."

**Seen in:** `main.py`, `FlakyLLM.__call__`; `../fake_llm.py`.

## if __name__ == "__main__"
*Type: Python syntax*

**1. Definition:** A guard that runs the scenarios only when the file is executed directly.
**2. In plain words:** Later steps can import `run_agent` and `call_with_retry` without printing the demo.
**3. Example to quote:**
> "The four scenarios are under the guard, so importing the retry helper is silent."

**Seen in:** `main.py`, bottom section.
