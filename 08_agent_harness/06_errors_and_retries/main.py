"""
GOAL   : Make the agent resilient: retry transient network failures with exponential backoff, ask the
         model to repair invalid JSON, and turn tool errors into observations instead of crashes.
RUN    : cd 08_agent_harness/06_errors_and_retries && python main.py
EXPECTED OUTPUT: four scenarios - flaky network (recovers after 2 retries), bad JSON (repaired),
         tool error (reported to model), permanent outage (clean failure message).
NEEDS  : nothing (mock; failures are injected by wrapper classes).
"""
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json, MockLLM
from fake_llm import fake_reply

# ---------------------------------------------------------------------------
# Failure injection: fake LLMs that misbehave on purpose (so we can test our defences)
# ---------------------------------------------------------------------------
class FlakyLLM:
    """Raises TimeoutError for the first `fail_times` calls, then behaves."""
    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0
    def __call__(self, messages, system=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("simulated: request timed out")
        return fake_reply(messages, system)

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

def as_mock(callable_brain):
    return MockLLM(rules=[(r"", callable_brain)])

# ---------------------------------------------------------------------------
# Layer 1: retry with EXPONENTIAL BACKOFF + jitter for TRANSIENT errors only
# ---------------------------------------------------------------------------
RETRYABLE = (TimeoutError, ConnectionError)        # network-ish problems that may go away on their own

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

# ---------------------------------------------------------------------------
# Tools + safe execution (tool errors become observations)
# ---------------------------------------------------------------------------
def calculator(a, op, b):
    operations = {"+": lambda x, y: x + y, "-": lambda x, y: x - y,
                  "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(operations[op](a, b))               # b == 0 with "/" raises ZeroDivisionError

TOOLS = {"calculator": calculator}

def run_tool(name, args):
    try:
        return TOOLS[name](**args)
    except KeyError as err:
        return f"ERROR: unknown tool or operator {err}"
    except Exception as err:
        return f"ERROR: {type(err).__name__}: {err}"

# ---------------------------------------------------------------------------
# The loop with: retries (layer 1), JSON repair (layer 2), tool errors as data (layer 3)
# ---------------------------------------------------------------------------
def run_agent(question, brain, max_steps=6, max_repairs=2):
    mock = as_mock(brain)
    messages, repairs = [{"role": "user", "content": question}], 0
    for step in range(1, max_steps + 1):
        try:
            reply = call_with_retry(lambda: generate(messages, mock=mock))   # lambda = tiny function with no args
        except RETRYABLE as err:
            return f"FAILED: model unavailable after retries ({type(err).__name__}). Try again later."
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
        messages.append({"role": "assistant", "content": reply})
        if "final" in decision:
            return decision["final"]
        result = run_tool(decision["tool"], decision["args"])  # layer 3: never raises
        print(f"      tool {decision['tool']} -> {result}")
        messages.append({"role": "user", "content": f"Observation: {result}"})
    return "STOPPED: max_steps"

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

# Rules of thumb: retry only TRANSIENT errors (timeouts, 429 rate limit, 5xx); never retry 4xx bugs (400/401/404).
# Always cap attempts AND total steps. Add JITTER so many clients do not retry at the same instant.
# TRY IT: raise base_delay to 0.5 and time the run. TRY IT: set max_repairs=0.
# BREAK IT: remove the `except RETRYABLE` in run_agent and run scenario D -> read the TimeoutError traceback.
