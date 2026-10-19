"""
DRILL 1 -- EXPECTED TO CRASH (with a safety net).   Bug: `while True` with no step limit / stop condition.
RUN: python broken_no_max_steps.py        Read the traceback, then FIX it with a bounded loop.

Scenario: the model keeps asking for the same tool (a real model can do this when a tool result confuses it).
Without max_steps, a real agent would loop forever and burn money.  We added a safety net so YOUR laptop
survives: after 25 iterations we raise RuntimeError.  Production code needs a REAL limit, not a safety net.
"""
from _common import calculator, extract_json, MockLLM, generate

always_same_tool = MockLLM(rules=[(r"", '{"tool": "calculator", "args": {"a": 1, "op": "+", "b": 1}}')])

messages = [{"role": "user", "content": "What is 1 + 1?"}]
iterations = 0
while True:                                             # BUG: nothing ever makes this loop stop
    iterations += 1
    if iterations > 25:                                 # safety net for the drill only
        raise RuntimeError(f"runaway loop detected after {iterations} iterations: the harness has no max_steps!")
    decision = extract_json(generate(messages, mock=always_same_tool))
    if "final" in decision:                             # the model never says 'final' here
        break
    result = calculator(**decision["args"])
    messages.append({"role": "user", "content": f"Observation: {result}"})

# FIX HINTS: (1) `for step in range(max_steps):`  (2) return a clear 'STOPPED' message after the loop
#            (3) detect repetition: if the same (tool, args) repeats 3 times, stop early and say why.
