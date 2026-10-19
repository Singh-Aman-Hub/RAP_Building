"""
DRILL 2 -- EXPECTED TO CRASH.   Bug: tool arguments from the model are used without validation.
RUN: python broken_unvalidated_args.py    Read the LAST line of the traceback first, then walk upwards.

The model sent a=\"6\" (a string) instead of 6.  Python happily calls the function and fails INSIDE it.
Worse variant (try it): with op "*" the call "6" * 7 does NOT crash - it silently returns '6666666'.
FIX: validate names, types and allowed values BEFORE calling (step 03) and return the problem as an Observation.
"""
from _common import calculator, MockLLM, generate, extract_json

llm_output = '{"tool": "calculator", "args": {"a": "6", "op": "+", "b": 7}}'     # what the "model" produced
decision = extract_json(llm_output)
print("calling:", decision)
print(calculator(**decision["args"]))       # BUG: no validation -> TypeError: can only concatenate str (not "int") to str
