"""
DRILL 3 -- EXPECTED TO CRASH.   Bug: parsing model output with json.loads directly and no error handling.
RUN: python broken_bad_json.py    Use the line/column in the error to find the first bad character.

Models often add chatter, use single quotes or forget quotes.  FIX: use extract_json inside try/except,
and on failure send the error back to the model (repair loop - step 06) with a retry cap.
"""
import json

reply = "Sure! Here is my action: {'tool': 'calculator', 'args': {'a': 1, 'op': '+', 'b': 2}}"
print("model said:", reply)
decision = json.loads(reply)                # BUG: (1) leading chatter  (2) single quotes -> JSONDecodeError
print(decision)
