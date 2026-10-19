"""
GOAL   : Ask the model for JSON, then NEVER trust it: parse, validate by hand AND with
         pydantic, and retry with error feedback when the reply is invalid.
RUN    : cd 07_prompt_engineering/04_structured_json && python main.py
EXPECTED OUTPUT: a failed first attempt (prose, wrong label), an automatic retry that succeeds,
         and a table showing which validator accepts/rejects which JSON.
NEEDS  : pydantic (installed in .venv).
"""
import json
import sys
from pathlib import Path

# shared/ lives two folders up from this file: <repo>/shared
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from llm_client import generate, extract_json, MockLLM
from pydantic import BaseModel, Field, ValidationError   # pydantic: 3rd-party validation library
from typing import Literal

ALLOWED = ("positive", "negative", "neutral", "unknown")

# ---------------------------------------------------------------------------
# Validator 1: hand-written. Shows exactly what "validation" means.
# ---------------------------------------------------------------------------
def validate_by_hand(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("reply must be a JSON object")
    if "label" not in data:
        raise ValueError("missing key 'label'")
    if data["label"] not in ALLOWED:
        raise ValueError(f"label {data['label']!r} not in {ALLOWED}")
    conf = data.get("confidence", 0.0)
    # bool is a subclass of int in Python, so exclude it explicitly
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
        raise ValueError("confidence must be a number between 0 and 1")
    extra = set(data) - {"label", "confidence"}
    if extra:
        raise ValueError(f"unexpected keys: {sorted(extra)}")
    return data

# ---------------------------------------------------------------------------
# Validator 2: pydantic. You DECLARE the schema; the library does the checking.
# ---------------------------------------------------------------------------
class Sentiment(BaseModel):                       # a class whose attributes are the fields
    label: Literal["positive", "negative", "neutral", "unknown"]   # only these exact values
    confidence: float = Field(ge=0, le=1)         # ge/le = greater-or-equal / less-or-equal

# Compare both validators on a few hand-made "model replies".
CASES = {
    "valid":             '{"label": "positive", "confidence": 0.9}',
    "bad label":         '{"label": "happy", "confidence": 0.9}',
    "confidence 7":      '{"label": "positive", "confidence": 7}',
    "confidence 'high'": '{"label": "positive", "confidence": "high"}',
    "missing key":       '{"confidence": 0.5}',
    "chatter around it": 'Sure! {"label": "neutral", "confidence": 0.4} hope it helps',
}
print(f"{'case':20s} {'by hand':35s} pydantic")
for name, raw in CASES.items():
    try:
        data = extract_json(raw)                 # step 1: find + parse JSON in the text
    except ValueError as err:
        print(f"{name:20s} PARSE FAIL {err}")
        continue
    try:
        validate_by_hand(data); hand = "ok"
    except ValueError as err:
        hand = f"REJECT: {err}"[:34]
    try:
        Sentiment.model_validate(data); pyd = "ok"      # model_validate(dict) -> Sentiment object
    except ValidationError as err:
        pyd = f"REJECT ({err.error_count()} error)"
    print(f"{name:20s} {hand:35s} {pyd}")

# ---------------------------------------------------------------------------
# What a REAL pydantic ValidationError looks like (read it: field path, message, offending input).
# ---------------------------------------------------------------------------
print("\nReal ValidationError for a reply with an illegal label AND an out-of-range confidence:")
try:
    Sentiment.model_validate({"label": "happy", "confidence": 7})
except ValidationError as err:
    print(err)                                   # str(err) lists every failing field
    print("  -> error_count():", err.error_count(), "| first error loc:", err.errors()[0]["loc"])

def feedback_from(err: Exception) -> str:
    """Turn an error into a short, model-readable instruction (field + problem, no stack trace)."""
    if isinstance(err, ValidationError):
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors())
        return f"Your JSON failed validation -> {problems}. Return ONLY corrected JSON."
    return f"Your reply was not parseable JSON ({err}). Return ONLY a JSON object."

# ---------------------------------------------------------------------------
# Retry loop: on invalid output, send the error BACK to the model and ask again.
# ---------------------------------------------------------------------------
def classify_json(text: str, mock: MockLLM, max_attempts: int = 3) -> Sentiment:
    system = 'Return ONLY JSON: {"label": "positive|negative|neutral|unknown", "confidence": 0-1}'
    messages = [{"role": "user", "content": f"Text: {text}"}]
    for attempt in range(1, max_attempts + 1):          # range(1, 4) -> 1, 2, 3
        reply = generate(messages, system=system, mock=mock)
        print(f"  attempt {attempt} raw reply: {reply!r}")
        try:
            return Sentiment.model_validate(extract_json(reply))
        except (ValueError, ValidationError) as err:    # ValidationError is a ValueError in pydantic v2
            # Feedback turn: show the model its own answer + what was wrong.
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": feedback_from(err)}]
            print(f"  feedback sent to model: {feedback_from(err)[:110]}")
    raise RuntimeError(f"no valid JSON after {max_attempts} attempts")   # fail loudly, never guess

print("\nRetry demo (the mock is SCRIPTED to fail twice, then behave):")
scripted = MockLLM(script=[
    "I think it is positive!",                              # no JSON at all
    '{"label": "happy", "confidence": 0.9}',                # JSON but illegal label
    '{"label": "positive", "confidence": 0.9}',             # finally valid
])
result = classify_json("I love it", scripted)
print("  result object:", result, "| as dict:", result.model_dump())

# Same pipeline against the repo's rule-based simulator (sim_llm.py): it answers in JSON when asked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_llm import make_sim                     # noqa: E402
print("\nSimulator, JSON requested:")
sim_reply = generate([{"role": "user", "content": "Text: The delivery was slow and the box was broken."}],
                     system='Return ONLY JSON {"label": ..., "confidence": ...}', mock=make_sim())
print("  raw:", sim_reply, "->", Sentiment.model_validate(extract_json(sim_reply)))

# TRY IT: set max_attempts=2 and watch the RuntimeError.
# TRY IT: add a `reason: str` field to Sentiment and see which cases now fail.
# BREAK IT: Sentiment.model_validate({"label": "happy", "confidence": 7})  -> the ValidationError printed above.
# BREAK IT: json.loads("{'label': 'x'}")  -> single quotes are not valid JSON (JSONDecodeError).
