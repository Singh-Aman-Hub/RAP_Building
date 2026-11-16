# Step 04 — Structured output: JSON, schemas, validation, retry

## The story so far
In [step 03](../03_prompt_templates/) you built clean, versioned prompts. The replies are still prose, and code cannot safely read prose.

## The pain point
Ask for JSON and the model sometimes adds chatter ("Sure! Here you go"), uses single quotes, invents a label like `happy`, or returns a confidence of 7. One bad reply that slips through crashes your program or, worse, corrupts your data quietly.

## The spark
Never trust the model's output; check it. Declare the schema as a Pydantic `BaseModel`, validate every reply against it, and on failure send the *exact* validation error back to the model and retry a bounded number of times. Pydantic is the enforcement mechanism, not the prompt.

## Goal
Get machine-readable output from a model and make sure bad output can never flow into the rest of your program.

## Run it
```bash
cd 07_prompt_engineering/04_structured_json && ../../.venv/bin/python main.py
```
Real output:
```
case                 by hand                             pydantic
valid                ok                                  ok
bad label            REJECT: label 'happy' not in ('pos  REJECT (1 error)
confidence 7         REJECT: confidence must be a numbe  REJECT (1 error)
confidence 'high'    REJECT: confidence must be a numbe  REJECT (1 error)
missing key          REJECT: missing key 'label'         REJECT (1 error)
chatter around it    ok                                  ok

Real ValidationError for a reply with an illegal label AND an out-of-range confidence:
2 validation errors for Sentiment
label
  Input should be 'positive', 'negative', 'neutral' or 'unknown' [type=literal_error, input_value='happy', input_type=str]
    For further information visit https://errors.pydantic.dev/2.14/v/literal_error
confidence
  Input should be less than or equal to 1 [type=less_than_equal, input_value=7, input_type=int]
    For further information visit https://errors.pydantic.dev/2.14/v/less_than_equal
  -> error_count(): 2 | first error loc: ('label',)

Retry demo (the mock is SCRIPTED to fail twice, then behave):
  attempt 1 raw reply: 'I think it is positive!'
  feedback sent to model: Your reply was not parseable JSON (No JSON object found in model reply: 'I think it is positive!'). Return ONL
  attempt 2 raw reply: '{"label": "happy", "confidence": 0.9}'
  feedback sent to model: Your JSON failed validation -> label: Input should be 'positive', 'negative', 'neutral' or 'unknown'. Return O
  attempt 3 raw reply: '{"label": "positive", "confidence": 0.9}'
  result object: label='positive' confidence=0.9 | as dict: {'label': 'positive', 'confidence': 0.9}

Simulator, JSON requested:
  raw: {"label": "negative", "confidence": 0.9} -> label='negative' confidence=0.9
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Structured output | JSON mode, schema-constrained decoding, structured outputs (API feature) | Near | Asking for JSON in the prompt does not guarantee it; API-level constrained decoding does |
| Validation | schema check, parsing + checking | Near | Parsing only checks syntax; validation checks keys, types, ranges |
| pydantic BaseModel | schema class, data model | Yes | A class that declares fields and validates dicts into objects |
| Literal | enum, closed set | Near | Allows only the listed exact values |
| Retry with feedback | repair loop, self-correction loop | Yes | Send the error back to the model and ask again |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–8)

```python
"""
GOAL   : Ask the model for JSON, then NEVER trust it: parse, validate by hand AND with
         pydantic, and retry with error feedback when the reply is invalid.
RUN    : cd 07_prompt_engineering/04_structured_json && python main.py
EXPECTED OUTPUT: a failed first attempt (prose, wrong label), an automatic retry that succeeds,
         and a table showing which validator accepts/rejects which JSON.
NEEDS  : pydantic (installed in .venv).
"""
```

- **In simple words:** Says: ask for JSON, never trust it, validate, retry.
- **Exactly:** Documents expected output: failed first attempt, retry succeeds, comparison table.
- **Why it is here:** Orients the reader.

### 2. Imports and path to shared client  (`main.py` lines 9–17)

```python
import json
import sys
from pathlib import Path

# shared/ lives two folders up from this file: <repo>/shared
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from llm_client import generate, extract_json, MockLLM
from pydantic import BaseModel, Field, ValidationError   # pydantic: 3rd-party validation library
from typing import Literal
```

- **In simple words:** Load helpers: the shared LLM client and pydantic.
- **Exactly:** `parents[2]/"shared"` is `<repo>/shared`. `extract_json` finds a `{...}` object inside text and `json.loads` it; raises `ValueError` if none. `BaseModel, Field, ValidationError` come from pydantic v2 (errors link to errors.pydantic.dev/2.14).
- **Why it is here:** Without the path insert `llm_client` cannot be imported.

### 3. Allowed labels constant  (`main.py` lines 19–19)

```python
ALLOWED = ("positive", "negative", "neutral", "unknown")
```

- **In simple words:** The only labels we accept.
- **Exactly:** A `tuple` of 4 strings, including `"unknown"` for empty input.
- **Why it is here:** One source of truth for the hand validator.

### 4. Hand-written validator  (`main.py` lines 24–38)

```python
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
```

- **In simple words:** Check each rule by hand: dict, has label, label allowed, confidence number in 0..1, no extra keys.
- **Exactly:** Raises `ValueError` with a message per rule. `isinstance(conf, bool)` is excluded because `True` is an `int` in Python. Returns the same dict on success.
- **Why it is here:** Shows exactly what validation means before a library does it for you.

### 5. pydantic schema  (`main.py` lines 43–45)

```python
class Sentiment(BaseModel):                       # a class whose attributes are the fields
    label: Literal["positive", "negative", "neutral", "unknown"]   # only these exact values
    confidence: float = Field(ge=0, le=1)         # ge/le = greater-or-equal / less-or-equal
```

- **In simple words:** Declare what a valid reply looks like and let the library check it.
- **Exactly:** `Literal[...]` accepts only those four strings. `Field(ge=0, le=1)` means 0 <= confidence <= 1. `float` also coerces `7` int to 7.0 and then rejects it by range; `"high"` fails type check.
- **Why it is here:** Fewer lines than the hand validator and gives structured error details.

### 6. Six sample model replies  (`main.py` lines 48–55)

```python
CASES = {
    "valid":             '{"label": "positive", "confidence": 0.9}',
    "bad label":         '{"label": "happy", "confidence": 0.9}',
    "confidence 7":      '{"label": "positive", "confidence": 7}',
    "confidence 'high'": '{"label": "positive", "confidence": "high"}',
    "missing key":       '{"confidence": 0.5}',
    "chatter around it": 'Sure! {"label": "neutral", "confidence": 0.4} hope it helps',
}
```

- **In simple words:** Make-believe model answers, some good and some bad.
- **Exactly:** `CASES` is a `dict[str, str]` of raw reply strings: valid, bad label, confidence 7, confidence 'high', missing key, chatter around valid JSON.
- **Why it is here:** You test the validators on realistic failures.

### 7. Compare both validators on each case  (`main.py` lines 56–71)

```python
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
```

- **In simple words:** Push every reply through parse, hand check and pydantic check and print a table.
- **Exactly:** `extract_json(raw)` parses first. `Sentiment.model_validate(data)` returns an object or raises `ValidationError`; `err.error_count()` counts failures. Real table: `valid` ok/ok; `bad label` REJECT/REJECT(1 error); `chatter around it` ok/ok because `extract_json` strips the chatter.
- **Why it is here:** Shows both validators agree and why parsing and validation are two steps.

### 8. Show a real ValidationError  (`main.py` lines 73–81)

```python
# ---------------------------------------------------------------------------
# What a REAL pydantic ValidationError looks like (read it: field path, message, offending input).
# ---------------------------------------------------------------------------
print("\nReal ValidationError for a reply with an illegal label AND an out-of-range confidence:")
try:
    Sentiment.model_validate({"label": "happy", "confidence": 7})
except ValidationError as err:
    print(err)                                   # str(err) lists every failing field
    print("  -> error_count():", err.error_count(), "| first error loc:", err.errors()[0]["loc"])
```

- **In simple words:** Print what pydantic says when two things are wrong.
- **Exactly:** Real output starts `2 validation errors for Sentiment`, lists `label` (literal_error, input 'happy') and `confidence` (less_than_equal, input 7). `err.errors()[0]["loc"]` is `('label',)`.
- **Why it is here:** You must be able to read these messages: field path, message, input value.

### 9. Turn an error into feedback for the model  (`main.py` lines 83–88)

```python
def feedback_from(err: Exception) -> str:
    """Turn an error into a short, model-readable instruction (field + problem, no stack trace)."""
    if isinstance(err, ValidationError):
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in err.errors())
        return f"Your JSON failed validation -> {problems}. Return ONLY corrected JSON."
    return f"Your reply was not parseable JSON ({err}). Return ONLY a JSON object."
```

- **In simple words:** Convert the technical error to a short instruction the model can follow.
- **Exactly:** For `ValidationError` joins `loc: msg` for each error. Otherwise reports unparseable JSON. Always ends with "Return ONLY corrected JSON".
- **Why it is here:** Raw stack traces waste tokens and confuse the model.

### 10. Retry loop with error feedback  (`main.py` lines 90–106)

```python
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
```

- **In simple words:** Ask; if the reply is bad, show the model its mistake and ask again, up to three times.
- **Exactly:** `range(1, max_attempts + 1)` gives 1,2,3. Success returns a `Sentiment`. On failure the loop appends an assistant message (the bad reply) and a user message (the feedback), so the next call sees the history. After 3 failures raises `RuntimeError`.
- **Why it is here:** Fixes most transient format errors, and the cap stops infinite loops and cost.

### 11. Scripted retry demo  (`main.py` lines 108–115)

```python
print("\nRetry demo (the mock is SCRIPTED to fail twice, then behave):")
scripted = MockLLM(script=[
    "I think it is positive!",                              # no JSON at all
    '{"label": "happy", "confidence": 0.9}',                # JSON but illegal label
    '{"label": "positive", "confidence": 0.9}',             # finally valid
])
result = classify_json("I love it", scripted)
print("  result object:", result, "| as dict:", result.model_dump())
```

- **In simple words:** A fake model that is set up to fail twice and then answer correctly.
- **Exactly:** `MockLLM(script=[...])` returns the three strings in order. Real run: attempt 1 prose (not parseable), attempt 2 `label: "happy"` (rejected), attempt 3 valid. Result `label='positive' confidence=0.9`, `model_dump()` gives `{'label': 'positive', 'confidence': 0.9}`.
- **Why it is here:** Deterministic way to test the retry code without a flaky real model.

### 12. Same pipeline with the simulator  (`main.py` lines 117–123)

```python
# Same pipeline against the repo's rule-based simulator (sim_llm.py): it answers in JSON when asked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim_llm import make_sim                     # noqa: E402
print("\nSimulator, JSON requested:")
sim_reply = generate([{"role": "user", "content": "Text: The delivery was slow and the box was broken."}],
                     system='Return ONLY JSON {"label": ..., "confidence": ...}', mock=make_sim())
print("  raw:", sim_reply, "->", Sentiment.model_validate(extract_json(sim_reply)))
```

- **In simple words:** Ask the rule-based simulator for JSON and validate the answer.
- **Exactly:** `make_sim()` replies `{"label": "negative", "confidence": 0.9}` when the system prompt asks for JSON; validated into `Sentiment`.
- **Why it is here:** Shows the full path: prompt, raw JSON, typed object.

### 13. TRY IT / BREAK IT  (`main.py` lines 125–128)

```python
# TRY IT: set max_attempts=2 and watch the RuntimeError.
# TRY IT: add a `reason: str` field to Sentiment and see which cases now fail.
# BREAK IT: Sentiment.model_validate({"label": "happy", "confidence": 7})  -> the ValidationError printed above.
# BREAK IT: json.loads("{'label': 'x'}")  -> single quotes are not valid JSON (JSONDecodeError).
```

- **In simple words:** Experiments.
- **Exactly:** Comments. `json.loads("{'label': 'x'}")` raises `JSONDecodeError` (single quotes are invalid JSON).
- **Why it is here:** Practice.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class Sentiment(BaseModel):` | a class that *inherits* pydantic's checking behaviour | `class A(Base): x: int` |
| `label: Literal["a","b"]` | type hint: only these exact values | `x: Literal["a"] = "a"` |
| `Field(ge=0, le=1)` | extra constraints: >= 0 and <= 1 | `Field(gt=0)` |
| `isinstance(x, (int, float))` | is `x` any of these types? | `isinstance(1, int)` -> True |
| `set(data) - {"label"}` | set difference: keys that are not allowed | `{1,2} - {1}` -> `{2}` |
| `for attempt in range(1, max_attempts + 1)` | counts 1..max_attempts | – |
| `except (ValueError, ValidationError) as err` | catch several exception types | – |
| `{err.error_count()}` | method call inside an f-string | – |

## Libraries & functions
| Name | Kind | Does | Args | Returns |
|---|---|---|---|---|
| `json.loads` (inside `extract_json`) | standard library | text -> Python dict | str | dict; `JSONDecodeError` if invalid |
| `extract_json` | shared helper | finds first `{...}` in chatty text and parses it | str | dict |
| `pydantic.BaseModel` | 3rd-party | declare a schema as a class | – | – |
| `Model.model_validate(dict)` | pydantic v2 | validate + build object | dict | model instance; `ValidationError` |
| `err.errors()` | pydantic v2 | list of dicts, one per failing field (`loc`, `msg`) | – | list |
| `obj.model_dump()` | pydantic v2 | object -> dict | – | dict |
| `MockLLM(script=[...])` | shared helper | returns canned replies in order | list | – |

## Execution trace
Attempt 1 -> no `{`, `extract_json` raises `ValueError` -> feedback turn appended. Attempt 2 -> valid JSON but `happy` is not in the `Literal` -> `ValidationError` -> feedback again. Attempt 3 -> valid -> returns a `Sentiment` object. The caller gets either a *valid object* or a loud `RuntimeError`, never garbage.

## Experiments
- Run the simulator at the end: `sim_llm.make_sim()` answers in JSON when the prompt asks for it, and the same `Sentiment.model_validate` accepts it.
- `max_attempts=2` -> `RuntimeError: no valid JSON after 2 attempts`.
- Add `reason: str` to `Sentiment` -> "valid" case now fails (missing field) — schema changes must match the prompt.
- Print `Sentiment.model_json_schema()` — pydantic can produce a JSON Schema you can paste into the prompt or an API's structured-output feature.

## Debugging practice
The enforcement is Pydantic: read the real `ValidationError` it printed above. It names each failing field (`label`, `confidence`), says what was expected, and echoes the bad `input_value`. `feedback_from` turns exactly that into the retry message, so the model is told *which field* to fix.

`json.loads("{'label': 'x'}")` -> `JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2`. Read it: *line/column* point to the first bad character; the message says what was expected. JSON needs double quotes.

## Rebuild checklist
- [ ] A hand validator (type, range, allowed values, extra keys).
- [ ] A pydantic model with `Literal` + `Field`.
- [ ] A bounded retry loop with error feedback.

## The new problem (cliffhanger)
Replies are now reliable and typed. But one prompt can still hide a hard task inside a single call, and the model gets it wrong without telling you which part failed. Next: [05_decomposition](../05_decomposition/).

> **Say it aloud (interview box):** I ask for JSON, declare the schema as a Pydantic model, and validate every reply before using it. When validation fails I retry with the real error message, up to a hard limit, and then raise instead of guessing. Validation proves the shape is right, not that the content is true.
