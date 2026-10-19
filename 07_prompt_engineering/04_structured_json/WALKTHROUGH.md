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
