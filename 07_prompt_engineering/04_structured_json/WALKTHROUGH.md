# Step 04 — Structured output: JSON, schemas, validation, retry

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

Retry demo (the mock is SCRIPTED to fail twice, then behave):
  attempt 1 raw reply: 'I think it is positive!'
  attempt 2 raw reply: '{"label": "happy", "confidence": 0.9}'
  attempt 3 raw reply: '{"label": "positive", "confidence": 0.9}'
  result object: label='positive' confidence=0.9 | as dict: {'label': 'positive', 'confidence': 0.9}
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
| `obj.model_dump()` | pydantic v2 | object -> dict | – | dict |
| `MockLLM(script=[...])` | shared helper | returns canned replies in order | list | – |

## Execution trace
Attempt 1 -> no `{`, `extract_json` raises `ValueError` -> feedback turn appended. Attempt 2 -> valid JSON but `happy` is not in the `Literal` -> `ValidationError` -> feedback again. Attempt 3 -> valid -> returns a `Sentiment` object. The caller gets either a *valid object* or a loud `RuntimeError`, never garbage.

## Experiments
- `max_attempts=2` -> `RuntimeError: no valid JSON after 2 attempts`.
- Add `reason: str` to `Sentiment` -> "valid" case now fails (missing field) — schema changes must match the prompt.
- Print `Sentiment.model_json_schema()` — pydantic can produce a JSON Schema you can paste into the prompt or an API's structured-output feature.

## Debugging practice
`json.loads("{'label': 'x'}")` -> `JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2`. Read it: *line/column* point to the first bad character; the message says what was expected. JSON needs double quotes.

## Rebuild checklist
- [ ] A hand validator (type, range, allowed values, extra keys).
- [ ] A pydantic model with `Literal` + `Field`.
- [ ] A bounded retry loop with error feedback.

## Say it aloud
"I ask for JSON with an explicit schema, extract it from the reply, then validate it — by hand or with pydantic — before using it. If it is invalid I retry with the error message, with a hard attempt limit. Limitation: validation proves the *shape* is right, not that the *content* is true."
