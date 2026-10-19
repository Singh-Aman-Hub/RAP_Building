# Terms — Step 04: Structured output (JSON, validation, retry)

Folder: `07_prompt_engineering/04_structured_json` · Files covered: `main.py`, `WALKTHROUGH.md` (uses `../../shared/llm_client.py`)
Back to module: [07 Prompt engineering README](../README.md)

**Terms in this folder:** [Structured output](#structured-output) · [JSON (JavaScript Object Notation)](#json-javascript-object-notation) · [json.loads](#jsonloads) · [extract_json](#extract_json) · [Schema](#schema) · [Validation](#validation) · [Hand-written validator (validate_by_hand)](#hand-written-validator-validate_by_hand) · [isinstance](#isinstance) · [Set difference](#set-difference) · [pydantic](#pydantic) · [BaseModel](#basemodel) · [Literal (typing)](#literal-typing) · [Field (pydantic)](#field-pydantic) · [model_validate](#model_validate) · [model_dump](#model_dump) · [ValidationError](#validationerror) · [Retry with error feedback](#retry-with-error-feedback) · [Bounded loop (range(1, max_attempts + 1))](#bounded-loop-range1-max_attempts--1) · [except with a tuple of exceptions](#except-with-a-tuple-of-exceptions) · [Scripted MockLLM](#scripted-mockllm) · [RuntimeError (fail loudly)](#runtimeerror-fail-loudly) · [JSONDecodeError](#jsondecodeerror) · [err.errors() and feedback_from](#errerrors-and-feedback_from)

---

## Structured output
*Type: concept*

**1. Definition:** Asking a model to reply in a machine-readable format (such as JSON that follows a schema) instead of free prose.
**2. In plain words:** Programs cannot reliably read "I think it's probably positive!", but they can read `{"label": "positive"}`. The catch: models can still break the format, so you must check what comes back.
**3. Example to quote:**
> "I ask for `{\"label\": ..., \"confidence\": ...}` so my code can read the fields directly, and I validate it before using it."

**Seen in:** `main.py`, `classify_json` system prompt.

## JSON (JavaScript Object Notation)
*Type: concept / data format*

**1. Definition:** A text format for data using objects (`{}` with double-quoted keys), arrays (`[]`), strings, numbers, booleans and `null`.
**2. In plain words:** It is the common language between programs and APIs. In Python, JSON objects become dicts. It is strict: single quotes are not allowed.
**3. Example to quote:**
> "`{'label': 'x'}` is not valid JSON because it uses single quotes; `json.loads` raises `JSONDecodeError`."

**Seen in:** `main.py`, `CASES`; BREAK IT comment.

## json.loads
*Type: Python standard library*

**1. Definition:** `json.loads(text)` parses a JSON string and returns the matching Python object; invalid text raises `json.JSONDecodeError`.
**2. In plain words:** It converts the model's text into a dict you can use. It is called inside the shared `extract_json` helper after the `{...}` part is cut out of the reply.
**3. Example to quote:**
> "`json.loads('{\"a\": 1}')` returns `{'a': 1}`. If the model returns broken JSON this raises, which my retry loop handles."

**Seen in:** `main.py` (import, and via `extract_json`).

## extract_json
*Type: project function*

**1. Definition:** `shared/llm_client.py::extract_json(text)` finds the first `{` to the last `}` in a reply and parses it as JSON; raises `ValueError` when no JSON object exists.
**2. In plain words:** Models often add chatter like "Sure! Here you go:". This function ignores the chatter and pulls out only the JSON, so parsing does not fail on harmless extra words.
**3. Example to quote:**
> "The case `'Sure! {\"label\": \"neutral\", \"confidence\": 0.4} hope it helps'` still parses because `extract_json` cuts out the braces part."

**Seen in:** `main.py`, parsing step and retry loop.

## Schema
*Type: concept*

**1. Definition:** A formal description of the allowed structure of data: which fields exist, their types and their allowed values or ranges.
**2. In plain words:** It is the contract between you and the model. If the reply breaks the contract it is rejected. You can write it by hand or declare it as a pydantic class.
**3. Example to quote:**
> "My schema says `label` must be one of four strings and `confidence` a number from 0 to 1."

**Seen in:** `main.py`, `ALLOWED` and the `Sentiment` class.

## Validation
*Type: concept*

**1. Definition:** Checking that data satisfies the rules of a schema (types, allowed values, ranges, required and unexpected keys) before it is used.
**2. In plain words:** Parsing only proves the text is valid JSON, not that its content is sensible. Validation catches `{"label": "happy"}` or `confidence: 7`. It proves shape, not truth.
**3. Example to quote:**
> "The reply parsed fine but `happy` is not an allowed label, so validation rejected it and I retried."

**Seen in:** `main.py`, `validate_by_hand` and `Sentiment.model_validate`.

## Hand-written validator (`validate_by_hand`)
*Type: project function*

**1. Definition:** A function that manually checks each rule of the schema with `if` statements and raises `ValueError` on the first violation.
**2. In plain words:** It shows exactly what validation involves: is it a dict, is the key there, is the value allowed, is the number in range, are there extra keys. Libraries like pydantic do this for you.
**3. Example to quote:**
> "My validator raises `ValueError: missing key 'label'` for `{\"confidence\": 0.5}`."

**Seen in:** `main.py`, `validate_by_hand`.

## isinstance
*Type: Python built-in*

**1. Definition:** `isinstance(obj, type_or_tuple)` returns True if the object is an instance of the type (or any type in the tuple).
**2. In plain words:** It is the standard type check. A gotcha used here: `True` counts as an `int` in Python, so the code excludes `bool` explicitly before accepting a number.
**3. Example to quote:**
> "`isinstance(conf, (int, float))` checks confidence is a number, and `isinstance(conf, bool)` blocks `true` from slipping through."

```python
isinstance(1, (int, float))   # True
```

**Seen in:** `main.py`, `validate_by_hand`.

## Set difference
*Type: Python syntax*

**1. Definition:** `a - b` on sets returns the elements in `a` that are not in `b`.
**2. In plain words:** It quickly finds forbidden extra keys: take the reply's keys and subtract the allowed ones; anything left is unexpected.
**3. Example to quote:**
> "`set(data) - {\"label\", \"confidence\"}` is empty for a good reply and contains the extra key otherwise."

```python
{1, 2} - {1}   # {2}
```

**Seen in:** `main.py`, `validate_by_hand`.

## pydantic
*Type: third-party: pydantic*

**1. Definition:** A Python library that validates data against type-annotated model classes and raises `ValidationError` when it does not conform.
**2. In plain words:** You declare what valid data looks like once, and the library checks it for you with clear error messages. FastAPI (module 09) uses the same library to validate requests.
**3. Example to quote:**
> "I replaced thirty lines of manual checks with a four-line pydantic class."

**Seen in:** `main.py`, `from pydantic import BaseModel, Field, ValidationError`.

## BaseModel
*Type: third-party: pydantic*

**1. Definition:** The pydantic base class; subclasses declare fields as annotated attributes and get automatic validation, conversion and serialisation.
**2. In plain words:** Inheriting from it turns a plain class into a validated data schema. Creating or validating an instance checks every field.
**3. Example to quote:**
> "`class Sentiment(BaseModel)` with `label` and `confidence` fields is my schema."

**Seen in:** `main.py`, `class Sentiment`.

## Literal (typing)
*Type: Python standard library (typing)*

**1. Definition:** `Literal["a", "b"]` is a type hint meaning the value must be exactly one of the listed constants.
**2. In plain words:** Pydantic uses it to enforce a closed set of allowed strings, such as the four sentiment labels.
**3. Example to quote:**
> "`label: Literal[\"positive\", \"negative\", \"neutral\", \"unknown\"]` rejects `happy` automatically."

**Seen in:** `main.py`, `Sentiment.label`.

## Field (pydantic)
*Type: third-party: pydantic*

**1. Definition:** A function used as a field's default to attach constraints and metadata, e.g. `Field(ge=0, le=1)` for greater-or-equal 0 and less-or-equal 1.
**2. In plain words:** It adds rules beyond the type. Here it forces confidence into the 0-1 range, so `7` is rejected.
**3. Example to quote:**
> "`confidence: float = Field(ge=0, le=1)` means a confidence of 7 fails validation."

**Seen in:** `main.py`, `Sentiment.confidence`.

## model_validate
*Type: third-party: pydantic v2*

**1. Definition:** A class method, `Model.model_validate(data)`, that validates a dict against the model and returns an instance or raises `ValidationError`.
**2. In plain words:** It is the step that turns the untrusted dict from the model into a trusted object with known fields. After it succeeds you can use `result.label`.
**3. Example to quote:**
> "`Sentiment.model_validate(extract_json(reply))` either gives me a `Sentiment` object or raises an error I can retry on."

**Seen in:** `main.py`, case loop and `classify_json`.

## model_dump
*Type: third-party: pydantic v2*

**1. Definition:** An instance method that converts a model object back into a plain dict.
**2. In plain words:** Use it when you need to log the result, return it from an API or store it as JSON.
**3. Example to quote:**
> "`result.model_dump()` returned `{'label': 'positive', 'confidence': 0.9}`."

**Seen in:** `main.py`, final print.

## ValidationError
*Type: third-party: pydantic*

**1. Definition:** The exception pydantic raises when data fails validation; it holds a list of errors and offers `error_count()`.
**2. In plain words:** Catch it to handle bad model output without crashing. In pydantic v2 it also behaves as a `ValueError`, which is why the file catches both types together.
**3. Example to quote:**
> "`err.error_count()` told me the bad-label reply had one error."

**Seen in:** `main.py`, both `except` clauses.

## Retry with error feedback
*Type: concept / pattern*

**1. Definition:** On invalid output, appending the model's reply and a message describing the error to the conversation and calling the model again, up to a limit.
**2. In plain words:** Telling the model what was wrong often fixes it on the next try. The attempt limit prevents endless loops and costs.
**3. Example to quote:**
> "Attempt one was prose, attempt two had an illegal label, attempt three was valid. After my limit of three I raise instead of guessing."

**Seen in:** `main.py`, `classify_json`.

## Bounded loop (`range(1, max_attempts + 1)`)
*Type: Python syntax*

**1. Definition:** `range(start, stop)` yields integers from `start` up to but not including `stop`, so `range(1, 4)` gives 1, 2, 3.
**2. In plain words:** It gives you a fixed number of attempts and a counter to print. Using `+ 1` makes the last attempt equal `max_attempts`.
**3. Example to quote:**
> "`for attempt in range(1, max_attempts + 1)` retries at most three times."

**Seen in:** `main.py`, `classify_json`.

## except with a tuple of exceptions
*Type: Python syntax*

**1. Definition:** `except (A, B) as err:` catches an exception that is an instance of any listed type.
**2. In plain words:** Parsing errors and validation errors both mean "this reply is unusable", so one handler covers both.
**3. Example to quote:**
> "`except (ValueError, ValidationError) as err` triggers the retry for both bad JSON and wrong fields."

**Seen in:** `main.py`, `classify_json`.

## Scripted MockLLM
*Type: project helper*

**1. Definition:** `MockLLM(script=[...])` from `shared/llm_client.py` returns the listed replies one by one, in order, each time it is called.
**2. In plain words:** It lets you replay a bad reply followed by a good one so you can test your retry code deterministically, without waiting for a real model to misbehave.
**3. Example to quote:**
> "I scripted three replies: prose, an illegal label, then valid JSON, so the retry loop is tested every run."

**Seen in:** `main.py`, `scripted = MockLLM(script=[...])`.

## RuntimeError (fail loudly)
*Type: Python built-in exception*

**1. Definition:** A general exception for errors that do not fit other categories; here raised when all attempts are used up.
**2. In plain words:** After the retries are exhausted, guessing a label would hide the problem. Raising an error makes the caller handle the failure.
**3. Example to quote:**
> "With `max_attempts=2` the script ends with `RuntimeError: no valid JSON after 2 attempts`."

**Seen in:** `main.py`, end of `classify_json`.

## JSONDecodeError
*Type: Python standard library (json)*

**1. Definition:** The exception raised by `json.loads` when the text is not valid JSON; it reports line and column of the first problem.
**2. In plain words:** The message tells you where to look. The classic cause is single quotes or a trailing comma.
**3. Example to quote:**
> "`json.loads(\"{'label': 'x'}\")` gives 'Expecting property name enclosed in double quotes: line 1 column 2'."

**Seen in:** `main.py` BREAK IT comment; `WALKTHROUGH.md` debugging practice.

## err.errors() and feedback_from
*Type: third-party: pydantic v2 / project function*

**1. Definition:** `ValidationError.errors()` returns a list of dicts, one per failing field, with `loc` (field path), `msg` and the bad `input`. `feedback_from(err)` joins them into one short instruction for the model.
**2. In plain words:** The raw error is for humans; the retry message should say only which field is wrong and what is allowed. That is what makes the second attempt succeed.
**3. Example to quote:**
> "My retry message says `label: Input should be 'positive', 'negative', 'neutral' or 'unknown'`, so the model knows exactly what to fix."

**Seen in:** `main.py`, `feedback_from` and `classify_json`.
