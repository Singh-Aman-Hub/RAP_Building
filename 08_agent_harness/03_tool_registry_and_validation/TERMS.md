# Terms — Step 03: Tool registry, dispatch and argument validation

Folder: `08_agent_harness/03_tool_registry_and_validation` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Tool registry](#tool-registry) · [Dispatch](#dispatch) · [Tool description](#tool-description) · [Single source of truth](#single-source-of-truth) · [Argument validation](#argument-validation) · [validate_call](#validate_call) · [execute (never crash the loop)](#execute-never-crash-the-loop) · [Observation as an error message](#observation-as-an-error-message) · [Nested dictionary holding functions](#nested-dictionary-holding-functions) · [isinstance with a tuple of types](#isinstance-with-a-tuple-of-types) · [bool is a subclass of int](#bool-is-a-subclass-of-int) · [Type hint -> list[str]](#type-hint---liststr) · [sorted](#sorted) · [f-string with !r](#f-string-with-r) · [try / except with a catch-all](#try--except-with-a-catch-all) · [pydantic](#pydantic) · [BaseModel and model_config extra forbid](#basemodel-and-model_config-extra-forbid) · [Lax mode and strict mode (pydantic)](#lax-mode-and-strict-mode-pydantic) · [ValidationError and err.errors()](#validationerror-and-errerrors) · [Keyword unpacking (CalcArgs(**args))](#keyword-unpacking-calcargsargs) · [Generator expression in join](#generator-expression-in-join) · [if __name__ == "__main__"](#if-__name__--__main__) · [Calculator and shout tools](#calculator-and-shout-tools) · [Fake agent LLM](#fake-agent-llm)

---

## Tool registry
*Type: concept*

**1. Definition:** A single dictionary that stores, for each tool name, its function, description, parameter types, required arguments and allowed values.
**2. In plain words:** Everything the harness knows about a tool lives in one place, so the prompt and the validation are generated from the same source and cannot drift apart. Adding a tool means adding one entry.
**3. Example to quote:**
> "`REGISTRY['calculator']` holds the function, 'Do arithmetic on two numbers', the parameter types and the allowed operators. I add `word_count` by adding one more entry."

**Seen in:** `main.py`, `REGISTRY`.

## Dispatch
*Type: concept*

**1. Definition:** Looking up a tool by the name the model gave and calling its function with the supplied arguments.
**2. In plain words:** The model says "calculator" and your code finds the right function. Dispatch should only happen after the call has been validated.
**3. Example to quote:**
> "`REGISTRY[tool_name]['fn'](**args)` is the dispatch line, and `execute` only reaches it after `validate_call` finds no problems."

**Seen in:** `main.py`, `execute`.

## Tool description
*Type: concept*

**1. Definition:** A short human-readable sentence stored with a tool that is shown to the model so it knows when to use it.
**2. In plain words:** The model chooses tools by reading these descriptions, so they should be accurate and specific.
**3. Example to quote:**
> "`tools_prompt()` prints `- calculator(a, op, b): Do arithmetic on two numbers.` into the system prompt."

**Seen in:** `main.py`, `REGISTRY` and `tools_prompt`.

## Single source of truth
*Type: concept / design principle*

**1. Definition:** Keeping each fact (here, tool definitions) in exactly one place and generating everything else from it.
**2. In plain words:** If the prompt text and the validation rules were written separately, one would go out of date. Generating the prompt from the registry avoids that.
**3. Example to quote:**
> "I add a tool only to `REGISTRY` and `tools_prompt()` lists it automatically."

**Seen in:** `main.py`, `tools_prompt`; TRY IT comment.

## Argument validation
*Type: concept*

**1. Definition:** Checking a tool call's name and arguments (known tool, required arguments present, no unexpected ones, correct types, allowed values) before executing it.
**2. In plain words:** The model only proposes a call; your code decides if it is acceptable. Real models invent tool names, forget arguments or send strings where numbers are expected.
**3. Example to quote:**
> "A call with `a: 'six'` was rejected with 'argument a has wrong type str' before the calculator ever ran."

**Seen in:** `main.py`, `validate_call`.

## validate_call
*Type: project function*

**1. Definition:** A function that returns a list of problem strings for a proposed call; an empty list means the call is valid.
**2. In plain words:** Returning all problems at once lets the model fix everything in a single retry. It never runs the tool itself.
**3. Example to quote:**
> "`validate_call('calculator', {'a': 6, 'op': '*'})` returns `[\"missing required argument 'b'\"]`."

**Seen in:** `main.py`, `validate_call`.

## execute (never crash the loop)
*Type: project function*

**1. Definition:** A wrapper that validates, runs the tool inside `try/except` and always returns a string observation, beginning with `ERROR:` on failure.
**2. In plain words:** One bad call must not kill the agent. Errors become text the model can read and react to.
**3. Example to quote:**
> "Dividing by zero returned `ERROR: division by zero` as an observation instead of crashing the program."

**Seen in:** `main.py`, `execute`.

## Observation as an error message
*Type: concept*

**1. Definition:** Returning a validation or runtime problem to the model as the next `Observation:` message so it can self-correct.
**2. In plain words:** A real model can read "missing required argument 'b'" and try again with the argument added.
**3. Example to quote:**
> "A bad call looks to the model like `Observation: ERROR: missing required argument 'b'`."

**Seen in:** `main.py`, final demo print.

## Nested dictionary holding functions
*Type: Python syntax*

**1. Definition:** A dict whose values are dicts, and one of whose values is a function object, such as `{"fn": calculator}`.
**2. In plain words:** Functions are ordinary values in Python, so you can store them and call them later by looking them up.
**3. Example to quote:**
> "`spec['fn'](**args)` fetches the function from the dict and calls it."

```python
d = {"f": len}; d["f"]("abc")   # 3
```

**Seen in:** `main.py`, `REGISTRY`, `execute`.

## isinstance with a tuple of types
*Type: Python built-in*

**1. Definition:** `isinstance(x, (int, float))` is True if `x` is an instance of any listed type.
**2. In plain words:** The registry stores accepted types, so a number parameter accepts both integers and decimals.
**3. Example to quote:**
> "`isinstance(2.0, (int, float))` is True, so a decimal number passes the check for `a`."

**Seen in:** `main.py`, `validate_call`.

## bool is a subclass of int
*Type: Python gotcha*

**1. Definition:** In Python, `True` equals 1 and `False` equals 0, and `isinstance(True, int)` is True.
**2. In plain words:** Without an explicit check, `{"a": true}` from the model would be accepted as the number 1. The code rejects booleans for numeric parameters explicitly.
**3. Example to quote:**
> "The 'bool as number' case returned `ERROR: argument 'a' has wrong type bool` because I check `isinstance(value, bool)` first."

**Seen in:** `main.py`, `validate_call` comment; BREAK IT comment.

## Type hint `-> list[str]`
*Type: Python syntax*

**1. Definition:** A return annotation saying the function returns a list of strings.
**2. In plain words:** It documents that `validate_call` returns problems as text, with an empty list meaning success.
**3. Example to quote:**
> "`def validate_call(tool_name, args) -> list[str]` returns the list of problems."

**Seen in:** `main.py`, `validate_call`.

## sorted
*Type: Python built-in*

**1. Definition:** `sorted(iterable)` returns a new sorted list; on a dict it sorts the keys.
**2. In plain words:** It lists the available tool names alphabetically in the error message, which helps the model pick a valid one.
**3. Example to quote:**
> "`sorted(REGISTRY)` gives `['calculator', 'shout']` for the 'unknown tool' message."

**Seen in:** `main.py`, `validate_call`.

## f-string with `!r`
*Type: Python syntax*

**1. Definition:** `{value!r}` inside an f-string inserts `repr(value)`, showing quotes and escapes.
**2. In plain words:** Error messages like `unknown tool 'weather'` are clearer because the quotes show exactly what string was received.
**3. Example to quote:**
> "`f\"unknown tool {tool_name!r}\"` printed `unknown tool 'weather'`."

**Seen in:** `main.py`, `validate_call`.

## try / except with a catch-all
*Type: Python syntax*

**1. Definition:** `except ZeroDivisionError:` handles a specific error and `except Exception as err:` catches any remaining exception.
**2. In plain words:** Specific handlers give clear messages; the catch-all is a last-resort net that reports the error instead of crashing. Use the catch-all sparingly and always report what happened.
**3. Example to quote:**
> "Any unexpected tool failure becomes `ERROR: tool raised ValueError: ...` rather than a crash."

**Seen in:** `main.py`, `execute`.

## pydantic
*Type: third-party: pydantic*

**1. Definition:** A library that validates data against type-annotated classes and raises `ValidationError` on mismatch.
**2. In plain words:** It does declaratively what `validate_call` does by hand. FastAPI uses the same library to check requests (module 09).
**3. Example to quote:**
> "My `CalcArgs` class replaces dozens of lines of manual checks with three field declarations."

**Seen in:** `main.py`, `from pydantic import BaseModel, ValidationError`.

## BaseModel and model_config extra forbid
*Type: third-party: pydantic v2*

**1. Definition:** `BaseModel` is pydantic's base class for schemas; `model_config = {"extra": "forbid"}` makes unknown fields an error instead of ignoring them.
**2. In plain words:** Forbidding extras catches a model inventing arguments such as `fast`.
**3. Example to quote:**
> "`{'a': 1, 'op': '+', 'b': 2, 'fast': 1}` failed with 'Extra inputs are not permitted'."

**Seen in:** `main.py`, `class CalcArgs`.

## Lax mode and strict mode (pydantic)
*Type: concept*

**1. Definition:** By default pydantic converts compatible values (the string `"5"` becomes 5.0); `{"strict": True}` in `model_config` disables that and demands exact types.
**2. In plain words:** Lax mode is convenient but hides model mistakes; strict mode shows them. Choose based on whether you want to be forgiving or exact.
**3. Example to quote:**
> "`{'a': '5'}` passed as 5.0 in lax mode. I would turn on strict mode if I want to catch a model sending strings for numbers."

**Seen in:** `main.py`, `validate_with_pydantic` comment and NOTE.

## ValidationError and err.errors()
*Type: third-party: pydantic*

**1. Definition:** The exception pydantic raises on invalid data; `err.errors()` returns a list of dicts with the field location (`loc`) and message (`msg`).
**2. In plain words:** You can turn those into one short line for the model, like `op: Field required`.
**3. Example to quote:**
> "For `{'a': 1, 'b': 2}` the error list said `op: Field required`."

**Seen in:** `main.py`, `validate_with_pydantic`.

## Keyword unpacking (`CalcArgs(**args)`)
*Type: Python syntax*

**1. Definition:** Passing a dict's items as keyword arguments to a constructor or function.
**2. In plain words:** The dict from the model becomes named arguments for the pydantic class, which then validates each field.
**3. Example to quote:**
> "`CalcArgs(**{'a': 1, 'op': '+', 'b': 2})` builds the validated object."

**Seen in:** `main.py`, `validate_with_pydantic`.

## Generator expression in join
*Type: Python syntax*

**1. Definition:** An expression like `"; ".join(f"..." for e in err.errors())` that produces strings one by one and joins them.
**2. In plain words:** It builds one readable error line from many errors without a separate loop.
**3. Example to quote:**
> "I join all pydantic errors into one `ERROR: ...` string with a generator expression."

**Seen in:** `main.py`, `validate_with_pydantic`, `tools_prompt`.

## if __name__ == "__main__"
*Type: Python syntax*

**1. Definition:** A guard that runs the block below it only when the file is executed directly, not when it is imported.
**2. In plain words:** This file defines the registry and validators that other steps import; the demo table only prints when you run this file.
**3. Example to quote:**
> "Later steps import `execute` from this file, and the guard stops the demo output from running at import time."

**Seen in:** `main.py`, bottom section.

## Calculator and shout tools
*Type: project functions*

**1. Definition:** Two plain functions registered as tools: `calculator(a, op, b)` for arithmetic via a dict of lambdas, and `shout(text)` which upper-cases text.
**2. In plain words:** The second tool shows that a registry holds any number of tools with different parameter schemas.
**3. Example to quote:**
> "`shout` takes only `text: str`; the registry knows that and rejects anything else."

**Seen in:** `main.py`, `calculator`, `shout`.

## Fake agent LLM
*Type: project helper*

**1. Definition:** `make_fake_agent_llm()` returns a scripted mock model used in this module; imported here for consistency though the demo calls `execute` directly.
**2. In plain words:** It is the stand-in "brain"; the step focuses on validation, which does not need it.
**3. Example to quote:**
> "`LLM = make_fake_agent_llm()` is defined but the validation table calls `execute` directly with hand-written calls."

**Seen in:** `main.py`, `LLM = make_fake_agent_llm()`.
