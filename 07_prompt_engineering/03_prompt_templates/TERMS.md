# Terms — Step 03: Prompt templates

Folder: `07_prompt_engineering/03_prompt_templates` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [07 Prompt engineering README](../README.md)

**Terms in this folder:** [Prompt template](#prompt-template) · [f-string](#f-string) · [str.format](#strformat) · [string.Template](#stringtemplate) · [Placeholder](#placeholder) · [Doubled braces ({{ and }})](#doubled-braces--and) · [KeyError](#keyerror) · [try / except ... as](#try--except--as) · [Loud failure (missing value)](#loud-failure-missing-value) · [Prompt builder function](#prompt-builder-function) · [Type hints](#type-hints) · [raise ValueError](#raise-valueerror) · [str.strip](#strstrip) · [Prompt versioning (PROMPT_V1, PROMPT_V2)](#prompt-versioning-prompt_v1-prompt_v2)

---

## Prompt template
*Type: concept*

**1. Definition:** A reusable prompt string with named placeholders that are filled with real values (labels, user text) at run time.
**2. In plain words:** Instead of pasting a new long string for every request, you write the wording once and plug the changing parts in. That makes prompts easier to version, review and test.
**3. Example to quote:**
> "My template is `Classify into {labels}.\nText: {text}`. Changing the wording happens in one place, not at every call site."

**Seen in:** `main.py`, `FORMAT_TEMPLATE`.

## f-string
*Type: Python syntax*

**1. Definition:** A string literal starting with `f` in which `{expression}` is evaluated and inserted at the moment the line runs.
**2. In plain words:** It is the quickest way to build text with variables, but the values are fixed right where the string is written, so it is not a stored template you can fill later.
**3. Example to quote:**
> "`f\"Classify into {', '.join(labels)}.\\nText: {text}\"` builds the prompt immediately from my two variables."

```python
f"{1 + 1}"   # '2'
```

**Seen in:** `main.py`, `fstring_prompt`.

## str.format
*Type: Python built-in*

**1. Definition:** A string method that replaces `{name}` slots in a string with the keyword arguments you pass; raises `KeyError` when a named slot has no value.
**2. In plain words:** It lets you keep the template as a constant and fill it later. Its weakness is that it treats every curly brace as a slot, which is a problem when the prompt itself contains JSON.
**3. Example to quote:**
> "`FORMAT_TEMPLATE.format(labels=..., text=...)` fills my constant template later, and forgetting `text` fails loudly with a `KeyError`."

```python
"{a}-{b}".format(a=1, b=2)   # '1-2'
```

**Seen in:** `main.py`, `format_prompt`, `build_prompt`.

## string.Template
*Type: Python standard library*

**1. Definition:** A class in the `string` module for `$name` style templates; `.substitute()` raises `KeyError` on a missing name, `.safe_substitute()` leaves the slot unchanged.
**2. In plain words:** Curly braces are ordinary characters here, so it is safer when the prompt contains JSON examples. Use `.substitute()` so mistakes are loud.
**3. Example to quote:**
> "My JSON-asking prompt uses `Template`, so the braces in `{\"label\": ...}` need no escaping."

```python
from string import Template
Template("hi $name").substitute(name="Asha")
```

**Seen in:** `main.py`, `JSON_TEMPLATE`, `template_prompt`.

## Placeholder
*Type: concept*

**1. Definition:** A named slot in a template (`{text}` or `$text`) that stands in for a value supplied later.
**2. In plain words:** It marks "a value goes here". Each template style has its own spelling, and every placeholder must receive a value or the rendering should fail.
**3. Example to quote:**
> "`{labels}` and `{text}` are the two placeholders in my template."

**Seen in:** `main.py`, templates.

## Doubled braces (`{{` and `}}`)
*Type: Python syntax*

**1. Definition:** In `str.format` and f-strings, `{{` and `}}` are escape sequences that produce literal `{` and `}` characters.
**2. In plain words:** JSON uses braces, and `str.format` would read them as placeholders. Doubling them tells Python "these are real braces".
**3. Example to quote:**
> "`'Return {{\"label\": \"x\"}} for {text}'.format(text='hi')` prints `Return {\"label\": \"x\"} for hi`."

**Seen in:** `main.py`, the `GOOD` string.

## KeyError
*Type: Python built-in exception*

**1. Definition:** The exception raised when a dictionary key or a named format placeholder is not found.
**2. In plain words:** For templates it means a slot has no value. The message shows the missing name in quotes, which tells you what to fix. Here the name `'"label"'` shows Python mistook the JSON key for a placeholder.
**3. Example to quote:**
> "I got `KeyError: '\"label\"'` because `str.format` read my JSON key as a placeholder; I fixed it by doubling the braces."

**Seen in:** `main.py`, the two `except KeyError as err` blocks.

## try / except ... as
*Type: Python syntax*

**1. Definition:** A statement that runs a block and, if the named exception occurs, runs the `except` block instead, binding the exception object to a name.
**2. In plain words:** It lets the program handle a predictable failure instead of crashing. Here it demonstrates errors on purpose and prints them.
**3. Example to quote:**
> "I wrap `BAD.format(text='hi')` in `try`/`except KeyError as err` to show the failure without stopping the script."

```python
try:
    {}["x"]
except KeyError as err:
    print(err)
```

**Seen in:** `main.py`, pitfall demos.

## Loud failure (missing value)
*Type: concept*

**1. Definition:** The design principle that a missing or invalid input should raise an error immediately rather than silently producing a bad result.
**2. In plain words:** A prompt with a blank slot still gets sent to the model and you pay for a nonsense answer. Raising an exception catches the bug where it happens.
**3. Example to quote:**
> "If someone forgets `text=`, `format` raises `KeyError: 'text'` rather than sending a half-empty prompt."

**Seen in:** `main.py`, pitfall 2; `WALKTHROUGH.md` experiments (`safe_substitute` warning).

## Prompt builder function
*Type: concept / pattern*

**1. Definition:** A function that combines a template with input validation to produce a prompt, e.g. `build_prompt(user_text, allowed)`.
**2. In plain words:** It puts the template and the safety checks in one place, so every part of your code builds prompts the same safe way.
**3. Example to quote:**
> "`build_prompt` strips the text and refuses an empty string, so I never send a prompt about nothing."

**Seen in:** `main.py`, `build_prompt`.

## Type hints
*Type: Python syntax*

**1. Definition:** Annotations such as `user_text: str` and `-> str` that declare expected types; Python does not enforce them at run time.
**2. In plain words:** They document what goes in and out, and editors and tools use them to warn you about mistakes. They are hints, not guards, so you still validate inputs.
**3. Example to quote:**
> "`def build_prompt(user_text: str, allowed: list[str]) -> str` tells readers it takes a string and a list of strings and returns a string."

**Seen in:** `main.py`, `build_prompt` signature.

## raise ValueError
*Type: Python syntax / built-in exception*

**1. Definition:** `raise` throws an exception; `ValueError` signals that a value has the right type but an unacceptable content.
**2. In plain words:** It is how your own code says "this input is not acceptable" with a clear message that appears in the traceback.
**3. Example to quote:**
> "`build_prompt('   ', labels)` raises `ValueError: user_text is empty`, a message I wrote myself."

**Seen in:** `main.py`, `build_prompt`.

## str.strip
*Type: Python built-in*

**1. Definition:** A string method returning a copy with leading and trailing whitespace removed.
**2. In plain words:** It cleans user input. It also lets you detect "empty" input: a string of spaces becomes `""`, which is falsy.
**3. Example to quote:**
> "`if not user_text.strip():` catches inputs that are only spaces."

```python
"  fine  ".strip()   # 'fine'
```

**Seen in:** `main.py`, `build_prompt`.

## Prompt versioning (PROMPT_V1, PROMPT_V2)
*Type: concept*

**1. Definition:** Keeping each revision of a prompt in a clearly named constant so versions can be compared and rolled back.
**2. In plain words:** When you improve a prompt you want to prove it is better and be able to go back. Named constants make that possible and are used in step 07.
**3. Example to quote:**
> "I keep `PROMPT_V1`, `PROMPT_V2` and run the same test set against each, so I know which wording is better."

**Seen in:** `main.py`, final comment.
