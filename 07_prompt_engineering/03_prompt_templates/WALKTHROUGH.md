# Step 03 — Prompt templates (f-string, `str.format`, `string.Template`)

## The story so far
In [step 02](../02_few_shot/) the examples made the model smarter, but the prompt grew into a long hand-edited string.

## The pain point
Prompts pasted together ad hoc cannot be versioned or tested. A stray `{` in a JSON example crashes `str.format` with a `KeyError`, and an empty input silently produces a broken prompt.

## The spark
Treat a prompt like code: a constant string with named slots, plus one function that fills it and checks the inputs. Name versions `PROMPT_V1`, `PROMPT_V2` so changes are traceable.

## Goal
Generate prompts from a reusable template so versions can be tracked, tested and reused.

## Run it
```bash
cd 07_prompt_engineering/03_prompt_templates && ../../.venv/bin/python main.py
```
Real output (excerpt):
```
--- string.Template ---
Return JSON like {"label": "<one of positive/negative/neutral>"} for this text:
Battery life is great but the screen scratches easily.

KeyError caught on purpose: '"label"'
Return {"label": "x"} for hi
Missing placeholder: 'text'
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| f-string | formatted string literal | Yes | Filled immediately where it is written |
| str.format | format template, `.format()` | No | Template stored first, filled later by name or position |
| string.Template | `$name` template, `substitute` | No | `$` placeholders; braces untouched, safe for JSON text |
| Prompt template | prompt builder, prompt version (PROMPT_V1) | Near | A reusable string with holes plus a function that fills them |
| KeyError | missing placeholder error | Yes (here) | Raised by `format` when a named placeholder has no value |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–8)

```python
"""
GOAL   : Build prompts from reusable TEMPLATES instead of pasting strings by hand, using the
         three Python tools beginners meet: f-strings, str.format, string.Template.
RUN    : cd 07_prompt_engineering/03_prompt_templates && python main.py
EXPECTED OUTPUT: the same prompt rendered three ways, a safe-rendering demo, and a KeyError
         that is caught on purpose when a placeholder value is missing.
NEEDS  : nothing (standard library only).
"""
```

- **In simple words:** Says this step is about building prompts from templates using three standard-library tools.
- **Exactly:** Expected output: same prompt three ways, a KeyError demo, a safe builder.
- **Why it is here:** Orients the reader.

### 2. Import Template and define sample data  (`main.py` lines 9–12)

```python
from string import Template  # Template is in Python's standard library

labels = ["positive", "negative", "neutral"]
text = "Battery life is great but the screen scratches easily."
```

- **In simple words:** Bring in `Template` and make two values to fill in.
- **Exactly:** `labels` is `list[str]` of 3; `text` is a `str`. `string.Template` is standard library, nothing to install.
- **Why it is here:** Fixed inputs make all three methods produce comparable output.

### 3. f-string prompt  (`main.py` lines 14–15)

```python
# 1) f-string: the {} are filled IMMEDIATELY, at the line where the string is written.
fstring_prompt = f"Classify into {', '.join(labels)}.\nText: {text}"
```

- **In simple words:** Write the values straight into the string.
- **Exactly:** `', '.join(labels)` yields `positive, negative, neutral`. Result: `"Classify into positive, negative, neutral.\nText: Battery life is great but the screen scratches easily."`
- **Why it is here:** The simplest option when the prompt is used once.

### 4. str.format template  (`main.py` lines 17–20)

```python
# 2) str.format: the template is a normal string with {names}; fill it LATER with .format(...).
#    Great for templates you keep as constants at the top of a file.
FORMAT_TEMPLATE = "Classify into {labels}.\nText: {text}"
format_prompt = FORMAT_TEMPLATE.format(labels=", ".join(labels), text=text)
```

- **In simple words:** Keep the template as a constant and fill it later.
- **Exactly:** `{labels}` and `{text}` are named fields; `.format(labels=..., text=...)` returns a new string identical to the f-string result in the real run.
- **Why it is here:** Lets you version and reuse templates (`PROMPT_V1`).

### 5. string.Template with JSON braces  (`main.py` lines 22–25)

```python
# 3) string.Template: placeholders look like $name. Curly braces are NOT special, so
#    it is safer when the template itself contains JSON (which is full of { }).
JSON_TEMPLATE = Template('Return JSON like {"label": "<one of $labels>"} for this text:\n$text')
template_prompt = JSON_TEMPLATE.substitute(labels="/".join(labels), text=text)
```

- **In simple words:** Use `$name` placeholders so the curly braces of JSON need no special care.
- **Exactly:** `substitute(labels="positive/negative/neutral", text=...)` replaces `$labels` and `$text`. Output: `Return JSON like {"label": "<one of positive/negative/neutral>"} for this text:` and the text on the next line.
- **Why it is here:** Avoids brace-escaping bugs in JSON-heavy prompts.

### 6. Print the three rendered prompts  (`main.py` lines 27–29)

```python
print("--- f-string ---\n" + fstring_prompt)
print("\n--- str.format ---\n" + format_prompt)
print("\n--- string.Template ---\n" + template_prompt)
```

- **In simple words:** Show the three results so you can compare.
- **Exactly:** `"\n--- f-string ---\n" + ...` prints each header and prompt. The first two are identical.
- **Why it is here:** Proves the methods are different routes to the same string.

### 7. Pitfall 1: unescaped JSON braces in str.format  (`main.py` lines 31–38)

```python
# --- Pitfall 1: JSON braces inside str.format must be doubled {{ }} ------------------
BAD = 'Return {"label": "x"} for {text}'
try:
    BAD.format(text="hi")
except KeyError as err:   # Python reads '"label"' as a placeholder name -> KeyError
    print("\nKeyError caught on purpose:", err)
GOOD = 'Return {{"label": "x"}} for {text}'    # doubled braces print as single braces
print(GOOD.format(text="hi"))
```

- **In simple words:** Curly braces in the text are read as placeholders, so a JSON example breaks it. Doubling them fixes it.
- **Exactly:** `BAD.format(text="hi")` raises `KeyError: '"label"'` (Python takes `"label"` as a field name). Real printed output: `KeyError caught on purpose: '"label"'`. `GOOD` uses `{{ }}`; prints `Return {"label": "x"} for hi`.
- **Why it is here:** One of the most common prompt-template bugs.

### 8. Pitfall 2: missing value fails loudly  (`main.py` lines 40–44)

```python
# --- Pitfall 2: a missing value should fail loudly, not silently produce a bad prompt --
try:
    FORMAT_TEMPLATE.format(labels="a, b")          # forgot text=
except KeyError as err:
    print("Missing placeholder:", err)
```

- **In simple words:** Forgetting a value stops the program instead of sending a broken prompt.
- **Exactly:** `FORMAT_TEMPLATE.format(labels="a, b")` raises `KeyError: 'text'`; printed `Missing placeholder: 'text'`.
- **Why it is here:** A silently half-filled prompt costs money and gives wrong answers.

### 9. Reusable prompt builder with validation  (`main.py` lines 46–52)

```python
# --- A tiny reusable function: a template + validation = a "prompt builder" -------------
def build_prompt(user_text: str, allowed: list[str]) -> str:   # type hints: str in, str out
    if not user_text.strip():
        raise ValueError("user_text is empty - refuse to build a prompt for nothing")
    return FORMAT_TEMPLATE.format(labels=", ".join(allowed), text=user_text.strip())

print("\n" + build_prompt("  fine, I guess  ", labels))
```

- **In simple words:** Wrap the template in a function that refuses empty input and trims spaces.
- **Exactly:** Signature `build_prompt(user_text: str, allowed: list[str]) -> str`. `.strip()` removes the surrounding spaces: `"  fine, I guess  "` becomes `fine, I guess`. Empty or whitespace input raises `ValueError`.
- **Why it is here:** Central place for validation; callers cannot forget it.

### 10. TRY IT / BREAK IT / versioning tip  (`main.py` lines 54–56)

```python
# TRY IT: add a third placeholder {tone} to FORMAT_TEMPLATE and fix every call.
# BREAK IT: call build_prompt("   ", labels) and read the traceback (ValueError, your own message).
# Versioning tip: store templates in constants named PROMPT_V1, PROMPT_V2 ... (used in step 07).
```

- **In simple words:** Experiments and a naming tip.
- **Exactly:** Comments. `build_prompt("   ", labels)` gives `ValueError: user_text is empty - refuse to build a prompt for nothing`.
- **Why it is here:** Prompt versions are compared in step 07.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| f-string `f"{x}"` | fills in `x` right now | `f"{1+1}"` -> `2` |
| `"{a}".format(a=1)` | fills named slots later | `"{a}-{b}".format(a=1, b=2)` |
| `{{` and `}}` | literal braces inside `format`/f-strings | `"{{x}}".format()` -> `{x}` |
| `Template("$a").substitute(a=1)` | `$name` slots, braces are normal | `Template("$a").substitute(a=1)` |
| `try / except KeyError as err` | catch a missing dict key / placeholder | see file |
| `def f(x: str, y: list[str]) -> str:` | function with type hints (hints are documentation, not enforced) | see file |
| `raise ValueError("msg")` | stop with your own error | – |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `str.format` | built-in | fills `{name}` slots | new str; `KeyError` if a name is missing |
| `string.Template` | standard library | `$name` templates; `.substitute()` raises on missing, `.safe_substitute()` leaves the slot | str |
| `str.strip` | built-in | trims whitespace both ends | str |
| `", ".join(list)` | built-in | list -> single string | str |

## Execution trace
Input: `labels` list and a text. Processing: three different mechanisms build equivalent text. The `BAD` template shows that `{"label": ...}` is parsed as a placeholder named `"label"`, hence `KeyError: '"label"'`. Output: identical prompts + two deliberate errors caught.

## Experiments
- Add `{tone}` to `FORMAT_TEMPLATE` -> every call site now fails until updated (this is a feature: loud failure).
- Replace `.substitute` with `.safe_substitute` and omit `text` -> `$text` stays in the prompt silently (dangerous!).

## Debugging practice
`KeyError: '"label"'` — the quotes inside the message tell you Python thought `"label"` was a placeholder name. Fix: double the braces or use `string.Template`.

## Rebuild checklist
- [ ] A constant template, a builder function with input validation, one deliberate failing call.

## The new problem (cliffhanger)
Now the prompt is clean, but the *reply* is still free text. The next line of your program needs `reply["label"]`, not a sentence. We must force a machine-readable format and refuse anything else. Next: [04_structured_json](../04_structured_json/).

> **Say it aloud (interview box):** A prompt template is a constant string with named slots plus a function that fills it and validates the inputs. I use f-strings for one-offs, `str.format` for reusable constants, and `string.Template` when the text contains many braces such as JSON. Versioned templates let me change one thing at a time and test the result.
