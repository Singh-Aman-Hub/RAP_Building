# Step 03 — Prompt templates (f-string, `str.format`, `string.Template`)

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

## Say it aloud
"A prompt template is a constant string with named slots, filled by a builder function that validates input. It makes prompts versionable and testable. Use `str.format` for simple prompts, double the braces — or use `string.Template` — when the prompt contains JSON. Never silently allow a missing slot."
