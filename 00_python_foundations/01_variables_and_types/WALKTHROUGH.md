# Walkthrough 01 — Variables, types, strings, f-strings

## Goal
Know what kinds of values Python has, how to inspect them with `type()`, and how to print them clearly.

## Run it
```bash
cd 00_python_foundations/01_variables_and_types
python main.py
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `x = 5` | Create a name `x` pointing to the value 5 (no type declaration) | `epochs = 3` |
| `# text` | Comment: ignored by Python | `# learning rate` |
| `f"...{x}..."` | f-string: `{}` is replaced by the value of the expression | `f"loss={loss:.3f}"` -> `loss=2.718` |
| `{x!r}` | Use `repr(x)` (shows quotes on strings) — best for debugging | `f"{'a'!r}"` -> `'a'` |
| `{x:.2f}` / `{x:,}` | Format: 2 decimals / thousands separators | `f"{1234567:,}"` -> `1,234,567` |
| `s[0:4]`, `s[-1]` | Slice (stop excluded), negative index = from the end | `"distilgpt2"[0:4]` -> `dist` |
| `//`, `%`, `**` | Floor division, remainder, power | `7 // 2` -> `3` |
| `1_000` | Underscores inside numbers are cosmetic | `124_439_808` |
| `for a, b in [(…),(…)]:` | Loop that unpacks each pair | see main.py |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `print()` | built-in | Writes text to the terminal | `end=`, `sep=` | `None` |
| `type()` | built-in | Tells the class of a value | value | a class object |
| `int()`, `float()`, `str()`, `bool()` | built-in | Convert between types | value | converted value (or `ValueError`) |
| `str.strip()` / `.lower()` / `.split()` / `.join()` | built-in methods | Clean / lowercase / cut / glue strings | `split(sep)`; `join(list)` | **new** string/list (strings are immutable) |

No third-party libraries in this step.

## Execution trace (real output)
Input: nothing (values are written in the script) -> processing: loop over 5 (name, value) pairs -> output:
```
learning_rate = 0.001  -> <class 'float'>
epochs = 3  -> <class 'int'>
model_name = 'distilgpt2'  -> <class 'str'>
is_training = True  -> <class 'bool'>
nothing_yet = None  -> <class 'NoneType'>
```
Later: `7 / 2 = 3.5`, `7 // 2 = 3`; `text.strip()` prints `Hello, Tokenizer!` while `text.lower()` still has the spaces (original untouched);
`bool(0), bool(""), bool([]), bool("x")` -> `False False False True`; last line `loss=2.718  params=124,439,808`.

## Experiments (change one thing)
1. Change `epochs = 3` to `epochs = "3"` and add `print(epochs * 2)`. Expected: `33` (string repetition!) — a type mix-up that doesn't crash.
2. Change `.3f` to `.1f` in the last f-string. Expected: `loss=2.7`.
3. Print `model_name[:-1]`. Expected: `distilgpt` (everything except the last character).

## Debugging practice
Uncomment `print("epochs: " + epochs)`. Traceback ends with `TypeError: can only concatenate str (not "int") to str`.
Read: error type = TypeError, message says you added str + int. Fix: `f"epochs: {epochs}"` or `str(epochs)`.

## Rebuild checklist
- [ ] Create 4 variables of 4 types and print each with its type using an f-string
- [ ] Format a float to 2 decimals and an int with commas
- [ ] Slice the first 4 characters and last character of a string

## Say it aloud
"Python variables are labels attached to values; the value has the type, not the name. I use `type()` to inspect it and f-strings to print values, with `!r` when I want to see exactly what is inside. Strings are immutable so methods like `strip()` return a new string. A common bug is mixing types, like adding a str and an int, which is why config values read from files or env vars must be converted with `int()`/`float()`."
Trade-off: dynamic typing is flexible but errors show up only at runtime — that's why type hints (step 03) help.
