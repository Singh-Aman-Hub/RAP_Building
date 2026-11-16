# Walkthrough 01 — Variables, types, strings, f-strings

## Goal
Know what kinds of values Python has, how to inspect them with `type()`, and how to print them clearly.

## Run it
```bash
cd 00_python_foundations/01_variables_and_types
python main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| variable | name, identifier, binding, reference | Roughly | In Python a variable is a name bound to an object, not a box holding a value. |
| `None` | null (Java/JS), nil, NoneType | Same idea, Python spelling | `None` is a real object of type `NoneType`; compare with `is None`. |
| `repr()` vs `str()` | `!r` vs `!s` in f-strings | No | `repr` is unambiguous for developers (quotes on strings); `str` is friendly for users. |
| `/` vs `//` | true division vs floor division | No | `/` always returns float; `//` rounds down. |
| immutable | unchangeable | Same | `str`, `int`, `tuple` cannot be modified in place; methods return new objects. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–12)

```python
"""
GOAL:  See Python's basic value types, how to inspect them, and how to print them with f-strings.

RUN:   cd 00_python_foundations/01_variables_and_types && python main.py
       (or from repo root: python 00_python_foundations/01_variables_and_types/main.py)

EXPECTED OUTPUT (first lines):
    learning_rate = 0.001  -> <class 'float'>
    ...

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** This block is the file's label: what it teaches, how to run it, what to expect. Python ignores it when running.
- **Exactly:** A triple-quoted string as the first statement is the module docstring (available as `__doc__`). It runs nothing; no output.
- **Why it is here:** Tells the next reader the goal and the run command without opening the walkthrough.

### 2. Create five variables of five types  (`main.py` lines 14–19)

```python
# A variable is just a name pointing at a value. No type declaration needed.
learning_rate = 0.001          # float  : number with a decimal point
epochs = 3                     # int    : whole number
model_name = "distilgpt2"      # str    : text
is_training = True             # bool   : True / False (capital first letter!)
nothing_yet = None             # NoneType : "no value" marker, often used for "not set"
```

- **In simple words:** A variable is a sticky label on a value. You write `name = value` and Python works out the type for you.
- **Exactly:** `=` binds a name to an object. Real types at runtime: `0.001` is `float`, `3` is `int`, `"distilgpt2"` is `str`, `True` is `bool`, `None` is `NoneType`. `True`/`False` need a capital first letter; `true` would be a `NameError`.
- **Why it is here:** Every later block uses these five values; they are the five basic shapes of data you meet constantly (learning rate, epoch count, model name, flag, unset).

### 3. Loop over names and print each type  (`main.py` lines 21–27)

```python
# type() tells you what kind of value a name points to. Built-in function.
for name, value in [("learning_rate", learning_rate), ("epochs", epochs),
                    ("model_name", model_name), ("is_training", is_training),
                    ("nothing_yet", nothing_yet)]:
    # f-string: put f before the quotes and write {expressions} inside the text.
    # {value!r} shows the "repr" (strings get quotes) which is better for debugging.
    print(f"{name} = {value!r}  -> {type(value)}")
```

- **In simple words:** Walk through a list of (label, value) pairs and print what each value is, with its type.
- **Exactly:** The list holds 5 tuples; `for name, value in ...` unpacks each tuple into two names. `type(value)` returns the class object, printed as `<class 'float'>`. `{value!r}` calls `repr()`, so the string prints as `'distilgpt2'` with quotes. Real first line: `learning_rate = 0.001  -> <class 'float'>`. The f-string (`f"..."`) evaluates `{...}` expressions inside text.
- **Why it is here:** Without `!r` you could not tell the string `'3'` from the int `3` in the output; this is the first debugging habit: look at the type, not just the value.

### 4. Arithmetic operators  (`main.py` lines 29–34)

```python
# ---- arithmetic ---------------------------------------------------------
print("\n--- arithmetic ---")
print("7 / 2  =", 7 / 2)      # true division always gives a float -> 3.5
print("7 // 2 =", 7 // 2)     # floor division -> 3
print("7 % 2  =", 7 % 2)      # remainder (modulo) -> 1
print("2 ** 10 =", 2 ** 10)   # power -> 1024
```

- **In simple words:** Python can do sums. The four non-obvious ones are true division, floor division, remainder and power.
- **Exactly:** `7 / 2` -> `3.5` (always float). `7 // 2` -> `3` (rounds down toward negative infinity). `7 % 2` -> `1` (remainder). `2 ** 10` -> `1024`. The leading `"\n"` in the section title print gives a blank line.
- **Why it is here:** `/` vs `//` is a classic source of off-by-one and float-vs-int bugs (e.g. using a float as a list index raises `TypeError`).

### 5. String methods and slicing  (`main.py` lines 36–44)

```python
# ---- strings ------------------------------------------------------------
print("\n--- strings ---")
text = "  Hello, Tokenizer!  "
print(text.strip())            # strip() removes spaces at both ends (returns a NEW string)
print(text.lower())            # strings are immutable: the original is unchanged
print(text.strip().split(", "))  # split() -> list of pieces
print("-".join(["a", "b", "c"]))  # join() glues a list of strings together
print(model_name[0:4])         # slicing: characters 0,1,2,3 -> 'dist'
print(model_name[-1])          # negative index counts from the end -> '2'
```

- **In simple words:** Strings have built-in tools for cleaning, splitting, joining and cutting pieces out.
- **Exactly:** `text = "  Hello, Tokenizer!  "`. `.strip()` returns `'Hello, Tokenizer!'`; `.lower()` returns a new lowercase string (spaces kept); `.split(", ")` returns `['Hello', 'Tokenizer!']`; `"-".join([...])` returns `'a-b-c'`; `model_name[0:4]` is `'dist'` (start inclusive, stop exclusive); `model_name[-1]` is `'2'`. Strings are immutable: every method returns a NEW string, `text` never changes.
- **Why it is here:** Text cleaning is step one of every NLP pipeline; knowing methods return new strings avoids the bug of calling `s.strip()` and ignoring the result.

### 6. Convert between types and truthiness  (`main.py` lines 46–51)

```python
# ---- converting between types (very common source of bugs) --------------
print("\n--- conversions ---")
print(int("42") + 1)           # str -> int
print(float("0.5") * 2)        # str -> float
print(str(123) + "!")          # int -> str
print(bool(0), bool(""), bool([]), bool("x"))  # "falsy" values: 0, "", [], None, {}
```

- **In simple words:** Python does not guess when you mix text and numbers; you convert explicitly. Empty or zero things count as False.
- **Exactly:** `int("42") + 1` -> `43`; `float("0.5") * 2` -> `1.0`; `str(123) + "!"` -> `'123!'`. `bool(0), bool(""), bool([]), bool("x")` -> `False False False True`. Falsy values: `0`, `0.0`, `""`, `[]`, `{}`, `None`, `False`.
- **Why it is here:** Data read from files, env vars and HTTP is almost always `str`; forgetting to convert is the number-one beginner `TypeError`.

### 7. Format numbers inside an f-string  (`main.py` lines 53–56)

```python
# f-string number formatting: {value:.2f} = 2 decimals, {value:,} = thousands commas
loss = 2.718281828
params = 124_439_808           # underscores in numbers are just for readability
print(f"loss={loss:.3f}  params={params:,}")
```

- **In simple words:** Control how many decimals show and add commas to big numbers.
- **Exactly:** `loss=2.718281828`, `params=124_439_808` (underscores ignored by Python). `{loss:.3f}` rounds to 3 decimals -> `2.718`; `{params:,}` -> `124,439,808`. Format spec comes after `:`.
- **Why it is here:** Raw floats print 10+ digits and big integers are unreadable; logs and tables need formatting.

### 8. TRY IT and BREAK IT comments  (`main.py` lines 58–61)

```python
# TRY IT: change epochs to "3" (a string) and re-run section "arithmetic" ideas:
#         what happens with epochs + 1 ?
# BREAK IT: uncomment the next line to see a TypeError traceback (read the last line first!)
# print("epochs: " + epochs)
```

- **In simple words:** Two suggestions for you to experiment with. They are comments, so Python skips them.
- **Exactly:** Line 61 is commented-out: `"epochs: " + epochs` would raise `TypeError: can only concatenate str (not "int") to str` because `+` does not convert for you.
- **Why it is here:** Gives you a safe place to cause the classic type error on purpose and practise reading the traceback.

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
