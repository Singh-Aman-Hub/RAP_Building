# Terms — Variables and Types

Folder: `00_python_foundations/01_variables_and_types` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [00 Python foundations README](../README.md)

**Terms in this folder:** [Variable](#variable) · [Comment](#comment) · [float](#float) · [int](#int) · [str](#str) · [bool](#bool) · [None / NoneType](#none--nonetype) · [type()](#type) · [print()](#print) · [for loop with tuple unpacking](#for-loop-with-tuple-unpacking) · [f-string](#f-string) · [repr / !r](#repr--r) · [Arithmetic operators](#arithmetic-operators) · [String methods](#string-methods) · [Immutability](#immutability) · [Slicing and negative indexing](#slicing-and-negative-indexing) · [Type conversion](#type-conversion) · [Truthy and falsy values](#truthy-and-falsy-values) · [Format specifiers](#format-specifiers) · [Numeric underscore separators](#numeric-underscore-separators) · [TypeError](#typeerror)

---

## Variable  `(concept)`
**1. Definition:** A variable is a name bound to a value (an object) in memory; assigning with `=` creates or rebinds the name, and the type belongs to the value, not the name.
**2. In plain words:** Think of a sticky label you attach to a box. You can move the label to a different box later. Python does not make you declare "this label is only for numbers"; `epochs = 3` just attaches the label `epochs` to the number 3. The flexibility is handy, but it also means a wrong type only shows up when the code runs.
**3. Example to quote:**
> "In my script `learning_rate = 0.001` just labels a float, and later I could rebind `learning_rate` to a string without Python complaining, so I use type hints and checks to stay safe."
```python
epochs = 3
epochs = "three"   # same name, new value, new type
```
**Seen in:** `main.py`, first block of assignments

## Comment  `(syntax)`
**1. Definition:** A comment is text after `#` on a line that the Python interpreter ignores.
**2. In plain words:** Comments are notes for humans. In this file they explain what each type is. Good comments say *why* something is done, not just repeat the code. They never change how the program runs.
**3. Example to quote:**
> "I comment `# bool : True / False (capital first letter!)` so the next reader remembers `true` in lowercase would be a NameError."
```python
is_training = True   # capital T matters
```
**Seen in:** `main.py`, every section

## float  `(Python built-in)`
**1. Definition:** `float` is the built-in type for double-precision floating-point numbers, i.e. numbers with a fractional part.
**2. In plain words:** Any number with a decimal point, like `0.001`, is a float. Learning rates, losses and probabilities are floats. They are approximate (for example `0.1 + 0.2` is not exactly `0.3`), which is fine for machine learning but surprising for money.
**3. Example to quote:**
> "My `learning_rate = 0.001` is a float; `type(learning_rate)` prints `<class 'float'>`, and `7 / 2` also returns a float, 3.5."
```python
print(type(0.001))   # <class 'float'>
```
**Seen in:** `main.py`, `learning_rate`

## int  `(Python built-in)`
**1. Definition:** `int` is the built-in type for whole numbers of unlimited size.
**2. In plain words:** Counts such as epochs, batch sizes or number of parameters are ints. Python ints never overflow, so `2 ** 1000` just works. Mixing ints and floats is fine in arithmetic: the result becomes a float.
**3. Example to quote:**
> "`epochs = 3` is an int, because you can't train 2.5 epochs in my loop; `type(epochs)` gives `<class 'int'>`."
```python
epochs = 3
print(epochs + 1)    # 4
```
**Seen in:** `main.py`, `epochs`

## str  `(Python built-in)`
**1. Definition:** `str` is the built-in immutable sequence type for text, made of Unicode characters.
**2. In plain words:** Anything in quotes is a string: `"distilgpt2"`. You can index, slice and call methods on it. Strings cannot be changed in place; every "change" gives you a new string. Numbers read from files arrive as strings and must be converted.
**3. Example to quote:**
> "My `model_name = \"distilgpt2\"` is a str, so `model_name[0:4]` gives `'dist'`."
```python
model_name = "distilgpt2"
print(len(model_name))   # 10
```
**Seen in:** `main.py`, `model_name`, strings section

## bool  `(Python built-in)`
**1. Definition:** `bool` is the built-in type with exactly two values, `True` and `False`; it is a subclass of `int`.
**2. In plain words:** A bool is a yes/no answer, like `is_training`. Note the capital first letter: `True`, not `true`. Comparisons such as `3 > 2` produce bools, and `if` statements use them to choose a path.
**3. Example to quote:**
> "I keep `is_training = True` as a flag so one function can switch between training and evaluation behaviour."
```python
print(3 > 2)   # True
```
**Seen in:** `main.py`, `is_training`

## None / NoneType  `(Python built-in)`
**1. Definition:** `None` is the single value of type `NoneType`, used to represent "no value".
**2. In plain words:** `None` means "nothing here yet". It is the usual way to say a setting has not been provided. A function that does not `return` anything also gives back `None`. Trying to call a method on `None` is a very common bug (`AttributeError`).
**3. Example to quote:**
> "I set `nothing_yet = None` as a placeholder, and check `if nothing_yet is None` before using it."
```python
result = None
print(result is None)   # True
```
**Seen in:** `main.py`, `nothing_yet`

## type()  `(Python built-in)`
**1. Definition:** `type(obj)` is a built-in function that returns the class of an object.
**2. In plain words:** When you are not sure what kind of thing a variable holds, call `type()`. It is the fastest first step when debugging a `TypeError`. It tells you what the value *is*, not what you hoped it would be.
**3. Example to quote:**
> "When my code crashed, I printed `type(value)` and saw `<class 'str'>` where I expected an int — the value came from a file."
```python
print(type(3.5))   # <class 'float'>
```
**Seen in:** `main.py`, the `for name, value in [...]` loop

## print()  `(Python built-in)`
**1. Definition:** `print(*objects, sep=' ', end='\n')` writes the text form of its arguments to standard output and returns `None`.
**2. In plain words:** `print` is how you see what your program is doing. Several arguments are separated by a space by default; `print("7 / 2 =", 7 / 2)` shows both. It is the simplest debugging tool, though real services should use `logging` instead.
**3. Example to quote:**
> "I printed `7 / 2 =` followed by the result to confirm that `/` returns 3.5, not 3."
```python
print("a", "b", sep="-")   # a-b
```
**Seen in:** `main.py`, throughout

## for loop with tuple unpacking  `(syntax)`
**1. Definition:** `for a, b in iterable:` repeats a block once per item and unpacks each item (here a pair) into the names `a` and `b`.
**2. In plain words:** The script has a list of pairs like `("epochs", 3)`. Instead of grabbing `pair[0]` and `pair[1]`, the loop splits each pair into `name` and `value` automatically. It makes the body read like plain English.
**3. Example to quote:**
> "I loop over `[(\"epochs\", 3), ...]` with `for name, value in` so each pass gets a label and its value without indexing."
```python
for name, value in [("a", 1), ("b", 2)]:
    print(name, value)
```
**Seen in:** `main.py`, top-level loop

## f-string  `(syntax)`
**1. Definition:** An f-string is a string literal prefixed with `f` in which `{expression}` placeholders are evaluated and inserted into the text.
**2. In plain words:** Put an `f` before the opening quote and write variables inside curly braces. Python swaps them for their values. It is much cleaner than gluing strings with `+`, and it avoids the str + int `TypeError`.
**3. Example to quote:**
> "I wrote `f\"loss={loss:.3f}\"` and it printed `loss=2.718`, so I don't need to convert the float to a string myself."
```python
loss = 2.718281828
print(f"loss={loss:.3f}")
```
**Seen in:** `main.py`, the loop's `print` and the last line

## repr / !r  `(Python built-in)`
**1. Definition:** `repr(x)` returns an unambiguous, developer-oriented string for `x`; inside an f-string, `{x!r}` applies `repr` to the value.
**2. In plain words:** `print("5")` and `print(5)` look identical, but `repr` adds quotes around strings so you can tell them apart. Use `!r` in debug prints to see exactly what a value is.
**3. Example to quote:**
> "`model_name = 'distilgpt2'` printed with quotes because I used `!r`, which instantly shows me it's a string and not a variable name."
```python
print(f"{'epochs'!r}")   # 'epochs'
```
**Seen in:** `main.py`, `{value!r}`

## Arithmetic operators  `(syntax)`
**1. Definition:** `/` is true division (always float), `//` is floor division, `%` is remainder, and `**` is exponentiation.
**2. In plain words:** `7 / 2` gives 3.5, `7 // 2` drops the fraction and gives 3, `7 % 2` gives what is left over (1), and `2 ** 10` is 1024. `//` and `%` are the pair you use for tasks like "how many full batches fit, and how many items remain".
**3. Example to quote:**
> "With 7 items and batches of 2, `7 // 2` is 3 full batches and `7 % 2` is 1 leftover item."
```python
print(7 // 2, 7 % 2, 2 ** 10)   # 3 1 1024
```
**Seen in:** `main.py`, arithmetic section

## String methods  `(Python built-in)`
**1. Definition:** Methods on `str` such as `strip()`, `lower()`, `split(sep)` and `sep.join(list)` return new strings or lists derived from the original.
**2. In plain words:** `strip()` trims spaces from both ends, `lower()` makes lowercase, `split(", ")` cuts text into a list at each separator, and `"-".join(["a","b"])` does the reverse. They are the everyday tools for cleaning text before tokenizing.
**3. Example to quote:**
> "I cleaned `'  Hello, Tokenizer!  '` with `.strip()` and then `.split(', ')` to get `['Hello', 'Tokenizer!']`."
```python
print("-".join(["a", "b", "c"]))   # a-b-c
```
**Seen in:** `main.py`, strings section

## Immutability  `(concept)`
**1. Definition:** An immutable object cannot be changed after creation; operations on it return new objects.
**2. In plain words:** After `text.lower()`, the original `text` still has its capital letters and spaces. If you want the lowercase text, you must store the result: `text = text.lower()`. Forgetting this is a classic beginner bug.
**3. Example to quote:**
> "I called `text.lower()` without assigning it and my text didn't change, because strings are immutable and the method returned a new string."
```python
s = "Hi"
s.lower()
print(s)   # Hi
```
**Seen in:** `main.py`, comment on `text.lower()`

## Slicing and negative indexing  `(syntax)`
**1. Definition:** `s[start:stop]` returns the items from `start` up to but not including `stop`; negative indices count from the end, so `s[-1]` is the last item.
**2. In plain words:** Position numbering starts at 0. `"distilgpt2"[0:4]` takes characters 0 to 3 and gives `'dist'`. `[-1]` is a shortcut for the last character, `'2'`. Remembering that the stop index is excluded prevents off-by-one errors.
**3. Example to quote:**
> "`model_name[0:4]` returned `'dist'`, and `model_name[-1]` returned `'2'`, the last character."
```python
print("distilgpt2"[:-1])   # distilgpt
```
**Seen in:** `main.py`, strings section

## Type conversion  `(Python built-in)`
**1. Definition:** `int()`, `float()`, `str()` and `bool()` construct a value of that type from another value, raising `ValueError` if the conversion is impossible.
**2. In plain words:** Data from files, environment variables and web forms arrives as text. Before doing math you convert it: `int("42") + 1` gives 43. Converting something like `int("abc")` fails with a `ValueError`, so validate input first.
**3. Example to quote:**
> "My batch size comes from an environment variable as `\"32\"`, so I wrap it in `int()` before using it in arithmetic."
```python
print(int("42") + 1)   # 43
```
**Seen in:** `main.py`, conversions section

## Truthy and falsy values  `(concept)`
**1. Definition:** In a boolean context, `0`, `0.0`, `""`, `[]`, `{}`, `set()` and `None` evaluate to `False` ("falsy"); almost everything else is `True` ("truthy").
**2. In plain words:** `if my_list:` is a short way of asking "is this list not empty?". The script shows `bool(0)`, `bool("")` and `bool([])` are all `False`, while `bool("x")` is `True`. The gotcha is that the string `"False"` is truthy because it is not empty.
**3. Example to quote:**
> "I wrote `if not results:` to detect an empty list, relying on empty containers being falsy."
```python
print(bool(0), bool(""), bool([]), bool("x"))   # False False False True
```
**Seen in:** `main.py`, conversions section

## Format specifiers  `(syntax)`
**1. Definition:** After a colon inside an f-string placeholder, a format spec controls presentation, e.g. `:.2f` (2 decimals) and `:,` (thousands separators).
**2. In plain words:** Raw numbers are hard to read. `{loss:.3f}` rounds the display to three decimals and `{params:,}` prints `124,439,808` with commas. Only the displayed text changes; the stored value is untouched.
**3. Example to quote:**
> "My log line shows `params=124,439,808` because I used `{params:,}` to make big numbers readable."
```python
print(f"{1234567:,}  {3.14159:.2f}")   # 1,234,567  3.14
```
**Seen in:** `main.py`, last `print`

## Numeric underscore separators  `(syntax)`
**1. Definition:** Underscores between digits in a numeric literal (`124_439_808`) are ignored by Python and exist purely for readability.
**2. In plain words:** Long numbers are hard to read at a glance. Writing `124_439_808` is the same as `124439808` but easier on the eyes. It has no effect on the value.
**3. Example to quote:**
> "I write parameter counts like `124_439_808` so I can tell millions from thousands quickly."
```python
print(1_000 == 1000)   # True
```
**Seen in:** `main.py`, `params`

## TypeError  `(concept)`
**1. Definition:** `TypeError` is the exception raised when an operation is applied to a value of an inappropriate type, such as adding a `str` and an `int`.
**2. In plain words:** Python refuses to guess how to combine text and numbers. `"epochs: " + epochs` fails with `can only concatenate str (not "int") to str`. Read the last line of the traceback first: it names the error and what clashed. Fix it with an f-string or `str()`.
**3. Example to quote:**
> "When I saw `TypeError: can only concatenate str (not \"int\") to str`, I knew I was adding text and a number, so I switched to `f\"epochs: {epochs}\"`."
```python
epochs = 3
print(f"epochs: {epochs}")
```
**Seen in:** `main.py` BREAK IT line; `WALKTHROUGH.md` debugging practice
