# Terms — Functions and Type Hints

Folder: `00_python_foundations/03_functions_and_type_hints` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [00 Python foundations README](../README.md)

**Terms in this folder:** [typing.Optional](#typingoptional) · [Function (def)](#function-def) · [return](#return) · [Default argument](#default-argument) · [Positional vs keyword arguments](#positional-vs-keyword-arguments) · [Type hint](#type-hint) · [Docstring](#docstring) · [sum() and len()](#sum-and-len) · [min() and max()](#min-and-max) · [Tuple return and unpacking](#tuple-return-and-unpacking) · [*args and **kwargs](#args-and-kwargs) · [Mutable default argument trap](#mutable-default-argument-trap) · [lambda](#lambda) · [sorted() with key](#sorted-with-key) · [try / except TypeError](#try--except-typeerror) · [Missing-argument TypeError](#missing-argument-typeerror)

---

## typing.Optional  `(standard library)`
**1. Definition:** `typing.Optional[X]` is a type-hint helper meaning "a value of type `X` or `None`".
**2. In plain words:** Some functions return a result only sometimes, like finding an item in a list. `Optional[int]` tells readers and tools "this gives an int, or None when nothing was found". It changes nothing at runtime; it's documentation. In modern Python you can also write `int | None`.
**3. Example to quote:**
> "`find_index` is hinted `Optional[int]` because it returns the position if the target exists and `None` if it doesn't."
```python
from typing import Optional
def f() -> Optional[int]: return None
```
**Seen in:** `main.py`, import and `find_index`

## Function (def)  `(syntax)`
**1. Definition:** A function is a named, reusable block of code defined with `def name(parameters):` that runs when called and may return a value.
**2. In plain words:** Instead of copy-pasting the same lines, wrap them in a function and call it with different inputs. `add(2, 3)` runs the body with `a=2`, `b=3`. Functions are the main way to keep code organised and testable.
**3. Example to quote:**
> "I defined `add(a, b)` once and called it with `add(2, 3)`, getting 5."
```python
def add(a, b):
    return a + b
```
**Seen in:** `main.py`, `add`

## return  `(syntax)`
**1. Definition:** `return value` ends the function and sends `value` back to the caller; a function without `return` gives back `None`.
**2. In plain words:** `return` is how the answer leaves the function. Printing inside a function shows text but doesn't give the caller a value to use. Forgetting `return` is a common bug that produces `None`.
**3. Example to quote:**
> "`find_index` returns `i` when it finds the target, and explicitly returns `None` at the end when it doesn't."
```python
def double(x): return x * 2
print(double(4))   # 8
```
**Seen in:** `main.py`, `add`, `find_index`

## Default argument  `(syntax)`
**1. Definition:** A parameter written as `name=value` uses that value when the caller does not supply the argument.
**2. In plain words:** Defaults make parameters optional. In `generate_text(prompt, max_new_tokens=5, temperature=1.0)` you only must give the prompt. This is exactly how library functions such as `model.generate(max_new_tokens=20)` work.
**3. Example to quote:**
> "`generate_text(\"Hi\")` printed `[Hi] + 5 tokens @ temp 1.0` because both optional parameters fell back to their defaults."
```python
def gen(p, n=5): return f"{p}:{n}"
print(gen("Hi"))   # Hi:5
```
**Seen in:** `main.py`, `generate_text`

## Positional vs keyword arguments  `(concept)`
**1. Definition:** Positional arguments are matched to parameters by order; keyword arguments are matched by name (`name=value`) and may appear in any order after the positional ones.
**2. In plain words:** `generate_text("Hi", 20)` puts 20 into the second slot, `max_new_tokens`. `generate_text("Hi", temperature=0.2)` skips the middle parameter by naming the one you want. Keywords make calls self-explanatory and safe when a function has many options.
**3. Example to quote:**
> "I called `generate_text(prompt=\"Hi\", temperature=0, max_new_tokens=1)` with keywords in a different order, and it still worked because the names decide."
```python
generate_text("Hi", temperature=0.2)
```
**Seen in:** `main.py`, calls of `generate_text`

## Type hint  `(syntax)`
**1. Definition:** A type hint is an annotation such as `x: int` or `-> float` that documents the expected types of parameters and return values; Python does not enforce it at runtime.
**2. In plain words:** Hints tell humans, editors and libraries (FastAPI, Pydantic) what a function expects. But if you pass strings to `average(numbers: list[float])`, Python still runs and fails inside `sum` with a normal `TypeError`. Hints are signposts, not guards.
**3. Example to quote:**
> "`def average(numbers: list[float]) -> float` tells readers it takes a list of floats, but passing `['1','2']` still crashes at runtime, so I validate input where it matters."
```python
def average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)
```
**Seen in:** `main.py`, `average`, `find_index`, `min_max`

## Docstring  `(syntax)`
**1. Definition:** A docstring is a triple-quoted string placed as the first statement of a function, class or module that documents it and is available via `help()`.
**2. In plain words:** It's the official place to explain what a function does. Editors show it when you hover over the function. Keep it short: what it returns and any special cases, like "or None if absent".
**3. Example to quote:**
> "In `find_index`, the docstring says it returns the index or None, so callers know to handle the missing case."
```python
def f():
    """Return 1."""
    return 1
```
**Seen in:** `main.py`, `find_index`

## sum() and len()  `(Python built-in)`
**1. Definition:** `sum(iterable)` adds up the items and `len(x)` returns how many items a container has.
**2. In plain words:** Together they compute an average: `sum(numbers) / len(numbers)`. `sum` fails with a `TypeError` if items can't be added to the starting 0, such as strings. `len` of an empty list is 0, so dividing by it would raise `ZeroDivisionError`.
**3. Example to quote:**
> "`average([1.0, 2.0, 6.0])` is `sum` = 9.0 divided by `len` = 3, so 3.0."
```python
print(sum([1, 2, 6]) / len([1, 2, 6]))   # 3.0
```
**Seen in:** `main.py`, `average`

## min() and max()  `(Python built-in)`
**1. Definition:** `min(iterable)` and `max(iterable)` return the smallest and largest items.
**2. In plain words:** They save you from writing a loop to find extremes. In `min_max([4, 9, 1])` they give 1 and 9. They raise `ValueError` on an empty list.
**3. Example to quote:**
> "`min_max([4, 9, 1])` returned `(1, 9)` using the two built-ins."
```python
print(min([4, 9, 1]), max([4, 9, 1]))   # 1 9
```
**Seen in:** `main.py`, `min_max`

## Tuple return and unpacking  `(syntax)`
**1. Definition:** `return a, b` returns a single tuple `(a, b)`, and `x, y = f()` unpacks it into separate variables.
**2. In plain words:** Python functions technically return one value, but a tuple lets you return several. The caller unpacks them in one line: `low, high = min_max([4, 9, 1])` gives `low = 1` and `high = 9`.
**3. Example to quote:**
> "I returned `min(values), max(values)` and unpacked it as `low, high`, so one call gave me both numbers."
```python
low, high = (1, 9)
```
**Seen in:** `main.py`, `min_max`

## *args and **kwargs  `(syntax)`
**1. Definition:** In a function definition, `*args` collects extra positional arguments into a tuple and `**kwargs` collects extra keyword arguments into a dict.
**2. In plain words:** They let a function accept "anything else" without naming each parameter. `show(1, 2, name="gpt2", layers=12)` gives `args = (1, 2)` and `kwargs = {'name': 'gpt2', 'layers': 12}`. Wrappers and decorators use them to pass arguments through.
**3. Example to quote:**
> "`show(1, 2, name=\"gpt2\", layers=12)` printed `args: (1, 2) | kwargs: {'name': 'gpt2', 'layers': 12}`."
```python
def show(*args, **kwargs): print(args, kwargs)
```
**Seen in:** `main.py`, `show`

## Mutable default argument trap  `(concept)`
**1. Definition:** A default value like `[]` is created once when the function is defined, so every call that uses the default shares the same list object.
**2. In plain words:** `bad_append(1)` and then `bad_append(2)` both return `[1, 2]` because they append to one shared list. The fix is to default to `None` and create a new list inside the function. It is a classic interview question.
**3. Example to quote:**
> "My `bad_append` printed `[1, 2] [1, 2]` because the default list is shared; `good_append` with `bucket=None` printed `[1] [2]`."
```python
def good(x, bucket=None):
    bucket = [] if bucket is None else bucket
    bucket.append(x); return bucket
```
**Seen in:** `main.py`, `bad_append`, `good_append`

## lambda  `(syntax)`
**1. Definition:** `lambda args: expression` creates a small anonymous function consisting of a single expression.
**2. In plain words:** When you need a throwaway function for one spot, like a sort key, a `lambda` avoids writing a full `def`. `lambda w: len(w)` means "given w, return its length".
**3. Example to quote:**
> "I sorted words by length using `key=lambda w: len(w)`, which gave `['fig', 'apple', 'banana']`."
```python
double = lambda x: x * 2
```
**Seen in:** `main.py`, end of file

## sorted() with key  `(Python built-in)`
**1. Definition:** `sorted(iterable, key=func)` returns a new sorted list, ordering items by the value `func(item)` instead of the item itself.
**2. In plain words:** By default strings sort alphabetically. Passing `key=` changes the rule: sort by length, by last letter, or by a dict field. The original list is left unchanged because `sorted` returns a new one.
**3. Example to quote:**
> "`sorted(words, key=lambda w: len(w))` ordered fig, apple, banana by length without altering `words`."
```python
print(sorted(["banana", "fig", "apple"], key=len))
```
**Seen in:** `main.py`, end of file

## try / except TypeError  `(syntax)`
**1. Definition:** `try:` runs code that might fail and `except TypeError as err:` catches that specific exception and runs a handler, binding the exception to `err`.
**2. In plain words:** The script wants to show that type hints don't prevent a crash, but also not stop the whole program. So it wraps the bad call and prints the error message. Catch specific exceptions rather than everything.
**3. Example to quote:**
> "I wrapped `average([\"1\", \"2\"])` in `try/except TypeError` and printed `unsupported operand type(s) for +: 'int' and 'str'`."
```python
try:
    sum(["1", "2"])
except TypeError as err:
    print(err)
```
**Seen in:** `main.py`, the "Hints are not enforced" block

## Missing-argument TypeError  `(concept)`
**1. Definition:** Calling a function without a required positional argument raises `TypeError: f() missing 1 required positional argument: 'name'`.
**2. In plain words:** Python tells you the function name and which parameter you forgot. `add(1)` fails because `b` has no default. Read the last line of the traceback first; it usually contains the fix.
**3. Example to quote:**
> "`add(1)` raised `TypeError: add() missing 1 required positional argument: 'b'`, so I knew I had forgotten the second value."
```python
# add(1)  -> TypeError
```
**Seen in:** `main.py` BREAK IT; `WALKTHROUGH.md` debugging practice
