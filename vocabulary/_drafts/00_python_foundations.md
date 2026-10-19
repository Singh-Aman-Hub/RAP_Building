### Variable
<!-- cat: python -->
- **Technical definition:** A name bound to an object in memory; assignment with `=` creates or rebinds the name.
- **Plain English:** A label stuck on a value.
- **Example / analogy:** A sticky note saying 'epochs' on the number 3.
- **Snippet:**
```python
epochs = 3
epochs = 5  # same label, new value
```
- **Snippet explained:** The name `epochs` first points to 3, then to 5. No type declaration is needed.
- **Related terms:** [[Data type]], [[Mutable vs immutable]]

### Data type
<!-- cat: python -->
- **Technical definition:** The class of a value (int, float, str, bool, list, dict ...) that decides which operations are allowed.
- **Plain English:** What kind of thing a value is.
- **Example / analogy:** You can add numbers, but adding text to a number is an error.
- **Snippet:**
```python
type(3), type(3.0), type('3')
```
- **Snippet explained:** Returns `<class 'int'>`, `<class 'float'>`, `<class 'str'>`.
- **Related terms:** [[Variable]], [[Type conversion]]

### Type conversion
<!-- cat: python -->
- **Technical definition:** Explicitly turning a value into another type with int(), float(), str(), bool().
- **Plain English:** Changing a value's kind, like text '5' into number 5.
- **Example / analogy:** Values read from env vars or files are strings and must be converted.
- **Snippet:**
```python
int('42') + 1
```
- **Snippet explained:** `int('42')` makes the number 42, so adding 1 gives 43.
- **Related terms:** [[Data type]], [[Environment variable]]

### f-string
<!-- cat: python -->
- **Technical definition:** A string literal prefixed with f whose {expressions} are evaluated and formatted into the text.
- **Plain English:** A text template with blanks Python fills in.
- **Example / analogy:** Like a mail-merge letter.
- **Snippet:**
```python
loss = 2.7183
print(f'loss={loss:.2f}')
```
- **Snippet explained:** `{loss:.2f}` inserts loss with 2 decimals -> `loss=2.72`.
- **Related terms:** [[Data type]]

### Slicing
<!-- cat: python -->
- **Technical definition:** Selecting a sub-sequence with `seq[start:stop:step]`; stop is exclusive and negative indices count from the end.
- **Plain English:** Cutting a piece out of a list or string.
- **Example / analogy:** Taking pages 2 to 4 of a book.
- **Snippet:**
```python
'distilgpt2'[0:4]  # 'dist'
'abc'[::-1]       # 'cba'
```
- **Snippet explained:** First gives characters 0-3; second reverses with step -1.
- **Related terms:** [[List]]

### Conditional (if/elif/else)
<!-- cat: python -->
- **Technical definition:** A statement that runs a block only when a boolean condition is true.
- **Plain English:** Making a decision in code.
- **Example / analogy:** If it rains take an umbrella, else wear sunglasses.
- **Snippet:**
```python
if loss > 2:
    print('learning')
elif loss > 0.5:
    print('better')
else:
    print('good')
```
- **Snippet explained:** Checks conditions top to bottom and runs the first true block.
- **Related terms:** [[Loop (for/while)]], [[Boolean]]

### Boolean
<!-- cat: python -->
- **Technical definition:** The type with values True and False; empty containers, 0 and None are 'falsy'.
- **Plain English:** A yes/no value.
- **Example / analogy:** bool([]) is False, bool('x') is True.
- **Snippet:**
```python
print(bool(0), bool(''), bool('false'))
```
- **Snippet explained:** Prints `False False True`: any non-empty string is truthy, even 'false'.
- **Related terms:** [[Conditional (if/elif/else)]], [[Type conversion]]

### Loop (for/while)
<!-- cat: python -->
- **Technical definition:** Constructs that repeat a block: `for` over an iterable, `while` while a condition holds.
- **Plain English:** Doing something again and again.
- **Example / analogy:** Washing every dish (for) vs scrubbing until clean (while).
- **Snippet:**
```python
for i in range(3):
    print(i)
```
- **Snippet explained:** Prints 0, 1, 2; `range(3)` stops before 3.
- **Related terms:** [[Conditional (if/elif/else)]], [[Iterator]]

### Iterator
<!-- cat: python -->
- **Technical definition:** An object that returns items one at a time via `next()`; any `for` loop uses one.
- **Plain English:** Something you can walk through step by step.
- **Example / analogy:** A conveyor belt delivering one box at a time.
- **Snippet:**
```python
it = iter([1, 2])
print(next(it), next(it))
```
- **Snippet explained:** `iter` creates the iterator; `next` pulls items; a third `next` raises StopIteration.
- **Related terms:** [[Generator]], [[Loop (for/while)]]

### enumerate
<!-- cat: python -->
- **Technical definition:** Built-in wrapping an iterable to yield (index, item) pairs.
- **Plain English:** A loop that also tells you the position.
- **Example / analogy:** Numbering the items on a to-do list.
- **Snippet:**
```python
for i, w in enumerate(['a', 'b']):
    print(i, w)
```
- **Snippet explained:** Prints `0 a` then `1 b`.
- **Related terms:** [[Loop (for/while)]], [[zip]]

### zip
<!-- cat: python -->
- **Technical definition:** Built-in that combines several iterables element by element into tuples, stopping at the shortest.
- **Plain English:** Walking two lists side by side.
- **Example / analogy:** A zipper pairing teeth.
- **Snippet:**
```python
list(zip('ab', [1, 2]))
```
- **Snippet explained:** Gives `[('a', 1), ('b', 2)]`.
- **Related terms:** [[enumerate]], [[Dictionary]]

### Function
<!-- cat: python -->
- **Technical definition:** A named, reusable block of code defined with `def`, taking parameters and optionally returning a value.
- **Plain English:** A mini-program you can call by name.
- **Example / analogy:** A recipe you follow with different ingredients.
- **Snippet:**
```python
def add(a, b):
    return a + b
add(2, 3)
```
- **Snippet explained:** `add` takes two inputs and returns their sum, 5.
- **Related terms:** [[Default argument]], [[Keyword argument]]

### Default argument
<!-- cat: python -->
- **Technical definition:** A parameter with a pre-set value used when the caller omits it.
- **Plain English:** An optional setting with a fallback.
- **Example / analogy:** `generate(max_new_tokens=20)` if you do not choose.
- **Snippet:**
```python
def gen(prompt, n=5):
    return prompt * n
gen('a')
```
- **Snippet explained:** `n` defaults to 5 so `gen('a')` returns 'aaaaa'.
- **Related terms:** [[Function]], [[Mutable default argument trap]]

### Keyword argument
<!-- cat: python -->
- **Technical definition:** An argument passed by parameter name (`f(x, temperature=0.2)`) instead of by position.
- **Plain English:** Naming the setting you change.
- **Example / analogy:** Saying 'a large coffee with oat milk' instead of by order.
- **Snippet:**
```python
gen('hi', n=2)
```
- **Snippet explained:** `n=2` sets only n; other parameters keep defaults.
- **Related terms:** [[Default argument]], [[args and kwargs]]

### args and kwargs
<!-- cat: python -->
- **Technical definition:** `*args` collects extra positional arguments into a tuple; `**kwargs` collects extra keyword arguments into a dict.
- **Plain English:** A catch-all for any extra inputs.
- **Example / analogy:** A wrapper that forwards everything it receives.
- **Snippet:**
```python
def show(*args, **kwargs):
    print(args, kwargs)
show(1, 2, a=3)
```
- **Snippet explained:** Prints `(1, 2) {'a': 3}`.
- **Related terms:** [[Decorator]], [[Keyword argument]]

### Mutable default argument trap
<!-- cat: python -->
- **Technical definition:** Using a mutable object (list/dict) as a default value; it is created once and shared by all calls.
- **Plain English:** A hidden shared notebook between calls.
- **Example / analogy:** Everyone writing in the same shared bucket.
- **Snippet:**
```python
def f(x, bucket=[]):
    bucket.append(x)
    return bucket
print(f(1), f(2))
```
- **Snippet explained:** Prints `[1, 2] [1, 2]`. Fix: default `None` and create the list inside.
- **Related terms:** [[Default argument]], [[Mutable vs immutable]]

### Type hint
<!-- cat: python -->
- **Technical definition:** Optional annotation of parameter and return types (`x: int -> str`) used by tools; not enforced at runtime.
- **Plain English:** A note telling readers and editors what type is expected.
- **Example / analogy:** A label on a box saying 'fragile'.
- **Snippet:**
```python
def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs)
```
- **Snippet explained:** The hints say xs is a list of floats and the result is a float; Python still runs it with any input.
- **Related terms:** [[Pydantic model]], [[Function]]

### Docstring
<!-- cat: python -->
- **Technical definition:** A string literal as the first statement in a function/class/module that documents it; shown by `help()`.
- **Plain English:** Built-in documentation text.
- **Example / analogy:** The instructions printed on a tool.
- **Snippet:**
```python
def f():
    """Do a thing."""
help(f)
```
- **Snippet explained:** `help(f)` prints the docstring.
- **Related terms:** [[Function]]

### List
<!-- cat: python -->
- **Technical definition:** An ordered, mutable sequence written with [].
- **Plain English:** A changeable ordered collection.
- **Example / analogy:** A shopping list you can edit.
- **Snippet:**
```python
xs = [1, 2]
xs.append(3)
```
- **Snippet explained:** The list grows to `[1, 2, 3]`.
- **Related terms:** [[Tuple]], [[Slicing]]

### Tuple
<!-- cat: python -->
- **Technical definition:** An ordered, immutable sequence written with ().
- **Plain English:** A fixed record.
- **Example / analogy:** (latitude, longitude).
- **Snippet:**
```python
shape = (2, 3)
rows, cols = shape
```
- **Snippet explained:** Unpacks the tuple into two variables.
- **Related terms:** [[List]], [[Mutable vs immutable]]

### Dictionary
<!-- cat: python -->
- **Technical definition:** A mutable mapping from hashable keys to values with average O(1) lookup.
- **Plain English:** A lookup table.
- **Example / analogy:** A phone book: name -> number.
- **Snippet:**
```python
vocab = {'the': 0}
vocab.get('cat', -1)
```
- **Snippet explained:** `.get` returns -1 when the key is missing instead of raising KeyError.
- **Related terms:** [[Set]], [[Hash map]]

### Set
<!-- cat: python -->
- **Technical definition:** An unordered collection of unique hashable items with average O(1) membership tests.
- **Plain English:** A bag with no duplicates.
- **Example / analogy:** A guest list where each name appears once.
- **Snippet:**
```python
seen = {1, 2, 2}
3 in seen
```
- **Snippet explained:** The duplicate 2 is dropped; the `in` test is fast.
- **Related terms:** [[Dictionary]], [[Hash map]]

### Comprehension
<!-- cat: python -->
- **Technical definition:** Compact syntax that builds a list, dict or set from an iterable with optional filtering.
- **Plain English:** A one-line loop that builds a collection.
- **Example / analogy:** A factory line producing squares.
- **Snippet:**
```python
[n * n for n in range(5) if n % 2 == 0]
```
- **Snippet explained:** Squares of even numbers: `[0, 4, 16]`.
- **Related terms:** [[List]], [[Generator]]

### Mutable vs immutable
<!-- cat: python -->
- **Technical definition:** Mutable objects (list, dict, set) can change in place; immutable ones (int, str, tuple) cannot.
- **Plain English:** Can the thing be edited after creation?
- **Example / analogy:** Pencil vs ink.
- **Snippet:**
```python
a = [1]; b = a; b.append(2); print(a)
```
- **Snippet explained:** Prints `[1, 2]`: both names point to the same mutable list.
- **Related terms:** [[List]], [[Tuple]]

### Class
<!-- cat: python -->
- **Technical definition:** A blueprint defining attributes and methods for objects created from it.
- **Plain English:** A template for making objects.
- **Example / analogy:** A cookie cutter.
- **Snippet:**
```python
class Counter:
    def __init__(self):
        self.n = 0
```
- **Snippet explained:** Each `Counter()` has its own `n`.
- **Related terms:** [[Method]], [[Dataclass]]

### Method
<!-- cat: python -->
- **Technical definition:** A function defined inside a class, called on an instance and receiving it as `self`.
- **Plain English:** A function that belongs to an object.
- **Example / analogy:** `model.generate(...)`.
- **Snippet:**
```python
class A:
    def hi(self):
        return 'hi'
A().hi()
```
- **Snippet explained:** `A()` creates the object; `.hi()` runs the method.
- **Related terms:** [[Class]], [[Classmethod]]

### Classmethod
<!-- cat: python -->
- **Technical definition:** A method decorated with @classmethod that receives the class (`cls`); often an alternative constructor.
- **Plain English:** A function on the class itself that builds objects.
- **Example / analogy:** `from_pretrained(...)` builds a model from a name.
- **Snippet:**
```python
class M:
    @classmethod
    def make(cls):
        return cls()
```
- **Snippet explained:** `M.make()` returns a new M without needing an instance.
- **Related terms:** [[Class]], [[Decorator]]

### Dataclass
<!-- cat: python -->
- **Technical definition:** A class decorated with @dataclass that auto-generates __init__, __repr__ and __eq__ from annotated fields.
- **Plain English:** A class for holding data with less typing.
- **Example / analogy:** A training config.
- **Snippet:**
```python
from dataclasses import dataclass
@dataclass
class Cfg:
    lr: float = 1e-4
```
- **Snippet explained:** `Cfg()` prints as `Cfg(lr=0.0001)` automatically.
- **Related terms:** [[Class]], [[Type hint]]

### Inheritance
<!-- cat: python -->
- **Technical definition:** A class reusing and extending another class's behaviour via `class Child(Parent)`; `super()` calls the parent.
- **Plain English:** A new class built on an old one.
- **Example / analogy:** A sports car 'is a' car with extra features.
- **Snippet:**
```python
class B(A):
    def hi(self):
        return super().hi() + '!'
```
- **Snippet explained:** B overrides `hi` but reuses A's version.
- **Related terms:** [[Class]], [[Method]]

### Exception
<!-- cat: python -->
- **Technical definition:** An object representing an error raised during execution and handled with try/except.
- **Plain English:** A controlled way to fail.
- **Example / analogy:** A fire alarm instead of silent smoke.
- **Snippet:**
```python
try:
    1 / 0
except ZeroDivisionError as e:
    print(e)
```
- **Snippet explained:** Catches the error and prints `division by zero` instead of crashing.
- **Related terms:** [[Traceback]], [[Raise]]

### Raise
<!-- cat: python -->
- **Technical definition:** Statement that signals an error by throwing an exception object.
- **Plain English:** Deliberately reporting a problem.
- **Example / analogy:** Pulling the emergency brake.
- **Snippet:**
```python
raise ValueError('bad value')
```
- **Snippet explained:** Stops normal flow with a ValueError unless caught.
- **Related terms:** [[Exception]]

### Context manager (with)
<!-- cat: python -->
- **Technical definition:** An object used in a `with` statement that sets up and guarantees cleanup (e.g. closing a file).
- **Plain English:** Auto-cleanup block.
- **Example / analogy:** A self-closing door.
- **Snippet:**
```python
with open('x.txt') as f:
    data = f.read()
```
- **Snippet explained:** The file is closed automatically even if an error occurs.
- **Related terms:** [[File handling]], [[Exception]]

### pathlib.Path
<!-- cat: python -->
- **Technical definition:** Standard-library object-oriented file path with operators like `/`, `.exists()`, `.read_text()`.
- **Plain English:** A smart address of a file.
- **Example / analogy:** A GPS address instead of a street description.
- **Snippet:**
```python
from pathlib import Path
HERE = Path(__file__).resolve().parent
cfg = HERE / 'config.json'
```
- **Snippet explained:** Builds a path next to the script that works from any cwd.
- **Related terms:** [[File handling]], [[Working directory]]

### File handling
<!-- cat: python -->
- **Technical definition:** Reading and writing files with open()/Path methods using modes r, w, a and an explicit encoding.
- **Plain English:** Opening, reading and saving files.
- **Example / analogy:** Taking a document from a cabinet and returning it.
- **Snippet:**
```python
Path('n.txt').write_text('hi', encoding='utf-8')
```
- **Snippet explained:** Writes 'hi' as UTF-8 text.
- **Related terms:** [[pathlib.Path]], [[Context manager (with)]]

### JSON
<!-- cat: python -->
- **Technical definition:** A text data format of objects, arrays, strings, numbers, booleans and null; handled by the json module.
- **Plain English:** A universal text format for structured data.
- **Example / analogy:** The language APIs use to send dictionaries over the wire.
- **Snippet:**
```python
import json
json.dumps({'ok': True})
```
- **Snippet explained:** Gives `'{"ok": true}'`; `json.loads` reverses it.
- **Related terms:** [[Dictionary]], [[Serialization]]

### Serialization
<!-- cat: python -->
- **Technical definition:** Converting an in-memory object to bytes/text for storage or transfer, and back (deserialization).
- **Plain English:** Packing an object so it can be saved or sent.
- **Example / analogy:** Flat-packing furniture.
- **Snippet:**
```python
import json
s = json.dumps([1, 2])
json.loads(s)
```
- **Snippet explained:** List -> text -> list.
- **Related terms:** [[JSON]]

### Module and import
<!-- cat: python -->
- **Technical definition:** A module is a Python file; `import` loads it, searching sys.path (script folder, site-packages, stdlib).
- **Plain English:** Reusing code from another file.
- **Example / analogy:** Borrowing a book from a library.
- **Snippet:**
```python
import math
from collections import Counter
```
- **Snippet explained:** The first imports the module; the second only one name from it.
- **Related terms:** [[Module shadowing]], [[Virtual environment]]

### Main guard
<!-- cat: python -->
- **Technical definition:** `if __name__ == "__main__":` block that runs only when the file is executed directly, not imported.
- **Plain English:** Code that runs only when you start this file.
- **Example / analogy:** A 'staff only' door.
- **Snippet:**
```python
def main():
    print('hi')
if __name__ == '__main__':
    main()
```
- **Snippet explained:** Importing the file does not print; running it does.
- **Related terms:** [[Module and import]]

### Decorator
<!-- cat: python -->
- **Technical definition:** A callable that takes a function and returns a replacement; `@d` above `def f` equals `f = d(f)`.
- **Plain English:** A wrapper that adds behaviour to a function.
- **Example / analogy:** Gift wrapping around a present.
- **Snippet:**
```python
def timed(f):
    def w(*a, **k):
        return f(*a, **k)
    return w
@timed
def g(): ...
```
- **Snippet explained:** `g` is replaced by `w`, which calls the original.
- **Related terms:** [[args and kwargs]], [[Function]]

### Generator
<!-- cat: python -->
- **Technical definition:** A function using `yield` that produces values lazily one at a time, pausing between them.
- **Plain English:** A lazy producer; memory friendly.
- **Example / analogy:** A faucet vs a full bucket.
- **Snippet:**
```python
def count(n):
    for i in range(n):
        yield i
list(count(3))
```
- **Snippet explained:** Produces 0, 1, 2 only as requested.
- **Related terms:** [[Iterator]], [[Comprehension]]

### Hash map
<!-- cat: python -->
- **Technical definition:** A data structure mapping keys to values via a hash function, giving average O(1) lookup (Python's dict).
- **Plain English:** Fast key lookup.
- **Example / analogy:** A cloakroom with numbered tickets.
- **Snippet:**
```python
seen = {}
seen['a'] = 1
'a' in seen
```
- **Snippet explained:** Membership and lookup take about constant time.
- **Related terms:** [[Dictionary]], [[Set]]
