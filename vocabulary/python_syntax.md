# Python Syntax, Standard Library, Debugging and DSA Terms

120 terms, alphabetical. Back to [index](index.md).

- [.env file](#env-file)
- [args and kwargs](#args-and-kwargs)
- [Assertion](#assertion)
- [Assertion (assert)](#assertion-assert)
- [Async/await](#asyncawait)
- [AttributeError](#attributeerror)
- [Big-O notation](#big-o-notation)
- [Binary search](#binary-search)
- [Boolean](#boolean)
- [Breakpoint](#breakpoint)
- [Brute force](#brute-force)
- [Class](#class)
- [Classmethod](#classmethod)
- [Complement lookup](#complement-lookup)
- [Comprehension](#comprehension)
- [Conditional (if/elif/else)](#conditional-ifelifelse)
- [Configuration file](#configuration-file)
- [ConnectionError](#connectionerror)
- [Context manager (with)](#context-manager-with)
- [Counter](#counter)
- [Data type](#data-type)
- [Dataclass](#dataclass)
- [Decorator](#decorator)
- [Default argument](#default-argument)
- [defaultdict](#defaultdict)
- [Dependency conflict](#dependency-conflict)
- [Dictionary](#dictionary)
- [Dictionary merge ({**a, **b})](#dictionary-merge-a-b)
- [Docker container logs](#docker-container-logs)
- [Docstring](#docstring)
- [dotenv file](#dotenv-file)
- [Dry run](#dry-run)
- [Duplicate detection](#duplicate-detection)
- [Edge case](#edge-case)
- [enumerate](#enumerate)
- [Environment variable](#environment-variable)
- [Exception](#exception)
- [Exception handling](#exception-handling)
- [f-string](#f-string)
- [File handling](#file-handling)
- [FileNotFoundError](#filenotfounderror)
- [Fixture](#fixture)
- [Frequency counting](#frequency-counting)
- [Function](#function)
- [Generator](#generator)
- [Generator expression](#generator-expression)
- [GIL](#gil)
- [Grouping by key](#grouping-by-key)
- [Hash map](#hash-map)
- [Hash map (dict)](#hash-map-dict)
- [Hash set](#hash-set)
- [Hashing](#hashing)
- [HTTP status code](#http-status-code)
- [ImportError / ModuleNotFoundError](#importerror-modulenotfounderror)
- [In-place algorithm](#in-place-algorithm)
- [Inheritance](#inheritance)
- [Input validation](#input-validation)
- [Iterator](#iterator)
- [joblib](#joblib)
- [JSON](#json)
- [JSON (config format)](#json-config-format)
- [JSONL](#jsonl)
- [KeyError](#keyerror)
- [Keyword argument](#keyword-argument)
- [Keyword arguments (**kwargs)](#keyword-arguments-kwargs)
- [Lambda](#lambda)
- [List](#list)
- [List comprehension](#list-comprehension)
- [Literal type](#literal-type)
- [Log level](#log-level)
- [Logging](#logging)
- [Loop (for/while)](#loop-forwhile)
- [Main guard](#main-guard)
- [Method](#method)
- [Minimal reproducible example](#minimal-reproducible-example)
- [Module and import](#module-and-import)
- [Module shadowing](#module-shadowing)
- [Mutable default argument trap](#mutable-default-argument-trap)
- [Mutable vs immutable](#mutable-vs-immutable)
- [pathlib.Path](#pathlibpath)
- [Pickle](#pickle)
- [pip](#pip)
- [Prefix sum](#prefix-sum)
- [Print debugging](#print-debugging)
- [Pydantic model](#pydantic-model)
- [pytest](#pytest)
- [Raise](#raise)
- [Regression test](#regression-test)
- [Regular expression](#regular-expression)
- [requirements.txt](#requirementstxt)
- [Retry with backoff](#retry-with-backoff)
- [Root-cause analysis](#root-cause-analysis)
- [Runtime error](#runtime-error)
- [Serialization](#serialization)
- [Set](#set)
- [Slicing](#slicing)
- [Sliding window](#sliding-window)
- [Sorting (stable)](#sorting-stable)
- [Space complexity](#space-complexity)
- [Stack frame](#stack-frame)
- [str.format](#strformat)
- [string.Template](#stringtemplate)
- [Subarray](#subarray)
- [SyntaxError](#syntaxerror)
- [sys.path](#syspath)
- [Time complexity](#time-complexity)
- [Timeout](#timeout)
- [Traceback](#traceback)
- [Tuple](#tuple)
- [Two pointers](#two-pointers)
- [Two sum](#two-sum)
- [Type conversion](#type-conversion)
- [Type hint](#type-hint)
- [TypeError](#typeerror)
- [Valid anagram](#valid-anagram)
- [ValidationError](#validationerror)
- [Variable](#variable)
- [Virtual environment](#virtual-environment)
- [Working directory](#working-directory)
- [zip](#zip)

---

### .env file
*From module: 09_api_development*

- **Technical definition:** A text file of KEY=value lines loaded into the environment by tools like python-dotenv (never committed when it holds secrets).
- **Plain English:** A local settings file for environment variables.
- **Example / analogy:** A notepad of your private settings next to the project.
- **Snippet:**
```python
from dotenv import load_dotenv
load_dotenv()   # real env vars win unless override=True
```
- **Snippet explained:** Loads .env into os.environ without overriding variables already set.
- **Related terms:** [Environment variable](python_syntax.md#environment-variable), [Configuration](api_and_deployment_terms.md#configuration)

---

### args and kwargs
*From module: 00_python_foundations*

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
- **Related terms:** [Decorator](python_syntax.md#decorator), [Keyword argument](python_syntax.md#keyword-argument)

---

### Assertion
*From module: 01_debugging_and_environment*

- **Technical definition:** `assert cond, msg` raises AssertionError when cond is false; used to state and check assumptions or tests.
- **Plain English:** A claim the program checks for you.
- **Example / analogy:** A smoke detector for wrong assumptions.
- **Snippet:**
```python
assert average([2, 4]) == 3, 'average broken'
```
- **Snippet explained:** Silent if true; raises with the message if false.
- **Related terms:** [Regression test](python_syntax.md#regression-test)

---

### Assertion (assert)
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** A statement that raises AssertionError if its condition is False; used for simple tests.
- **Plain English:** A tripwire: 'I claim this is true; crash loudly if not.'
- **Example / analogy:** A smoke detector: silent when all is fine.
- **Snippet:**
```python
assert sum([1, 2]) == 3
```
- **Snippet explained:** If the claim is false Python stops with AssertionError and the failing line.
- **Related terms:** [Edge case](python_syntax.md#edge-case), [Dry run](python_syntax.md#dry-run)

---

### Async/await
*From module: 09_api_development*

- **Technical definition:** Syntax for coroutines: `async def` defines a function that can be paused; `await` pauses it until an awaitable completes, letting others run.
- **Plain English:** A way to say 'while I wait for this, go do something else'.
- **Example / analogy:** Putting the kettle on, then making toast while it heats.
- **Snippet:**
```python
async def f():
    await asyncio.sleep(1)
    return 'done'
```
- **Snippet explained:** Pauses 1 s without blocking the event loop; must itself be awaited or run with asyncio.run.
- **Related terms:** [Event loop](api_and_deployment_terms.md#event-loop), [Concurrency](api_and_deployment_terms.md#concurrency)

---

### AttributeError
*From module: 01_debugging_and_environment*

- **Technical definition:** Raised when an object has no such attribute or method; often the object is None.
- **Plain English:** Asked an object for something it doesn't have.
- **Example / analogy:** Asking a bicycle for its steering-wheel size.
- **Snippet:**
```python
None.upper()
```
- **Snippet explained:** `AttributeError: 'NoneType' object has no attribute 'upper'` usually means a missing `return`.
- **Related terms:** [TypeError](python_syntax.md#typeerror)

---

### Big-O notation
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Asymptotic upper bound that describes the growth rate of an algorithm's cost, ignoring constants and lower-order terms.
- **Plain English:** A label for how fast the work grows: O(1), O(log n), O(n), O(n log n), O(n^2).
- **Example / analogy:** Washing one dish is O(1); washing n dishes O(n); comparing every pair of guests at a party O(n^2).
- **Snippet:**
```python
n = 1000
print(n * (n - 1) // 2)   # pair comparisons: 499500 ~ O(n^2)
```
- **Snippet explained:** The number of pairs grows with the square of n, which is why nested loops get slow.
- **Related terms:** [Time complexity](python_syntax.md#time-complexity), [Brute force](python_syntax.md#brute-force), [Space complexity](python_syntax.md#space-complexity)

---

### Binary search
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Search a sorted array by repeatedly halving the range; O(log n).
- **Plain English:** Guess the middle, throw away the half that can't contain the answer.
- **Example / analogy:** Finding a word in a dictionary by opening to the middle.
- **Snippet:**
```python
mid = (lo + hi) // 2
if nums[mid] < target:
    lo = mid + 1
else:
    hi = mid - 1
```
- **Snippet explained:** Moves the search bounds into the half that may contain the target.
- **Related terms:** [Time complexity](python_syntax.md#time-complexity), [Sorting (stable)](python_syntax.md#sorting-stable)

---

### Boolean
*From module: 00_python_foundations*

- **Technical definition:** The type with values True and False; empty containers, 0 and None are 'falsy'.
- **Plain English:** A yes/no value.
- **Example / analogy:** bool([]) is False, bool('x') is True.
- **Snippet:**
```python
print(bool(0), bool(''), bool('false'))
```
- **Snippet explained:** Prints `False False True`: any non-empty string is truthy, even 'false'.
- **Related terms:** [Conditional (if/elif/else)](python_syntax.md#conditional-ifelifelse), [Type conversion](python_syntax.md#type-conversion)

---

### Breakpoint
*From module: 01_debugging_and_environment*

- **Technical definition:** `breakpoint()` pauses the program in the pdb debugger so you can inspect variables and step through lines.
- **Plain English:** Pause button inside your code.
- **Example / analogy:** Freezing a movie to look at one frame.
- **Snippet:**
```python
breakpoint()  # commands: n, s, p x, c, q
```
- **Snippet explained:** n runs the next line, p prints a variable, c continues.
- **Related terms:** [Print debugging](python_syntax.md#print-debugging), [Assertion](python_syntax.md#assertion)

---

### Brute force
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** A solution that tries every possible candidate without exploiting structure in the problem.
- **Plain English:** The simplest approach that obviously works, even if slow.
- **Example / analogy:** Trying every key on a ring until the door opens.
- **Snippet:**
```python
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        if nums[i] + nums[j] == target:
            return [i, j]
```
- **Snippet explained:** Checks all pairs, O(n^2); a correct baseline to compare an optimised solution against.
- **Related terms:** [Time complexity](python_syntax.md#time-complexity), [Complement lookup](python_syntax.md#complement-lookup), [Dry run](python_syntax.md#dry-run)

---

### Class
*From module: 00_python_foundations*

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
- **Related terms:** [Method](python_syntax.md#method), [Dataclass](python_syntax.md#dataclass)

---

### Classmethod
*From module: 00_python_foundations*

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
- **Related terms:** [Class](python_syntax.md#class), [Decorator](python_syntax.md#decorator)

---

### Complement lookup
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Searching a hash structure for target - x instead of scanning for a partner of x.
- **Plain English:** Instead of searching for a partner, ask 'have I already seen the number that completes me?'
- **Example / analogy:** At a dance, checking the registry for your partner instead of asking everyone.
- **Snippet:**
```python
complement = target - x
if complement in seen:
    return [seen[complement], i]
```
- **Snippet explained:** Constant-time dict check replaces an inner loop in Two Sum.
- **Related terms:** [Hash map (dict)](python_syntax.md#hash-map-dict), [Brute force](python_syntax.md#brute-force), [Two sum](python_syntax.md#two-sum)

---

### Comprehension
*From module: 00_python_foundations*

- **Technical definition:** Compact syntax that builds a list, dict or set from an iterable with optional filtering.
- **Plain English:** A one-line loop that builds a collection.
- **Example / analogy:** A factory line producing squares.
- **Snippet:**
```python
[n * n for n in range(5) if n % 2 == 0]
```
- **Snippet explained:** Squares of even numbers: `[0, 4, 16]`.
- **Related terms:** [List](python_syntax.md#list), [Generator](python_syntax.md#generator)

---

### Conditional (if/elif/else)
*From module: 00_python_foundations*

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
- **Related terms:** [Loop (for/while)](python_syntax.md#loop-forwhile), [Boolean](python_syntax.md#boolean)

---

### Configuration file
*From module: 05_llm_parameters*

- **Technical definition:** A file (JSON, YAML, .env) holding settings separate from code so they can change without editing code.
- **Plain English:** The settings menu of your program.
- **Example / analogy:** config.json holds training, generation and limit parameters validated by load_config.
- **Snippet:**
```python
import json
with open('config.json') as f:
    cfg = json.load(f)
```
- **Snippet explained:** Opens the file safely and parses JSON into nested dicts and lists.
- **Related terms:** [Hyperparameter](ml_terms.md#hyperparameter), [JSON (config format)](python_syntax.md#json-config-format), [Reproducibility](ml_terms.md#reproducibility)

---

### ConnectionError
*From module: 01_debugging_and_environment*

- **Technical definition:** Raised when a network connection cannot be established (server down, wrong host/port).
- **Plain English:** Nobody is listening at that address.
- **Example / analogy:** Dialling a number that is out of service.
- **Snippet:**
```python
try:
    requests.get('http://127.0.0.1:9', timeout=1)
except requests.ConnectionError:
    print('unreachable')
```
- **Snippet explained:** Catches the unreachable-server case separately from a slow server.
- **Related terms:** [Timeout](python_syntax.md#timeout), [HTTP status code](python_syntax.md#http-status-code)

---

### Context manager (with)
*From module: 00_python_foundations*

- **Technical definition:** An object used in a `with` statement that sets up and guarantees cleanup (e.g. closing a file).
- **Plain English:** Auto-cleanup block.
- **Example / analogy:** A self-closing door.
- **Snippet:**
```python
with open('x.txt') as f:
    data = f.read()
```
- **Snippet explained:** The file is closed automatically even if an error occurs.
- **Related terms:** [File handling](python_syntax.md#file-handling), [Exception](python_syntax.md#exception)

---

### Counter
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** collections.Counter: a dict subclass for counting hashable items; offers most_common(k) and arithmetic.
- **Plain English:** A ready-made tally chart.
- **Example / analogy:** A scoreboard that adds one each time you call out a name.
- **Snippet:**
```python
from collections import Counter
c = Counter([1, 1, 2])
print(c.most_common(1))   # [(1, 2)]
```
- **Snippet explained:** most_common(1) returns the single most frequent (item, count) pair.
- **Related terms:** [Frequency counting](python_syntax.md#frequency-counting), [defaultdict](python_syntax.md#defaultdict)

---

### Data type
*From module: 00_python_foundations*

- **Technical definition:** The class of a value (int, float, str, bool, list, dict ...) that decides which operations are allowed.
- **Plain English:** What kind of thing a value is.
- **Example / analogy:** You can add numbers, but adding text to a number is an error.
- **Snippet:**
```python
type(3), type(3.0), type('3')
```
- **Snippet explained:** Returns `<class 'int'>`, `<class 'float'>`, `<class 'str'>`.
- **Related terms:** [Variable](python_syntax.md#variable), [Type conversion](python_syntax.md#type-conversion)

---

### Dataclass
*From module: 00_python_foundations*

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
- **Related terms:** [Class](python_syntax.md#class), [Type hint](python_syntax.md#type-hint)

---

### Decorator
*From module: 00_python_foundations*

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
- **Related terms:** [args and kwargs](python_syntax.md#args-and-kwargs), [Function](python_syntax.md#function)

---

### Default argument
*From module: 00_python_foundations*

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
- **Related terms:** [Function](python_syntax.md#function), [Mutable default argument trap](python_syntax.md#mutable-default-argument-trap)

---

### defaultdict
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** collections.defaultdict: a dict that creates a default value for missing keys via a factory function.
- **Plain English:** A dict that never raises KeyError; it makes an empty entry for you.
- **Example / analogy:** A filing cabinet that adds a new empty folder when you ask for a missing one.
- **Snippet:**
```python
from collections import defaultdict
groups = defaultdict(list)
groups['a'].append(1)
```
- **Snippet explained:** Accessing groups['a'] auto-creates [] then append adds to it.
- **Related terms:** [Grouping by key](python_syntax.md#grouping-by-key), [Hash map (dict)](python_syntax.md#hash-map-dict)

---

### Dependency conflict
*From module: 01_debugging_and_environment*

- **Technical definition:** Two packages need incompatible versions of a third, so they can't both be satisfied.
- **Plain English:** Libraries disagreeing about which version to use.
- **Example / analogy:** Two recipes demanding different oven temperatures.
- **Snippet:**
```bash
python -m pip check
```
- **Snippet explained:** Reports installed packages whose requirements are not met.
- **Related terms:** [pip](python_syntax.md#pip), [Virtual environment](python_syntax.md#virtual-environment)

---

### Dictionary
*From module: 00_python_foundations*

- **Technical definition:** A mutable mapping from hashable keys to values with average O(1) lookup.
- **Plain English:** A lookup table.
- **Example / analogy:** A phone book: name -> number.
- **Snippet:**
```python
vocab = {'the': 0}
vocab.get('cat', -1)
```
- **Snippet explained:** `.get` returns -1 when the key is missing instead of raising KeyError.
- **Related terms:** [Set](python_syntax.md#set), [Hash map](python_syntax.md#hash-map)

---

### Dictionary merge ({**a, **b})
*From module: 05_llm_parameters*

- **Technical definition:** Creating a new dict from several others using ** unpacking; later dicts override earlier keys.
- **Plain English:** Layering settings: defaults first, overrides on top.
- **Example / analogy:** {**base_settings, **overrides} is how each experiment row changes only one parameter.
- **Snippet:**
```python
base = {'temperature': 0.8, 'top_k': 50}
print({**base, **{'temperature': 0.2}})
```
- **Snippet explained:** Output {'temperature': 0.2, 'top_k': 50}: the right-hand value wins.
- **Related terms:** [Configuration file](python_syntax.md#configuration-file), [Hyperparameter](ml_terms.md#hyperparameter)

---

### Docker container logs
*From module: 01_debugging_and_environment*

- **Technical definition:** Output from a container's stdout/stderr read with `docker logs`; the first place to look when a container misbehaves.
- **Plain English:** The container's printed output, tracebacks included.
- **Example / analogy:** The black-box recorder of a flight.
- **Snippet:**
```bash
docker ps -a
docker logs --tail 50 mycontainer
```
- **Snippet explained:** Shows whether it exited and the last 50 log lines.
- **Related terms:** [Traceback](python_syntax.md#traceback), [Logging](python_syntax.md#logging)

---

### Docstring
*From module: 00_python_foundations*

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
- **Related terms:** [Function](python_syntax.md#function)

---

### dotenv file
*From module: 01_debugging_and_environment*

- **Technical definition:** .env file of KEY=VALUE lines loaded into the environment by python-dotenv; never committed to git.
- **Plain English:** A local file of settings and secrets.
- **Example / analogy:** A private settings notebook.
- **Snippet:**
```python
from dotenv import load_dotenv
load_dotenv()
```
- **Snippet explained:** Reads `.env` and copies its pairs into os.environ.
- **Related terms:** [Environment variable](python_syntax.md#environment-variable)

---

### Dry run
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Manually executing code with a small example, recording variable values after each step.
- **Plain English:** Playing computer on paper before (or after) you code.
- **Example / analogy:** Following a recipe step by step with a tiny portion to check it works.
- **Snippet:**
```python
def total(nums):
    t = 0
    for i, x in enumerate(nums):
        t += x
        print(i, x, t)   # dry run printed
    return t
```
- **Snippet explained:** The print shows index, value and running total each loop so you can check your mental trace.
- **Related terms:** [Edge case](python_syntax.md#edge-case), [Assertion (assert)](python_syntax.md#assertion-assert), [Brute force](python_syntax.md#brute-force)

---

### Duplicate detection
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Determining whether any value repeats, usually with a set (O(n)) or by sorting (O(n log n)).
- **Plain English:** Checking if anything shows up twice.
- **Example / analogy:** Ticking names on an attendance list and noticing someone signs twice.
- **Snippet:**
```python
def has_dup(nums):
    return len(set(nums)) != len(nums)
```
- **Snippet explained:** A set drops duplicates; if its size is smaller the list had repeats.
- **Related terms:** [Hash set](python_syntax.md#hash-set), [Duplicate detection](python_syntax.md#duplicate-detection)

---

### Edge case
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** An input at the boundary of valid input that often breaks naive code: empty, single element, duplicates, negatives, maximum size.
- **Plain English:** The weird inputs you must test: nothing, one item, all the same, negatives.
- **Example / analogy:** A door that works for adults but jams for toddlers or giants.
- **Snippet:**
```python
assert find_max([]) is None
assert find_max([-5, -2]) == -2
```
- **Snippet explained:** Two edge cases: empty input and an all-negative list.
- **Related terms:** [Dry run](python_syntax.md#dry-run), [Assertion (assert)](python_syntax.md#assertion-assert)

---

### enumerate
*From module: 00_python_foundations*

- **Technical definition:** Built-in wrapping an iterable to yield (index, item) pairs.
- **Plain English:** A loop that also tells you the position.
- **Example / analogy:** Numbering the items on a to-do list.
- **Snippet:**
```python
for i, w in enumerate(['a', 'b']):
    print(i, w)
```
- **Snippet explained:** Prints `0 a` then `1 b`.
- **Related terms:** [Loop (for/while)](python_syntax.md#loop-forwhile), [zip](python_syntax.md#zip)

---

### Environment variable
*From module: 01_debugging_and_environment*

- **Technical definition:** A named string value supplied by the operating system/process (read via os.environ).
- **Plain English:** A setting given from outside the program.
- **Example / analogy:** A sticky note on the desk the program reads on startup.
- **Snippet:**
```python
import os
port = int(os.environ.get('PORT', '8000'))
```
- **Snippet explained:** Reads PORT as text, falling back to '8000', then converts to int.
- **Related terms:** [dotenv file](python_syntax.md#dotenv-file), [Type conversion](python_syntax.md#type-conversion)

---

### Exception
*From module: 00_python_foundations*

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
- **Related terms:** [Traceback](python_syntax.md#traceback), [Raise](python_syntax.md#raise)

---

### Exception handling
*From module: 99_extra_linked_terms*

- **Technical definition:** Using `try` / `except` / `else` / `finally` to catch errors raised at run time and respond instead of letting the program crash.
- **Plain English:** You say "try this; if it fails in this specific way, do that instead". Catch only the errors you can actually handle.
- **Example / analogy:** A safety net under a trapeze act.
- **Snippet:**
```python
try:
    value = int("abc")
except ValueError as err:
    print("not a number:", err)
```
- **Snippet explained:** `int("abc")` raises `ValueError`; the `except` block runs and the program continues.
- **Related terms:** [ValidationError](python_syntax.md#validationerror)

---

### f-string
*From module: 00_python_foundations*

- **Technical definition:** A string literal prefixed with f whose {expressions} are evaluated and formatted into the text.
- **Plain English:** A text template with blanks Python fills in.
- **Example / analogy:** Like a mail-merge letter.
- **Snippet:**
```python
loss = 2.7183
print(f'loss={loss:.2f}')
```
- **Snippet explained:** `{loss:.2f}` inserts loss with 2 decimals -> `loss=2.72`.
- **Related terms:** [Data type](python_syntax.md#data-type)

---

### File handling
*From module: 00_python_foundations*

- **Technical definition:** Reading and writing files with open()/Path methods using modes r, w, a and an explicit encoding.
- **Plain English:** Opening, reading and saving files.
- **Example / analogy:** Taking a document from a cabinet and returning it.
- **Snippet:**
```python
Path('n.txt').write_text('hi', encoding='utf-8')
```
- **Snippet explained:** Writes 'hi' as UTF-8 text.
- **Related terms:** [pathlib.Path](python_syntax.md#pathlibpath), [Context manager (with)](python_syntax.md#context-manager-with)

---

### FileNotFoundError
*From module: 01_debugging_and_environment*

- **Technical definition:** Raised when opening a path that doesn't exist; often a wrong cwd, spelling or case.
- **Plain English:** The file isn't where Python is looking.
- **Example / analogy:** Searching the wrong drawer.
- **Snippet:**
```python
from pathlib import Path
print(Path('x.txt').resolve())
```
- **Snippet explained:** Prints the exact absolute path Python would try.
- **Related terms:** [Working directory](python_syntax.md#working-directory)

---

### Fixture
*From module: 99_extra_linked_terms*

- **Technical definition:** In pytest, a function marked `@pytest.fixture` that prepares something tests need (a client, a temp file) and is injected by naming it as a test argument.
- **Plain English:** Shared set-up written once and reused by many tests.
- **Example / analogy:** Laying out the tools before each experiment.
- **Snippet:**
```python
import pytest
@pytest.fixture
def numbers():
    return [1, 2, 3]
def test_sum(numbers):
    assert sum(numbers) == 6
```
- **Snippet explained:** pytest sees the argument name `numbers` and passes in the fixture's return value.
- **Related terms:** [pytest](python_syntax.md#pytest)

---

### Frequency counting
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Counting how many times each distinct item occurs, typically with a dict or collections.Counter.
- **Plain English:** A tally chart.
- **Example / analogy:** Counting votes by making a mark next to each candidate name.
- **Snippet:**
```python
from collections import Counter
print(Counter('banana'))   # Counter({'a': 3, 'n': 2, 'b': 1})
```
- **Snippet explained:** Counter builds the item -> count mapping in one call.
- **Related terms:** [Counter](python_syntax.md#counter), [Hash map (dict)](python_syntax.md#hash-map-dict), [Valid anagram](python_syntax.md#valid-anagram)

---

### Function
*From module: 00_python_foundations*

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
- **Related terms:** [Default argument](python_syntax.md#default-argument), [Keyword argument](python_syntax.md#keyword-argument)

---

### Generator
*From module: 00_python_foundations*

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
- **Related terms:** [Iterator](python_syntax.md#iterator), [Comprehension](python_syntax.md#comprehension)

---

### Generator expression
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Lazy comprehension written with parentheses that yields items one at a time instead of building a list.
- **Plain English:** A list comprehension that doesn't store everything.
- **Example / analogy:** A tap versus a filled bucket.
- **Snippet:**
```python
total = sum(1 for x in nums if x > 0)
```
- **Snippet explained:** Counts positives without creating an intermediate list.
- **Related terms:** [List comprehension](python_syntax.md#list-comprehension)

---

### GIL
*From module: 99_extra_linked_terms*

- **Technical definition:** The Global Interpreter Lock: a lock in standard CPython that lets only one thread execute Python bytecode at a time.
- **Plain English:** Threads do not speed up CPU-heavy pure-Python code, though they help while waiting on network or disk. For CPU-bound work use multiple processes (as Gunicorn workers do) or libraries that release the lock.
- **Example / analogy:** One microphone in a room: many people can wait, only one speaks at once.
- **Snippet:** n/a
- **Snippet explained:** n/a
- **Related terms:** [Concurrency](api_and_deployment_terms.md#concurrency)

---

### Grouping by key
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Placing items into buckets keyed by a function of the item (e.g., sorted letters for anagrams).
- **Plain English:** Sorting things into piles by a label.
- **Example / analogy:** Sorting laundry by colour.
- **Snippet:**
```python
groups = defaultdict(list)
for w in words:
    groups[''.join(sorted(w))].append(w)
```
- **Snippet explained:** Anagrams share the sorted-letters key and land in the same bucket.
- **Related terms:** [defaultdict](python_syntax.md#defaultdict), [Tuple](python_syntax.md#tuple), [Valid anagram](python_syntax.md#valid-anagram)

---

### Hash map
*From module: 00_python_foundations*

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
- **Related terms:** [Dictionary](python_syntax.md#dictionary), [Set](python_syntax.md#set)

---

### Hash map (dict)
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** A key-value structure using a hash function to locate entries; average O(1) insert, lookup and delete. In Python: dict.
- **Plain English:** A lookup table: give a key, instantly get its value.
- **Example / analogy:** A coat-check: ticket number straight to your coat, no searching.
- **Snippet:**
```python
ages = {'amy': 30}
ages['bob'] = 25
print(ages['amy'])
```
- **Snippet explained:** Creates a dict, adds a key, reads a value; each step is a hash lookup.
- **Related terms:** [Hash set](python_syntax.md#hash-set), [Hashing](python_syntax.md#hashing), [Frequency counting](python_syntax.md#frequency-counting)

---

### Hash set
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** An unordered collection of unique hashable items with average O(1) membership test. In Python: set.
- **Plain English:** A bag that remembers what it has seen and ignores duplicates.
- **Example / analogy:** A guest list where checking a name is instant.
- **Snippet:**
```python
seen = set()
seen.add(3)
print(3 in seen)   # True
```
- **Snippet explained:** Adds 3 and tests membership in constant average time.
- **Related terms:** [Hash map (dict)](python_syntax.md#hash-map-dict), [Duplicate detection](python_syntax.md#duplicate-detection)

---

### Hashing
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Mapping a key to an integer (hash) used to pick a storage slot; keys must be immutable/hashable.
- **Plain English:** Turning a key into a number that says where to store it.
- **Example / analogy:** A librarian who computes a shelf number from the book title.
- **Snippet:**
```python
print(hash('abc') == hash('abc'))   # True within one run
# hash([1, 2])  -> TypeError: unhashable type: 'list'
```
- **Snippet explained:** Equal strings hash equally; lists are mutable so Python refuses to hash them.
- **Related terms:** [Hash map (dict)](python_syntax.md#hash-map-dict), [Tuple](python_syntax.md#tuple)

---

### HTTP status code
*From module: 01_debugging_and_environment*

- **Technical definition:** A 3-digit number in an HTTP response: 2xx success, 4xx client error, 5xx server error.
- **Plain English:** The server's short answer: ok, you erred, or I erred.
- **Example / analogy:** A shop assistant saying 'here you go', 'we don't have that' or 'our till is broken'.
- **Snippet:**
```python
r = requests.get(url, timeout=3)
r.status_code
```
- **Snippet explained:** 200 ok, 404 not found, 422 invalid body, 500 server bug.
- **Related terms:** [Timeout](python_syntax.md#timeout), [ConnectionError](python_syntax.md#connectionerror)

---

### ImportError / ModuleNotFoundError
*From module: 01_debugging_and_environment*

- **Technical definition:** Raised when a module cannot be found (ModuleNotFoundError) or a name cannot be imported from it (ImportError).
- **Plain English:** Python can't find or load what you asked to import.
- **Example / analogy:** A book not in this library branch.
- **Snippet:**
```python
import numpyy
```
- **Snippet explained:** `No module named 'numpyy'`: typo or not installed in this interpreter.
- **Related terms:** [Virtual environment](python_syntax.md#virtual-environment), [Module shadowing](python_syntax.md#module-shadowing)

---

### In-place algorithm
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Algorithm that transforms the input using O(1) extra space by modifying the input structure.
- **Plain English:** Edit the list itself instead of making a new one.
- **Example / analogy:** Rearranging books on the same shelf rather than copying them to another shelf.
- **Snippet:**
```python
nums[i], nums[j] = nums[j], nums[i]
```
- **Snippet explained:** Tuple swap modifies the list directly.
- **Related terms:** [Two pointers](python_syntax.md#two-pointers), [Space complexity](python_syntax.md#space-complexity)

---

### Inheritance
*From module: 00_python_foundations*

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
- **Related terms:** [Class](python_syntax.md#class), [Method](python_syntax.md#method)

---

### Input validation
*From module: 01_debugging_and_environment*

- **Technical definition:** Checking that data has the expected type, range and shape before using it.
- **Plain English:** Checking ingredients before cooking.
- **Example / analogy:** A bouncer checking IDs.
- **Snippet:**
```python
if not isinstance(n, int) or n <= 0:
    raise ValueError('n must be a positive int')
```
- **Snippet explained:** Rejects bad input early with a clear message.
- **Related terms:** [Pydantic model](python_syntax.md#pydantic-model), [TypeError](python_syntax.md#typeerror)

---

### Iterator
*From module: 00_python_foundations*

- **Technical definition:** An object that returns items one at a time via `next()`; any `for` loop uses one.
- **Plain English:** Something you can walk through step by step.
- **Example / analogy:** A conveyor belt delivering one box at a time.
- **Snippet:**
```python
it = iter([1, 2])
print(next(it), next(it))
```
- **Snippet explained:** `iter` creates the iterator; `next` pulls items; a third `next` raises StopIteration.
- **Related terms:** [Generator](python_syntax.md#generator), [Loop (for/while)](python_syntax.md#loop-forwhile)

---

### joblib
*From module: 10_ml_system_architecture*

- **Technical definition:** A library for saving/loading Python objects (like pickle) with efficient handling of large numpy arrays; the standard way to store scikit-learn models. Same security caveats as pickle.
- **Plain English:** Pickle tuned for machine-learning models.
- **Example / analogy:** A better suitcase for bulky data.
- **Snippet:**
```python
import joblib
joblib.dump(model, 'm.joblib', compress=3)
```
- **Snippet explained:** Saves with compression level 3.
- **Related terms:** [Pickle](python_syntax.md#pickle), [Serialization](python_syntax.md#serialization)

---

### JSON
*From module: 00_python_foundations*

- **Technical definition:** A text data format of objects, arrays, strings, numbers, booleans and null; handled by the json module.
- **Plain English:** A universal text format for structured data.
- **Example / analogy:** The language APIs use to send dictionaries over the wire.
- **Snippet:**
```python
import json
json.dumps({'ok': True})
```
- **Snippet explained:** Gives `'{"ok": true}'`; `json.loads` reverses it.
- **Related terms:** [Dictionary](python_syntax.md#dictionary), [Serialization](python_syntax.md#serialization)

---

### JSON (config format)
*From module: 05_llm_parameters*

- **Technical definition:** A text format for nested dictionaries, lists, numbers, strings, true/false/null. No comments or trailing commas.
- **Plain English:** A universal way to write structured data as text.
- **Example / analogy:** {"temperature": 0.8, "top_k": 50, "do_sample": true}
- **Snippet:**
```python
import json
json.loads('{"a": 1}')   # {'a': 1}
```
- **Snippet explained:** loads parses a JSON string; json.load reads from a file object.
- **Related terms:** [Configuration file](python_syntax.md#configuration-file)

---

### JSONL
*From module: 99_extra_linked_terms*

- **Technical definition:** "JSON Lines": a text format with one complete JSON object per line, so files can be appended to and read line by line.
- **Plain English:** A log file where each line is a self-contained record you can parse with `json.loads`; one broken line does not ruin the rest.
- **Example / analogy:** A notebook where every line is a separate, complete index card.
- **Snippet:**
```python
import json
with open("trace.jsonl", "a") as f:
    f.write(json.dumps({"tool": "calc", "ok": True}) + "\n")
```
- **Snippet explained:** `json.dumps` makes one line of JSON; the newline separates records; mode `"a"` appends.
- **Related terms:** [JSON (JavaScript Object Notation)](api_and_deployment_terms.md#json-javascript-object-notation)

---

### KeyError
*From module: 01_debugging_and_environment*

- **Technical definition:** Raised when a dict key is not found.
- **Plain English:** Asked for something not in the lookup table.
- **Example / analogy:** Looking up a name missing from the phone book.
- **Snippet:**
```python
d = {'a': 1}
d.get('b', 0)
```
- **Snippet explained:** `.get` avoids KeyError by returning a default.
- **Related terms:** [Dictionary](python_syntax.md#dictionary)

---

### Keyword argument
*From module: 00_python_foundations*

- **Technical definition:** An argument passed by parameter name (`f(x, temperature=0.2)`) instead of by position.
- **Plain English:** Naming the setting you change.
- **Example / analogy:** Saying 'a large coffee with oat milk' instead of by order.
- **Snippet:**
```python
gen('hi', n=2)
```
- **Snippet explained:** `n=2` sets only n; other parameters keep defaults.
- **Related terms:** [Default argument](python_syntax.md#default-argument), [args and kwargs](python_syntax.md#args-and-kwargs)

---

### Keyword arguments (**kwargs)
*From module: 08_agent_harness*

- **Technical definition:** Passing a dict as named arguments with `f(**d)`, or collecting extra named arguments with `def f(**kwargs)`.
- **Plain English:** Turning a dict into named inputs for a function.
- **Example / analogy:** Handing over a form where each box already has a label.
- **Snippet:**
```python
def add(a, b):
    return a + b
print(add(**{"a": 1, "b": 2}))
```
- **Snippet explained:** Prints 3: the dict keys must match parameter names or `TypeError` is raised.
- **Related terms:** [Tool dispatch](llm_terms.md#tool-dispatch), [Argument validation](llm_terms.md#argument-validation)

---

### Lambda
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** An anonymous single-expression function written with the lambda keyword.
- **Plain English:** A tiny throw-away function.
- **Example / analogy:** A sticky note instruction rather than a full manual.
- **Snippet:**
```python
sorted(words, key=lambda w: len(w))
```
- **Snippet explained:** The lambda tells sort to order words by length.
- **Related terms:** [Sorting (stable)](python_syntax.md#sorting-stable), [List comprehension](python_syntax.md#list-comprehension)

---

### List
*From module: 00_python_foundations*

- **Technical definition:** An ordered, mutable sequence written with [].
- **Plain English:** A changeable ordered collection.
- **Example / analogy:** A shopping list you can edit.
- **Snippet:**
```python
xs = [1, 2]
xs.append(3)
```
- **Snippet explained:** The list grows to `[1, 2, 3]`.
- **Related terms:** [Tuple](python_syntax.md#tuple), [Slicing](python_syntax.md#slicing)

---

### List comprehension
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Compact syntax [expr for item in iterable if cond] to build a list.
- **Plain English:** A one-line loop that builds a list.
- **Example / analogy:** A conveyor belt that transforms each item as it passes.
- **Snippet:**
```python
squares = [x * x for x in range(5) if x % 2 == 0]   # [0, 4, 16]
```
- **Snippet explained:** Loops 0..4, keeps evens, squares them.
- **Related terms:** [Generator expression](python_syntax.md#generator-expression), [Lambda](python_syntax.md#lambda)

---

### Literal type
*From module: 07_prompt_engineering*

- **Technical definition:** A type hint (`typing.Literal`) restricting a value to specific constants.
- **Plain English:** A field that can only be one of these exact values.
- **Example / analogy:** A traffic light: red, amber or green only.
- **Snippet:**
```python
from typing import Literal
Mode = Literal["fast", "safe"]
```
- **Snippet explained:** A variable annotated `Mode` should be exactly `"fast"` or `"safe"`.
- **Related terms:** [Pydantic model](python_syntax.md#pydantic-model), [Type hint](python_syntax.md#type-hint)

---

### Log level
*From module: 01_debugging_and_environment*

- **Technical definition:** A severity label (DEBUG < INFO < WARNING < ERROR < CRITICAL); messages below the configured level are dropped.
- **Plain English:** How important a message is.
- **Example / analogy:** Volume knob for diagnostics.
- **Snippet:**
```python
log.warning('empty input')
```
- **Snippet explained:** Shown at INFO or DEBUG levels, hidden if level is ERROR.
- **Related terms:** [Logging](python_syntax.md#logging)

---

### Logging
*From module: 01_debugging_and_environment*

- **Technical definition:** The logging module records messages with levels (DEBUG..CRITICAL), timestamps and destinations.
- **Plain English:** A structured diary of what the program did.
- **Example / analogy:** A ship's log vs scribbled notes.
- **Snippet:**
```python
import logging
logging.basicConfig(level='INFO')
logging.getLogger('app').info('started')
```
- **Snippet explained:** Prints a timestamped INFO line; DEBUG lines would be hidden.
- **Related terms:** [Print debugging](python_syntax.md#print-debugging), [Log level](python_syntax.md#log-level)

---

### Loop (for/while)
*From module: 00_python_foundations*

- **Technical definition:** Constructs that repeat a block: `for` over an iterable, `while` while a condition holds.
- **Plain English:** Doing something again and again.
- **Example / analogy:** Washing every dish (for) vs scrubbing until clean (while).
- **Snippet:**
```python
for i in range(3):
    print(i)
```
- **Snippet explained:** Prints 0, 1, 2; `range(3)` stops before 3.
- **Related terms:** [Conditional (if/elif/else)](python_syntax.md#conditional-ifelifelse), [Iterator](python_syntax.md#iterator)

---

### Main guard
*From module: 00_python_foundations*

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
- **Related terms:** [Module and import](python_syntax.md#module-and-import)

---

### Method
*From module: 00_python_foundations*

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
- **Related terms:** [Class](python_syntax.md#class), [Classmethod](python_syntax.md#classmethod)

---

### Minimal reproducible example
*From module: 01_debugging_and_environment*

- **Technical definition:** The smallest self-contained code and data that still shows a bug, with expected vs actual output.
- **Plain English:** A tiny version of the problem anyone can run.
- **Example / analogy:** Isolating the squeak in a car by removing parts.
- **Snippet:**
```python
def add_item(x, bucket=[]):
    bucket.append(x); return bucket
print(add_item(1), add_item(2))
```
- **Snippet explained:** Six lines reproduce a bug found in a 40-line program.
- **Related terms:** [Root-cause analysis](python_syntax.md#root-cause-analysis), [Mutable default argument trap](python_syntax.md#mutable-default-argument-trap)

---

### Module and import
*From module: 00_python_foundations*

- **Technical definition:** A module is a Python file; `import` loads it, searching sys.path (script folder, site-packages, stdlib).
- **Plain English:** Reusing code from another file.
- **Example / analogy:** Borrowing a book from a library.
- **Snippet:**
```python
import math
from collections import Counter
```
- **Snippet explained:** The first imports the module; the second only one name from it.
- **Related terms:** [Module shadowing](python_syntax.md#module-shadowing), [Virtual environment](python_syntax.md#virtual-environment)

---

### Module shadowing
*From module: 01_debugging_and_environment*

- **Technical definition:** A local file or folder with the same name as a library hides the real one because the script folder is first on sys.path.
- **Plain English:** Your file named like a library breaks the library.
- **Example / analogy:** A fake sign hiding the real shop.
- **Snippet:**
```python
# a local file named json.py makes this fail:
import json
json.loads('1')
```
- **Snippet explained:** Python imports your json.py, which lacks `loads`.
- **Related terms:** [Module and import](python_syntax.md#module-and-import), [ImportError / ModuleNotFoundError](python_syntax.md#importerror-modulenotfounderror)

---

### Mutable default argument trap
*From module: 00_python_foundations*

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
- **Related terms:** [Default argument](python_syntax.md#default-argument), [Mutable vs immutable](python_syntax.md#mutable-vs-immutable)

---

### Mutable vs immutable
*From module: 00_python_foundations*

- **Technical definition:** Mutable objects (list, dict, set) can change in place; immutable ones (int, str, tuple) cannot.
- **Plain English:** Can the thing be edited after creation?
- **Example / analogy:** Pencil vs ink.
- **Snippet:**
```python
a = [1]; b = a; b.append(2); print(a)
```
- **Snippet explained:** Prints `[1, 2]`: both names point to the same mutable list.
- **Related terms:** [List](python_syntax.md#list), [Tuple](python_syntax.md#tuple)

---

### pathlib.Path
*From module: 00_python_foundations*

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
- **Related terms:** [File handling](python_syntax.md#file-handling), [Working directory](python_syntax.md#working-directory)

---

### Pickle
*From module: 10_ml_system_architecture*

- **Technical definition:** Python's built-in binary serialization format for arbitrary objects; loading a pickle can execute arbitrary code, so only load trusted files.
- **Plain English:** Python's own 'save anything' format - convenient but unsafe from strangers.
- **Example / analogy:** A sealed box that may contain a booby trap if you don't trust the sender.
- **Snippet:**
```python
with open('m.pkl', 'wb') as f:
    pickle.dump(model, f)
```
- **Snippet explained:** Writes the object in binary mode.
- **Related terms:** [Serialization](python_syntax.md#serialization), [joblib](python_syntax.md#joblib), [safetensors](ml_terms.md#safetensors)

---

### pip
*From module: 01_debugging_and_environment*

- **Technical definition:** Python's package installer; `python -m pip` ensures it matches the interpreter.
- **Plain English:** The app store for Python libraries.
- **Example / analogy:** Ordering parts from a catalogue.
- **Snippet:**
```bash
python -m pip install numpy
python -m pip show numpy
python -m pip check
```
- **Snippet explained:** Install, inspect and check dependency consistency.
- **Related terms:** [Virtual environment](python_syntax.md#virtual-environment), [requirements.txt](python_syntax.md#requirementstxt)

---

### Prefix sum
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** An array where prefix[i] is the sum of the first i elements, enabling O(1) range-sum queries.
- **Plain English:** A running total you keep so you can subtract two totals to get any range.
- **Example / analogy:** Odometer readings: distance between two stops = difference of readings.
- **Snippet:**
```python
prefix = [0]
for x in nums:
    prefix.append(prefix[-1] + x)
range_sum = prefix[r + 1] - prefix[l]
```
- **Snippet explained:** Build running totals once, subtract two entries to get sum of nums[l..r].
- **Related terms:** [Sliding window](python_syntax.md#sliding-window), [Subarray](python_syntax.md#subarray)

---

### Print debugging
*From module: 01_debugging_and_environment*

- **Technical definition:** Inserting print statements to inspect values while running.
- **Plain English:** Quick-and-dirty peeking at values.
- **Example / analogy:** Torch in a dark room.
- **Snippet:**
```python
print(type(x), repr(x), len(x))
```
- **Snippet explained:** Shows the kind, exact value and size of x.
- **Related terms:** [Logging](python_syntax.md#logging), [Breakpoint](python_syntax.md#breakpoint)

---

### Pydantic model
*From module: 01_debugging_and_environment*

- **Technical definition:** A class deriving from BaseModel whose type-hinted fields are validated and coerced when instances are created.
- **Plain English:** A schema that checks data automatically.
- **Example / analogy:** A form that rejects invalid answers.
- **Snippet:**
```python
from pydantic import BaseModel, Field
class R(BaseModel):
    n: int = Field(gt=0)
R(n='5')
```
- **Snippet explained:** The string '5' is coerced to 5; `-1` would raise ValidationError.
- **Related terms:** [Input validation](python_syntax.md#input-validation), [Type hint](python_syntax.md#type-hint)

---

### pytest
*From module: 09_api_development*

- **Technical definition:** A Python test framework that discovers `test_*` functions, uses plain `assert`, fixtures and parametrization.
- **Plain English:** A tool that runs your checks and reports which fail.
- **Example / analogy:** A checklist an inspector ticks off automatically.
- **Snippet:**
```python
def test_health(client):
    assert client.get('/health').status_code == 200
```
- **Snippet explained:** Fails with a readable diff if the status is not 200.
- **Related terms:** [Test client](api_and_deployment_terms.md#test-client), [Fixture](python_syntax.md#fixture), [Smoke test](api_and_deployment_terms.md#smoke-test)

---

### Raise
*From module: 00_python_foundations*

- **Technical definition:** Statement that signals an error by throwing an exception object.
- **Plain English:** Deliberately reporting a problem.
- **Example / analogy:** Pulling the emergency brake.
- **Snippet:**
```python
raise ValueError('bad value')
```
- **Snippet explained:** Stops normal flow with a ValueError unless caught.
- **Related terms:** [Exception](python_syntax.md#exception)

---

### Regression test
*From module: 01_debugging_and_environment*

- **Technical definition:** A test added after a bug fix that fails on the old behaviour and passes on the fixed one, preventing it returning.
- **Plain English:** A guard so the bug cannot return unnoticed.
- **Example / analogy:** A plug in the hole you just fixed.
- **Snippet:**
```python
assert add_item_fixed('b') == ['b']
```
- **Snippet explained:** Would have failed with the buggy shared list.
- **Related terms:** [Assertion](python_syntax.md#assertion), [Root-cause analysis](python_syntax.md#root-cause-analysis)

---

### Regular expression
*From module: 99_extra_linked_terms*

- **Technical definition:** A compact pattern language for matching and extracting text, available in Python through the `re` module.
- **Plain English:** A search pattern more powerful than "contains": for example "the first {...} block in this reply".
- **Example / analogy:** Find-and-replace on steroids.
- **Snippet:**
```python
import re
re.search(r"\d+", "order 66 shipped").group()   # '66'
```
- **Snippet explained:** `\d+` means one or more digits; `search` finds the first match and `.group()` returns its text.
- **Related terms:** [Parser (reply parser)](llm_terms.md#parser-reply-parser)

---

### requirements.txt
*From module: 01_debugging_and_environment*

- **Technical definition:** A text file listing package names (and versions) so an environment can be recreated with `pip install -r`.
- **Plain English:** A shopping list for the project's libraries.
- **Example / analogy:** A recipe's ingredient list.
- **Snippet:**
```bash
numpy==2.5.4
requests>=2.30
```
- **Snippet explained:** One requirement per line; pins keep versions reproducible.
- **Related terms:** [pip](python_syntax.md#pip), [Dependency conflict](python_syntax.md#dependency-conflict)

---

### Retry with backoff
*From module: 01_debugging_and_environment*

- **Technical definition:** Re-attempting a failed transient operation after increasing delays.
- **Plain English:** Try again, waiting longer each time.
- **Example / analogy:** Knocking again after 1 s, then 2 s, then 4 s.
- **Snippet:**
```python
import time
for wait in [1, 2, 4]:
    try:
        ...
        break
    except requests.Timeout:
        time.sleep(wait)
```
- **Snippet explained:** Waits longer after each failed attempt.
- **Related terms:** [Timeout](python_syntax.md#timeout), [HTTP status code](python_syntax.md#http-status-code)

---

### Root-cause analysis
*From module: 01_debugging_and_environment*

- **Technical definition:** Systematically tracing a failure back to the underlying cause, testing hypotheses one at a time.
- **Plain English:** Finding why, not just what.
- **Example / analogy:** Treating the illness, not the fever.
- **Snippet:**
n/a
- **Snippet explained:** Observe, reproduce, shrink, hypothesise, test, fix, verify.
- **Related terms:** [Minimal reproducible example](python_syntax.md#minimal-reproducible-example), [Regression test](python_syntax.md#regression-test)

---

### Runtime error
*From module: 01_debugging_and_environment*

- **Technical definition:** An error that occurs while a valid program is executing (KeyError, TypeError, ZeroDivisionError ...).
- **Plain English:** A crash during running.
- **Example / analogy:** A recipe that is fine on paper but fails when the egg is missing.
- **Snippet:**
```python
1 / 0
```
- **Snippet explained:** Raises ZeroDivisionError at run time.
- **Related terms:** [SyntaxError](python_syntax.md#syntaxerror), [Exception](python_syntax.md#exception)

---

### Serialization
*From module: 00_python_foundations*

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
- **Related terms:** [JSON](python_syntax.md#json)

---

### Set
*From module: 00_python_foundations*

- **Technical definition:** An unordered collection of unique hashable items with average O(1) membership tests.
- **Plain English:** A bag with no duplicates.
- **Example / analogy:** A guest list where each name appears once.
- **Snippet:**
```python
seen = {1, 2, 2}
3 in seen
```
- **Snippet explained:** The duplicate 2 is dropped; the `in` test is fast.
- **Related terms:** [Dictionary](python_syntax.md#dictionary), [Hash map](python_syntax.md#hash-map)

---

### Slicing
*From module: 00_python_foundations*

- **Technical definition:** Selecting a sub-sequence with `seq[start:stop:step]`; stop is exclusive and negative indices count from the end.
- **Plain English:** Cutting a piece out of a list or string.
- **Example / analogy:** Taking pages 2 to 4 of a book.
- **Snippet:**
```python
'distilgpt2'[0:4]  # 'dist'
'abc'[::-1]       # 'cba'
```
- **Snippet explained:** First gives characters 0-3; second reverses with step -1.
- **Related terms:** [List](python_syntax.md#list)

---

### Sliding window
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Maintain a contiguous range [left, right] and update its aggregate incrementally while moving the edges.
- **Plain English:** A frame that slides along the list, adding one item and dropping one.
- **Example / analogy:** A magnifying glass moved along a line of text.
- **Snippet:**
```python
window += nums[right] - nums[right - k]
```
- **Snippet explained:** One element enters, one leaves; the sum updates in O(1).
- **Related terms:** [Two pointers](python_syntax.md#two-pointers), [Prefix sum](python_syntax.md#prefix-sum)

---

### Sorting (stable)
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Sorting where equal elements keep their original relative order; Python's sorted()/list.sort() use stable Timsort, O(n log n).
- **Plain English:** Ties stay in the order they were in before.
- **Example / analogy:** Sorting a class list by grade: students with the same grade stay alphabetical if they were before.
- **Snippet:**
```python
sorted([('a', 2), ('b', 1), ('c', 2)], key=lambda p: p[1])
```
- **Snippet explained:** Result keeps ('a',2) before ('c',2) because the sort is stable.
- **Related terms:** [Lambda](python_syntax.md#lambda), [Binary search](python_syntax.md#binary-search)

---

### Space complexity
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** How much extra memory an algorithm needs as input size grows, not counting the input itself.
- **Plain English:** How much extra scratch paper your solution needs.
- **Example / analogy:** Two Sum with a dict keeps up to n entries: O(n) extra space.
- **Snippet:**
```python
seen = {}   # grows with the input -> O(n) space
```
- **Snippet explained:** The dict stores at most one entry per element, so memory grows linearly.
- **Related terms:** [Time complexity](python_syntax.md#time-complexity), [Hash map (dict)](python_syntax.md#hash-map-dict)

---

### Stack frame
*From module: 01_debugging_and_environment*

- **Technical definition:** One entry in the call stack: a function call with its file, line number and local variables.
- **Plain English:** One step in the chain of calls.
- **Example / analogy:** One floor in a tall building you are descending.
- **Snippet:**
```python
import traceback
for fr in traceback.extract_tb(err.__traceback__):
    print(fr.name, fr.lineno)
```
- **Snippet explained:** Lists each function and line in the failing chain.
- **Related terms:** [Traceback](python_syntax.md#traceback)

---

### str.format
*From module: 07_prompt_engineering*

- **Technical definition:** A string method that replaces `{name}` placeholders with keyword/positional arguments; braces must be doubled to appear literally.
- **Plain English:** A string with named slots filled later.
- **Example / analogy:** A mail-merge template.
- **Snippet:**
```python
"{a}-{b}".format(a=1, b=2)
'{{"k": 1}}'.format()
```
- **Snippet explained:** First gives `1-2`; second prints `{"k": 1}` using doubled braces.
- **Related terms:** [f-string](python_syntax.md#f-string), [Prompt template](llm_terms.md#prompt-template), [string.Template](python_syntax.md#stringtemplate)

---

### string.Template
*From module: 07_prompt_engineering*

- **Technical definition:** Standard-library template class using `$name` placeholders; braces are not special, so it suits text containing JSON.
- **Plain English:** A template whose slots look like $name.
- **Example / analogy:** Mail-merge that won't be confused by curly braces.
- **Snippet:**
```python
from string import Template
Template('{"label": "$x"}').substitute(x="ok")
```
- **Snippet explained:** Substitutes `$x`; braces stay untouched.
- **Related terms:** [str.format](python_syntax.md#strformat), [Prompt template](llm_terms.md#prompt-template)

---

### Subarray
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** A contiguous slice of an array (unlike a subsequence, which may skip elements).
- **Plain English:** A continuous chunk of the list.
- **Example / analogy:** A few adjacent pages of a book.
- **Snippet:**
```python
nums[2:5]   # elements at index 2,3,4
```
- **Snippet explained:** Slicing gives a contiguous subarray (copy).
- **Related terms:** [Sliding window](python_syntax.md#sliding-window), [Prefix sum](python_syntax.md#prefix-sum)

---

### SyntaxError
*From module: 01_debugging_and_environment*

- **Technical definition:** An error found while compiling source code, before any of it runs, because the text is not valid Python.
- **Plain English:** Grammar mistake in the code.
- **Example / analogy:** A sentence with a missing full stop that nobody can read.
- **Snippet:**
```python
compile('if x print(1)', 'f', 'exec')
```
- **Snippet explained:** Raises SyntaxError, so no line of the file would have executed.
- **Related terms:** [Traceback](python_syntax.md#traceback), [Runtime error](python_syntax.md#runtime-error)

---

### sys.path
*From module: 01_debugging_and_environment*

- **Technical definition:** The list of directories Python searches when importing, starting with the script's folder.
- **Plain English:** Where Python looks for modules.
- **Example / analogy:** The list of shelves checked in order.
- **Snippet:**
```python
import sys
print(sys.path[:2])
```
- **Snippet explained:** Shows the first two search locations.
- **Related terms:** [Module shadowing](python_syntax.md#module-shadowing)

---

### Time complexity
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** A function describing how the number of basic operations grows with input size n, usually expressed in Big-O notation.
- **Plain English:** How much slower your code gets when the input gets bigger.
- **Example / analogy:** Looking for a name in an unsorted phone list (read every line) vs jumping straight to the page in an index.
- **Snippet:**
```python
for x in nums:      # one pass -> O(n)
    pass
```
- **Snippet explained:** A single loop does work proportional to len(nums), so doubling the list doubles the work.
- **Related terms:** [Space complexity](python_syntax.md#space-complexity), [Big-O notation](python_syntax.md#big-o-notation), [Brute force](python_syntax.md#brute-force)

---

### Timeout
*From module: 01_debugging_and_environment*

- **Technical definition:** The maximum time to wait for a response before raising an error; requests has none by default.
- **Plain English:** How long you're willing to wait.
- **Example / analogy:** Hanging up if nobody answers after 10 rings.
- **Snippet:**
```python
requests.get(url, timeout=3)
```
- **Snippet explained:** Raises `requests.Timeout` if no response within 3 s.
- **Related terms:** [HTTP status code](python_syntax.md#http-status-code), [ConnectionError](python_syntax.md#connectionerror)

---

### Traceback
*From module: 01_debugging_and_environment*

- **Technical definition:** The printed call stack at the point an exception was raised, oldest call first and the exception type/message last.
- **Plain English:** The trail of function calls that led to the crash.
- **Example / analogy:** Footprints leading to the spot where something broke.
- **Snippet:**
```python
def f(): return {}['x']
f()
```
- **Snippet explained:** Last line `KeyError: 'x'`; the frame above points to the `return` line.
- **Related terms:** [Exception](python_syntax.md#exception), [Stack frame](python_syntax.md#stack-frame)

---

### Tuple
*From module: 00_python_foundations*

- **Technical definition:** An ordered, immutable sequence written with ().
- **Plain English:** A fixed record.
- **Example / analogy:** (latitude, longitude).
- **Snippet:**
```python
shape = (2, 3)
rows, cols = shape
```
- **Snippet explained:** Unpacks the tuple into two variables.
- **Related terms:** [List](python_syntax.md#list), [Mutable vs immutable](python_syntax.md#mutable-vs-immutable)

---

### Two pointers
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Technique with two indices moving through a sequence (often from both ends of a sorted array) to avoid nested loops.
- **Plain English:** Two fingers walking along the list.
- **Example / analogy:** Two people meeting from opposite ends of a corridor.
- **Snippet:**
```python
left, right = 0, len(nums) - 1
while left < right:
    left += 1
    right -= 1
```
- **Snippet explained:** Pointers move inward until they meet.
- **Related terms:** [Sliding window](python_syntax.md#sliding-window), [In-place algorithm](python_syntax.md#in-place-algorithm)

---

### Two sum
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Classic problem: find indices of two numbers in an array that add to a target; O(n) with a dict.
- **Plain English:** Find the pair that adds up to the goal.
- **Example / analogy:** Finding two coins in a jar that total exactly 9 rupees.
- **Snippet:**
```python
seen = {}
for i, x in enumerate(nums):
    if target - x in seen:
        return [seen[target - x], i]
    seen[x] = i
```
- **Snippet explained:** Check for the complement before storing the current number.
- **Related terms:** [Complement lookup](python_syntax.md#complement-lookup), [Two pointers](python_syntax.md#two-pointers)

---

### Type conversion
*From module: 00_python_foundations*

- **Technical definition:** Explicitly turning a value into another type with int(), float(), str(), bool().
- **Plain English:** Changing a value's kind, like text '5' into number 5.
- **Example / analogy:** Values read from env vars or files are strings and must be converted.
- **Snippet:**
```python
int('42') + 1
```
- **Snippet explained:** `int('42')` makes the number 42, so adding 1 gives 43.
- **Related terms:** [Data type](python_syntax.md#data-type), [Environment variable](python_syntax.md#environment-variable)

---

### Type hint
*From module: 00_python_foundations*

- **Technical definition:** Optional annotation of parameter and return types (`x: int -> str`) used by tools; not enforced at runtime.
- **Plain English:** A note telling readers and editors what type is expected.
- **Example / analogy:** A label on a box saying 'fragile'.
- **Snippet:**
```python
def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs)
```
- **Snippet explained:** The hints say xs is a list of floats and the result is a float; Python still runs it with any input.
- **Related terms:** [Pydantic model](python_syntax.md#pydantic-model), [Function](python_syntax.md#function)

---

### TypeError
*From module: 01_debugging_and_environment*

- **Technical definition:** Raised when an operation gets a value of the wrong type or wrong number of arguments.
- **Plain English:** Mixing incompatible kinds of values.
- **Example / analogy:** Adding 'apples' to 3.
- **Snippet:**
```python
'age: ' + 30
```
- **Snippet explained:** Fails: cannot concatenate str and int; use f-strings.
- **Related terms:** [Type conversion](python_syntax.md#type-conversion), [Type hint](python_syntax.md#type-hint)

---

### Valid anagram
*From module: 02_dsa_arrays_hashmaps*

- **Technical definition:** Check that two strings contain the same characters with the same counts; sort-compare O(n log n) or count O(n).
- **Plain English:** Same letters, different order.
- **Example / analogy:** 'listen' and 'silent'.
- **Snippet:**
```python
from collections import Counter
Counter('rat') == Counter('tar')   # True
```
- **Snippet explained:** Equal letter counts means anagram.
- **Related terms:** [Frequency counting](python_syntax.md#frequency-counting), [Counter](python_syntax.md#counter), [Grouping by key](python_syntax.md#grouping-by-key)

---

### ValidationError
*From module: 07_prompt_engineering*

- **Technical definition:** The exception pydantic raises when data does not satisfy a model; `.errors()` lists each failing field with location and message.
- **Plain English:** Pydantic's way of saying 'this data is wrong, here is why'.
- **Example / analogy:** A rejected form with red marks on bad fields.
- **Snippet:**
```python
from pydantic import BaseModel, ValidationError
class A(BaseModel):
    x: int
try:
    A(x="no")
except ValidationError as e:
    print(e.errors()[0]["loc"])
```
- **Snippet explained:** Prints `('x',)`, the field that failed.
- **Related terms:** [Pydantic model](python_syntax.md#pydantic-model), [Exception handling](python_syntax.md#exception-handling)

---

### Variable
*From module: 00_python_foundations*

- **Technical definition:** A name bound to an object in memory; assignment with `=` creates or rebinds the name.
- **Plain English:** A label stuck on a value.
- **Example / analogy:** A sticky note saying 'epochs' on the number 3.
- **Snippet:**
```python
epochs = 3
epochs = 5  # same label, new value
```
- **Snippet explained:** The name `epochs` first points to 3, then to 5. No type declaration is needed.
- **Related terms:** [Data type](python_syntax.md#data-type), [Mutable vs immutable](python_syntax.md#mutable-vs-immutable)

---

### Virtual environment
*From module: 01_debugging_and_environment*

- **Technical definition:** An isolated directory with its own Python interpreter and installed packages, created with `python -m venv .venv`.
- **Plain English:** A private toolbox per project.
- **Example / analogy:** Separate kitchens so recipes don't share ingredients.
- **Snippet:**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install requests
```
- **Snippet explained:** Creates, activates and installs packages only into that project.
- **Related terms:** [pip](python_syntax.md#pip), [Dependency conflict](python_syntax.md#dependency-conflict)

---

### Working directory
*From module: 01_debugging_and_environment*

- **Technical definition:** The folder a process starts in; relative paths are resolved against it (`os.getcwd()`).
- **Plain English:** The folder you launched the command from.
- **Example / analogy:** Where you are standing in the building when you ask for 'the red folder'.
- **Snippet:**
```python
import os
print(os.getcwd())
```
- **Snippet explained:** Shows the cwd; it differs from the script's folder if you run from elsewhere.
- **Related terms:** [pathlib.Path](python_syntax.md#pathlibpath), [FileNotFoundError](python_syntax.md#filenotfounderror)

---

### zip
*From module: 00_python_foundations*

- **Technical definition:** Built-in that combines several iterables element by element into tuples, stopping at the shortest.
- **Plain English:** Walking two lists side by side.
- **Example / analogy:** A zipper pairing teeth.
- **Snippet:**
```python
list(zip('ab', [1, 2]))
```
- **Snippet explained:** Gives `[('a', 1), ('b', 2)]`.
- **Related terms:** [enumerate](python_syntax.md#enumerate), [Dictionary](python_syntax.md#dictionary)

---
