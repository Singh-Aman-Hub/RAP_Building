# Terms — Modules, Decorators and Generators

Folder: `00_python_foundations/07_modules_decorators_generators` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [00 Python foundations README](../README.md)

**Terms in this folder:** [import](#import) · [from ... import](#from--import) · [import ... as (alias)](#import--as-alias) · [Module search path and shadowing](#module-search-path-and-shadowing) · [math.sqrt](#mathsqrt) · [collections.Counter](#collectionscounter) · [collections.defaultdict](#collectionsdefaultdict) · [Decorator](#decorator) · [Wrapper function and closure](#wrapper-function-and-closure) · [time.perf_counter](#timeperf_counter) · [func.__name__](#func__name__) · [Generator and yield](#generator-and-yield) · [next()](#next) · [StopIteration](#stopiteration) · [Lazy evaluation](#lazy-evaluation) · [Batching](#batching) · [if __name__ == "__main__"](#if-__name__--__main__) · [AttributeError from a shadowed module](#attributeerror-from-a-shadowed-module)

---

## import  `(syntax)`
**1. Definition:** `import module` loads a module (a `.py` file or package) once and binds its name, so you access its contents as `module.name`.
**2. In plain words:** Python's standard library and third-party packages hold ready-made tools. `import math` gives you `math.sqrt`. Importing keeps names organised under the module's name so they don't clash with yours.
**3. Example to quote:**
> "I wrote `import math` and then `math.sqrt(16)` to get 4.0."
```python
import math
print(math.sqrt(16))
```
**Seen in:** `main.py`, imports section

## from ... import  `(syntax)`
**1. Definition:** `from module import name1, name2` brings specific names directly into the current namespace.
**2. In plain words:** It saves typing the module prefix: `Counter("banana")` instead of `collections.Counter("banana")`. The cost is that it is less obvious where a name came from, so import only what you use.
**3. Example to quote:**
> "I imported `Counter` and `defaultdict` straight from `collections` so I can use them without a prefix."
```python
from collections import Counter, defaultdict
```
**Seen in:** `main.py`, imports section

## import ... as (alias)  `(syntax)`
**1. Definition:** `import module as alias` binds the module to a shorter or different name.
**2. In plain words:** It is common for long names (`import numpy as np`). Here `import time as t` lets the code call `t.perf_counter()`. Pick aliases that others recognise.
**3. Example to quote:**
> "I aliased `time` as `t` so my timer reads `t.perf_counter()`."
```python
import time as t
```
**Seen in:** `main.py`, imports section

## Module search path and shadowing  `(concept)`
**1. Definition:** When you `import X`, Python searches `sys.path` in order (the script's folder first, then installed packages, then the standard library) and uses the first match.
**2. In plain words:** If you create a file named `math.py` next to your script, `import math` loads your file instead of the real one. This "shadowing" causes bewildering errors. Never name your files after standard-library or installed modules.
**3. Example to quote:**
> "My script broke because I had a `random.py` in the folder; it shadowed the standard library, so I renamed it."
```python
import sys
print(sys.path[0])   # script folder is first
```
**Seen in:** `main.py`, comment under imports; BREAK IT

## math.sqrt  `(standard library)`
**1. Definition:** `math.sqrt(x)` returns the square root of `x` as a float.
**2. In plain words:** It's the standard way to take roots, for example for standard deviation or vector length. Negative inputs raise `ValueError`.
**3. Example to quote:**
> "`math.sqrt(16)` printed `4.0`; it always returns a float."
```python
print(math.sqrt(9))   # 3.0
```
**Seen in:** `main.py`, first print

## collections.Counter  `(standard library)`
**1. Definition:** `collections.Counter` is a dict subclass for counting hashable items; `Counter(iterable)` returns a mapping from each item to its number of occurrences.
**2. In plain words:** It does the frequency-counting loop for you: `Counter("banana")` gives `{'a': 3, 'n': 2, 'b': 1}`. Handy for LeetCode problems like Valid Anagram, and for counting tokens.
**3. Example to quote:**
> "`Counter('banana')` counted the letters in one call: 3 a's, 2 n's, 1 b."
```python
print(Counter("banana").most_common(1))   # [('a', 3)]
```
**Seen in:** `main.py`, imports section

## collections.defaultdict  `(standard library)`
**1. Definition:** `defaultdict(factory)` is a dict that, when a missing key is accessed, creates a value by calling `factory()` and stores it.
**2. In plain words:** With `defaultdict(list)` you can do `groups["fruit"].append("apple")` without first checking whether `"fruit"` exists. It's the standard tool for grouping items, as in "group anagrams".
**3. Example to quote:**
> "I used `defaultdict(list)` so `groups['fruit'].append('apple')` worked even though 'fruit' was new."
```python
groups = defaultdict(list)
groups["a"].append(1)
```
**Seen in:** `main.py`, `groups`

## Decorator  `(syntax)`
**1. Definition:** A decorator is a function that takes a function and returns a modified function; `@decorator` above a `def` is shorthand for `f = decorator(f)`.
**2. In plain words:** It lets you add behaviour, like timing or logging, around a function without editing its body. In frameworks you will meet `@app.get("/health")` (FastAPI), `@st.cache_resource` (Streamlit) and `@torch.no_grad()` (PyTorch).
**3. Example to quote:**
> "`@timed` above `slow_sum` replaced it with a wrapper that prints how long the call took."
```python
@timed
def slow_sum(n): return sum(range(n))
```
**Seen in:** `main.py`, decorators section

## Wrapper function and closure  `(concept)`
**1. Definition:** A wrapper is an inner function defined inside a decorator that calls the original function; it keeps access to the outer variables (a closure).
**2. In plain words:** `timed` defines `wrapper(*args, **kwargs)` so it can accept any arguments and pass them on. The wrapper remembers `func`, the function it wraps. The trade-off: stack traces now pass through `wrapper`, which can make debugging a little harder.
**3. Example to quote:**
> "`wrapper(*args, **kwargs)` forwards whatever arguments it gets to `func`, so `timed` works on any function."
```python
def timed(func):
    def wrapper(*a, **k): return func(*a, **k)
    return wrapper
```
**Seen in:** `main.py`, `timed`

## time.perf_counter  `(standard library)`
**1. Definition:** `time.perf_counter()` returns a high-resolution timer value in fractional seconds, suitable for measuring short durations.
**2. In plain words:** Call it before and after some code and subtract to get elapsed time. Multiply by 1000 for milliseconds. It is the right tool for benchmarking rather than `time.time()`.
**3. Example to quote:**
> "`slow_sum took 13.45 ms` came from subtracting two `perf_counter()` readings."
```python
start = t.perf_counter()
elapsed = t.perf_counter() - start
```
**Seen in:** `main.py`, `timed`

## func.__name__  `(Python built-in)`
**1. Definition:** Every function object has a `__name__` attribute holding its name as a string.
**2. In plain words:** The decorator uses it to print which function was timed, such as `slow_sum`. It's handy for logging.
**3. Example to quote:**
> "I print `func.__name__` in the timer so the message says 'slow_sum took ...'."
```python
print(len.__name__)   # len
```
**Seen in:** `main.py`, `timed`

## Generator and yield  `(syntax)`
**1. Definition:** A generator function contains `yield`; calling it returns an iterator that produces one value at a time, pausing at each `yield` and resuming where it left off.
**2. In plain words:** Instead of building a full list in memory, `batches` hands back one chunk, waits to be asked again, then gives the next. This is how data loaders feed a model one batch at a time. Trade-off: a generator can be used only once.
**3. Example to quote:**
> "`batches(list(range(7)), 3)` yielded `[0, 1, 2]`, `[3, 4, 5]`, then `[6]`, one chunk at a time."
```python
def count():
    yield 1
    yield 2
```
**Seen in:** `main.py`, `batches`

## next()  `(Python built-in)`
**1. Definition:** `next(iterator)` retrieves the next item from an iterator or generator, raising `StopIteration` when exhausted.
**2. In plain words:** It is the manual way to pull one value, which a `for` loop does automatically. `print(next(gen), next(gen))` gave the first two chunks.
**3. Example to quote:**
> "`next(gen)` twice returned `[1, 2]` and `[3, 4]`; a third call would raise `StopIteration`."
```python
g = iter([1, 2]); print(next(g), next(g))
```
**Seen in:** `main.py`, generator section

## StopIteration  `(Python built-in)`
**1. Definition:** `StopIteration` is the exception raised by `next()` when an iterator has no more items.
**2. In plain words:** It signals "exhausted". `for` loops catch it internally to know when to stop, so you normally never see it unless you call `next()` yourself.
**3. Example to quote:**
> "After two `next(gen)` calls on a two-chunk generator, a third call raises `StopIteration`."
```python
# next(iter([]))  -> StopIteration
```
**Seen in:** `main.py`, comment after `next(gen)`

## Lazy evaluation  `(concept)`
**1. Definition:** Lazy evaluation means values are computed only when requested instead of all at once.
**2. In plain words:** A generator function's body doesn't run until you iterate. That's why decorating `batches` with `@timed` reports about 0 ms: no work happened yet. Laziness saves memory and time for large data.
**3. Example to quote:**
> "Calling `batches(...)` did nothing until the `for` loop asked for chunks, which is why timing it showed about 0 ms."
```python
gen = batches([1, 2, 3], 2)   # nothing computed yet
```
**Seen in:** `main.py` TRY IT comment; `WALKTHROUGH.md` experiment 1

## Batching  `(concept)`
**1. Definition:** Batching means splitting a sequence into fixed-size chunks (the last one possibly smaller) to process them together.
**2. In plain words:** Models process data in batches for speed and memory control. `range(0, len(items), size)` produces the start positions and `items[i:i + size]` slices each chunk.
**3. Example to quote:**
> "With 7 items and size 3, my generator produced batches of 3, 3 and 1, just like a data loader."
```python
items = list(range(7))
print(items[0:3], items[6:9])   # [0, 1, 2] [6]
```
**Seen in:** `main.py`, `batches`

## if __name__ == "__main__"  `(syntax)`
**1. Definition:** `__name__` is `"__main__"` when a file is run directly and the module's own name when it is imported, so this check runs code only on direct execution.
**2. In plain words:** It lets one file be both a reusable module and a runnable script. Another file can `import main` without triggering the script part. Many test helpers and servers rely on this.
**3. Example to quote:**
> "Running `python main.py` printed 'running as a script, __name__ = __main__', but importing the file wouldn't call `main()`."
```python
if __name__ == "__main__":
    main()
```
**Seen in:** `main.py`, end of file

## AttributeError from a shadowed module  `(concept)`
**1. Definition:** If a local file shadows a library, accessing a missing attribute gives `AttributeError: module 'math' has no attribute 'sqrt'` because Python imported your file instead.
**2. In plain words:** The message names the module and the missing attribute, which is the clue that you imported the wrong thing. Check `math.__file__` to see which file was loaded, then rename or delete your file and its `__pycache__`.
**3. Example to quote:**
> "I got `module 'math' has no attribute 'sqrt'` because I had created `math.py`; printing `math.__file__` pointed at my own file."
```python
import math
print(math.__file__)
```
**Seen in:** `main.py` BREAK IT; `WALKTHROUGH.md` debugging practice
