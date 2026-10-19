# Terms — Classes and Dataclasses

Folder: `00_python_foundations/05_classes_and_dataclasses` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [00 Python foundations README](../README.md)

**Terms in this folder:** [dataclasses (module)](#dataclasses-module) · [Class](#class) · [Instance (object)](#instance-object) · [__init__ (constructor)](#__init__-constructor) · [self](#self) · [Attribute](#attribute) · [Method](#method) · [Method chaining](#method-chaining) · [__repr__](#__repr__) · [Inheritance](#inheritance) · [super()](#super) · [@classmethod and cls](#classmethod-and-cls) · [from_pretrained (pattern)](#from_pretrained-pattern) · [Decorator](#decorator) · [__call__](#__call__) · [Dunder (magic) methods](#dunder-magic-methods) · [@dataclass](#dataclass) · [dataclass field](#dataclass-field) · [field(default_factory=...)](#fielddefault_factory) · [Dataclass equality (__eq__)](#dataclass-equality-__eq__) · [UnboundLocalError](#unboundlocalerror)

---

## dataclasses (module)  `(standard library)`
**1. Definition:** `dataclasses` is a standard-library module providing the `@dataclass` decorator and the `field()` helper to generate boilerplate methods for data-holding classes.
**2. In plain words:** If a class mostly stores values, writing `__init__`, `__repr__` and `__eq__` by hand is tedious. This module writes them for you from a list of annotated fields. Nothing to install; it ships with Python.
**3. Example to quote:**
> "I imported `dataclass` and `field` from `dataclasses` so my `TrainConfig` needed no hand-written constructor."
```python
from dataclasses import dataclass, field
```
**Seen in:** `main.py`, first import

## Class  `(syntax)`
**1. Definition:** A class, defined with `class Name:`, is a blueprint that bundles data (attributes) and behaviour (methods) for the objects created from it.
**2. In plain words:** A class is like a cookie cutter; each cookie is an object. `Counter` describes what a counter has (a name and a value) and what it can do (increment). Every ML library is built from classes: models, tokenizers and configs are all classes.
**3. Example to quote:**
> "`Counter` is a class; it defines what every counter has and can do, and `Counter('clicks')` builds one concrete counter."
```python
class Counter:
    pass
```
**Seen in:** `main.py`, `Counter`

## Instance (object)  `(concept)`
**1. Definition:** An instance is a concrete object created by calling a class, e.g. `c = Counter("clicks")`.
**2. In plain words:** The class is the plan, the instance is the thing built from it. Two instances of the same class hold their own separate data: `Counter("a")` and `Counter("b")` have independent values.
**3. Example to quote:**
> "`c = Counter('clicks')` creates one instance whose `name` is 'clicks' and whose `value` starts at 0."
```python
c = Counter("clicks")
```
**Seen in:** `main.py`, `c = Counter("clicks")`

## __init__ (constructor)  `(syntax)`
**1. Definition:** `__init__(self, ...)` is the special method Python calls automatically right after creating an instance, used to set up its initial attributes.
**2. In plain words:** When you write `Counter("clicks")`, Python makes an empty object and passes it to `__init__` along with "clicks". Inside, you store things on `self`. It's where "starting state" is defined.
**3. Example to quote:**
> "`__init__` stores `self.name = name` and sets `self.value = 0` so every counter starts from zero."
```python
def __init__(self, name):
    self.name = name
    self.value = 0
```
**Seen in:** `main.py`, `Counter.__init__`, `TinyModel.__init__`

## self  `(syntax)`
**1. Definition:** `self` is the conventional name of a method's first parameter, which refers to the specific instance the method is being called on.
**2. In plain words:** When you call `c.increment()`, Python passes `c` in as `self`, so the method knows whose `value` to change. Forgetting `self.` in front of an attribute name creates a local variable instead and often causes an error.
**3. Example to quote:**
> "Inside `increment`, `self.value += by` changes the value stored on that particular counter."
```python
self.value += by
```
**Seen in:** `main.py`, all methods

## Attribute  `(concept)`
**1. Definition:** An attribute is a named piece of data stored on an object and accessed with dot notation, e.g. `c.value`.
**2. In plain words:** Attributes are the object's data. In `model.config.vocab_size`, `config` is an attribute that holds another object, which in turn has an attribute `vocab_size`. Dots simply walk through these layers.
**3. Example to quote:**
> "`print(c.value, c.name)` read two attributes with dot notation and printed `3 clicks`."
```python
print(c.value, c.name)
```
**Seen in:** `main.py`, after `c.increment()`

## Method  `(concept)`
**1. Definition:** A method is a function defined inside a class that operates on an instance, called as `obj.method()`.
**2. In plain words:** Methods are what the object can do. `c.increment()` is a method call; `model.generate(...)` in ML libraries is the same idea. The only difference from a plain function is the automatic `self`.
**3. Example to quote:**
> "`increment` is a method of `Counter`, so I call it as `c.increment()` rather than `increment(c)`."
```python
def increment(self, by=1):
    self.value += by
```
**Seen in:** `main.py`, `Counter.increment`

## Method chaining  `(concept)`
**1. Definition:** Method chaining is calling several methods in a row on the same object, which works when each method returns `self`.
**2. In plain words:** Because `increment` ends with `return self`, the result is the same counter again, so you can immediately call `.increment()` on it. `c.increment().increment().increment()` raises the value to 3 in one line.
**3. Example to quote:**
> "I chained three `increment()` calls because each one returns `self`; the final print showed `Counter 'clicks' = 3`."
```python
c.increment().increment().increment()
```
**Seen in:** `main.py`, `c.increment().increment().increment()`

## __repr__  `(syntax)`
**1. Definition:** `__repr__(self)` is the special method that returns the developer-friendly string shown by `repr()` and used by `print()` when no `__str__` exists.
**2. In plain words:** Without it, printing an object shows something ugly like `<__main__.Counter object at 0x...>`. Defining `__repr__` makes print and debugging output readable, e.g. `Counter 'clicks' = 3`.
**3. Example to quote:**
> "I defined `__repr__` so `print(c)` shows the name and value instead of a memory address."
```python
def __repr__(self):
    return f"Counter {self.name!r} = {self.value}"
```
**Seen in:** `main.py`, `Counter.__repr__`

## Inheritance  `(concept)`
**1. Definition:** Inheritance lets a class (`class Child(Parent):`) reuse and extend the attributes and methods of another class.
**2. In plain words:** `LoudCounter` is a `Counter` that also prints a message. It gets everything from `Counter` for free and only overrides what differs. Use it carefully: deep inheritance trees become hard to follow.
**3. Example to quote:**
> "`LoudCounter(Counter)` inherits all counter behaviour and overrides `increment` just to add a print."
```python
class LoudCounter(Counter):
    pass
```
**Seen in:** `main.py`, `LoudCounter`

## super()  `(Python built-in)`
**1. Definition:** `super()` returns a proxy that lets a subclass call the parent class's version of a method.
**2. In plain words:** When you override a method but still want the original behaviour, call `super().increment(by)`. That way `LoudCounter` prints first and then does the normal counting instead of duplicating code.
**3. Example to quote:**
> "In `LoudCounter.increment` I print 'incrementing!' and then call `super().increment(by)` to reuse the parent's logic."
```python
return super().increment(by)
```
**Seen in:** `main.py`, `LoudCounter.increment`

## @classmethod and cls  `(syntax)`
**1. Definition:** A method decorated with `@classmethod` receives the class itself as its first argument (conventionally `cls`) instead of an instance.
**2. In plain words:** It belongs to the class, not to a particular object, so you can call it before any instance exists. It's often used as an alternative constructor. Inside, `cls(...)` is the same as calling the class by name.
**3. Example to quote:**
> "`TinyModel.from_pretrained('gpt2')` is a class method; inside it, `cls(vocab_size=50257)` builds and returns an instance."
```python
@classmethod
def from_pretrained(cls, name):
    return cls(vocab_size=50257)
```
**Seen in:** `main.py`, `TinyModel.from_pretrained`

## from_pretrained (pattern)  `(concept)`
**1. Definition:** `from_pretrained(name)` is the common Hugging Face convention: a class method that downloads or loads saved weights/config and returns a ready-to-use instance.
**2. In plain words:** In real code, `AutoModelForCausalLM.from_pretrained("gpt2")` fetches a model. The `TinyModel` in this file imitates it with a print statement, so the syntax stops feeling magical: it's just a class method returning an object.
**3. Example to quote:**
> "I recognise `from_pretrained` as a class method that returns an instance, exactly like my `TinyModel` imitation."
```python
model = TinyModel.from_pretrained("gpt2")
```
**Seen in:** `main.py`, comment block and `TinyModel`

## Decorator  `(syntax)`
**1. Definition:** A decorator is a callable applied above a function or class with `@name` that wraps or modifies it.
**2. In plain words:** `@classmethod` and `@dataclass` change how the thing below them behaves without you rewriting it. Think of it as a modifier sticker. Step 07 shows how to write your own.
**3. Example to quote:**
> "`@dataclass` above my class asks Python to generate `__init__`, `__repr__` and `__eq__` for me."
```python
@dataclass
class A:
    x: int = 0
```
**Seen in:** `main.py`, `@classmethod`, `@dataclass`

## __call__  `(syntax)`
**1. Definition:** Defining `__call__(self, ...)` makes an instance callable like a function: `obj(args)` runs `obj.__call__(args)`.
**2. In plain words:** That's why you can write `model("hi")` or `model(input_ids)` in PyTorch: the model object is callable. It's a method hook, not magic.
**3. Example to quote:**
> "`model('hi')` returned `output for 'hi'` because my `TinyModel` defines `__call__`."
```python
def __call__(self, text):
    return f"output for {text!r}"
```
**Seen in:** `main.py`, `TinyModel.__call__`

## Dunder (magic) methods  `(concept)`
**1. Definition:** Dunder ("double underscore") methods such as `__init__`, `__repr__`, `__call__` and `__eq__` are special hooks Python calls implicitly in response to syntax or built-in functions.
**2. In plain words:** They let your class plug into Python itself: `print(obj)` triggers `__repr__`, `obj(x)` triggers `__call__`, `a == b` triggers `__eq__`. You rarely call them directly.
**3. Example to quote:**
> "When I wrote `print(c)`, Python used my class's `__repr__` behind the scenes."
```python
# print(c) -> Counter.__repr__(c)
```
**Seen in:** `main.py`, `__init__`, `__repr__`, `__call__`

## @dataclass  `(syntax)`
**1. Definition:** `@dataclass` is a class decorator that auto-generates `__init__`, `__repr__`, `__eq__` (and more) from the class's annotated fields.
**2. In plain words:** You list the fields with types and defaults, and Python writes the boilerplate. It's perfect for configs like learning rate and epochs. `TrainConfig(epochs=5)` overrides one value and keeps the other defaults.
**3. Example to quote:**
> "I used a dataclass for `TrainConfig`, and `print(cfg)` gave `TrainConfig(learning_rate=0.0001, epochs=5, batch_size=8, tags=[])` for free."
```python
@dataclass
class TrainConfig:
    epochs: int = 3
```
**Seen in:** `main.py`, `TrainConfig`

## dataclass field  `(syntax)`
**1. Definition:** Each annotated class variable in a dataclass, like `epochs: int = 3`, is a field with a type hint and an optional default value.
**2. In plain words:** Fields describe what the data object holds. Fields with defaults are optional when creating the object. The type hints are documentation and are not enforced at runtime.
**3. Example to quote:**
> "`learning_rate: float = 1e-4` declares a field with a type and default, so I can omit it when creating the config."
```python
learning_rate: float = 1e-4
```
**Seen in:** `main.py`, `TrainConfig`

## field(default_factory=...)  `(Python built-in)`
**1. Definition:** `dataclasses.field(default_factory=callable)` sets a field's default by calling `callable()` for each new instance, which is required for mutable defaults like lists.
**2. In plain words:** Writing `tags: list = []` would share one list among all instances, the same trap as mutable default arguments (step 03). `default_factory=list` creates a fresh empty list each time.
**3. Example to quote:**
> "I wrote `tags: list = field(default_factory=list)` so each config gets its own list instead of sharing one."
```python
tags: list = field(default_factory=list)
```
**Seen in:** `main.py`, `TrainConfig.tags`

## Dataclass equality (__eq__)  `(concept)`
**1. Definition:** A dataclass auto-generates `__eq__`, so two instances are equal when all their fields are equal.
**2. In plain words:** For ordinary classes, two objects with the same data are not equal by default. For dataclasses, `cfg == TrainConfig(epochs=5)` is `True`, which is handy for tests and comparing configs.
**3. Example to quote:**
> "`cfg == TrainConfig(epochs=5)` returned `True` because all fields matched, without me writing `__eq__`."
```python
print(TrainConfig() == TrainConfig())   # True
```
**Seen in:** `main.py`, last `print`

## UnboundLocalError  `(Python built-in)`
**1. Definition:** `UnboundLocalError` is raised when a local variable is read (or augmented) before being assigned within a function.
**2. In plain words:** Writing `value += by` without `self.` makes Python think `value` is a local variable that was never defined. The traceback points to that line. The fix is `self.value += by`.
**3. Example to quote:**
> "I removed `self.` and got `UnboundLocalError: cannot access local variable 'value'`, which told me attributes need `self.`."
```python
self.value += by   # correct
```
**Seen in:** `main.py` BREAK IT; `WALKTHROUGH.md` debugging practice
