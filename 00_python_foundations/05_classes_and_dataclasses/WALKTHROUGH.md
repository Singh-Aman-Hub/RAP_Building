# Walkthrough 05 — Classes and dataclasses

## Goal
Read `model = SomeClass.from_pretrained(...)`, `model.generate(...)` and `model.config.x` without feeling lost.

## Run it
```bash
cd 00_python_foundations/05_classes_and_dataclasses
python main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| class vs instance (object) | blueprint vs thing built from it | No | One class, many instances, each with its own attribute values. |
| method vs function | member function | Nearly | A method is a function defined in a class whose first parameter is `self`. |
| attribute vs property vs field | instance variable, member, data field | Rough | Attribute is any `obj.x`; field is a dataclass-declared attribute. |
| `@classmethod` vs `@staticmethod` | alternate constructor vs plain function in a class | No | classmethod gets `cls`; staticmethod gets nothing automatically. |
| `__repr__` vs `__str__` | developer vs user text | No | `print` uses `__str__`, falling back to `__repr__`. |
| dataclass vs Pydantic model | config holders | No | Dataclass does no validation; Pydantic validates and converts types. |

## Code, step by step

### 1. File header (docstring)  (`main.py` lines 1–11)

```python
"""
GOAL:  Understand classes (objects with data + behaviour) because every ML library is built from them.

RUN:   cd 00_python_foundations/05_classes_and_dataclasses && python main.py

EXPECTED OUTPUT (first lines):
    Counter 'clicks' = 3
    ...

NEEDS: nothing (standard library only), no internet.
"""
```

- **In simple words:** The file's label.
- **Exactly:** Module docstring; no runtime effect.
- **Why it is here:** Orients the reader.

### 2. Import dataclass and field  (`main.py` lines 13–13)

```python
from dataclasses import dataclass, field  # dataclass: auto-writes __init__/__repr__ for data holders
```

- **In simple words:** Bring in two helpers that let Python write boilerplate class code for us.
- **Exactly:** `dataclass` is a decorator that generates `__init__`, `__repr__`, `__eq__`; `field` customises one attribute (used later for `default_factory`).
- **Why it is here:** Without it you hand-write all three methods for every data-holder class.

### 3. The Counter class  (`main.py` lines 16–29)

```python
# ---- a plain class -------------------------------------------------------
class Counter:
    """class = blueprint.  An *instance* (object) is one thing built from it."""

    def __init__(self, name):          # __init__ runs when you call Counter("clicks")
        self.name = name               # `self` = this particular object; self.x stores data on it
        self.value = 0

    def increment(self, by=1):         # a method = function defined inside a class
        self.value += by
        return self                    # returning self allows chaining (shown below)

    def __repr__(self):                # controls how print() shows the object
        return f"Counter {self.name!r} = {self.value}"
```

- **In simple words:** A class is a blueprint; each object built from it carries its own data (`name`, `value`) and abilities (methods).
- **Exactly:** `__init__(self, name)` runs on construction and stores `self.name`, `self.value = 0`. `increment(self, by=1)` adds to `self.value` and returns `self` so calls can be chained. `__repr__` returns the text `print()` shows, e.g. `Counter 'clicks' = 3`. `self` is the specific instance, passed automatically.
- **Why it is here:** This is the basic unit of every ML library (`Tokenizer`, `Model`, `Trainer` are classes).

### 4. Create an instance and chain methods  (`main.py` lines 32–35)

```python
c = Counter("clicks")                  # create an instance (calls __init__)
c.increment().increment().increment()  # method chaining
print(c)
print(c.value, c.name)                 # attributes are read with dot notation
```

- **In simple words:** Build one Counter and bump it three times in one line.
- **Exactly:** `Counter("clicks")` calls `__init__`. `c.increment().increment().increment()` works because each call returns `self`; `value` ends at 3. `print(c)` uses `__repr__` -> `Counter 'clicks' = 3`; `print(c.value, c.name)` -> `3 clicks`.
- **Why it is here:** Shows instance creation, attribute access with dots and why returning `self` enables fluent APIs.

### 5. Inheritance with super()  (`main.py` lines 38–45)

```python
# ---- inheritance: a class that extends another --------------------------
class LoudCounter(Counter):
    def increment(self, by=1):
        print("incrementing!")
        return super().increment(by)   # super() calls the parent's version


LoudCounter("loud").increment(5)
```

- **In simple words:** A child class reuses the parent and changes only what it needs, like a model variant with one tweak.
- **Exactly:** `LoudCounter(Counter)` inherits `__init__`/`__repr__`. Its `increment` prints `incrementing!` then `super().increment(by)` runs the parent's version. `LoudCounter("loud").increment(5)` prints `incrementing!` (value becomes 5 but the return is not printed).
- **Why it is here:** Inheritance is how PyTorch `nn.Module` subclasses work (`super().__init__()`).

### 6. A tiny imitation of from_pretrained  (`main.py` lines 47–66)

```python
# ---- how ML libraries look from the outside ------------------------------
# model = AutoModelForCausalLM.from_pretrained("gpt2")   <- a CLASSMETHOD that returns an instance
# out = model.generate(...)                              <- a method on that instance
# model.config.vocab_size                                <- an attribute holding another object
# Below is a tiny imitation so the syntax stops looking magical.
class TinyModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    @classmethod                       # decorator: method receives the class (cls), not an instance
    def from_pretrained(cls, name):
        print(f"(pretend we downloaded {name})")
        return cls(vocab_size=50257)   # cls(...) is the same as TinyModel(...)

    def __call__(self, text):          # makes the instance callable: model("hi")
        return f"output for {text!r}"


model = TinyModel.from_pretrained("gpt2")
print(model.vocab_size, model("hi"))
```

- **In simple words:** Shows that `Model.from_pretrained(...)` and `model(...)` are ordinary Python features, not magic.
- **Exactly:** `@classmethod` makes `from_pretrained` receive the class as `cls`; `cls(vocab_size=50257)` builds an instance. `__call__` makes an instance callable like a function. Output: `(pretend we downloaded gpt2)` then `50257 output for 'hi'`.
- **Why it is here:** Demystifies the Hugging Face / PyTorch call style you will read constantly.

### 7. A dataclass for training config  (`main.py` lines 68–79)

```python
# ---- dataclass: classes that mostly just hold values ---------------------
@dataclass
class TrainConfig:
    learning_rate: float = 1e-4        # field: type hint + default value
    epochs: int = 3
    batch_size: int = 8
    tags: list = field(default_factory=list)   # mutable defaults need default_factory


cfg = TrainConfig(epochs=5)            # override one value, keep others
print(cfg)                             # free, readable __repr__
print(cfg.learning_rate, cfg == TrainConfig(epochs=5))  # free __eq__
```

- **In simple words:** A class that mostly stores settings, where Python writes `__init__`, printing and comparison for you.
- **Exactly:** Each annotated line with a default is a field. `tags: list = field(default_factory=list)` creates a fresh list per object (a bare `[]` default raises `ValueError`). `TrainConfig(epochs=5)` overrides one value. `print(cfg)` -> `TrainConfig(learning_rate=0.0001, epochs=5, batch_size=8, tags=[])`; `cfg == TrainConfig(epochs=5)` -> `True` because `__eq__` compares all fields.
- **Why it is here:** Configs are everywhere in ML; dataclasses give readable, comparable, less error-prone config objects.

### 8. TRY IT and BREAK IT comments  (`main.py` lines 81–82)

```python
# TRY IT: add a `weight_decay: float = 0.01` field and print it.
# BREAK IT: remove `self.` in `self.value += by` -> UnboundLocalError. Read which line it names.
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** Dropping `self.` in `self.value += by` raises `UnboundLocalError` since `value` would be treated as a local variable.
- **Why it is here:** Teaches what `self.` does.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class Name:` | Blueprint for objects | `class Counter:` |
| `def __init__(self, x):` | Constructor: runs on `Name(x)` | `self.value = 0` |
| `self` | The specific object a method is working on | `self.name` |
| method | Function inside a class, called `obj.method()` | `c.increment()` |
| `__repr__`, `__call__`, `__eq__` | "Dunder" methods: special hooks | `model("hi")` calls `__call__` |
| `class Child(Parent):` / `super()` | Inheritance / call the parent version | `LoudCounter` |
| `@classmethod` + `cls` | Method on the class itself, often an alternative constructor | `from_pretrained` |
| `@dataclass` | Auto-generates `__init__`, `__repr__`, `__eq__` | `TrainConfig` |
| `field(default_factory=list)` | Safe default for mutable fields | `tags` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does |
|---|---|---|
| `dataclasses.dataclass`, `field` | standard library | Reduce boilerplate for data-holder classes |
| `super()` | built-in | Access parent-class methods |

## Execution trace (real output)
`c.increment().increment().increment()` — each call adds 1 and returns `self`, so calls chain:
```
Counter 'clicks' = 3
3 clicks
```
`TinyModel.from_pretrained("gpt2")` prints `(pretend we downloaded gpt2)`, builds the object via `cls(vocab_size=50257)`; then `model("hi")` runs `__call__`:
```
50257 output for 'hi'
TrainConfig(learning_rate=0.0001, epochs=5, batch_size=8, tags=[])
0.0001 True
```

## Experiments
1. Add `weight_decay: float = 0.01` to `TrainConfig`; print it.
2. Add a `reset()` method to `Counter` that sets `value` to 0.
3. Remove `__repr__` and print the object: you see `<__main__.Counter object at 0x...>`.

## Debugging practice
Change `self.value += by` to `value += by` -> `UnboundLocalError: cannot access local variable 'value'...`. The traceback line points to the method; lesson: attributes need `self.`.

## Rebuild checklist
- [ ] Class with `__init__`, one method and `__repr__`
- [ ] Subclass overriding a method and calling `super()`
- [ ] Dataclass config with 3 fields and a default

## Say it aloud
"A class bundles data and behaviour. `__init__` sets up the object and `self` refers to it. In ML libraries a model is an instance: `from_pretrained` is a class method that builds it, and `generate` is a method that uses its stored weights and config. I use dataclasses for configs because they cut boilerplate."
Trade-off: inheritance couples classes tightly; prefer small classes and composition.
