# Walkthrough 05 — Classes and dataclasses

## Goal
Read `model = SomeClass.from_pretrained(...)`, `model.generate(...)` and `model.config.x` without feeling lost.

## Run it
```bash
cd 00_python_foundations/05_classes_and_dataclasses
python main.py
```

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
