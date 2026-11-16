# Terms — LLD foundations: classes, contracts and the vocabulary of object design

Folder: `20_llm_system_lld/01_lld_foundations` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Low-level design (LLD, object design, class design)](#low-level-design-lld-object-design-class-design--concept) · [SOLID](#solid--concept) · [Protocol (structural typing, duck typing with types)](#protocol-structural-typing-duck-typing-with-types--syntax) · [Abstract base class (ABC, nominal typing)](#abstract-base-class-abc-nominal-typing--syntax) · [Dataclass (frozen, slots)](#dataclass-frozen-slots--syntax) · [Value object](#value-object--concept) · [Enum (closed set)](#enum-closed-set--syntax) · [Dependency injection (DI)](#dependency-injection-di--concept) · [Test double (fake, stub, mock)](#test-double-fake-stub-mock--concept) · [Class diagram and sequence diagram](#class-diagram-and-sequence-diagram--concept) · [AttributeError: 'str' object has no attribute 'generate'](#attributeerror-str-object-has-no-attribute-generate--error)

---

## Low-level design (LLD, object design, class design)  `(concept)`
**1. Definition:** The design of the classes, interfaces, methods and collaborations inside one service, as opposed to the architecture between services (HLD).
**2. In plain words:** The 'how do the objects fit together' round: you produce a class diagram, a sequence diagram, edge cases and tests.
**3. Example to quote:** > "LLD is where I decide which things are interfaces and which are values, so the model and the vector DB are swappable."
**Seen in:** `WALKTHROUGH.md`, theory

## SOLID  `(concept)`
**1. Definition:** Five object-design principles: single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion.
**2. In plain words:** Rules of thumb that keep code easy to change and test. In ragkit: one job per class, new plugins by registration, any store passes the same contract test, small Protocols, constructor injection.
**3. Example to quote:** > "I use SOLID as a checklist, not a religion: each letter has a concrete LLM-app symptom it prevents."
**Seen in:** `WALKTHROUGH.md`, theory

## Protocol (structural typing, duck typing with types)  `(syntax)`
**1. Definition:** A `typing.Protocol` class declares methods; any object having them is accepted as that type without inheriting.
**2. In plain words:** A plug-shape: if it fits, it works. `@runtime_checkable` lets `isinstance` test it, but only by attribute names.
**3. Example to quote:** > "Protocols let a vendor SDK wrapper and a test fake satisfy the same contract with zero coupling."
```python
class Shouter:
    def generate(self, system, user): return user.upper()
isinstance(Shouter(), Generator)  # True
```
**Seen in:** `main.py`, block 1

## Abstract base class (ABC, nominal typing)  `(syntax)`
**1. Definition:** A class deriving from `abc.ABC` with `@abstractmethod` members; subclasses must implement them or cannot be instantiated.
**2. In plain words:** A formal membership: you must inherit and complete the work.
**3. Example to quote:** > "I use an ABC when I also ship shared behaviour, as in the Template Method chunker."
```python
class Lazy(BaseGenerator): pass
Lazy()  # TypeError: Can't instantiate abstract class
```
**Seen in:** `main.py`, block 2

## Dataclass (frozen, slots)  `(syntax)`
**1. Definition:** `@dataclass` generates `__init__`, `__repr__`, `__eq__`; `frozen=True` forbids assignment and adds `__hash__`; `slots=True` stores fields in slots instead of a `__dict__`.
**2. In plain words:** A record class with the boilerplate written for you; frozen+slots makes it safe and small.
**3. Example to quote:** > "Frozen is shallow: a list field can still be mutated, so I use tuples and read-only mappings inside value objects."
```python
@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
```
**Seen in:** `main.py`, block 5

## Value object  `(concept)`
**1. Definition:** An object defined only by its field values, equal when the fields are equal, ideally immutable.
**2. In plain words:** Like a banknote: any 10-rupee note is as good as another.
**3. Example to quote:** > "A chunk is a value object; a conversation session is an entity with identity and changing state."
**Seen in:** `WALKTHROUGH.md`, theory

## Enum (closed set)  `(syntax)`
**1. Definition:** `enum.Enum` defines a fixed set of named members; `class Role(str, Enum)` also behaves as a string.
**2. In plain words:** A multiple-choice answer instead of free text.
**3. Example to quote:** > "Enums reject bad values at the edge with a ValueError instead of letting a typo travel."
```python
Role('admin')  # ValueError: 'admin' is not a valid Role
```
**Seen in:** `main.py`, block 7

## Dependency injection (DI)  `(concept)`
**1. Definition:** Passing a class the objects it depends on (usually via its constructor) instead of letting it create them.
**2. In plain words:** A chef is handed ingredients rather than growing them.
**3. Example to quote:** > "Constructor arguments are enough; the composition root is the one place that creates real objects."
```python
class Summarizer:
    def __init__(self, generator): self.generator = generator
```
**Seen in:** `main.py`, block 9

## Test double (fake, stub, mock)  `(concept)`
**1. Definition:** An object that stands in for a real dependency in a test. A stub returns canned data, a fake is a simplified working version, a mock records calls.
**2. In plain words:** A crash-test dummy.
**3. Example to quote:** > "`FakeLLM` is 3 lines and makes the whole suite instant, free and deterministic."
**Seen in:** `main.py`, FakeLLM

## Class diagram and sequence diagram  `(concept)`
**1. Definition:** A class diagram shows types and their relationships (implements, inherits, composes, uses); a sequence diagram shows calls between objects over time.
**2. In plain words:** A floor plan versus a film storyboard.
**3. Example to quote:** > "I draw the class diagram first (structure), then one sequence diagram for the main flow and one for the failure path."
```python
mermaid:
classDiagram
  Generator <|.. FakeLLM
```
**Seen in:** `WALKTHROUGH.md`, theory

## AttributeError: 'str' object has no attribute 'generate'  `(error)`
**1. Definition:** Raised when code calls a method on an object that does not have it; here a string was passed where a Generator was expected.
**2. In plain words:** You handed the manager a sandwich instead of a worker.
**3. Example to quote:** > "Type hints do not stop this at runtime; the error appears at the first call, far from the cause."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
