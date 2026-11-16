"""
GOAL:  The vocabulary of low-level design, shown in 60 lines of running code: Protocol vs ABC, frozen+slots
       dataclasses, enums, and dependency injection (the thing that makes LLM code testable offline).

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/01_lld_foundations/main.py
EXPECTED OUTPUT (shortened):
    Shouter is a Generator? True (no inheritance)
    ABC without the method -> TypeError: Can't instantiate abstract class ...
    frozen -> FrozenInstanceError | slots typo -> TypeError | mutable -> silently changed
    DI: real pipeline answer vs test answer, same Summarizer class
NEEDS: only the standard library + the ragkit package next to this folder (no models, no network).
"""
import sys
from abc import ABC, abstractmethod
from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # make `import ragkit` work from any cwd
from ragkit.protocols import Generator                              # a typing.Protocol: "anything with generate(system, user)"


# ---- 1. Protocol (structural) vs ABC (nominal) -----------------------------------------------------------
class Shouter:                                                      # does NOT inherit from Generator
    def generate(self, system: str, user: str) -> str:
        return user.upper()


print("Shouter is a Generator?", isinstance(Shouter(), Generator), "(no inheritance)")


class BaseGenerator(ABC):                                           # nominal: you MUST inherit and implement
    @abstractmethod
    def generate(self, system: str, user: str) -> str: ...


class Lazy(BaseGenerator):                                          # forgot to implement generate()
    pass


try:
    Lazy()
except TypeError as e:
    print("ABC without the method -> TypeError:", str(e)[:60], "...")


# ---- 2. dataclasses: mutable vs frozen vs slots ----------------------------------------------------------
@dataclass
class MutableChunk:
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str


m, c = MutableChunk("a"), Chunk("a")
m.text = "changed"                                                  # allowed: shared mutable state is a bug factory
print("mutable ->", m.text)
try:
    c.text = "changed"
except FrozenInstanceError as e:
    print("frozen  -> FrozenInstanceError:", e)
try:
    c.txt = "typo"
except (AttributeError, TypeError) as e:
    print("slots typo ->", type(e).__name__)
print("hashable value object:", {c: 1}[Chunk("a")], "| equal by value:", c == Chunk("a"))


# ---- 3. Enum: a closed set of values instead of magic strings --------------------------------------------
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


print("Role('user') ->", Role("user"), "| Role.USER == 'user' ->", Role.USER == "user")
try:
    Role("admin")
except ValueError as e:
    print("bad role rejected:", e)


# ---- 4. Dependency injection: the class receives its collaborator, it does not build it ---------------------
class Summarizer:
    def __init__(self, generator: Generator):                       # depends on the abstraction, not on a vendor
        self.generator = generator

    def summarize(self, text: str) -> str:
        return self.generator.generate("Summarize in one line.", text)


class FakeLLM:                                                      # a 3-line test double: instant, free, deterministic
    def generate(self, system: str, user: str) -> str:
        return f"summary({len(user.split())} words)"


print("DI with a fake :", Summarizer(FakeLLM()).summarize("one two three four"))
print("DI with another:", Summarizer(Shouter()).summarize("one two three four"))

# TRY IT: add a method `def stream(self)` to the Generator Protocol in ragkit/protocols.py: is Shouter still a Generator?
# TRY IT: change `Chunk` to `@dataclass(frozen=True)` (no slots) and rerun: which line stops raising?
# BREAK IT: call Summarizer("not a generator").summarize("x") and read the AttributeError: Python checked nothing in advance.
