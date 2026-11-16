"""Solutions for step 01 exercises 1.1 (modify), 1.2 (break), 1.3 (rebuild). Run: python sol_01.py"""
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.protocols import Generator

# 1.1 Add `stream` to a Generator-like Protocol: is Shouter still one?
@runtime_checkable
class StreamingGenerator(Protocol):
    def generate(self, system: str, user: str) -> str: ...
    def stream(self, system: str, user: str): ...

class Shouter:
    def generate(self, system, user):
        return user.upper()

class StreamingShouter(Shouter):
    def stream(self, system, user):
        yield from user.upper().split()

print("1.1 Shouter is a Generator:", isinstance(Shouter(), Generator),
      "| Shouter is a StreamingGenerator:", isinstance(Shouter(), StreamingGenerator),
      "| StreamingShouter is:", isinstance(StreamingShouter(), StreamingGenerator))

# 1.2 Break it: an object that is not a generator, caught EARLY with an explicit check in the constructor
class Summarizer:
    def __init__(self, generator):
        if not isinstance(generator, Generator):
            raise TypeError(f"generator must implement generate(system, user); got {type(generator).__name__}")
        self.generator = generator

try:
    Summarizer("not a generator")
except TypeError as e:
    print("1.2 early error:", e)

# 1.3 Rebuild: an Embedder Protocol, a fake, and an ABC version of the same contract
@runtime_checkable
class Embedder(Protocol):
    dim: int
    def embed(self, texts): ...

class FakeEmbedder:                                   # satisfies the Protocol structurally
    dim = 3
    def embed(self, texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]

class EmbedderABC(ABC):
    @abstractmethod
    def embed(self, texts): ...

class Incomplete(EmbedderABC):
    pass

print("1.3 FakeEmbedder is an Embedder:", isinstance(FakeEmbedder(), Embedder))
try:
    Incomplete()
except TypeError as e:
    print("    ABC version refuses early:", str(e)[:70])
