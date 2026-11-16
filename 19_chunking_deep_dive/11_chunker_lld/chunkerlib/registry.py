"""registry.py -- the Chunker Protocol and a name -> class registry (strategy pattern)."""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from .model import Chunk, ChunkConfig, TokenCounter


@runtime_checkable
class Chunker(Protocol):
    """Anything with these members is a chunker (structural typing: no inheritance needed)."""
    chunker_id: str
    version: str

    def chunk(self, doc_id: str, text: str, base_offset: int = 0) -> list[Chunk]: ...


_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Class decorator: @register("recursive") makes the class constructible by name from a config file."""
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"chunker {name!r} already registered")
        cls.chunker_id = name
        _REGISTRY[name] = cls
        return cls
    return deco


def create(name: str, config: ChunkConfig, count_tokens: TokenCounter) -> Chunker:
    if name not in _REGISTRY:
        raise KeyError(f"unknown chunker {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](config, count_tokens)


def available() -> list[str]:
    return sorted(_REGISTRY)
