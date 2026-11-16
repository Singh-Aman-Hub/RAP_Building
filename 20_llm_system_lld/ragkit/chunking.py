"""chunking.py -- Template Method: BaseChunker.chunk() fixes the skeleton, subclasses fill one hook (steps 03, 05, 10)."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Sequence

from .domain import Chunk, Document, Element, ElementKind


class BaseChunker(ABC):
    """Template Method pattern. `chunk` is the fixed algorithm; `_groups` is the hook that varies."""
    name = "base"

    def chunk(self, doc: Document, elements: Sequence[Element]) -> list[Chunk]:
        chunks, seen = [], set()
        for section, text in self._groups(elements):              # the hook: (section title, text) pairs
            if not text.strip():
                continue
            c = Chunk.create(doc.doc_id, text.strip(), section, position=len(chunks), source=doc.source)
            if c.chunk_id in seen:                                # identical text in the same section -> keep one
                continue
            seen.add(c.chunk_id)
            chunks.append(c)
        return chunks

    @abstractmethod
    def _groups(self, elements: Sequence[Element]) -> list[tuple[str, str]]:
        ...


class HeadingChunker(BaseChunker):
    """One chunk per section (heading + its paragraphs); a section longer than max_words is split by paragraph."""

    def __init__(self, max_words: int = 120):
        self.max_words = max_words
        self.name = f"heading(max_words={max_words})"

    def _groups(self, elements):
        groups, section, buf = [], "", []

        def flush():
            if buf:
                groups.append((section, " ".join(buf)))
                buf.clear()

        for el in elements:
            if el.kind is ElementKind.HEADING:
                flush()
                section = el.text
            else:
                if buf and sum(len(b.split()) for b in buf) + len(el.text.split()) > self.max_words:
                    flush()                                        # budget reached: start a new chunk, same section
                buf.append(el.text)
        flush()
        return groups


class FixedWordChunker(BaseChunker):
    """Sliding window of `size` words, advancing by size - overlap."""

    def __init__(self, size: int = 50, overlap: int = 10):
        if not 0 <= overlap < size:
            raise ValueError("need 0 <= overlap < size")
        self.size, self.overlap = size, overlap
        self.name = f"fixed(size={size},overlap={overlap})"

    def _groups(self, elements):
        words = " ".join(e.text for e in elements if e.kind is not ElementKind.HEADING).split()
        step = self.size - self.overlap
        return [("", " ".join(words[i:i + self.size])) for i in range(0, max(len(words), 1), step)
                if words[i:i + self.size]]


class SentenceChunker(BaseChunker):
    """Pack whole sentences until max_words would be exceeded."""

    def __init__(self, max_words: int = 40):
        self.max_words = max_words
        self.name = f"sentence(max_words={max_words})"

    def _groups(self, elements):
        text = " ".join(e.text for e in elements if e.kind is not ElementKind.HEADING)
        out, buf, n = [], [], 0
        for s in re.split(r"(?<=[.!?])\s+", text):
            w = len(s.split())
            if buf and n + w > self.max_words:
                out.append(("", " ".join(buf)))
                buf, n = [], 0
            buf.append(s)
            n += w
        if buf:
            out.append(("", " ".join(buf)))
        return out
