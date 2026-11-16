"""streaming.py -- chunk a document that does not fit in memory: cut it into segments at blank lines, chunk each segment."""
from __future__ import annotations

from typing import Iterable, Iterator

from .model import Chunk
from .registry import Chunker


def chunk_stream(lines: Iterable[str], chunker: Chunker, doc_id: str, segment_chars: int = 20_000) -> Iterator[Chunk]:
    """Read line by line, close a segment at the first blank line after `segment_chars`, chunk it, yield, forget it.
    Peak memory ~ one segment, not the whole file. Offsets stay global via base_offset."""
    buf, offset, pos = [], 0, 0
    for line in lines:
        buf.append(line)
        pos += len(line)
        if pos - offset >= segment_chars and not line.strip():     # a safe boundary: never inside a paragraph
            yield from chunker.chunk(doc_id, "".join(buf), base_offset=offset)
            buf, offset = [], pos
    if buf:
        yield from chunker.chunk(doc_id, "".join(buf), base_offset=offset)
