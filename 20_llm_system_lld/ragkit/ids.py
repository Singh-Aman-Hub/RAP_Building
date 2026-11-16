"""ids.py -- how entities get their identity (step 02).

Two kinds of id:
  content_hash : deterministic. Same input -> same id. Used for chunks so re-ingesting an unchanged
                 document produces the SAME ids (idempotent upsert).
  new_id       : random (uuid4). Used for things that are events, not content: traces, requests, feedback.
"""
from __future__ import annotations

import hashlib
import uuid


def content_hash(*parts: str, length: int = 16) -> str:
    """sha256 over the parts, joined with an unambiguous separator; return the first `length` hex chars."""
    joined = "\x1f".join(parts)                      # \x1f (unit separator) so ("ab","c") != ("a","bc")
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def new_id(prefix: str = "") -> str:
    """Random 12-hex-char id with an optional readable prefix, e.g. 'trace-3f9a1c0b77de'."""
    return f"{prefix}{'-' if prefix else ''}{uuid.uuid4().hex[:12]}"
