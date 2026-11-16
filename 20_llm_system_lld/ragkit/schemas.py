"""schemas.py -- pydantic v2 models for data that crosses a TRUST boundary (HTTP bodies, LLM JSON output).

Rule of thumb used in ragkit:
  dataclass (domain.py)  -> objects we build ourselves, already trusted, hot paths
  pydantic  (here)       -> text from outside: parse + coerce + validate + good error messages + JSON Schema
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChunkModel(BaseModel):
    """Same fields as domain.Chunk, but validated and coerced on construction."""
    model_config = ConfigDict(frozen=True)
    chunk_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    section: str = ""
    position: int = Field(default=0, ge=0)


class AskRequest(BaseModel):
    """What an HTTP endpoint `POST /ask` would accept."""
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)


class StructuredAnswer(BaseModel):
    """What we ask the LLM to return as JSON (step 08)."""
    answer: str
    citations: list[int] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
