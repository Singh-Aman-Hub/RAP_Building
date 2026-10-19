"""Solution for exercise 04: add a `reason` field to the schema, the prompt and the validator.
RUN: ../../.venv/bin/python ex04_reason_field.py   (from this folder)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from llm_client import extract_json
from pydantic import BaseModel, ValidationError
from typing import Literal


class Sentiment(BaseModel):
    reason: str                                   # NEW required field (put it first: reasoning before answer)
    label: Literal["positive", "negative", "neutral", "unknown"]


old_style = '{"label": "positive"}'                # an old-format reply now FAILS - schema and prompt must change together
new_style = '{"reason": "uses the word love", "label": "positive"}'
for raw in (old_style, new_style):
    try:
        print("ok  ", Sentiment.model_validate(extract_json(raw)))
    except ValidationError as err:
        print("FAIL", err.errors()[0]["msg"], "->", err.errors()[0]["loc"])
