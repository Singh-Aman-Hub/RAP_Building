"""Exercise 7: a pydantic contract for one element with a bbox-inside-page check, and a negative test.
RUN: python s07_contract_validator.py"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator


class El(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["title", "section_header", "text", "table", "page_footer"]
    text: str
    page: int
    bbox: list[float]
    reading_order: int
    level: int | None = None


class Doc(BaseModel):
    pages: dict[int, tuple[float, float]]               # page -> (width, height) in points
    elements: list[El]

    @model_validator(mode="after")
    def inside(self):
        for e in self.elements:
            w, h = self.pages[e.page]
            x0, y0, x1, y1 = e.bbox
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                raise ValueError(f"bbox {e.bbox} is outside the {w}x{h} page (pixel/point mix-up?)")
        return self


ok = dict(pages={1: (600, 480)}, elements=[dict(type="title", text="Report", page=1, bbox=[31.5, 23.25, 370.5, 48.0], reading_order=0)])
Doc.model_validate(ok)
bad = dict(pages={1: (600, 480)}, elements=[dict(type="title", text="Report", page=1, bbox=[42, 31, 794, 64], reading_order=0)])   # pixel box: x1=794 > 600
try:
    Doc.model_validate(bad)
    raise SystemExit("bug: pixel box accepted")
except ValidationError as e:
    print("rejected:", e.errors()[0]["msg"])
try:
    Doc.model_validate(dict(ok, elements=[dict(ok["elements"][0], type="heading")]))
    raise SystemExit("bug: heading accepted")
except ValidationError as e:
    print("rejected:", e.errors()[0]["loc"], e.errors()[0]["msg"][:60])
print("ok")
