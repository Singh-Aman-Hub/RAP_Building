"""
GOAL:  Catch wrong-type / wrong-value input early using isinstance checks and Pydantic.

RUN:   cd 01_debugging_and_environment/05_types_and_validation && python main.py
       Crashing version (EXPECTED):  python broken_types.py

EXPECTED OUTPUT: friendly validation errors instead of crashes deep inside the code.

NEEDS: pydantic (installed in .venv), no internet.
"""

from pydantic import BaseModel, Field, ValidationError  # pydantic: validates data using type hints


# ---- hand-written validation ---------------------------------------------------------
def set_max_tokens(value):
    if not isinstance(value, int):                 # isinstance(obj, type) -> bool
        raise TypeError(f"max_tokens must be int, got {type(value).__name__}: {value!r}")
    if value <= 0:
        raise ValueError(f"max_tokens must be > 0, got {value}")
    return value


for candidate in [20, "20", -1, None]:
    try:
        print("ok:", set_max_tokens(candidate))
    except (TypeError, ValueError) as err:         # catch several types with a tuple
        print(f"rejected -> {type(err).__name__}: {err}")


# ---- Pydantic: the same idea declared with type hints -----------------------------------
class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1)              # at least one character
    max_tokens: int = Field(default=20, gt=0, le=512)   # gt = greater than, le = less or equal
    temperature: float = Field(default=1.0, ge=0, le=2)


good = GenerationRequest(prompt="Hello", max_tokens="30")   # "30" is coerced to int 30
print("\nvalid:", good, "| type of max_tokens:", type(good.max_tokens).__name__)

try:
    GenerationRequest(prompt="", max_tokens=-5, temperature=9)
except ValidationError as err:
    print(f"\n{err.error_count()} problems found:")
    for e in err.errors():                         # list of dicts: where / what / why
        print("  -", e["loc"], e["msg"])

# This is exactly what FastAPI does for you in module 09 (it answers HTTP 422 with this list).

# TRY IT: add a field `stop: list[str] = []` and pass stop="x" -> read the error message.
# BREAK IT: see broken_types.py for the 3 most common type bugs.
