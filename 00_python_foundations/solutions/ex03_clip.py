"""Ex 03: function with defaults + type hints + validation."""


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Force value into [low, high]."""
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return max(low, min(value, high))


assert clip(1.7) == 1.0 and clip(-3) == 0.0 and clip(0.4) == 0.4
assert clip(5, high=10) == 5
print("ok")
