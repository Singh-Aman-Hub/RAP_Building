"""
GOAL:  Understand imports, `if __name__ == "__main__"`, decorators and generators (all appear in FastAPI/PyTorch code).

RUN:   cd 00_python_foundations/07_modules_decorators_generators && python main.py

EXPECTED OUTPUT (first lines):
    sqrt(16) = 4.0
    ...

NEEDS: nothing (standard library only), no internet.
"""

# ---- imports ----------------------------------------------------------------
import math                            # import a whole module; use math.sqrt(...)
from collections import Counter, defaultdict  # import specific names
import time as t                       # alias with `as`
# `import X` finds X in: the current folder, installed packages (site-packages), the standard library.
# If two places have the same name, the first found wins -> naming your file random.py breaks `import random`!

print("sqrt(16) =", math.sqrt(16))
print(Counter("banana"))                # Counter: dict subclass for counting (also useful for LeetCode)
groups = defaultdict(list)              # dict that creates a missing value automatically
groups["fruit"].append("apple")
print(dict(groups))

# ---- decorators ----------------------------------------------------------------
# A decorator is a function that takes a function and returns an improved function.
# `@something` above a def is shorthand for: f = something(f)
def timed(func):
    def wrapper(*args, **kwargs):       # accepts anything, passes it through
        start = t.perf_counter()
        result = func(*args, **kwargs)  # call the original function
        elapsed = t.perf_counter() - start
        print(f"{func.__name__} took {elapsed * 1000:.2f} ms")
        return result
    return wrapper


@timed                                  # same as: slow_sum = timed(slow_sum)
def slow_sum(n):
    return sum(range(n))


print(slow_sum(1_000_000))

# You will meet decorators as: @app.get("/health") in FastAPI, @st.cache_resource in Streamlit,
# @torch.no_grad() in PyTorch, @dataclass (module 05 of this chapter).

# ---- generators: functions that `yield` one value at a time -------------------
def batches(items, size):
    """Yield successive chunks. Memory-friendly: only one chunk exists at a time."""
    for i in range(0, len(items), size):    # range(start, stop, step)
        yield items[i:i + size]             # pause here, hand back a chunk, resume next time


for batch in batches(list(range(7)), 3):    # this is how data loaders feed a model
    print("batch:", batch)

gen = batches([1, 2, 3, 4], 2)
print(next(gen), next(gen))                 # next() pulls one value manually
# next(gen) again would raise StopIteration: the generator is exhausted.

# ---- the main guard --------------------------------------------------------------
# When you run `python main.py`, __name__ == "__main__".
# When another file does `import main`, __name__ == "main" so this block is skipped.
# That lets a file be both a reusable module AND a runnable script.
def main():
    print("running as a script, __name__ =", __name__)


if __name__ == "__main__":
    main()

# TRY IT: add @timed to `batches`... what does it print and why is the time ~0? (Hint: generators are lazy.)
# BREAK IT: create a file named math.py next to this script, then run again -> weird AttributeError. Delete it after!
