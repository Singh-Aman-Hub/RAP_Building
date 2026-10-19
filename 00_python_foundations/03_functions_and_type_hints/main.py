"""
GOAL:  Write functions with parameters, defaults, return values and type hints.

RUN:   cd 00_python_foundations/03_functions_and_type_hints && python main.py

EXPECTED OUTPUT (first lines):
    add(2, 3) = 5
    ...

NEEDS: nothing (standard library only), no internet.
"""

from typing import Optional  # Optional[int] means "an int or None" (type-hint helper)


# def name(parameters): body.  `return` hands a value back to the caller.
def add(a, b):
    return a + b


print("add(2, 3) =", add(2, 3))

# ---- default values and keyword arguments --------------------------------
# Parameters with defaults are optional. This is exactly how generate(max_new_tokens=20) works.
def generate_text(prompt, max_new_tokens=5, temperature=1.0):
    return f"[{prompt}] + {max_new_tokens} tokens @ temp {temperature}"


print(generate_text("Hi"))                                 # uses both defaults
print(generate_text("Hi", 20))                             # positional: fills max_new_tokens
print(generate_text("Hi", temperature=0.2))                # keyword: skip a parameter by name
print(generate_text(prompt="Hi", temperature=0, max_new_tokens=1))  # keywords can be in any order

# ---- type hints ----------------------------------------------------------
# `: type` after a parameter and `-> type` after the parentheses are HINTS for humans and tools.
# Python does NOT enforce them at runtime (see the demo below), but editors and Pydantic/FastAPI use them.
def average(numbers: list[float]) -> float:
    return sum(numbers) / len(numbers)          # sum() and len() are built-ins


def find_index(items: list[str], target: str) -> Optional[int]:
    """Return the index of target, or None if absent.  (A docstring: triple-quoted text that documents.)"""
    for i, item in enumerate(items):
        if item == target:
            return i
    return None                                  # explicit "not found"


print("average:", average([1.0, 2.0, 6.0]))
print("find_index:", find_index(["a", "b"], "b"), find_index(["a", "b"], "z"))

# Hints are not enforced:
print("average of strings 'works' until it crashes:", end=" ")
try:
    print(average(["1", "2"]))                   # wrong type, no hint error, a TypeError inside sum
except TypeError as err:
    print("TypeError ->", err)

# ---- returning several values (a tuple) and unpacking --------------------
def min_max(values: list[int]) -> tuple[int, int]:
    return min(values), max(values)              # returns ONE tuple (low, high)


low, high = min_max([4, 9, 1])                   # tuple unpacking
print("low, high =", low, high)

# ---- *args and **kwargs --------------------------------------------------
# *args collects extra positional arguments into a tuple; **kwargs collects extra keywords into a dict.
def show(*args, **kwargs):
    print("args:", args, "| kwargs:", kwargs)


show(1, 2, name="gpt2", layers=12)

# ---- scope and the mutable-default trap (classic interview question) -----
def bad_append(x, bucket=[]):                    # the [] is created ONCE, shared by every call
    bucket.append(x)
    return bucket


print(bad_append(1), bad_append(2))              # surprise: [1, 2] [1, 2]  (same list!)


def good_append(x, bucket=None):
    if bucket is None:                           # create a fresh list per call
        bucket = []
    bucket.append(x)
    return bucket


print(good_append(1), good_append(2))            # [1] [2]

# ---- lambda: a tiny anonymous function, handy for sorting keys -----------
words = ["banana", "fig", "apple"]
print(sorted(words, key=lambda w: len(w)))       # sort by length

# TRY IT: give generate_text a new parameter top_k=50 and call it with top_k=1.
# BREAK IT: call add(1) (missing argument) and read the TypeError message.
