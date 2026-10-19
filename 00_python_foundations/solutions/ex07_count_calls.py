"""Ex 07: a decorator that counts calls."""
import functools


def count_calls(func):
    @functools.wraps(func)                         # keeps func.__name__ / docstring
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0                              # functions are objects: we can attach attributes
    return wrapper


@count_calls
def greet(name):
    return f"hi {name}"


greet("a"); greet("b")
print(greet.__name__, greet.calls)
assert greet.calls == 2
