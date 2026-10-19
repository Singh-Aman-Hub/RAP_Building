# Exercises — Module 00 (Python foundations)

For every step do the same 4 things. Worked answers for the "modify" tasks are in [solutions/](solutions/) — **try first**.

1. **Modify it** (change ONE thing, predict the output, then run)
2. **Break it** (cause the error, read the traceback bottom-up, name the type + cause + fix)
3. **Rebuild** (close the file; recreate from the prompt in 10 minutes)
4. **Say it aloud** (60 seconds, no notes)

| Step | Modify it | Break it | Rebuild from memory | Explain aloud |
|---|---|---|---|---|
| 01 types | Print a 3-row table (model name left-aligned, parameter count right-aligned with commas) -> [ex01](solutions/ex01_format_table.py) | `"epochs: " + 3` | 4 variables of 4 types + `type()` loop | Why does `bool("false")` equal `True`? |
| 02 control flow | Add a `max_steps` guard to a `while` loop whose loss grows -> [ex02](solutions/ex02_max_steps.py) | Infinite loop + Ctrl+C | FizzBuzz 1-30 | Difference between `break`, `continue`, and `for ... else` |
| 03 functions | Write `clip(value, low=0.0, high=1.0)` with hints and validation -> [ex03](solutions/ex03_clip.py) | `clip()` with missing arg; call with `low > high` | A function with 2 defaults called 3 ways | The mutable default argument bug |
| 04 collections | Group words by length; dedupe keeping order -> [ex04](solutions/ex04_group_dedupe.py) | `d["missing"]`; mutate a list while looping over it | Frequency count without `Counter` | Why is `in` on a set faster than on a list? |
| 05 classes | `Stack` class (push/pop/max size from a dataclass) -> [ex05](solutions/ex05_stack.py) | Drop `self.` before an attribute | Counter class with `__repr__` | What does `from_pretrained` return and why a classmethod? |
| 06 errors/files/json | `load_config` with defaults, missing file and bad JSON handled -> [ex06](solutions/ex06_load_config.py) | Open a missing file; `json.loads("{'a':1}")` | Write dict -> JSON -> read back | Why `with open(...)`? Why not bare `except:`? |
| 07 decorators/generators | `count_calls` decorator -> [ex07](solutions/ex07_count_calls.py) | Name your file `math.py` (shadowing) | A `batches(items, size)` generator | What does `if __name__ == "__main__"` do? |

## Mixed challenge (after all 7)
Write `summarize(path)` that reads a text file, counts word frequencies (lowercased, punctuation stripped), and writes the top 3 to `summary.json`. Requirements: type hints, docstring, `Path`, `with`, a dict for counting, `try/except FileNotFoundError`, and an `if __name__ == "__main__"` block. Then break it three ways and write a debug journal ([template](../01_debugging_and_environment/DEBUG_JOURNAL_TEMPLATE.md)) for one.

## Python syntax self-quiz (answer without running)
1. `"abc"[::-1]`  2. `[x for x in range(5) if x % 2]`  3. `{1, 2} | {2, 3}`  4. `dict(zip("ab", [1, 2]))`
5. `a, *rest = [1, 2, 3]` — what is `rest`?  6. `sorted(["b", "A", "c"], key=str.lower)`
7. What does `print(*[1, 2, 3], sep="-")` print?  8. `x = y = []; x.append(1); print(y)`
Answers: `'cba'`; `[1, 3]`; `{1, 2, 3}`; `{'a': 1, 'b': 2}`; `[2, 3]`; `['A', 'b', 'c']`; `1-2-3`; `[1]`.
