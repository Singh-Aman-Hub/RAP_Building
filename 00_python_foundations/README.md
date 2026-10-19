# Module 00 — Python Foundations

## Why this matters
The interviewers explicitly test **Python syntax** first: variables, functions, classes, imports, loops, conditionals, exceptions, comprehensions, type hints, file handling. Everything later (PyTorch, FastAPI, agents) is just these constructs in bigger code. If you can read and write this module fluently, "explain the code you generated" becomes easy.

## Where this fits
**Builds on:** nothing — this is the starting point (install Python 3.10+, create a venv: [01 step 03](../01_debugging_and_environment/03_import_and_environment/)).

**Leads to:**
- Functions + defaults + keyword arguments -> `model.generate(max_new_tokens=..., temperature=...)` in [04](../04_llm_fundamentals/) and [05](../05_llm_parameters/).
- Dicts/sets/loops -> every LeetCode problem in [02](../02_dsa_arrays_hashmaps/).
- Classes + `from_pretrained` -> Hugging Face models in [04](../04_llm_fundamentals/); dataclasses -> training configs in [05](../05_llm_parameters/).
- Exceptions + JSON -> validating tool arguments and retries in [08](../08_agent_harness/); parsing LLM JSON in [07](../07_prompt_engineering/).
- Decorators -> `@app.get("/health")` in [09](../09_api_development/), `@st.cache_resource` in [12](../12_streamlit_gradio/).
- Type hints -> Pydantic validation in [09](../09_api_development/) (preview in [01 step 05](../01_debugging_and_environment/05_types_and_validation/)).
- Generators/batching -> data loaders in [03](../03_ml_foundations/) and [06](../06_llm_training_and_fast_inference/).
- Files/pathlib -> saving models ([03](../03_ml_foundations/), [10](../10_ml_system_architecture/)) and Docker `COPY` paths ([11](../11_docker/)).

## Concept map
```
values & types (01) ──► control flow (02) ──► functions (03) ──┐
        │                      │                  │            │
        ▼                      ▼                  ▼            ▼
 collections & comprehensions (04) ◄──────── classes/dataclasses (05)
        │                                           │
        └──────────► exceptions + files + JSON (06) ◄┘
                                │
                                ▼
            modules, decorators, generators, __main__ (07)
                                │
                                ▼
          every later module (LLMs, agents, APIs, Docker, UI)
```

## Study path
| Step | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_variables_and_types](01_variables_and_types/) | types, strings, f-strings, conversions | `python main.py` | 45 min |
| 2 | [02_control_flow_and_loops](02_control_flow_and_loops/) | if/elif, for/while, enumerate/zip, break/continue | same | 45 min |
| 3 | [03_functions_and_type_hints](03_functions_and_type_hints/) | defaults, kwargs, `*args`, hints, mutable-default trap | same | 1 h |
| 4 | [04_collections_and_comprehensions](04_collections_and_comprehensions/) | list/tuple/dict/set, comprehensions, aliasing | same | 1.5 h |
| 5 | [05_classes_and_dataclasses](05_classes_and_dataclasses/) | classes, inheritance, classmethod, dataclass | same | 1 h |
| 6 | [06_exceptions_files_json](06_exceptions_files_json/) | try/except, `with`, pathlib, json | same | 1 h |
| 7 | [07_modules_decorators_generators](07_modules_decorators_generators/) | imports, decorators, generators, `__main__` | same | 1 h |

Total about 7 h; then [exercises.md](exercises.md) (about 3 h). Per step follow the loop in each `WALKTHROUGH.md`: read -> run -> trace -> modify -> break -> rebuild -> say aloud.

## Syllabus coverage (Python syntax list from the syllabus)
variables (01) · functions (03) · classes (05) · imports (07) · loops (02) · conditionals (02) · exceptions (06) · comprehensions (04) · type hints (03) · file handling (06).

## Concepts in plain English
- **Variable:** a label on a value; the value has the type. **Immutable** (str, int, tuple) can't change in place; **mutable** (list, dict, set) can — which is why aliasing (`b = a`) surprises people.
- **Function:** reusable block; defaults + keyword args let one function serve many cases. Trade-off: hidden state (mutable defaults, globals) causes bugs.
- **Comprehension:** a compact loop that builds a collection. Limit: unreadable if nested deeply.
- **Class:** data + behaviour bundled; `self` is "this object". Library models are instances of classes.
- **Exception:** a controlled failure. Catch the narrowest type you can handle.
- **Context manager (`with`):** guarantees cleanup (files closed).
- **Decorator / generator:** function wrappers / lazy producers. Generators save memory but can be used once.

## Common mistakes & gotchas
- Mutable default arguments (`def f(x, bucket=[])`).
- `=` vs `==`; forgetting `:`; mixing tabs and spaces.
- Modifying a list while iterating over it.
- `b = a` doesn't copy a list (use `.copy()`).
- Naming your file after a library (`random.py`, `json.py`) -> shadows the real one.
- Strings are immutable: `s.upper()` returns a new string.
- Reading numbers from files/env vars gives strings -> convert.

## Self-check
- [ ] I can write a function with defaults, keyword arguments and type hints
- [ ] I can explain list vs tuple vs dict vs set and give a use for each
- [ ] I can write a list/dict/set comprehension with a filter
- [ ] I can write a class with `__init__`, a method and `__repr__`; and a dataclass
- [ ] I can handle an exception and raise my own
- [ ] I can read and write text and JSON files using `pathlib` and `with`
- [ ] I can explain `import` vs `from ... import`, `__name__ == "__main__"`, decorators and `yield`
- [ ] I did all 7 "break it" drills and read the traceback bottom-up

## Interview questions
1. **List vs tuple?** List is mutable, tuple immutable (hashable if elements are -> can be dict keys). Use tuples for fixed records; lists for collections that change.
2. **Why is the mutable default argument a bug?** Defaults are evaluated once at function definition, so every call shares the same object. Use `None` and create inside. (Demonstrated in 03 and 01/09.)
3. **`is` vs `==`?** `==` compares values, `is` compares identity (same object). Use `is None`.
4. **What does `with open(...) as f` do?** Context manager: closes the file even if an error occurs.
5. **Explain a list comprehension vs a generator expression.** Both loop and filter; the list builds everything in memory, the generator yields lazily. Trade-off: memory vs reuse.
6. **What is a decorator?** A function that takes a function and returns a wrapped one; `@d` above `def f` equals `f = d(f)`. Example: `@app.get`.
7. **What does `if __name__ == "__main__"` do?** Runs code only when the file is executed directly, not when imported.
8. **dict vs set lookup complexity?** Average O(1) via hashing; list membership is O(n). Basis for Two Sum ([02](../02_dsa_arrays_hashmaps/)).
9. **`*args` and `**kwargs`?** Collect extra positional/keyword arguments into a tuple/dict; used to pass options through (e.g. wrapper functions).
10. **Type hints — are they enforced?** No. Tools (mypy, editors, Pydantic, FastAPI) use them; Python ignores them at runtime. Limitation: no runtime safety unless a library validates.
11. **Shallow vs deep copy?** `.copy()` copies the outer container only; nested objects are shared. `copy.deepcopy` copies recursively.
12. **How do you catch several exceptions?** `except (TypeError, ValueError) as e:`; prefer specific types over bare `except`.
