# Step 03 — Tool registry, dispatch and argument validation

## The story so far
In [step 02](../02_agent_loop/) the loop runs any tool call the model writes, straight into `TOOLS[name](**args)`.

## The pain point
Models invent tool names, forget arguments and pass strings where numbers belong. Unchecked, that raises `KeyError` or `TypeError` inside your loop and the agent dies.

## The spark
Describe each tool once in a registry (function, description, parameter types, allowed values). Validate every call against it before running, and return any problem to the model as an `Observation: ERROR ...` so it can correct itself.

## Goal
Describe each tool once (function, description, parameter types, allowed values), validate every call before running it, and send problems back to the model as observations.

## Run it
```bash
cd 08_agent_harness/03_tool_registry_and_validation && ../../.venv/bin/python main.py
```
Real output:
```
case             observation
valid            42
unknown tool     ERROR: unknown tool 'weather'; available: ['calculator', 'shout']
missing arg      ERROR: missing required argument 'b'
wrong type       ERROR: argument 'a' has wrong type str
extra arg        ERROR: unexpected argument 'fast'
bad operator     ERROR: argument 'op'='**' not in ['+', '-', '*', '/']
divide by zero   ERROR: division by zero
bool as number   ERROR: argument 'a' has wrong type bool

Pydantic version of the same checks:
  {'a': 1, 'op': '+', 'b': 2} -> a=1.0 op='+' b=2.0
  {'a': '5', 'op': '+', 'b': 2} -> a=5.0 op='+' b=2.0
  {'a': 'five', 'op': '+', 'b': 2} -> ERROR: a: Input should be a valid number, unable to parse string as a number
  {'a': 1, 'b': 2} -> ERROR: op: Field required
  {'a': 1, 'op': '+', 'b': 2, 'fast': 1} -> ERROR: fast: Extra inputs are not permitted
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| nested dict `REGISTRY = {"calculator": {"fn": calculator, ...}}` | a function stored as a value | `d = {"f": len}; d["f"]("abc")` -> 3 |
| `(int, float)` as a "type" | a tuple of types, accepted by `isinstance` | `isinstance(2.0, (int, float))` |
| `-> list[str]` | return type hint: a list of strings | – |
| `spec["fn"](**args)` | call the stored function with unpacked args | – |
| `except Exception as err:` | catch-all net (use sparingly, report the error) | – |
| `model_config = {"extra": "forbid"}` | pydantic class setting: reject unknown fields | – |
| `err.errors()` | pydantic: list of dicts with `loc` and `msg` | – |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `isinstance(x, T)` | built-in | type check | bool |
| `sorted(dict)` | built-in | sorted keys | list |
| `pydantic.BaseModel` | 3rd-party | schema as a class | – |
| `ValidationError` | pydantic | raised when data does not fit | – |

## Execution trace
`execute(name, args)`: (1) `validate_call` returns a list of problems; if non-empty, return `"ERROR: ..."` without running anything; (2) otherwise call the function inside `try/except`; (3) always return a string. The loop appends that string as `Observation:` so a real model can read the error and try again.

## Experiments
- Add `word_count` to `REGISTRY` only; `tools_prompt()` lists it automatically (single source of truth).
- Remove the `bool` check: `True` passes as `1` silently.
- Pydantic lax mode converts `"5"` to `5.0` — convenient, but hides model mistakes. Try `{"strict": True}` in `model_config`.

## Debugging practice
`TypeError: unsupported operand type(s)` inside a tool = the harness did not validate ([`broken_unvalidated_args.py`](../broken_examples/broken_unvalidated_args.py)). Read the last traceback line first (what), then upwards (where it was called from).

## Rebuild checklist
- [ ] Registry dict with params/required/allowed. [ ] `validate_call` returning problems. [ ] `execute` that never raises.

## The new problem (cliffhanger)
Each turn adds messages, and the agent runs for many steps. The message list keeps growing until it no longer fits in the model's context window. Next: [04_history_and_context](../04_history_and_context/).

> **Say it aloud (interview box):** I describe every tool once in a registry with its parameters and allowed values, and I validate each model-proposed call before executing it. Problems go back to the model as error observations instead of crashing the loop. The registry also generates the tool descriptions in the prompt, so code and prompt cannot drift apart.
