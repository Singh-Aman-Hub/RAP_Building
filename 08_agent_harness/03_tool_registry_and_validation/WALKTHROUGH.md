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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Tool registry | tool table, tool catalogue, function table | Same | A dict holding each tool's function, description and parameter rules. |
| Validation | schema check, input sanitising | Near-same | Validation checks shape/type/allowed values; sanitising changes the data. |
| JSON Schema | tool `input_schema`, parameters schema | Close | The standard language for describing args; our `params` dict is a tiny hand-made version. |
| Pydantic model | dataclass, schema class, BaseModel | Not same | Pydantic validates and coerces at runtime; a dataclass does not validate. |
| Lax vs strict mode | coercion vs exact types | Opposites | Lax turns `"5"` into `5.0`; strict rejects it. |

## Code, step by step

### 1. Imports, path setup, fake model  (`main.py` lines 9–20)

```python
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json
from fake_llm import make_fake_agent_llm
from pydantic import BaseModel, ValidationError

LLM = make_fake_agent_llm()
```

- **In simple words:** Load helpers, pydantic, and the scripted model.
- **Exactly:** `BaseModel` is pydantic's schema base class; `ValidationError` is raised when data does not fit. `LLM` is the mock brain. `json`, `generate`, `extract_json` are imported but not used in this file.
- **Why it is here:** Pydantic is the library version of the hand checks coming below.

### 2. Two plain tools  (`main.py` lines 25–30)

```python
def calculator(a, op, b):
    ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(ops[op](a, b))

def shout(text):
    return text.upper()
```

- **In simple words:** A calculator and a function that shouts text.
- **Exactly:** `calculator(6,'*',7)` returns `'42'` (a string). `shout(text)` returns `text.upper()`. No type hints, because the registry carries the rules.
- **Why it is here:** Tools stay ordinary functions; all checking lives outside them.

### 3. The registry: one entry per tool  (`main.py` lines 37–43)

```python
REGISTRY = {
    "calculator": {"fn": calculator, "description": "Do arithmetic on two numbers.",
                   "params": {"a": (int, float), "op": str, "b": (int, float)},
                   "required": ["a", "op", "b"], "allowed": {"op": ["+", "-", "*", "/"]}},
    "shout": {"fn": shout, "description": "Upper-case some text.",
              "params": {"text": str}, "required": ["text"], "allowed": {}},
}
```

- **In simple words:** One dict that says everything about each tool: the function, what it does, and what arguments it takes.
- **Exactly:** `params` maps argument name to an expected type or tuple of types, e.g. `"a": (int, float)`. `required` lists mandatory names. `allowed` restricts values, e.g. `op` in `+ - * /`.
- **Why it is here:** One source of truth: the prompt, validator and dispatcher all read it, so they cannot disagree.

### 4. Generate the tool list for the prompt  (`main.py` lines 45–48)

```python
def tools_prompt() -> str:
    """Generate the tool list for the system prompt FROM the registry (single source of truth)."""
    lines = [f'- {name}({", ".join(spec["params"])}): {spec["description"]}' for name, spec in REGISTRY.items()]
    return "Available tools:\n" + "\n".join(lines)
```

- **In simple words:** Build the text menu of tools automatically from the registry.
- **Exactly:** Real output: `- calculator(a, op, b): Do arithmetic on two numbers.` and `- shout(text): Upper-case some text.` `", ".join(spec["params"])` joins the dict keys.
- **Why it is here:** Adding a tool to the registry updates the prompt for free; no stale hand-written prompt.

### 5. Check tool name and args container  (`main.py` lines 53–58)

```python
def validate_call(tool_name, args) -> list[str]:
    if tool_name not in REGISTRY:
        return [f"unknown tool {tool_name!r}; available: {sorted(REGISTRY)}"]
    spec, problems = REGISTRY[tool_name], []
    if not isinstance(args, dict):
        return ["args must be a JSON object"]
```

- **In simple words:** Reject calls to tools that do not exist or args that are not a dict.
- **Exactly:** Returns a `list[str]` of problems (empty means OK). Unknown tool: `unknown tool 'weather'; available: ['calculator', 'shout']`. `{name!r}` uses `repr` so quotes show.
- **Why it is here:** A real model can invent tool names; early return avoids a `KeyError`.

### 6. Check required arguments  (`main.py` lines 59–61)

```python
    for name in spec["required"]:
        if name not in args:
            problems.append(f"missing required argument {name!r}")
```

- **In simple words:** Every must-have argument has to be present.
- **Exactly:** Loops `spec["required"]`; with `{"a":6,"op":"*"}` it records `missing required argument 'b'`.
- **Why it is here:** Missing args would otherwise raise `TypeError` inside the tool.

### 7. Reject unknown arguments  (`main.py` lines 62–65)

```python
    for name, value in args.items():
        if name not in spec["params"]:
            problems.append(f"unexpected argument {name!r}")
            continue
```

- **In simple words:** Arguments we never declared are errors.
- **Exactly:** `fast` in `{"a":1,"op":"+","b":2,"fast":True}` gives `unexpected argument 'fast'`; `continue` skips the type checks for it.
- **Why it is here:** Passing extra kwargs to a function would crash it with `TypeError`.

### 8. Check the type, including the bool trap  (`main.py` lines 66–68)

```python
        # bool is a subclass of int in Python (True == 1) -> reject it for numeric params explicitly
        if isinstance(value, bool) or not isinstance(value, spec["params"][name]):
            problems.append(f"argument {name!r} has wrong type {type(value).__name__}")
```

- **In simple words:** Make sure each value is the right kind, and do not let `True` pass as the number 1.
- **Exactly:** `isinstance(value, spec["params"][name])` accepts a tuple of types. `bool` is a subclass of `int`, so it is rejected first. Real: `"six"` -> `argument 'a' has wrong type str`; `True` -> `wrong type bool`.
- **Why it is here:** Without the bool test `True + 2` silently returns 3.

### 9. Check allowed values, return problems  (`main.py` lines 69–71)

```python
        elif name in spec["allowed"] and value not in spec["allowed"][name]:
            problems.append(f"argument {name!r}={value!r} not in {spec['allowed'][name]}")
    return problems
```

- **In simple words:** Operators must be on the allowed list.
- **Exactly:** `"**"` gives `argument 'op'='**' not in ['+', '-', '*', '/']`. The function returns the full problem list.
- **Why it is here:** Stops values the tool cannot handle (would be `KeyError`).

### 10. Safe execution that always returns a string  (`main.py` lines 73–83)

```python
def execute(tool_name, args) -> str:
    """Validate -> run -> ALWAYS return a string observation (never crash the loop)."""
    problems = validate_call(tool_name, args)
    if problems:
        return "ERROR: " + "; ".join(problems)
    try:
        return REGISTRY[tool_name]["fn"](**args)
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except Exception as err:                       # last-resort net: report, don't crash
        return f"ERROR: tool raised {type(err).__name__}: {err}"
```

- **In simple words:** Validate, run, and turn every failure into text instead of crashing.
- **Exactly:** Problems join into `ERROR: ...`. `fn(**args)` runs the tool; `ZeroDivisionError` -> `ERROR: division by zero` (real output); any other exception -> `ERROR: tool raised <Type>: <msg>`.
- **Why it is here:** A crash would kill the agent; an error observation lets the model correct itself.

### 11. A pydantic schema for the calculator  (`main.py` lines 88–92)

```python
class CalcArgs(BaseModel):
    a: float
    op: str
    b: float
    model_config = {"extra": "forbid"}             # unknown keys -> error (pydantic v2 setting)
```

- **In simple words:** Describe the arguments as a class and let the library check them.
- **Exactly:** Fields `a: float`, `op: str`, `b: float`. `model_config = {"extra": "forbid"}` is pydantic v2 syntax: unknown keys raise a validation error.
- **Why it is here:** Less code than hand-written checks, and it produces structured errors.

### 12. Validate with pydantic and format errors  (`main.py` lines 94–98)

```python
def validate_with_pydantic(args):
    try:
        return CalcArgs(**args)                    # coerces "5" -> 5.0 by default (lax mode!)
    except ValidationError as err:
        return "ERROR: " + "; ".join(f"{e['loc'][0] if e['loc'] else '?'}: {e['msg']}" for e in err.errors())
```

- **In simple words:** Build the model from the args; if that fails, turn the errors into one message.
- **Exactly:** `CalcArgs(**args)` returns an object (`a=1.0 op='+' b=2.0`) or raises `ValidationError`. `err.errors()` is a list of dicts with `loc` (location tuple) and `msg`. Real: `a: Input should be a valid number, unable to parse string as a number`.
- **Why it is here:** Shows the library way; lax mode coerces `"5"` to `5.0`, which hides mistakes.

### 13. Main guard and show the tool menu  (`main.py` lines 100–101)

```python
if __name__ == "__main__":
    print(tools_prompt(), "\n")
```

- **In simple words:** Only run the demo when the file is run directly.
- **Exactly:** `if __name__ == "__main__":` is False on import, so other steps can import `REGISTRY`/`execute` without printing.
- **Why it is here:** Keeps the file reusable.

### 14. Run eight test calls  (`main.py` lines 102–114)

```python
    CALLS = [
        ("valid",            "calculator", {"a": 6, "op": "*", "b": 7}),
        ("unknown tool",     "weather",    {"city": "Pune"}),
        ("missing arg",      "calculator", {"a": 6, "op": "*"}),
        ("wrong type",       "calculator", {"a": "six", "op": "*", "b": 7}),
        ("extra arg",        "calculator", {"a": 1, "op": "+", "b": 2, "fast": True}),
        ("bad operator",     "calculator", {"a": 1, "op": "**", "b": 2}),
        ("divide by zero",   "calculator", {"a": 1, "op": "/", "b": 0}),
        ("bool as number",   "calculator", {"a": True, "op": "+", "b": 2}),
    ]
    print(f"{'case':16s} observation")
    for label, tool, args in CALLS:
        print(f"{label:16s} {execute(tool, args)}")
```

- **In simple words:** Try good and bad calls and print what the model would see.
- **Exactly:** Real table: valid -> `42`, unknown tool, missing arg, wrong type, extra arg, bad operator, divide by zero, bool as number -> each an `ERROR:` message.
- **Why it is here:** Each case checks one defence.

### 15. Run the pydantic checks  (`main.py` lines 116–119)

```python
    print("\nPydantic version of the same checks:")
    for args in [{"a": 1, "op": "+", "b": 2}, {"a": "5", "op": "+", "b": 2}, {"a": "five", "op": "+", "b": 2},
                 {"a": 1, "b": 2}, {"a": 1, "op": "+", "b": 2, "fast": 1}]:
        print(f"  {args} -> {validate_with_pydantic(args)}")
```

- **In simple words:** Same kinds of bad input through pydantic.
- **Exactly:** `{"a":"5",...}` is accepted as `a=5.0` (lax coercion); `"five"` fails; missing `op` -> `op: Field required`; `fast` -> `fast: Extra inputs are not permitted`.
- **Why it is here:** Contrasts lax pydantic with the stricter hand check.

### 16. See a bad call as the model would  (`main.py` lines 121–122)

```python
    # The errors are RETURNED to the model as an Observation, so a real model can self-correct:
    print("\nHow a bad call looks to the model:", {"role": "user", "content": "Observation: " + execute("calculator", {"a": 1})})
```

- **In simple words:** Show the error wrapped as a chat message.
- **Exactly:** Prints `{'role': 'user', 'content': "Observation: ERROR: missing required argument 'op'; missing required argument 'b'"}`.
- **Why it is here:** That message is how a real model learns what to fix.

### 17. Experiment comments  (`main.py` lines 124–128)

```python
# TRY IT: add a new tool 'word_count' (param text: str) to REGISTRY ONLY. tools_prompt() updates automatically.
# TRY IT: add a numeric range to 'allowed' - think about how you'd extend validate_call.
# BREAK IT: delete the isinstance(bool) test and run the 'bool as number' case: True is silently treated as 1.
# NOTE: pydantic's default 'lax' mode converts "5" to 5.0 - convenient but it hides model mistakes; use
#       model_config = {"strict": True} when you want exact types.
```

- **In simple words:** Ideas to try and a note on pydantic strict mode.
- **Exactly:** Comments only. `model_config = {"strict": True}` makes pydantic reject `"5"`.
- **Why it is here:** Tells you where to extend.

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
