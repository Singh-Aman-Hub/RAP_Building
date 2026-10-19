"""
GOAL   : Turn the tool dict into a REGISTRY with descriptions + parameter schemas, validate every
         tool call BEFORE running it, and return problems to the model as observations.
RUN    : cd 08_agent_harness/03_tool_registry_and_validation && python main.py
EXPECTED OUTPUT: a table of tool calls: valid ones run; unknown tool / missing arg / wrong type /
         extra arg / bad operator are rejected with a clear message (and the same checks with pydantic).
NEEDS  : pydantic (installed).
"""
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

# ---------------------------------------------------------------------------
# Tools (plain functions)
# ---------------------------------------------------------------------------
def calculator(a, op, b):
    ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(ops[op](a, b))

def shout(text):
    return text.upper()

# ---------------------------------------------------------------------------
# REGISTRY: everything the harness knows about a tool lives in ONE dict entry.
#   description -> shown to the model        params -> {arg name: expected Python type}
#   required   -> args that must be present  allowed -> optional allowed values per arg
# ---------------------------------------------------------------------------
REGISTRY = {
    "calculator": {"fn": calculator, "description": "Do arithmetic on two numbers.",
                   "params": {"a": (int, float), "op": str, "b": (int, float)},
                   "required": ["a", "op", "b"], "allowed": {"op": ["+", "-", "*", "/"]}},
    "shout": {"fn": shout, "description": "Upper-case some text.",
              "params": {"text": str}, "required": ["text"], "allowed": {}},
}

def tools_prompt() -> str:
    """Generate the tool list for the system prompt FROM the registry (single source of truth)."""
    lines = [f'- {name}({", ".join(spec["params"])}): {spec["description"]}' for name, spec in REGISTRY.items()]
    return "Available tools:\n" + "\n".join(lines)

# ---------------------------------------------------------------------------
# VALIDATION by hand: return a list of problems (empty list = OK)
# ---------------------------------------------------------------------------
def validate_call(tool_name, args) -> list[str]:
    if tool_name not in REGISTRY:
        return [f"unknown tool {tool_name!r}; available: {sorted(REGISTRY)}"]
    spec, problems = REGISTRY[tool_name], []
    if not isinstance(args, dict):
        return ["args must be a JSON object"]
    for name in spec["required"]:
        if name not in args:
            problems.append(f"missing required argument {name!r}")
    for name, value in args.items():
        if name not in spec["params"]:
            problems.append(f"unexpected argument {name!r}")
            continue
        # bool is a subclass of int in Python (True == 1) -> reject it for numeric params explicitly
        if isinstance(value, bool) or not isinstance(value, spec["params"][name]):
            problems.append(f"argument {name!r} has wrong type {type(value).__name__}")
        elif name in spec["allowed"] and value not in spec["allowed"][name]:
            problems.append(f"argument {name!r}={value!r} not in {spec['allowed'][name]}")
    return problems

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

# ---------------------------------------------------------------------------
# Same idea with PYDANTIC: declare the schema as a class instead of writing checks.
# ---------------------------------------------------------------------------
class CalcArgs(BaseModel):
    a: float
    op: str
    b: float
    model_config = {"extra": "forbid"}             # unknown keys -> error (pydantic v2 setting)

def validate_with_pydantic(args):
    try:
        return CalcArgs(**args)                    # coerces "5" -> 5.0 by default (lax mode!)
    except ValidationError as err:
        return "ERROR: " + "; ".join(f"{e['loc'][0] if e['loc'] else '?'}: {e['msg']}" for e in err.errors())

if __name__ == "__main__":
    print(tools_prompt(), "\n")
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

    print("\nPydantic version of the same checks:")
    for args in [{"a": 1, "op": "+", "b": 2}, {"a": "5", "op": "+", "b": 2}, {"a": "five", "op": "+", "b": 2},
                 {"a": 1, "b": 2}, {"a": 1, "op": "+", "b": 2, "fast": 1}]:
        print(f"  {args} -> {validate_with_pydantic(args)}")

    # The errors are RETURNED to the model as an Observation, so a real model can self-correct:
    print("\nHow a bad call looks to the model:", {"role": "user", "content": "Observation: " + execute("calculator", {"a": 1})})

# TRY IT: add a new tool 'word_count' (param text: str) to REGISTRY ONLY. tools_prompt() updates automatically.
# TRY IT: add a numeric range to 'allowed' - think about how you'd extend validate_call.
# BREAK IT: delete the isinstance(bool) test and run the 'bool as number' case: True is silently treated as 1.
# NOTE: pydantic's default 'lax' mode converts "5" to 5.0 - convenient but it hides model mistakes; use
#       model_config = {"strict": True} when you want exact types.
