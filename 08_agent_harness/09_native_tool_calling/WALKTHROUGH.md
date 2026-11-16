# Step 09 — Native tool/function calling (reference)

## The story so far
In [step 08](../08_trace_and_full_agent/) the whole agent runs on a text protocol that we invented and parse ourselves.

## The pain point
Parsing JSON out of prose is fragile, and every provider's models behave differently with our home-made format.

## The spark
Hosted APIs accept a `tools` schema and return structured `tool_use` blocks, so parsing disappears. The registry from step 03 can generate that schema. The loop, validation, permissions, retries and trace are still yours either way.

## Goal
Understand how hosted APIs expose tool calling natively, and that your harness (validation, permissions, retries, tracing) is needed either way.

## Run it
```bash
cd 08_agent_harness/09_native_tool_calling && ../../.venv/bin/python main.py      # no network
# optional, needs a key:  ANTHROPIC_API_KEY=... python native_tool_use_reference.py   (NOT run by default)
```
Real output (excerpt):
```
NATIVE FORMAT: the API receives `tools=[...]` with JSON Schema
{ "name": "calculator", "description": "Do arithmetic on two numbers.",
  "input_schema": {"type": "object", "properties": {"a": {"type": "number"}, "op": {"type": "string", "enum": ["+","-","*","/"]}, "b": {"type": "number"}}, "required": ["a","op","b"]} }

validation problems: []
message to send back: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01ABC", "content": "42", "is_error": false}]}
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Native tool calling | function calling, tool use, tools API | Same idea, different names per vendor | The API takes a structured `tools` list and returns structured call blocks; Anthropic says `tool_use`, OpenAI says `tool_calls`. |
| JSON Schema | `input_schema` (Anthropic), `parameters` (OpenAI) | Same schema language | Same standard, different field name. |
| Text protocol | prompt-based tool calling, ReAct-style JSON | Opposite of native | You write the format and parse it yourself; works with any model. |
| tool_use / tool_result | tool call / tool message, `function_call` / `function_call_output` | Same roles, vendor naming | The request block and the reply block, linked by an id. |
| Schema vs validation | `strict` tools, structured outputs vs your own checks | Not the same | A schema tells the model the shape; it does not stop bad values reaching your code. |

## Code, step by step

### 1. Imports and load the step-03 registry  (`main.py` lines 9–15)

```python
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_schema import REGISTRY, validate_call, execute, to_native_schema    # reuses step 03 by path
```

- **In simple words:** Bring in the registry, validator, executor and a schema converter.
- **Exactly:** `native_schema` loads step 03's `main.py` by file path (see below) and exposes `REGISTRY`, `validate_call`, `execute` and `to_native_schema`.
- **Why it is here:** Shows that the same checking code serves both protocols.

### 2. Describe the text protocol  (`main.py` lines 17–20)

```python
# --- 1. How OUR protocol tells the model about tools: plain text in the system prompt -----------
print("TEXT PROTOCOL  (step 02/03): the tool list is words in the system prompt; the model answers with JSON text")
print('  system: Reply with ONLY {"tool": "calculator", "args": {...}} or {"final": ...}')
print('  model : {"tool": "calculator", "args": {"a": 6, "op": "*", "b": 7}}\n')
```

- **In simple words:** Print how our own protocol looks.
- **Exactly:** The tool list lives as words in the system prompt and the model answers with JSON text; the printed example reply is `{"tool": "calculator", "args": {"a": 6, "op": "*", "b": 7}}`.
- **Why it is here:** Sets the baseline for comparison.

### 3. Print the native tool definition  (`main.py` lines 22–24)

```python
# --- 2. How a NATIVE API is told: a structured `tools` list with JSON Schema (see native_schema.py) -----
print("NATIVE FORMAT: the API receives `tools=[...]` with JSON Schema")
print(json.dumps(to_native_schema("calculator", REGISTRY["calculator"]), indent=2), "\n")
```

- **In simple words:** Convert the calculator entry into the JSON the API expects.
- **Exactly:** Real output: `{"name": "calculator", "description": "Do arithmetic on two numbers.", "input_schema": {"type": "object", "properties": {"a": {"type": "number"}, "op": {"type": "string", "enum": ["+","-","*","/"]}, "b": {"type": "number"}}, "required": ["a","op","b"]}}`. `json.dumps(indent=2)` pretty-prints.
- **Why it is here:** This is what you pass as `tools=[...]`.

### 4. A simulated native response  (`main.py` lines 26–33)

```python
# --- 3. A native response is STRUCTURED: no JSON-in-text parsing needed ---------------------------------
simulated_native_response = {
    "stop_reason": "tool_use",                                   # the model stopped because it wants a tool
    "content": [
        {"type": "text", "text": "I'll multiply those."},
        {"type": "tool_use", "id": "toolu_01ABC", "name": "calculator", "input": {"a": 6, "op": "*", "b": 7}},
    ],
}
```

- **In simple words:** A hand-written dict that looks like a real API reply.
- **Exactly:** `stop_reason` is `"tool_use"`; `content` has a text block and a `tool_use` block with `id`, `name` and `input` (already a dict, no JSON-in-text parsing).
- **Why it is here:** Lets the demo run offline without an API key.

### 5. Find tool_use blocks, validate, run  (`main.py` lines 34–37)

```python
for block in simulated_native_response["content"]:
    if block["type"] == "tool_use":
        problems = validate_call(block["name"], block["input"])  # STILL validate: schema != guarantee
        result = execute(block["name"], block["input"])
```

- **In simple words:** Loop the blocks; for each tool request check and run it.
- **Exactly:** `validate_call(name, input)` returns `[]` for the good call. `execute(...)` returns `'42'`.
- **Why it is here:** A schema in the request does not guarantee a valid call; models can still send bad input.

### 6. Build the tool_result message  (`main.py` lines 38–41)

```python
        tool_result_message = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": block["id"],  # must echo the id so the model can match call <-> result
             "content": result, "is_error": result.startswith("ERROR")},
        ]}
```

- **In simple words:** Package the result to send back.
- **Exactly:** A `user` message whose content is a list with one `tool_result` block: `tool_use_id` must equal the block's `id` (`toolu_01ABC`), `content` is the result text, `is_error` is `result.startswith("ERROR")`, here `False`.
- **Why it is here:** The id lets the model match each result to its call, essential with parallel calls.

### 7. Print the results  (`main.py` lines 42–43)

```python
        print("validation problems:", problems)
        print("message to send back:", json.dumps(tool_result_message))
```

- **In simple words:** Show problems and the message.
- **Exactly:** Real: `validation problems: []` and `message to send back: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01ABC", "content": "42", "is_error": false}]}`.
- **Why it is here:** Makes the wire format visible.

### 8. Summarise the differences  (`main.py` lines 45–52)

```python
print("""
DIFFERENCES to remember
  text protocol : works with ANY model, you parse + repair JSON yourself, you design the protocol.
  native calling: model is trained for it (fewer format errors), structured blocks, ids for parallel calls,
                  provider-specific field names (Anthropic tool_use/tool_result vs OpenAI tool_calls).
  both          : the model only REQUESTS a call - YOUR code runs it, so validation, permissions, retries,
                  step limits and tracing (steps 03-08) are needed either way.
""")
```

- **In simple words:** Compare text and native calling.
- **Exactly:** A printed block: text protocol works with any model but you parse and repair; native is trained-in, structured, uses ids; either way your code runs the tool.
- **Why it is here:** The interview takeaway: safety and validation are your job in both.

### 9. Experiment comments  (`main.py` lines 54–55)

```python
# TRY IT: change simulated 'input' to {"a": "six", ...} and read the validator's complaint.
# TRY IT: open native_tool_use_reference.py and map each line to a step in 02-08.
```

- **In simple words:** Try bad input.
- **Exactly:** Comments only; `"a": "six"` would give a type problem.
- **Why it is here:** Shows validation still matters.

### 10. Load step 03 by file path  (`native_schema.py` lines 9–17)

```python
import importlib.util
from pathlib import Path

_step03 = Path(__file__).resolve().parents[1] / "03_tool_registry_and_validation" / "main.py"
_spec = importlib.util.spec_from_file_location("step03_registry", _step03)   # (module name, file path)
step03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(step03)                                              # actually run the file

REGISTRY, validate_call, execute = step03.REGISTRY, step03.validate_call, step03.execute
```

- **In simple words:** Import code from a folder whose name starts with a digit.
- **Exactly:** Python names cannot start with a digit, so `importlib.util.spec_from_file_location(name, path)` creates a spec, `module_from_spec` creates the module and `exec_module` runs the file. Then `REGISTRY`, `validate_call`, `execute` are taken from it. Running step 03 imports it with `__name__ != "__main__"`, so its demo does not print.
- **Why it is here:** Avoids copying code and avoids clashing `main.py` names.

### 11. Convert a registry entry to JSON Schema  (`native_schema.py` lines 20–29)

```python
def to_native_schema(name, spec):
    """Build an Anthropic-style tool definition from our registry entry."""
    json_type = {str: "string", int: "number", float: "number", (int, float): "number"}
    properties = {}
    for param, py_type in spec["params"].items():
        properties[param] = {"type": json_type[py_type]}
        if param in spec["allowed"]:
            properties[param]["enum"] = spec["allowed"][param]                # allowed values -> JSON Schema enum
    return {"name": name, "description": spec["description"],
            "input_schema": {"type": "object", "properties": properties, "required": spec["required"]}}
```

- **In simple words:** Translate Python types into JSON Schema types.
- **Exactly:** `json_type` maps `str`->`"string"`, `int`/`float`/`(int, float)`->`"number"`. For each param build `{"type": ...}`; if in `allowed`, add `"enum": [...]`. Return `{"name", "description", "input_schema": {"type":"object","properties","required"}}`, the Anthropic format.
- **Why it is here:** One registry feeds both your validator and the API schema, so they cannot drift apart.

### 12. Imports (read, not run)  (`native_tool_use_reference.py` lines 8–15)

```python
import os
import sys
from pathlib import Path

import anthropic   # 3rd-party SDK: pip install anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_schema import to_native_schema, REGISTRY, execute   # converter + our validated dispatch
```

- **In simple words:** Needs the Anthropic SDK and our converter.
- **Exactly:** `anthropic` is the official SDK (`pip install anthropic`; not installed here). Checked Nov 2026 against the Anthropic docs: `tools` with `name`, `description`, `input_schema`, `tool_use`/`tool_result` blocks are current.
- **Why it is here:** Reference only; no key or network here.

### 13. Pick the model  (`native_tool_use_reference.py` lines 17–17)

```python
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
```

- **In simple words:** Read the model name from an environment variable.
- **Exactly:** `os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")`. `claude-haiku-5-5` is a valid API ID in the Nov 2026 models overview.
- **Why it is here:** Lets you swap models without editing code.

### 14. Create client, tools, messages  (`native_tool_use_reference.py` lines 19–22)

```python
def run_native_agent(question: str, max_steps: int = 5) -> str:
    client = anthropic.Anthropic()                          # reads ANTHROPIC_API_KEY
    tools = [to_native_schema(name, spec) for name, spec in REGISTRY.items()]
    messages = [{"role": "user", "content": question}]
```

- **In simple words:** Set up the client and convert every registry tool.
- **Exactly:** `anthropic.Anthropic()` reads `ANTHROPIC_API_KEY`. `tools` is a list of schema dicts.
- **Why it is here:** Native calling needs the schema list on every request.

### 15. Call the model; stop if no tool is requested  (`native_tool_use_reference.py` lines 24–28)

```python
    for _ in range(max_steps):                              # same bounded loop as step 02
        response = client.messages.create(model=MODEL, max_tokens=500, tools=tools, messages=messages)

        if response.stop_reason != "tool_use":              # model produced a normal answer -> done
            return "".join(b.text for b in response.content if b.type == "text")
```

- **In simple words:** Ask the model; if it did not stop for a tool, return its text.
- **Exactly:** `client.messages.create(model, max_tokens=500, tools, messages)`. `response.stop_reason == "tool_use"` means a tool is wanted. Text is joined from `text` blocks.
- **Why it is here:** The bounded loop is the same as step 02.

### 16. Echo the assistant turn  (`native_tool_use_reference.py` lines 30–31)

```python
        # IMPORTANT: put the assistant's whole turn (text + tool_use blocks) back into the history
        messages.append({"role": "assistant", "content": response.content})
```

- **In simple words:** Put the model's whole answer in the history.
- **Exactly:** `response.content` is the list of text and tool_use blocks; it goes in unchanged as the assistant message.
- **Why it is here:** The API expects each `tool_result` to match a `tool_use` in the previous assistant turn.

### 17. Run tools and return all results in one message  (`native_tool_use_reference.py` lines 33–39)

```python
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":                    # block.name, block.input (dict), block.id
                result = execute(block.name, block.input)   # our validation + permissions + error handling
                tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                     "content": result, "is_error": result.startswith("ERROR")})
        messages.append({"role": "user", "content": tool_results})   # ALL results in ONE user message
```

- **In simple words:** Execute each requested tool and send all results together.
- **Exactly:** `block.id`, `block.name`, `block.input` (dict). `execute` validates, gates, and returns text. All `tool_result` blocks go in a single user message.
- **Why it is here:** Missing or split results cause an API error or confuse parallel calls.

### 18. Step limit and entry point  (`native_tool_use_reference.py` lines 41–44)

```python
    return "STOPPED: max_steps"

if __name__ == "__main__":
    print(run_native_agent("What is 6 x 7?"))
```

- **In simple words:** Stop after `max_steps` and run an example.
- **Exactly:** Returns `STOPPED: max_steps` when exhausted; the main guard runs `"What is 6 x 7?"`.
- **Why it is here:** Prevents unbounded loops and cost.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `importlib.util.spec_from_file_location(name, path)` | load a Python file by path under a name you choose (needed because folder names start with digits) | see `native_schema.py` |
| dict comprehension with `{**a, **b}` | build `properties` dynamically | – |
| `block.type == "tool_use"` | SDK response blocks are objects with a `.type` | – |
| `"".join(b.text for b in blocks if b.type == "text")` | join all text blocks | – |

## Libraries & functions
| Name | Kind | Does | Important args | Returns |
|---|---|---|---|---|
| `anthropic.Anthropic()` | 3rd-party SDK | API client; reads `ANTHROPIC_API_KEY` | – | client |
| `client.messages.create` | 3rd-party SDK | one model call | `model`, `max_tokens`, `tools`, `messages`, `system` | response with `.content` blocks and `.stop_reason` |
| `stop_reason == "tool_use"` | API field | model wants tool(s) run | – | – |
| `tool_use` block | API object | `id`, `name`, `input` (dict) | – | – |
| `tool_result` block | message content | result for a given `tool_use_id`; `is_error` flag | – | – |

## Execution trace (native loop)
1. Send `messages` + `tools` schema. 2. Response has `stop_reason="tool_use"` and `tool_use` blocks. 3. Append the **whole assistant turn** to history. 4. For each block: validate -> execute -> build a `tool_result` with the same id. 5. Send all results in one user message. 6. Repeat until `stop_reason` is not `tool_use`. Same loop as step 02 with structured blocks instead of parsed text.

## Text protocol vs native
| | Text protocol (02-08) | Native tool calling |
|---|---|---|
| Works with | any model | models/APIs that support it |
| Format reliability | you validate + repair | model trained for it; still validate |
| Parallel calls | you design it | built in via ids |
| Portability | high | field names differ per provider |

## Experiments
- Edit the simulated `input` to `{"a": "six", ...}` — the validator still catches it (schemas guide the model, they do not guarantee).
- Map each line in `native_tool_use_reference.py` to a step in 02–08 (loop=02, validation=03, errors=06, permissions=07).

## Debugging practice
Typical native-API errors: `400` when a `tool_result` is missing for a `tool_use` id, or when history order is wrong. Read the error's message: it names the offending message index.

## The new problem (cliffhanger)
The agent now works end to end, but only inside your terminal. Other programs, a website or a teammate cannot call it. We need to put it behind a network interface. Next: [09_api_development](../../09_api_development/).

> **Say it aloud (interview box):** Native tool calling means I send a tools schema and the API returns structured tool-use blocks instead of text I must parse. The agent loop is the same: execute the tool, append the whole assistant turn and the tool result, and repeat. My harness still owns validation, permissions, retries and tracing, because the API does not provide those.
