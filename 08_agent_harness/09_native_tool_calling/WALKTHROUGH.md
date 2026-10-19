# Step 09 — Native tool/function calling (reference)

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

## Say it aloud
"Native tool calling lets me pass tool schemas to the API and get back structured tool_use blocks instead of parsing JSON from text. It's more reliable, but the model still only *requests* actions — I still validate arguments, enforce permissions, bound the loop, retry and trace."
