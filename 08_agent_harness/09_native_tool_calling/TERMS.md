# Terms — Step 09: Native tool/function calling

Folder: `08_agent_harness/09_native_tool_calling` · Files covered: `main.py`, `native_schema.py`, `native_tool_use_reference.py` (read only, not run), `WALKTHROUGH.md`
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Native tool calling (function calling)](#native-tool-calling-function-calling) · [Text protocol (comparison)](#text-protocol-comparison) · [JSON Schema](#json-schema) · [input_schema](#input_schema) · [enum (allowed values)](#enum-allowed-values) · [to_native_schema](#to_native_schema) · [Python type to JSON type mapping](#python-type-to-json-type-mapping) · [stop_reason](#stop_reason) · [Content blocks (text, tool_use)](#content-blocks-text-tool_use) · [tool_use block](#tool_use-block) · [tool_result message](#tool_result-message) · [tool_use_id](#tool_use_id) · [Validate even with a schema](#validate-even-with-a-schema) · [Parallel tool calls](#parallel-tool-calls) · [importlib.util (load a file by path)](#importlibutil-load-a-file-by-path) · [Reusing code by path](#reusing-code-by-path) · [Anthropic SDK client](#anthropic-sdk-client) · [client.messages.create parameters](#clientmessagescreate-parameters) · [Assistant turn goes back into history](#assistant-turn-goes-back-into-history) · [Generator expression with join for text blocks](#generator-expression-with-join-for-text-blocks) · [Attribute access on SDK objects vs dict keys](#attribute-access-on-sdk-objects-vs-dict-keys) · [os.getenv for the model name](#osgetenv-for-the-model-name) · [Reference-only file (py_compile)](#reference-only-file-py_compile) · [json.dumps with indent](#jsondumps-with-indent) · [Triple-quoted print and for loop over content blocks](#triple-quoted-print-and-for-loop-over-content-blocks) · [Provider-specific field names](#provider-specific-field-names)

---

## Native tool calling (function calling)
*Type: concept*

**1. Definition:** An API feature where you pass tool definitions as structured data (names, descriptions, JSON Schema for inputs) and the model replies with structured tool-request blocks instead of free text containing JSON.
**2. In plain words:** The model is trained for this format, so there are fewer formatting mistakes and you do not parse JSON out of prose. It still only requests an action; your code decides whether to run it, so validation and safety are still your job.
**3. Example to quote:**
> "With native calling the response has a `tool_use` block with `name` and `input`; I still run my validator before executing it."

**Seen in:** `main.py`, section 2 and 3; `WALKTHROUGH.md`.

## Text protocol (comparison)
*Type: concept*

**1. Definition:** The approach of earlier steps: the tool list is plain words in the system prompt and the model replies with a JSON object as text, which you parse and repair yourself.
**2. In plain words:** It works with any model and you control the design, but you carry the parsing and repair burden. Native calling is more reliable but field names differ per provider.
**3. Example to quote:**
> "Text protocol works with any model; native calling works only where the API supports it but needs less repair code."

**Seen in:** `main.py`, section 1 and the differences printout.

## JSON Schema
*Type: concept*

**1. Definition:** A standard vocabulary for describing the shape of JSON data: types, properties, required fields and allowed values (`enum`).
**2. In plain words:** It tells the model exactly what arguments a tool accepts. Schemas guide the model but do not guarantee correct output, so you still validate.
**3. Example to quote:**
> "The calculator schema says `a` and `b` are numbers and `op` is a string with an `enum` of four operators."

**Seen in:** `native_schema.py`, `to_native_schema`.

## input_schema
*Type: API field (Anthropic)*

**1. Definition:** The field of a tool definition that holds its JSON Schema: `{"type": "object", "properties": {...}, "required": [...]}`.
**2. In plain words:** It replaces the hand-written tool list in the system prompt. The converter builds it from the registry.
**3. Example to quote:**
> "`input_schema` lists `a`, `op` and `b` as properties and marks all three required."

**Seen in:** `native_schema.py`; `main.py` printout.

## enum (allowed values)
*Type: concept (JSON Schema)*

**1. Definition:** A JSON Schema keyword listing the only values allowed for a property.
**2. In plain words:** The registry's `allowed` operators become `enum: ["+", "-", "*", "/"]`, steering the model away from inventing operators.
**3. Example to quote:**
> "`properties['op']['enum'] = spec['allowed']['op']` puts the four operators into the schema."

**Seen in:** `native_schema.py`, `to_native_schema`.

## to_native_schema
*Type: project function*

**1. Definition:** Converts one registry entry into an Anthropic-style tool definition with `name`, `description` and `input_schema`.
**2. In plain words:** It proves the single registry can serve both approaches: the text prompt and the native API.
**3. Example to quote:**
> "`to_native_schema('calculator', REGISTRY['calculator'])` prints the dict I would pass as `tools=[...]`."

**Seen in:** `native_schema.py`.

## Python type to JSON type mapping
*Type: Python syntax / concept*

**1. Definition:** A dict mapping Python types to JSON Schema type names: `str` to `"string"`, `int`/`float` and the tuple `(int, float)` to `"number"`.
**2. In plain words:** The registry stores Python types; the API wants JSON type names. A dict is the simplest translator. A tuple works as a dict key because tuples are hashable.
**3. Example to quote:**
> "`json_type[(int, float)]` returns `'number'`."

**Seen in:** `native_schema.py`, `json_type`.

## stop_reason
*Type: API field*

**1. Definition:** A field of the response telling why the model stopped; `"tool_use"` means it wants one or more tools run, anything else (such as normal end of turn) means it answered.
**2. In plain words:** It is the native equivalent of checking for `final` in the JSON text. The loop continues while it equals `tool_use`.
**3. Example to quote:**
> "`if response.stop_reason != 'tool_use': return ...` ends the loop with the model's text answer."

**Seen in:** `main.py`, simulated response; `native_tool_use_reference.py`.

## Content blocks (`text`, `tool_use`)
*Type: API concept*

**1. Definition:** A response's `content` is a list of typed blocks; `text` blocks hold words and `tool_use` blocks hold `id`, `name` and `input` (a dict).
**2. In plain words:** One reply can contain an explanation and a tool request, even several tool requests. You loop over the blocks and handle each by type.
**3. Example to quote:**
> "The simulated reply has a text block 'I'll multiply those.' and a `tool_use` block for the calculator."

**Seen in:** `main.py`, `simulated_native_response`.

## tool_use block
*Type: API object*

**1. Definition:** A block with `type: "tool_use"`, a unique `id`, the tool `name` and the arguments `input`.
**2. In plain words:** It is the structured replacement for `{"tool": ..., "args": ...}`. The `id` links the request to its result.
**3. Example to quote:**
> "`block['input']` is `{'a': 6, 'op': '*', 'b': 7}`, ready to validate."

**Seen in:** `main.py`; `native_tool_use_reference.py`.

## tool_result message
*Type: API object*

**1. Definition:** A user message containing `tool_result` blocks that carry the output of a tool back to the model, each with the matching `tool_use_id`, `content` and an `is_error` flag.
**2. In plain words:** Echoing the id lets the model match each result to its call, which matters for parallel calls. All results from one assistant turn go in one user message.
**3. Example to quote:**
> "My result for `toolu_01ABC` was content `42` with `is_error` false."

**Seen in:** `main.py`, `tool_result_message`.

## tool_use_id
*Type: API field*

**1. Definition:** The identifier of a specific tool request that a result must echo back.
**2. In plain words:** If you forget or mismatch it, the API returns a 400 error because a `tool_use` has no corresponding `tool_result`.
**3. Example to quote:**
> "I copy `block['id']` into `tool_use_id` so the model can pair the result with its request."

**Seen in:** `main.py`; `native_tool_use_reference.py`.

## Validate even with a schema
*Type: concept*

**1. Definition:** Running your own validation on native tool inputs, because a schema guides the model but does not guarantee conforming arguments.
**2. In plain words:** The harness is needed either way: validation, permissions, step limits, retries and tracing.
**3. Example to quote:**
> "If I change the simulated input to `a: 'six'` the validator still reports a type problem."

**Seen in:** `main.py`, `validate_call(block["name"], block["input"])`.

## Parallel tool calls
*Type: concept*

**1. Definition:** A single model turn can request several tools at once, each with its own id.
**2. In plain words:** Ids make it possible to return several results in one message and keep them matched.
**3. Example to quote:**
> "If the model asks for two lookups in one turn, I append both `tool_result` blocks to the same user message."

**Seen in:** `WALKTHROUGH.md`, comparison table.

## importlib.util (load a file by path)
*Type: Python standard library*

**1. Definition:** `spec_from_file_location(name, path)`, `module_from_spec(spec)` and `spec.loader.exec_module(module)` load a Python file as a module under a name you choose.
**2. In plain words:** A folder named `03_tool_registry...` cannot be written in an `import` statement because names cannot start with a digit, and several files are called `main.py`. `importlib` loads the file explicitly.
**3. Example to quote:**
> "`native_schema.py` loads step 03's `main.py` by its path and reuses its `REGISTRY`, `validate_call` and `execute`."

**Seen in:** `native_schema.py`.

## Reusing code by path
*Type: concept*

**1. Definition:** Importing code from a sibling step through its file path rather than duplicating it.
**2. In plain words:** The registry and validator exist once in step 03; this step proves the same code serves the native API too.
**3. Example to quote:**
> "`REGISTRY, validate_call, execute = step03.REGISTRY, step03.validate_call, step03.execute`."

**Seen in:** `native_schema.py`, last lines.

## Anthropic SDK client
*Type: third-party: anthropic*

**1. Definition:** `anthropic.Anthropic()` creates an API client that reads the `ANTHROPIC_API_KEY` environment variable; `client.messages.create(...)` sends a request.
**2. In plain words:** It is the official library for calling the hosted model. This step's real-SDK file is reference only and is not run by default, since it needs a key and network access.
**3. Example to quote:**
> "`client.messages.create(model=MODEL, max_tokens=500, tools=tools, messages=messages)` is the native call."

**Seen in:** `native_tool_use_reference.py`.

## client.messages.create parameters
*Type: third-party: anthropic*

**1. Definition:** Main arguments: `model` (which model), `max_tokens` (cap on reply length), `tools` (list of tool definitions), `messages` (conversation) and `system` (standing instructions).
**2. In plain words:** It returns a response with `.content` blocks and a `.stop_reason`.
**3. Example to quote:**
> "I pass `tools=[to_native_schema(...)]` so the model knows what it may call."

**Seen in:** `native_tool_use_reference.py`.

## Assistant turn goes back into history
*Type: concept*

**1. Definition:** After a `tool_use` response you must append the whole assistant message (text plus `tool_use` blocks) to the history before sending the results.
**2. In plain words:** The model needs to see its own request next to the result; leaving it out leads to API errors.
**3. Example to quote:**
> "`messages.append({'role': 'assistant', 'content': response.content})` keeps the request in the conversation."

**Seen in:** `native_tool_use_reference.py`, loop.

## Generator expression with join for text blocks
*Type: Python syntax*

**1. Definition:** `"".join(b.text for b in response.content if b.type == "text")` concatenates the text of all text blocks.
**2. In plain words:** The reply may be split into several text blocks; this collects them into one answer string.
**3. Example to quote:**
> "When the model finishes, I return the joined text of its blocks."

**Seen in:** `native_tool_use_reference.py`.

## Attribute access on SDK objects vs dict keys
*Type: Python syntax*

**1. Definition:** SDK responses are objects, so you write `block.type`, `block.name`, `block.input`; the simulated response in `main.py` is a dict, so it uses `block["type"]`.
**2. In plain words:** Same idea, different syntax; mixing them up raises `AttributeError` or `TypeError`.
**3. Example to quote:**
> "In the reference file I write `block.name`; in the simulation I write `block['name']`."

**Seen in:** `native_tool_use_reference.py`; `main.py`.

## os.getenv for the model name
*Type: Python standard library*

**1. Definition:** `os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")` reads an environment variable with a default.
**2. In plain words:** You can change the model without editing code.
**3. Example to quote:**
> "`MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-haiku-5-5')` falls back to the default model name."

**Seen in:** `native_tool_use_reference.py`.

## Reference-only file (`py_compile`)
*Type: concept / command*

**1. Definition:** A file meant to be read, not run; `python -m py_compile file.py` checks that it has valid syntax without executing it.
**2. In plain words:** The real SDK example needs an API key and network, so it is only compile-checked.
**3. Example to quote:**
> "I ran `python -m py_compile native_tool_use_reference.py` to confirm the syntax without calling the API."

```bash
python -m py_compile native_tool_use_reference.py
```

**Seen in:** `native_tool_use_reference.py` docstring.

## json.dumps with indent
*Type: Python standard library*

**1. Definition:** `json.dumps(obj, indent=2)` converts to JSON text with two-space indentation.
**2. In plain words:** The script prints the native tool definition readably.
**3. Example to quote:**
> "`print(json.dumps(to_native_schema(...), indent=2))` shows the schema as nested JSON."

**Seen in:** `main.py`.

## Triple-quoted print and for loop over content blocks
*Type: Python syntax*

**1. Definition:** A triple-quoted string prints the multi-line differences summary, and a `for block in ...["content"]` loop handles each block by type.
**2. In plain words:** The loop is the native version of "decide whether the model gave a final answer or a tool request".
**3. Example to quote:**
> "`if block['type'] == 'tool_use':` is where validation and execution happen."

**Seen in:** `main.py`.

## Provider-specific field names
*Type: concept*

**1. Definition:** Different providers name the same ideas differently (Anthropic `tool_use`/`tool_result`, OpenAI `tool_calls`).
**2. In plain words:** A text protocol is portable; native calling ties you to a provider's format, so keep a thin adapter layer.
**3. Example to quote:**
> "My registry is provider-neutral, and `to_native_schema` is the only Anthropic-specific part."

**Seen in:** `main.py`, differences printout.
