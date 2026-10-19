<!-- vocabulary draft for 08_agent_harness (merged by the build script) -->

### Agent (LLM agent)
<!-- cat: llm -->
- **Technical definition:** A program that repeatedly asks an LLM to choose an action, executes it through tools, feeds back the result and stops on a final answer or limit.
- **Plain English:** A loop around a model that can use tools.
- **Example / analogy:** A manager who delegates to staff, reads reports and decides the next step.
- **Snippet:**
```python
for step in range(max_steps):
    action = decide(messages)
    if "final" in action:
        break
    messages.append(run_tool(action))
```
- **Snippet explained:** Bounded loop: decide, maybe act, append the result, repeat.
- **Related terms:** [[Agent loop]], [[Tool (function calling)]], [[Max steps]]

### Agent loop
<!-- cat: llm -->
- **Technical definition:** The control flow ask-model -> parse action -> validate -> execute tool -> append observation -> repeat.
- **Plain English:** The repeat cycle that makes a model act step by step.
- **Example / analogy:** Think, act, look at the result, think again.
- **Snippet:**
```python
while steps < max_steps:
    reply = generate(messages)
    ...
```
- **Snippet explained:** Each pass is one step; exit on `final` or the limit.
- **Related terms:** [[Agent (LLM agent)]], [[Stop condition]], [[Observation]]

### Tool (function calling)
<!-- cat: llm -->
- **Technical definition:** A named function exposed to the model with a description and parameter schema; the model requests calls, the harness executes them.
- **Plain English:** An ability you let the model request, like a calculator.
- **Example / analogy:** A tool in a toolbox that only you can pick up.
- **Snippet:**
```python
def calculator(a, op, b):
    return str({"+": a + b, "*": a * b}[op])
```
- **Snippet explained:** A tool is an ordinary Python function returning a string.
- **Related terms:** [[Tool registry]], [[Tool dispatch]], [[Native tool calling]]

### Tool registry
<!-- cat: llm -->
- **Technical definition:** A dict mapping tool names to functions plus metadata (description, parameter types, permissions).
- **Plain English:** A directory of tools the agent may use.
- **Example / analogy:** A phone book of available helpers.
- **Snippet:**
```python
REGISTRY = {"calculator": {"fn": calculator, "params": {"a": float, "op": str, "b": float}}}
```
- **Snippet explained:** One entry per tool; prompts and validators are generated from it.
- **Related terms:** [[Tool dispatch]], [[Argument validation]]

### Tool dispatch
<!-- cat: llm -->
- **Technical definition:** Looking up the requested tool by name and calling it with the model-provided arguments.
- **Plain English:** Finding the right function and running it.
- **Example / analogy:** A switchboard operator connecting a call.
- **Snippet:**
```python
result = REGISTRY[name]["fn"](**args)
```
- **Snippet explained:** `**args` unpacks the dict into keyword arguments.
- **Related terms:** [[Tool registry]], [[Keyword arguments (**kwargs)]]

### Keyword arguments (**kwargs)
<!-- cat: python -->
- **Technical definition:** Passing a dict as named arguments with `f(**d)`, or collecting extra named arguments with `def f(**kwargs)`.
- **Plain English:** Turning a dict into named inputs for a function.
- **Example / analogy:** Handing over a form where each box already has a label.
- **Snippet:**
```python
def add(a, b):
    return a + b
print(add(**{"a": 1, "b": 2}))
```
- **Snippet explained:** Prints 3: the dict keys must match parameter names or `TypeError` is raised.
- **Related terms:** [[Tool dispatch]], [[Argument validation]]

### Argument validation
<!-- cat: llm -->
- **Technical definition:** Checking tool name, required/unknown arguments, types and allowed values before executing a tool call.
- **Plain English:** Checking the model's request makes sense before acting on it.
- **Example / analogy:** Checking a cheque is filled in correctly before cashing it.
- **Snippet:**
```python
if "b" not in args:
    return "ERROR: missing required argument 'b'"
```
- **Snippet explained:** A problem is returned as an observation so the model can retry.
- **Related terms:** [[Tool registry]], [[Pydantic model]], [[Observation]]

### Observation
<!-- cat: llm -->
- **Technical definition:** The text result of a tool call appended to the conversation so the model can see what happened.
- **Plain English:** The tool's answer, shown back to the model.
- **Example / analogy:** A lab report returned to the doctor.
- **Snippet:**
```python
messages.append({"role": "user", "content": f"Observation: {result}"})
```
- **Snippet explained:** Results re-enter the conversation as a message.
- **Related terms:** [[Agent loop]], [[Message list (chat history)]]

### Message list (chat history)
<!-- cat: llm -->
- **Technical definition:** The ordered list of role/content dicts sent on every API call; the only memory a stateless model has.
- **Plain English:** The conversation so far, resent each time.
- **Example / analogy:** Giving a new assistant the full meeting minutes before every question.
- **Snippet:**
```python
history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
```
- **Snippet explained:** Append both user and assistant turns to keep context.
- **Related terms:** [[Context window]], [[Short-term memory]]

### Context window
<!-- cat: llm -->
- **Technical definition:** The maximum number of tokens (prompt + reply) a model can consider in one request.
- **Plain English:** How much text the model can read at once.
- **Example / analogy:** The size of a desk: more fits, but everything else falls off.
- **Snippet:**
```python
def estimate_tokens(ms):
    return sum(len(m["content"]) for m in ms) // 4
```
- **Snippet explained:** Rough estimate (about 4 characters per token) to decide when to trim.
- **Related terms:** [[Truncation (history)]], [[Summarisation (history)]]

### Truncation (history)
<!-- cat: llm -->
- **Technical definition:** Dropping older messages (usually keeping the first and the newest) to fit the token budget.
- **Plain English:** Cutting off the oldest part of the chat.
- **Example / analogy:** Tearing the oldest pages out of a notebook.
- **Snippet:**
```python
kept = [first] + messages[-6:]
```
- **Snippet explained:** Keep the goal message plus recent turns.
- **Related terms:** [[Summarisation (history)]], [[Context window]]

### Summarisation (history)
<!-- cat: llm -->
- **Technical definition:** Replacing old messages with a model-written summary to save tokens while keeping key facts.
- **Plain English:** Shrinking the old chat into a short recap.
- **Example / analogy:** Meeting minutes instead of a full transcript.
- **Snippet:**
```python
summary = generate([{"role": "user", "content": "Summarise:\n" + transcript}])
```
- **Snippet explained:** One extra model call produces the recap message.
- **Related terms:** [[Truncation (history)]], [[Context window]]

### Short-term memory
<!-- cat: llm -->
- **Technical definition:** Information held in the current run's message list; lost when the run ends.
- **Plain English:** What the agent remembers during this conversation.
- **Example / analogy:** Working memory.
- **Snippet:**
```python
messages = [{"role": "user", "content": question}]  # fresh each run
```
- **Snippet explained:** A new list each run means no carry-over.
- **Related terms:** [[Long-term memory]], [[Message list (chat history)]]

### Long-term memory
<!-- cat: llm -->
- **Technical definition:** Information persisted outside the model (file, database, vector store) and accessed through tools across runs.
- **Plain English:** Facts the agent can look up later, even in a new session.
- **Example / analogy:** A notebook kept on the shelf.
- **Snippet:**
```python
MEMORY_FILE.write_text(json.dumps({"name": "Priya"}))
```
- **Snippet explained:** Writes a dict as JSON text to disk.
- **Related terms:** [[Short-term memory]], [[Memory poisoning]]

### Memory poisoning
<!-- cat: llm -->
- **Technical definition:** Storing attacker-controlled text in long-term memory so that it later influences the agent as if trusted.
- **Plain English:** Bad data saved now, causing bad behaviour later.
- **Example / analogy:** Someone forging an entry in your diary.
- **Snippet:**
```python
remember("note", web_page_text)  # unsafe: untrusted text saved verbatim
```
- **Snippet explained:** Untrusted text can contain instructions the model will later read.
- **Related terms:** [[Long-term memory]], [[Prompt injection]]

### Stop condition
<!-- cat: llm -->
- **Technical definition:** A rule that ends the agent loop: model returns final, step limit reached, repeated calls, budget exhausted or fatal error.
- **Plain English:** How the agent knows when to stop.
- **Example / analogy:** A finish line and a time limit.
- **Snippet:**
```python
if "final" in decision:
    return decision["final"]
```
- **Snippet explained:** Returns immediately on the model's final answer.
- **Related terms:** [[Max steps]], [[Agent loop]]

### Max steps
<!-- cat: llm -->
- **Technical definition:** A hard upper bound on loop iterations that prevents runaway cost and infinite loops.
- **Plain English:** The most times the agent may go round the loop.
- **Example / analogy:** A circuit breaker.
- **Snippet:**
```python
for step in range(1, max_steps + 1):
    ...
return "STOPPED: max_steps"
```
- **Snippet explained:** Bounded `range` guarantees the loop ends.
- **Related terms:** [[Stop condition]], [[Agent loop]]

### Exponential backoff
<!-- cat: api -->
- **Technical definition:** A retry strategy that waits progressively longer between attempts (base * 2^(n-1)), usually with random jitter, and a retry cap.
- **Plain English:** Wait a bit, then longer, then longer again before retrying.
- **Example / analogy:** Knocking on a door with growing pauses.
- **Snippet:**
```python
import random, time
delay = 0.05 * 2 ** (attempt - 1) + random.uniform(0, 0.05)
time.sleep(delay)
```
- **Snippet explained:** Attempt 1 waits ~0.05s, attempt 2 ~0.10s, attempt 3 ~0.20s.
- **Related terms:** [[Jitter]], [[Transient error]], [[Rate limit (429)]]

### Jitter
<!-- cat: api -->
- **Technical definition:** A small random amount added to retry delays so many clients do not retry at the same moment.
- **Plain English:** Randomness in the wait time.
- **Example / analogy:** People leaving a stadium at slightly different times.
- **Snippet:**
```python
random.uniform(0, 0.05)
```
- **Snippet explained:** Returns a random float between 0 and 0.05.
- **Related terms:** [[Exponential backoff]]

### Transient error
<!-- cat: api -->
- **Technical definition:** A temporary failure (timeout, 429, 5xx, dropped connection) that may succeed if retried.
- **Plain English:** A hiccup that may go away.
- **Example / analogy:** A busy phone line.
- **Snippet:**
```python
RETRYABLE = (TimeoutError, ConnectionError)
```
- **Snippet explained:** Only these exception types are retried.
- **Related terms:** [[Exponential backoff]], [[Permanent error]]

### Permanent error
<!-- cat: api -->
- **Technical definition:** A failure that retrying cannot fix (e.g. 400 bad request, 401 unauthorised, 404 not found, bug in code).
- **Plain English:** An error that will keep happening.
- **Example / analogy:** Dialling a number that does not exist.
- **Snippet:**
```python
except ValueError:
    raise  # do not retry bugs
```
- **Snippet explained:** Non-retryable errors propagate immediately.
- **Related terms:** [[Transient error]], [[HTTP status code]]

### Rate limit (429)
<!-- cat: api -->
- **Technical definition:** An API response (HTTP 429) telling the client it sent too many requests; the client should wait and retry later.
- **Plain English:** 'Slow down, you are asking too fast.'
- **Example / analogy:** A turnstile that lets in limited people per minute.
- **Snippet:**
```python
if response.status_code == 429:
    time.sleep(int(response.headers.get("retry-after", 1)))
```
- **Snippet explained:** Honour the server's `retry-after` header if present.
- **Related terms:** [[Exponential backoff]], [[Transient error]]

### JSON repair loop
<!-- cat: llm -->
- **Technical definition:** On unparsable model output, send the error back and ask for corrected JSON, up to a fixed number of attempts.
- **Plain English:** Ask the model to fix its own formatting.
- **Example / analogy:** A proofreader returning a page marked in red.
- **Snippet:**
```python
repairs += 1
if repairs > max_repairs:
    return "FAILED"
```
- **Snippet explained:** A counter bounds the number of repairs.
- **Related terms:** [[Retry with feedback]], [[Argument validation]]

### Least privilege
<!-- cat: api -->
- **Technical definition:** Giving a component only the permissions and tools it needs for its task.
- **Plain English:** Don't hand over more power than needed.
- **Example / analogy:** A cleaner gets a room key, not the safe combination.
- **Snippet:**
```python
agent = Agent(tools=["calculator"], mode="read_only")
```
- **Snippet explained:** Only one read-only tool is exposed.
- **Related terms:** [[Read-only mode]], [[Tool safety]]

### Read-only mode
<!-- cat: api -->
- **Technical definition:** A harness setting that refuses tools which modify state (write, delete, send).
- **Plain English:** The agent may look but not touch.
- **Example / analogy:** A museum rope.
- **Snippet:**
```python
if mode == "read_only" and not spec["read_only"]:
    return "ERROR: permission denied"
```
- **Snippet explained:** Checks the tool's own `read_only` flag.
- **Related terms:** [[Least privilege]], [[Confirmation hook]]

### Path sandboxing
<!-- cat: api -->
- **Technical definition:** Restricting file access to one directory by resolving the requested path and verifying it is inside that directory.
- **Plain English:** Only letting the agent open files in one folder.
- **Example / analogy:** A playpen with walls.
- **Snippet:**
```python
p = (SANDBOX / user_path).resolve()
if not p.is_relative_to(SANDBOX):
    raise PermissionError
```
- **Snippet explained:** `resolve()` removes `..`; `is_relative_to` checks containment.
- **Related terms:** [[Path traversal]], [[Least privilege]]

### Path traversal
<!-- cat: api -->
- **Technical definition:** An attack using `..` segments or absolute paths to reach files outside an intended directory.
- **Plain English:** Sneaking out of the allowed folder with ../..
- **Example / analogy:** Using a back staircase to reach a restricted floor.
- **Snippet:**
```python
Path("sandbox") / "../../.env"
```
- **Snippet explained:** Without resolving and checking, this reaches a parent directory file.
- **Related terms:** [[Path sandboxing]]

### Confirmation hook
<!-- cat: api -->
- **Technical definition:** A function called before a risky tool runs that returns whether a human (or policy) approves it.
- **Plain English:** A 'are you sure?' step before dangerous actions.
- **Example / analogy:** Two-person approval for a payment.
- **Snippet:**
```python
def ask_human(name, args):
    return input(f"Allow {name}({args})? [y/N] ").lower() == "y"
```
- **Snippet explained:** Returns True only for an explicit yes.
- **Related terms:** [[Read-only mode]], [[Least privilege]]

### Tool safety
<!-- cat: llm -->
- **Technical definition:** The set of controls (validation, permissions, sandbox, confirmation, size limits, logging) that bound what an agent's tools can do.
- **Plain English:** Everything that stops tools from doing harm.
- **Example / analogy:** Guard rails on a mountain road.
- **Snippet:**
```python
result = guarded_execute(name, args, mode, confirm)
```
- **Snippet explained:** One gate function applies all checks.
- **Related terms:** [[Least privilege]], [[Path sandboxing]], [[Confirmation hook]]

### JSONL (JSON Lines)
<!-- cat: api -->
- **Technical definition:** A text format with one JSON object per line; append-friendly and easy to grep or stream.
- **Plain English:** A log file where each line is a small JSON record.
- **Example / analogy:** A ledger with one entry per line.
- **Snippet:**
```python
with open("trace.jsonl", "a") as f:
    f.write(json.dumps({"event": "start"}) + "\n")
```
- **Snippet explained:** Opens in append mode and writes one object plus a newline.
- **Related terms:** [[Trace (agent trace)]], [[JSON (JavaScript Object Notation)]]

### Trace (agent trace)
<!-- cat: llm -->
- **Technical definition:** A chronological record of agent events (model replies, tool calls with args and results, errors, final status) used for debugging and monitoring.
- **Plain English:** The agent's diary of what it did.
- **Example / analogy:** A flight recorder.
- **Snippet:**
```python
{"event": "tool_call", "tool": "calculator", "ok": true, "ms": 0.01}
```
- **Snippet explained:** One JSONL record for a tool call.
- **Related terms:** [[JSONL (JSON Lines)]], [[Logging]]

### Native tool calling
<!-- cat: llm -->
- **Technical definition:** Provider API feature where tool schemas are passed in a `tools` parameter and the model returns structured tool_use blocks instead of JSON in text.
- **Plain English:** The API's built-in way for the model to ask for tools.
- **Example / analogy:** A built-in order form versus handwriting a note.
- **Snippet:**
```python
resp = client.messages.create(model=M, max_tokens=500, tools=tools, messages=messages)
resp.stop_reason  # "tool_use"
```
- **Snippet explained:** `stop_reason == "tool_use"` signals a requested call.
- **Related terms:** [[Tool (function calling)]], [[JSON Schema]], [[tool_result]]

### JSON Schema
<!-- cat: api -->
- **Technical definition:** A standard JSON vocabulary (type, properties, required, enum) describing the structure of JSON data; used to describe tool inputs.
- **Plain English:** A machine-readable description of what the JSON must look like.
- **Example / analogy:** A blueprint for a data form.
- **Snippet:**
```json
{"type": "object", "properties": {"a": {"type": "number"}}, "required": ["a"]}
```
- **Snippet explained:** Declares an object that must contain numeric `a`.
- **Related terms:** [[Native tool calling]], [[Pydantic model]]

### tool_result
<!-- cat: llm -->
- **Technical definition:** A message content block returned to the model containing a tool's output, keyed by the `tool_use_id` of the request, optionally flagged `is_error`.
- **Plain English:** The reply slip matching a tool request.
- **Example / analogy:** A receipt carrying the order number.
- **Snippet:**
```json
{"type": "tool_result", "tool_use_id": "toolu_01", "content": "42", "is_error": false}
```
- **Snippet explained:** The id lets the model match result to request.
- **Related terms:** [[Native tool calling]], [[Observation]]

### Hallucinated tool call
<!-- cat: llm -->
- **Technical definition:** A model request for a tool or arguments that do not exist or are invalid.
- **Plain English:** The model asking for something that is not there.
- **Example / analogy:** Asking a shop assistant for a product they don't sell.
- **Snippet:**
```python
if name not in REGISTRY:
    return f"ERROR: unknown tool {name!r}"
```
- **Snippet explained:** Reject and list available tools.
- **Related terms:** [[Argument validation]], [[Tool registry]]
