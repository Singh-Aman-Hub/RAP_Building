# Step 08 — Tracing and the full agent

## The story so far
In [step 07](../07_safety_and_permissions/) the agent is guarded. All the pieces from steps 01 to 07 now exist; they just sit in separate folders.

## The pain point
When an agent misbehaves you only see the final answer. Which prompt did the model see, which tool ran, which call failed, and how long did each take?

## The spark
Log one JSON line per event (start, model reply, tool call, end) to a JSONL trace, and write a small reader. Put every earlier idea into one readable `agent.py` loop of about 200 lines.

## Goal
Record every step to a JSONL trace so the agent can be debugged afterwards, and see all earlier ideas combined in [`agent.py`](agent.py).

## Run it
```bash
cd 08_agent_harness/08_trace_and_full_agent
../../.venv/bin/python main.py && ../../.venv/bin/python show_trace.py
```
Real output (excerpt; run ids differ):
```
final      steps=2  'What is 6 x 7?'                     -> Result: 42
final      steps=3  'Calculate 12 + 8 then * 3'          -> Result: 60
final      steps=2  'read file ../../.env'               -> Sorry, the tool failed: ERROR: permission denied: path '../../.env' is outside the sandbox
final      steps=2  'delete file notes.txt'              -> Sorry, the tool failed: ERROR: permission denied: delete_file needs confirmation and was not approved
final      steps=1  'Tell me a joke'                     -> I don't know how to help with that.

max_steps=1: {'status': 'max_steps', 'answer': 'Stopped: step limit reached.', 'steps': 1, ...}

49 events, 10 runs
event counts: {'start': 10, 'llm_reply': 19, 'tool_call': 10, 'end': 10}

Failed tool calls (the first place to look when an agent misbehaves):
  [7be13922] step 1: read_file({'path': '../../.env'}) -> ERROR: permission denied: path '../../.env' is outside the sandbox
  [c13bbf7b] step 1: calculator({'a': 5, 'op': '/', 'b': 0}) -> ERROR: ZeroDivisionError: division by zero
  [4550bf88] step 1: delete_file({'path': 'notes.txt'}) -> ERROR: permission denied: delete_file needs confirmation and was not approved
```

The one loop everything else guards (see `run` in [`agent.py`](agent.py)):
```
Prompt -> LLM call -> parse tool call -> execute tool -> return observation -> iterate until final answer
```

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class Agent:` + `__init__(self, ...)` + methods | bundles settings and behaviour; `self` = this instance | `a = Agent(max_steps=3); a.run("q")` |
| default args `max_steps=6` | optional parameters with defaults | – |
| `{**dict1, "k": v}` / `**fields` | merge/unpack dicts; `**fields` collects keyword args | `{**{"a":1}, "b":2}` |
| `with path.open("a") as f:` | open a file in append mode and close it automatically | – |
| JSONL | one JSON object per line: append-only, greppable, streamable | `{"a":1}\n{"a":2}\n` |
| `[json.loads(l) for l in text.splitlines() if l.strip()]` | list comprehension with a filter | – |
| generator expression `(x for x in events if ...)` | lazy loop, no list built | – |
| `uuid.uuid4().hex[:8]` | short random id for a run | – |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `uuid.uuid4()` | standard library | random unique id | UUID |
| `time.time()` | standard library | seconds since 1970 (for timing) | float |
| `collections.Counter` | standard library | counts items | dict-like |
| `Agent.run(question)` | our code | the whole loop | dict(status, answer, steps, run_id) |

## Execution trace
`Agent.run` logs `start`; per step it logs `llm_reply`, then either `bad_json` or `tool_call` (with args, result, ok, ms) and finally `end` with a **status** (`final`, `max_steps`, `llm_unavailable`, `bad_json`). The trace file lets you answer "what did it do and where did it go wrong?" without re-running — and without asking an AI.

## Experiments
- Switch `mode` to `read_only`; compare traces.
- `Agent(tools=["calculator"])` then ask to read a file -> `unknown tool` (least privilege).
- Use `grep '"ok": false' traces/agent_trace.jsonl`.

## Debugging practice
Use `show_trace.py` to find the **first** failing event of a bad run, then reproduce it with the smallest input (module [01](../../01_debugging_and_environment/)).

## Rebuild checklist
- [ ] `log()` appending JSON lines. [ ] statuses for every exit path. [ ] a summary script.

## The new problem (cliffhanger)
The full agent works, but it parses tool calls from plain text with our own protocol. Hosted APIs offer a native tool-calling format. Next: [09_native_tool_calling](../09_native_tool_calling/).

> **Say it aloud (interview box):** I write one JSON line per event to a trace file so any run can be replayed and failed tool calls are easy to find. The full agent is one loop: prompt, model call, parse, validate and execute the tool through a permission gate, append the observation, repeat until final or the step limit. Tracing is what turns a mysterious failure into a debuggable one.
