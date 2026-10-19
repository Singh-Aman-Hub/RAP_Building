# Terms — Step 08: Tracing and the full agent

Folder: `08_agent_harness/08_trace_and_full_agent` · Files covered: `agent.py`, `main.py`, `show_trace.py`, `WALKTHROUGH.md`, `data/memory.json`, `traces/agent_trace.jsonl`
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Trace (agent trace)](#trace-agent-trace) · [JSONL (JSON Lines)](#jsonl-json-lines) · [Event types (start, llm_reply, tool_call, end, ...)](#event-types-start-llm_reply-tool_call-end) · [Run id](#run-id) · [uuid.uuid4](#uuiduuid4) · [Status (every exit path)](#status-every-exit-path) · [Agent class](#agent-class) · [__init__ with default arguments](#__init__-with-default-arguments) · [Least-privilege tool selection](#least-privilege-tool-selection) · [Dictionary comprehension](#dictionary-comprehension) · [Registry with dict(...) entries](#registry-with-dict-entries) · [log method (append a JSON line)](#log-method-append-a-json-line) · [**fields (collecting keyword arguments) and dict merge](#fields-collecting-keyword-arguments-and-dict-merge) · [with Path.open("a")](#with-pathopena) · [json.dumps and json.loads](#jsondumps-and-jsonloads) · [time.time and duration in milliseconds](#timetime-and-duration-in-milliseconds) · [System prompt generated from the tools](#system-prompt-generated-from-the-tools) · [Validation inside the agent (validate)](#validation-inside-the-agent-validate) · [Permission gate (execute)](#permission-gate-execute) · [Confirm hook (lambda default)](#confirm-hook-lambda-default) · [Retry with exponential backoff (call_llm)](#retry-with-exponential-backoff-call_llm) · [History trimming (trim)](#history-trimming-trim) · [Local lambda (size)](#local-lambda-size) · [JSON repair and max_repairs](#json-repair-and-max_repairs) · [dict.get with defaults for the decision](#dictget-with-defaults-for-the-decision) · [Replay of a run](#replay-of-a-run) · [collections.Counter](#collectionscounter) · [List comprehension with a filter and set comprehension](#list-comprehension-with-a-filter-and-set-comprehension) · [Generator expression](#generator-expression) · [Dict comprehension with a filter](#dict-comprehension-with-a-filter) · [Debugging from evidence (first failing event)](#debugging-from-evidence-first-failing-event) · [sys.path.insert(0, HERE)](#syspathinsert0-here) · [Path.mkdir and Path.unlink for a repeatable demo](#pathmkdir-and-pathunlink-for-a-repeatable-demo) · [Task list with expected behaviours](#task-list-with-expected-behaviours) · [Fake agent LLM](#fake-agent-llm) · [memory.json and agent_trace.jsonl (artifacts)](#memoryjson-and-agent_tracejsonl-artifacts)

---

## Trace (agent trace)
*Type: concept*

**1. Definition:** A chronological record of everything an agent run did: model replies, tool calls with arguments and results, errors and the final status.
**2. In plain words:** When an agent misbehaves, the trace is your evidence. You read it to find the first failing event instead of guessing or asking an AI. Traces can hold sensitive data, so redact secrets and set a retention period.
**3. Example to quote:**
> "The trace showed the first failing event was `read_file('../../.env')` denied by the sandbox, so I knew the model, not my code, had asked for it."

**Seen in:** `agent.py`, `Agent.log`; `traces/agent_trace.jsonl`.

## JSONL (JSON Lines)
*Type: concept / data format*

**1. Definition:** A text format with one complete JSON object per line, appended to a file as events happen.
**2. In plain words:** Append-only logs are simple and crash-safe, and you can `grep` them, stream them or load them with a script. Each line stands alone, so a half-written file is still mostly readable.
**3. Example to quote:**
> "`grep '\"ok\": false' traces/agent_trace.jsonl` lists every failed tool call in one command."

```text
{"event": "start", ...}
{"event": "llm_reply", ...}
```

**Seen in:** `agent.py`, `log`; `show_trace.py`.

## Event types (`start`, `llm_reply`, `tool_call`, `end`, ...)
*Type: concept*

**1. Definition:** Labels in each trace record saying what happened: `start`, `llm_reply`, `bad_json`, `tool_call`, `llm_error`, `history_trimmed`, `end`.
**2. In plain words:** Consistent event names make the log searchable and let a summary script count and group them.
**3. Example to quote:**
> "The run produced 49 events: 10 `start`, 19 `llm_reply`, 10 `tool_call` and 10 `end`."

**Seen in:** `agent.py`, `Agent.run`; `show_trace.py`, `Counter`.

## Run id
*Type: concept*

**1. Definition:** A short unique identifier attached to every event of one agent run so that interleaved events can be grouped.
**2. In plain words:** When many runs write to one file, the run id tells you which events belong together, like a ticket number.
**3. Example to quote:**
> "Run `7be13922` is the one that tried to read `.env`; I replayed only its events."

**Seen in:** `agent.py`, `self.run_id`.

## uuid.uuid4
*Type: Python standard library*

**1. Definition:** `uuid.uuid4()` generates a random universally unique identifier; `.hex[:8]` takes the first eight hex characters as a short id.
**2. In plain words:** It gives each run a name without any coordination. Eight characters are short enough to read and unique enough for a demo.
**3. Example to quote:**
> "`uuid.uuid4().hex[:8]` produced ids like `c13bbf7b`."

```python
import uuid
uuid.uuid4().hex[:8]
```

**Seen in:** `agent.py`, `Agent.run`.

## Status (every exit path)
*Type: concept*

**1. Definition:** A value returned and logged when a run ends, one of `final`, `max_steps`, `llm_unavailable` or `bad_json`.
**2. In plain words:** Each way a run can end gets a name, so you can count and investigate them. An agent that fails silently is much harder to debug.
**3. Example to quote:**
> "With `max_steps=1` the status was `max_steps` and the answer was 'Stopped: step limit reached.'"

**Seen in:** `agent.py`, `Agent.finish`.

## Agent class
*Type: Python syntax / design*

**1. Definition:** A class bundling the agent's settings (tools, mode, limits, trace file) in `__init__` with its behaviour as methods (`run`, `execute`, `validate`, `trim`, `call_llm`, `log`, `finish`).
**2. In plain words:** `self` refers to the instance, so each agent keeps its own configuration. This is the whole framework-free harness in about 200 lines.
**3. Example to quote:**
> "`Agent(mock=llm, trace_file=TRACE, mode='read_write')` builds one agent, and `agent.run(task)` runs the loop."

**Seen in:** `agent.py`, `class Agent`; `main.py`, `agent = Agent(...)`.

## __init__ with default arguments
*Type: Python syntax*

**1. Definition:** The constructor method; parameters such as `max_steps=6` have defaults used when the caller omits them.
**2. In plain words:** Sensible defaults let you create an agent with one line and override only what you need.
**3. Example to quote:**
> "`Agent(tools=['calculator'])` limits the agent to one tool and keeps the other defaults."

**Seen in:** `agent.py`, `Agent.__init__`.

## Least-privilege tool selection
*Type: concept (security)*

**1. Definition:** `self.tools = {name: REGISTRY[name] for name in (tools or REGISTRY)}` builds the agent's tool set from a list of allowed names, defaulting to all.
**2. In plain words:** If you pass only `['calculator']`, any request for another tool becomes 'unknown tool'. The agent cannot use what it was not given.
**3. Example to quote:**
> "`Agent(tools=['calculator'])` then 'read file notes.txt' returns an unknown-tool error."

**Seen in:** `agent.py`, `__init__`; `main.py` BREAK IT comment.

## Dictionary comprehension
*Type: Python syntax*

**1. Definition:** `{k: v for k in iterable}` builds a dict from a loop in one expression.
**2. In plain words:** It filters the registry down to the permitted tools.
**3. Example to quote:**
> "`{name: REGISTRY[name] for name in tools}` copies only the allowed tool entries."

```python
{x: x * 2 for x in [1, 2]}   # {1: 2, 2: 4}
```

**Seen in:** `agent.py`, `__init__`.

## Registry with `dict(...)` entries
*Type: Python syntax / concept*

**1. Definition:** Each tool is described as `dict(fn=..., params=..., read_only=..., confirm=..., allowed=...)`, the keyword form of creating a dictionary.
**2. In plain words:** One entry holds the function and every rule about it, so validation, permissions and the system prompt all read the same data.
**3. Example to quote:**
> "`delete_file` has `read_only=False, confirm=True`, so it is blocked in read-only mode and needs approval otherwise."

**Seen in:** `agent.py`, `REGISTRY`.

## log method (append a JSON line)
*Type: project function*

**1. Definition:** `Agent.log(event, **fields)` builds a record with timestamp, run id, event name and extra fields and appends it as one JSON line to the trace file.
**2. In plain words:** `**fields` collects any keyword arguments, so each event can carry different details. If no trace file is set, it does nothing.
**3. Example to quote:**
> "`self.log('tool_call', step=step, tool=name, args=args, result=result, ok=..., ms=...)` records everything about one tool call."

**Seen in:** `agent.py`, `Agent.log`.

## `**fields` (collecting keyword arguments) and dict merge
*Type: Python syntax*

**1. Definition:** In a function definition `**fields` gathers extra keyword arguments into a dict; in a dict literal `{**a, "k": v}` merges `a` into the new dict.
**2. In plain words:** It lets one `log` function accept any event details and place them all in the record.
**3. Example to quote:**
> "`record = {'ts': ..., 'event': event, **fields}` puts every extra field into the JSON line."

```python
{**{"a": 1}, "b": 2}   # {'a': 1, 'b': 2}
```

**Seen in:** `agent.py`, `Agent.log`.

## with Path.open("a")
*Type: Python syntax / standard library*

**1. Definition:** `with path.open("a") as f:` opens a file in append mode and guarantees it is closed when the block ends, even after an error.
**2. In plain words:** Append mode adds to the end rather than overwriting, which is what a log needs. The `with` statement saves you from forgetting to close the file.
**3. Example to quote:**
> "Each event opens the file, writes one line with `f.write(json.dumps(record) + '\\n')`, and closes it."

**Seen in:** `agent.py`, `Agent.log`.

## json.dumps and json.loads
*Type: Python standard library*

**1. Definition:** `dumps` turns a Python object into JSON text; `loads` parses JSON text into Python objects.
**2. In plain words:** Writing uses `dumps` per event; the summary script reads each line back with `loads`.
**3. Example to quote:**
> "`json.loads(line)` on one trace line gives a dict with `event`, `run_id`, `ts` and the other fields."

**Seen in:** `agent.py`, `log`; `show_trace.py`.

## time.time and duration in milliseconds
*Type: Python standard library*

**1. Definition:** `time.time()` returns the current time in seconds since 1970; subtracting two readings gives elapsed seconds, and multiplying by 1000 gives milliseconds.
**2. In plain words:** Logging how long each model reply and tool call took shows where time goes.
**3. Example to quote:**
> "`round((time.time() - started) * 1000)` is the `ms` value stored with each `llm_reply`."

**Seen in:** `agent.py`, `Agent.run`.

## System prompt generated from the tools
*Type: concept*

**1. Definition:** `system_prompt()` builds the instruction text, including the tool names and their parameters, from the agent's current tool set.
**2. In plain words:** If you give the agent fewer tools, the prompt lists fewer tools, so what the model is told always matches what is allowed.
**3. Example to quote:**
> "With `tools=['calculator']` the prompt lists only `calculator(a, op, b)`."

**Seen in:** `agent.py`, `Agent.system_prompt`.

## Validation inside the agent (`validate`)
*Type: project function*

**1. Definition:** A method returning a list of problems for a proposed call: unknown tool, args not an object, missing or unexpected arguments, wrong types (including booleans), values outside allowed lists.
**2. In plain words:** It is the step-03 validation merged into the class. The model proposes; this decides.
**3. Example to quote:**
> "A call with `a: 'six'` returns `argument 'a' has wrong type str` and never reaches the calculator."

**Seen in:** `agent.py`, `Agent.validate`.

## Permission gate (`execute`)
*Type: project function*

**1. Definition:** The single method every call passes through: validate, check read-only mode, require confirmation for risky tools, run inside `try/except`, always return a string.
**2. In plain words:** It combines steps 03, 06 and 07, so there is exactly one place to audit.
**3. Example to quote:**
> "`delete file notes.txt` returned 'permission denied: delete_file needs confirmation and was not approved' because the default confirm hook says no."

**Seen in:** `agent.py`, `Agent.execute`.

## Confirm hook (lambda default)
*Type: Python syntax / concept*

**1. Definition:** The `confirm` parameter defaults to `lambda name, args: False`, a tiny function that denies every risky action unless you pass a different one.
**2. In plain words:** Safe by default: unattended agents do not delete anything. To allow it you pass `confirm=lambda n, a: True`.
**3. Example to quote:**
> "I passed `confirm=lambda n, a: True` for a test and the delete went through."

**Seen in:** `agent.py`, `__init__`; `main.py` TRY IT comment.

## Retry with exponential backoff (`call_llm`)
*Type: project function*

**1. Definition:** Calls the model up to `attempts` times, catching `TimeoutError` and `ConnectionError`, waiting `base_delay * 2 ** (attempt - 1)` plus jitter, and logging each failure as `llm_error`.
**2. In plain words:** It is the step-06 retry logic inside the class, with the failures recorded in the trace.
**3. Example to quote:**
> "Each timeout writes an `llm_error` event with the attempt number, so I can see how often the model was flaky."

**Seen in:** `agent.py`, `Agent.call_llm`.

## History trimming (`trim`)
*Type: project function*

**1. Definition:** Keeps the first message and the newest messages that fit `token_budget`, drops a leading assistant message, and logs a `history_trimmed` event with before and after counts.
**2. In plain words:** It is the step-04 truncation applied automatically before each model call.
**3. Example to quote:**
> "A `history_trimmed` event with, for example, `before=40 after=11` tells me exactly when the agent started forgetting."

**Seen in:** `agent.py`, `Agent.trim`.

## Local lambda (`size`)
*Type: Python syntax*

**1. Definition:** `size = lambda ms: sum(len(m["content"]) for m in ms) // 4` defines a small function in one line and names it.
**2. In plain words:** It estimates tokens as characters divided by 4 for any message list.
**3. Example to quote:**
> "`size(messages) <= self.token_budget` decides whether trimming is needed."

**Seen in:** `agent.py`, `Agent.trim`.

## JSON repair and `max_repairs`
*Type: concept*

**1. Definition:** When the reply has no JSON the agent appends an error observation and asks again, up to `max_repairs`, then ends with status `bad_json`.
**2. In plain words:** One malformed reply should not kill the run, but an endless complaining loop should not continue either.
**3. Example to quote:**
> "After two failed repairs the agent returns 'Sorry, I could not produce a valid action.'"

**Seen in:** `agent.py`, `Agent.run`.

## dict.get with defaults for the decision
*Type: Python built-in*

**1. Definition:** `decision.get("tool")` and `decision.get("args", {})` read keys without raising `KeyError`, supplying `{}` when `args` is missing.
**2. In plain words:** A model that omits `args` produces a validation error instead of a crash.
**3. Example to quote:**
> "If the model replies `{'tool': 'calculator'}` with no args, validation reports the missing arguments."

**Seen in:** `agent.py`, `Agent.run`.

## Replay of a run
*Type: concept*

**1. Definition:** Printing all events that share one run id in order, to see the full story of that run.
**2. In plain words:** Replay turns a mixed log into a readable narrative: what the model said, which tool ran, what came back.
**3. Example to quote:**
> "I replayed the first run: start, llm_reply, tool_call, llm_reply, end."

**Seen in:** `show_trace.py`, last section.

## collections.Counter
*Type: Python standard library*

**1. Definition:** A dict subclass that counts hashable items.
**2. In plain words:** The summary uses it to show how many events of each type happened.
**3. Example to quote:**
> "`Counter(e['event'] for e in events)` printed `{'start': 10, 'llm_reply': 19, 'tool_call': 10, 'end': 10}`."

**Seen in:** `show_trace.py`.

## List comprehension with a filter and set comprehension
*Type: Python syntax*

**1. Definition:** `[json.loads(line) for line in lines if line.strip()]` parses only non-empty lines; `{e['run_id'] for e in events}` builds a set of unique run ids.
**2. In plain words:** The first line reads the whole log in one expression; the set counts distinct runs.
**3. Example to quote:**
> "`len({e['run_id'] for e in events})` is the number of runs in the file."

**Seen in:** `show_trace.py`, top section.

## Generator expression
*Type: Python syntax*

**1. Definition:** `(x for x in events if x["run_id"] == first_run)` produces matching items one at a time without building a list.
**2. In plain words:** It is used in the replay loop to walk only that run's events.
**3. Example to quote:**
> "The replay loop iterates a generator expression filtered by run id."

**Seen in:** `show_trace.py`, replay loop.

## Dict comprehension with a filter
*Type: Python syntax*

**1. Definition:** `{k: v for k, v in e.items() if k not in ("ts", "run_id", "event")}` builds a dict keeping only selected keys.
**2. In plain words:** It strips the fields every event shares so only the interesting details are printed.
**3. Example to quote:**
> "The replay prints just `step`, `tool`, `args` and `result` for each event."

**Seen in:** `show_trace.py`.

## Debugging from evidence (first failing event)
*Type: concept*

**1. Definition:** Using the trace to locate the earliest event that went wrong, then reproducing it with the smallest possible input.
**2. In plain words:** Later failures are often just consequences of an earlier one. Fix the first.
**3. Example to quote:**
> "The failed tool call list shows three failures and each was expected: a sandbox block, a zero division and a denied delete."

**Seen in:** `show_trace.py`, failed tool calls section; `WALKTHROUGH.md`.

## sys.path.insert(0, HERE)
*Type: Python standard library*

**1. Definition:** Adds the script's own folder to Python's import search list so `import agent` finds `agent.py` regardless of the working directory.
**2. In plain words:** Without it, running from another folder could raise `ModuleNotFoundError`.
**3. Example to quote:**
> "`sys.path.insert(0, str(HERE))` lets `from agent import Agent` work from the repo root too."

**Seen in:** `main.py`, import setup.

## Path.mkdir and Path.unlink for a repeatable demo
*Type: Python standard library*

**1. Definition:** `TRACE.parent.mkdir(exist_ok=True)` creates the folder if needed and `unlink(missing_ok=True)` deletes old files without failing if they do not exist.
**2. In plain words:** Starting from a clean state each run makes the output reproducible.
**3. Example to quote:**
> "I delete the old trace, memory and `out.txt` at the top so every demo run starts fresh."

**Seen in:** `main.py`, setup lines.

## Task list with expected behaviours
*Type: concept*

**1. Definition:** A list of nine questions that each exercise a different path: arithmetic, memory write and read, file read, sandbox block, tool error, denied confirmation, no tool.
**2. In plain words:** A varied set of tasks shows each behaviour in the trace, so the trace becomes a test report.
**3. Example to quote:**
> "'Tell me a joke' finishes in one step because the model has no tool for it."

**Seen in:** `main.py`, `TASKS`.

## Fake agent LLM
*Type: project helper*

**1. Definition:** `make_fake_agent_llm()` returns the scripted mock used for all tasks.
**2. In plain words:** It makes the nine-task demo deterministic. With `LLM_BACKEND=anthropic` the same agent runs against a real model.
**3. Example to quote:**
> "I pass the same fake model to two agents, one normal and one with `max_steps=1`."

**Seen in:** `main.py`, `llm = make_fake_agent_llm()`.

## memory.json and agent_trace.jsonl (artifacts)
*Type: data files*

**1. Definition:** `data/memory.json` stores the agent's long-term memory (here `{"name": "Priya"}`) and `traces/agent_trace.jsonl` stores the event log; both are regenerated on each run.
**2. In plain words:** They are outputs, not source code. Delete them freely; the demo recreates them.
**3. Example to quote:**
> "After the demo, `memory.json` holds Priya's name and the trace holds 49 lines."

**Seen in:** `data/memory.json`, `traces/agent_trace.jsonl`.
