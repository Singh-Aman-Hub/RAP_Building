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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Trace | log, audit trail, telemetry, span log | Near-same | A trace records ordered events of one run (with a `run_id`); a plain log may be unstructured. |
| JSONL | JSON Lines, NDJSON | Yes | One JSON object per line, easy to append and grep. |
| Observability | monitoring, tracing, LLMOps tooling (LangSmith, Langfuse, OpenTelemetry) | Related | Same idea at product scale; our file is the minimal version. |
| Agent | assistant, bot, harness + model | Not exactly | An agent is the loop around the model plus tools, not the model itself. |
| max_steps | step limit, iteration cap, recursion limit | Same | A hard stop on loop iterations. |

## Code, step by step

### 1. Imports and make `agent.py` importable  (`main.py` lines 7–13)

```python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                              # so `import agent` works from any cwd
from agent import Agent, MEMORY_FILE, SANDBOX
from fake_llm import make_fake_agent_llm
```

- **In simple words:** Add this folder to the search path and import the agent and the scripted model.
- **Exactly:** `Agent` is the class from `agent.py`; `MEMORY_FILE` and `SANDBOX` are its path constants; `make_fake_agent_llm` returns the mock brain (importable because `agent.py` puts `08_agent_harness` on the path).
- **Why it is here:** Lets the script run from any folder.

### 2. Reset the trace, memory and output file  (`main.py` lines 15–19)

```python
TRACE = HERE / "traces" / "agent_trace.jsonl"
TRACE.parent.mkdir(exist_ok=True)
TRACE.unlink(missing_ok=True)                              # fresh trace for a repeatable demo
MEMORY_FILE.unlink(missing_ok=True)
(SANDBOX / "out.txt").unlink(missing_ok=True)
```

- **In simple words:** Start the demo from a clean state.
- **Exactly:** `TRACE = HERE/"traces"/"agent_trace.jsonl"`; `mkdir(exist_ok=True)` creates `traces/`; `unlink(missing_ok=True)` deletes old files if present.
- **Why it is here:** A stale trace would mix old and new runs and break repeatability.

### 3. Create the model and the agent  (`main.py` lines 21–22)

```python
llm = make_fake_agent_llm()
agent = Agent(mock=llm, trace_file=TRACE, mode="read_write")
```

- **In simple words:** Build the fake brain and an agent that writes a trace.
- **Exactly:** `Agent(mock=llm, trace_file=TRACE, mode="read_write")`: all six tools allowed, write tools enabled, default `max_steps=6`, confirmation hook denies everything.
- **Why it is here:** `read_write` is needed so remember/write tools can run.

### 4. The nine tasks  (`main.py` lines 24–34)

```python
TASKS = [
    "What is 6 x 7?",
    "Calculate 12 + 8 then * 3",                           # two tool calls in one run
    "Remember that my name is Priya",                      # writes long-term memory
    "What is my name?",                                    # reads it back
    "read file notes.txt",
    "read file ../../.env",                                # blocked by the sandbox
    "What is 5 / 0?",                                      # tool error becomes an observation
    "delete file notes.txt",                               # needs confirmation -> denied by default
    "Tell me a joke",                                      # model has no tool for this
]
```

- **In simple words:** A list of questions covering each behaviour.
- **Exactly:** Covers two-tool chaining, memory write/read, file read, sandbox escape, divide by zero, confirmation-needed delete, and a question with no tool.
- **Why it is here:** One run shows every code path in the agent.

### 5. Run each task and print a summary line  (`main.py` lines 35–37)

```python
for task in TASKS:
    result = agent.run(task)
    print(f"{result['status']:10s} steps={result['steps']}  {task!r:36s} -> {result['answer']}")
```

- **In simple words:** Call `agent.run(task)` and print status, steps and answer.
- **Exactly:** `run` returns a dict `{status, answer, steps, run_id}`. Real: `final steps=2 'What is 6 x 7?' -> Result: 42`; the `../../.env` task answers `Sorry, the tool failed: ERROR: permission denied: path '../../.env' is outside the sandbox`; delete -> `... needs confirmation and was not approved`; joke -> `steps=1`.
- **Why it is here:** Shows each outcome as data, not an exception.

### 6. Show max_steps in action  (`main.py` lines 39–41)

```python
# max_steps demonstration: 1 step is not enough for a two-tool question
tiny = Agent(mock=llm, trace_file=TRACE, max_steps=1)
print("\nmax_steps=1:", tiny.run("Calculate 12 + 8 then * 3"))
```

- **In simple words:** A second agent allowed only one step cannot finish a two-tool job.
- **Exactly:** Real: `{'status': 'max_steps', 'answer': 'Stopped: step limit reached.', 'steps': 1, 'run_id': ...}`. It shares the same trace file.
- **Why it is here:** Demonstrates the budget stop and that it is logged.

### 7. Tell the user where the trace is  (`main.py` lines 42–42)

```python
print(f"\nTrace written to {TRACE.relative_to(HERE.parents[1])}  -> now run: python show_trace.py")
```

- **In simple words:** Print the relative path of the trace.
- **Exactly:** `Path.relative_to(HERE.parents[1])` makes the path relative to the repo root.
- **Why it is here:** Points you at the evidence for debugging.

### 8. Experiment comments  (`main.py` lines 44–46)

```python
# TRY IT: change mode to "read_only" and re-run: which tasks change?   TRY IT: pass confirm=lambda n, a: True.
# TRY IT: open traces/agent_trace.jsonl in an editor - one JSON object per line.
# BREAK IT: Agent(tools=["calculator"]) then ask "read file notes.txt" -> 'unknown tool' (least privilege).
```

- **In simple words:** Ideas to try.
- **Exactly:** Comments only; restricting `tools=["calculator"]` makes `read_file` unknown.
- **Why it is here:** Shows least privilege.

### 9. Imports, paths, constants  (`agent.py` lines 15–28)

```python
import json
import random
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json

SANDBOX = (HERE.parent / "sandbox_files").resolve()
MEMORY_FILE = HERE / "data" / "memory.json"
```

- **In simple words:** Standard library tools plus the shared LLM helper; two paths the tools use.
- **Exactly:** `uuid` makes run ids, `random` jitter, `time` timestamps and sleeps. `SANDBOX` is the only folder files may touch; `MEMORY_FILE` is `data/memory.json` in this folder.
- **Why it is here:** One file holds the whole harness, so every dependency is visible.

### 10. Permission error and safe path check  (`agent.py` lines 30–40)

```python
# ============================== TOOLS =======================================================
class PermissionDenied(Exception):
    pass

def safe_path(user_path):
    p = (SANDBOX / user_path).resolve()
    if not p.is_relative_to(SANDBOX):
        raise PermissionDenied(f"path {user_path!r} is outside the sandbox")
    if p.suffix not in {".txt", ".md"}:
        raise PermissionDenied(f"file type {p.suffix!r} not allowed")
    return p
```

- **In simple words:** Our own error type and the path guard from step 07.
- **Exactly:** Resolves `SANDBOX / user_path`; rejects paths outside the sandbox or suffixes other than `.txt`/`.md`.
- **Why it is here:** The model controls the path, so it must be treated as hostile input.

### 11. Calculator tool  (`agent.py` lines 42–44)

```python
def calculator(a, op, b):
    ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(ops[op](a, b))
```

- **In simple words:** Arithmetic with a dict of four operations.
- **Exactly:** Returns `str(result)`; may raise `KeyError`/`ZeroDivisionError`, which `execute` catches.
- **Why it is here:** No `eval`.

### 12. Memory tools  (`agent.py` lines 46–56)

```python
def _load():
    return json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else {}

def remember(key, value):
    mem = _load(); mem[key] = str(value)[:200]
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(mem, indent=2))
    return f"stored {key!r}"

def recall(key):
    return _load().get(key, f"nothing stored for {key!r}")
```

- **In simple words:** Load, store and look up facts in the JSON file.
- **Exactly:** `_load` returns `{}` if no file. `remember` slices to 200 characters, creates the folder, writes indented JSON, returns `stored 'name'`. `recall` uses `.get(key, default)`.
- **Why it is here:** Gives the agent persistence between runs.

### 13. File tools  (`agent.py` lines 58–68)

```python
def read_file(path):
    text = safe_path(path).read_text()
    return text[:500] + ("...[truncated]" if len(text) > 500 else "")

def write_file(path, content):
    safe_path(path).write_text(content)
    return f"wrote {len(content)} characters to {path}"

def delete_file(path):
    safe_path(path).unlink()
    return f"deleted {path}"
```

- **In simple words:** Read (capped at 500 chars), write and delete inside the sandbox.
- **Exactly:** Every function calls `safe_path` first. `read_file` appends `...[truncated]` when longer than 500 characters.
- **Why it is here:** Cap protects the context window; `safe_path` protects the disk.

### 14. The registry  (`agent.py` lines 70–78)

```python
NUMBER = (int, float)
REGISTRY = {  # fn, params {name: type(s)}, read_only, needs_confirmation, optional allowed values
    "calculator":  dict(fn=calculator,  params={"a": NUMBER, "op": str, "b": NUMBER}, read_only=True,  confirm=False, allowed={"op": ["+", "-", "*", "/"]}),
    "remember":    dict(fn=remember,    params={"key": str, "value": str},            read_only=False, confirm=False, allowed={}),
    "recall":      dict(fn=recall,      params={"key": str},                          read_only=True,  confirm=False, allowed={}),
    "read_file":   dict(fn=read_file,   params={"path": str},                         read_only=True,  confirm=False, allowed={}),
    "write_file":  dict(fn=write_file,  params={"path": str, "content": str},         read_only=False, confirm=False, allowed={}),
    "delete_file": dict(fn=delete_file, params={"path": str},                         read_only=False, confirm=True,  allowed={}),
}
```

- **In simple words:** One dict describing every tool: function, parameter types, risk flags, allowed values.
- **Exactly:** Six entries. `read_only` False for remember/write/delete; `confirm` True only for `delete_file`. `NUMBER = (int, float)` is a tuple usable by `isinstance`.
- **Why it is here:** Validation, permissions and the prompt all read this table.

### 15. Retryable errors  (`agent.py` lines 80–81)

```python
# ============================== THE AGENT ====================================================
RETRYABLE = (TimeoutError, ConnectionError)
```

- **In simple words:** Which exceptions count as transient.
- **Exactly:** `(TimeoutError, ConnectionError)`.
- **Why it is here:** Only these are retried.

### 16. Agent constructor  (`agent.py` lines 83–91)

```python
class Agent:
    def __init__(self, mock=None, tools=None, mode="read_only", confirm=lambda name, args: False,
                 max_steps=6, max_repairs=2, token_budget=2000, trace_file=None):
        self.mock = mock
        self.tools = {name: REGISTRY[name] for name in (tools or REGISTRY)}   # least privilege: only what's listed
        self.mode, self.confirm = mode, confirm
        self.max_steps, self.max_repairs, self.token_budget = max_steps, max_repairs, token_budget
        self.trace_file = Path(trace_file) if trace_file else None
        self.run_id = None
```

- **In simple words:** Store settings. The default is safe: read-only mode and deny-all confirmation.
- **Exactly:** `self.tools` is a dict comprehension over `tools or REGISTRY` (least privilege). Defaults: `max_steps=6`, `max_repairs=2`, `token_budget=2000` (tokens, estimated chars/4). `trace_file` is a `Path` or `None`.
- **Why it is here:** Safe defaults mean a forgotten option cannot grant power.

### 17. Trace logger  (`agent.py` lines 93–100)

```python
    # ---- trace: one JSON object per line (JSONL) -> easy to grep, load with pandas, replay ----
    def log(self, event, **fields):
        if not self.trace_file:
            return
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": event, **fields}
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_file.open("a") as f:          # "a" = append; `with` closes the file for us
            f.write(json.dumps(record) + "\n")
```

- **In simple words:** Append one JSON line per event.
- **Exactly:** Record `{"ts", "run_id", "event", **fields}` written with `open("a")` as `json.dumps(record) + "\n"`. Does nothing when no trace file.
- **Why it is here:** Append-only lines survive a crash and are easy to grep.

### 18. Build the system prompt from the tools  (`agent.py` lines 102–105)

```python
    def system_prompt(self):
        lines = [f'  {n}({", ".join(s["params"])})' for n, s in self.tools.items()]
        return ("You are an assistant with tools. Reply with ONLY one JSON object:\n"
                '  {"tool": "<name>", "args": {...}}  or  {"final": "<answer>"}\nTools:\n' + "\n".join(lines))
```

- **In simple words:** Generate the instructions and tool list.
- **Exactly:** Lines like `  calculator(a, op, b)`; asks for exactly one JSON object `{"tool":..., "args":...}` or `{"final":...}`.
- **Why it is here:** A prompt generated from the registry cannot list a tool that does not exist.

### 19. Validate a tool call  (`agent.py` lines 107–122)

```python
    # ---- step 03: validation ----
    def validate(self, name, args):
        spec = self.tools.get(name)
        if spec is None:
            return [f"unknown tool {name!r}; available: {sorted(self.tools)}"]
        if not isinstance(args, dict):
            return ["args must be an object"]
        problems = [f"missing argument {p!r}" for p in spec["params"] if p not in args]
        for k, v in args.items():
            if k not in spec["params"]:
                problems.append(f"unexpected argument {k!r}")
            elif isinstance(v, bool) or not isinstance(v, spec["params"][k]):
                problems.append(f"argument {k!r} has wrong type {type(v).__name__}")
            elif k in spec["allowed"] and v not in spec["allowed"][k]:
                problems.append(f"argument {k!r}={v!r} not in {spec['allowed'][k]}")
        return problems
```

- **In simple words:** Check name, args type, required, unexpected, types and allowed values.
- **Exactly:** Returns a list of problem strings (empty = OK); same logic as step 03 with bool rejected for numbers.
- **Why it is here:** Models make mistakes; the harness must catch them before the tool runs.

### 20. The gate every tool call passes  (`agent.py` lines 124–139)

```python
    # ---- steps 03 + 06 + 07: the single gate every tool call passes through ----
    def execute(self, name, args):
        problems = self.validate(name, args)
        if problems:
            return "ERROR: " + "; ".join(problems)
        spec = self.tools[name]
        if self.mode == "read_only" and not spec["read_only"]:
            return f"ERROR: permission denied: {name} is a write tool and mode is read_only"
        if spec["confirm"] and not self.confirm(name, args):
            return f"ERROR: permission denied: {name} needs confirmation and was not approved"
        try:
            return spec["fn"](**args)
        except PermissionDenied as err:
            return f"ERROR: permission denied: {err}"
        except Exception as err:
            return f"ERROR: {type(err).__name__}: {err}"
```

- **In simple words:** Validate, check mode, check confirmation, run, convert errors to text.
- **Exactly:** Always returns a string. Real: `ERROR: permission denied: delete_file needs confirmation and was not approved`; `ERROR: ZeroDivisionError: division by zero`.
- **Why it is here:** One choke point for safety and reliability.

### 21. Model call with retry and backoff  (`agent.py` lines 141–150)

```python
    # ---- step 06: retry with backoff ----
    def call_llm(self, messages, attempts=4, base_delay=0.05):
        for attempt in range(1, attempts + 1):
            try:
                return generate(messages, system=self.system_prompt(), mock=self.mock)
            except RETRYABLE as err:
                self.log("llm_error", attempt=attempt, error=type(err).__name__)
                if attempt == attempts:
                    raise
                time.sleep(base_delay * 2 ** (attempt - 1) + random.uniform(0, base_delay))
```

- **In simple words:** Try up to 4 times with growing waits.
- **Exactly:** Delay `base_delay * 2**(attempt-1) + random.uniform(0, base_delay)`; each failure logs `llm_error`; last failure re-raises.
- **Why it is here:** Network calls fail; this keeps a hiccup from killing the run.

### 22. Trim history to the budget  (`agent.py` lines 152–165)

```python
    # ---- step 04: keep history inside the budget (keep first message + newest) ----
    def trim(self, messages):
        size = lambda ms: sum(len(m["content"]) for m in ms) // 4     # ~tokens
        if size(messages) <= self.token_budget:
            return messages
        first, kept = messages[0], []
        for m in reversed(messages[1:]):
            if size([first] + [m] + kept) > self.token_budget:
                break
            kept.insert(0, m)
        while kept and kept[0]["role"] == "assistant":
            kept.pop(0)
        self.log("history_trimmed", before=len(messages), after=1 + len(kept))
        return [first] + kept
```

- **In simple words:** If the message list is too long, keep the first message and as many of the newest as fit.
- **Exactly:** `size` is a lambda estimating tokens (chars//4). Pinned first message; reversed walk; leading assistant messages are removed; logs `history_trimmed` with before/after counts.
- **Why it is here:** Prevents exceeding the context window.

### 23. Start a run  (`agent.py` lines 167–172)

```python
    # ---- step 02: the loop ----
    def run(self, question):
        self.run_id = uuid.uuid4().hex[:8]
        self.log("start", question=question, mode=self.mode, tools=sorted(self.tools))
        messages, repairs = [{"role": "user", "content": question}], 0
        for step in range(1, self.max_steps + 1):
```

- **In simple words:** Create a run id and log the start.
- **Exactly:** `uuid.uuid4().hex[:8]` e.g. `d9386d02`. Logs question, mode and sorted tool names. Messages start with the question; `repairs = 0`.
- **Why it is here:** The run id ties all events of one run together.

### 24. Call the model for this step  (`agent.py` lines 173–178)

```python
            started = time.time()
            try:
                reply = self.call_llm(self.trim(messages))        # prompt -> LLM call
            except RETRYABLE:
                return self.finish("llm_unavailable", "Sorry, the model is unavailable right now.", step)
            self.log("llm_reply", step=step, reply=reply, ms=round((time.time() - started) * 1000))
```

- **In simple words:** Time it, call with retry, handle an outage.
- **Exactly:** `self.call_llm(self.trim(messages))`. A `RETRYABLE` error after all attempts returns status `llm_unavailable`. Logs `llm_reply` with `ms`.
- **Why it is here:** The ms field shows slow steps; the clean return avoids a crash.

### 25. Parse, or ask the model to repair  (`agent.py` lines 179–189)

```python
            try:
                decision = extract_json(reply)                    # parse tool call
            except ValueError:
                repairs += 1
                self.log("bad_json", step=step, repairs=repairs)
                if repairs > self.max_repairs:
                    return self.finish("bad_json", "Sorry, I could not produce a valid action.", step)
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": "Observation: ERROR: reply was not valid JSON. "
                                                         "Use {\"tool\":..,\"args\":..} or {\"final\":..}."}]
                continue
```

- **In simple words:** If the reply is not JSON, count a repair and re-ask; give up after `max_repairs`.
- **Exactly:** `extract_json` raises `ValueError`; logs `bad_json`; status `bad_json` on exhaustion; otherwise adds an error Observation and `continue`s.
- **Why it is here:** Chatty replies are common with real models.

### 26. Record reply and stop on final  (`agent.py` lines 190–192)

```python
            messages.append({"role": "assistant", "content": reply})
            if "final" in decision:
                return self.finish("final", decision["final"], step)
```

- **In simple words:** Save the assistant turn; return if the model says final.
- **Exactly:** `finish("final", ...)` logs the `end` event and returns the result dict.
- **Why it is here:** Normal exit path.

### 27. Run the tool, log it, feed back the result  (`agent.py` lines 193–198)

```python
            name, args = decision.get("tool"), decision.get("args", {})
            t0 = time.time()
            result = self.execute(name, args)                     # execute tool (gated)
            self.log("tool_call", step=step, tool=name, args=args, result=result,
                     ok=not result.startswith("ERROR"), ms=round((time.time() - t0) * 1000, 2))
            messages.append({"role": "user", "content": f"Observation: {result}"})   # return observation, iterate
```

- **In simple words:** Execute the gated tool, log details and append an Observation.
- **Exactly:** `decision.get("tool")`, `decision.get("args", {})` tolerate missing keys. Log has `ok=not result.startswith("ERROR")` and tool time in ms.
- **Why it is here:** `ok` makes failed calls easy to filter in the trace.

### 28. Step limit and the finish helper  (`agent.py` lines 199–203)

```python
        return self.finish("max_steps", "Stopped: step limit reached.", self.max_steps)

    def finish(self, status, answer, steps):
        self.log("end", status=status, answer=answer, steps=steps)
        return {"status": status, "answer": answer, "steps": steps, "run_id": self.run_id}
```

- **In simple words:** If the loop ends without a final answer, report a stop; `finish` logs and returns the result.
- **Exactly:** Returns `{"status","answer","steps","run_id"}`; status `max_steps` when the budget is used up.
- **Why it is here:** Every exit path is logged exactly once.

### 29. Imports and load the trace  (`show_trace.py` lines 6–13)

```python
import json
from collections import Counter
from pathlib import Path

TRACE = Path(__file__).resolve().parent / "traces" / "agent_trace.jsonl"

# one line = one JSON object; a list comprehension parses every non-empty line
events = [json.loads(line) for line in TRACE.read_text().splitlines() if line.strip()]
```

- **In simple words:** Read the JSONL file into a list of dicts.
- **Exactly:** `splitlines()` then `json.loads` per non-empty line. Real run: `49 events, 10 runs` (after one `main.py` run).
- **Why it is here:** The trace is just data; no special tool needed.

### 30. Count events  (`show_trace.py` lines 14–16)

```python
print(f"{len(events)} events, {len({e['run_id'] for e in events})} runs\n")

print("event counts:", dict(Counter(e["event"] for e in events)))
```

- **In simple words:** Total runs and how many of each event type.
- **Exactly:** Set comprehension counts distinct `run_id`s; `Counter` gives `{'start': 10, 'llm_reply': 19, 'tool_call': 10, 'end': 10}`.
- **Why it is here:** A quick health check: starts and ends should match.

### 31. List each run's status  (`show_trace.py` lines 18–22)

```python
print("\nrun_id   status       steps  question")
questions = {e["run_id"]: e["question"] for e in events if e["event"] == "start"}
for e in events:
    if e["event"] == "end":
        print(f"{e['run_id']} {e['status']:12s} {e['steps']:<5d}  {questions[e['run_id']]}")
```

- **In simple words:** A table of run id, status, steps and question.
- **Exactly:** `questions` maps `run_id` to the question from `start` events; each `end` event prints one row, e.g. `c2881f19 max_steps 1 Calculate 12 + 8 then * 3`.
- **Why it is here:** Spot failed or stopped runs at a glance.

### 32. Show failed tool calls  (`show_trace.py` lines 24–27)

```python
print("\nFailed tool calls (the first place to look when an agent misbehaves):")
for e in events:
    if e["event"] == "tool_call" and not e["ok"]:
        print(f"  [{e['run_id']}] step {e['step']}: {e['tool']}({e['args']}) -> {e['result']}")
```

- **In simple words:** Print tool calls that returned errors.
- **Exactly:** Real: three failures (path escape, divide by zero, delete not approved).
- **Why it is here:** The first place to look when debugging.

### 33. Replay the first run  (`show_trace.py` lines 29–34)

```python
# Replay one run in order: this is the full story of what the agent did.
first_run = events[0]["run_id"]
print(f"\nReplay of run {first_run}:")
for e in (x for x in events if x["run_id"] == first_run):    # generator expression
    detail = {k: v for k, v in e.items() if k not in ("ts", "run_id", "event")}
    print(f"  {e['event']:10s} {detail}")
```

- **In simple words:** Print every event of one run in order.
- **Exactly:** Generator expression filters by `run_id`; dict comprehension drops `ts`, `run_id`, `event`. Real run shows `start`, `llm_reply`, `tool_call` (`result: '42'`), `llm_reply`, `end`.
- **Why it is here:** The replay is the story of exactly what the agent did.

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
