"""
agent.py -- the FULL framework-free agent harness. Every idea from steps 01-07 in one readable file:

  step 01 LLM call + message list      step 05 long-term memory tools (JSON file)
  step 02 the loop + stop conditions   step 06 retries/backoff, JSON repair, errors as observations
  step 03 registry + validation        step 07 permissions, sandbox, confirmation
  step 04 history trimming             step 08 (new) JSONL TRACE of everything that happens

THE LOOP (method `run`, below), in one line:
  Prompt -> LLM call -> parse tool call -> execute tool -> return observation -> iterate until final answer
  (everything else in this file - validation, retries, trimming, permissions, trace - guards one of those arrows).

Read it top to bottom. It is ~200 lines because there is no framework hiding anything.
"""
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

def calculator(a, op, b):
    ops = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: x / y}
    return str(ops[op](a, b))

def _load():
    return json.loads(MEMORY_FILE.read_text()) if MEMORY_FILE.exists() else {}

def remember(key, value):
    mem = _load(); mem[key] = str(value)[:200]
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(mem, indent=2))
    return f"stored {key!r}"

def recall(key):
    return _load().get(key, f"nothing stored for {key!r}")

def read_file(path):
    text = safe_path(path).read_text()
    return text[:500] + ("...[truncated]" if len(text) > 500 else "")

def write_file(path, content):
    safe_path(path).write_text(content)
    return f"wrote {len(content)} characters to {path}"

def delete_file(path):
    safe_path(path).unlink()
    return f"deleted {path}"

NUMBER = (int, float)
REGISTRY = {  # fn, params {name: type(s)}, read_only, needs_confirmation, optional allowed values
    "calculator":  dict(fn=calculator,  params={"a": NUMBER, "op": str, "b": NUMBER}, read_only=True,  confirm=False, allowed={"op": ["+", "-", "*", "/"]}),
    "remember":    dict(fn=remember,    params={"key": str, "value": str},            read_only=False, confirm=False, allowed={}),
    "recall":      dict(fn=recall,      params={"key": str},                          read_only=True,  confirm=False, allowed={}),
    "read_file":   dict(fn=read_file,   params={"path": str},                         read_only=True,  confirm=False, allowed={}),
    "write_file":  dict(fn=write_file,  params={"path": str, "content": str},         read_only=False, confirm=False, allowed={}),
    "delete_file": dict(fn=delete_file, params={"path": str},                         read_only=False, confirm=True,  allowed={}),
}

# ============================== THE AGENT ====================================================
RETRYABLE = (TimeoutError, ConnectionError)

class Agent:
    def __init__(self, mock=None, tools=None, mode="read_only", confirm=lambda name, args: False,
                 max_steps=6, max_repairs=2, token_budget=2000, trace_file=None):
        self.mock = mock
        self.tools = {name: REGISTRY[name] for name in (tools or REGISTRY)}   # least privilege: only what's listed
        self.mode, self.confirm = mode, confirm
        self.max_steps, self.max_repairs, self.token_budget = max_steps, max_repairs, token_budget
        self.trace_file = Path(trace_file) if trace_file else None
        self.run_id = None

    # ---- trace: one JSON object per line (JSONL) -> easy to grep, load with pandas, replay ----
    def log(self, event, **fields):
        if not self.trace_file:
            return
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": event, **fields}
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_file.open("a") as f:          # "a" = append; `with` closes the file for us
            f.write(json.dumps(record) + "\n")

    def system_prompt(self):
        lines = [f'  {n}({", ".join(s["params"])})' for n, s in self.tools.items()]
        return ("You are an assistant with tools. Reply with ONLY one JSON object:\n"
                '  {"tool": "<name>", "args": {...}}  or  {"final": "<answer>"}\nTools:\n' + "\n".join(lines))

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

    # ---- step 02: the loop ----
    def run(self, question):
        self.run_id = uuid.uuid4().hex[:8]
        self.log("start", question=question, mode=self.mode, tools=sorted(self.tools))
        messages, repairs = [{"role": "user", "content": question}], 0
        for step in range(1, self.max_steps + 1):
            started = time.time()
            try:
                reply = self.call_llm(self.trim(messages))        # prompt -> LLM call
            except RETRYABLE:
                return self.finish("llm_unavailable", "Sorry, the model is unavailable right now.", step)
            self.log("llm_reply", step=step, reply=reply, ms=round((time.time() - started) * 1000))
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
            messages.append({"role": "assistant", "content": reply})
            if "final" in decision:
                return self.finish("final", decision["final"], step)
            name, args = decision.get("tool"), decision.get("args", {})
            t0 = time.time()
            result = self.execute(name, args)                     # execute tool (gated)
            self.log("tool_call", step=step, tool=name, args=args, result=result,
                     ok=not result.startswith("ERROR"), ms=round((time.time() - t0) * 1000, 2))
            messages.append({"role": "user", "content": f"Observation: {result}"})   # return observation, iterate
        return self.finish("max_steps", "Stopped: step limit reached.", self.max_steps)

    def finish(self, status, answer, steps):
        self.log("end", status=status, answer=answer, steps=steps)
        return {"status": status, "answer": answer, "steps": steps, "run_id": self.run_id}
