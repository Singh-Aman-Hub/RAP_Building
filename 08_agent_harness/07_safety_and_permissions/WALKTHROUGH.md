# Step 07 — Tool safety and permissions

## The story so far
In [step 06](../06_errors_and_retries/) the agent recovers from errors. But the model controls the tool *arguments*, and the model reads untrusted text (see [07/06](../../07_prompt_engineering/06_prompt_injection/)).

## The pain point
A hijacked or confused model can ask for `../../.env`, `/etc/passwd`, or `delete_file`. If your tools do exactly what they are told, the agent can leak or destroy things.

## The spark
Enforce policy in the harness, never in the prompt: a tool allowlist (unknown tools do not exist), read-only vs write modes, path confinement to a sandbox folder, a file-type allowlist, capped output, and human confirmation for destructive tools.

## Goal
Limit what an agent can do: least privilege, read-only vs write, sandboxed paths, human confirmation.

## Run it
```bash
cd 08_agent_harness/07_safety_and_permissions && ../../.venv/bin/python main.py
```
Real output (the agent's final text is shown after each case):
```
read inside sandbox        [read_only ] Result: Shopping list: (3 lines, trimmed)
path traversal  ../        [read_only ] Sorry, the tool failed: ERROR: permission denied: path '../../.env' is outside the sandbox
absolute path              [read_only ] Sorry, the tool failed: ERROR: permission denied: path '/etc/passwd' is outside the sandbox
wrong file type            [read_only ] Sorry, the tool failed: ERROR: permission denied: file type '.json' not allowed
write in read_only mode    [read_only ] Sorry, the tool failed: ERROR: permission denied: write_file is a write tool and the agent is in read_only mode
write in read_write mode   [read_write] Result: wrote 5 characters to out.txt
delete, not confirmed      [read_write] Sorry, the tool failed: ERROR: permission denied: delete_file({'path': 'out.txt'}) needs confirmation and was not approved
delete, confirmed          [read_write] Result: deleted out.txt

unknown tool 'run_shell'  : ERROR: tool 'run_shell' not available
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Least privilege | minimal permissions, minimal tool set | Same | Give the agent only the tools and access it needs. |
| Allowlist | whitelist, permitted list | Same | Only listed items work; everything else is refused (the opposite is a denylist/blocklist). |
| Sandbox folder | jail, chroot, working directory limit | Not same as container | A path check in code; a container/VM is a real isolation boundary. |
| Path traversal | directory traversal, `../` escape | Same | Using `..` or absolute paths to leave the allowed folder. |
| Human-in-the-loop | confirmation, approval hook, HITL | Same | A person approves risky actions before they run. |

## Code, step by step

### 1. Imports and path setup  (`main.py` lines 10–17)

```python
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json
from fake_llm import make_fake_agent_llm
```

- **In simple words:** Make helpers importable.
- **Exactly:** Same pattern as earlier steps. `generate` and `extract_json` run the loop; `LLM` is scripted.
- **Why it is here:** Required before any code uses the helpers.

### 2. Sandbox settings  (`main.py` lines 19–22)

```python
LLM = make_fake_agent_llm()
SANDBOX = (HERE.parent / "sandbox_files").resolve()      # the ONLY folder the agent may touch
ALLOWED_SUFFIXES = {".txt", ".md"}
MAX_READ_CHARS = 500                                     # cap what flows back into the model's context
```

- **In simple words:** Choose the one allowed folder, the allowed file types and a read cap.
- **Exactly:** `SANDBOX = (HERE.parent / "sandbox_files").resolve()` is an absolute `Path`. `ALLOWED_SUFFIXES = {".txt", ".md"}` is a set. `MAX_READ_CHARS = 500`.
- **Why it is here:** Limits where, what, and how much the agent can touch.

### 3. Our own exception type  (`main.py` lines 24–25)

```python
class PermissionDenied(Exception):                       # our own exception class: subclass of Exception
    pass
```

- **In simple words:** A named error for "not allowed".
- **Exactly:** `class PermissionDenied(Exception)` with `pass` body; can be caught separately from other errors.
- **Why it is here:** Lets the gate tell policy refusals apart from ordinary bugs.

### 4. Check a path is safe  (`main.py` lines 30–37)

```python
def safe_path(user_path: str) -> Path:
    candidate = (SANDBOX / user_path).resolve()          # .resolve() collapses '..' and follows symlinks
    # NOTE: if user_path is ABSOLUTE ("/etc/passwd"), `SANDBOX / "/etc/passwd"` becomes just "/etc/passwd"!
    if not candidate.is_relative_to(SANDBOX):            # is it still INSIDE the sandbox after resolving?
        raise PermissionDenied(f"path {user_path!r} is outside the sandbox")
    if candidate.suffix not in ALLOWED_SUFFIXES:
        raise PermissionDenied(f"file type {candidate.suffix!r} not allowed")
    return candidate
```

- **In simple words:** Turn the model's path into a real path and make sure it stays inside the sandbox with an allowed type.
- **Exactly:** `(SANDBOX / user_path).resolve()` collapses `..` and symlinks; an absolute `user_path` replaces `SANDBOX` in `/`. `is_relative_to(SANDBOX)` (Python 3.9+) is False for escapes. `.suffix` is e.g. `.json`. Real: `path '../../.env' is outside the sandbox`, `file type '.json' not allowed`.
- **Why it is here:** Checking after resolving defeats `../` tricks; checking the raw string would not.

### 5. Read tool  (`main.py` lines 42–44)

```python
def read_file(path):
    text = safe_path(path).read_text()
    return text[:MAX_READ_CHARS] + ("...[truncated]" if len(text) > MAX_READ_CHARS else "")
```

- **In simple words:** Read a file and cut it if long.
- **Exactly:** `safe_path(path).read_text()`; `text[:500]` plus `...[truncated]` if longer. The sandbox notes file reads `Shopping list: - rice - lentils - tea`.
- **Why it is here:** Huge files would flood the model's context.

### 6. Write tool  (`main.py` lines 46–48)

```python
def write_file(path, content):
    safe_path(path).write_text(content)
    return f"wrote {len(content)} characters to {path}"
```

- **In simple words:** Write text to a safe path.
- **Exactly:** `write_text(content)` returns characters written; real: `wrote 5 characters to out.txt`.
- **Why it is here:** Writes need the same path checks as reads.

### 7. Delete tool  (`main.py` lines 50–52)

```python
def delete_file(path):
    safe_path(path).unlink()                             # removes the file
    return f"deleted {path}"
```

- **In simple words:** Remove a safe file.
- **Exactly:** `Path.unlink()` deletes; real: `deleted out.txt`.
- **Why it is here:** Destructive, so later guarded by confirmation.

### 8. Declare each tool's risk  (`main.py` lines 54–59)

```python
# Each tool declares its risk. The harness - not the model - enforces it.
TOOLS = {
    "read_file":   {"fn": read_file,   "read_only": True,  "needs_confirmation": False},
    "write_file":  {"fn": write_file,  "read_only": False, "needs_confirmation": False},
    "delete_file": {"fn": delete_file, "read_only": False, "needs_confirmation": True},
}
```

- **In simple words:** Label each tool as read-only or not, and whether a human must confirm.
- **Exactly:** Dict entries `{"fn", "read_only", "needs_confirmation"}`. Only `delete_file` needs confirmation.
- **Why it is here:** The harness enforces the policy; the model cannot argue its way around it.

### 9. Confirmation hooks  (`main.py` lines 61–71)

```python
# ---------------------------------------------------------------------------
# Confirmation hooks: a function (tool_name, args) -> bool. Swap them like plugs.
# ---------------------------------------------------------------------------
def deny_all(tool_name, args):          # safe default for unattended runs
    return False

def ask_human(tool_name, args):         # interactive CLI version (not used in the demo - needs a keyboard)
    return input(f"Allow {tool_name}({args})? [y/N] ").strip().lower() == "y"

def approve_all_for_tests(tool_name, args):
    return True
```

- **In simple words:** Three interchangeable ways to answer "may I do this?".
- **Exactly:** Each is `(tool_name, args) -> bool`: `deny_all` returns False, `ask_human` uses `input()` and accepts only `y`, `approve_all_for_tests` returns True.
- **Why it is here:** Swapping a function changes policy without changing the gate.

### 10. The gate: availability, mode, confirmation  (`main.py` lines 76–83)

```python
def guarded_execute(name, args, mode, confirm):
    spec = TOOLS.get(name)
    if spec is None:
        return f"ERROR: tool {name!r} not available"
    if mode == "read_only" and not spec["read_only"]:
        return f"ERROR: permission denied: {name} is a write tool and the agent is in read_only mode"
    if spec["needs_confirmation"] and not confirm(name, args):
        return f"ERROR: permission denied: {name}({args}) needs confirmation and was not approved"
```

- **In simple words:** Every tool call passes three checks before it runs.
- **Exactly:** Unknown name -> `ERROR: tool 'run_shell' not available`. In `read_only` mode non-read tools are blocked. If `needs_confirmation`, `confirm(name, args)` must return True. Real: `delete_file({'path': 'out.txt'}) needs confirmation and was not approved`.
- **Why it is here:** One choke point means no tool can bypass the rules.

### 11. Run the tool and convert errors to text  (`main.py` lines 84–91)

```python
    try:
        return spec["fn"](**args)
    except PermissionDenied as err:
        return f"ERROR: permission denied: {err}"
    except FileNotFoundError:
        return "ERROR: file not found"
    except Exception as err:
        return f"ERROR: {type(err).__name__}: {err}"
```

- **In simple words:** Execute, and turn any failure into an error observation.
- **Exactly:** `PermissionDenied` -> `ERROR: permission denied: ...`; `FileNotFoundError` -> `ERROR: file not found`; anything else -> `ERROR: <Type>: <msg>`.
- **Why it is here:** A crash would kill the agent and leak tracebacks.

### 12. The agent loop using the gate  (`main.py` lines 93–103)

```python
def run_agent(question, mode="read_only", confirm=deny_all, max_steps=4):
    messages = [{"role": "user", "content": question}]
    for _ in range(max_steps):
        reply = generate(messages, mock=LLM)
        decision = extract_json(reply)
        messages.append({"role": "assistant", "content": reply})
        if "final" in decision:
            return decision["final"]
        result = guarded_execute(decision["tool"], decision["args"], mode, confirm)
        messages.append({"role": "user", "content": f"Observation: {result}"})
    return "STOPPED: max_steps"
```

- **In simple words:** The usual loop but every call goes through `guarded_execute`.
- **Exactly:** Defaults `mode="read_only"`, `confirm=deny_all`: safe by default. At most `max_steps=4` iterations.
- **Why it is here:** Safe defaults mean forgetting to configure does not grant power.

### 13. Main guard, reset demo file  (`main.py` lines 105–106)

```python
if __name__ == "__main__":
    (SANDBOX / "out.txt").unlink(missing_ok=True)       # repeatable demo
```

- **In simple words:** Remove a leftover output file.
- **Exactly:** `unlink(missing_ok=True)`.
- **Why it is here:** Makes the demo repeatable.

### 14. Eight demo requests  (`main.py` lines 107–118)

```python
    cases = [
        ("read inside sandbox",         "read file notes.txt",                  "read_only", deny_all),
        ("path traversal  ../",         "read file ../../.env",                 "read_only", deny_all),
        ("absolute path",               "read file /etc/passwd",                "read_only", deny_all),
        ("wrong file type",             "read file config.json",                 "read_only", deny_all),
        ("write in read_only mode",     "write 'hello' to out.txt",             "read_only", deny_all),
        ("write in read_write mode",    "write 'hello' to out.txt",             "read_write", deny_all),
        ("delete, not confirmed",       "delete file out.txt",                  "read_write", deny_all),
        ("delete, confirmed",           "delete file out.txt",                  "read_write", approve_all_for_tests),
    ]
    for label, question, mode, confirm in cases:
        print(f"{label:26s} [{mode:10s}] {run_agent(question, mode, confirm)}")
```

- **In simple words:** Allowed and forbidden requests in one table.
- **Exactly:** Real: read ok; `../../.env` and `/etc/passwd` blocked; `.json` blocked; write blocked in read_only; write ok in read_write; delete blocked then allowed with `approve_all_for_tests`.
- **Why it is here:** Each row proves one defence layer.

### 15. Unknown tool  (`main.py` lines 120–121)

```python
    # Allowlist: a tool the model invents is not in TOOLS, so it is simply unavailable.
    print("\nunknown tool 'run_shell'  :", guarded_execute("run_shell", {"cmd": "rm -rf /"}, "read_write", deny_all))
```

- **In simple words:** A tool the model invented is simply unavailable.
- **Exactly:** Real: `ERROR: tool 'run_shell' not available`.
- **Why it is here:** Allowlist beats trying to block bad names.

### 16. Show the real traceback  (`main.py` lines 123–129)

```python
    # The raw failure: what safe_path raises BEFORE guarded_execute turns it into an Observation.
    import traceback
    print("\nReal trace of a path-escape attempt (the harness catches this and tells the model):")
    try:
        read_file("../../.env")
    except PermissionDenied:
        print(traceback.format_exc().rstrip())
```

- **In simple words:** See what is raised before the harness turns it into text.
- **Exactly:** `traceback.format_exc()` prints the stack ending in `PermissionDenied: path '../../.env' is outside the sandbox` at `safe_path`.
- **Why it is here:** Teaches how to read a traceback from bottom to top.

### 17. Defence-in-depth notes  (`main.py` lines 131–136)

```python
# Defence in depth: (1) expose few tools  (2) read-only default  (3) sandbox paths  (4) allow-list file types
# (5) cap output size  (6) human confirmation for destructive actions  (7) log everything (step 08)
# LIMITATION: a sandbox folder is NOT a security boundary against malicious code (use containers/VMs - module 11),
#             and symlink / race-condition (TOCTOU) edge cases exist. Never give an agent `eval`, `exec` or raw shell.
# TRY IT: remove the is_relative_to check and run 'path traversal' - what comes back? (then restore it!)
# TRY IT: add a 'max file size' check to write_file. BREAK IT: use `confirm=ask_human` and answer n.
```

- **In simple words:** Comments listing the layers and limits.
- **Exactly:** Comments only; a sandbox folder is not a security boundary against malicious code (use containers/VMs).
- **Why it is here:** Avoids overclaiming safety.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class PermissionDenied(Exception): pass` | define your own exception type | – |
| `Path.resolve()` | absolute path with `..` and symlinks removed | `Path("/a/b/../c").resolve()` -> `/a/c` |
| `a.is_relative_to(b)` | is path `a` inside folder `b`? | `Path("/a/b").is_relative_to("/a")` -> True |
| `SANDBOX / "/etc/passwd"` | joining an ABSOLUTE path discards the left side — the classic trap | `Path("/x") / "/etc"` -> `/etc` |
| passing functions as arguments (`confirm=deny_all`) | swap behaviour without changing the loop | – |
| `{"read_file": {"fn": read_file, "read_only": True}}` | tool metadata next to the function | – |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `pathlib.Path.read_text / write_text / unlink` | standard library | file read/write/delete | str / int / None |
| `Path.suffix` | standard library | file extension, e.g. `.txt` | str |
| `input()` | built-in | read a line from the keyboard (used by `ask_human`) | str |

## Execution trace
`guarded_execute` is the single gate: unknown tool? -> mode check (read-only vs write) -> confirmation for risky tools -> run inside `try/except PermissionDenied`. The path check happens inside the tool via `safe_path`: resolve, then require "still inside SANDBOX" and an allowed suffix.

## Experiments
- Remove `is_relative_to` and run the traversal case (then restore it!).
- Use `confirm=ask_human` interactively and answer `n`.
- Add a maximum file size for `write_file`.

## Debugging practice
Run [`broken_path_escape.py`](../broken_examples/broken_path_escape.py): a file outside the sandbox is read successfully — the naive `SANDBOX / path` trusts the model. Fix by resolving and checking.

## Rebuild checklist
- [ ] `safe_path`. [ ] tool metadata (`read_only`, `needs_confirmation`). [ ] one `guarded_execute` gate. [ ] a swappable `confirm` hook.

## The new problem (cliffhanger)
With guardrails the agent is safe to run, and with logging in hand we can see why it did what it did. But when it misbehaves at 2 a.m., what exactly happened? We need a record of every step. Next: [08_trace_and_full_agent](../08_trace_and_full_agent/).

> **Say it aloud (interview box):** I treat every model-chosen argument as hostile input and enforce policy in the harness: a tool allowlist, read-only versus write modes, path confinement to a sandbox folder, and confirmation for destructive actions. Prompt rules are only hints, so the checks live in code. A sandbox folder is not a real security boundary, so for untrusted code I would use a container.
