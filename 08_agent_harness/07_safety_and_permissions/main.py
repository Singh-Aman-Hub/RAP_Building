"""
GOAL   : Control what an agent may do: expose only needed tools (least privilege), mark tools
         read-only vs write, confine file access to a sandbox folder, and require confirmation for
         dangerous actions.
RUN    : cd 08_agent_harness/07_safety_and_permissions && python main.py
EXPECTED OUTPUT: a table of requests: reads inside the sandbox work, path-escape attempts and
         writes in read-only mode are blocked, delete needs confirmation.
NEEDS  : nothing (mock). Files are created only inside 08_agent_harness/sandbox_files/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "shared"))
sys.path.insert(0, str(HERE.parent))
from llm_client import generate, extract_json
from fake_llm import make_fake_agent_llm

LLM = make_fake_agent_llm()
SANDBOX = (HERE.parent / "sandbox_files").resolve()      # the ONLY folder the agent may touch
ALLOWED_SUFFIXES = {".txt", ".md"}
MAX_READ_CHARS = 500                                     # cap what flows back into the model's context

class PermissionDenied(Exception):                       # our own exception class: subclass of Exception
    pass

# ---------------------------------------------------------------------------
# Path sandboxing: the model controls `path`, so treat it like hostile user input.
# ---------------------------------------------------------------------------
def safe_path(user_path: str) -> Path:
    candidate = (SANDBOX / user_path).resolve()          # .resolve() collapses '..' and follows symlinks
    # NOTE: if user_path is ABSOLUTE ("/etc/passwd"), `SANDBOX / "/etc/passwd"` becomes just "/etc/passwd"!
    if not candidate.is_relative_to(SANDBOX):            # is it still INSIDE the sandbox after resolving?
        raise PermissionDenied(f"path {user_path!r} is outside the sandbox")
    if candidate.suffix not in ALLOWED_SUFFIXES:
        raise PermissionDenied(f"file type {candidate.suffix!r} not allowed")
    return candidate

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
def read_file(path):
    text = safe_path(path).read_text()
    return text[:MAX_READ_CHARS] + ("...[truncated]" if len(text) > MAX_READ_CHARS else "")

def write_file(path, content):
    safe_path(path).write_text(content)
    return f"wrote {len(content)} characters to {path}"

def delete_file(path):
    safe_path(path).unlink()                             # removes the file
    return f"deleted {path}"

# Each tool declares its risk. The harness - not the model - enforces it.
TOOLS = {
    "read_file":   {"fn": read_file,   "read_only": True,  "needs_confirmation": False},
    "write_file":  {"fn": write_file,  "read_only": False, "needs_confirmation": False},
    "delete_file": {"fn": delete_file, "read_only": False, "needs_confirmation": True},
}

# ---------------------------------------------------------------------------
# Confirmation hooks: a function (tool_name, args) -> bool. Swap them like plugs.
# ---------------------------------------------------------------------------
def deny_all(tool_name, args):          # safe default for unattended runs
    return False

def ask_human(tool_name, args):         # interactive CLI version (not used in the demo - needs a keyboard)
    return input(f"Allow {tool_name}({args})? [y/N] ").strip().lower() == "y"

def approve_all_for_tests(tool_name, args):
    return True

# ---------------------------------------------------------------------------
# The permission gate: ONE function every tool call must pass through.
# ---------------------------------------------------------------------------
def guarded_execute(name, args, mode, confirm):
    spec = TOOLS.get(name)
    if spec is None:
        return f"ERROR: tool {name!r} not available"
    if mode == "read_only" and not spec["read_only"]:
        return f"ERROR: permission denied: {name} is a write tool and the agent is in read_only mode"
    if spec["needs_confirmation"] and not confirm(name, args):
        return f"ERROR: permission denied: {name}({args}) needs confirmation and was not approved"
    try:
        return spec["fn"](**args)
    except PermissionDenied as err:
        return f"ERROR: permission denied: {err}"
    except FileNotFoundError:
        return "ERROR: file not found"
    except Exception as err:
        return f"ERROR: {type(err).__name__}: {err}"

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

if __name__ == "__main__":
    (SANDBOX / "out.txt").unlink(missing_ok=True)       # repeatable demo
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

# Defence in depth: (1) expose few tools  (2) read-only default  (3) sandbox paths  (4) allow-list file types
# (5) cap output size  (6) human confirmation for destructive actions  (7) log everything (step 08)
# LIMITATION: a sandbox folder is NOT a security boundary against malicious code (use containers/VMs - module 11),
#             and symlink / race-condition (TOCTOU) edge cases exist. Never give an agent `eval`, `exec` or raw shell.
# TRY IT: remove the is_relative_to check and run 'path traversal' - what comes back? (then restore it!)
# TRY IT: add a 'max file size' check to write_file. BREAK IT: use `confirm=ask_human` and answer n.
