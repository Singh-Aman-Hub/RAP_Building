# Terms — Step 07: Tool safety and permissions

Folder: `08_agent_harness/07_safety_and_permissions` · Files covered: `main.py`, `WALKTHROUGH.md` (uses `../sandbox_files/notes.txt` and `../fake_llm.py`)
Back to module: [08 Agent harness README](../README.md)

**Terms in this folder:** [Least privilege](#least-privilege) · [Untrusted model (the harness enforces)](#untrusted-model-the-harness-enforces) · [Path sandboxing](#path-sandboxing) · [Path traversal](#path-traversal) · [Path.resolve](#pathresolve) · [Path.is_relative_to](#pathis_relative_to) · [Absolute path joining trap](#absolute-path-joining-trap) · [Allow-listed file types](#allow-listed-file-types) · [Path.suffix](#pathsuffix) · [Output size cap (MAX_READ_CHARS)](#output-size-cap-max_read_chars) · [Custom exception (PermissionDenied)](#custom-exception-permissiondenied) · [Read-only vs write tool](#read-only-vs-write-tool) · [Tool metadata (read_only, needs_confirmation)](#tool-metadata-read_only-needs_confirmation) · [Permission gate (guarded_execute)](#permission-gate-guarded_execute) · [Human confirmation hook](#human-confirmation-hook) · [Functions as arguments](#functions-as-arguments) · [input](#input) · [Pathlib file operations (read_text, write_text, unlink)](#pathlib-file-operations-read_text-write_text-unlink) · [try / except with several handlers](#try--except-with-several-handlers) · [Dictionary .get for the registry](#dictionary-get-for-the-registry) · [Defence in depth](#defence-in-depth) · [Sandbox is not a security boundary](#sandbox-is-not-a-security-boundary) · [Table-driven test cases](#table-driven-test-cases) · [Fake agent LLM (fake_llm.py)](#fake-agent-llm-fake_llmpy)

---

## Least privilege
*Type: concept (security)*

**1. Definition:** Giving a system only the tools and access it needs and nothing more.
**2. In plain words:** If the model is tricked or makes a mistake, the damage is limited to what you allowed. A reading assistant that has no write or delete tool cannot destroy files.
**3. Example to quote:**
> "My agent defaults to `read_only` mode, so even a bad instruction cannot write or delete anything."

**Seen in:** `main.py`, docstring and the defence-in-depth comment.

## Untrusted model (the harness enforces)
*Type: concept (security)*

**1. Definition:** Treating the model's output as a proposal from an untrusted component; the harness code, not the model, decides what is allowed.
**2. In plain words:** The model can be wrong, manipulated by prompt injection, or just creative. Safety rules therefore live in code and permissions, not in the prompt.
**3. Example to quote:**
> "The model asked to read `../../.env`; my `safe_path` function refused it regardless of what the model intended."

**Seen in:** `main.py`, comment above `TOOLS`; `WALKTHROUGH.md`, Say it aloud.

## Path sandboxing
*Type: concept (security)*

**1. Definition:** Restricting file access to one folder (the sandbox) by resolving each requested path and rejecting any that falls outside it.
**2. In plain words:** The model chooses the path, so treat it like hostile user input. Attempts like `../../.env` or `/etc/passwd` must be blocked.
**3. Example to quote:**
> "Reading `notes.txt` works, but `../../.env` and `/etc/passwd` both fail with 'outside the sandbox'."

**Seen in:** `main.py`, `SANDBOX`, `safe_path`.

## Path traversal
*Type: concept (security)*

**1. Definition:** An attack using `..` segments or absolute paths to reach files outside an intended folder.
**2. In plain words:** `SANDBOX / "../../.env"` looks like it is inside the sandbox but points two folders up. You must resolve the path and then check where it really lands.
**3. Example to quote:**
> "The 'path traversal ../' case is blocked because after `resolve()` the path is no longer inside `SANDBOX`."

**Seen in:** `main.py`, `safe_path`, test cases.

## Path.resolve
*Type: Python standard library*

**1. Definition:** `Path.resolve()` returns the absolute path with `..` segments collapsed and symbolic links followed.
**2. In plain words:** It turns a tricky path into its real destination so you can compare it with the allowed folder.
**3. Example to quote:**
> "`Path('/a/b/../c').resolve()` gives `/a/c`, which is why I resolve before checking."

```python
from pathlib import Path
Path("/a/b/../c").resolve()   # /a/c
```

**Seen in:** `main.py`, `SANDBOX`, `safe_path`.

## Path.is_relative_to
*Type: Python standard library (3.9+)*

**1. Definition:** `a.is_relative_to(b)` returns True if path `a` is located inside (or equal to) path `b`.
**2. In plain words:** It is the actual containment test after resolving. If it is False, access is denied.
**3. Example to quote:**
> "`candidate.is_relative_to(SANDBOX)` is False for `/etc/passwd`, so I raise `PermissionDenied`."

```python
Path("/a/b").is_relative_to("/a")   # True
```

**Seen in:** `main.py`, `safe_path`.

## Absolute path joining trap
*Type: Python gotcha*

**1. Definition:** In `pathlib`, joining with an absolute path discards the left side: `Path("/x") / "/etc"` becomes `/etc`.
**2. In plain words:** A naive `SANDBOX / user_path` would silently escape the sandbox if the model supplies `/etc/passwd`. Resolving and checking containment catches it.
**3. Example to quote:**
> "`SANDBOX / '/etc/passwd'` is just `/etc/passwd`, so the containment check after resolving is essential."

**Seen in:** `main.py`, comment in `safe_path`; `WALKTHROUGH.md`.

## Allow-listed file types
*Type: concept (security)*

**1. Definition:** Accepting only specified file extensions (here `.txt` and `.md`) and rejecting everything else.
**2. In plain words:** Even inside the sandbox some files should not be read, such as configuration or scripts.
**3. Example to quote:**
> "Reading `config.json` is denied because `.json` is not in `ALLOWED_SUFFIXES`."

**Seen in:** `main.py`, `ALLOWED_SUFFIXES`, `safe_path`.

## Path.suffix
*Type: Python standard library*

**1. Definition:** The file extension of a path including the dot, such as `.txt`.
**2. In plain words:** It is the quick way to check what type of file is being requested.
**3. Example to quote:**
> "`candidate.suffix not in ALLOWED_SUFFIXES` rejects any extension I did not list."

**Seen in:** `main.py`, `safe_path`.

## Output size cap (`MAX_READ_CHARS`)
*Type: concept*

**1. Definition:** A limit on how much tool output is returned to the model, here 500 characters followed by `...[truncated]`.
**2. In plain words:** A huge file would fill the context window and cost money, and could carry hidden instructions. Capping keeps results manageable.
**3. Example to quote:**
> "`text[:MAX_READ_CHARS] + '...[truncated]'` stops a big file flooding the conversation."

**Seen in:** `main.py`, `MAX_READ_CHARS`, `read_file`.

## Custom exception (`PermissionDenied`)
*Type: Python syntax*

**1. Definition:** `class PermissionDenied(Exception): pass` defines a new exception type by subclassing `Exception`.
**2. In plain words:** A named exception lets you catch exactly your permission failures, separately from other errors, and report a clear message.
**3. Example to quote:**
> "`safe_path` raises `PermissionDenied`, and `guarded_execute` catches it and returns `ERROR: permission denied: ...`."

**Seen in:** `main.py`, `class PermissionDenied`.

## Read-only vs write tool
*Type: concept*

**1. Definition:** A classification of tools by whether they change anything; read-only tools only fetch information, write tools modify state.
**2. In plain words:** Each tool declares its risk in the registry, and the harness blocks write tools in read-only mode.
**3. Example to quote:**
> "`write_file` in `read_only` mode returns 'permission denied: write_file is a write tool'."

**Seen in:** `main.py`, `TOOLS` metadata, `guarded_execute`.

## Tool metadata (`read_only`, `needs_confirmation`)
*Type: concept*

**1. Definition:** Extra fields stored next to each tool function describing its risk, enforced by the harness.
**2. In plain words:** Keeping metadata beside the function makes the permission rules visible and easy to change in one place.
**3. Example to quote:**
> "`delete_file` is marked `needs_confirmation: True`, so it only runs when the confirm hook approves."

**Seen in:** `main.py`, `TOOLS`.

## Permission gate (`guarded_execute`)
*Type: project function*

**1. Definition:** One function every tool call must pass through, which checks tool existence, mode, confirmation and handles permission errors.
**2. In plain words:** A single choke point is easier to audit than checks scattered across tools. The order is: unknown tool, mode check, confirmation, then execution inside `try/except`.
**3. Example to quote:**
> "All eight demo cases go through `guarded_execute`, which is why every denial has a consistent message."

**Seen in:** `main.py`, `guarded_execute`.

## Human confirmation hook
*Type: concept*

**1. Definition:** A function `(tool_name, args) -> bool` passed to the agent that decides whether a dangerous action may proceed, such as `deny_all`, `ask_human` or `approve_all_for_tests`.
**2. In plain words:** Because it is a parameter, you can swap behaviour without changing the loop: unattended runs deny, interactive runs ask a person, tests approve.
**3. Example to quote:**
> "'delete, not confirmed' uses `deny_all` and is refused; 'delete, confirmed' uses `approve_all_for_tests` and succeeds."

**Seen in:** `main.py`, `deny_all`, `ask_human`, `approve_all_for_tests`.

## Functions as arguments
*Type: Python syntax*

**1. Definition:** Passing a function as a parameter, for example `run_agent(question, mode, confirm=deny_all)`, and calling it inside.
**2. In plain words:** Functions are ordinary values, so behaviour can be plugged in. Here the confirmation policy is chosen by the caller.
**3. Example to quote:**
> "`confirm(name, args)` inside `guarded_execute` calls whichever policy function I passed in."

**Seen in:** `main.py`, `guarded_execute`, `run_agent`.

## input
*Type: Python built-in*

**1. Definition:** `input(prompt)` reads a line from the keyboard and returns it as a string.
**2. In plain words:** `ask_human` uses it to ask "Allow delete_file(...)? [y/N]". The demo does not use it because it needs a person at the keyboard.
**3. Example to quote:**
> "`input(...).strip().lower() == 'y'` accepts only an explicit yes; anything else is treated as no."

**Seen in:** `main.py`, `ask_human`.

## Pathlib file operations (`read_text`, `write_text`, `unlink`)
*Type: Python standard library*

**1. Definition:** `read_text()` returns a file's text, `write_text(s)` writes text and returns the number of characters, `unlink()` deletes the file.
**2. In plain words:** They are the three actions the file tools perform. `unlink(missing_ok=True)` is used to reset the demo.
**3. Example to quote:**
> "`safe_path(path).write_text(content)` only runs after the path has passed the sandbox check."

**Seen in:** `main.py`, `read_file`, `write_file`, `delete_file`.

## try / except with several handlers
*Type: Python syntax*

**1. Definition:** Multiple `except` clauses are checked in order, so specific exceptions (`PermissionDenied`, `FileNotFoundError`) are handled before the general `Exception`.
**2. In plain words:** Each failure gets a clear message, and anything unexpected is still reported rather than crashing the agent.
**3. Example to quote:**
> "A missing file returns `ERROR: file not found` instead of a traceback."

**Seen in:** `main.py`, `guarded_execute`.

## Dictionary `.get` for the registry
*Type: Python built-in*

**1. Definition:** `TOOLS.get(name)` returns the entry or `None` when the tool is unknown.
**2. In plain words:** It lets the gate answer "tool not available" without a `KeyError`.
**3. Example to quote:**
> "`spec = TOOLS.get(name)` followed by `if spec is None` handles invented tool names."

**Seen in:** `main.py`, `guarded_execute`.

## Defence in depth
*Type: concept (security)*

**1. Definition:** Stacking several independent protections so that no single failure is catastrophic.
**2. In plain words:** The file lists seven: few tools, read-only default, sandbox paths, allowed file types, output caps, human confirmation, and logging.
**3. Example to quote:**
> "Even if confirmation were bypassed, the sandbox would still stop a write outside the folder."

**Seen in:** `main.py`, closing comment.

## Sandbox is not a security boundary
*Type: concept (security limitation)*

**1. Definition:** Restricting paths in code does not isolate a process from the system; malicious code, symlink tricks and race conditions (TOCTOU) can still cause harm.
**2. In plain words:** For real isolation use containers or virtual machines (module 11) and never give an agent `eval`, `exec` or raw shell access.
**3. Example to quote:**
> "My folder check is one layer; if I needed to run untrusted code I would put it in a container."

**Seen in:** `main.py`, LIMITATION comment.

## Table-driven test cases
*Type: concept / Python syntax*

**1. Definition:** A list of tuples `(label, question, mode, confirm)` iterated by a loop to run many scenarios with the same code.
**2. In plain words:** It makes the demo and your own tests easy to extend: add a row to add a case.
**3. Example to quote:**
> "I added a ninth row for a huge file and the same loop ran it."

**Seen in:** `main.py`, `cases` in the `__main__` block.

## Fake agent LLM (`fake_llm.py`)
*Type: project helper*

**1. Definition:** `make_fake_agent_llm()` returns a scripted mock whose regex rules turn questions like `read file notes.txt` into tool-call JSON.
**2. In plain words:** It lets the permission logic be tested deterministically without a real model.
**3. Example to quote:**
> "The question 'read file ../../.env' becomes a `read_file` tool call, which my gate then denies."

**Seen in:** `main.py`, `LLM = make_fake_agent_llm()`; `../fake_llm.py`.
