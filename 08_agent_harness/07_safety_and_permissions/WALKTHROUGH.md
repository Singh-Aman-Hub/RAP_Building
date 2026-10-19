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
