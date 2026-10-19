# Step 07 — Tool safety and permissions

## Goal
Limit what an agent can do: least privilege, read-only vs write, sandboxed paths, human confirmation.

## Run it
```bash
cd 08_agent_harness/07_safety_and_permissions && ../../.venv/bin/python main.py
```
Real output (the agent's final text is shown after each case):
```
read inside sandbox        [read_only ] Result: Shopping list: ...
path traversal  ../        [read_only ] Sorry, the tool failed: ERROR: permission denied: path '../../.env' is outside the sandbox
absolute path              [read_only ] Sorry, the tool failed: ERROR: permission denied: path '/etc/passwd' is outside the sandbox
wrong file type            [read_only ] Sorry, the tool failed: ERROR: permission denied: file type '.json' not allowed
write in read_only mode    [read_only ] Sorry, the tool failed: ERROR: permission denied: write_file is a write tool and the agent is in read_only mode
write in read_write mode   [read_write] Result: wrote 5 characters to out.txt
delete, not confirmed      [read_write] Sorry, the tool failed: ERROR: permission denied: delete_file({'path': 'out.txt'}) needs confirmation and was not approved
delete, confirmed          [read_write] Result: deleted out.txt
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

## Say it aloud
"The model is an untrusted component: it only proposes actions. My harness enforces least privilege — only needed tools, read-only by default, file access confined to a sandbox with resolved paths, allow-listed file types, size caps and human confirmation for destructive actions — and logs everything. A sandbox folder is not a security boundary against malicious code; for that I need containers or VMs."
