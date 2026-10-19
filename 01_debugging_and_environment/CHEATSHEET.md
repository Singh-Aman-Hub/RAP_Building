# Debugging cheat sheet (print this)

## The loop
**Read the error -> reproduce -> shrink -> one hypothesis -> test -> fix the cause -> verify.**

## Reading output
1. Last line = exception type + message. 2. Frame above = line that raised. 3. Frames above = callers (oldest on top).
4. Look at *your* files first; library frames are usually fine — the bad value came from your code.

## Error type -> first thing to check
| Error | First check |
|---|---|
| `SyntaxError` | Line shown **and** the line before: missing `:` `)` `]` quotes, `=` vs `==` |
| `IndentationError` | Mixed tabs/spaces; block under `if/def` |
| `NameError` | Typo? defined above? inside another function's scope? |
| `TypeError` | `print(type(x), repr(x))` for each operand; arg count; `None` returned |
| `AttributeError: 'NoneType'...` | A function forgot `return`, or a lookup returned None |
| `KeyError` | Print the dict keys; use `.get()`; check spelling/case |
| `IndexError` | `len()`; off-by-one; empty list |
| `ValueError` | Right type, wrong content (e.g. `int("abc")`); shapes in ML |
| `ZeroDivisionError` | Empty list / zero count |
| `FileNotFoundError` | `os.getcwd()`, absolute path, spelling, case (Linux) |
| `ModuleNotFoundError` | `sys.executable`, `python -m pip show pkg`, venv active? shadowing file? |
| `ImportError: cannot import name` | Version mismatch or wrong name: `help(module)` / `dir(module)` |
| `UnicodeDecodeError` | Open with `encoding="utf-8"` / binary file |
| `RecursionError` | Missing base case |
| `KeyboardInterrupt` | You pressed Ctrl+C (possibly an infinite loop) |
| `requests.Timeout` | Server slow/hung; raise timeout or fix server; retry with back-off |
| `requests.ConnectionError` | Server not running, wrong host/port, container bound to 127.0.0.1 |
| HTTP 404 / 422 / 500 | Wrong route / invalid body (read response JSON) / server traceback (read server logs) |
| `RuntimeError: CUDA out of memory` / `Killed` | Smaller batch / max tokens / model; more memory (see module 06) |
| `torch ... size mismatch` / shape errors | Print `.shape` of every tensor along the path (module 03) |

## Environment checks (copy-paste)
```bash
which python; python --version
python -c "import sys; print(sys.executable)"
python -m pip list | grep -i NAME
python -m pip show NAME
python -m pip check
env | grep -i NAME                 # env vars (don't print secrets in shared logs)
pwd; ls -la                        # where am I, what's here
lsof -i :8000                      # who owns port 8000 (macOS/Linux)
curl -i http://127.0.0.1:8000/health
```

## Python inspection snippets
```python
print(type(x), repr(x))            # what is it really?
print(len(x), x[:3])               # size and a peek
print(dir(obj)); help(obj.method)  # what can I call?
import inspect; print(inspect.signature(fn))
breakpoint()                       # pdb: n next | s step in | p x | l list | c continue | q quit
assert x > 0, f"x was {x}"         # state the assumption in code
python -X faulthandler script.py   # tracebacks on hard crashes
python -m pdb script.py            # post-mortem on crash
```

## Docker quick map (see module 11)
`docker ps -a` -> `docker logs --tail 50 C` -> `docker exec -it C sh` -> `docker inspect C` -> `docker build --progress=plain .`

## Anti-patterns
- Changing many things at once. - Fixing the symptom (`try/except: pass`). - Skipping the traceback. - Pasting secrets into chats.
- Trusting "it works on my machine" (check Python version, OS case-sensitivity, env vars, cwd).
