# Step 07 — Gunicorn, Uvicorn, workers and concurrency

## The story so far
Step 06 served one request nicely. A real product has many users at the same moment, and the framework alone cannot decide how requests overlap.

## The pain point
One worker doing a 1-second job handles one request per second. Four users arriving together wait 1, 2, 3 and 4 seconds, and a blocking call inside `async def` freezes everybody.

## The spark
Let the server manage concurrency: more worker processes for parallel CPU work, threads or async to overlap waiting. Measure with real servers, because the numbers surprise you.

## Goal
Separate two ideas: the **framework** (Flask/FastAPI: your routes) and the **server** (Gunicorn/Uvicorn: sockets, processes, concurrency). Then measure how workers, threads and `async` change how many requests finish at once.

## Run it (~40 s; starts and stops its own servers)
```bash
cd 09_api_development/07_servers_workers_concurrency
python main.py
```
Real output:
```
gunicorn 2 workers: distinct worker PIDs seen = 2  [2721, 2722]

4 parallel 1-second requests:
  gunicorn  1 sync worker                 4.1 s
  gunicorn  2 sync workers                2.1 s
  gunicorn  1 worker x 4 threads          1.1 s
  uvicorn   async def (await sleep)       1.0 s
  uvicorn   def (threadpool)              1.0 s
  uvicorn   async def + BLOCKING          4.1 s
```

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Gunicorn vs Uvicorn | Python application servers | No | Gunicorn serves WSGI (Flask) with a pre-fork worker model; Uvicorn serves ASGI (FastAPI) with an event loop |
| Worker vs process vs thread | Units of concurrency | Worker = usually a process | A worker is a Gunicorn/Uvicorn child process with its own memory; threads live inside a process and share memory |
| Concurrency vs parallelism | Doing many things | No | Concurrency = overlapping waits (one CPU, interleaved); parallelism = literally simultaneous on several CPUs |
| async def vs def (in FastAPI) | Coroutine vs plain function | No | `async def` runs on the event loop (must not block); `def` runs in a thread pool |
| I/O-bound vs CPU-bound | Waiting vs computing | No | I/O-bound work benefits from threads/async; CPU-bound work needs more processes |
| `terminate()` vs `kill()` | SIGTERM vs SIGKILL | No | terminate lets the process clean up; kill is immediate and cannot be caught |
| Event loop vs thread pool | Where code runs | No | The loop is one thread cooperatively switching between `await` points; the pool is several OS threads |

## Code, step by step

### 1. Docstring: goal, expected output, needs  (`main.py` lines 1–22)
```python
"""
GOAL:   Understand the SERVER layer (Gunicorn for Flask, Uvicorn for FastAPI), what a "worker" is,
        and why sync vs async changes how many requests are handled at once.

RUN:    cd 09_api_development/07_servers_workers_concurrency && python main.py     (~40 s)

WHAT IT DOES: starts real servers as background processes, fires requests with httpx, prints
        timings, then stops every server it started.  Equivalent manual commands are in the
        COMMANDS dict below and in WALKTHROUGH.md.

EXPECTED OUTPUT (shape):
        gunicorn 2 workers: distinct worker PIDs seen = 2
        4 parallel 1-second requests:
          gunicorn  1 sync worker            ~4.0 s
          gunicorn  2 sync workers           ~2.0 s
          gunicorn  1 worker x 4 threads     ~1.0 s
          uvicorn   async def (await sleep)  ~1.0 s
          uvicorn   def (threadpool)         ~1.0 s
          uvicorn   async def + BLOCKING     ~4.0 s   <- the classic mistake

NEEDS:  gunicorn, uvicorn, httpx (all in requirements.txt)
"""
```
- **In simple words:** A header that says what the script demonstrates and what you should see.
- **Exactly:** Documents the six timings you should get (about 4.0, 2.0, 1.0, 1.0, 1.0, 4.0 s). Real run: 4.4, 2.3, 1.1, 1.1, 1.1, 4.3 s: slightly above ideal because of process start-up and HTTP overhead.
- **Why it is here:** Tells you what "correct" looks like before you run it.

### 2. Imports  (`main.py` lines 24–30)
```python
import subprocess      # built-in: start other programs (here: servers) from Python
import sys
import time
from concurrent.futures import ThreadPoolExecutor   # built-in: fire several requests at the same time
from pathlib import Path

import httpx           # third-party HTTP client (modern requests-like API)
```
- **In simple words:** Bring in tools to launch programs, wait, run parallel threads, find folders and make HTTP calls.
- **Exactly:** `subprocess` starts other programs; `ThreadPoolExecutor` runs functions in parallel threads; `pathlib.Path` handles paths; `httpx` is the HTTP client (requests-like API).
- **Why it is here:** Everything here is about starting real servers and timing them, so each import has a job.

### 3. Locate the apps and the Python interpreter  (`main.py` lines 32–33)
```python
APP_DIR = Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"  # where the apps live
PY = sys.executable    # the Python running THIS script (our venv), so `-m gunicorn` finds our packages
```
- **In simple words:** Find the step 06 folder (where the apps live) and remember which Python is running us.
- **Exactly:** `Path(__file__).resolve().parents[1]` is the parent folder of this step (`09_api_development`), so `/ "06_model_api_flask_vs_fastapi"` gives the app folder. `sys.executable` is the venv's python path.
- **Why it is here:** Using `sys.executable -m gunicorn` guarantees the same venv packages; plain `gunicorn` might be a different install.

### 4. Table of server commands and the port  (`main.py` lines 35–42)
```python
# The exact commands you would type in a terminal. {port} is filled in below.
COMMANDS = {
    "gunicorn_1w":  [PY, "-m", "gunicorn", "flask_app:app", "--workers", "1", "--bind", "127.0.0.1:{port}", "--log-level", "warning"],
    "gunicorn_2w":  [PY, "-m", "gunicorn", "flask_app:app", "--workers", "2", "--bind", "127.0.0.1:{port}", "--log-level", "warning"],
    "gunicorn_1w4t": [PY, "-m", "gunicorn", "flask_app:app", "--workers", "1", "--threads", "4", "--bind", "127.0.0.1:{port}", "--log-level", "warning"],
    "uvicorn_1w":   [PY, "-m", "uvicorn", "fastapi_app:app", "--port", "{port}", "--log-level", "warning"],
}
PORT = 8123
```
- **In simple words:** Four recipes for starting servers, each as a list of words you would type in a terminal.
- **Exactly:** Each value is an argv list (command split into parts). `--workers N` = N processes; `--threads 4` = 4 threads per worker (gthread class); `flask_app:app` = module `flask_app`, variable `app`; `{port}` is a placeholder filled by `.format`. `PORT = 8123`.
- **Why it is here:** Keeping commands as data avoids repeated code and matches what you type by hand.

### 5. Server context manager: remember the command  (`main.py` lines 45–49)
```python
class Server:
    """Context manager: `with Server(...) as base_url:` starts the server and ALWAYS stops it afterwards."""

    def __init__(self, key: str):
        self.cmd = [part.format(port=PORT) for part in COMMANDS[key]]   # fill in {port}
```
- **In simple words:** A helper object that will start a server and always stop it. First it remembers which command to run.
- **Exactly:** `COMMANDS[key]` is a list; the list comprehension calls `part.format(port=PORT)` on each piece, turning `"127.0.0.1:{port}"` into `"127.0.0.1:8123"`.
- **Why it is here:** Without it each run would repeat start/stop code, and failures could leave servers alive.

### 6. __enter__: start the server and wait until ready  (`main.py` lines 51–61)
```python
    def __enter__(self) -> str:
        self.proc = subprocess.Popen(self.cmd, cwd=APP_DIR,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{PORT}"
        for _ in range(100):                          # wait up to ~20 s for the server to accept connections
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    return base
            except httpx.TransportError:              # connection refused = not up yet
                time.sleep(0.2)
        raise RuntimeError(f"server did not start: {self.cmd}")
```
- **In simple words:** Launch the server in the background and keep knocking until it answers.
- **Exactly:** `subprocess.Popen(cmd, cwd=APP_DIR, stdout/stderr=DEVNULL)` starts the process without blocking and hides its logs. Up to 100 tries x 0.2 s sleep (~20 s) of `httpx.get(base+"/health", timeout=1)`; a connection refused raises `httpx.TransportError`, caught, then retry. Returns the base URL `http://127.0.0.1:8123` (bound to the `as base` name). Raises `RuntimeError` if never up.
- **Why it is here:** Servers take a moment to boot and load the model; sending requests too early would fail with "connection refused".

### 7. __exit__: stop the server, force if needed  (`main.py` lines 63–68)
```python
    def __exit__(self, *exc_info):
        self.proc.terminate()                         # polite stop (SIGTERM); gunicorn also stops its workers
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()                          # force stop if it ignores us
```
- **In simple words:** When the `with` block ends, politely ask the server to stop and kill it if it does not.
- **Exactly:** `terminate()` sends SIGTERM; Gunicorn forwards it to its workers. `wait(timeout=10)` blocks until exit; `TimeoutExpired` triggers `kill()` (SIGKILL).
- **Why it is here:** Leaked servers would keep port 8123 busy and the next run would fail with "Address already in use".

### 8. timed_parallel: fire n requests at once and time them  (`main.py` lines 71–76)
```python
def timed_parallel(base: str, path: str, n: int = 4) -> float:
    """Send n requests AT THE SAME TIME and return total wall-clock seconds."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda _: httpx.get(base + path, timeout=30), range(n)))
    return time.perf_counter() - start
```
- **In simple words:** Send 4 requests at the same instant and measure how long until all are done.
- **Exactly:** `ThreadPoolExecutor(max_workers=n)` creates n threads; `pool.map(lambda _: httpx.get(...), range(n))` submits n calls; `list(...)` forces waiting for all; `perf_counter` difference is wall-clock seconds. Each call is a 1-second sleep endpoint, so ideal parallel = 1 s, fully serial = 4 s.
- **Why it is here:** This is the measurement that exposes concurrency: total time, not per-request time.

### 9. Part 1: how many distinct worker processes?  (`main.py` lines 79–83)
```python
if __name__ == "__main__":
    # Part 1: what is a worker?  Each worker is a separate OS process with its own copy of the app + model.
    with Server("gunicorn_2w") as base:
        pids = {httpx.get(base + "/whoami").json()["pid"] for _ in range(30)}   # set comprehension -> unique pids
        print(f"gunicorn 2 workers: distinct worker PIDs seen = {len(pids)}  {sorted(pids)}")
```
- **In simple words:** Start Gunicorn with 2 workers, ask "who are you?" 30 times, and count the different answers.
- **Exactly:** A set comprehension `{... ["pid"] for _ in range(30)}` collects unique pids. Real output: `gunicorn 2 workers: distinct worker PIDs seen = 2  [22778, 22779]`.
- **Why it is here:** Proves a worker is a separate OS process with its own memory and its own copy of the model.

### 10. Part 2: the experiment plan  (`main.py` lines 85–94)
```python
    # Part 2: same 4 simultaneous 1-second requests against different setups.
    print("\n4 parallel 1-second requests:")
    plan = [
        ("gunicorn  1 sync worker",           "gunicorn_1w",   "/slow?seconds=1"),
        ("gunicorn  2 sync workers",          "gunicorn_2w",   "/slow?seconds=1"),
        ("gunicorn  1 worker x 4 threads",    "gunicorn_1w4t", "/slow?seconds=1"),
        ("uvicorn   async def (await sleep)", "uvicorn_1w",    "/slow?seconds=1"),
        ("uvicorn   def (threadpool)",        "uvicorn_1w",    "/slow-sync?seconds=1"),
        ("uvicorn   async def + BLOCKING",    "uvicorn_1w",    "/slow-blocking?seconds=1"),
    ]
```
- **In simple words:** A list of six experiments: label, which server, which URL.
- **Exactly:** Each tuple `(label, key, path)`. `/slow?seconds=1` sleeps 1 s; `/slow-sync` is a plain `def` (thread pool); `/slow-blocking` is `async def` plus `time.sleep`.
- **Why it is here:** Same load against different server setups so the only variable is the concurrency model.

### 11. Run each experiment and print the time  (`main.py` lines 95–100)
```python
    for label, key, path in plan:
        with Server(key) as base:
            print(f"  {label:<36}{timed_parallel(base, path):5.1f} s")
    # TRY IT: change n=4 to n=8 in timed_parallel's default and predict the new numbers first.
    # TRY IT: add "--workers 4" to uvicorn_1w and run /whoami many times (needs an import string, as we have).
    # BREAK IT: set PORT to a port a server is already using and read "Address already in use".
```
- **In simple words:** For each setup: start the server, send 4 parallel requests, print the time, stop the server.
- **Exactly:** `{label:<36}` left-pads the label; `{...:5.1f}` prints width 5, one decimal. Real run: 1 sync worker 4.4 s; 2 sync workers 2.3 s; 1 worker x 4 threads 1.1 s; uvicorn async 1.1 s; uvicorn def 1.1 s; uvicorn async+blocking 4.3 s.
- **Why it is here:** The numbers show: workers multiply parallel capacity, threads/async overlap waiting, and a blocking call inside `async def` serialises everything.


## Commands to type yourself
```bash
cd 09_api_development/06_model_api_flask_vs_fastapi
gunicorn flask_app:app --workers 2 --bind 127.0.0.1:5001        # Flask (WSGI) under Gunicorn
gunicorn flask_app:app --workers 1 --threads 4 --bind 127.0.0.1:5001   # threaded worker
uvicorn fastapi_app:app --port 8000                             # FastAPI (ASGI), 1 process
uvicorn fastapi_app:app --workers 2 --port 8000                 # 2 processes
gunicorn fastapi_app:app -k uvicorn.workers.UvicornWorker -w 2 -b 127.0.0.1:8000   # Gunicorn as process manager, Uvicorn workers
```
`flask_app:app` = *file name* (no `.py`) + `:` + *variable name*. `--reload` (Uvicorn) is for development only. Never use `app.run()`/Flask debug mode in production.

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `subprocess.Popen([...])` | Start another program and keep control of it | `Popen(["ls"])` |
| `class Server:` with `__enter__`/`__exit__` | A **context manager**: `with Server(k) as base:` always runs cleanup, even on exceptions | — |
| `{... for _ in range(30)}` | **Set comprehension**: unique values only | `{x%3 for x in range(9)}` → `{0,1,2}` |
| `ThreadPoolExecutor.map` | Run a function over inputs in parallel threads | — |
| `_` as a variable name | "I don't need this value" | `for _ in range(3)` |
| `[part.format(port=PORT) for part in cmd]` | Fill `{port}` in each string | — |
| `try: ... finally:` / `__exit__` | Cleanup guarantee | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `subprocess.Popen(cmd, cwd=, stdout=DEVNULL)` | stdlib | Launch server in another process, working directory `cwd` |
| `proc.terminate()` / `proc.wait(timeout)` / `proc.kill()` | stdlib | Polite stop → wait → force stop |
| `httpx.get/Client` | 3rd-party | HTTP client; `timeout=` is essential |
| `time.perf_counter()` | stdlib | Wall-clock stopwatch |
| `gunicorn --workers N` | 3rd-party CLI | N separate OS processes, each with its own copy of the app |
| `gunicorn --threads T` | CLI | T threads inside each worker (threads share memory, one model copy) |
| `uvicorn --workers N` | CLI | N processes of the ASGI app |

## What the numbers teach (4 requests that each wait 1 s)
| Setup | Time | Why |
|---|---|---|
| 1 sync worker | 4 s | one request at a time → 1+1+1+1 |
| 2 sync workers | 2 s | two at a time |
| 1 worker × 4 threads | 1 s | threads overlap the *waiting* |
| `async def` + `await asyncio.sleep` | 1 s | event loop serves others while one waits |
| plain `def` in FastAPI | 1 s | FastAPI runs it in a thread pool (default ~40 threads) |
| `async def` + `time.sleep` | 4 s | **blocking call inside async blocks the whole event loop** — never do this |

### Workers and the model in memory
Each worker is a **separate process**, so a 2 GB model with 4 workers needs ~8 GB RAM (unless the OS can share pages via `--preload`, which helps only for read-only data). Python threads can't run CPU-bound code in parallel (the GIL), so *CPU-heavy inference scales by processes*, *waiting-heavy I/O scales by async/threads*. Rule of thumb for Gunicorn workers: `(2 × CPU cores) + 1` for I/O-bound apps; for model serving size by **memory and per-request CPU time**, then verify with a load test (module 10 step 08).

### Sync vs async: when it matters
- Calling a **remote LLM API** or DB (waiting): async/threads win big.
- Running a **local PyTorch/sklearn model** (computing): `async` does not speed it up; use `def` routes (thread pool keeps the event loop responsive) and more processes/replicas, or batch requests.

## Execution trace
`/slow-blocking` with 4 parallel clients on one Uvicorn worker: request 1 enters `async def`, calls `time.sleep(1)` (the single event-loop thread is stuck), requests 2–4 sit in the socket queue → completion at 1, 2, 3, 4 s → total 4.1 s. With `await asyncio.sleep(1)` the loop is free, all four start immediately → 1.0 s.

## Experiments
1. Change the default `n=4` to `n=8`; predict each number before running.
2. Run `gunicorn flask_app:app --workers 2` manually, then `for i in $(seq 10); do curl -s localhost:5001/whoami; done` — see two PIDs alternate.
3. Add `--preload` and read the startup logs: the model-loading line prints once in the master instead of once per worker.

## Debugging practice
Start a second server on the same port → `OSError: [Errno 48] Address already in use` (macOS). Find the owner with `lsof -i :8123` and stop it, or change the port. If `gunicorn` says `ModuleNotFoundError: No module named 'flask_app'`, you are in the wrong directory (use `--chdir`) — see module 01.

## Rebuild checklist
- [ ] Write the Gunicorn command for Flask and the Uvicorn command for FastAPI from memory.
- [ ] Explain why `async def` + `time.sleep` is a bug.
- [ ] Explain why workers multiply memory use.

## The new problem (cliffhanger)
Worker counts, ports and model paths are now scattered in commands and code. Changing them per environment must not mean editing code. Next: [08_config_and_env](../08_config_and_env/).

## Say it aloud
> **Say it aloud (interview box):** Flask and FastAPI define the app, while Gunicorn and Uvicorn run it, and a worker is a process with its own memory. More workers give more parallel requests but also more copies of the model, and async only overlaps waiting, not CPU-bound inference. I size by running a load test, because too few workers cause queueing and too many cost RAM.
