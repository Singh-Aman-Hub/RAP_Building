# Terms — Step 07: Servers, workers and concurrency

Folder: `09_api_development/07_servers_workers_concurrency` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [Worker](#worker-concept) · [Gunicorn](#gunicorn-third-party-gunicorn) · [Uvicorn](#uvicorn-third-party-uvicorn) · [Import string (module:variable)](#import-string-modulevariable-syntax) · [--workers, --threads, --bind, --log-level](#--workers---threads---bind---log-level-command) · [Sync worker](#sync-worker-concept) · [Thread vs process](#thread-vs-process-concept) · [GIL (Global Interpreter Lock)](#gil-global-interpreter-lock-concept) · [Concurrency vs parallelism](#concurrency-vs-parallelism-concept) · [async def with await asyncio.sleep](#async-def-with-await-asynciosleep-syntax) · [Blocking call inside async def](#blocking-call-inside-async-def-concept) · [Plain def route in FastAPI (thread pool)](#plain-def-route-in-fastapi-thread-pool-concept) · [subprocess.Popen](#subprocesspopen-standard-library) · [proc.terminate / wait / kill](#procterminate-wait-kill-standard-library) · [sys.executable](#sysexecutable-standard-library) · [python -m module](#python--m-module-command) · [Context manager class (__enter__ / __exit__)](#context-manager-class-__enter__-__exit__-syntax) · [Readiness polling loop](#readiness-polling-loop-concept) · [httpx](#httpx-third-party-httpx) · [ThreadPoolExecutor](#threadpoolexecutor-standard-library) · [lambda](#lambda-syntax) · [Wall-clock timing](#wall-clock-timing-concept) · [Set comprehension](#set-comprehension-syntax) · [Underscore as throwaway variable](#underscore-as-throwaway-variable-syntax) · [String .format with named fields](#string-format-with-named-fields-python-built-in) · [Path(__file__).resolve().parents[1]](#path__file__resolveparents1-standard-library) · [Address already in use](#address-already-in-use-concept) · [Worker count sizing](#worker-count-sizing-concept) · [--preload (Gunicorn)](#--preload-gunicorn-command) · [gunicorn -k uvicorn.workers.UvicornWorker](#gunicorn--k-uvicornworkersuvicornworker-command) · [--reload (development only)](#--reload-development-only-command) · [lsof -i](#lsof--i-command)

---

## Worker  `(concept)`
**1. Definition:** A worker is a server process (or thread) that handles incoming requests; `--workers N` starts N separate OS processes, each with its own copy of the app and its memory.
**2. In plain words:** Think of workers as cashiers: more cashiers serve more customers at once. But each cashier needs their own desk, so each worker also needs its own copy of your app and model in memory. A 2 GB model with 4 workers needs roughly 8 GB of RAM.
**3. Example to quote:**
> "With `--workers 2` I saw two different pids from `/whoami`; that means two processes, each loaded the model, so memory doubled."

**Seen in:** `Part 1` of `main.py`, `WALKTHROUGH.md`

## Gunicorn  `(third-party: gunicorn)`
**1. Definition:** A production WSGI HTTP server for Python that manages a pool of worker processes running a WSGI app such as Flask.
**2. In plain words:** Gunicorn is the server in "Flask + Gunicorn". It listens on a port, accepts connections and forwards them to workers, restarting any that die. It handles WSGI only; for FastAPI you pair it with Uvicorn workers.
**3. Example to quote:**
> "I serve Flask with `gunicorn flask_app:app --workers 2`; Flask has my routes, Gunicorn handles sockets and processes."

**Seen in:** `COMMANDS`

## Uvicorn  `(third-party: uvicorn)`
**1. Definition:** A fast ASGI server for Python that runs async frameworks such as FastAPI on an event loop; it can run multiple worker processes with `--workers`.
**2. In plain words:** It is the server half of "FastAPI + Uvicorn". One Uvicorn process can interleave many waiting requests thanks to async. For more CPU parallelism you start several processes.
**3. Example to quote:**
> "`uvicorn fastapi_app:app --port 8000` runs my FastAPI app; with `--workers 2` it runs two processes."

**Seen in:** `COMMANDS["uvicorn_1w"]`

## Import string (module:variable)  `(syntax)`
**1. Definition:** The `flask_app:app` notation tells the server which Python module to import and which variable in it holds the application object.
**2. In plain words:** The part before the colon is the file name without `.py`; the part after is the variable. If the server runs in the wrong directory you get `ModuleNotFoundError: No module named 'flask_app'`; fix with the right working directory or `--chdir`.
**3. Example to quote:**
> "`gunicorn flask_app:app` means import `flask_app.py` and use the object called `app`."

**Seen in:** `COMMANDS`, `WALKTHROUGH.md`

## --workers, --threads, --bind, --log-level  `(command)`
**1. Definition:** Gunicorn flags: `--workers N` sets the number of processes, `--threads T` sets threads per worker, `--bind host:port` the listening address, and `--log-level` the verbosity.
**2. In plain words:** `--workers` multiplies memory use, `--threads` shares one copy of memory but only helps with waiting, not CPU work, and `--bind` decides who can reach the server. `--log-level warning` keeps the demo output quiet.
**3. Example to quote:**
> "`--workers 1 --threads 4` handled four 1-second waits in about 1 second, using one model copy."

```bash
gunicorn flask_app:app --workers 1 --threads 4 --bind 127.0.0.1:5001
```

**Seen in:** `COMMANDS`

## Sync worker  `(concept)`
**1. Definition:** A Gunicorn worker (the default type) that handles exactly one request at a time from start to finish.
**2. In plain words:** If a request takes 1 second and four arrive together, a single sync worker finishes them one after another, in about 4 seconds. Adding workers or threads shortens the queue.
**3. Example to quote:**
> "With one sync worker four 1-second requests took 4.1 s; with two workers 2.1 s."

**Seen in:** `plan`, `WALKTHROUGH.md` table

## Thread vs process  `(concept)`
**1. Definition:** A process has its own memory space; threads run inside a process and share its memory.
**2. In plain words:** Threads are cheap and share one model copy but, because of Python's GIL, cannot run Python code in parallel on several cores. Processes cost more memory but give true parallelism for CPU-bound work like model inference.
**3. Example to quote:**
> "For waiting-heavy work I add threads; for CPU-heavy inference I add processes, accepting the extra memory."

**Seen in:** `WALKTHROUGH.md` "Workers and the model in memory"

## GIL (Global Interpreter Lock)  `(concept)`
**1. Definition:** A lock in CPython that allows only one thread to execute Python bytecode at a time within a process.
**2. In plain words:** It means Python threads help when they are waiting (network, disk) but not when all of them are computing in pure Python. That is why CPU-bound serving scales with processes. Libraries like NumPy and PyTorch release the GIL in their native code, so the picture is more nuanced for them.
**3. Example to quote:**
> "Because of the GIL, adding threads won't speed up CPU-bound Python; I add worker processes instead."

**Seen in:** `WALKTHROUGH.md`

## Concurrency vs parallelism  `(concept)`
**1. Definition:** Concurrency means making progress on several tasks by interleaving them; parallelism means actually running several tasks at the same instant on different cores.
**2. In plain words:** One cashier juggling orders while the kitchen cooks is concurrency; two cashiers serving at once is parallelism. Async and threads give concurrency for waiting work; multiple processes give parallelism.
**3. Example to quote:**
> "Uvicorn's single process handled four waiting requests concurrently in 1 s, but it doesn't use several cores unless I add workers."

**Seen in:** `WALKTHROUGH.md`

## async def with await asyncio.sleep  `(syntax)`
**1. Definition:** An `async def` function that awaits `asyncio.sleep` pauses only itself, letting the event loop serve other requests meanwhile.
**2. In plain words:** It models waiting on a database or remote API without blocking others. Four parallel requests finish in about 1 second total.
**3. Example to quote:**
> "`/slow` uses `await asyncio.sleep`, so four simultaneous requests all complete in about one second."

**Seen in:** `plan` entry "uvicorn async def"

## Blocking call inside async def  `(concept)`
**1. Definition:** Calling a blocking function such as `time.sleep` inside `async def` freezes the entire event loop for its duration.
**2. In plain words:** Only one thread runs the event loop, so a blocking call stops every other request on that worker. It is the classic async mistake and made four requests take about 4 seconds.
**3. Example to quote:**
> "`/slow-blocking` uses `time.sleep` in an async function, so the 4 requests ran back to back in 4.1 s."

**Seen in:** `plan` entry "uvicorn async def + BLOCKING"

## Plain def route in FastAPI (thread pool)  `(concept)`
**1. Definition:** FastAPI runs a non-async `def` route in a thread from an internal thread pool, so blocking code does not freeze the event loop.
**2. In plain words:** You can write ordinary blocking code (like calling a sklearn model) in a plain `def` route and still handle several requests at once, up to the pool size (about 40 by default).
**3. Example to quote:**
> "`/slow-sync` is a plain `def` with `time.sleep`, and four parallel requests still finished in about 1 second."

**Seen in:** `plan`, `WALKTHROUGH.md`

## subprocess.Popen  `(standard library)`
**1. Definition:** `subprocess.Popen(cmd, cwd=..., stdout=..., stderr=...)` starts another program as a child process and returns a handle for controlling it.
**2. In plain words:** It lets Python start Gunicorn or Uvicorn the same way you would type the command, so the script can test them. `cwd` sets the working directory; `DEVNULL` discards the server's output.
**3. Example to quote:**
> "`Popen(self.cmd, cwd=APP_DIR, stdout=DEVNULL, stderr=DEVNULL)` launches the server in the folder where `flask_app.py` lives."

**Seen in:** `Server.__enter__`

## proc.terminate / wait / kill  `(standard library)`
**1. Definition:** `terminate()` sends SIGTERM to politely ask a process to stop, `wait(timeout)` blocks until it exits, and `kill()` force-stops it.
**2. In plain words:** A good shutdown sequence is: ask nicely, wait a bit, then force if needed. Gunicorn also stops its workers on SIGTERM. `subprocess.TimeoutExpired` is raised if the wait times out.
**3. Example to quote:**
> "I `terminate()` the server, `wait(timeout=10)`, and only `kill()` if it ignored me, so no stray servers are left running."

**Seen in:** `Server.__exit__`

## sys.executable  `(standard library)`
**1. Definition:** The path of the Python interpreter running the current script.
**2. In plain words:** Launching `[sys.executable, "-m", "gunicorn", ...]` guarantees the child uses the same virtual environment, so it finds the same installed packages. Using a bare `gunicorn` might pick a different install.
**3. Example to quote:**
> "`PY = sys.executable` makes the child servers use my venv's Python, so `-m gunicorn` finds the packages I installed."

**Seen in:** `PY`

## python -m module  `(command)`
**1. Definition:** `python -m gunicorn ...` runs an installed module as a script using that Python interpreter.
**2. In plain words:** It avoids PATH confusion, because the module is looked up in the interpreter you chose.
**3. Example to quote:**
> "`python -m uvicorn fastapi_app:app` uses exactly the interpreter I invoked, so there's no version mix-up."

**Seen in:** `COMMANDS`

## Context manager class (__enter__ / __exit__)  `(syntax)`
**1. Definition:** A class defining `__enter__` and `__exit__` can be used in a `with` statement; `__exit__` runs on leaving the block even if an exception occurred.
**2. In plain words:** `with Server("gunicorn_2w") as base:` starts the server, gives you its URL, and always stops it afterwards, so a failing test cannot leave a server running.
**3. Example to quote:**
> "My `Server` class starts the process in `__enter__` and terminates it in `__exit__`, guaranteeing cleanup."

**Seen in:** `class Server`

## Readiness polling loop  `(concept)`
**1. Definition:** Repeatedly calling a health endpoint until it answers, with a time limit, to know when a server is ready.
**2. In plain words:** A server needs time to start (and load its model). The loop tries `/health`, treats "connection refused" as "not yet", sleeps 0.2 s and retries, giving up after about 20 seconds with a clear error.
**3. Example to quote:**
> "I poll `/health` up to 100 times with 0.2 s sleeps, and raise `RuntimeError` with the command if the server never starts."

**Seen in:** `Server.__enter__`

## httpx  `(third-party: httpx)`
**1. Definition:** A modern Python HTTP client with a requests-like API, supporting timeouts, async and HTTP/2.
**2. In plain words:** It is used here to call the servers. Always pass `timeout=`, because the default behaviour can hang on a stuck server. `httpx.TransportError` covers connection failures.
**3. Example to quote:**
> "`httpx.get(base + '/health', timeout=1)` checks readiness; a `TransportError` means the server isn't up yet."

**Seen in:** `Server.__enter__`, `timed_parallel`

## ThreadPoolExecutor  `(standard library)`
**1. Definition:** `concurrent.futures.ThreadPoolExecutor(max_workers=n)` manages a pool of threads; `pool.map(fn, inputs)` runs `fn` over inputs concurrently.
**2. In plain words:** It lets the test script fire four requests at the same moment to see how the server copes. Wrapping `pool.map` in `list(...)` forces all results to complete before timing stops.
**3. Example to quote:**
> "`list(pool.map(lambda _: httpx.get(...), range(4)))` sends four simultaneous requests and waits for all of them."

**Seen in:** `timed_parallel`

## lambda  `(syntax)`
**1. Definition:** `lambda args: expression` creates a small anonymous function.
**2. In plain words:** `lambda _: httpx.get(...)` ignores its input and makes a request; handy when `map` needs a function but you do not want to name one.
**3. Example to quote:**
> "`lambda _: httpx.get(base + path, timeout=30)` is a one-line function that sends one request."

```python
print((lambda x: x + 1)(2))   # 3
```

**Seen in:** `timed_parallel`

## Wall-clock timing  `(concept)`
**1. Definition:** Measuring elapsed real time between a start and end point, here with `time.perf_counter()`.
**2. In plain words:** For concurrency tests the total time for all requests matters (about 4.1 s versus 1.0 s), not per-request CPU time.
**3. Example to quote:**
> "I note `perf_counter()` before and after the four parallel requests and print the difference as the total seconds."

**Seen in:** `timed_parallel`

## Set comprehension  `(syntax)`
**1. Definition:** `{expr for x in iterable}` builds a set, keeping only unique values.
**2. In plain words:** Collecting the pid from 30 responses into a set leaves only the distinct pids, which shows how many workers answered.
**3. Example to quote:**
> "`{httpx.get(...).json()['pid'] for _ in range(30)}` gave two pids, proving two worker processes."

```python
print({x % 3 for x in range(9)})   # {0, 1, 2}
```

**Seen in:** Part 1

## Underscore as throwaway variable  `(syntax)`
**1. Definition:** `_` is a conventional name for a value you do not use, as in `for _ in range(30)`.
**2. In plain words:** It tells readers the loop variable is irrelevant; you just want the loop to repeat.
**3. Example to quote:**
> "`for _ in range(100)` repeats the readiness check up to 100 times without caring about the index."

**Seen in:** `Server.__enter__`, Part 1

## String .format with named fields  `(Python built-in)`
**1. Definition:** `"...{port}...".format(port=8123)` replaces named placeholders in a string.
**2. In plain words:** The command templates contain `{port}`; a list comprehension fills in the real port for every part of the command.
**3. Example to quote:**
> "`[part.format(port=PORT) for part in COMMANDS[key]]` turns `127.0.0.1:{port}` into `127.0.0.1:8123`."

```python
print("127.0.0.1:{port}".format(port=8123))
```

**Seen in:** `Server.__init__`

## Path(__file__).resolve().parents[1]  `(standard library)`
**1. Definition:** `Path(__file__)` is this script's path, `.resolve()` makes it absolute, and `.parents[1]` is the folder two levels up from the file.
**2. In plain words:** It lets the script find the sibling folder `06_model_api_flask_vs_fastapi` no matter which directory you run it from.
**3. Example to quote:**
> "`Path(__file__).resolve().parents[1] / '06_model_api_flask_vs_fastapi'` points to step 06 from anywhere."

**Seen in:** `APP_DIR`

## Address already in use  `(concept)`
**1. Definition:** An `OSError` (errno 48 on macOS) raised when a program tries to listen on a port that another process already holds.
**2. In plain words:** Only one program can own a port. Find the owner with `lsof -i :8123` and stop it, or choose another port.
**3. Example to quote:**
> "Starting a second server on port 8123 gives 'Address already in use'; `lsof -i :8123` shows which process holds it."

**Seen in:** `BREAK IT` comment, `WALKTHROUGH.md`

## Worker count sizing  `(concept)`
**1. Definition:** Choosing the number of workers from CPU cores, memory and per-request CPU time, then verifying with a load test.
**2. In plain words:** For waiting-heavy apps a common rule is `(2 x cores) + 1` workers. For model serving, memory per copy of the model is usually the limit, so you size by RAM and measure.
**3. Example to quote:**
> "With a 2 GB model and 8 GB free, I'd start with 2 or 3 workers and confirm with a load test."

**Seen in:** `WALKTHROUGH.md`

## --preload (Gunicorn)  `(command)`
**1. Definition:** A Gunicorn flag that loads the application in the master process before forking workers.
**2. In plain words:** Read-only data such as model weights can then be shared between workers through copy-on-write, possibly saving memory and startup time. It helps only for data that isn't modified.
**3. Example to quote:**
> "With `--preload`, my model-loading print appears once in the master instead of once per worker."

**Seen in:** `WALKTHROUGH.md` experiment 3

## gunicorn -k uvicorn.workers.UvicornWorker  `(command)`
**1. Definition:** Runs a FastAPI (ASGI) app under Gunicorn as process manager by selecting Uvicorn's worker class with `-k`.
**2. In plain words:** You get Gunicorn's process supervision with Uvicorn's async serving. It is a common production setup.
**3. Example to quote:**
> "`gunicorn fastapi_app:app -k uvicorn.workers.UvicornWorker -w 2` gives me two ASGI workers managed by Gunicorn."

**Seen in:** `WALKTHROUGH.md` commands

## --reload (development only)  `(command)`
**1. Definition:** A Uvicorn flag that restarts the server when source files change.
**2. In plain words:** It is convenient while developing but adds overhead and should not be used in production.
**3. Example to quote:**
> "I use `--reload` locally, and drop it in the Docker command."

**Seen in:** `WALKTHROUGH.md`

## lsof -i  `(command)`
**1. Definition:** A command that lists open files and network connections; `lsof -i :PORT` shows which process is using a port.
**2. In plain words:** It is the quick way to find who owns a port when you see 'Address already in use'.
**3. Example to quote:**
> "`lsof -i :8123` printed the pid holding the port, so I could stop it."

```bash
lsof -i :8123
```

**Seen in:** `WALKTHROUGH.md` debugging practice
