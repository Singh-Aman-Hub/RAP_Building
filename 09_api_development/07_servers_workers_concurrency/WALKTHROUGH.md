# Step 07 — Gunicorn, Uvicorn, workers and concurrency

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

## Say it aloud
"Flask and FastAPI define the app; Gunicorn and Uvicorn run it. A worker is a process with its own memory, so more workers means more parallel requests but also more copies of the model. Async lets one worker overlap waiting, which helps with network calls but not CPU-bound inference. I confirm sizing with a load test. Trade-off: more workers cost RAM; too few cause queueing and high tail latency."
