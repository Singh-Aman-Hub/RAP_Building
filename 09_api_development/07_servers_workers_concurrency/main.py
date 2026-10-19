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

import subprocess      # built-in: start other programs (here: servers) from Python
import sys
import time
from concurrent.futures import ThreadPoolExecutor   # built-in: fire several requests at the same time
from pathlib import Path

import httpx           # third-party HTTP client (modern requests-like API)

APP_DIR = Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"  # where the apps live
PY = sys.executable    # the Python running THIS script (our venv), so `-m gunicorn` finds our packages

# The exact commands you would type in a terminal. {port} is filled in below.
COMMANDS = {
    "gunicorn_1w":  [PY, "-m", "gunicorn", "flask_app:app", "--workers", "1", "--bind", "127.0.0.1:{port}", "--log-level", "warning"],
    "gunicorn_2w":  [PY, "-m", "gunicorn", "flask_app:app", "--workers", "2", "--bind", "127.0.0.1:{port}", "--log-level", "warning"],
    "gunicorn_1w4t": [PY, "-m", "gunicorn", "flask_app:app", "--workers", "1", "--threads", "4", "--bind", "127.0.0.1:{port}", "--log-level", "warning"],
    "uvicorn_1w":   [PY, "-m", "uvicorn", "fastapi_app:app", "--port", "{port}", "--log-level", "warning"],
}
PORT = 8123


class Server:
    """Context manager: `with Server(...) as base_url:` starts the server and ALWAYS stops it afterwards."""

    def __init__(self, key: str):
        self.cmd = [part.format(port=PORT) for part in COMMANDS[key]]   # fill in {port}

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

    def __exit__(self, *exc_info):
        self.proc.terminate()                         # polite stop (SIGTERM); gunicorn also stops its workers
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()                          # force stop if it ignores us


def timed_parallel(base: str, path: str, n: int = 4) -> float:
    """Send n requests AT THE SAME TIME and return total wall-clock seconds."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(lambda _: httpx.get(base + path, timeout=30), range(n)))
    return time.perf_counter() - start


if __name__ == "__main__":
    # Part 1: what is a worker?  Each worker is a separate OS process with its own copy of the app + model.
    with Server("gunicorn_2w") as base:
        pids = {httpx.get(base + "/whoami").json()["pid"] for _ in range(30)}   # set comprehension -> unique pids
        print(f"gunicorn 2 workers: distinct worker PIDs seen = {len(pids)}  {sorted(pids)}")

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
    for label, key, path in plan:
        with Server(key) as base:
            print(f"  {label:<36}{timed_parallel(base, path):5.1f} s")
    # TRY IT: change n=4 to n=8 in timed_parallel's default and predict the new numbers first.
    # TRY IT: add "--workers 4" to uvicorn_1w and run /whoami many times (needs an import string, as we have).
    # BREAK IT: set PORT to a port a server is already using and read "Address already in use".
