"""
load_test.py -- a tiny load tester: N concurrent clients hammer an endpoint for a few seconds; we report throughput and
latency percentiles. Real tools: locust, k6, wrk, ab.  Writing one yourself shows exactly what they measure.

RUN:    cd 10_ml_system_architecture/08_scalability_and_reliability && python load_test.py   (~20 s)

It starts the module-09 Flask API under Gunicorn with 1, 2 and 4 workers, calls GET /slow?seconds=0.05
(a stand-in for a 50 ms model), and prints the results. Expect throughput to scale roughly with workers
(~20 req/s per worker; measured numbers run slightly high because requests that finish just after the deadline are still counted) while per-request latency stays near 50 ms until workers are saturated -- then it queues up.
"""

import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

APP_DIR = Path(__file__).resolve().parents[2] / "09_api_development" / "06_model_api_flask_vs_fastapi"
PORT = 8125
BASE = f"http://127.0.0.1:{PORT}"


def client_loop(stop_at: float, latencies: list, errors: list) -> None:
    """One simulated user: send requests back-to-back until the deadline."""
    with httpx.Client(base_url=BASE, timeout=30) as client:
        while time.perf_counter() < stop_at:
            start = time.perf_counter()
            try:
                ok = client.get("/slow", params={"seconds": 0.05}).status_code == 200
            except httpx.HTTPError:
                ok = False
            (latencies if ok else errors).append((time.perf_counter() - start) * 1000)   # list.append is thread-safe in CPython


def run(workers: int, clients: int = 16, duration: float = 3.0) -> None:
    server = subprocess.Popen([sys.executable, "-m", "gunicorn", "flask_app:app", "--workers", str(workers),
                               "--bind", f"127.0.0.1:{PORT}", "--log-level", "error"], cwd=APP_DIR)
    try:
        for _ in range(100):
            try:
                httpx.get(BASE + "/health", timeout=1)
                break
            except httpx.TransportError:
                time.sleep(0.2)
        latencies: list = []
        errors: list = []
        stop_at = time.perf_counter() + duration
        threads = [threading.Thread(target=client_loop, args=(stop_at, latencies, errors)) for _ in range(clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        q = statistics.quantiles(latencies, n=100)       # 99 cut points -> q[49]=p50, q[94]=p95
        print(f"workers={workers}  clients={clients}  throughput={len(latencies)/duration:6.1f} req/s  "
              f"p50={q[49]:6.1f} ms  p95={q[94]:6.1f} ms  errors={len(errors)}")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    for w in (1, 2, 4):
        run(w)
    # TRY IT: raise clients to 64 and watch p95 explode on workers=1 (requests queue up waiting for a free worker).
    # TRY IT: point it at /predict instead (POST) -- a microsecond model makes HTTP overhead the bottleneck.
