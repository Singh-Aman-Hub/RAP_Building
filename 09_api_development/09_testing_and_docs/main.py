"""
GOAL:   Test a REAL running server over the network (not in-process): start uvicorn, call it with
        httpx, check status codes + JSON, read the auto docs, then shut the server down.
        This is a "smoke test" -- the quick check you run after deploying.

RUN:    cd 09_api_development/09_testing_and_docs
        python main.py            # smoke test with a real server
        pytest -v                 # the in-process unit tests (fast; no ports)
        bash curl_cheatsheet.sh   # same checks with curl (needs a server running)

EXPECTED OUTPUT (shape):
        GET /health -> 200 {...}
        POST /predict -> 200 {...}
        ...
        All smoke checks passed.

NEEDS:  uvicorn, httpx, pytest
"""

import subprocess
import sys
import time
from pathlib import Path

import httpx

APP_DIR = Path(__file__).resolve().parents[1] / "06_model_api_flask_vs_fastapi"
PORT = 8124
BASE = f"http://127.0.0.1:{PORT}"


def check(label: str, resp: httpx.Response, expected_status: int) -> None:
    print(f"{label:<28}-> {resp.status_code} {resp.text[:90]}")
    assert resp.status_code == expected_status, f"{label}: expected {expected_status}, got {resp.status_code}"


if __name__ == "__main__":
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "fastapi_app:app", "--port", str(PORT),
                               "--log-level", "warning"], cwd=APP_DIR)
    try:                                                   # try/finally guarantees the server is stopped
        for _ in range(100):                               # wait until it answers
            try:
                httpx.get(BASE + "/health", timeout=1)
                break
            except httpx.TransportError:
                time.sleep(0.2)

        with httpx.Client(base_url=BASE, timeout=5) as client:   # a Client reuses the connection and a base URL
            check("GET /health", client.get("/health"), 200)
            check("POST /predict", client.post("/predict", json={"text": "great, I love it"}), 200)
            check("POST /predict (empty)", client.post("/predict", json={"text": ""}), 422)
            check("GET /predict (wrong method)", client.get("/predict"), 405)
            check("GET /nope", client.get("/nope"), 404)
            check("GET /openapi.json", client.get("/openapi.json"), 200)
            check("GET /docs", client.get("/docs"), 200)
        print("All smoke checks passed.")
    finally:
        server.terminate()
        server.wait(timeout=10)
    # TRY IT: change an expected status to 201 and read the AssertionError message.
    # TRY IT: open http://127.0.0.1:8000/docs in a browser after `uvicorn fastapi_app:app` and use "Try it out".
