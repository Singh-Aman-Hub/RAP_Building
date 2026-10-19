"""
GOAL:  See HTTP status codes, timeouts and connection errors using a tiny LOCAL server (no internet needed).

RUN:   cd 01_debugging_and_environment/06_http_and_network && python main.py
       Crashing version (EXPECTED):  python broken_http.py

EXPECTED OUTPUT: status codes 200/404/500, a Timeout, and a ConnectionError each handled separately.

NEEDS: requests (installed in .venv).
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer   # standard-library web server

import requests  # 3rd-party HTTP client: requests.get(url, timeout=...) returns a Response


class Handler(BaseHTTPRequestHandler):
    """Pretend API with four behaviours."""

    def do_GET(self):                       # called for every GET request
        if self.path == "/ok":
            self._reply(200, b'{"status": "healthy"}')
        elif self.path == "/boom":
            self._reply(500, b'{"error": "server crashed"}')
        elif self.path == "/slow":
            time.sleep(2)                   # simulates a model taking 2 s to respond
            self._reply(200, b'{"status": "finally"}')
        else:
            self._reply(404, b'{"error": "no such route"}')

    def _reply(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):           # silence the default per-request logging
        pass


def start_server(port):
    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()  # run in background thread
    return server


if __name__ == "__main__":
    server = start_server(8765)
    base = "http://127.0.0.1:8765"

    # 1) Status codes. 2xx = success, 4xx = YOUR request is wrong, 5xx = THE SERVER failed.
    for path in ["/ok", "/missing", "/boom"]:
        r = requests.get(base + path, timeout=3)      # ALWAYS set a timeout (default is "wait forever")
        print(f"GET {path:8s} -> {r.status_code}  body={r.json()}")

    # 2) raise_for_status() turns 4xx/5xx into an exception (HTTPError)
    try:
        requests.get(base + "/boom", timeout=3).raise_for_status()
    except requests.HTTPError as err:
        print("HTTPError:", err)

    # 3) Timeout: server too slow
    try:
        requests.get(base + "/slow", timeout=0.5)
    except requests.Timeout as err:
        print("Timeout:", type(err).__name__)

    # 4) Connection error: nothing is listening on that port
    try:
        requests.get("http://127.0.0.1:9/ok", timeout=1)
    except requests.ConnectionError as err:
        print("ConnectionError: server not running / wrong port / wrong host")

    server.shutdown()

# TRY IT: change the timeout in step 3 to 5 and see the call succeed (and take 2 s).
# BREAK IT: change base to port 8766 -> ConnectionError on every call. Which line of the traceback names the port?
