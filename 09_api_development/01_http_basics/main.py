"""
GOAL:   See HTTP with your own eyes: start a tiny web server using only the standard library,
        send GET and POST requests to it, and inspect status codes, headers and bodies.

RUN:    cd 09_api_development/01_http_basics && python main.py
        (from repo root: python 09_api_development/01_http_basics/main.py)

EXPECTED OUTPUT (shape):
        GET  /health   -> 200 {'status': 'ok'}
        POST /echo     -> 200 {'you_sent': {'text': 'hello'}}
        POST /echo bad -> 400 {'error': 'body is not valid JSON'}
        GET  /nope     -> 404 {'error': 'not found'}

NEEDS:  Python standard library only. No internet.
"""

import json                                              # built-in: text <-> Python dict/list
import threading                                         # built-in: run the server in the background
import urllib.error                                      # built-in: HTTPError (raised for 4xx/5xx)
import urllib.request                                    # built-in: a basic HTTP client
from http.server import BaseHTTPRequestHandler, HTTPServer  # built-in: a minimal web server


class Handler(BaseHTTPRequestHandler):
    """One instance of this class handles ONE incoming request.

    `class Handler(BaseHTTPRequestHandler)` = inheritance: we get all the HTTP parsing for
    free and only override the methods for the HTTP verbs we care about.
    """

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")       # dict -> JSON text -> bytes (HTTP sends bytes)
        self.send_response(status)                       # status line, e.g. "HTTP/1.0 200 OK"
        self.send_header("Content-Type", "application/json")  # tells the client how to read the body
        self.send_header("Content-Length", str(len(body)))    # tells the client how many bytes follow
        self.end_headers()                               # blank line = end of headers
        self.wfile.write(body)                           # the body itself

    def do_GET(self):                                    # called automatically for GET requests
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):                                   # called automatically for POST requests
        if self.path != "/echo":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))  # how many body bytes to read
        raw = self.rfile.read(length)                    # read the request body (bytes)
        try:
            data = json.loads(raw)                       # bytes/str -> Python object
        except json.JSONDecodeError:
            self._send_json(400, {"error": "body is not valid JSON"})  # 400 = client's fault
            return
        self._send_json(200, {"you_sent": data})

    def log_message(self, format, *args):                # silence the default per-request log line
        pass


def call(method: str, url: str, body: bytes | None = None) -> tuple[int, dict]:
    """Tiny client helper: returns (status_code, parsed_json)."""
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:        # `with` closes the connection for us
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:                # urllib raises on 4xx/5xx; we still want the body
        return err.code, json.loads(err.read())


if __name__ == "__main__":                               # only runs when executed directly, not when imported
    server = HTTPServer(("127.0.0.1", 8765), Handler)    # (host, port): 127.0.0.1 = this computer only
    threading.Thread(target=server.serve_forever, daemon=True).start()  # serve in background thread
    base = "http://127.0.0.1:8765"

    print("GET  /health   ->", *call("GET", base + "/health"))
    print("POST /echo     ->", *call("POST", base + "/echo", json.dumps({"text": "hello"}).encode()))
    print("POST /echo bad ->", *call("POST", base + "/echo", b"{not json"))   # BREAK IT: send garbage
    print("GET  /nope     ->", *call("GET", base + "/nope"))
    # TRY IT: add an `elif self.path == "/time":` branch that returns the current time.
    # TRY IT: change the port to 8766 in BOTH places and re-run.
    server.shutdown()                                    # stop the server cleanly
