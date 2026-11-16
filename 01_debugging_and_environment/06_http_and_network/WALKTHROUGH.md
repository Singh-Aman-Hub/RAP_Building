# Walkthrough 06 — HTTP errors, timeouts, connection failures

## Goal
Distinguish client errors (4xx), server errors (5xx), slow servers (timeout) and unreachable servers (connection error) — and handle each differently. Prepares you for [module 09](../../09_api_development/) and LLM API calls in [module 08](../../08_agent_harness/).

## Run it
```bash
cd 01_debugging_and_environment/06_http_and_network
python main.py            # local fake API on port 8765, offline
python broken_http.py     # EXPECTED crash: ReadTimeout (port 8766)
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| HTTP status code | response code, status | Same | 2xx ok, 3xx redirect, 4xx client mistake, 5xx server failure. |
| `Timeout` vs `ConnectionError` | slow vs unreachable | No | Timeout: server reachable but too slow; ConnectionError: nobody answered. |
| `ConnectTimeout` vs `ReadTimeout` | phase of the timeout | No | Connect timeout = could not open the connection; read timeout = connected but no data in time. |
| `HTTPError` vs `ConnectionError` | bad status vs no connection | No | HTTPError only appears if you call `raise_for_status()`. |
| client vs server | caller vs responder | No | `requests` is the client; `http.server` is the server. |
| `localhost` vs `127.0.0.1` | loopback address | Nearly | `localhost` is a name that resolves to `127.0.0.1` (or `::1` for IPv6). |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–10)

```python
"""
GOAL:  See HTTP status codes, timeouts and connection errors using a tiny LOCAL server (no internet needed).

RUN:   cd 01_debugging_and_environment/06_http_and_network && python main.py
       Crashing version (EXPECTED):  python broken_http.py

EXPECTED OUTPUT: status codes 200/404/500, a Timeout, and a ConnectionError each handled separately.

NEEDS: requests (installed in .venv).
"""
```

- **In simple words:** Label for the file: HTTP outcomes shown with a fake local server, so no internet is needed.
- **Exactly:** Docstring; nothing runs.
- **Why it is here:** Keeps the lesson reproducible offline.

### 2. Imports: threads, time, web server, requests  (`main.py` lines 12–16)

```python
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer   # standard-library web server

import requests  # 3rd-party HTTP client: requests.get(url, timeout=...) returns a Response
```

- **In simple words:** Gather what we need to run a tiny server in the background and talk to it.
- **Exactly:** `threading` runs the server beside the main code; `time.sleep` simulates slowness; `http.server` (standard library) provides `HTTPServer` and `BaseHTTPRequestHandler`; `requests` is the third-party HTTP client.
- **Why it is here:** We need both a server and a client to observe status codes, timeouts and refused connections.

### 3. Handler.do_GET: four fake routes  (`main.py` lines 19–31)

```python
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
```

- **In simple words:** The server's brain: look at the URL and decide what to answer.
- **Exactly:** `do_GET` is called by the base class for every GET; `self.path` is the URL path. `/ok` -> 200 `{"status": "healthy"}`; `/boom` -> 500; `/slow` sleeps 2 s then 200; anything else -> 404.
- **Why it is here:** One server gives us success, client error (4xx), server error (5xx) and slowness on demand.

### 4. Helper to send a reply, and silence logging  (`main.py` lines 33–40)

```python
    def _reply(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):           # silence the default per-request logging
        pass
```

- **In simple words:** Build the response: status line, header, blank line, body. Hide the noisy default log.
- **Exactly:** `send_response(code)` writes the status line; `send_header("Content-Type", ...)` adds a header; `end_headers()` finishes the header section; `wfile.write(body)` sends bytes (`b'...'`). `log_message` is overridden with `pass` so requests are not printed.
- **Why it is here:** An HTTP response is exactly status + headers + body; a missing `end_headers()` would hang the client.

### 5. Start the server in a background thread  (`main.py` lines 43–46)

```python
def start_server(port):
    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()  # run in background thread
    return server
```

- **In simple words:** Run the server so the main code can keep going and act as the client.
- **Exactly:** `HTTPServer(("127.0.0.1", port), Handler)` binds to the local machine on a port. `serve_forever` runs in a `daemon=True` thread (it dies with the program). Returns the server object so it can be shut down.
- **Why it is here:** Without a thread, `serve_forever()` would block and the client code would never run.

### 6. Main guard, start server, set base URL  (`main.py` lines 49–51)

```python
if __name__ == "__main__":
    server = start_server(8765)
    base = "http://127.0.0.1:8765"
```

- **In simple words:** Run the demo only when executed directly, using port 8765.
- **Exactly:** `if __name__ == "__main__":` skips this block when `broken_http.py` does `from main import start_server`. `base` is `http://127.0.0.1:8765`.
- **Why it is here:** Lets other files import `start_server` without side effects.

### 7. Status codes: 200, 404, 500  (`main.py` lines 53–56)

```python
    # 1) Status codes. 2xx = success, 4xx = YOUR request is wrong, 5xx = THE SERVER failed.
    for path in ["/ok", "/missing", "/boom"]:
        r = requests.get(base + path, timeout=3)      # ALWAYS set a timeout (default is "wait forever")
        print(f"GET {path:8s} -> {r.status_code}  body={r.json()}")
```

- **In simple words:** Request three routes and show the status and body.
- **Exactly:** `requests.get(url, timeout=3)` returns a `Response`; `r.status_code` is an int, `r.json()` parses the body. Real output: `GET /ok -> 200 body={'status': 'healthy'}`, `GET /missing -> 404 ...`, `GET /boom -> 500 ...`. Note `requests` does NOT raise for 404/500 by itself.
- **Why it is here:** Shows 2xx = success, 4xx = your request is wrong, 5xx = the server failed. `timeout` has no default: without it a call can wait forever.

### 8. raise_for_status turns errors into exceptions  (`main.py` lines 58–62)

```python
    # 2) raise_for_status() turns 4xx/5xx into an exception (HTTPError)
    try:
        requests.get(base + "/boom", timeout=3).raise_for_status()
    except requests.HTTPError as err:
        print("HTTPError:", err)
```

- **In simple words:** Ask requests to raise if the status was 4xx or 5xx.
- **Exactly:** `raise_for_status()` raises `requests.HTTPError`. Real output: `HTTPError: 500 Server Error: Internal Server Error for url: http://127.0.0.1:8765/boom`.
- **Why it is here:** Without it a failed call looks like a normal response and the bug shows up later when you parse the body.

### 9. Timeout: server too slow  (`main.py` lines 64–68)

```python
    # 3) Timeout: server too slow
    try:
        requests.get(base + "/slow", timeout=0.5)
    except requests.Timeout as err:
        print("Timeout:", type(err).__name__)
```

- **In simple words:** Allow only half a second for a server that needs two.
- **Exactly:** `timeout=0.5` is the seconds to wait for the connect and for each read. `/slow` sleeps 2 s, so `requests.Timeout` fires (specifically `ReadTimeout`). Output: `Timeout: ReadTimeout`.
- **Why it is here:** Model servers can be slow; a timeout lets you retry or fail fast instead of hanging.

### 10. ConnectionError: nobody listening  (`main.py` lines 70–74)

```python
    # 4) Connection error: nothing is listening on that port
    try:
        requests.get("http://127.0.0.1:9/ok", timeout=1)
    except requests.ConnectionError as err:
        print("ConnectionError: server not running / wrong port / wrong host")
```

- **In simple words:** Call a port where nothing runs.
- **Exactly:** Port 9 on localhost has no server, so the connection is refused: `requests.ConnectionError`. Output: `ConnectionError: server not running / wrong port / wrong host`.
- **Why it is here:** Separates "server answered with an error" from "could not reach the server at all".

### 11. Shut the server down  (`main.py` lines 76–76)

```python
    server.shutdown()
```

- **In simple words:** Stop the background server.
- **Exactly:** `server.shutdown()` ends `serve_forever`.
- **Why it is here:** Clean exit; frees the port.

### 12. TRY IT and BREAK IT comments  (`main.py` lines 78–79)

```python
# TRY IT: change the timeout in step 3 to 5 and see the call succeed (and take 2 s).
# BREAK IT: change base to port 8766 -> ConnectionError on every call. Which line of the traceback names the port?
```

- **In simple words:** Experiments for you (comments only).
- **Exactly:** Port 8766 has no server in main.py so every call gives `ConnectionError`; the traceback text names `port=8766`.
- **Why it is here:** Practice reading connection errors.

### 13. broken_http.py: docstring  (`broken_http.py` lines 1–4)

```python
"""
EXPECTED TO CRASH. A client that forgot error handling.  Run it, read the traceback bottom-up, then
add specific except branches for Timeout and HTTPError and print a helpful message for each.
"""
```

- **In simple words:** Says this client has no error handling; read the traceback bottom-up and add specific except branches.
- **Exactly:** Docstring only.
- **Why it is here:** Defines the exercise.

### 14. broken_http: reuse the fake server  (`broken_http.py` lines 5–6)

```python
from main import start_server           # reuse the local fake API from main.py
import requests
```

- **In simple words:** Import the server starter from main.py, and the HTTP client.
- **Exactly:** `from main import start_server` runs `main.py` as a module; the `__main__` guard keeps its demo from running. Also `import requests`.
- **Why it is here:** Reuses the fake API and shows why the main guard matters.

### 15. broken_http: start a server and call /slow with a short timeout  (`broken_http.py` lines 8–10)

```python
start_server(8766)
resp = requests.get("http://127.0.0.1:8766/slow", timeout=0.5)   # server needs 2 s
print(resp.json()["status"])
```

- **In simple words:** Start on port 8766, then ask the slow route with a half-second limit.
- **Exactly:** `start_server(8766)` begins serving. `requests.get(..., timeout=0.5)` waits 0.5 s but `/slow` needs 2 s, so the call raises `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=8766): Read timed out. (read timeout=0.5)`. The traceback is long: low-level frames (`socket.py` `TimeoutError: timed out`) at the top, then urllib3, then requests; chained with "The above exception was the direct cause of the following exception". Your own line is line 9; line 10 never runs. Fix: wrap in `try/except requests.Timeout` and `requests.HTTPError`, or increase the timeout.
- **Why it is here:** Teaches reading a long chained traceback: take the last line, then find the first frame that is in YOUR file.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `threading.Thread(target=fn, daemon=True).start()` | Run `fn` in the background; daemon threads stop when the main program ends |
| `class Handler(BaseHTTPRequestHandler): def do_GET(self)` | Handler method called for each GET request |
| `if __name__ == "__main__":` | Lets `broken_http.py` import `start_server` without re-running the demo |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Returns |
|---|---|---|---|
| `http.server.HTTPServer` | standard lib | Tiny web server | server object (`serve_forever`, `shutdown`) |
| `requests.get(url, timeout=)` | 3rd-party | HTTP GET | `Response` (`.status_code`, `.json()`, `.text`) |
| `Response.raise_for_status()` | 3rd-party | Raises `requests.HTTPError` on 4xx/5xx | `None` |
| `requests.Timeout`, `requests.ConnectionError`, `requests.HTTPError` | 3rd-party | Exception classes | — |

## Execution trace (real output)
```
GET /ok       -> 200  body={'status': 'healthy'}
GET /missing  -> 404  body={'error': 'no such route'}
GET /boom     -> 500  body={'error': 'server crashed'}
HTTPError: 500 Server Error: Internal Server Error for url: http://127.0.0.1:8765/boom
Timeout: ReadTimeout
ConnectionError: server not running / wrong port / wrong host
```
Meaning of codes: 200 OK, 201 created, 400 bad request, 401/403 auth, 404 not found, 422 validation failed (FastAPI), 429 too many requests (retry later), 500 server bug, 503 overloaded/unavailable.
`broken_http.py` ends with `requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=8766): Read timed out. (read timeout=0.5)`.
Handling plan: timeout -> retry with back-off (maybe raise `timeout`); 429/503 -> retry later; 4xx -> don't retry, fix the request; ConnectionError -> check URL/port/server running.

## Experiments
1. Increase `timeout` to 5 in the `/slow` call: success after ~2 s.
2. Use port 8766 for the base URL in `main.py` -> `ConnectionError` for all calls.
3. Call `requests.get(base + "/ok")` without `timeout` and think about what happens if the server hangs forever.

## Debugging practice
Take `broken_http.py`: wrap the request in `try/except requests.Timeout`, print "server too slow (>0.5 s)", and retry once with `timeout=5`. Verify the second attempt prints the status.

## Rebuild checklist
- [ ] Explain 2xx/4xx/5xx
- [ ] Always set a timeout; catch Timeout vs ConnectionError vs HTTPError separately
- [ ] Use `curl -i URL` to inspect status and headers from the terminal

## Say it aloud
"4xx means the request is wrong, 5xx means the server failed. A timeout means the server was reachable but slow; a connection error means nothing answered at that host/port. I always set timeouts and handle each failure separately, retrying only transient ones."
Trade-off: aggressive retries can overload an already struggling server; use back-off and limits.
