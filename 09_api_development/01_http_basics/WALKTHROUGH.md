# Step 01 — HTTP basics with the standard library

## The story so far
Chapter 20 ended with an agent that works in a notebook, and nobody else can call it. For another program to use it (or any model) there must be a shared language over the network. That language is HTTP, and this step shows it with nothing hiding it. Previous step: [08_agent_harness](../../08_agent_harness/).

## The pain point
A notebook cell can only be run by the person who owns the notebook. A teammate, a web page or another service needs a way to send a question over the network and get an answer back, and most people have never seen what actually travels on the wire.

## The spark
HTTP is just text: a request (method, path, headers, body) and a response (status, headers, body). If you can write a tiny server that reads one and writes the other, no framework will ever feel like magic.

## Goal
See a request and a response with your own eyes, before any framework hides them: a server that answers `GET /health` and `POST /echo`, and a client that checks status codes.

## Run it
```bash
cd 09_api_development/01_http_basics
python main.py
```
Real output:
```
GET  /health   -> 200 {'status': 'ok'}
POST /echo     -> 200 {'you_sent': {'text': 'hello'}}
POST /echo bad -> 400 {'error': 'body is not valid JSON'}
GET  /nope     -> 404 {'error': 'not found'}
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| HTTP method / verb | GET, POST, PUT, DELETE, PATCH | Yes, these are the verbs | The method is the verb on the request line; the "path" is the noun it acts on. |
| Status code | HTTP status, response code, `err.code` | Yes | Three-digit number; 2xx ok, 4xx client mistake, 5xx server mistake. |
| Header vs body | Metadata vs payload | No | Headers describe the message (type, length); the body is the data itself. |
| `localhost` | `127.0.0.1`, loopback | Nearly | localhost is a name that resolves to 127.0.0.1 (or ::1 for IPv6); the IP is the address itself. |
| `Content-Type` vs `Content-Length` | MIME type vs byte count | No | What the bytes are vs how many of them there are. |
| 400 vs 404 | Bad request vs not found | No | 400: the request itself is malformed; 404: well-formed but the resource does not exist. |
| `HTTPServer` vs `BaseHTTPRequestHandler` | server vs handler | No | The server owns the socket and loop; a handler instance deals with one request. |

## Code, step by step

### 1. The file header: goal, how to run, expected output  (`main.py` lines 1–15)

```python
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
```

- **In simple words:** A note to the reader. Python ignores it when running; it tells you what the file is for.
- **Exactly:** A triple-quoted string at the top of a module is its docstring (`__doc__`). It runs no code. It lists the four expected lines and says only the standard library is needed.
- **Why it is here:** Without it you must read all the code to know what to expect from a run.

### 2. Import the standard-library pieces we need  (`main.py` lines 17–21)

```python
import json                                              # built-in: text <-> Python dict/list
import threading                                         # built-in: run the server in the background
import urllib.error                                      # built-in: HTTPError (raised for 4xx/5xx)
import urllib.request                                    # built-in: a basic HTTP client
from http.server import BaseHTTPRequestHandler, HTTPServer  # built-in: a minimal web server
```

- **In simple words:** Pull in the toolboxes: JSON, threads, a web client, and a web server.
- **Exactly:** `json` converts between Python objects and JSON text. `threading` runs the server in the background. `urllib.request` is the client; `urllib.error` gives `HTTPError`. `http.server` supplies `HTTPServer` (listens on a socket) and `BaseHTTPRequestHandler` (parses one HTTP request).
- **Why it is here:** Without them there is no server and no client. Using only the standard library means no pip install, so HTTP is visible with nothing hiding it.

### 3. Declare the handler class (one instance per request)  (`main.py` lines 24–29)

```python
class Handler(BaseHTTPRequestHandler):
    """One instance of this class handles ONE incoming request.

    `class Handler(BaseHTTPRequestHandler)` = inheritance: we get all the HTTP parsing for
    free and only override the methods for the HTTP verbs we care about.
    """
```

- **In simple words:** Think of a waiter: the server creates a fresh waiter for every customer request, and the waiter knows how to read the order and reply.
- **Exactly:** `class Handler(BaseHTTPRequestHandler)` is inheritance: the parent parses the request line, headers and socket I/O, giving us `self.path`, `self.headers`, `self.rfile` (input stream) and `self.wfile` (output stream). `HTTPServer` builds a new `Handler` object for each request.
- **Why it is here:** Without inheriting, you would parse raw bytes like `GET /health HTTP/1.1\r\n...` yourself.

### 4. Helper that sends a JSON response  (`main.py` lines 31–37)

```python
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")       # dict -> JSON text -> bytes (HTTP sends bytes)
        self.send_response(status)                       # status line, e.g. "HTTP/1.0 200 OK"
        self.send_header("Content-Type", "application/json")  # tells the client how to read the body
        self.send_header("Content-Length", str(len(body)))    # tells the client how many bytes follow
        self.end_headers()                               # blank line = end of headers
        self.wfile.write(body)                           # the body itself
```

- **In simple words:** One place that writes a reply correctly: status first, then labels (headers), then the content (body).
- **Exactly:** `json.dumps(payload)` turns a dict into text; `.encode("utf-8")` makes bytes. For `{"status": "ok"}` that is 16 bytes. `send_response(200)` writes the status line (`HTTP/1.0 200 OK`; this class defaults to HTTP/1.0). Two `send_header` calls add `Content-Type: application/json` and `Content-Length: 16`. `end_headers()` writes the blank line `\r\n` that ends the headers. `wfile.write(body)` sends the body. The order is fixed: status, headers, blank line, body.
- **Why it is here:** Skip `Content-Length` and the client does not know where the body ends. Skip the blank line and the client reads your body as a header. Having every route call this helper stops each one repeating those five lines.

### 5. Handle GET requests: /health or 404  (`main.py` lines 39–43)

```python
    def do_GET(self):                                    # called automatically for GET requests
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})
```

- **In simple words:** A receptionist: if asked for the health desk, answer OK; anything else, say there is no such desk.
- **Exactly:** The server calls `do_GET` automatically when the request method is `GET` (the name is `do_` + method). `self.path` is the URL path including any query string, e.g. `"/health"`. Match gives `200 {"status": "ok"}`, otherwise `404 {"error": "not found"}`.
- **Why it is here:** This is routing in its rawest form: a manual `if` on the path. Flask and FastAPI replace it with decorators. Without the `else`, unknown paths would get no response and the client would hang.

### 6. Handle POST: only the /echo path exists  (`main.py` lines 45–48)

```python
    def do_POST(self):                                   # called automatically for POST requests
        if self.path != "/echo":
            self._send_json(404, {"error": "not found"})
            return
```

- **In simple words:** Same receptionist for deliveries: only the /echo door is open.
- **Exactly:** `do_POST` runs for `POST`. A path other than `/echo` returns 404 JSON and `return` stops the method early, so nothing is read afterwards.
- **Why it is here:** Without the early return, a POST to a wrong path would go on and try to echo.

### 7. Read the request body by its declared length  (`main.py` lines 49–50)

```python
        length = int(self.headers.get("Content-Length", 0))  # how many body bytes to read
        raw = self.rfile.read(length)                    # read the request body (bytes)
```

- **In simple words:** The client says "my message is N bytes long"; we read exactly N bytes.
- **Exactly:** `self.headers.get("Content-Length", 0)` returns the header as a string (default `0`); `int(...)` converts it. For the demo body `{"text": "hello"}` it is 17. `self.rfile.read(17)` returns exactly 17 bytes (`bytes`). For `{not json` it is 9.
- **Why it is here:** A socket is a stream with no built-in end marker. Reading without a length blocks forever waiting for more data. This is why `Content-Length` exists.

### 8. Parse the JSON, return 400 if it is garbage, else echo  (`main.py` lines 51–56)

```python
        try:
            data = json.loads(raw)                       # bytes/str -> Python object
        except json.JSONDecodeError:
            self._send_json(400, {"error": "body is not valid JSON"})  # 400 = client's fault
            return
        self._send_json(200, {"you_sent": data})
```

- **In simple words:** Try to understand the message; if it is nonsense, tell the client the mistake was theirs (400); otherwise repeat it back.
- **Exactly:** `json.loads(raw)` accepts bytes or str and returns a Python object (`{'text': 'hello'}`). On invalid input it raises `json.JSONDecodeError`; we catch that and send `400 {'error': 'body is not valid JSON'}` and `return`. 400 means "the client's request is malformed". Success is `200 {'you_sent': {'text': 'hello'}}`.
- **Why it is here:** Without `try/except` bad JSON would crash the handler method and the client would get a dropped connection or a 500 instead of an explained 400. Choosing the right status code is a core API skill.

### 9. Silence the default request log  (`main.py` lines 58–59)

```python
    def log_message(self, format, *args):                # silence the default per-request log line
        pass
```

- **In simple words:** Stop the server from printing a line for every request, so our demo output stays clean.
- **Exactly:** `BaseHTTPRequestHandler.log_message(format, *args)` normally writes to stderr. Overriding it with `pass` makes it a no-op. (Step 02 uses `wsgiref`, which does not have this override, which is why its run shows a log line.)
- **Why it is here:** Without it, four log lines would be mixed into the four result lines.

### 10. The client helper: send a request, return (status, JSON)  (`main.py` lines 62–70)

```python
def call(method: str, url: str, body: bytes | None = None) -> tuple[int, dict]:
    """Tiny client helper: returns (status_code, parsed_json)."""
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:        # `with` closes the connection for us
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as err:                # urllib raises on 4xx/5xx; we still want the body
        return err.code, json.loads(err.read())
```

- **In simple words:** A tiny browser: send a request, then give back the number and the message, even when the server said "error".
- **Exactly:** `urllib.request.Request(url, data=body, method=method, headers=...)` builds the request object; `data=None` for GET. `urlopen(req)` sends it and returns a response object used as a context manager (closed on exit). `resp.status` is an int and `resp.read()` is bytes, parsed by `json.loads`. For 4xx/5xx, `urllib` raises `HTTPError`, which is also a response: `err.code` is the status and `err.read()` the body. The function returns the tuple `(status, dict)`.
- **Why it is here:** Without the `except HTTPError` branch the 400 and 404 cases would crash the script instead of showing the status code. Without it we could not see error bodies.

### 11. Start the server in a background thread  (`main.py` lines 73–76)

```python
if __name__ == "__main__":                               # only runs when executed directly, not when imported
    server = HTTPServer(("127.0.0.1", 8765), Handler)    # (host, port): 127.0.0.1 = this computer only
    threading.Thread(target=server.serve_forever, daemon=True).start()  # serve in background thread
    base = "http://127.0.0.1:8765"
```

- **In simple words:** Open the shop (server) and let it serve customers while the main program is the customer.
- **Exactly:** `if __name__ == "__main__":` is true only when the file is run directly. `HTTPServer(("127.0.0.1", 8765), Handler)` binds a socket to host 127.0.0.1 (loopback: only this machine) port 8765 and remembers the handler class. `serve_forever` loops accepting requests; `Thread(..., daemon=True).start()` runs it in a second thread so the main thread can continue. `daemon=True` means the thread dies with the program. `base` is the URL prefix.
- **Why it is here:** `serve_forever()` blocks. Without the thread, the script would never reach the client calls. Without the `__main__` guard, importing the file would start a server.

### 12. Fire four requests and print the results  (`main.py` lines 78–81)

```python
    print("GET  /health   ->", *call("GET", base + "/health"))
    print("POST /echo     ->", *call("POST", base + "/echo", json.dumps({"text": "hello"}).encode()))
    print("POST /echo bad ->", *call("POST", base + "/echo", b"{not json"))   # BREAK IT: send garbage
    print("GET  /nope     ->", *call("GET", base + "/nope"))
```

- **In simple words:** Ask the server four questions: one good, one echo, one with garbage, one for a missing page.
- **Exactly:** `call(...)` returns `(status, dict)` and `print("...", *tuple)` unpacks it into two arguments. Real output: `GET /health -> 200 {'status': 'ok'}`, `POST /echo -> 200 {'you_sent': {'text': 'hello'}}`, `POST /echo bad -> 400 {'error': 'body is not valid JSON'}`, `GET /nope -> 404 {'error': 'not found'}`. `json.dumps({...}).encode()` makes the body bytes.
- **Why it is here:** This is the proof: one success, one POST with a body, and two different error classes (client-malformed 400, not-found 404).

### 13. Stop the server; ideas to try  (`main.py` lines 82–84)

```python
    # TRY IT: add an `elif self.path == "/time":` branch that returns the current time.
    # TRY IT: change the port to 8766 in BOTH places and re-run.
    server.shutdown()                                    # stop the server cleanly
```

- **In simple words:** Close the shop. The two TRY IT lines are suggested experiments.
- **Exactly:** `server.shutdown()` makes `serve_forever` return. It must be called from a different thread than the one running `serve_forever` (here: the main thread), otherwise it deadlocks.
- **Why it is here:** Without it the port stays bound until the process exits; in a longer script a second run would fail with `Address already in use`.


## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `class Handler(BaseHTTPRequestHandler):` | **Inheritance** — `Handler` gets everything the parent class can do, and we override only `do_GET`/`do_POST` | `class Dog(Animal): ...` |
| `self` | The current object; `self.path` is *this* request's path | `self.path` |
| `def _send_json(...)` | A leading underscore is a convention for "internal helper, don't call from outside" | `def _helper(): ...` |
| `-> None`, `: int`, `dict \| None` | **Type hints**: documentation for humans/tools, not enforced at runtime | `def f(x: int) -> str:` |
| `b"..."` / `.encode()` | **bytes**: HTTP sends bytes, not text; `.encode("utf-8")` converts str → bytes | `"hi".encode()` → `b'hi'` |
| `with ... as resp:` | **Context manager**: guarantees cleanup (closing the connection) even on errors | `with open(f) as fh:` |
| `try / except HTTPError` | Catch an error and keep going | `try: int("x")\nexcept ValueError: ...` |
| `if __name__ == "__main__":` | Run this block only when the file is executed directly, not imported | — |
| `print("a", *call(...))` | `*` **unpacks** a tuple into separate arguments | `print(*(1, 2))` → `1 2` |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `http.server.HTTPServer` | stdlib | Listens on a socket, creates a `Handler` per request | `(host, port)`, handler class | server object |
| `server.serve_forever()` | stdlib | Loop: accept → handle → repeat | — | never (until `shutdown`) |
| `threading.Thread(target=..., daemon=True)` | stdlib | Run something in the background | `target`, `daemon` | thread |
| `json.dumps / json.loads` | stdlib | dict ⇄ JSON text | the object / text | str / object |
| `urllib.request.Request/urlopen` | stdlib | A bare-bones HTTP client | `data` (bytes), `method`, `headers` | response object |
| `self.send_response(code)` / `send_header` / `end_headers` / `wfile.write` | stdlib | Write the status line, headers, blank line, body — in that order | status int | — |

## Execution trace (input → processing → output)
`POST /echo` with body `{"text": "hello"}`:
1. **Input:** client sends `POST /echo HTTP/1.1`, header `Content-Type: application/json`, header `Content-Length: 17`, blank line, body bytes.
2. **Processing:** `do_POST` runs → reads `Content-Length` (17) → `self.rfile.read(17)` → `json.loads(...)` gives `{'text': 'hello'}`.
3. **Output:** `_send_json(200, {'you_sent': {...}})` writes `HTTP/1.0 200 OK`, `Content-Type`, `Content-Length`, blank line, then the JSON body. The client prints `200 {'you_sent': {'text': 'hello'}}`.

The anatomy to remember: **request** = method + path + headers + (optional) body; **response** = status code + headers + body.

## Experiments (change ONE thing)
1. Add `elif self.path == "/time":` returning `{"time": ...}` → new endpoint appears.
2. Change the port in both places → still works; change in only one → `Connection refused` (`URLError`).
3. Send a GET to `/echo` → 404, because only `do_POST` knows `/echo` (method matters!).

## Debugging practice
Remove `.encode("utf-8")` in `_send_json`: `self.wfile.write(body)` raises `TypeError: a bytes-like object is required, not 'str'`. Read the traceback bottom-up: last line = error, line above = where. Fix: encode the string.

## Rebuild checklist
- [ ] Write a handler class with `do_GET` returning JSON from memory.
- [ ] Remember the response order: status → headers → blank line → body.
- [ ] Explain 200 / 201 / 400 / 404 / 405 / 422 / 500.

## The new problem (cliffhanger)
This stdlib server handles one request at a time, and you parsed and encoded everything by hand. Real servers must run any Python web app, so how does a server talk to your code in a standard way? Next: [02_wsgi_asgi_by_hand](../02_wsgi_asgi_by_hand/).

## Say it aloud
> **Say it aloud (interview box):** HTTP is request and response text: a request has a method, path, headers and optional body, and a response has a status code, headers and body. Python's `http.server` shows those mechanics by hand, and frameworks like Flask and FastAPI do the same parsing for me. Its limit is that it serves one request at a time with no routing or validation, so it is only for learning.
