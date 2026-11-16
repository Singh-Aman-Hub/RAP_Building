# Step 02 — WSGI vs ASGI, written by hand

## The story so far
Step 01 left us parsing sockets by hand, with a server that cannot be swapped for a better one. We need a standard handshake between a server (owns the port) and an app (owns the logic).

## The pain point
If every framework invented its own way to talk to servers, each framework would only run on its own server. Your code would be locked in, and nobody could improve the server without touching every app.

## The spark
Agree on a contract: the app is just a callable and the server calls it. WSGI is the synchronous version `(environ, start_response)`; ASGI is the asynchronous version `(scope, receive, send)`. Write both by hand once and the words stop being scary.

## Goal
Remove the magic: a **web framework** (Flask/FastAPI) and a **web server** (Gunicorn/Uvicorn) talk through a tiny contract. WSGI is the synchronous contract; ASGI is the asynchronous one.

## Run it
```bash
cd 09_api_development/02_wsgi_asgi_by_hand
python main.py
uvicorn asgi_app:app --port 8000        # optional: serve the ASGI app with a real server
curl -i http://127.0.0.1:8000/hi
```
Real output (the first line is the WSGI reference server's access log):
```
127.0.0.1 - - [19/Oct/2026 11:03:00] "GET /hi HTTP/1.1" 200 25
WSGI  -> 200 OK b'Hello from WSGI, path=/hi'
ASGI  -> messages sent: [('http.response.start', 200), ('http.response.body', b'Hello from ASGI, path=/hi')]
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| WSGI | PEP 3333, Web Server Gateway Interface | No | Synchronous `(environ, start_response)` contract; one request occupies one worker. |
| ASGI | Asynchronous Server Gateway Interface | No | Async `(scope, receive, send)` contract; also covers WebSockets and lifespan events. |
| `environ` vs `scope` | request dict (WSGI) vs connection dict (ASGI) | Similar role | Both describe the request; WSGI also carries the body stream in it, ASGI delivers the body via `receive`. |
| Web server vs app server vs WSGI/ASGI server | gunicorn, uvicorn, waitress, nginx | Not the same | nginx is a reverse proxy/web server; gunicorn/uvicorn/waitress run your Python app through WSGI/ASGI. |
| Coroutine | `async def` function call result, awaitable | Related | A coroutine is the object returned by calling an `async def` function; it runs only when awaited or passed to `asyncio.run`. |
| Framework vs protocol | Flask/FastAPI vs WSGI/ASGI | No | Flask and FastAPI are libraries; WSGI and ASGI are the interfaces they implement. |

## Code, step by step

### 1. The file header  (`main.py` lines 1–14)

```python
"""
GOAL:   Understand WSGI vs ASGI by writing the smallest possible app of each kind by hand --
        no Flask, no FastAPI. Flask is "a nicer way to write a WSGI app"; FastAPI is
        "a nicer way to write an ASGI app".

RUN:    cd 09_api_development/02_wsgi_asgi_by_hand && python main.py
        Optional real ASGI server:  uvicorn asgi_app:app --port 8000   (see asgi_app.py)

EXPECTED OUTPUT (shape):
        WSGI  -> 200 OK  b'Hello from WSGI, path=/hi'
        ASGI  -> messages sent: [('http.response.start', 200), ('http.response.body', b'Hello from ASGI, path=/hi')]

NEEDS:  standard library only.
"""
```

- **In simple words:** A note saying what this demo is and what to expect.
- **Exactly:** Docstring only, no code runs. It states the idea: Flask is a nicer way to write a WSGI app, FastAPI a nicer way to write an ASGI app.
- **Why it is here:** Orientation for the reader.

### 2. Imports, including our own ASGI app  (`main.py` lines 16–21)

```python
import asyncio                                           # built-in: runs `async def` code
import threading
import urllib.request
from wsgiref.simple_server import make_server            # built-in: a reference WSGI server (for learning, not production)

from asgi_app import app as asgi_app                     # our hand-written ASGI app lives next door
```

- **In simple words:** Bring in the tools, plus the ASGI app written in the neighbouring file.
- **Exactly:** `asyncio` runs coroutines; `threading` runs the WSGI server in the background; `urllib.request` is the client; `wsgiref.simple_server.make_server` is the standard-library reference WSGI server. `from asgi_app import app as asgi_app` imports the object `app` from `asgi_app.py` and renames it to avoid clashing with other names. It works because Python puts the script's folder on `sys.path`, which is why you must run from this folder.
- **Why it is here:** Without the import there is no ASGI app to call. Without `make_server` there is nothing to serve the WSGI function over a real socket.

### 3. WSGI app signature and its contract (docstring)  (`main.py` lines 24–32)

```python
# ----------------------------- WSGI: a plain function ------------------------------
def wsgi_app(environ, start_response):
    """The WSGI contract: a callable taking (environ, start_response) and returning an iterable of bytes.

    environ        : dict with the request (REQUEST_METHOD, PATH_INFO, QUERY_STRING, wsgi.input ...)
    start_response : function you call ONCE with the status line and headers
    return value   : iterable of bytes = the response body
    WSGI is SYNCHRONOUS: one worker handles one request at a time from start to end.
    """
```

- **In simple words:** WSGI is a promise: "I am a function that takes the request and a reply-starter, and returns the body."
- **Exactly:** A WSGI app is any callable `(environ, start_response) -> iterable of bytes`. `environ` is a plain dict (`REQUEST_METHOD`, `PATH_INFO`, `QUERY_STRING`, `wsgi.input`...). `start_response` is a callback supplied by the server.
- **Why it is here:** This tiny interface is what lets any WSGI server (gunicorn, waitress, wsgiref) run any WSGI framework (Flask, Django).

### 4. WSGI body: build bytes, start the response, return  (`main.py` lines 33–37)

```python
    path = environ["PATH_INFO"]
    body = f"Hello from WSGI, path={path}".encode()      # f-string: {path} is replaced by its value
    start_response("200 OK", [("Content-Type", "text/plain"),
                              ("Content-Length", str(len(body)))])
    return [body]
```

- **In simple words:** Read the path, write a greeting, tell the server the status and headers, hand back the body.
- **Exactly:** `environ["PATH_INFO"]` is `"/hi"` in the run. The f-string gives `Hello from WSGI, path=/hi`; `.encode()` makes 25 bytes (the log line ends `200 25`). `start_response("200 OK", [(name, value), ...])` takes the status as a string and headers as a list of tuples; call it once, before returning. `return [body]` is a list of one bytes chunk.
- **Why it is here:** WSGI needs bytes, not str. Forget `start_response` or return a str and the server raises an error or sends a broken reply.

### 5. Start of the fake ASGI server: build the scope  (`main.py` lines 40–43)

```python
# ----------------- ASGI: drive the app by hand, no server needed -------------------
async def call_asgi(app, path: str) -> list:
    """Pretend to be an ASGI server: build a `scope`, a `receive` and a `send`, then call the app."""
    scope = {"type": "http", "method": "GET", "path": path, "headers": []}
```

- **In simple words:** To test an ASGI app without a real server we play the server's part ourselves.
- **Exactly:** `async def` defines a coroutine function. `scope` is a dict describing the connection: `{"type": "http", "method": "GET", "path": "/hi", "headers": []}` (real servers add more keys such as `query_string`, `server`). `sent = []` collects what the app sends.
- **Why it is here:** It shows that ASGI's input is just a dict plus two callables, nothing magical.

### 6. Define receive and send channels  (`main.py` lines 46–50)

```python
    async def receive():                                 # the server -> app channel (request body events)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):                             # the app -> server channel (response events)
        sent.append(message)
```

- **In simple words:** Two pipes: one carrying the request into the app, one carrying the response out.
- **Exactly:** `receive()` is an async function returning an event dict `{"type": "http.request", "body": b"", "more_body": False}` (an empty body, nothing more to come). `send(message)` appends each message the app emits to `sent`. A real server's `send` would write to the socket.
- **Why it is here:** The app is written against these two abstractions, so it works the same with a real server or a test harness.

### 7. Call the app and return what it sent  (`main.py` lines 52–53)

```python
    await app(scope, receive, send)                      # `await` = "wait for this async call to finish"
    return sent
```

- **In simple words:** Run the app, wait for it to finish, then give back its recorded replies.
- **Exactly:** `await app(scope, receive, send)` calls the coroutine and suspends our function until it completes. It returns `sent`, a list of two dicts in this run.
- **Why it is here:** Without `await` the call just creates a coroutine object that never runs (the BREAK IT experiment: a `RuntimeWarning: coroutine ... was never awaited`).

### 8. Serve the WSGI app on a real socket and call it  (`main.py` lines 56–62)

```python
if __name__ == "__main__":
    # --- WSGI over a real socket using the stdlib reference server ---
    httpd = make_server("127.0.0.1", 8766, wsgi_app)     # server + app glued together
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with urllib.request.urlopen("http://127.0.0.1:8766/hi") as r:
        print("WSGI  ->", r.status, r.reason, r.read())
    httpd.shutdown()
```

- **In simple words:** Start a real mini web server for the WSGI function, make one request, print the result, stop it.
- **Exactly:** `make_server("127.0.0.1", 8766, wsgi_app)` glues host, port and app. A thread runs `serve_forever`. `urlopen` fetches `/hi`. Printed: `WSGI  -> 200 OK b'Hello from WSGI, path=/hi'` (`r.status` 200, `r.reason` "OK", `r.read()` bytes). `wsgiref` also prints an access log line to stderr (`GET /hi HTTP/1.1" 200 25`). `httpd.shutdown()` stops it.
- **Why it is here:** Shows WSGI end to end over real HTTP, with the standard library only.

### 9. Drive the ASGI app by hand and summarise the messages  (`main.py` lines 64–69)

```python
    # --- ASGI called directly: shows the (scope, receive, send) protocol ---
    messages = asyncio.run(call_asgi(asgi_app, "/hi"))   # asyncio.run starts an event loop and runs the coroutine
    summary = [(m["type"], m.get("status", m.get("body"))) for m in messages]  # list comprehension
    print("ASGI  -> messages sent:", summary)
    # TRY IT: change the path to "/other" in both places and watch the body change.
    # BREAK IT: remove the `await` in front of app(...) and read the RuntimeWarning.
```

- **In simple words:** Call the ASGI app without any server and look at the two messages it produced.
- **Exactly:** `asyncio.run(coro)` creates an event loop, runs the coroutine to completion and returns its result. The list comprehension maps each message to `(type, status-or-body)`: `m.get("status", m.get("body"))` takes `status` if present, else `body`. Result: `[('http.response.start', 200), ('http.response.body', b'Hello from ASGI, path=/hi')]`.
- **Why it is here:** It proves ASGI's response is two events (start then body), which is what makes streaming possible: many body events.

### 10. asgi_app.py header: how to run it with uvicorn  (`asgi_app.py` lines 1–9)

```python
"""
A hand-written ASGI application. Run it with a real ASGI server:

    cd 09_api_development/02_wsgi_asgi_by_hand
    uvicorn asgi_app:app --port 8000
    curl -i http://127.0.0.1:8000/hi

`asgi_app:app` = "in the file asgi_app.py, use the object called app".
"""
```

- **In simple words:** A note: this file can also be run for real with uvicorn.
- **Exactly:** `uvicorn asgi_app:app` means "module `asgi_app`, object `app`". uvicorn is an ASGI server.
- **Why it is here:** Without this note people do not know how to run the file on a real server.

### 11. The ASGI app: signature, contract, and non-HTTP guard  (`asgi_app.py` lines 12–21)

```python
async def app(scope, receive, send):
    """ASGI contract: an ASYNC callable taking (scope, receive, send).

    scope   : dict describing the connection (type, method, path, headers ...)
    receive : `await receive()` gets incoming events (e.g. request body chunks)
    send    : `await send({...})` pushes response events to the server
    Because it is async, one worker can juggle thousands of waiting requests (e.g. waiting on a DB).
    """
    if scope["type"] != "http":                          # ASGI also covers websockets/lifespan; we ignore them
        return
```

- **In simple words:** An async function that receives the request description and two channels; ignore anything that is not plain HTTP.
- **Exactly:** `async def app(scope, receive, send)`. `scope["type"]` is `"http"`, `"websocket"` or `"lifespan"`; we just `return` for the latter two. Being async lets the server run other requests while this one awaits I/O.
- **Why it is here:** Real servers send `lifespan` events at startup; returning cleanly for unknown types keeps uvicorn happy.

### 12. Build the body and send the two response events  (`asgi_app.py` lines 22–26)

```python
    body = f"Hello from ASGI, path={scope['path']}".encode()
    # A response is TWO events: first the status+headers, then the body.
    await send({"type": "http.response.start", "status": 200,
                "headers": [(b"content-type", b"text/plain")]})
    await send({"type": "http.response.body", "body": body})
```

- **In simple words:** Write the greeting, then send the "status + headers" message and the "body" message.
- **Exactly:** `scope['path']` is `/hi`. Event 1: `http.response.start` with `status` 200 and `headers` as a list of **bytes** tuples (`(b"content-type", b"text/plain")`). Event 2: `http.response.body` with `body` bytes (`b'Hello from ASGI, path=/hi'`). `more_body` defaults to False so the response ends.
- **Why it is here:** Splitting start and body is what lets ASGI stream (send many body events with `more_body: True`). Headers must be bytes (unlike WSGI's str).


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `async def f(): ...` | Defines a **coroutine**: a function that can pause (`await`) so others can run | `async def f(): await g()` |
| `await x` | Pause here until `x` finishes, letting the event loop do other work meanwhile | — |
| `asyncio.run(coro())` | Start an event loop and run one coroutine to completion | `asyncio.run(main())` |
| f-string `f"path={path}"` | Insert variables into a string | `f"{1+1}"` → `2` |
| nested `async def receive()` | A function defined inside a function (a *closure*): it can use the outer variables (`sent`) | — |
| `[(m["type"], ...) for m in messages]` | **List comprehension**: build a list in one line | `[x*2 for x in range(3)]` |

## Libraries & functions
| Name | Kind | What it does | Notes |
|---|---|---|---|
| `wsgiref.simple_server.make_server(host, port, app)` | stdlib | Tiny WSGI server for learning | Not for production |
| `asyncio.run` | stdlib | Runs a coroutine | creates + closes an event loop |
| `uvicorn asgi_app:app` | 3rd-party CLI | Production ASGI server | `module:variable` import string |

## The two contracts side by side
| | WSGI | ASGI |
|---|---|---|
| Signature | `app(environ, start_response)` → iterable of bytes | `async app(scope, receive, send)` |
| Concurrency | Sync: one request occupies a worker until finished | Async: one worker can interleave many waiting requests |
| Protocols | HTTP only | HTTP + WebSockets + lifespan events |
| Frameworks | Flask, Django (classic) | FastAPI, Starlette, Django (async) |
| Servers | Gunicorn, uWSGI, Waitress | Uvicorn, Hypercorn, Daphne |

## Execution trace
WSGI: server parses the HTTP request → builds `environ` (`PATH_INFO="/hi"`) → calls `wsgi_app(environ, start_response)` → app calls `start_response("200 OK", headers)` and returns `[b"Hello from WSGI, path=/hi"]` → server writes status, headers, body.
ASGI: we play the server: build `scope={"type":"http","path":"/hi",...}` → `await app(scope, receive, send)` → app awaits `send({"type":"http.response.start","status":200,...})` then `send({"type":"http.response.body","body":...})` → we collected both events.

## Experiments
1. Change the path to `/other` in both places → body changes in both.
2. Make `wsgi_app` return a 404 for any path except `/hi`.
3. Run `uvicorn asgi_app:app` and `curl -i` — compare the headers you sent with those uvicorn added (`date`, `server`).

## Debugging practice
Remove `await` before `app(scope, receive, send)` in `call_asgi`: you get `RuntimeWarning: coroutine 'app' was never awaited` — calling an async function without `await` only creates the coroutine object; nothing runs. This is the #1 async beginner bug.

## Rebuild checklist
- [ ] Write the WSGI signature and what `start_response` is for.
- [ ] Write the ASGI signature and the two `send` events of a response.
- [ ] Say which of Flask/FastAPI is WSGI/ASGI and which server runs each.

## The new problem (cliffhanger)
A bare callable still makes you route paths and parse query strings yourself. We need a framework that hides the contract behind decorators. Next: [03_flask_basics](../03_flask_basics/).

## Say it aloud
> **Say it aloud (interview box):** Flask and FastAPI are frameworks, and a server such as Gunicorn or Uvicorn owns the port and calls the app through the WSGI or ASGI contract. WSGI is a synchronous callable, while ASGI is asynchronous and also supports WebSockets and streaming. Async only helps when the work is waiting, so CPU-bound model inference still needs more processes.
