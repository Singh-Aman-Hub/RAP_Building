# Terms — Step 02: WSGI vs ASGI by hand

Folder: `09_api_development/02_wsgi_asgi_by_hand` · Files covered: `main.py`, `asgi_app.py`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [Web framework vs web server](#web-framework-vs-web-server-concept) · [WSGI](#wsgi-concept) · [ASGI](#asgi-concept) · [environ (WSGI)](#environ-wsgi-concept) · [start_response (WSGI)](#start_response-wsgi-concept) · [scope, receive, send (ASGI)](#scope-receive-send-asgi-concept) · [Synchronous vs asynchronous](#synchronous-vs-asynchronous-concept) · [async def / coroutine](#async-def-coroutine-syntax) · [await](#await-syntax) · [asyncio.run](#asynciorun-standard-library) · [Event loop](#event-loop-concept) · [Closure (nested function)](#closure-nested-function-syntax) · [wsgiref.simple_server.make_server](#wsgirefsimple_servermake_server-standard-library) · [uvicorn asgi_app:app](#uvicorn-asgi_appapp-command) · [Uvicorn](#uvicorn-third-party-uvicorn) · [curl -i](#curl--i-command) · [urllib.request.urlopen](#urllibrequesturlopen-standard-library) · [threading.Thread (daemon)](#threadingthread-daemon-standard-library) · [f-string](#f-string-syntax) · [List comprehension](#list-comprehension-syntax) · [dict.get with default](#dictget-with-default-python-built-in) · [RuntimeWarning: coroutine was never awaited](#runtimewarning-coroutine-was-never-awaited-concept)

---

## Web framework vs web server  `(concept)`
**1. Definition:** A web framework (Flask, FastAPI) provides routing, request parsing and helpers for writing application logic; a web server (Gunicorn, Uvicorn) listens on a network port, speaks HTTP and calls the framework's app through a standard interface.
**2. In plain words:** The framework is the code you write your endpoints in; the server is the program that actually accepts connections and hands each request to your code. Keeping them separate lets you swap servers or tune worker counts without touching your endpoints. In interviews, always explain the two roles separately.
**3. Example to quote:**
> "In production I run `gunicorn flask_app:app`: Flask is the framework with my routes, Gunicorn is the server that accepts connections and calls Flask."

**Seen in:** `WALKTHROUGH.md` goal and "Say it aloud"

## WSGI  `(concept)`
**1. Definition:** The Web Server Gateway Interface (PEP 3333), a synchronous contract in which a server calls `app(environ, start_response)` and the app returns an iterable of bytes.
**2. In plain words:** WSGI is the agreed handshake between classic Python servers and frameworks like Flask. It is synchronous: one worker handles one request from start to finish before taking the next. To serve more requests at once you add more worker processes or threads.
**3. Example to quote:**
> "Flask is a WSGI framework. My hand-written `wsgi_app(environ, start_response)` returned `[b'Hello from WSGI, path=/hi']`, and a server wrote that as the response."

**Seen in:** `wsgi_app` in `main.py`

## ASGI  `(concept)`
**1. Definition:** The Asynchronous Server Gateway Interface, an async contract in which a server awaits `app(scope, receive, send)`; it supports HTTP, WebSockets and lifespan events.
**2. In plain words:** ASGI is the async successor to WSGI. Because the app can pause while waiting for I/O, a single worker can interleave many requests. It does not make CPU-heavy work faster; heavy model inference still blocks unless run in another thread or process.
**3. Example to quote:**
> "FastAPI is ASGI. My `async def app(scope, receive, send)` sent two events: `http.response.start` with status 200, then `http.response.body`."

**Seen in:** `asgi_app.py`

## environ (WSGI)  `(concept)`
**1. Definition:** The dictionary a WSGI server passes to the app, containing the request details such as `REQUEST_METHOD`, `PATH_INFO`, `QUERY_STRING` and `wsgi.input`.
**2. In plain words:** It is the "whole request in one dict". Our app only needed `environ["PATH_INFO"]`, the path like `/hi`. Flask reads this dict for you and gives you the friendlier `request` object.
**3. Example to quote:**
> "In `wsgi_app` I read `environ['PATH_INFO']` to build the greeting; that is the path part of the URL."

**Seen in:** `wsgi_app`

## start_response (WSGI)  `(concept)`
**1. Definition:** A callable supplied by the WSGI server that the app calls once with a status string and a list of header tuples before returning the body.
**2. In plain words:** It is how a WSGI app says "the status is 200 OK and here are the headers". The body is then returned from the function as a list of bytes. Calling it twice, or not at all, breaks the protocol.
**3. Example to quote:**
> "`start_response('200 OK', [('Content-Type', 'text/plain'), ('Content-Length', str(len(body)))])` sets the status line and headers, then I `return [body]`."

**Seen in:** `wsgi_app`

## scope, receive, send (ASGI)  `(concept)`
**1. Definition:** The three arguments of an ASGI app: `scope` is a dict describing the connection, `receive` is an awaitable that yields incoming events, and `send` is an awaitable that pushes outgoing events.
**2. In plain words:** `scope` says who is calling and what they want (type, method, path). `receive` is the channel from client to app (e.g. body chunks) and `send` is the channel back. A response is two `send` events: start (status and headers) and body.
**3. Example to quote:**
> "My ASGI app checks `scope['type'] == 'http'`, then awaits `send` twice: once for status and headers, once for the body."

**Seen in:** `asgi_app.py`, `call_asgi`

## Synchronous vs asynchronous  `(concept)`
**1. Definition:** Synchronous code runs one step at a time, blocking while it waits; asynchronous code can suspend at `await` points so other work runs while waiting.
**2. In plain words:** Think of a cashier who serves one customer to the end (sync) versus one who takes an order, hands it to the kitchen and serves the next customer while waiting (async). Async only helps when the wait is for I/O such as a database or another API; it does not help if the CPU is busy computing.
**3. Example to quote:**
> "Async helps my API when it waits on a database, but a model forward pass is CPU-bound, so I'd still add worker processes."

**Seen in:** `WALKTHROUGH.md` contract table

## async def / coroutine  `(syntax)`
**1. Definition:** `async def` defines a coroutine function; calling it returns a coroutine object that only runs when awaited or scheduled on an event loop.
**2. In plain words:** An `async def` function can pause at `await`. Just calling it does not run its body; you must `await` it or pass it to `asyncio.run`. Forgetting `await` is the classic async bug and gives "coroutine was never awaited".
**3. Example to quote:**
> "`async def app(scope, receive, send)` is a coroutine. If I call it without `await`, Python warns 'coroutine app was never awaited' and nothing happens."

```python
async def f():
    return 1
```

**Seen in:** `app`, `call_asgi`, `receive`, `send`

## await  `(syntax)`
**1. Definition:** `await x` suspends the current coroutine until the awaitable `x` completes, letting the event loop run other tasks in the meantime.
**2. In plain words:** It is the "I'll wait here, but you can do other things" keyword. It can only be used inside an `async def`. Each `await send(...)` hands control back to the server until the message is delivered.
**3. Example to quote:**
> "I `await send({...})` for each response event; while the server flushes the data, it can work on other requests."

**Seen in:** `asgi_app.py`, `call_asgi`

## asyncio.run  `(standard library)`
**1. Definition:** `asyncio.run(coro)` creates a new event loop, runs the coroutine to completion, and closes the loop.
**2. In plain words:** It is the bridge from normal synchronous code (like a script's `__main__`) into async code. You usually call it once, at the top level.
**3. Example to quote:**
> "`messages = asyncio.run(call_asgi(asgi_app, '/hi'))` starts the event loop, runs my fake-server coroutine and returns what the app sent."

```python
import asyncio
print(asyncio.run(asyncio.sleep(0, "done")))   # done
```

**Seen in:** `if __name__ == "__main__":`

## Event loop  `(concept)`
**1. Definition:** The scheduler in an async program that runs coroutines and resumes each when the thing it awaited is ready.
**2. In plain words:** It is the single "traffic controller" that decides which coroutine runs next. A coroutine that never awaits (a long CPU loop) blocks the controller and freezes everything else on that loop.
**3. Example to quote:**
> "Uvicorn runs one event loop per worker; if my handler does a long computation without awaiting, every other request on that worker waits."

**Seen in:** `asyncio.run` explanation

## Closure (nested function)  `(syntax)`
**1. Definition:** A function defined inside another function that can read variables from the enclosing function's scope.
**2. In plain words:** `receive` and `send` are defined inside `call_asgi`, so `send` can append to the `sent` list defined just above it. It is a compact way to build small callbacks that remember context.
**3. Example to quote:**
> "My inner `send` function appends each message to the outer `sent` list, so after the call I can see everything the app sent."

**Seen in:** `call_asgi`

## wsgiref.simple_server.make_server  `(standard library)`
**1. Definition:** `make_server(host, port, app)` returns a small reference WSGI server bound to the given address and serving the given WSGI app.
**2. In plain words:** It lets you run a WSGI app with no installs to see it work over a real socket. It is intended for learning and testing, not production; use Gunicorn there.
**3. Example to quote:**
> "`make_server('127.0.0.1', 8766, wsgi_app)` glues the server and my function together, so `urlopen` on port 8766 gets 'Hello from WSGI'."

**Seen in:** `if __name__ == "__main__":`

## uvicorn asgi_app:app  `(command)`
**1. Definition:** A command that starts the Uvicorn ASGI server using an import string in the form `module:variable`, here the object `app` in `asgi_app.py`.
**2. In plain words:** The part before the colon is the Python file (without `.py`) and the part after is the variable holding the app. `--port 8000` chooses the port. Getting this string wrong gives an "Error loading ASGI app" message.
**3. Example to quote:**
> "`uvicorn asgi_app:app --port 8000` serves the app object in `asgi_app.py`; then `curl -i http://127.0.0.1:8000/hi` shows the response."

```bash
uvicorn asgi_app:app --port 8000
```

**Seen in:** `asgi_app.py` docstring, `WALKTHROUGH.md`

## Uvicorn  `(third-party: uvicorn)`
**1. Definition:** A fast ASGI web server for Python that runs an ASGI app (such as FastAPI) and handles HTTP and WebSockets.
**2. In plain words:** Uvicorn is the "server" half of the FastAPI pairing. It listens on a port, parses HTTP, and calls your app through the ASGI contract. For several CPU cores you run multiple workers.
**3. Example to quote:**
> "I serve my FastAPI app with Uvicorn; the framework is FastAPI, the server is Uvicorn."

**Seen in:** `WALKTHROUGH.md`

## curl -i  `(command)`
**1. Definition:** `curl` is a command-line HTTP client; the `-i` flag prints the response headers along with the body.
**2. In plain words:** It is the quickest way to test an endpoint from a terminal. With `-i` you also see the status line and headers, such as `content-type` and `date`.
**3. Example to quote:**
> "`curl -i http://127.0.0.1:8000/hi` shows `HTTP/1.1 200 OK`, the content-type I sent, and headers Uvicorn added."

```bash
curl -i http://127.0.0.1:8000/hi
```

**Seen in:** `asgi_app.py` docstring

## urllib.request.urlopen  `(standard library)`
**1. Definition:** `urlopen(url_or_request)` sends an HTTP request and returns a response object with `.status`, `.reason` and `.read()`.
**2. In plain words:** Used here as a tiny client to call the WSGI server. It must be used in a `with` block (or closed manually) so the connection is released.
**3. Example to quote:**
> "I call `urlopen('http://127.0.0.1:8766/hi')` and print `r.status`, `r.reason` and `r.read()` to prove the WSGI app answered."

**Seen in:** `if __name__ == "__main__":`

## threading.Thread (daemon)  `(standard library)`
**1. Definition:** `threading.Thread(target=f, daemon=True)` runs `f` on a background thread that will not keep the process alive on exit.
**2. In plain words:** `serve_forever` blocks, so the WSGI server runs in a background thread while the main thread acts as the client. Without a thread the script would never reach the client call.
**3. Example to quote:**
> "I start `httpd.serve_forever` in a daemon thread so the same script can then call it with `urlopen`."

**Seen in:** `if __name__ == "__main__":`

## f-string  `(syntax)`
**1. Definition:** A string literal prefixed with `f` in which expressions in braces are evaluated and inserted, e.g. `f"path={path}"`.
**2. In plain words:** It is the readable way to build text from variables. Here it builds the response body from the requested path. Remember to `.encode()` it before sending, because HTTP needs bytes.
**3. Example to quote:**
> "`f\"Hello from WSGI, path={path}\"` becomes `'Hello from WSGI, path=/hi'` when the path is `/hi`."

```python
path = "/hi"
print(f"path={path}")   # path=/hi
```

**Seen in:** `wsgi_app`, `app`

## List comprehension  `(syntax)`
**1. Definition:** An expression `[expr for item in iterable]` that builds a new list by evaluating `expr` for each item.
**2. In plain words:** It is a one-line loop that collects results. Here it turns each recorded ASGI message into a short `(type, status-or-body)` tuple for printing. If it grows too complex, a regular loop is clearer.
**3. Example to quote:**
> "`[(m['type'], m.get('status', m.get('body'))) for m in messages]` summarises both ASGI events into two tuples."

```python
print([x * 2 for x in range(3)])   # [0, 2, 4]
```

**Seen in:** `if __name__ == "__main__":`

## dict.get with default  `(Python built-in)`
**1. Definition:** `d.get(key, default)` returns `d[key]` if the key exists, otherwise `default` (or `None`), without raising `KeyError`.
**2. In plain words:** The two events have different keys: the start event has `status` and the body event has `body`. `m.get("status", m.get("body"))` picks whichever exists, with no crash.
**3. Example to quote:**
> "`m.get('status', m.get('body'))` returns the status for the start event and the body bytes for the body event."

```python
print({"a": 1}.get("b", "none"))   # none
```

**Seen in:** `if __name__ == "__main__":`

## RuntimeWarning: coroutine was never awaited  `(concept)`
**1. Definition:** A Python warning issued when a coroutine object is created by calling an `async def` function but is never awaited or scheduled.
**2. In plain words:** It means you forgot `await`, so the function body never ran. It is the first thing to check when an async call seems to do nothing.
**3. Example to quote:**
> "If I remove `await` in front of `app(scope, receive, send)`, I get 'coroutine app was never awaited' and no messages are collected."

**Seen in:** `BREAK IT` comment, `WALKTHROUGH.md` debugging practice
