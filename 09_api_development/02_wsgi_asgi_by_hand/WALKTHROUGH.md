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
