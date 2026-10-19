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

import asyncio                                           # built-in: runs `async def` code
import threading
import urllib.request
from wsgiref.simple_server import make_server            # built-in: a reference WSGI server (for learning, not production)

from asgi_app import app as asgi_app                     # our hand-written ASGI app lives next door


# ----------------------------- WSGI: a plain function ------------------------------
def wsgi_app(environ, start_response):
    """The WSGI contract: a callable taking (environ, start_response) and returning an iterable of bytes.

    environ        : dict with the request (REQUEST_METHOD, PATH_INFO, QUERY_STRING, wsgi.input ...)
    start_response : function you call ONCE with the status line and headers
    return value   : iterable of bytes = the response body
    WSGI is SYNCHRONOUS: one worker handles one request at a time from start to end.
    """
    path = environ["PATH_INFO"]
    body = f"Hello from WSGI, path={path}".encode()      # f-string: {path} is replaced by its value
    start_response("200 OK", [("Content-Type", "text/plain"),
                              ("Content-Length", str(len(body)))])
    return [body]


# ----------------- ASGI: drive the app by hand, no server needed -------------------
async def call_asgi(app, path: str) -> list:
    """Pretend to be an ASGI server: build a `scope`, a `receive` and a `send`, then call the app."""
    scope = {"type": "http", "method": "GET", "path": path, "headers": []}
    sent = []                                            # we record what the app sends

    async def receive():                                 # the server -> app channel (request body events)
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):                             # the app -> server channel (response events)
        sent.append(message)

    await app(scope, receive, send)                      # `await` = "wait for this async call to finish"
    return sent


if __name__ == "__main__":
    # --- WSGI over a real socket using the stdlib reference server ---
    httpd = make_server("127.0.0.1", 8766, wsgi_app)     # server + app glued together
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    with urllib.request.urlopen("http://127.0.0.1:8766/hi") as r:
        print("WSGI  ->", r.status, r.reason, r.read())
    httpd.shutdown()

    # --- ASGI called directly: shows the (scope, receive, send) protocol ---
    messages = asyncio.run(call_asgi(asgi_app, "/hi"))   # asyncio.run starts an event loop and runs the coroutine
    summary = [(m["type"], m.get("status", m.get("body"))) for m in messages]  # list comprehension
    print("ASGI  -> messages sent:", summary)
    # TRY IT: change the path to "/other" in both places and watch the body change.
    # BREAK IT: remove the `await` in front of app(...) and read the RuntimeWarning.
