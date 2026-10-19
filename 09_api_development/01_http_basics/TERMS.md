# Terms — Step 01: HTTP basics

Folder: `09_api_development/01_http_basics` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [HTTP request and response](#http-request-and-response-concept) · [GET vs POST](#get-vs-post-concept) · [HTTP status code](#http-status-code-concept) · [HTTP header](#http-header-concept) · [Content-Type](#content-type-concept) · [Content-Length](#content-length-concept) · [json (module)](#json-module-standard-library) · [json.dumps](#jsondumps-standard-library) · [json.loads](#jsonloads-standard-library) · [json.JSONDecodeError](#jsonjsondecodeerror-standard-library) · [http.server](#httpserver-standard-library) · [HTTPServer](#httpserver-standard-library) · [BaseHTTPRequestHandler](#basehttprequesthandler-standard-library) · [Class inheritance](#class-inheritance-syntax) · [self.path](#selfpath-standard-library) · [do_GET / do_POST](#do_get-do_post-standard-library) · [Underscore-prefixed helper](#underscore-prefixed-helper-syntax) · [Type hints](#type-hints-syntax) · [bytes and .encode()](#bytes-and-encode-python-built-in) · [send_response / send_header / end_headers](#send_response-send_header-end_headers-standard-library) · [wfile.write / rfile.read](#wfilewrite-rfileread-standard-library) · [headers.get](#headersget-standard-library) · [log_message override](#log_message-override-standard-library) · [urllib.request (urlopen, Request)](#urllibrequest-urlopen-request-standard-library) · [urllib.error.HTTPError](#urlliberrorhttperror-standard-library) · [with statement (context manager)](#with-statement-context-manager-syntax) · [try / except](#try-except-syntax) · [if __name__ == "__main__"](#if-__name__-__main__-syntax) · [Host and port (127.0.0.1)](#host-and-port-127001-concept) · [threading.Thread (daemon)](#threadingthread-daemon-standard-library) · [serve_forever / shutdown](#serve_forever-shutdown-standard-library) · [Argument unpacking with *](#argument-unpacking-with-syntax)

---

## HTTP request and response  `(concept)`
**1. Definition:** HTTP is a text-based request/response protocol on top of TCP. A request consists of a method, a path, headers and an optional body; a response consists of a status code, headers and an optional body.
**2. In plain words:** Every web API conversation is "the client asks, the server answers". The request says what you want (`GET /health`) and may carry data; the response says how it went (a number like 200) and carries the result. Frameworks hide this, but the shape never changes. HTTP is stateless: the server does not remember the previous request unless you send something (like an id) that lets it look it up.
**3. Example to quote:**
> "When my client calls `POST /echo` with `{"text": "hello"}`, that is a request with method POST, path `/echo`, a `Content-Type` header and a JSON body. The server answers with status 200 and a JSON body `{'you_sent': {...}}`."

**Seen in:** whole step; `WALKTHROUGH.md` "Execution trace"

## GET vs POST  `(concept)`
**1. Definition:** GET asks the server to return a resource and should not change server state; POST sends data in the body for the server to process, usually creating something or triggering work.
**2. In plain words:** Use GET to read ("give me the health status") and POST to send something the server must act on ("here is some text, classify it"). GET data usually travels in the URL, which is visible and length-limited, so large or sensitive inputs go in a POST body. In this step the same path can behave differently per method: `/echo` only exists for POST.
**3. Example to quote:**
> "`GET /health` just reads the status, so it is a GET. `POST /echo` sends a JSON payload, so it is a POST. If I GET `/echo` I get 404 here, because only `do_POST` knows that path."

**Seen in:** `do_GET`, `do_POST`

## HTTP status code  `(concept)`
**1. Definition:** A three-digit number in the response that summarises the result: 2xx success, 3xx redirect, 4xx client error, 5xx server error.
**2. In plain words:** The status code lets a client react without parsing the body. 200 means OK, 400 means "your request was malformed", 404 means "no such path", 500 means "the server crashed". Choosing the right one matters: 4xx says "caller, fix your request", 5xx says "we have a problem". This step returns 200, 400 and 404.
**3. Example to quote:**
> "When the body isn't valid JSON I return 400, because that's the client's fault. When the path doesn't exist I return 404. If my own code crashed, it should be a 500."

**Seen in:** `_send_json(400, ...)`, `_send_json(404, ...)`

## HTTP header  `(concept)`
**1. Definition:** A key-value pair sent with a request or response that carries metadata about the message, such as `Content-Type` or `Content-Length`.
**2. In plain words:** Headers are the "labels on the envelope": they describe the body without being part of it. They come before a blank line, and the body comes after. Servers and clients read them to decide how to interpret the bytes that follow.
**3. Example to quote:**
> "My response has two headers: `Content-Type: application/json` so the client knows to parse JSON, and `Content-Length` so it knows how many bytes to read."

**Seen in:** `send_header(...)`, `self.headers.get(...)`

## Content-Type  `(concept)`
**1. Definition:** An HTTP header naming the media type of the message body, for example `application/json` or `text/plain`.
**2. In plain words:** Bytes alone have no meaning; `Content-Type` says whether they are JSON, HTML, an image, and so on. When a client sends JSON it sets `application/json` so the server knows to parse it that way. Flask's `jsonify` and FastAPI set it for you, but here we set it by hand to see it.
**3. Example to quote:**
> "I set `Content-Type: application/json` on the response so the client's `json.loads` is the right way to read it."

**Seen in:** `_send_json`, `call`

## Content-Length  `(concept)`
**1. Definition:** An HTTP header giving the size of the message body in bytes.
**2. In plain words:** TCP is a stream with no built-in message boundaries, so the receiver needs to know how many bytes belong to the body. In `do_POST` the server reads this header to know how much to pull from the socket; in `_send_json` it is computed with `len(body)` on the encoded bytes, not on the string, because characters and bytes can differ in length.
**3. Example to quote:**
> "The body `{\"text\": \"hello\"}` is 17 bytes, so the client sends `Content-Length: 17` and my handler reads exactly 17 bytes with `rfile.read(17)`."

**Seen in:** `_send_json`, `do_POST`

## json (module)  `(standard library)`
**1. Definition:** Python's standard module for converting between Python objects (dict, list, str, numbers) and JSON text.
**2. In plain words:** JSON is the text format APIs speak. `json.dumps` turns a dict into a JSON string to send, and `json.loads` turns received JSON back into a dict. It is built in, so no installation is needed. Only JSON-compatible types work: a `set` or a custom object raises `TypeError`.
**3. Example to quote:**
> "I use `json.dumps({'status': 'ok'})` to build the response body and `json.loads(raw)` to parse the request body."

**Seen in:** `_send_json`, `do_POST`, `call`

## json.dumps  `(standard library)`
**1. Definition:** `json.dumps(obj)` serialises a Python object to a JSON-formatted `str`.
**2. In plain words:** It is "dict to text". Note it returns a string, not bytes, so before putting it on the network we still need `.encode("utf-8")`. The "s" stands for "string"; `json.dump` (no s) writes to a file instead.
**3. Example to quote:**
> "`json.dumps({'status': 'ok'})` gives the string `'{\"status\": \"ok\"}'`."

```python
import json
print(json.dumps({"status": "ok"}))   # {"status": "ok"}
```

**Seen in:** `_send_json`

## json.loads  `(standard library)`
**1. Definition:** `json.loads(s)` parses a JSON `str`, `bytes` or `bytearray` into the corresponding Python object, raising `json.JSONDecodeError` on invalid input.
**2. In plain words:** It is "text to dict". It accepts bytes directly, which is handy because HTTP bodies arrive as bytes. If the text is not valid JSON, it raises an error, which is why `do_POST` wraps it in `try/except` and returns 400.
**3. Example to quote:**
> "`json.loads(b'{\"text\": \"hello\"}')` returns `{'text': 'hello'}`. With `b'{not json'` it raises `JSONDecodeError`, so I answer 400."

```python
import json
print(json.loads(b'{"text": "hello"}'))   # {'text': 'hello'}
```

**Seen in:** `do_POST`, `call`

## json.JSONDecodeError  `(standard library)`
**1. Definition:** The exception raised by `json.loads` when its input is not valid JSON; it is a subclass of `ValueError`.
**2. In plain words:** It is the signal that the client sent garbage. Catching this specific exception, instead of a bare `except`, means you only handle the case you expect and still see genuine bugs.
**3. Example to quote:**
> "I catch `json.JSONDecodeError` around `json.loads` and return `400 {'error': 'body is not valid JSON'}` instead of letting the server crash."

**Seen in:** `do_POST`

## http.server  `(standard library)`
**1. Definition:** A standard-library module providing simple HTTP server classes, including `HTTPServer` and `BaseHTTPRequestHandler`.
**2. In plain words:** It is the smallest way to run a web server in Python with no installs. It parses the raw HTTP for you and calls your handler. It handles one request at a time and has no routing or validation, so it is for learning, not production.
**3. Example to quote:**
> "I used `http.server` to see HTTP without a framework. Flask and FastAPI do the same parsing but add routing, validation and convenience."

**Seen in:** imports; `HTTPServer(("127.0.0.1", 8765), Handler)`

## HTTPServer  `(standard library)`
**1. Definition:** `http.server.HTTPServer(address, handler_class)` listens on a socket at `(host, port)` and creates an instance of the handler class for each incoming request.
**2. In plain words:** This is the object that actually "listens". You give it where to listen and which class should answer. Calling `serve_forever()` makes it loop accepting connections.
**3. Example to quote:**
> "`HTTPServer(('127.0.0.1', 8765), Handler)` means: listen on this computer, port 8765, and use `Handler` to answer each request."

**Seen in:** `if __name__ == "__main__":`

## BaseHTTPRequestHandler  `(standard library)`
**1. Definition:** The base class in `http.server` that parses an incoming HTTP request and dispatches it to methods named `do_<VERB>` such as `do_GET` and `do_POST`.
**2. In plain words:** You subclass it and only write the verbs you care about. The parent class has already read the request line and headers, so `self.path` and `self.headers` are ready to use. Any verb you did not define (like DELETE) gets an automatic error from the parent.
**3. Example to quote:**
> "My `Handler` extends `BaseHTTPRequestHandler`, so I only write `do_GET` and `do_POST`; the parsing is inherited."

**Seen in:** `class Handler(BaseHTTPRequestHandler)`

## Class inheritance  `(syntax)`
**1. Definition:** Writing `class Child(Parent):` creates a class that gets all attributes and methods of `Parent` and may add or override some.
**2. In plain words:** Instead of rewriting HTTP parsing, `Handler` borrows it from `BaseHTTPRequestHandler`. You override only the methods whose behaviour you want to change. Inheritance is also how frameworks let you plug your logic into their machinery.
**3. Example to quote:**
> "`class Handler(BaseHTTPRequestHandler)` says 'a Handler is a request handler'; it inherits the parsing and I override `do_GET`."

```python
class Animal:
    def speak(self): return "..."
class Dog(Animal):
    def speak(self): return "woof"
```

**Seen in:** `class Handler(...)`

## self.path  `(standard library)`
**1. Definition:** An attribute of the request handler holding the request target (the path and query string) as a string, such as `"/health"`.
**2. In plain words:** It is how the handler knows what the client asked for. Comparing `self.path` against strings is a hand-made version of what a framework's routing does with `@app.get("/health")`. Note that it still includes any query string, such as `/add?a=2`.
**3. Example to quote:**
> "In `do_GET` I branch on `self.path`: `/health` returns 200, anything else returns 404."

**Seen in:** `do_GET`, `do_POST`

## do_GET / do_POST  `(standard library)`
**1. Definition:** Methods that `BaseHTTPRequestHandler` automatically calls when a request with the matching HTTP verb arrives.
**2. In plain words:** The method name is the contract: `do_GET` for GET, `do_POST` for POST. You never call them yourself. If a verb has no matching method, the server replies "501 Unsupported method".
**3. Example to quote:**
> "I never call `do_POST` myself; the server calls it when a POST arrives, and inside it I read the body and reply."

**Seen in:** `Handler`

## Underscore-prefixed helper  `(syntax)`
**1. Definition:** By convention, a name starting with a single underscore (such as `_send_json`) marks an internal helper that is not meant to be called from outside the class or module.
**2. In plain words:** Python does not enforce privacy, but the underscore tells other developers "this is an implementation detail". Here `_send_json` avoids repeating the same five lines in every response path.
**3. Example to quote:**
> "`_send_json` is a private helper so every response goes through one place and always has the right headers."

**Seen in:** `_send_json`

## Type hints  `(syntax)`
**1. Definition:** Optional annotations like `status: int` and `-> None` that declare the expected types of parameters and return values; Python does not enforce them at runtime.
**2. In plain words:** They document intent and let editors and tools catch mistakes. `bytes | None` means "bytes or None". Frameworks like FastAPI later use them for real validation, but here they are only documentation.
**3. Example to quote:**
> "`def call(method: str, url: str, body: bytes | None = None) -> tuple[int, dict]` says it takes a method and URL, optionally a body, and returns a status code plus a dict."

**Seen in:** `_send_json`, `call`

## bytes and .encode()  `(Python built-in)`
**1. Definition:** `bytes` is an immutable sequence of raw byte values; `str.encode("utf-8")` converts text to bytes and `bytes.decode()` converts back.
**2. In plain words:** Networks move bytes, not characters. A Python string must be encoded before writing it to a socket, and body bytes read from a socket must be decoded or passed to `json.loads`. Forgetting this gives the classic `TypeError: a bytes-like object is required, not 'str'`.
**3. Example to quote:**
> "`json.dumps(payload).encode('utf-8')` converts the JSON string to bytes so `wfile.write` can send it."

```python
print("hi".encode())   # b'hi'
```

**Seen in:** `_send_json`, `call`

## send_response / send_header / end_headers  `(standard library)`
**1. Definition:** Handler methods that write an HTTP response in order: `send_response(code)` writes the status line, `send_header(name, value)` adds a header, and `end_headers()` writes the blank line that ends the headers.
**2. In plain words:** An HTTP response has a strict order: status, headers, blank line, body. These three calls are that order made visible. If you write the body before `end_headers()`, the client gets a malformed response.
**3. Example to quote:**
> "I call `send_response(200)`, then `send_header` twice, then `end_headers()`, and only then write the body."

**Seen in:** `_send_json`

## wfile.write / rfile.read  `(standard library)`
**1. Definition:** `self.wfile` is a writable stream to the client and `self.rfile` a readable stream from the client; `write(bytes)` sends bytes and `read(n)` reads exactly `n` bytes.
**2. In plain words:** They are the pipes to and from the client. `rfile.read(length)` needs the length from `Content-Length`; if you call `read()` without a size on a POST it can block forever waiting for more data.
**3. Example to quote:**
> "I read the header `Content-Length`, call `self.rfile.read(length)` to get the body bytes, and answer with `self.wfile.write(body)`."

**Seen in:** `do_POST`, `_send_json`

## headers.get  `(standard library)`
**1. Definition:** `self.headers.get(name, default)` returns a request header's value as a string, or the default if the header is missing.
**2. In plain words:** It is like `dict.get`: it never raises when the header is absent. Header values are always strings, so the code wraps it in `int(...)` to turn the length into a number.
**3. Example to quote:**
> "`int(self.headers.get('Content-Length', 0))` gives me the body size, or 0 if the client sent no body."

**Seen in:** `do_POST`

## log_message override  `(standard library)`
**1. Definition:** `BaseHTTPRequestHandler.log_message(format, *args)` is the method that prints each request to stderr; overriding it changes or silences that logging.
**2. In plain words:** By default the server prints a line per request, which would clutter the demo output. Defining `log_message` with `pass` as its body turns that off. In a real service you would send these lines to proper logging instead.
**3. Example to quote:**
> "I override `log_message` with `pass` so my demo output only shows my own prints."

**Seen in:** `Handler.log_message`

## urllib.request (urlopen, Request)  `(standard library)`
**1. Definition:** The standard-library HTTP client module: `urllib.request.Request(url, data, method, headers)` describes a request and `urlopen(request)` sends it and returns a response object.
**2. In plain words:** It is the no-install way to call an HTTP server from Python. You pass `data` as bytes for a POST body. Libraries like `requests` and `httpx` are friendlier, but `urllib` shows what is happening.
**3. Example to quote:**
> "My `call` helper builds a `Request` with the method, body and a JSON header, then `urlopen` returns an object with `.status` and `.read()`."

**Seen in:** `call`

## urllib.error.HTTPError  `(standard library)`
**1. Definition:** The exception `urlopen` raises for responses with status 4xx or 5xx; the exception object is also a response, with `.code` and `.read()`.
**2. In plain words:** `urllib` treats error statuses as exceptions, but we want to inspect them, so `call` catches `HTTPError` and returns `(err.code, parsed_body)` the same way it does for success. Other libraries (like `httpx`) don't raise unless you ask.
**3. Example to quote:**
> "A 404 would crash my client, so I catch `HTTPError` and return the status and JSON body instead."

**Seen in:** `call`

## with statement (context manager)  `(syntax)`
**1. Definition:** `with expr as name:` runs a block with a resource and guarantees its cleanup code runs afterwards, even if an exception occurs.
**2. In plain words:** It is the safe way to use things that must be closed, such as files and network connections. When the indented block ends, the connection is closed automatically; you cannot forget.
**3. Example to quote:**
> "`with urllib.request.urlopen(req) as resp:` closes the connection for me when the block ends, even on an error."

```python
with open("notes.txt") as fh:
    text = fh.read()
```

**Seen in:** `call`

## try / except  `(syntax)`
**1. Definition:** A construct that runs code in `try` and, if a listed exception is raised, jumps to the matching `except` block instead of crashing.
**2. In plain words:** It lets the program handle expected failures gracefully, such as bad JSON from a client. Catch the specific exception you expect so real bugs are not hidden.
**3. Example to quote:**
> "I wrap `json.loads(raw)` in `try` and catch `JSONDecodeError` to return a clear 400 rather than a server error."

```python
try:
    int("x")
except ValueError:
    print("not a number")
```

**Seen in:** `do_POST`, `call`

## if __name__ == "__main__"  `(syntax)`
**1. Definition:** A guard that is true only when the file is executed directly (`python main.py`), not when it is imported as a module.
**2. In plain words:** It lets a file be both a script and an importable module. The demo server starts only when you run the file, so importing helpers from it does not start anything.
**3. Example to quote:**
> "The server and the demo calls sit under `if __name__ == '__main__':`, so importing this file in a test won't start a server."

**Seen in:** bottom of `main.py`

## Host and port (127.0.0.1)  `(concept)`
**1. Definition:** A host identifies a machine and a port identifies a service on it; `127.0.0.1` (localhost) always means "this computer".
**2. In plain words:** Binding to `127.0.0.1` makes the server reachable only from your own machine, which is safe for experiments. Two programs cannot listen on the same port at once, which is why you may need a different port number. Inside Docker later you will bind to `0.0.0.0` instead.
**3. Example to quote:**
> "I bound to `127.0.0.1:8765`, so only my computer can reach it. Changing the port in only one place gives 'Connection refused'."

**Seen in:** `HTTPServer(("127.0.0.1", 8765), Handler)`

## threading.Thread (daemon)  `(standard library)`
**1. Definition:** `threading.Thread(target=f, daemon=True)` creates a thread that runs `f` concurrently; a daemon thread does not keep the program alive when the main thread ends.
**2. In plain words:** The server's `serve_forever()` never returns, so we run it on a background thread to let the main thread keep making requests. `daemon=True` means the program can still exit without waiting for it.
**3. Example to quote:**
> "I start `server.serve_forever` in a daemon thread so my script can act as the client in the same process."

**Seen in:** `if __name__ == "__main__":`

## serve_forever / shutdown  `(standard library)`
**1. Definition:** `server.serve_forever()` loops accepting and handling requests until `server.shutdown()` is called from another thread.
**2. In plain words:** `serve_forever` is the server's main loop; `shutdown` stops it cleanly so the port is released. Calling `shutdown` from the same thread that runs `serve_forever` would deadlock, which is another reason for the background thread.
**3. Example to quote:**
> "After my four test calls I call `server.shutdown()` so the port is freed."

**Seen in:** `if __name__ == "__main__":`

## Argument unpacking with *  `(syntax)`
**1. Definition:** Putting `*` before an iterable in a call, such as `print("a", *call(...))`, passes its items as separate positional arguments.
**2. In plain words:** `call` returns a tuple `(status, body)`. With `*`, `print` receives them as two separate values and prints `200 {'status': 'ok'}` instead of `(200, {...})`.
**3. Example to quote:**
> "`print('GET /health ->', *call(...))` unpacks the status and the body, so the output reads `200 {'status': 'ok'}`."

```python
print(*(1, 2))   # 1 2
```

**Seen in:** `if __name__ == "__main__":`
