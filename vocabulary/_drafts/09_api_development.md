### HTTP
<!-- cat: api -->
- **Technical definition:** HyperText Transfer Protocol: a request/response protocol where a client sends a method, path, headers and optional body and the server replies with a status code, headers and body.
- **Plain English:** The language browsers and programs use to ask a server for things and get answers.
- **Example / analogy:** Ordering at a restaurant: you ask (request), the kitchen answers (response) with a status like 'ready' or 'sold out'.
- **Snippet:**
```bash
curl -i http://127.0.0.1:8000/health
```
- **Snippet explained:** `-i` prints the status line and response headers before the body.
- **Related terms:** [[HTTP method]], [[Status code]], [[REST]]

### HTTP method
<!-- cat: api -->
- **Technical definition:** The verb of a request (GET, POST, PUT, PATCH, DELETE) describing the intended action on the resource at the path.
- **Plain English:** What you want to DO: read, create, replace, change, delete.
- **Example / analogy:** GET = look at the menu, POST = place an order, DELETE = cancel it.
- **Snippet:**
```python
@app.get("/items")
@app.post("/items")
```
- **Snippet explained:** Same path, two routes: the method decides which function runs.
- **Related terms:** [[GET vs POST]], [[Route]], [[REST]]

### GET vs POST
<!-- cat: api -->
- **Technical definition:** GET retrieves data and has no body (parameters live in the URL, results may be cached); POST sends a body to create or process something and is not cached or idempotent by default.
- **Plain English:** GET asks, POST sends something along.
- **Example / analogy:** GET is reading a notice board; POST is handing a form to the clerk.
- **Snippet:**
```bash
curl 'http://127.0.0.1:8000/add?a=1&b=2'
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{"text":"hi"}'
```
- **Snippet explained:** First sends data in the URL; second sends a JSON body, which suits long text and keeps it out of URLs/logs.
- **Related terms:** [[HTTP method]], [[Idempotent]], [[Request body]]

### Status code
<!-- cat: api -->
- **Technical definition:** A three-digit number in the response: 2xx success, 3xx redirect, 4xx client error, 5xx server error.
- **Plain English:** A one-number summary of how the request went.
- **Example / analogy:** Like a traffic light on the answer: green 200, amber 4xx 'you did something wrong', red 5xx 'we broke'.
- **Snippet:**
```python
return jsonify(error="not found"), 404
```
- **Snippet explained:** Flask's (body, status) tuple sets the status code.
- **Related terms:** [[HTTP]], [[422 Unprocessable Content]], [[HTTP 500]]

### 422 Unprocessable Content
<!-- cat: api -->
- **Technical definition:** Status code meaning the request is well-formed JSON but its values fail validation; FastAPI returns it automatically for Pydantic errors.
- **Plain English:** 'I understood your message but the content is not acceptable.'
- **Example / analogy:** Handing in a form with your age written as 'banana'.
- **Snippet:**
```bash
curl -s -o /dev/null -w '%{http_code}' -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":""}'
```
- **Snippet explained:** Prints 422 because `text` fails `min_length=1`.
- **Related terms:** [[Status code]], [[Pydantic]], [[Validation]]

### HTTP 500
<!-- cat: api -->
- **Technical definition:** Internal Server Error: an unexpected failure in the server code; the server, not the client, must be fixed.
- **Plain English:** 'Something broke on our side.'
- **Example / analogy:** The kitchen caught fire, regardless of what you ordered.
- **Snippet:**
```python
@app.errorhandler(Exception)
def unexpected(e):
    app.logger.exception("unhandled")
    return jsonify(error="internal server error"), 500
```
- **Snippet explained:** Logs the traceback privately and returns a safe generic message to the client.
- **Related terms:** [[Status code]], [[Error handler]], [[Logging]]

### REST
<!-- cat: api -->
- **Technical definition:** An API style where URLs identify resources (nouns), HTTP methods express actions, and responses use standard status codes and usually JSON.
- **Plain English:** A tidy convention for naming web endpoints so they're predictable.
- **Example / analogy:** A library catalogue: /books/7 is a book, GET reads it, DELETE removes it.
- **Snippet:**
```python
@app.get("/items/{item_id}")
@app.post("/items")
```
- **Snippet explained:** Resource path + method pairs follow REST naming.
- **Related terms:** [[HTTP method]], [[Route]], [[JSON]]

### Route
<!-- cat: api -->
- **Technical definition:** A mapping from (HTTP method, URL path pattern) to a handler function.
- **Plain English:** A rule saying 'when someone asks for THIS address with THIS verb, run THAT function'.
- **Example / analogy:** A phone switchboard: dial extension 12, reach the person at desk 12.
- **Snippet:**
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```
- **Snippet explained:** Registers `health` as the handler for GET /health.
- **Related terms:** [[Decorator]], [[HTTP method]], [[Path parameter]]

### Path parameter
<!-- cat: api -->
- **Technical definition:** A variable part of the URL path, e.g. /items/7, captured and passed to the handler.
- **Plain English:** A blank in the address that you fill in.
- **Example / analogy:** Room number in a hotel address.
- **Snippet:**
```python
@app.get("/items/{item_id}")
def get_item(item_id: int): ...
```
- **Snippet explained:** `item_id` receives 7 for /items/7 and FastAPI converts it to int.
- **Related terms:** [[Query parameter]], [[Route]]

### Query parameter
<!-- cat: api -->
- **Technical definition:** Key=value pairs after `?` in a URL; always strings until converted.
- **Plain English:** Optional extra settings tacked onto the address.
- **Example / analogy:** Filters on a shopping site: ?color=red&size=M.
- **Snippet:**
```python
@app.get("/add")
def add(a: float, b: float):
    return {"result": a + b}
```
- **Snippet explained:** In FastAPI, non-path typed arguments become query params: /add?a=1&b=2.
- **Related terms:** [[Path parameter]], [[Request body]]

### Request body
<!-- cat: api -->
- **Technical definition:** The payload of a request (typically JSON) sent after the headers, used with POST/PUT/PATCH.
- **Plain English:** The actual data you send along with your request.
- **Example / analogy:** The contents of the envelope.
- **Snippet:**
```python
data = request.get_json(silent=True)
```
- **Snippet explained:** Flask parses the JSON body; `silent=True` returns None instead of raising on bad JSON.
- **Related terms:** [[JSON]], [[GET vs POST]], [[Pydantic]]

### JSON
<!-- cat: api -->
- **Technical definition:** JavaScript Object Notation: a text format of nested key/value objects, arrays, strings, numbers, booleans and null; the standard API data format.
- **Plain English:** A way to write dictionaries and lists as plain text.
- **Example / analogy:** A universal shipping label both sides can read.
- **Snippet:**
```python
import json
json.loads('{"a": 1}')   # -> {"a": 1}
json.dumps({"a": 1})  # -> '{"a": 1}'
```
- **Snippet explained:** `loads` text -> Python object; `dumps` Python object -> text.
- **Related terms:** [[Request body]], [[Serialization]]

### Header
<!-- cat: api -->
- **Technical definition:** A name/value metadata line on a request or response (Content-Type, Content-Length, Authorization...).
- **Plain English:** Notes written on the envelope, not inside the letter.
- **Example / analogy:** 'Fragile' sticker on a parcel.
- **Snippet:**
```bash
curl -H 'Content-Type: application/json' ...
```
- **Snippet explained:** Tells the server the body is JSON; Flask's get_json depends on it.
- **Related terms:** [[HTTP]], [[Request body]]

### Idempotent
<!-- cat: api -->
- **Technical definition:** An operation that has the same effect whether done once or many times (GET, PUT, DELETE are; POST usually isn't). Safe-to-retry operations must be idempotent.
- **Plain English:** Doing it twice is the same as doing it once.
- **Example / analogy:** Pressing the lift's 'up' button repeatedly.
- **Snippet:**
```python
# retry only if repeating is harmless
for attempt in range(3):
    ...
```
- **Snippet explained:** Idempotency decides whether automatic retries are safe.
- **Related terms:** [[Retry with backoff]], [[HTTP method]]

### Flask
<!-- cat: api -->
- **Technical definition:** A lightweight Python micro-framework for web apps and APIs that speaks WSGI; provides routing and request/response helpers and leaves validation/docs to you.
- **Plain English:** A small toolkit for making web endpoints in Python.
- **Example / analogy:** A basic toolbox: few tools, you add what you need.
- **Snippet:**
```python
from flask import Flask, jsonify
app = Flask(__name__)
@app.get("/health")
def health():
    return jsonify(status="ok")
```
- **Snippet explained:** Creates an app and a health route returning JSON.
- **Related terms:** [[WSGI]], [[Gunicorn]], [[FastAPI]]

### FastAPI
<!-- cat: api -->
- **Technical definition:** A modern Python web framework on ASGI (Starlette + Pydantic) that uses type hints for parameter parsing, validation and automatic OpenAPI docs.
- **Plain English:** A web framework where your type hints do the checking and the docs write themselves.
- **Example / analogy:** A toolbox with built-in measuring tape and instruction manual.
- **Snippet:**
```python
from fastapi import FastAPI
app = FastAPI()
@app.get("/add")
def add(a: float, b: float):
    return {"result": a + b}
```
- **Snippet explained:** `a` and `b` are parsed and validated as floats automatically.
- **Related terms:** [[ASGI]], [[Pydantic]], [[Uvicorn]]

### Pydantic
<!-- cat: api -->
- **Technical definition:** A Python library that validates and converts data using type annotations on a BaseModel class, raising ValidationError with all problems.
- **Plain English:** A strict bouncer that checks every field of incoming data matches the shape you declared.
- **Example / analogy:** A form with rules: age must be a number 0-120.
- **Snippet:**
```python
from pydantic import BaseModel, Field
class Req(BaseModel):
    text: str = Field(min_length=1)
Req(text="hi")
```
- **Snippet explained:** Declares a required non-empty string; invalid data raises ValidationError.
- **Related terms:** [[Validation]], [[FastAPI]], [[422 Unprocessable Content]]

### Validation
<!-- cat: api -->
- **Technical definition:** Checking that input has the expected type, range and size before using it; rejecting it with a clear client error otherwise.
- **Plain English:** Making sure what came in is acceptable before you act on it.
- **Example / analogy:** Airport security before boarding.
- **Snippet:**
```python
if not isinstance(text, str) or not text.strip():
    return jsonify(error="text required"), 422
```
- **Snippet explained:** A hand-written validation check as done in Flask.
- **Related terms:** [[Pydantic]], [[422 Unprocessable Content]]

### WSGI
<!-- cat: api -->
- **Technical definition:** Web Server Gateway Interface: the synchronous Python standard where a server calls app(environ, start_response) and receives an iterable of bytes.
- **Plain English:** The old agreed handshake between a Python web app and the server running it.
- **Example / analogy:** A standard plug shape so any appliance fits any socket.
- **Snippet:**
```python
def app(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"hello"]
```
- **Snippet explained:** The smallest WSGI app.
- **Related terms:** [[ASGI]], [[Gunicorn]], [[Flask]]

### ASGI
<!-- cat: api -->
- **Technical definition:** Asynchronous Server Gateway Interface: async successor to WSGI where the app is `async def app(scope, receive, send)`, supporting HTTP, WebSockets and lifespan events.
- **Plain English:** The modern handshake that allows apps to wait without blocking, and supports live connections.
- **Example / analogy:** A phone line that lets you put callers on hold while helping others.
- **Snippet:**
```python
async def app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"hi"})
```
- **Snippet explained:** The smallest ASGI app sends two events.
- **Related terms:** [[WSGI]], [[Uvicorn]], [[FastAPI]]

### Gunicorn
<!-- cat: api -->
- **Technical definition:** A production-grade pre-fork WSGI HTTP server that manages multiple worker processes (and can run Uvicorn workers for ASGI apps).
- **Plain English:** The program that actually listens on the port and runs several copies of your Flask app.
- **Example / analogy:** A restaurant manager who hires several waiters.
- **Snippet:**
```bash
gunicorn flask_app:app --workers 2 --bind 127.0.0.1:5001
```
- **Snippet explained:** Starts 2 worker processes serving the `app` object from flask_app.py.
- **Related terms:** [[Worker]], [[WSGI]], [[Uvicorn]]

### Uvicorn
<!-- cat: api -->
- **Technical definition:** A fast ASGI server (built on uvloop/httptools when available) used to run FastAPI/Starlette apps; supports --workers and --reload.
- **Plain English:** The program that listens on the port and runs your FastAPI app.
- **Example / analogy:** The shop front that passes orders to the kitchen.
- **Snippet:**
```bash
uvicorn fastapi_app:app --port 8000 --workers 2
```
- **Snippet explained:** Serves `app` from fastapi_app.py with 2 processes.
- **Related terms:** [[ASGI]], [[Gunicorn]], [[Worker]]

### Worker
<!-- cat: api -->
- **Technical definition:** An operating-system process (or thread) launched by the server to handle requests; each process has its own memory, so state and models are duplicated per worker.
- **Plain English:** One copy of your app doing the work; more workers = more requests in parallel.
- **Example / analogy:** Extra checkout lanes in a shop - each needs its own cashier (and its own copy of the price list).
- **Snippet:**
```bash
curl localhost:5001/whoami   # pid differs between workers
```
- **Snippet explained:** Different `pid` values prove separate processes.
- **Related terms:** [[Gunicorn]], [[Uvicorn]], [[Concurrency]]

### Concurrency
<!-- cat: api -->
- **Technical definition:** Making progress on several requests at overlapping times, via processes, threads or async I/O.
- **Plain English:** Handling more than one thing at once (or seeming to).
- **Example / analogy:** A chef stirring one pot while another simmers.
- **Snippet:**
```python
await asyncio.sleep(1)   # lets the event loop serve others meanwhile
```
- **Snippet explained:** Async waiting does not block other requests; time.sleep would.
- **Related terms:** [[Async/await]], [[Worker]], [[GIL]]

### Event loop
<!-- cat: api -->
- **Technical definition:** The scheduler in async programs that runs coroutines and resumes them when their awaited I/O finishes; it runs in a single thread, so blocking calls stall everything.
- **Plain English:** The traffic controller deciding which waiting task goes next.
- **Example / analogy:** One waiter checking which tables are ready.
- **Snippet:**
```python
import asyncio
asyncio.run(main())
```
- **Snippet explained:** Creates an event loop, runs `main()` to completion, closes it.
- **Related terms:** [[Concurrency]], [[ASGI]], [[Async/await]]

### Async/await
<!-- cat: python -->
- **Technical definition:** Syntax for coroutines: `async def` defines a function that can be paused; `await` pauses it until an awaitable completes, letting others run.
- **Plain English:** A way to say 'while I wait for this, go do something else'.
- **Example / analogy:** Putting the kettle on, then making toast while it heats.
- **Snippet:**
```python
async def f():
    await asyncio.sleep(1)
    return 'done'
```
- **Snippet explained:** Pauses 1 s without blocking the event loop; must itself be awaited or run with asyncio.run.
- **Related terms:** [[Event loop]], [[Concurrency]]

### Dependency injection (FastAPI)
<!-- cat: api -->
- **Technical definition:** FastAPI's `Depends(...)` mechanism for supplying shared resources (settings, DB sessions, auth) to route functions declaratively.
- **Plain English:** Letting the framework hand your function the tools it needs.
- **Example / analogy:** A waiter bringing you the right cutlery for each course.
- **Snippet:**
```python
from fastapi import Depends
@app.get('/x')
def x(settings=Depends(get_settings)): ...
```
- **Snippet explained:** FastAPI calls get_settings() and passes the result in.
- **Related terms:** [[FastAPI]], [[Config]]

### Lifespan (startup/shutdown)
<!-- cat: api -->
- **Technical definition:** FastAPI's context manager for code that runs once per worker process before serving (e.g. loading a model) and after shutdown.
- **Plain English:** A place to load heavy things once when the server starts.
- **Example / analogy:** Opening the shop and switching on the lights before the first customer.
- **Snippet:**
```python
@asynccontextmanager
async def lifespan(app):
    app.state.model = load_model()
    yield
```
- **Snippet explained:** Loads the model before the first request; code after `yield` runs at shutdown.
- **Related terms:** [[FastAPI]], [[Model loading]], [[Worker]]

### Lazy loading
<!-- cat: ml -->
- **Technical definition:** Deferring expensive initialisation (like importing or loading a big model) until it is first needed, then caching it.
- **Plain English:** Don't unpack the heavy thing until someone actually asks for it.
- **Example / analogy:** Not starting the oven until an order for pizza arrives.
- **Snippet:**
```python
_GEN = None
def gen():
    global _GEN
    if _GEN is None:
        _GEN = load_big_model()
    return _GEN
```
- **Snippet explained:** First call pays the loading cost; later calls reuse the cached object.
- **Related terms:** [[Model loading]], [[Lifespan (startup/shutdown)]]

### Environment variable
<!-- cat: python -->
- **Technical definition:** A named string value set outside the program (by the shell, container or platform) and read by the process, used for configuration and secrets.
- **Plain English:** Settings handed to a program from outside, not written in its code.
- **Example / analogy:** Instructions on a sticky note attached to the machine.
- **Snippet:**
```python
import os
port = int(os.getenv("PORT", "8000"))
```
- **Snippet explained:** Reads PORT (a string) with a default and converts to int.
- **Related terms:** [[.env file]], [[Configuration]], [[Docker ENV]]

### .env file
<!-- cat: python -->
- **Technical definition:** A text file of KEY=value lines loaded into the environment by tools like python-dotenv (never committed when it holds secrets).
- **Plain English:** A local settings file for environment variables.
- **Example / analogy:** A notepad of your private settings next to the project.
- **Snippet:**
```python
from dotenv import load_dotenv
load_dotenv()   # real env vars win unless override=True
```
- **Snippet explained:** Loads .env into os.environ without overriding variables already set.
- **Related terms:** [[Environment variable]], [[Configuration]]

### Configuration
<!-- cat: api -->
- **Technical definition:** All settings that change behaviour between environments (paths, versions, limits, keys), kept out of code and read in one place.
- **Plain English:** The knobs and switches of your program.
- **Example / analogy:** A car's seat and mirror settings, separate from the car's engine.
- **Snippet:**
```python
@dataclass(frozen=True)
class Settings:
    model_path: str | None
    max_text_length: int
```
- **Snippet explained:** A frozen dataclass collects settings in one typed object.
- **Related terms:** [[Environment variable]], [[.env file]]

### OpenAPI / Swagger docs
<!-- cat: api -->
- **Technical definition:** OpenAPI is a machine-readable API description; FastAPI generates it at /openapi.json and renders interactive Swagger UI at /docs from your code.
- **Plain English:** Automatic, clickable documentation for your API.
- **Example / analogy:** A menu that writes itself and lets you taste-test each dish.
- **Snippet:**
```bash
curl -s localhost:8000/openapi.json | head -c 200
```
- **Snippet explained:** Prints the start of the generated machine-readable spec.
- **Related terms:** [[FastAPI]], [[Pydantic]]

### pytest
<!-- cat: python -->
- **Technical definition:** A Python test framework that discovers `test_*` functions, uses plain `assert`, fixtures and parametrization.
- **Plain English:** A tool that runs your checks and reports which fail.
- **Example / analogy:** A checklist an inspector ticks off automatically.
- **Snippet:**
```python
def test_health(client):
    assert client.get('/health').status_code == 200
```
- **Snippet explained:** Fails with a readable diff if the status is not 200.
- **Related terms:** [[Test client]], [[Fixture]], [[Smoke test]]

### Test client
<!-- cat: api -->
- **Technical definition:** A helper that calls a web app in-process without a network server (Flask `app.test_client()`, FastAPI `TestClient`).
- **Plain English:** A pretend browser used inside tests.
- **Example / analogy:** A flight simulator instead of a real flight.
- **Snippet:**
```python
client = app.test_client()
resp = client.post('/predict', json={'text': 'hi'})
```
- **Snippet explained:** Sends a POST in-process; `resp.status_code`, `resp.get_json()` hold the result.
- **Related terms:** [[pytest]], [[Smoke test]]

### Smoke test
<!-- cat: api -->
- **Technical definition:** A quick end-to-end check against a really running service that its basic endpoints respond correctly, typically after deploy.
- **Plain English:** 'Does it turn on and do the basics?'
- **Example / analogy:** Switching on a new appliance to see if smoke comes out.
- **Snippet:**
```python
assert httpx.get(BASE + '/health').status_code == 200
```
- **Snippet explained:** The simplest smoke check.
- **Related terms:** [[Test client]], [[Health check]]

### curl
<!-- cat: api -->
- **Technical definition:** A command-line tool for making HTTP requests; flags: -i headers, -X method, -H header, -d body, -w output format, -o output file.
- **Plain English:** Typing a web request in the terminal.
- **Example / analogy:** A hand-operated browser.
- **Snippet:**
```bash
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text":"great"}'
```
- **Snippet explained:** POSTs JSON; -s hides the progress meter.
- **Related terms:** [[HTTP]], [[GET vs POST]]

### Health check
<!-- cat: api -->
- **Technical definition:** A cheap endpoint (usually GET /health) reporting that the process is alive (liveness) and optionally that the model is loaded (readiness); used by load balancers and orchestrators.
- **Plain English:** A 'are you okay?' address orchestrators ping.
- **Example / analogy:** A pulse check.
- **Snippet:**
```python
@app.get("/health")
def health():
    return {"status": "ok"}
```
- **Snippet explained:** Returns 200 when the app can answer.
- **Related terms:** [[Smoke test]], [[Docker health check]]

### Decorator
<!-- cat: python -->
- **Technical definition:** A function that wraps another function; written with `@name` above a def. Web frameworks use it to register routes.
- **Plain English:** A label that attaches extra behaviour to a function.
- **Example / analogy:** A gift-wrap around a present.
- **Snippet:**
```python
@app.get('/health')
def health(): ...
```
- **Snippet explained:** Equivalent to health = app.get('/health')(health).
- **Related terms:** [[Route]]

### Error handler
<!-- cat: api -->
- **Technical definition:** A function registered to turn an exception or status (404, 405, any Exception) into a custom response, e.g. JSON instead of HTML, without leaking stack traces.
- **Plain English:** A safety net that decides what the user sees when something goes wrong.
- **Example / analogy:** A polite receptionist when the manager is unavailable.
- **Snippet:**
```python
@app.errorhandler(404)
def nf(e):
    return jsonify(error='route not found'), 404
```
- **Snippet explained:** Replaces Flask's HTML 404 page with JSON.
- **Related terms:** [[Status code]], [[HTTP 500]]

### Port
<!-- cat: api -->
- **Technical definition:** A numeric endpoint (0-65535) on a host where a server listens; two programs cannot listen on the same port.
- **Plain English:** A numbered door on the computer.
- **Example / analogy:** Apartment numbers in one building address.
- **Snippet:**
```bash
lsof -i :8000
```
- **Snippet explained:** Shows which process is using port 8000 ('Address already in use' fix).
- **Related terms:** [[Host]], [[Docker port mapping]]
