# Step 01 — HTTP basics with the standard library

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

## Say it aloud
"HTTP is request/response text over a socket. A request has a method, path, headers and optional body; a response has a status code, headers and body. `http.server` shows the raw mechanics; frameworks like Flask and FastAPI do the same parsing and let me write just the logic. Trade-off: this stdlib server handles requests one at a time and has no routing/validation, so it's only for learning."
