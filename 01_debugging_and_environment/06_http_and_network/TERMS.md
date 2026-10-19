# Terms — HTTP Errors, Timeouts and Connection Failures

Folder: `01_debugging_and_environment/06_http_and_network` · Files covered: `main.py`, `broken_http.py`, `WALKTHROUGH.md`
Back to module: [01 Debugging and environment README](../README.md)

**Terms in this folder:** [HTTP status code](#http-status-code) · [2xx / 4xx / 5xx families](#2xx--4xx--5xx-families) · [requests (library)](#requests-library) · [requests.get](#requestsget) · [Response object](#response-object) · [timeout parameter](#timeout-parameter) · [raise_for_status()](#raise_for_status) · [requests.HTTPError](#requestshttperror) · [requests.Timeout / ReadTimeout](#requeststimeout--readtimeout) · [requests.ConnectionError](#requestsconnectionerror) · [http.server.HTTPServer](#httpserverhttpserver) · [BaseHTTPRequestHandler and do_GET](#basehttprequesthandler-and-do_get) · [Response headers and body (send_response, wfile)](#response-headers-and-body-send_response-wfile) · [threading.Thread (daemon)](#threadingthread-daemon) · [localhost and ports](#localhost-and-ports) · [Retry with back-off](#retry-with-back-off) · [Fake API (local test server)](#fake-api-local-test-server) · [if __name__ == "__main__" for reuse](#if-__name__--__main__-for-reuse) · [curl -i](#curl--i) · [JSON body (r.json())](#json-body-rjson)

---

## HTTP status code  `(concept)`
**1. Definition:** An HTTP status code is a three-digit number in a server's response indicating the outcome of the request, e.g. 200 OK, 404 Not Found, 500 Internal Server Error.
**2. In plain words:** It's the server's short answer: "worked", "you asked for something wrong", or "I failed". Reading it tells you whether to fix your request or wait for the server.
**3. Example to quote:**
> "`GET /missing` returned 404, meaning my request was wrong, while `GET /boom` returned 500, meaning the server failed."
```python
r = requests.get(url, timeout=3)
print(r.status_code)
```
**Seen in:** `main.py`, status codes loop

## 2xx / 4xx / 5xx families  `(concept)`
**1. Definition:** 2xx codes mean success, 4xx codes mean a client error (the request is wrong), and 5xx codes mean a server error (the server failed).
**2. In plain words:** Common ones: 200 OK, 201 created, 400 bad request, 401/403 auth, 404 not found, 422 validation failed (FastAPI), 429 too many requests, 500 server bug, 503 overloaded. Retrying a 4xx never helps; retrying a 503 later might.
**3. Example to quote:**
> "A 4xx tells me to fix my request; a 5xx or 429 may succeed if I retry later with back-off."
```text
200 OK · 404 Not Found · 500 Server Error
```
**Seen in:** `main.py` comments; `WALKTHROUGH.md`

## requests (library)  `(third-party: requests)`
**1. Definition:** `requests` is a third-party Python library for sending HTTP requests with a simple API.
**2. In plain words:** It's the usual way to call web APIs from scripts, including LLM APIs. It isn't in the standard library, so it must be installed in your environment.
**3. Example to quote:**
> "I use `requests` to call my health endpoint and check the status code."
```python
import requests
```
**Seen in:** `main.py`, import

## requests.get  `(third-party: requests)`
**1. Definition:** `requests.get(url, timeout=...)` sends an HTTP GET request and returns a `Response` object.
**2. In plain words:** GET asks a server for data without changing anything. You pass the URL and, importantly, a timeout.
**3. Example to quote:**
> "`requests.get(base + '/ok', timeout=3)` returned a response with status 200."
```python
r = requests.get("http://127.0.0.1:8765/ok", timeout=3)
```
**Seen in:** `main.py`

## Response object  `(third-party: requests)`
**1. Definition:** The `Response` returned by `requests` exposes `.status_code`, `.json()`, `.text`, `.headers` and related fields.
**2. In plain words:** It bundles everything the server sent back. Use `.status_code` first, then `.json()` to parse the body.
**3. Example to quote:**
> "I printed `r.status_code` and `r.json()` to see `404` and `{'error': 'no such route'}`."
```python
print(r.status_code, r.text)
```
**Seen in:** `main.py`

## timeout parameter  `(concept)`
**1. Definition:** The `timeout` argument of `requests.get` is the maximum number of seconds to wait for the server before raising an error; the default is no timeout.
**2. In plain words:** Without it, a hung server can freeze your program forever. Always set one. The slow endpoint takes 2 seconds, so `timeout=0.5` fails and `timeout=5` succeeds.
**3. Example to quote:**
> "I always pass `timeout=3`, because the default is to wait forever and a hung model server would freeze my client."
```python
requests.get(url, timeout=3)
```
**Seen in:** `main.py`, comments and `/slow` call

## raise_for_status()  `(third-party: requests)`
**1. Definition:** `Response.raise_for_status()` raises `requests.HTTPError` if the response status is 4xx or 5xx, and does nothing otherwise.
**2. In plain words:** It turns "bad status" into an exception so you can't accidentally ignore it. Without it, a 500 response is just another response object.
**3. Example to quote:**
> "`requests.get(base + '/boom').raise_for_status()` raised `HTTPError: 500 Server Error`."
```python
r.raise_for_status()
```
**Seen in:** `main.py`, part 2

## requests.HTTPError  `(third-party: requests)`
**1. Definition:** `requests.HTTPError` is the exception raised by `raise_for_status()` for 4xx and 5xx responses.
**2. In plain words:** It means the server answered, but with an error code. Handle it differently from timeouts or connection failures.
**3. Example to quote:**
> "I caught `requests.HTTPError` and printed `500 Server Error: Internal Server Error for url ...`."
```python
except requests.HTTPError as err:
    print(err)
```
**Seen in:** `main.py`

## requests.Timeout / ReadTimeout  `(third-party: requests)`
**1. Definition:** `requests.Timeout` is raised when a request exceeds its timeout; `ReadTimeout` means the connection was made but the server did not respond in time.
**2. In plain words:** The server is reachable but slow, like a model taking 2 seconds. Retry with a longer timeout or back-off, or report "server too slow".
**3. Example to quote:**
> "`/slow` takes 2 s, so `timeout=0.5` raised `ReadTimeout`; I'd retry once with `timeout=5`."
```python
try:
    requests.get(url, timeout=0.5)
except requests.Timeout:
    print("too slow")
```
**Seen in:** `main.py`; `broken_http.py`

## requests.ConnectionError  `(third-party: requests)`
**1. Definition:** `requests.ConnectionError` is raised when no connection can be made to the host and port.
**2. In plain words:** Nothing is listening: the server isn't running, the port is wrong, or the host is wrong. Check the URL, port and whether the server started.
**3. Example to quote:**
> "Requesting port 9 gave a `ConnectionError`, so I knew the server wasn't running or the port was wrong."
```python
except requests.ConnectionError:
    print("server not running / wrong port / wrong host")
```
**Seen in:** `main.py`, part 4

## http.server.HTTPServer  `(standard library)`
**1. Definition:** `http.server.HTTPServer` is a basic standard-library web server that listens on an address and delegates each request to a handler class.
**2. In plain words:** It lets the demo run a tiny fake API locally with no internet and no extra libraries. It's for demos and tests, not production.
**3. Example to quote:**
> "I started a fake API on `127.0.0.1:8765` with `HTTPServer` so I could demonstrate error codes offline."
```python
server = HTTPServer(("127.0.0.1", 8765), Handler)
```
**Seen in:** `main.py`, `start_server`

## BaseHTTPRequestHandler and do_GET  `(standard library)`
**1. Definition:** `BaseHTTPRequestHandler` is the base class for request handlers; defining `do_GET(self)` makes it respond to GET requests, with the requested URL in `self.path`.
**2. In plain words:** Python calls `do_GET` for each GET. The code checks `self.path` to decide between `/ok`, `/boom`, `/slow` and a 404 for everything else.
**3. Example to quote:**
> "My handler's `do_GET` returns 200 for `/ok`, 500 for `/boom` and 404 for unknown paths."
```python
def do_GET(self):
    if self.path == "/ok":
        self._reply(200, b"{}")
```
**Seen in:** `main.py`, `Handler`

## Response headers and body (send_response, wfile)  `(standard library)`
**1. Definition:** In a handler, `send_response(code)` sets the status, `send_header(name, value)` adds a header, `end_headers()` finishes the header section, and `wfile.write(bytes)` sends the body.
**2. In plain words:** These are the steps of replying: status line, headers (like `Content-Type: application/json`), blank line, then the body as bytes (`b'...'`).
**3. Example to quote:**
> "`_reply` sends the status, a JSON content-type header and the body bytes."
```python
self.send_response(200)
self.send_header("Content-Type", "application/json")
self.end_headers()
self.wfile.write(b'{"ok": true}')
```
**Seen in:** `main.py`, `_reply`

## threading.Thread (daemon)  `(standard library)`
**1. Definition:** `threading.Thread(target=fn, daemon=True).start()` runs `fn` in a background thread; daemon threads are stopped automatically when the main program exits.
**2. In plain words:** The server's `serve_forever()` blocks, so it's started in a background thread, letting the main code make requests to it. Daemon status means no cleanup is needed at exit.
**3. Example to quote:**
> "I ran `server.serve_forever` in a daemon thread so my main script could send requests to it."
```python
threading.Thread(target=server.serve_forever, daemon=True).start()
```
**Seen in:** `main.py`, `start_server`

## localhost and ports  `(concept)`
**1. Definition:** `127.0.0.1` (localhost) refers to your own machine, and a port number (like 8765) identifies one program listening on it.
**2. In plain words:** An address needs both host and port. If the server listens on 8766 but you call 8765, nothing answers and you get a `ConnectionError`. In Docker, `localhost` inside a container is the container itself.
**3. Example to quote:**
> "The URL was `http://127.0.0.1:8765`: my own machine, port 8765. A wrong port gave `ConnectionError`."
```python
base = "http://127.0.0.1:8765"
```
**Seen in:** `main.py`, `base`; BREAK IT

## Retry with back-off  `(concept)`
**1. Definition:** Retry with back-off means repeating a failed request after progressively longer waits and with a limit on attempts.
**2. In plain words:** Timeouts, 429 and 503 may be temporary, so retry later. 4xx errors (your mistake) should not be retried. Aggressive retries can overload a struggling server, so wait longer each time.
**3. Example to quote:**
> "On a timeout I retry once with a longer timeout; on a 404 I don't retry because the request itself is wrong."
```python
# attempt 1 -> wait 1s -> attempt 2 -> wait 2s -> give up
```
**Seen in:** `WALKTHROUGH.md`, handling plan and trade-off

## Fake API (local test server)  `(concept)`
**1. Definition:** A fake API is a small local server that imitates real behaviours (success, errors, slowness) so client code can be tested without the internet.
**2. In plain words:** It makes failures reproducible on demand: `/boom` always returns 500, `/slow` always takes 2 seconds. Great for practising error handling.
**3. Example to quote:**
> "I tested my client against a local fake API with `/ok`, `/boom` and `/slow` endpoints to practise each failure."
```python
server = start_server(8765)
```
**Seen in:** `main.py`, `Handler`

## if __name__ == "__main__" for reuse  `(syntax)`
**1. Definition:** Placing demo code under `if __name__ == "__main__":` lets other files import functions from the module without running the demo.
**2. In plain words:** `broken_http.py` does `from main import start_server`; thanks to the guard, importing `main` does not run the whole demo again.
**3. Example to quote:**
> "`broken_http.py` imports `start_server` from `main.py` safely because the demo is under the main guard."
```python
from main import start_server
```
**Seen in:** `main.py`, `broken_http.py`

## curl -i  `(command)`
**1. Definition:** `curl -i URL` sends an HTTP request from the terminal and prints the response headers (including the status line) along with the body.
**2. In plain words:** It is the quickest way to inspect what a server actually returns, independent of your Python code.
**3. Example to quote:**
> "I ran `curl -i http://127.0.0.1:8765/boom` to see the 500 status and headers."
```bash
curl -i http://127.0.0.1:8765/ok
```
**Seen in:** `WALKTHROUGH.md`, rebuild checklist

## JSON body (r.json())  `(third-party: requests)`
**1. Definition:** `Response.json()` parses the response body as JSON and returns Python objects, raising an error if the body is not valid JSON.
**2. In plain words:** APIs usually reply in JSON. `r.json()` turns the text into a dict you can index, such as `r.json()["status"]`.
**3. Example to quote:**
> "`r.json()` returned `{'status': 'healthy'}` which I printed beside the status code."
```python
print(r.json()["status"])
```
**Seen in:** `main.py`; `broken_http.py`
