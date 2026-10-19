# Walkthrough 06 — HTTP errors, timeouts, connection failures

## Goal
Distinguish client errors (4xx), server errors (5xx), slow servers (timeout) and unreachable servers (connection error) — and handle each differently. Prepares you for [module 09](../../09_api_development/) and LLM API calls in [module 08](../../08_agent_harness/).

## Run it
```bash
cd 01_debugging_and_environment/06_http_and_network
python main.py            # local fake API on port 8765, offline
python broken_http.py     # EXPECTED crash: ReadTimeout (port 8766)
```

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
