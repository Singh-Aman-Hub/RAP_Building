# Step 10 — Streaming tokens with Server-Sent Events (Chapter 21, The Castle Gate)

## The story so far
Step 09 left us with a tested API where each endpoint returns its answer only when the whole answer is ready. That is fine for a sentiment label (0.3 ms). A chat model needs several seconds to write a paragraph, and the gate we built has no way to let a word out early.

## The pain point
The user presses Enter and sees a blank screen for N seconds, then the whole answer appears at once. They cannot tell if the service is working, and every second feels longer. In our demo, `/blocking` produces 8 tokens and the first byte arrives only after all 8 exist.

## The spark
The model already makes one token at a time (module 04 step 08, the generation loop). Do not wait for the loop to finish: turn the loop into a generator and send each token the moment it exists. The simplest standard way over plain HTTP is Server-Sent Events: the response never ends, has `Content-Type: text/event-stream`, and each message looks like `data: ...` followed by a blank line.

## Goal
Build a FastAPI app that loads its model once at startup, streams tokens with `StreamingResponse`, and a `httpx` client that prints each token with a timestamp so you can see time-to-first-token (TTFT) versus total time. Then break the framing on purpose and watch a client disconnect.

## Run it
```bash
cd 09_api_development/10_streaming_sse
python main.py            # fake tokens, 0.15 s each, no model needed (a few seconds: it starts uvicorn on a free port)
python main.py --real     # same demo streaming distilgpt2 from the Hugging Face cache
pytest -q                 # 2 tests with the fake model, no server process
uvicorn main:app --port 8000   # then: curl -N "http://127.0.0.1:8000/stream?prompt=hi"
```
Real output of `python main.py`:
```
== 1. BLOCKING endpoint (8 tokens): the user sees nothing, then everything ==
   first byte after  1.24s   total  1.24s   text=' The castle gate opens and tokens walk out'

== 2. STREAMING endpoint (SSE): tokens appear as they are made ==
   content-type: text/event-stream; charset=utf-8
   t= 0.16s  token=' The'
   t= 0.32s  token=' castle'
   t= 0.47s  token=' gate'
   t= 0.63s  token=' opens'
   t= 0.78s  token=' and'
   t= 0.93s  token=' tokens'
   t= 1.09s  token=' walk'
   t= 1.24s  token=' out'
   time-to-first-token  0.16s   total  1.24s   (user starts reading 1.08s earlier)

== 3. BREAK IT: forget the blank line ==
   raw lines on the wire : ['data: tok0', 'data: tok1', 'data: tok2', 'data: tok3']
   events a real SSE client dispatches: []  <- nothing, the event never 'ends'

== 4. BREAK IT: wrong media type ==
   content-type: text/plain; charset=utf-8  -> EventSource would REFUSE: wrong MIME type

== 5. Client disconnects mid-stream ==
   client read event 0: {"token": " The"}
   client read event 1: {"token": " castle"}
   client read event 2: {"token": " gate"}
   client closes the connection now (the user clicked 'stop')
   server-side disconnects counted: 0 -> 1 (the generator was cancelled, so no more tokens are made)
```
(The server also prints one line, `model loaded once in 0.00s (fake)`; with `--real` it says `8.33s (distilgpt2)`, paid once at startup, not per request. With `--real` the whole demo prints in the same shape, with TTFT about 0.04 s and total about 0.18 s on this machine.)

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `StreamingResponse(gen, media_type="text/event-stream")` | Send the generator's chunks as they are yielded, keep the connection open | `main.py` `/stream` |
| `yield f"data: {payload}\n\n"` | One SSE event: field line, then a blank line | `sse_event()` |
| `async def event_generator(): ... yield` | An async generator: each `yield` becomes a network write | — |
| `await asyncio.to_thread(next, tokens, None)` | Run one blocking `next()` in a worker thread so the event loop stays free | — |
| `except asyncio.CancelledError: ... raise` | Runs when the client hangs up; re-raise so cleanup completes | disconnect counter |
| `@asynccontextmanager async def lifespan(app)` | Code before `yield` runs once at startup | model loaded once |
| `with httpx.stream("GET", url) as resp:` | Open a response without reading the whole body | client |
| `resp.iter_lines()` | Yield lines as they arrive | `parse_sse()` |

## Libraries & functions
| Name | Kind | What it does | Important args |
|---|---|---|---|
| `StreamingResponse` | fastapi (Starlette) | Streams an iterator as the body | `media_type`, `headers` |
| `httpx.stream` | httpx | Streaming HTTP client | `params=`, `timeout=` |
| `subprocess.Popen([... "-m", "uvicorn", "main:app", "--port", p])` | stdlib | Real server on a free port for the demo | `cwd`, `env` |
| `socket.bind(("127.0.0.1", 0))` | stdlib | Ask the OS for a free port | — |
| `TestClient(app)` | fastapi | In-process client used by pytest (`with` runs the lifespan) | — |
| `model.stream(prompt, n)` | our code | Generator yielding one token per forward pass (fake or distilgpt2 with KV cache) | — |

## The SSE wire format (what actually travels)
```
HTTP/1.1 200 OK
content-type: text/event-stream; charset=utf-8
cache-control: no-cache

data: {"token": " The"}

data: {"token": " castle"}

data: [DONE]

```
Rules: lines starting with `data:` carry the payload; a **blank line ends the event**; `[DONE]` is our own convention for "finished" (the browser's `EventSource` also reconnects on its own if the stream closes, so a clear end marker matters). `Cache-Control: no-cache` and `X-Accel-Buffering: no` stop proxies (such as nginx) from holding tokens in a buffer.

## Execution trace
`GET /stream?max_new_tokens=8`: uvicorn gives the request to FastAPI, which builds a `StreamingResponse` and sends the headers immediately. Then for each token: `asyncio.to_thread(next, tokens)` waits for the model (0.15 s) in a thread, `sse_event` wraps it as `data: {...}\n\n`, and the framework writes it to the socket. The client's `iter_lines()` hands each line to `parse_sse`, which dispatches an event at every blank line. Result: tokens at 0.16, 0.32, ... 1.24 s. If the client closes early, uvicorn cancels the generator task, `CancelledError` is raised at the next `await`, and the server counter goes 0 to 1.

## Experiments
1. Set `TOKEN_DELAY = 0.4`: blocking total and streaming total both grow, but TTFT stays about one token. That is the whole point: streaming does not make the model faster, it makes waiting visible.
2. Run `python main.py --real` and compare TTFT with the fake model.
3. Open two terminals and `curl -N` the stream twice; both receive tokens at the same time because the event loop is free (the blocking `next()` runs in threads).

## Debugging practice
Break it on purpose (sections 3 to 5 of the output do it for you):
- **Forget the blank line** (`yield "data: tok\n"`): the raw bytes look fine, but a spec-following client never dispatches an event, so the demo printed `events a real SSE client dispatches: []`. A browser would show nothing, then silently drop the pending data at the end.
- **Wrong media type** (`text/plain`): the bytes are fine, but `EventSource` refuses the connection (`MIME type that is not "text/event-stream"`), and proxies may buffer the response.
- **Client disconnects**: you must handle `CancelledError` (or check `await request.is_disconnected()`), otherwise the server keeps generating tokens for nobody.
- **Blocking call inside `async def`**: calling the model's `next()` directly blocks the event loop and freezes every other stream. That is why the demo uses `asyncio.to_thread`.

## Rebuild checklist
- [ ] A generator that yields one token at a time.
- [ ] `StreamingResponse(gen, media_type="text/event-stream")` with `data: ...\n\n` framing.
- [ ] Model loaded once in `lifespan`, not in the handler.
- [ ] A client that reads with `httpx.stream` and prints TTFT versus total.
- [ ] A test that checks the media type and the blank-line framing.

## The new problem (cliffhanger)
The gate is open: the model loads once, tokens stream, and the API is tested. But this runs correctly for one request on one afternoon. How do we keep it correct over months, as data changes, models get retrained and users multiply? That is a system, not an endpoint. Next: [../10_ml_system_architecture](../../10_ml_system_architecture/).

## Say it aloud
> **Say it aloud (interview box):** To stop users staring at a blank screen, I turn the model's token loop into a generator and return it in a FastAPI `StreamingResponse` with media type `text/event-stream`, framing each token as `data: ...` followed by a blank line. The model is loaded once at startup, blocking calls run in a thread so the event loop stays free, and I handle client disconnects so I stop generating for nobody. Streaming does not make the model faster, it lowers time-to-first-token, and proxies that buffer responses can break it.
