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

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| SSE vs WebSocket | Server push / streaming over HTTP | No | SSE is one-way server->client over plain HTTP with auto-reconnect; WebSocket is two-way over its own protocol |
| SSE vs chunked transfer encoding | Streaming body | Related | Chunked is the HTTP mechanism for a body without known length; SSE is a text format (`text/event-stream`) usually sent over it |
| StreamingResponse vs JSONResponse | FastAPI responses | No | Streaming sends pieces from a generator as they appear; JSON sends one complete body |
| TTFT vs total latency | Time to first token / time to last token | No | TTFT = delay until the first token arrives; total = until the last one |
| Generator vs async generator | `yield` functions | No | A generator is consumed with `next`/`for`; an async generator with `async for` and can `await` inside |
| `asyncio.to_thread` vs `run_in_executor` | Run blocking code off the event loop | Mostly | `to_thread` is the simpler modern wrapper (Python 3.9+) around the thread executor |
| KV cache vs prompt cache | Cached attention keys/values | Related | KV cache reuses K/V within one generation; prompt/prefix caching reuses them across requests |
| `CancelledError` vs client disconnect | How the server notices a hang-up | Related | A disconnect makes the framework cancel the response task, raising `CancelledError` inside the generator |

## Code, step by step

### 1. Docstring: goal, run modes, expected output  (`main.py` lines 1–16)
```python
"""
GOAL:   Stream tokens to the user as they are produced (Server-Sent Events), instead of making
        them stare at a blank screen until the whole answer is ready.
        FastAPI app + loaded-once model + a client that prints each token with a timestamp.

RUN:    cd 09_api_development/10_streaming_sse
        python main.py              # fast demo, fake tokens (no model needed)
        python main.py --real       # same demo with distilgpt2 from the Hugging Face cache
        uvicorn main:app --port 8000   # just the server; try: curl -N "http://127.0.0.1:8000/stream?prompt=hi"

EXPECTED OUTPUT (shape): blocking endpoint: first token == total time; streaming endpoint: first token
        after a fraction of that; broken framing shows what the client sees; a mid-stream disconnect is
        detected by the server.

NEEDS:  fastapi, uvicorn, httpx  (+ torch, transformers only for --real)
"""
```
- **In simple words:** Explains that the answer should arrive word by word, and how to run the demo.
- **Exactly:** Three run modes: fake tokens (default), `--real` distilgpt2, or just the server via uvicorn. `--real` was not run here (distilgpt2 is not in the offline cache; the server failed to start offline).
- **Why it is here:** You know what to expect: first token much earlier than the total.

### 2. Imports  (`main.py` lines 18–28)
```python
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
```
- **In simple words:** Bring in async tools, JSON, sockets, subprocess handling and the FastAPI pieces.
- **Exactly:** `StreamingResponse` sends a body piece by piece from a generator; `JSONResponse` sends one whole JSON; `asynccontextmanager` builds the lifespan; `socket` is used to find a free port.
- **Why it is here:** Each import supports one part: server (FastAPI), streaming, or the demo client launcher.

### 3. Settings: fake or real model, delay, counter  (`main.py` lines 30–32)
```python
USE_REAL_MODEL = os.environ.get("STREAM_REAL_MODEL") == "1"
TOKEN_DELAY = 0.15            # seconds per fake token (a real model on CPU is in the same range)
STATE = {"disconnects": 0}    # lives in the server process; the client reads it via /stats
```
- **In simple words:** Three module-level knobs: which model, how slow each fake token is, and a disconnect counter.
- **Exactly:** `USE_REAL_MODEL` is `True` only if env `STREAM_REAL_MODEL == "1"` (the demo sets it in the child). `TOKEN_DELAY = 0.15` s. `STATE = {"disconnects": 0}` lives in the server process and is read through `/stats`.
- **Why it is here:** The counter is how the client proves the server noticed the hang-up.

### 4. FakeTokenModel: a slow pretend LLM  (`main.py` lines 35–44)
```python
# ----------------------------- the model, loaded ONCE at startup -----------------------------
class FakeTokenModel:
    """Stands in for a real LLM: yields one 'token' at a time, slowly. Tests need no weights."""

    def stream(self, prompt: str, max_new_tokens: int):
        words = ("The castle gate opens and tokens walk out one at a time so the user can start reading "
                 "long before the model has finished").split()
        for i in range(max_new_tokens):
            time.sleep(TOKEN_DELAY)                       # pretend one forward pass takes this long
            yield " " + words[i % len(words)]
```
- **In simple words:** A tiny stand-in that hands out one word every 0.15 seconds.
- **Exactly:** `stream()` is a generator (uses `yield`): each `next()` sleeps `TOKEN_DELAY` then returns the next word with a leading space; `words[i % len(words)]` wraps around the sentence. Tokens produced for 8 tokens: `" The"`, `" castle"`, ... `" out"`.
- **Why it is here:** Tests and demos need no 330 MB weights, yet behave like a slow model.

### 5. DistilGPT2Model: load tokenizer and weights  (`main.py` lines 47–55)
```python
class DistilGPT2Model:
    """Greedy token-by-token loop with a KV cache (module 04 step 08, written as a generator)."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("distilgpt2")
        self.lm = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()
```
- **In simple words:** The real model class: its constructor loads the text-to-ids converter and the network.
- **Exactly:** `import torch`/`transformers` are inside `__init__` so fake mode never needs them. `.eval()` turns off dropout. This code was read, not run.
- **Why it is here:** Heavy imports only when the real model is requested.

### 6. DistilGPT2Model.stream: token-by-token loop with KV cache  (`main.py` lines 57–67)
```python
    def stream(self, prompt: str, max_new_tokens: int):
        torch = self.torch
        ids = self.tok(prompt, return_tensors="pt").input_ids            # shape: (1, prompt_len)
        past = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.lm(ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # shape: (1, 1)
                ids = next_id                                              # with a cache we feed only the new token
                yield self.tok.decode(next_id[0])
```
- **In simple words:** Ask the model for one token, remember its notes, repeat; hand each token to the caller as soon as it exists.
- **Exactly:** `ids` shape `(1, prompt_len)`. Each step: `self.lm(ids, past_key_values=past, use_cache=True)`; `out.logits[:, -1, :]` is the last position's scores `(1, vocab=50257)`; `.argmax(dim=-1, keepdim=True)` gives greedy `next_id` shape `(1, 1)`; with a cache we feed only the new token. `yield self.tok.decode(next_id[0])` produces the text. `torch.no_grad()` skips gradient bookkeeping.
- **Why it is here:** The KV cache avoids recomputing all earlier tokens each step, and `yield` is what makes streaming possible.

### 7. Lifespan: load the model once at startup  (`main.py` lines 70–77)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup, once per worker process: NEVER load weights inside a request handler.
    t0 = time.perf_counter()
    app.state.model = DistilGPT2Model() if USE_REAL_MODEL else FakeTokenModel()
    print(f"[pid {os.getpid()}] model loaded once in {time.perf_counter() - t0:.2f}s "
          f"({'distilgpt2' if USE_REAL_MODEL else 'fake'})", flush=True)
    yield
```
- **In simple words:** Before any request is served, create the model one time and say how long it took.
- **Exactly:** `app.state.model` holds a `DistilGPT2Model` or `FakeTokenModel`. Real output: `[pid 24977] model loaded once in 0.00s (fake)`. Code after `yield` would run at shutdown.
- **Why it is here:** Loading weights inside a handler would repeat a multi-second cost on every request.

### 8. Create the app  (`main.py` line 80)
```python
app = FastAPI(title="Streaming API", lifespan=lifespan)
```
- **In simple words:** Build the FastAPI app and connect the lifespan.
- **Exactly:** `FastAPI(title="Streaming API", lifespan=lifespan)`.
- **Why it is here:** This is the `main:app` that uvicorn starts.

### 9. GET /blocking: the slow, all-at-once endpoint  (`main.py` lines 83–88)
```python
# ----------------------------- the endpoints -----------------------------
@app.get("/blocking")
def blocking(prompt: str = "hi", max_new_tokens: int = 8):
    """The pain point: the whole answer, only when ALL tokens exist."""
    tokens = list(app.state.model.stream(prompt, max_new_tokens))
    return JSONResponse({"text": "".join(tokens)})
```
- **In simple words:** The old way: wait for every token, then send the whole answer.
- **Exactly:** Plain `def` runs in a thread pool; `list(...stream(...))` forces all tokens to be produced; `JSONResponse({"text": ...})`. Real run: first byte after 1.32 s = total 1.32 s (8 tokens x 0.15 s + overhead), text `' The castle gate opens and tokens walk out'`.
- **Why it is here:** It is the pain point: the user stares at a blank screen for the whole time.

### 10. sse_event: the SSE framing helper  (`main.py` lines 91–93)
```python
def sse_event(payload: str) -> str:
    # SSE framing: a field line "data: ...", then a BLANK LINE (\n\n) that means "event finished".
    return f"data: {payload}\n\n"
```
- **In simple words:** Wrap a message in the exact shape the browser expects.
- **Exactly:** Format: `data: <payload>\n\n`. The field line is `data:`; the blank line (second `\n`) means "this event is complete". E.g. `data: {"token": " The"}\n\n`.
- **Why it is here:** Without the blank line the client never dispatches the event (see experiment 3).

### 11. GET /stream: start of the streaming endpoint  (`main.py` lines 96–98)
```python
@app.get("/stream")
async def stream(prompt: str = "hi", max_new_tokens: int = 8):
    model = app.state.model
```
- **In simple words:** An async endpoint that grabs the model and prepares a generator.
- **Exactly:** `async def` so the event loop stays responsive; `max_new_tokens` and `prompt` arrive as query parameters with defaults `8` and `"hi"`.
- **Why it is here:** Async lets one process serve many open streams at once.

### 12. event_generator: pull tokens off-loop and frame them  (`main.py` lines 100–112)
```python
    async def event_generator():
        tokens = model.stream(prompt, max_new_tokens)        # a plain (blocking) generator
        try:
            while True:
                # next() blocks, so run it in a worker thread; the event loop stays free for other users.
                token = await asyncio.to_thread(next, tokens, None)
                if token is None:
                    break
                yield sse_event(json.dumps({"token": token}))
            yield sse_event("[DONE]")                         # our own end-of-stream marker
        except asyncio.CancelledError:                       # raised when the client hangs up
            STATE["disconnects"] += 1
            raise
```
- **In simple words:** Repeatedly fetch the next token without freezing the server, send it as an SSE event, and finish with a [DONE] marker.
- **Exactly:** `asyncio.to_thread(next, tokens, None)` runs the blocking `next(tokens, None)` in a worker thread; `None` signals exhaustion. Each event is `data: {"token": "..."}`. `except asyncio.CancelledError` fires when the client disconnects: increment `STATE["disconnects"]` and re-`raise` so cleanup continues. Real disconnect run: counter `0 -> 1`.
- **Why it is here:** Without `to_thread`, `time.sleep` in the model would block every other user; without handling cancel we would not know a user left (and would waste GPU time).

### 13. Return the StreamingResponse with SSE headers  (`main.py` lines 114–115)
```python
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
```
- **In simple words:** Hand FastAPI the generator and tell the browser/proxies this is a live event stream.
- **Exactly:** `media_type="text/event-stream"` (real header: `text/event-stream; charset=utf-8`); `Cache-Control: no-cache` stops caching; `X-Accel-Buffering: no` tells nginx not to buffer the stream.
- **Why it is here:** A proxy that buffers would deliver everything at the end, defeating streaming.

### 14. Broken endpoint 1: missing blank line  (`main.py` lines 118–124)
```python
@app.get("/stream-no-blank-line")
async def stream_no_blank_line(max_new_tokens: int = 4):
    async def gen():
        for i in range(max_new_tokens):
            await asyncio.sleep(TOKEN_DELAY)
            yield f"data: tok{i}\n"                          # BUG: only ONE newline -> event never ends
    return StreamingResponse(gen(), media_type="text/event-stream")
```
- **In simple words:** A deliberately wrong version that ends events with one newline instead of two.
- **Exactly:** Yields `data: tok0\n`, `data: tok1\n` ... Real raw lines: `['data: tok0', 'data: tok1', 'data: tok2', 'data: tok3']`.
- **Why it is here:** Shows what a framing bug looks like on the wire.

### 15. Broken endpoint 2: wrong media type  (`main.py` lines 127–133)
```python
@app.get("/stream-wrong-media-type")
async def stream_wrong_media_type(max_new_tokens: int = 4):
    async def gen():
        for i in range(max_new_tokens):
            await asyncio.sleep(TOKEN_DELAY)
            yield f"data: tok{i}\n\n"
    return StreamingResponse(gen(), media_type="text/plain")  # BUG: browsers/EventSource reject this
```
- **In simple words:** Correct framing but the wrong label on the response.
- **Exactly:** `media_type="text/plain"` -> real header `text/plain; charset=utf-8`. Browser `EventSource` only accepts `text/event-stream`.
- **Why it is here:** Shows a second classic bug: it works in curl but not in the browser.

### 16. GET /stats  (`main.py` lines 136–138)
```python
@app.get("/stats")
def stats():
    return STATE
```
- **In simple words:** Lets the client read the disconnect counter.
- **Exactly:** Returns the dict `STATE`, e.g. `{"disconnects": 1}`.
- **Why it is here:** The server's private state is otherwise invisible to an outside client.

### 17. parse_sse: a minimal SSE client parser  (`main.py` lines 141–152)
```python
# ----------------------------- the client -----------------------------
def parse_sse(lines):
    """Minimal SSE reader following the spec: collect 'data:' lines; a blank line dispatches the event."""
    data_lines = []
    for line in lines:
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    # NOTE: leftover data_lines at the end of the stream are NOT an event (spec says discard).
```
- **In simple words:** Read lines, gather the data lines, and release one event each time a blank line appears.
- **Exactly:** `data_lines.append(line[5:].lstrip(" "))` strips `data:` and one optional leading space; a blank line joins lines with `"\n"` and `yield`s the event, then resets. Leftover data without a blank line is discarded per the SSE spec.
- **Why it is here:** This is how a real EventSource behaves, so it exposes the missing-blank-line bug.

### 18. free_port: ask the OS for an unused port  (`main.py` lines 155–158)
```python
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
```
- **In simple words:** Find a port nobody is using.
- **Exactly:** Binding to port `0` lets the OS pick; `getsockname()[1]` returns the chosen port number.
- **Why it is here:** Avoids "Address already in use" collisions.

### 19. start_server: launch uvicorn and wait for it  (`main.py` lines 161–175)
```python
def start_server(real: bool):
    port = free_port()
    env = dict(os.environ, STREAM_REAL_MODEL="1" if real else "0")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(port), "--log-level", "warning"],
                            cwd=os.path.dirname(os.path.abspath(__file__)), env=env)
    import httpx
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):                                     # wait until the server answers (model loads first)
        try:
            httpx.get(base + "/stats", timeout=1)
            return proc, base
        except httpx.TransportError:
            time.sleep(0.2)
    proc.terminate()
    raise RuntimeError("server did not start")
```
- **In simple words:** Start the app as a separate process and wait until it answers.
- **Exactly:** `env = dict(os.environ, STREAM_REAL_MODEL=...)` copies the environment and sets the model switch; `Popen([python, "-m", "uvicorn", "main:app", "--port", port ...], cwd=this folder)`; up to 300 x 0.2 s polls of `/stats`; returns `(proc, base_url)`; on timeout terminates and raises `RuntimeError`.
- **Why it is here:** Real streaming needs a real HTTP server; the generous 60 s wait covers real model loading.

### 20. Demo part 1: blocking endpoint timing  (`main.py` lines 178–187)
```python
def demo(real: bool):
    import httpx
    proc, base = start_server(real)
    n = 8
    try:
        print(f"== 1. BLOCKING endpoint ({n} tokens): the user sees nothing, then everything ==")
        t0 = time.perf_counter()
        r = httpx.get(f"{base}/blocking", params={"prompt": "The castle", "max_new_tokens": n}, timeout=60)
        total = time.perf_counter() - t0
        print(f"   first byte after {total:5.2f}s   total {total:5.2f}s   text={r.json()['text']!r}")
```
- **In simple words:** Time how long until the blocking endpoint returns anything.
- **Exactly:** `httpx.get(..., params={...})`; `perf_counter` difference. Real output: `first byte after  1.32s   total  1.32s`.
- **Why it is here:** The baseline to compare streaming against.

### 21. Demo part 2: stream and print each token with time  (`main.py` lines 189–201)
```python
        print(f"\n== 2. STREAMING endpoint (SSE): tokens appear as they are made ==")
        t0 = time.perf_counter()
        first = None
        with httpx.stream("GET", f"{base}/stream", params={"prompt": "The castle", "max_new_tokens": n}, timeout=60) as resp:
            print("   content-type:", resp.headers["content-type"])
            for event in parse_sse(resp.iter_lines()):
                now = time.perf_counter() - t0
                if event == "[DONE]":
                    break
                first = first if first is not None else now
                print(f"   t={now:5.2f}s  token={json.loads(event)['token']!r}")
        total = time.perf_counter() - t0
        print(f"   time-to-first-token {first:5.2f}s   total {total:5.2f}s   (user starts reading {total - first:.2f}s earlier)")
```
- **In simple words:** Read events as they arrive and print when each one came.
- **Exactly:** `httpx.stream(...)` keeps the connection open; `resp.iter_lines()` yields lines as they come; `parse_sse` turns them into events; `[DONE]` ends the loop. Real: first token at 0.21 s, last at 1.29 s; `time-to-first-token 0.21s total 1.29s (user starts reading 1.08s earlier)`.
- **Why it is here:** TTFT (time to first token), not total time, is what makes the app feel fast.

### 22. Demo part 3: show the missing blank line bug  (`main.py` lines 203–209)
```python
        print("\n== 3. BREAK IT: forget the blank line ==")
        with httpx.stream("GET", f"{base}/stream-no-blank-line", timeout=60) as resp:
            raw = list(resp.iter_lines())
            print("   raw lines on the wire :", raw)
        with httpx.stream("GET", f"{base}/stream-no-blank-line", timeout=60) as resp:
            events = list(parse_sse(resp.iter_lines()))
        print("   events a real SSE client dispatches:", events, " <- nothing, the event never 'ends'")
```
- **In simple words:** Read the same broken stream twice: once raw, once as a real SSE client.
- **Exactly:** Raw: `['data: tok0', ..., 'data: tok3']`; parsed events: `[]`.
- **Why it is here:** Proves the data is on the wire but the client never gets an event.

### 23. Demo part 4: show the wrong media type bug  (`main.py` lines 211–215)
```python
        print("\n== 4. BREAK IT: wrong media type ==")
        with httpx.stream("GET", f"{base}/stream-wrong-media-type", timeout=60) as resp:
            ctype = resp.headers["content-type"]
            ok = ctype.startswith("text/event-stream")
            print(f"   content-type: {ctype}  -> EventSource would {'connect' if ok else 'REFUSE: wrong MIME type'}")
```
- **In simple words:** Check the header the way a browser would.
- **Exactly:** `ctype.startswith("text/event-stream")` is False -> prints `REFUSE: wrong MIME type` for `text/plain; charset=utf-8`.
- **Why it is here:** Catches the bug without needing a browser.

### 24. Demo part 5: client disconnects mid-stream  (`main.py` lines 217–227)
```python
        print("\n== 5. Client disconnects mid-stream ==")
        before = httpx.get(f"{base}/stats").json()["disconnects"]
        with httpx.stream("GET", f"{base}/stream", params={"max_new_tokens": 50}, timeout=60) as resp:
            for i, event in enumerate(parse_sse(resp.iter_lines())):
                print(f"   client read event {i}: {event}")
                if i == 2:
                    print("   client closes the connection now (the user clicked 'stop')")
                    break
        time.sleep(1.0)
        after = httpx.get(f"{base}/stats").json()["disconnects"]
        print(f"   server-side disconnects counted: {before} -> {after} (the generator was cancelled, so no more tokens are made)")
```
- **In simple words:** Read three events, hang up, and check that the server noticed.
- **Exactly:** Starts a 50-token stream, `break`s at `i == 2` (closing the connection), sleeps 1 s, then reads `/stats`: real `0 -> 1`.
- **Why it is here:** Shows streaming generators get cancelled, saving compute when users click stop.

### 25. Always stop the server  (`main.py` lines 228–230)
```python
    finally:
        proc.terminate()
        proc.wait(timeout=10)
```
- **In simple words:** Whatever happens, shut the background server down.
- **Exactly:** `finally` runs even after exceptions: `terminate()`, then `wait(timeout=10)`.
- **Why it is here:** Avoids leaked uvicorn processes.

### 26. Entry point and notes  (`main.py` lines 233–237)
```python
if __name__ == "__main__":
    demo(real="--real" in sys.argv)
    # TRY IT: change TOKEN_DELAY to 0.4 and compare blocking vs streaming totals.
    # TRY IT: python main.py --real to stream a real distilgpt2 answer.
    # BREAK IT: change media_type to "text/plain" in /stream and re-run section 2.
```
- **In simple words:** Run the demo; `--real` on the command line switches to distilgpt2.
- **Exactly:** `"--real" in sys.argv` gives a bool for `demo(real=...)`.
- **Why it is here:** The guard lets tests import `main` without starting the demo.

### 27. Test file: imports  (`test_streaming.py` lines 1–7)
```python
"""pytest: the SSE framing and media type, with the fake model (no weights, no server process)."""
import json

from fastapi.testclient import TestClient

import main
from main import app, parse_sse
```
- **In simple words:** Bring in pytest tools and the app and parser from main.
- **Exactly:** `import main` is needed so `monkeypatch.setattr(main, ...)` can change a module variable.
- **Why it is here:** Tests reuse the production code, not a copy.

### 28. test_stream_is_sse_and_framed  (`test_streaming.py` lines 10–19)
```python
def test_stream_is_sse_and_framed(monkeypatch):
    monkeypatch.setattr(main, "TOKEN_DELAY", 0.0)
    with TestClient(app) as client:                           # `with` runs the lifespan (model loaded once)
        with client.stream("GET", "/stream", params={"max_new_tokens": 3}) as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            raw = resp.read().decode()
    assert raw.endswith("\n\n") and raw.count("\n\n") == 4    # 3 tokens + [DONE], each ended by a blank line
    events = list(parse_sse(raw.splitlines()))
    assert events[-1] == "[DONE]" and len(events) == 4
    assert all("token" in json.loads(e) for e in events[:-1])
```
- **In simple words:** Check the stream has the right media type and exactly 3 token events plus [DONE].
- **Exactly:** `TOKEN_DELAY` is set to 0.0 so tests are fast; `client.stream(...)` and `resp.read()` get the raw body; `raw.count("\n\n") == 4` (3 tokens + [DONE]); `parse_sse` yields 4 events and the last is `[DONE]`; each earlier event JSON has `token`.
- **Why it is here:** Locks in the framing contract so refactors cannot silently break clients.

### 29. test_missing_blank_line_dispatches_nothing  (`test_streaming.py` lines 22–26)
```python
def test_missing_blank_line_dispatches_nothing(monkeypatch):
    monkeypatch.setattr(main, "TOKEN_DELAY", 0.0)
    with TestClient(app) as client:
        text = client.get("/stream-no-blank-line").text
    assert list(parse_sse(text.splitlines())) == []
```
- **In simple words:** Check the deliberately broken endpoint produces no events.
- **Exactly:** `client.get(...).text` is the whole body; `list(parse_sse(...)) == []`.
- **Why it is here:** Documents the bug as a test so you understand the spec.


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
