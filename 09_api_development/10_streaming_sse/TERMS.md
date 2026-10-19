# Terms — Step 10: Streaming with Server-Sent Events

Folder: `09_api_development/10_streaming_sse` · Files covered: `main.py`, `test_streaming.py`, `WALKTHROUGH.md`
Back to module: [09 API development](../README.md)

**Terms in this folder:** [Server-Sent Events (SSE)](#server-sent-events-sse-concept) · [text/event-stream](#texteventstream-concept) · [data: ... blank line framing](#data--blank-line-framing-concept) · [Time to first token (TTFT)](#time-to-first-token-ttft-concept) · [StreamingResponse](#streamingresponse-third-party-fastapi) · [Async generator](#async-generator-syntax) · [asyncio.to_thread](#asynciotothread-standard-library) · [asyncio.CancelledError (client disconnect)](#asynciocancellederror-client-disconnect-standard-library) · [Lifespan (load once)](#lifespan-load-once-concept) · [httpx.stream / iter_lines](#httpxstream--iter_lines-third-party-httpx) · [Proxy buffering](#proxy-buffering-concept) · [Token-by-token generator](#token-by-token-generator-concept)

---

## Server-Sent Events (SSE)  `(concept)`
**1. Definition:** A web standard where the server keeps one HTTP response open and pushes a sequence of text events to the client, one-way, using `Content-Type: text/event-stream`.
**2. In plain words:** Normal HTTP is "ask, wait, get the whole answer". SSE is "ask, then the server keeps talking until it is done". It is simpler than WebSockets because it is plain HTTP and only goes server to client, which is exactly what token streaming needs.
**3. Example to quote:**
> "I stream LLM tokens with SSE because it is plain HTTP, one-way, and browsers reconnect automatically."

**Seen in:** `/stream` in `main.py`

## text/event-stream  `(concept)`
**1. Definition:** The media type (Content-Type) that tells the client the body is an SSE stream.
**2. In plain words:** Without it a browser's `EventSource` refuses to connect, and proxies may treat the response as a normal file and buffer it. Set it through `media_type=` on `StreamingResponse`.
**3. Example to quote:**
> "With `media_type='text/plain'` my client said wrong MIME type and refused; `text/event-stream` fixed it."

**Seen in:** `/stream` and `/stream-wrong-media-type`

## data: ... blank line framing  `(concept)`
**1. Definition:** Each SSE event is one or more lines starting with `data:` and is ended by an empty line (`\n\n`).
**2. In plain words:** The blank line means "this message is finished, deliver it". If you send only one newline, the client keeps waiting for the rest of the message and dispatches nothing.
**3. Example to quote:**
> "I yielded `data: tok\n` with one newline and the client dispatched zero events; it must be `\n\n`."

**Seen in:** `sse_event()`, `parse_sse()`

## Time to first token (TTFT)  `(concept)`
**1. Definition:** The time from sending the request until the first token reaches the user.
**2. In plain words:** It is the delay the user feels. Streaming leaves total time about the same but cuts the wait for the first word from the full answer time to one token time.
**3. Example to quote:**
> "Blocking took 1.24 s before anything appeared; streaming showed the first token at 0.16 s, with the same total."

**Seen in:** `demo()` in `main.py`

## StreamingResponse  `(third-party: fastapi)`
**1. Definition:** A response class that takes an iterator or generator and sends each yielded chunk to the client as it is produced.
**2. In plain words:** A normal return builds the full body first. `StreamingResponse` sends headers right away and then writes chunks as your generator makes them.
**3. Example to quote:**
> "`StreamingResponse(event_generator(), media_type='text/event-stream')` sends each token as the generator yields it."

**Seen in:** `from fastapi.responses import StreamingResponse`

## Async generator  `(syntax)`
**1. Definition:** An `async def` function that contains `yield`; it produces values one by one and can `await` between them.
**2. In plain words:** Each `yield` hands one chunk to the framework to write; while the generator waits, the event loop serves other users.
**3. Example to quote:**
> "`async def event_generator(): ... yield sse_event(...)` becomes the response body, chunk by chunk."

**Seen in:** `event_generator()`

## asyncio.to_thread  `(standard library)`
**1. Definition:** `await asyncio.to_thread(fn, *args)` runs a blocking function in a worker thread and returns its result without blocking the event loop.
**2. In plain words:** The model's `next()` blocks while computing. Putting it in a thread keeps the server responsive to everyone else.
**3. Example to quote:**
> "I wrapped `next(tokens)` in `asyncio.to_thread` so a slow forward pass does not freeze other streams."

**Seen in:** `/stream`

## asyncio.CancelledError (client disconnect)  `(standard library)`
**1. Definition:** The exception raised inside a running task when it is cancelled; the server cancels the response generator when the client closes the connection.
**2. In plain words:** It is how your code learns the user clicked stop. Catch it to clean up or count it, and re-raise so the cancellation finishes.
**3. Example to quote:**
> "On disconnect the generator got `CancelledError`, the counter went 0 to 1, and generation stopped."

**Seen in:** `except asyncio.CancelledError` in `/stream`

## Lifespan (load once)  `(concept)`
**1. Definition:** FastAPI's startup and shutdown hook (`lifespan=`): code before `yield` runs once per worker process before the first request.
**2. In plain words:** Weights are loaded here and stored on `app.state`, so requests only use the model and never load it. Loading inside the handler would repeat the multi-second cost for every user.
**3. Example to quote:**
> "The distilgpt2 weights loaded once in 8.33 s at startup; every request after that pays nothing."

**Seen in:** `lifespan()`

## httpx.stream / iter_lines  `(third-party: httpx)`
**1. Definition:** `httpx.stream(method, url)` opens a response without downloading the body; `resp.iter_lines()` yields each line as soon as it arrives.
**2. In plain words:** This is how a Python client watches a stream live instead of waiting for the end, so it can print tokens with timestamps.
**3. Example to quote:**
> "`for line in resp.iter_lines()` prints tokens as they arrive, so I can measure TTFT."

**Seen in:** `demo()`

## Proxy buffering  `(concept)`
**1. Definition:** Some proxies (for example nginx) collect response bytes into a buffer before forwarding them.
**2. In plain words:** A buffering proxy turns your stream back into a blank screen followed by everything at once. Send `Cache-Control: no-cache` and `X-Accel-Buffering: no`, and disable buffering in the proxy config.
**3. Example to quote:**
> "Streaming worked locally but not behind nginx until I added `X-Accel-Buffering: no`."

**Seen in:** `headers=` in `/stream`

## Token-by-token generator  `(concept)`
**1. Definition:** A generator that runs one forward pass per iteration and yields the new token, usually reusing the KV cache.
**2. In plain words:** This is the generation loop from 04_llm_fundamentals step 08 written with `yield` instead of building a list, so the server can send each token the moment it exists.
**3. Example to quote:**
> "`DistilGPT2Model.stream` yields each decoded token; the API turns each yield into an SSE event."

**Seen in:** `DistilGPT2Model.stream`, `FakeTokenModel.stream`
