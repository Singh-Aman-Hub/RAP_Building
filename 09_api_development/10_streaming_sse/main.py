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

USE_REAL_MODEL = os.environ.get("STREAM_REAL_MODEL") == "1"
TOKEN_DELAY = 0.15            # seconds per fake token (a real model on CPU is in the same range)
STATE = {"disconnects": 0}    # lives in the server process; the client reads it via /stats


# ----------------------------- the model, loaded ONCE at startup -----------------------------
class FakeTokenModel:
    """Stands in for a real LLM: yields one 'token' at a time, slowly. Tests need no weights."""

    def stream(self, prompt: str, max_new_tokens: int):
        words = ("The castle gate opens and tokens walk out one at a time so the user can start reading "
                 "long before the model has finished").split()
        for i in range(max_new_tokens):
            time.sleep(TOKEN_DELAY)                       # pretend one forward pass takes this long
            yield " " + words[i % len(words)]


class DistilGPT2Model:
    """Greedy token-by-token loop with a KV cache (module 04 step 08, written as a generator)."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("distilgpt2")
        self.lm = AutoModelForCausalLM.from_pretrained("distilgpt2").eval()

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup, once per worker process: NEVER load weights inside a request handler.
    t0 = time.perf_counter()
    app.state.model = DistilGPT2Model() if USE_REAL_MODEL else FakeTokenModel()
    print(f"[pid {os.getpid()}] model loaded once in {time.perf_counter() - t0:.2f}s "
          f"({'distilgpt2' if USE_REAL_MODEL else 'fake'})", flush=True)
    yield


app = FastAPI(title="Streaming API", lifespan=lifespan)


# ----------------------------- the endpoints -----------------------------
@app.get("/blocking")
def blocking(prompt: str = "hi", max_new_tokens: int = 8):
    """The pain point: the whole answer, only when ALL tokens exist."""
    tokens = list(app.state.model.stream(prompt, max_new_tokens))
    return JSONResponse({"text": "".join(tokens)})


def sse_event(payload: str) -> str:
    # SSE framing: a field line "data: ...", then a BLANK LINE (\n\n) that means "event finished".
    return f"data: {payload}\n\n"


@app.get("/stream")
async def stream(prompt: str = "hi", max_new_tokens: int = 8):
    model = app.state.model

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

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/stream-no-blank-line")
async def stream_no_blank_line(max_new_tokens: int = 4):
    async def gen():
        for i in range(max_new_tokens):
            await asyncio.sleep(TOKEN_DELAY)
            yield f"data: tok{i}\n"                          # BUG: only ONE newline -> event never ends
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/stream-wrong-media-type")
async def stream_wrong_media_type(max_new_tokens: int = 4):
    async def gen():
        for i in range(max_new_tokens):
            await asyncio.sleep(TOKEN_DELAY)
            yield f"data: tok{i}\n\n"
    return StreamingResponse(gen(), media_type="text/plain")  # BUG: browsers/EventSource reject this


@app.get("/stats")
def stats():
    return STATE


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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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

        print("\n== 3. BREAK IT: forget the blank line ==")
        with httpx.stream("GET", f"{base}/stream-no-blank-line", timeout=60) as resp:
            raw = list(resp.iter_lines())
            print("   raw lines on the wire :", raw)
        with httpx.stream("GET", f"{base}/stream-no-blank-line", timeout=60) as resp:
            events = list(parse_sse(resp.iter_lines()))
        print("   events a real SSE client dispatches:", events, " <- nothing, the event never 'ends'")

        print("\n== 4. BREAK IT: wrong media type ==")
        with httpx.stream("GET", f"{base}/stream-wrong-media-type", timeout=60) as resp:
            ctype = resp.headers["content-type"]
            ok = ctype.startswith("text/event-stream")
            print(f"   content-type: {ctype}  -> EventSource would {'connect' if ok else 'REFUSE: wrong MIME type'}")

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
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    demo(real="--real" in sys.argv)
    # TRY IT: change TOKEN_DELAY to 0.4 and compare blocking vs streaming totals.
    # TRY IT: python main.py --real to stream a real distilgpt2 answer.
    # BREAK IT: change media_type to "text/plain" in /stream and re-run section 2.
