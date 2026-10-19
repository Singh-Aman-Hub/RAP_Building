"""pytest: the SSE framing and media type, with the fake model (no weights, no server process)."""
import json

from fastapi.testclient import TestClient

import main
from main import app, parse_sse


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


def test_missing_blank_line_dispatches_nothing(monkeypatch):
    monkeypatch.setattr(main, "TOKEN_DELAY", 0.0)
    with TestClient(app) as client:
        text = client.get("/stream-no-blank-line").text
    assert list(parse_sse(text.splitlines())) == []
