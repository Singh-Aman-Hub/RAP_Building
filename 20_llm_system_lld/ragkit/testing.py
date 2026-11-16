"""testing.py -- fakes used by tests AND demos: a controllable clock and providers that fail on cue (step 06)."""
from __future__ import annotations


class FakeClock:
    """Time that only moves when you say so: sleep(2) advances now() by 2 instantly. Makes backoff tests instant + exact."""

    def __init__(self, start: float = 1000.0):
        self.t, self.sleeps = start, []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FlakyGenerator:
    """Generator that follows a script: each item is 'ok', an exception instance/class, or a string reply."""

    def __init__(self, script, reply: str = "ok-reply"):
        self.script, self.reply, self.calls = list(script), reply, 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        step = self.script.pop(0) if self.script else "ok"
        if step == "ok":
            return self.reply
        if isinstance(step, str):
            return step
        raise step() if isinstance(step, type) else step


class FakeJsonProvider:
    """Gateway Provider that answers like a JSON-mode LLM, built on the offline ExtractiveGenerator (step 11).
    `modes` is a script, one entry per call: good | bad_json | bad_schema | ungrounded | email_leak | declines."""

    def __init__(self, name: str, modes=(), model: str = "fake"):
        self.name, self.modes, self.calls = name, list(modes), 0

    def complete(self, req):
        import json
        from .gateway import LLMResponse, Usage
        from .query import ExtractiveGenerator, WordTokenCounter
        self.calls += 1
        mode = self.modes.pop(0) if self.modes else "good"
        user = req.messages[-1].content if "<documents>" in req.messages[-1].content else next(
            m.content for m in req.messages if "<documents>" in m.content)
        text = ExtractiveGenerator().generate(req.system, user)               # "<sentence> [n]" or "I don't know."
        if mode == "bad_json":
            out = "Sure! Here is the answer: " + text
        elif mode == "bad_schema":
            out = json.dumps({"answer": text, "citations": [1], "confidence": 7})
        elif mode == "ungrounded":
            out = json.dumps({"answer": "The moon is made of cheese [1].", "citations": [1], "confidence": 0.9})
        elif mode == "email_leak":
            out = json.dumps({"answer": text + " Mail agent.smith@example.com", "citations": [1], "confidence": 0.8})
        elif mode == "declines":
            out = json.dumps({"answer": "I don't know.", "citations": [], "confidence": 0.9})
        else:
            out = json.dumps({"answer": text, "citations": [1], "confidence": 0.85})
        count = WordTokenCounter().count
        return LLMResponse(out, Usage(count(req.system + " ".join(m.content for m in req.messages)), count(out)),
                           self.name, req.model)

    def stream(self, req):
        yield self.complete(req).text
