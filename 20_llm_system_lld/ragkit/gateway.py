"""gateway.py -- one door to many LLM providers: unified types, streaming, structured output, tools, cost (step 08)."""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Iterator, Protocol, Sequence, Type

from pydantic import BaseModel, ValidationError

from .adapters import _use_module14
from .ids import new_id
from .query import WordTokenCounter
from .resilience import FallbackChain, PermanentError


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"                                              # result of a tool call, fed back to the model


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call: name + description + JSON Schema of its arguments."""
    name: str
    description: str
    parameters: dict
    model: Type[BaseModel] | None = field(default=None, compare=False, hash=False)

    @classmethod
    def from_model(cls, name: str, description: str, model: Type[BaseModel]) -> "ToolSpec":
        return cls(name, description, model.model_json_schema(), model)

    def to_anthropic(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}

    def to_openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}

    def validate_args(self, args: dict) -> dict:
        if self.model is None:
            return args
        return self.model.model_validate(args).model_dump()


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[Message, ...]
    model: str = "default"                                     # a route ALIAS, not a vendor model id
    system: str = ""
    max_tokens: int = 300
    temperature: float = 0.0
    tools: tuple[ToolSpec, ...] = ()
    request_id: str = field(default_factory=lambda: new_id("req"))   # doubles as an idempotency key

    @classmethod
    def user(cls, text: str, **kw) -> "LLMRequest":
        return cls((Message(Role.USER, text),), **kw)


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    estimated: bool = True                                     # True when we counted words, not the provider's tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    usage: Usage
    provider: str
    model: str
    finish_reason: str = "stop"
    tool_calls: tuple[ToolCall, ...] = ()
    latency_s: float = 0.0


@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: str                                                  # "delta" (a piece of text) or "final" (the whole response)
    text: str = ""
    response: LLMResponse | None = None


class Provider(Protocol):
    name: str
    def complete(self, req: LLMRequest) -> LLMResponse: ...
    def stream(self, req: LLMRequest) -> Iterator[str]: ...


_COUNT = WordTokenCounter()


class SharedClientProvider:
    """Adapter over shared/llm_client.generate: backend 'mock' | 'local' | 'anthropic'."""

    def __init__(self, backend: str = "mock", mock=None, name: str | None = None):
        _use_module14()
        import llm_client
        self._llm, self.backend, self.mock = llm_client, backend, mock
        self.name = name or backend

    def complete(self, req: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()
        msgs = [{"role": "assistant" if m.role is Role.ASSISTANT else "user", "content": m.content}
                for m in req.messages if m.role is not Role.SYSTEM]       # shared client knows only user/assistant
        text = self._llm.generate(msgs, system=req.system or None, max_tokens=req.max_tokens,
                                  temperature=req.temperature, backend=self.backend, mock=self.mock)
        usage = Usage(_COUNT.count(req.system + " ".join(m.content for m in req.messages)), _COUNT.count(text))
        return LLMResponse(text, usage, self.name, req.model, "stop", (), time.perf_counter() - t0)

    def stream(self, req: LLMRequest) -> Iterator[str]:
        """Simulated streaming: generate fully, then yield word by word. (A real provider yields tokens as decoded.)"""
        for w in re.findall(r"\S+\s*", self.complete(req).text):
            yield w

    async def astream(self, req: LLMRequest) -> AsyncIterator[str]:
        for w in self.stream(req):
            await asyncio.sleep(0)                             # hand control back to the event loop between pieces
            yield w


class ScriptedProvider:
    """Deterministic provider for tests: replies come from a list (or callable); optionally raises."""

    def __init__(self, name: str, replies: Sequence[Any], model: str = "scripted"):
        self.name, self.replies, self.calls = name, list(replies), 0

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls += 1
        r = self.replies.pop(0) if self.replies else "default reply"
        if isinstance(r, BaseException) or (isinstance(r, type) and issubclass(r, BaseException)):
            raise r() if isinstance(r, type) else r
        if isinstance(r, tuple):                               # ("text", [ToolCall...]) to script a tool call
            text, calls = r
        else:
            text, calls = r, ()
        usage = Usage(_COUNT.count(" ".join(m.content for m in req.messages)), _COUNT.count(text))
        return LLMResponse(text, usage, self.name, req.model, "tool_use" if calls else "stop", tuple(calls))

    def stream(self, req: LLMRequest) -> Iterator[str]:
        yield from re.findall(r"\S+\s*", self.complete(req).text)


# ------------------------------------------------------------------ cost accounting
PRICES_PER_MTOK = {                                            # (input, output) USD per 1M tokens -- ILLUSTRATIVE, not real prices
    "mock": (0.0, 0.0),
    "small": (0.25, 1.25),
    "large": (3.0, 15.0),
}


class BudgetExceeded(PermanentError):
    pass


class CostMeter:
    def __init__(self, prices: dict[str, tuple[float, float]] | None = None, budget_usd: float | None = None):
        self.prices, self.budget = prices or PRICES_PER_MTOK, budget_usd
        self.spent = 0.0
        self.input_tokens = self.output_tokens = self.requests = 0
        self.by_model: dict[str, float] = {}

    def cost(self, model: str, usage: Usage) -> float:
        pin, pout = self.prices.get(model, (0.0, 0.0))
        return (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6

    def check(self) -> None:
        if self.budget is not None and self.spent >= self.budget:
            raise BudgetExceeded(f"budget ${self.budget:.4f} exhausted (spent ${self.spent:.4f})")

    def record(self, resp: LLMResponse) -> float:
        c = self.cost(resp.model, resp.usage)
        self.spent += c
        self.requests += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        self.by_model[resp.model] = self.by_model.get(resp.model, 0.0) + c
        return c


# ------------------------------------------------------------------ the gateway
class Gateway:
    """routes: alias -> ordered providers (primary first). The caller names an alias; the gateway picks, falls back, meters."""

    def __init__(self, routes: dict[str, list], meter: CostMeter | None = None, bus=None):
        self.routes, self.meter, self.bus = routes, meter or CostMeter(), bus or (lambda n, p: None)
        self.last_chain: FallbackChain | None = None

    def complete(self, req: LLMRequest) -> LLMResponse:
        self.meter.check()                                     # refuse BEFORE spending
        chain = FallbackChain([(p.name, p.complete) for p in self.routes[req.model]])
        self.last_chain = chain
        resp = chain(req)
        cost = self.meter.record(resp)
        self.bus("llm_call", {"provider": resp.provider, "model": resp.model, "in": resp.usage.input_tokens,
                              "out": resp.usage.output_tokens, "cost": cost, "fell_back": bool(chain.failures)})
        return resp

    def stream(self, req: LLMRequest) -> Iterator[StreamEvent]:
        self.meter.check()
        provider = self.routes[req.model][0]
        parts: list[str] = []
        for piece in provider.stream(req):
            parts.append(piece)
            yield StreamEvent("delta", piece)
        text = "".join(parts)
        usage = Usage(_COUNT.count(" ".join(m.content for m in req.messages)), _COUNT.count(text))
        resp = LLMResponse(text, usage, provider.name, req.model)
        self.meter.record(resp)
        yield StreamEvent("final", "", resp)

    def complete_structured(self, req: LLMRequest, schema: Type[BaseModel], max_repairs: int = 1) -> tuple[BaseModel, int]:
        """Ask for JSON, validate with pydantic; on failure show the model its own error and retry (repair loop).
        Returns (parsed_object, repairs_used)."""
        messages = list(req.messages)
        last_err = ""
        for attempt in range(max_repairs + 1):
            resp = self.complete(LLMRequest(tuple(messages), req.model, req.system, req.max_tokens, req.temperature))
            try:
                return schema.model_validate(extract_json(resp.text)), attempt
            except (ValueError, ValidationError) as exc:       # ValidationError is a ValueError subclass in pydantic v2
                last_err = str(exc)
                messages += [Message(Role.ASSISTANT, resp.text),
                             Message(Role.USER, f"Your JSON was invalid: {last_err[:300]}\nReturn ONLY corrected JSON.")]
        raise PermanentError(f"no valid structured output after {max_repairs} repair(s): {last_err[:200]}")


def extract_json(text: str) -> dict:
    """First '{' to last '}' -- models often wrap JSON in chatter or code fences."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:80]!r}")
    return json.loads(m.group(0))


class GatewayGenerator:
    """Adapter: Gateway -> the Generator Protocol, so RagService can run on top of the gateway unchanged."""

    def __init__(self, gateway: Gateway, model: str = "default", max_tokens: int = 300):
        self.gateway, self.model, self.max_tokens = gateway, model, max_tokens

    def generate(self, system: str, user: str) -> str:
        return self.gateway.complete(LLMRequest.user(user, model=self.model, system=system, max_tokens=self.max_tokens)).text
