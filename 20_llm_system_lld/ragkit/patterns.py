"""patterns.py -- three small, generic design patterns (step 05): Registry/Factory, Observer, Chain of Responsibility."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from .domain import Answer

T = TypeVar("T")


class UnknownPluginError(KeyError):
    pass


# ------------------------------------------------------------------ Registry + Factory
class Registry(Generic[T]):
    """name -> factory function. New implementations register themselves; callers only know the NAME (from config)."""

    def __init__(self, kind: str):
        self.kind, self._factories = kind, {}

    def register(self, name: str) -> Callable:
        def deco(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ValueError(f"{self.kind} {name!r} already registered")
            self._factories[name] = factory
            return factory
        return deco

    def create(self, name: str, **kwargs: Any) -> T:
        try:
            factory = self._factories[name]
        except KeyError:
            raise UnknownPluginError(f"unknown {self.kind} {name!r}; available: {sorted(self._factories)}") from None
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)


# ------------------------------------------------------------------ Observer
class EventBus:
    """Publish/subscribe. Components emit (name, payload); subscribers react. A broken subscriber never breaks the emitter."""

    def __init__(self):
        self._subs: dict[str, list[Callable[[str, dict], None]]] = defaultdict(list)
        self.errors: list[str] = []

    def subscribe(self, name: str, fn: Callable[[str, dict], None]) -> None:
        self._subs[name].append(fn)                           # name "*" = every event

    def emit(self, name: str, payload: dict | None = None) -> None:
        for fn in self._subs.get(name, []) + self._subs.get("*", []):
            try:
                fn(name, payload or {})
            except Exception as exc:                           # isolate subscribers from the main flow
                self.errors.append(f"{fn!r} on {name}: {exc}")

    __call__ = emit                                           # a bus can be passed anywhere an on_event callable is expected


# ------------------------------------------------------------------ Chain of Responsibility
@dataclass
class QueryContext:
    question: str
    blocked: str | None = None
    notes: list[str] = field(default_factory=list)


class Handler(ABC):
    """Each handler does its job, then passes the context to the next one -- or stops the chain."""
    _next: "Handler | None" = None

    def set_next(self, nxt: "Handler") -> "Handler":
        self._next = nxt
        return nxt                                            # returning nxt allows a.set_next(b).set_next(c)

    def handle(self, ctx: QueryContext) -> QueryContext:
        ctx = self.process(ctx)
        if ctx.blocked or self._next is None:
            return ctx
        return self._next.handle(ctx)

    @abstractmethod
    def process(self, ctx: QueryContext) -> QueryContext: ...


class Normalize(Handler):
    def process(self, ctx):
        ctx.question = " ".join(ctx.question.split())
        return ctx


class RedactPII(Handler):
    def process(self, ctx):
        q = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", ctx.question)
        q = re.sub(r"\b\d{10}\b", "[PHONE]", q)
        if q != ctx.question:
            ctx.notes.append("pii_redacted")
        ctx.question = q
        return ctx


class MaxLength(Handler):
    def __init__(self, limit: int = 500):
        self.limit = limit

    def process(self, ctx):
        if len(ctx.question) > self.limit:
            ctx.blocked = f"question_too_long({len(ctx.question)}>{self.limit})"
        return ctx


class InjectionGuard(Handler):
    def process(self, ctx):
        if re.search(r"ignore (all |any )?(previous|prior) instructions", ctx.question, re.I):
            ctx.blocked = "prompt_injection_pattern"
        return ctx


def build_chain(*handlers: Handler) -> Handler:
    for a, b in zip(handlers, handlers[1:]):
        a.set_next(b)
    return handlers[0]


class GuardedAnswerer:
    """Decorator: run the chain first; blocked questions never reach the model."""

    def __init__(self, inner, chain: Handler):
        self.inner, self.chain = inner, chain

    def answer(self, question: str) -> Answer:
        ctx = self.chain.handle(QueryContext(question))
        if ctx.blocked:
            return Answer(f"Request blocked ({ctx.blocked}).", refused=True, refusal_reason=ctx.blocked)
        return self.inner.answer(ctx.question)
