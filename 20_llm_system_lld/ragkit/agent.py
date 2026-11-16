"""agent.py -- a tool-calling agent runtime: registry, validated tool calls, step limit, loop guard (step 10, problem 6)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from .gateway import Gateway, LLMRequest, Message, Role, ToolCall, ToolSpec


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}

    def register(self, spec: ToolSpec, fn: Callable[..., Any]) -> None:
        self._tools[spec.name] = (spec, fn)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(s for s, _ in self._tools.values())

    def call(self, call: ToolCall) -> ToolResult:
        """Never raises: every failure becomes a ToolResult the MODEL can read and react to."""
        if call.name not in self._tools:
            return ToolResult(False, f"unknown tool {call.name!r}; available: {sorted(self._tools)}")
        spec, fn = self._tools[call.name]
        try:
            args = spec.validate_args(call.arguments)
        except ValidationError as exc:
            return ToolResult(False, f"invalid arguments: {exc.errors()[0]['msg']} at {exc.errors()[0]['loc']}")
        try:
            return ToolResult(True, str(fn(**args)))
        except Exception as exc:                                      # tool bug / external failure
            return ToolResult(False, f"tool error: {type(exc).__name__}: {exc}")


@dataclass
class AgentResult:
    text: str
    stopped: str                                                      # "final" | "max_steps" | "loop"
    steps: int
    transcript: list[str] = field(default_factory=list)


class AgentRuntime:
    def __init__(self, gateway: Gateway, tools: ToolRegistry, model: str = "default", max_steps: int = 5,
                 system: str = "You may call tools. Reply with the final answer when done."):
        self.gateway, self.tools, self.model, self.max_steps, self.system = gateway, tools, model, max_steps, system

    def run(self, user_text: str) -> AgentResult:
        messages = [Message(Role.USER, user_text)]
        seen: set[tuple[str, str]] = set()
        log: list[str] = []
        for step in range(1, self.max_steps + 1):
            resp = self.gateway.complete(LLMRequest(tuple(messages), self.model, self.system, tools=self.tools.specs()))
            if not resp.tool_calls:
                log.append(f"step {step}: final answer")
                return AgentResult(resp.text, "final", step, log)
            messages.append(Message(Role.ASSISTANT, resp.text or f"(calling {resp.tool_calls[0].name})"))
            for call in resp.tool_calls:
                sig = (call.name, json.dumps(call.arguments, sort_keys=True))
                if sig in seen:                                       # same tool + same args again: the model is stuck
                    log.append(f"step {step}: repeated call {call.name} -> stop")
                    return AgentResult("stopped: repeated identical tool call", "loop", step, log)
                seen.add(sig)
                result = self.tools.call(call)
                log.append(f"step {step}: {call.name}({call.arguments}) -> {'ok' if result.ok else 'ERROR'}: {result.content}")
                messages.append(Message(Role.TOOL, f"[tool_result {call.name}] {result.content}"))
        return AgentResult("stopped: step limit reached", "max_steps", self.max_steps, log)
