# Step 08 — The LLM gateway: provider abstraction, streaming, structured output, tools and cost

## The story so far
[Step 07](../07_caching/WALKTHROUGH.md) cached answers. Until now the model was `Generator.generate(system, user) -> str`. Module [08 step 01](../../08_agent_harness/01_llm_api_call/WALKTHROUGH.md) and `shared/llm_client.py` gave us a text-in/text-out client with three backends (mock, local, Anthropic).

## The pain point
A string is not enough. You need to know how many tokens a call used (money), why it stopped, whether the model wants to call a tool, and to show text as it is produced. Each vendor has different request shapes (`system` as a field or as a message), tool formats and error types; if they leak into application code, switching a provider means touching everything. And a model that must return JSON will sometimes return JSON with chatter around it, or with a field of the wrong type.

## The spark
A **gateway**: one set of immutable request/response types, a tiny `Provider` interface, routes from a model **alias** to an ordered list of providers (so fallback is configuration), a **cost meter** with a budget guard, **streaming** as a generator (sync) or async iterator, a **repair loop** for structured output validated by pydantic, and a **tool spec** that renders to each vendor's format. An Adapter turns the gateway back into the `Generator` that `RagService` expects.

## Goal
Read `ragkit/gateway.py` block by block; run a completion with fallback, a stream (sync and async), a structured output that needs one repair, a tool schema in two vendor formats, cost accounting, and the whole RAG service running on top of the gateway.

## The theory (read this before the interview)
### Unified types (what crosses the seam)
| Type | Fields | Notes |
|---|---|---|
| `Message` | `role` (system/user/assistant/tool), `content` | `Role` is an Enum |
| `ToolSpec` | `name`, `description`, JSON-Schema `parameters` | built from a pydantic model with `model_json_schema()` |
| `ToolCall` | `id`, `name`, `arguments: dict` | what the model asks us to run |
| `LLMRequest` | messages, **model alias**, system, max_tokens, temperature, tools, `request_id` | frozen; `request_id` doubles as an idempotency key |
| `Usage` | input_tokens, output_tokens, `estimated` | `estimated=True` when we counted words, not the provider's tokens |
| `LLMResponse` | text, usage, provider, model, finish_reason, tool_calls, latency | everything a caller or a meter needs |
| `StreamEvent` | `kind` = `delta` or `final`, text, response | the final event carries usage |
The **alias** (`small`, `large`) is what application code names; the gateway's route table maps it to concrete providers, so a model upgrade is a config change.

### Providers behind one interface
`Provider` = `name`, `complete(req) -> LLMResponse`, `stream(req) -> Iterator[str]`. `SharedClientProvider` adapts `shared/llm_client.generate` (mock / local Qwen / Anthropic). `ScriptedProvider` is a test double with scripted replies, errors and tool calls. Real vendor SDKs would be two more adapters; none of them leaks into `RagService`.

### Routing and fallback
`Gateway.routes = {alias: [provider1, provider2, ...]}`. `complete` builds a `FallbackChain` (step 06) over the providers' `complete`; only `TransientError`/`CircuitOpenError` move on. Add a per-provider `CircuitBreaker` and `RetryingGenerator`-style wrappers around each `complete` for the full stack. Streaming uses the first provider only: **you cannot fall back mid-stream** without duplicating or losing text (decide up front, or buffer the first chunk).

### Streaming
- **Sync**: a generator that `yield`s pieces; `Gateway.stream` re-yields `StreamEvent('delta', piece)` and ends with `StreamEvent('final', response)` so usage is known only at the end.
- **Async**: `async def astream(...)` with `async for` consumption lets one process serve many streams; `await asyncio.sleep(0)` yields control to the event loop. FastAPI's `StreamingResponse` / SSE endpoints consume exactly this (module 09).
- Hosted APIs stream **server-sent events** (for the Claude API: events such as message start, content block deltas, message delta with the stop reason and usage, message stop). An error can arrive **after** a 200 response, mid-stream, so streaming code needs its own error path (see the API errors page in References).
- The mock "streams" by splitting a finished reply into words; a real stream yields tokens as they are decoded. UX metric: **time to first token**; throughput: tokens/second.

### Structured output with a repair loop
1. Ask for JSON (system prompt, or the provider's native structured-output / strict tool feature when available).
2. `extract_json`: first `{` to last `}` (models add chatter or code fences).
3. `schema.model_validate(...)` with pydantic v2: types, ranges (`confidence <= 1`), required fields.
4. On failure: append the model's own reply and a user message containing the validation error ("Your JSON was invalid: ... Return ONLY corrected JSON") and retry, up to `max_repairs`; then raise `PermanentError`.
Native constrained decoding (the provider guarantees the schema) is better when available, but validation remains useful as defence in depth. Every repair is a paid call: log the repair rate.

### Tool-call schema
A tool is a function description the model may call. The wire formats differ: Anthropic `{"name", "description", "input_schema"}`; OpenAI chat-completions `{"type": "function", "function": {"name", "description", "parameters"}}`. One pydantic model gives the JSON Schema and **validates the arguments the model sends back** (coercing `'5'` to `5`, rejecting a missing field). Step 10's agent runtime executes calls and feeds results back as `tool` messages.

### Cost and token accounting
`cost = (input_tokens * price_in + output_tokens * price_out) / 1e6`. Prices come from a table (the numbers in this module are **illustrative placeholders**; use the provider's pricing page). `CostMeter` accumulates spend per model and enforces a **budget**: before a call, if `spent >= budget` raise `BudgetExceeded` (a `PermanentError`, so retries and fallbacks do not mask it). Output tokens usually cost several times input tokens; cached input tokens cost less (step 07). Always prefer the provider's reported usage over your own estimate; the demo estimates because the mock has no tokenizer.

## Same thing, different names
| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| LLM gateway | LLM proxy, router, model router, AI gateway | Mostly | Gateway = one entry point with policy (budget, auth, logging); router = the part that picks the provider; proxy = network-level pass-through. |
| Provider | backend, vendor, driver, connector | Mostly | The thing that actually runs a model; our `Provider` is the interface, vendors are implementations. |
| Streaming | SSE, token streaming, incremental decoding | Close | SSE is the HTTP mechanism; streaming is the behaviour of yielding partial output. |
| Structured output | JSON mode, schema-constrained generation, function calling for extraction | No | JSON mode only promises valid JSON; schema-constrained promises your schema; validation + repair works with any model. |
| Tool call | function call, tool use, action | Same | Vendors differ in field names; the idea (model requests, app executes) is identical. |
| Token accounting | usage metering, cost tracking, FinOps for LLMs | Close | Counting tokens is the measurement; cost tracking multiplies by a price table; budgets add enforcement. |
| Model alias | logical model name, route key, deployment name | Close | A stable name for application code that the gateway maps to a concrete model. |

## Run it
```bash
# from this step folder
HF_HUB_OFFLINE=1 python main.py
# or from the repo root
HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/08_llm_gateway_and_streaming/main.py
```


## Code, step by step
Two files: `main.py` and `ragkit/gateway.py`. Every block is listed.

#### Walking `main.py`

### 1. Header docstring  (`main.py` lines 1–14)
```python
"""
GOAL:  One door to many LLM providers: unified request/response types, routing + fallback, streaming (sync generator
       and async iterator), structured output with a pydantic repair loop, a tool-call schema, and cost accounting.
       Then the gateway is plugged UNDER the RAG service from step 04 without changing the service.

RUN (from this step folder):  python main.py                       (mock backend; LLM_BACKEND=local / anthropic also work)
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/08_llm_gateway_and_streaming/main.py
EXPECTED OUTPUT (shortened):
    complete: provider=mock-b (fell back from mock-a) tokens in/out=.. cost=$...
    stream: ['Refunds ', 'take ', ...] then final usage
    structured: repaired after 1 attempt -> StructuredAnswer(...)
    RAG on top of the gateway: ... | total spend $0.00..
NEEDS: ragkit + shared/llm_client.py. Default backend is the scripted mock: offline, deterministic.
"""
```
- **In simple words:** What the file shows.
- **Exactly:** GOAL/RUN/EXPECTED/NEEDS; default backend is the scripted mock.
- **Why it is here:** Orientation.

### 2. Imports and the shared client  (`main.py` lines 15–36)
```python
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import BaseModel

from ragkit.adapters import _use_module14
from ragkit.factory import build_system
from ragkit.gateway import (CostMeter, Gateway, GatewayGenerator, LLMRequest, ScriptedProvider, SharedClientProvider,
                            ToolCall, ToolSpec)
from ragkit.ingest import DirectoryLoader
from ragkit.resilience import TransientError
from ragkit.schemas import StructuredAnswer

_use_module14()
from llm_client import MockLLM                                      # shared/llm_client.py (module 07/08's tiny client)

KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"
BACKEND = os.getenv("LLM_BACKEND", "mock")
```
- **In simple words:** Everything from the gateway, pydantic, and `MockLLM` from `shared/llm_client.py`.
- **Exactly:** `_use_module14()` adds `shared/` to the path so `from llm_client import MockLLM` works; `BACKEND = os.getenv('LLM_BACKEND', 'mock')`.
- **Why it is here:** The same file can run against `local` or `anthropic` by an environment variable.

### 3. Build a gateway with a failing primary  (`main.py` lines 38–44)
```python
# ---- 1. Providers behind one interface; a route is an ALIAS -> ordered list of providers ----------------------------
mock = MockLLM(rules=[(r"refund", "Refunds take 9 days after delivery [1]."), (r"", "I am the default mock reply.")])
primary = ScriptedProvider("mock-a", [TransientError("503 overloaded")])        # fails once, then replies
backup = SharedClientProvider(BACKEND, mock=mock, name=f"{BACKEND}-b")
prices = {"small": (0.25, 1.25), "large": (3.0, 15.0)}                           # ILLUSTRATIVE USD per 1M tokens
events = []
gw = Gateway({"small": [primary, backup]}, CostMeter(prices, budget_usd=0.05), bus=lambda n, p: events.append(p))
```
- **In simple words:** A route `small` -> [scripted provider that fails once, shared-client provider].
- **Exactly:** `MockLLM(rules=[(r'refund', 'Refunds take 9 days after delivery [1].'), (r'', ...)])`; `primary = ScriptedProvider('mock-a', [TransientError('503 overloaded')])`; `CostMeter(prices, budget_usd=0.05)`; `bus=lambda n, p: events.append(p)`.
- **Why it is here:** Fallback, metering and events from configuration alone.

### 4. Complete with fallback and accounting  (`main.py` lines 46–49)
```python
resp = gw.complete(LLMRequest.user("How many days for a refund?", model="small", system="Be brief."))
print(f"complete: provider={resp.provider} text={resp.text!r}")
print(f"          fell back: {gw.last_chain.failures} | tokens in/out={resp.usage.input_tokens}/{resp.usage.output_tokens} "
      f"(estimated={resp.usage.estimated}) | cost=${gw.meter.spent:.8f} | latency={resp.latency_s * 1000:.2f} ms")
```
- **In simple words:** One call, two providers, and a receipt.
- **Exactly:** Text from `mock-b`; `fell back: [('mock-a', 'TransientError')]`; tokens in/out `10/10` (`estimated=True`); cost `$0.00001500` = (10 * 0.25 + 10 * 1.25) / 1e6; latency 0.16 ms.
- **Why it is here:** The caller sees one response; the gateway hides the failure but records it.

### 5. Stream pieces, then a final event  (`main.py` lines 51–58)
```python
# ---- 2. Streaming: a generator of deltas, then one final event with usage -----------------------------------------------
pieces = []
gw_s = Gateway({"small": [backup]}, CostMeter(prices))              # streaming uses ONE provider: no mid-stream fallback
for ev in gw_s.stream(LLMRequest.user("How many days for a refund?", model="small")):
    if ev.kind == "delta":
        pieces.append(ev.text)
    else:
        print(f"stream  : {pieces} then final usage={ev.response.usage}")
```
- **In simple words:** Print each word as it arrives, then the usage.
- **Exactly:** Pieces `['Refunds ', 'take ', '9 ', 'days ', 'after ', 'delivery ', '[1].']`, final `Usage(input_tokens=8, output_tokens=10, estimated=True)`. A separate `Gateway({'small': [backup]})` is used because streaming only talks to the first provider.
- **Why it is here:** Usage is known only after the last delta.

### 6. Async iteration  (`main.py` lines 60–62)
```python
async def consume():
    return [p async for p in backup.astream(LLMRequest.user("refund?", model="small"))]
print("async   :", asyncio.run(consume()), "(async iterator: await between pieces keeps the event loop free)")
```
- **In simple words:** The same stream consumed with `async for`.
- **Exactly:** `[p async for p in backup.astream(...)]` run by `asyncio.run`; output is the same seven words.
- **Why it is here:** Servers handle many streams concurrently on one thread.

### 7. A repair loop with pydantic  (`main.py` lines 64–69)
```python
# ---- 3. Structured output: validate with pydantic, repair loop on failure ------------------------------------------------
reply_bad = '{"answer": "9 days", "citations": [1], "confidence": 7}'                       # confidence must be <= 1
reply_good = 'Here you go: {"answer": "9 days", "citations": [1], "confidence": 0.9}'       # chatter around the JSON
gw2 = Gateway({"small": [ScriptedProvider("scripted", [reply_bad, reply_good])]})
obj, repairs = gw2.complete_structured(LLMRequest.user("answer as JSON", model="small"), StructuredAnswer)
print(f"structured: repairs used={repairs} -> {obj!r}")
```
- **In simple words:** First reply is invalid (`confidence: 7`), second is valid but wrapped in chatter.
- **Exactly:** `complete_structured(..., StructuredAnswer)` returns `(StructuredAnswer(answer='9 days', citations=[1], confidence=0.9), repairs=1)`.
- **Why it is here:** Shows validation, JSON extraction from chatter and one repair.

### 8. One tool spec, two vendor formats  (`main.py` lines 71–80)
```python
# ---- 4. Tool-call schema: one spec, two vendor formats ---------------------------------------------------------------------------
class SearchArgs(BaseModel):
    query: str
    top_k: int = 3


tool = ToolSpec.from_model("search_docs", "Search the knowledge base", SearchArgs)
print("anthropic format:", json.dumps(tool.to_anthropic())[:150], "...")
print("openai format   :", json.dumps(tool.to_openai())[:150], "...")
print("validate args   :", tool.validate_args({"query": "refund", "top_k": "5"}), "(top_k '5' coerced to int)")
```
- **In simple words:** Describe a search tool and print both renderings.
- **Exactly:** `ToolSpec.from_model('search_docs', ..., SearchArgs)`; anthropic: `{'name','description','input_schema': {...}}`; openai: `{'type':'function','function':{...,'parameters': {...}}}`; `validate_args({'query':'refund','top_k':'5'})` -> `{'query': 'refund', 'top_k': 5}`.
- **Why it is here:** The schema comes from the pydantic model, so docs, validation and the model's view cannot drift.

### 9. A reply that asks for a tool  (`main.py` lines 81–82)
```python
tc = ScriptedProvider("t", [("", [ToolCall("call_1", "search_docs", {"query": "refund"})])]).complete(LLMRequest.user("x"))
print("tool-call reply : finish_reason =", tc.finish_reason, "| calls =", tc.tool_calls)
```
- **In simple words:** The scripted provider returns a tool call instead of text.
- **Exactly:** `finish_reason = 'tool_use'`, `tool_calls = (ToolCall(id='call_1', name='search_docs', arguments={'query': 'refund'}),)`.
- **Why it is here:** The shape the agent runtime (step 10) consumes.

### 10. RAG on top of the gateway  (`main.py` lines 84–92)
```python
# ---- 5. The gateway under the RAG service: Adapter makes it a Generator ------------------------------------------------------------
system = build_system({}, DirectoryLoader(KB))
system.pipeline.run()
gw3 = Gateway({"small": [SharedClientProvider(BACKEND, mock=MockLLM(rules=[(r"", "Refunds are issued within 9 days of delivery. [1]")]))]},
              CostMeter(prices), bus=lambda n, p: events.append(p))
system.service.generator = GatewayGenerator(gw3, model="small")
a = system.service.answer("How many days do I have to get a refund?")
print(f"RAG on gateway: {a.text} | citations ok={a.meta['citations_ok']} | gateway requests={gw3.meter.requests} spend=${gw3.meter.spent:.8f}")
print("events seen by the bus:", [(e['provider'], e['fell_back']) for e in events])
```
- **In simple words:** Replace the service's generator with `GatewayGenerator(gw3, model='small')`.
- **Exactly:** Answer `Refunds are issued within 9 days of delivery. [1]`, `citations ok=True`, `gateway requests=1`, `spend=$0.00007050`; the event bus saw `[('mock-b', True), ('mock', False)]` (provider, fell_back).
- **Why it is here:** `RagService` did not change: the Adapter made the gateway a Generator.

### 11. Exercises  (`main.py` lines 94–96)
```python
# TRY IT: set the budget to 0.0000001 on gw and call complete() twice: read the BudgetExceeded error.
# TRY IT: LLM_BACKEND=local python main.py (needs the cached Qwen model); only the SharedClientProvider changes.
# BREAK IT: make complete_structured(max_repairs=0) with the bad reply: PermanentError after one call.
```
- **In simple words:** Things to try.
- **Exactly:** Comment lines.
- **Why it is here:** See exercises.md.


#### Walking `ragkit/gateway.py`

### 12. gateway.py header and imports  (`ragkit/gateway.py` lines 1–17)
```python
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
```
- **In simple words:** Everything for talking to providers.
- **Exactly:** Imports `asyncio`, `json`, `re`, pydantic's `BaseModel`/`ValidationError`, `FallbackChain`/`PermanentError` from step 06 and `WordTokenCounter` from step 04.
- **Why it is here:** The gateway reuses earlier steps instead of reimplementing them.

### 13. Role enum  (`ragkit/gateway.py` lines 20–24)
```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"                                              # result of a tool call, fed back to the model
```
- **In simple words:** Who said it.
- **Exactly:** `SYSTEM`, `USER`, `ASSISTANT`, `TOOL` as a `str` Enum.
- **Why it is here:** TOOL carries tool results back to the model.

### 14. Message  (`ragkit/gateway.py` lines 27–30)
```python
@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str
```
- **In simple words:** One chat message.
- **Exactly:** Frozen, slotted dataclass `(role, content)`.
- **Why it is here:** Immutable messages can be shared between a request and its retries.

### 15. ToolSpec: fields  (`ragkit/gateway.py` lines 33–39)
```python
@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool the model may call: name + description + JSON Schema of its arguments."""
    name: str
    description: str
    parameters: dict
    model: Type[BaseModel] | None = field(default=None, compare=False, hash=False)
```
- **In simple words:** What the model is told about a tool.
- **Exactly:** `name`, `description`, `parameters` (JSON Schema dict), and optional `model` (the pydantic class) excluded from compare/hash.
- **Why it is here:** Keeping the pydantic class lets us validate arguments later.

### 16. ToolSpec: build, render, validate  (`ragkit/gateway.py` lines 41–55)
```python
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
```
- **In simple words:** Make a spec from a pydantic model; render per vendor; validate arguments.
- **Exactly:** `from_model` uses `model_json_schema()`; `to_anthropic` -> `input_schema`; `to_openai` -> `{'type':'function','function':{... 'parameters'}}`; `validate_args` -> `model_validate(args).model_dump()`.
- **Why it is here:** One source of truth, two wire formats.

### 17. ToolCall  (`ragkit/gateway.py` lines 58–62)
```python
@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict
```
- **In simple words:** A request from the model to run a tool.
- **Exactly:** `id` (to match the result), `name`, `arguments` dict.
- **Why it is here:** The id lets parallel tool calls be matched with their results.

### 18. LLMRequest  (`ragkit/gateway.py` lines 65–77)
```python
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
```
- **In simple words:** Everything needed to ask a model.
- **Exactly:** Frozen: `messages` tuple, `model` alias (default `'default'`), `system`, `max_tokens=300`, `temperature=0.0`, `tools`, and `request_id = new_id('req')`. `LLMRequest.user(text, **kw)` builds the common one-message case.
- **Why it is here:** A request is a value: retries and caches can reuse it safely.

### 19. Usage  (`ragkit/gateway.py` lines 80–84)
```python
@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    estimated: bool = True                                     # True when we counted words, not the provider's tokens
```
- **In simple words:** Token counts and whether they are estimates.
- **Exactly:** `input_tokens`, `output_tokens`, `estimated=True`.
- **Why it is here:** Honesty about measured vs estimated numbers.

### 20. LLMResponse  (`ragkit/gateway.py` lines 87–95)
```python
@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    usage: Usage
    provider: str
    model: str
    finish_reason: str = "stop"
    tool_calls: tuple[ToolCall, ...] = ()
    latency_s: float = 0.0
```
- **In simple words:** What came back.
- **Exactly:** `text`, `usage`, `provider`, `model`, `finish_reason` ('stop', 'tool_use', ...), `tool_calls`, `latency_s`.
- **Why it is here:** Callers and meters need more than the text.

### 21. StreamEvent  (`ragkit/gateway.py` lines 98–102)
```python
@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: str                                                  # "delta" (a piece of text) or "final" (the whole response)
    text: str = ""
    response: LLMResponse | None = None
```
- **In simple words:** One item of a stream.
- **Exactly:** `kind` is 'delta' or 'final'; deltas carry `text`, the final carries the whole `response`.
- **Why it is here:** A single event type keeps consumers simple.

### 22. Provider: the contract  (`ragkit/gateway.py` lines 105–108)
```python
class Provider(Protocol):
    name: str
    def complete(self, req: LLMRequest) -> LLMResponse: ...
    def stream(self, req: LLMRequest) -> Iterator[str]: ...
```
- **In simple words:** Two methods and a name.
- **Exactly:** `name`, `complete(req) -> LLMResponse`, `stream(req) -> Iterator[str]`.
- **Why it is here:** Everything vendor-specific lives behind these.

### 23. A module-level token estimator  (`ragkit/gateway.py` lines 111–111)
```python
_COUNT = WordTokenCounter()
```
- **In simple words:** A shared counter for estimates.
- **Exactly:** `ceil(words / 0.75)` (step 04).
- **Why it is here:** A real provider adapter would read usage from the response instead.

### 24. SharedClientProvider: constructor  (`ragkit/gateway.py` lines 114–121)
```python
class SharedClientProvider:
    """Adapter over shared/llm_client.generate: backend 'mock' | 'local' | 'anthropic'."""

    def __init__(self, backend: str = "mock", mock=None, name: str | None = None):
        _use_module14()
        import llm_client
        self._llm, self.backend, self.mock = llm_client, backend, mock
        self.name = name or backend
```
- **In simple words:** Adapt `shared/llm_client.generate`.
- **Exactly:** `backend` 'mock' | 'local' | 'anthropic'; optional `mock` (a `MockLLM`); `name` for reports; loads `llm_client` lazily via `_use_module14()`.
- **Why it is here:** One class covers three backends.

### 25. SharedClientProvider.complete  (`ragkit/gateway.py` lines 123–130)
```python
    def complete(self, req: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()
        msgs = [{"role": "assistant" if m.role is Role.ASSISTANT else "user", "content": m.content}
                for m in req.messages if m.role is not Role.SYSTEM]       # shared client knows only user/assistant
        text = self._llm.generate(msgs, system=req.system or None, max_tokens=req.max_tokens,
                                  temperature=req.temperature, backend=self.backend, mock=self.mock)
        usage = Usage(_COUNT.count(req.system + " ".join(m.content for m in req.messages)), _COUNT.count(text))
        return LLMResponse(text, usage, self.name, req.model, "stop", (), time.perf_counter() - t0)
```
- **In simple words:** Translate request -> messages, call, wrap the reply.
- **Exactly:** Messages become `[{'role': 'user'|'assistant', 'content': ...}]` (system is passed separately; TOOL maps to user); text -> `Usage(estimated)`; `time.perf_counter()` gives `latency_s`.
- **Why it is here:** Translating once at the edge keeps vendor formats out of the core.

### 26. SharedClientProvider.stream (simulated)  (`ragkit/gateway.py` lines 132–135)
```python
    def stream(self, req: LLMRequest) -> Iterator[str]:
        """Simulated streaming: generate fully, then yield word by word. (A real provider yields tokens as decoded.)"""
        for w in re.findall(r"\S+\s*", self.complete(req).text):
            yield w
```
- **In simple words:** Cut a finished reply into words.
- **Exactly:** `re.findall(r'\S+\s*', text)` yields each word with its trailing space.
- **Why it is here:** Good enough to test streaming code; a real adapter would yield decoded tokens.

### 27. SharedClientProvider.astream  (`ragkit/gateway.py` lines 137–140)
```python
    async def astream(self, req: LLMRequest) -> AsyncIterator[str]:
        for w in self.stream(req):
            await asyncio.sleep(0)                             # hand control back to the event loop between pieces
            yield w
```
- **In simple words:** The async version.
- **Exactly:** `async for` over `stream`, with `await asyncio.sleep(0)` between pieces to yield to the event loop.
- **Why it is here:** Without an `await`, an async generator would hog the loop.

### 28. ScriptedProvider: a test double  (`ragkit/gateway.py` lines 143–147)
```python
class ScriptedProvider:
    """Deterministic provider for tests: replies come from a list (or callable); optionally raises."""

    def __init__(self, name: str, replies: Sequence[Any], model: str = "scripted"):
        self.name, self.replies, self.calls = name, list(replies), 0
```
- **In simple words:** A provider that follows a script.
- **Exactly:** `replies` list; each item a string, an exception (class or instance), or `(text, [ToolCall...])`; counts `calls`.
- **Why it is here:** Deterministic failures and tool calls for tests and demos.

### 29. ScriptedProvider.complete and stream  (`ragkit/gateway.py` lines 149–162)
```python
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
```
- **In simple words:** Pop the next scripted outcome.
- **Exactly:** Raises scripted exceptions; a `(text, calls)` tuple makes `finish_reason='tool_use'`; estimates usage; `stream` yields the words of the same reply.
- **Why it is here:** Covers success, failure and tool-call paths in one class.

### 30. Price table and BudgetExceeded  (`ragkit/gateway.py` lines 165–174)
```python
# ------------------------------------------------------------------ cost accounting
PRICES_PER_MTOK = {                                            # (input, output) USD per 1M tokens -- ILLUSTRATIVE, not real prices
    "mock": (0.0, 0.0),
    "small": (0.25, 1.25),
    "large": (3.0, 15.0),
}


class BudgetExceeded(PermanentError):
    pass
```
- **In simple words:** Illustrative USD per million tokens.
- **Exactly:** `PRICES_PER_MTOK = {'mock': (0,0), 'small': (0.25, 1.25), 'large': (3.0, 15.0)}` are placeholders, not real prices. `BudgetExceeded` subclasses `PermanentError` so retry/fallback never mask it.
- **Why it is here:** A budget error must stop the request, not trigger a failover to a pricier model.

### 31. CostMeter: state and cost()  (`ragkit/gateway.py` lines 177–186)
```python
class CostMeter:
    def __init__(self, prices: dict[str, tuple[float, float]] | None = None, budget_usd: float | None = None):
        self.prices, self.budget = prices or PRICES_PER_MTOK, budget_usd
        self.spent = 0.0
        self.input_tokens = self.output_tokens = self.requests = 0
        self.by_model: dict[str, float] = {}

    def cost(self, model: str, usage: Usage) -> float:
        pin, pout = self.prices.get(model, (0.0, 0.0))
        return (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6
```
- **In simple words:** Totals and a per-call cost function.
- **Exactly:** `spent`, `input_tokens`, `output_tokens`, `requests`, `by_model`; `cost = (in * price_in + out * price_out) / 1e6`.
- **Why it is here:** Unknown models cost 0 here; production should raise instead.

### 32. CostMeter.check  (`ragkit/gateway.py` lines 188–190)
```python
    def check(self) -> None:
        if self.budget is not None and self.spent >= self.budget:
            raise BudgetExceeded(f"budget ${self.budget:.4f} exhausted (spent ${self.spent:.4f})")
```
- **In simple words:** Refuse when the budget is used up.
- **Exactly:** `if budget is not None and spent >= budget: raise BudgetExceeded(...)`.
- **Why it is here:** Called BEFORE spending.

### 33. CostMeter.record  (`ragkit/gateway.py` lines 192–202)
```python
    def record(self, resp: LLMResponse) -> float:
        c = self.cost(resp.model, resp.usage)
        self.spent += c
        self.requests += 1
        self.input_tokens += resp.usage.input_tokens
        self.output_tokens += resp.usage.output_tokens
        self.by_model[resp.model] = self.by_model.get(resp.model, 0.0) + c
        return c


# ------------------------------------------------------------------ the gateway
```
- **In simple words:** Add a response's cost to the totals.
- **Exactly:** Updates `spent`, token totals, `requests`, `by_model`; returns this call's cost.
- **Why it is here:** Feeds dashboards and the cost-per-answer metric.

### 34. Gateway: constructor  (`ragkit/gateway.py` lines 203–208)
```python
class Gateway:
    """routes: alias -> ordered providers (primary first). The caller names an alias; the gateway picks, falls back, meters."""

    def __init__(self, routes: dict[str, list], meter: CostMeter | None = None, bus=None):
        self.routes, self.meter, self.bus = routes, meter or CostMeter(), bus or (lambda n, p: None)
        self.last_chain: FallbackChain | None = None
```
- **In simple words:** Routes plus meter plus event hook.
- **Exactly:** `routes: dict[alias -> list[Provider]]`, `meter`, `bus` (callable, default no-op), `last_chain` (for inspection).
- **Why it is here:** All policy lives here, not in the application.

### 35. Gateway.complete  (`ragkit/gateway.py` lines 210–218)
```python
    def complete(self, req: LLMRequest) -> LLMResponse:
        self.meter.check()                                     # refuse BEFORE spending
        chain = FallbackChain([(p.name, p.complete) for p in self.routes[req.model]])
        self.last_chain = chain
        resp = chain(req)
        cost = self.meter.record(resp)
        self.bus("llm_call", {"provider": resp.provider, "model": resp.model, "in": resp.usage.input_tokens,
                              "out": resp.usage.output_tokens, "cost": cost, "fell_back": bool(chain.failures)})
        return resp
```
- **In simple words:** Budget check, fallback chain, record, emit.
- **Exactly:** `meter.check()`; `FallbackChain([(p.name, p.complete) ...])`; `resp = chain(req)`; `cost = meter.record(resp)`; `bus('llm_call', {provider, model, in, out, cost, fell_back})`.
- **Why it is here:** Every call is metered and observable.

### 36. Gateway.stream  (`ragkit/gateway.py` lines 220–231)
```python
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
```
- **In simple words:** Stream from the first provider, then meter.
- **Exactly:** Yields `delta` events, joins the text, estimates usage, records it, yields the `final` event.
- **Why it is here:** No fallback mid-stream.

### 37. Gateway.complete_structured: repair loop  (`ragkit/gateway.py` lines 233–246)
```python
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
```
- **In simple words:** Ask, parse, validate, repair.
- **Exactly:** Loop `attempt in range(max_repairs + 1)`: `complete(...)`; `schema.model_validate(extract_json(text))`; on `ValueError`/`ValidationError` append the assistant reply and a user message with the error (first 300 chars) and loop; after the last attempt raise `PermanentError`.
- **Why it is here:** Giving the model its own error is the cheapest effective repair.

### 38. extract_json  (`ragkit/gateway.py` lines 249–254)
```python
def extract_json(text: str) -> dict:
    """First '{' to last '}' -- models often wrap JSON in chatter or code fences."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in reply: {text[:80]!r}")
    return json.loads(m.group(0))
```
- **In simple words:** Cut the JSON out of surrounding text.
- **Exactly:** Regex `\{.*\}` with `DOTALL` (first `{` to last `}`), then `json.loads`; raises `ValueError` if none.
- **Why it is here:** Models add 'Sure! Here you go:' and code fences.

### 39. GatewayGenerator: Gateway as a Generator  (`ragkit/gateway.py` lines 257–264)
```python
class GatewayGenerator:
    """Adapter: Gateway -> the Generator Protocol, so RagService can run on top of the gateway unchanged."""

    def __init__(self, gateway: Gateway, model: str = "default", max_tokens: int = 300):
        self.gateway, self.model, self.max_tokens = gateway, model, max_tokens

    def generate(self, system: str, user: str) -> str:
        return self.gateway.complete(LLMRequest.user(user, model=self.model, system=system, max_tokens=self.max_tokens)).text
```
- **In simple words:** Adapter back to the step 04 interface.
- **Exactly:** `generate(system, user)` -> `gateway.complete(LLMRequest.user(user, model=..., system=system, max_tokens=...)).text`.
- **Why it is here:** Lets the whole RAG service run on the gateway unchanged.

## Syntax explained
| Syntax | Meaning |
|---|---|
| `async def` / `async for` / `await` | Coroutines and async iteration; `asyncio.run(coro)` runs one |
| `yield` inside `async def` | Async generator |
| `field(default_factory=lambda: new_id('req'))` | Per-instance default computed at construction |
| `Type[BaseModel]` | A pydantic model class (not an instance) |
| `model_validate`, `model_dump`, `model_json_schema` | pydantic v2 parse / export / schema |
| `re.findall(r'\S+\s*', s)` | Words with trailing whitespace preserved |

## Libraries & functions
| Name | What it does here |
|---|---|
| `pydantic` v2 | Validation of LLM JSON and tool arguments; JSON Schema generation |
| `asyncio` | Async iteration demo (`asyncio.run`) |
| `shared/llm_client.py` | `generate`, `MockLLM`, three backends |
| `ragkit.resilience.FallbackChain` | Provider fallback |
| `json`, `re` | JSON extraction |

## Execution trace (real run)
```text
complete: provider=mock-b text='Refunds take 9 days after delivery [1].'
          fell back: [('mock-a', 'TransientError')] | tokens in/out=10/10 (estimated=True) | cost=$0.00001500 | latency=21.10 ms
stream  : ['Refunds ', 'take ', '9 ', 'days ', 'after ', 'delivery ', '[1].'] then final usage=Usage(input_tokens=8, output_tokens=10, estimated=True)
async   : ['Refunds ', 'take ', '9 ', 'days ', 'after ', 'delivery ', '[1].'] (async iterator: await between pieces keeps the event loop free)
structured: repairs used=1 -> StructuredAnswer(answer='9 days', citations=[1], confidence=0.9)
anthropic format: {"name": "search_docs", "description": "Search the knowledge base", "input_schema": {"properties": {"query": {"title": "Query", "type": "string"}, "to ...
openai format   : {"type": "function", "function": {"name": "search_docs", "description": "Search the knowledge base", "parameters": {"properties": {"query": {"title":  ...
validate args   : {'query': 'refund', 'top_k': 5} (top_k '5' coerced to int)
tool-call reply : finish_reason = tool_use | calls = (ToolCall(id='call_1', name='search_docs', arguments={'query': 'refund'}),)
RAG on gateway: Refunds are issued within 9 days of delivery. [1] | citations ok=True | gateway requests=1 spend=$0.00007050
events seen by the bus: [('mock-b', True), ('mock', False)]
```
- The first provider raised a 503; the gateway answered from `mock-b` and recorded the failover. Cost is computed from estimated tokens: 10 in, 10 out at the small-model illustrative prices.
- Streaming delivers seven pieces and a final usage event; the async version yields the same pieces.
- Structured output: the first reply failed validation (`confidence` 7 > 1), one repair call fixed it; `repairs used=1`.
- The tool spec is the same schema in two shapes; `top_k: '5'` was coerced to `5` by pydantic.
- The RAG service answered through the gateway with one metered request; the bus recorded two LLM calls overall (first section's fallback call and this one).

## Experiments (change ONE thing)
Each snippet runs from this step folder (`cd` into it first). Outputs below are from real runs.

### Cost of 1000 requests: small vs large
```python
from ragkit.gateway import *
prices = {"small": (0.25, 1.25), "large": (3.0, 15.0)}
for m in ("small", "large"):
    cm = CostMeter(prices); u = Usage(2000, 300)
    print(m, "1000 requests of 2000 in / 300 out tokens: $%.2f" % (1000 * cm.cost(m, u)))
```
Observed:
```text
small 1000 requests of 2000 in / 300 out tokens: $0.88
large 1000 requests of 2000 in / 300 out tokens: $10.50
```
Same traffic, 12x the cost: $0.88 vs $10.50 per 1000 requests at the illustrative prices. Routing easy questions to the small alias and hard ones to the large alias is a major FinOps lever; the gateway is where that rule lives.

### Repairs: how many chances does the model get?
```python
from ragkit.gateway import *
from ragkit.schemas import StructuredAnswer
from ragkit.resilience import PermanentError
bad = '{"answer": "x", "confidence": 7}'; good = '{"answer": "x", "confidence": 0.5}'
for repairs in (0, 1):
    gw = Gateway({"default": [ScriptedProvider("p", [bad, good])]})
    try:
        o, n = gw.complete_structured(LLMRequest.user("q"), StructuredAnswer, max_repairs=repairs)
        print("max_repairs", repairs, "-> ok after", n, "repair(s) | provider calls", gw.meter.requests)
    except PermanentError as e:
        print("max_repairs", repairs, "-> PermanentError:", str(e)[:80], "| provider calls", gw.meter.requests)
```
Observed:
```text
max_repairs 0 -> PermanentError: no valid structured output after 0 repair(s): 1 validation error for StructuredA | provider calls 1
max_repairs 1 -> ok after 1 repair(s) | provider calls 2
```
With `max_repairs=0` the same bad reply raises `PermanentError` after **1** provider call; with one repair it succeeds after **2** calls. A repair doubles latency and cost for that request: track the repair rate as a quality metric.

### What the repair prompt looks like
```python
from ragkit.gateway import *
from ragkit.schemas import StructuredAnswer
seen = []
p = ScriptedProvider("p", ['{"answer": "x", "confidence": 7}', '{"answer": "x", "confidence": 0.5}'])
orig = p.complete
p.complete = lambda r: (seen.append(r), orig(r))[1]
Gateway({"default": [p]}).complete_structured(LLMRequest.user("answer as JSON"), StructuredAnswer)
for m in seen[1].messages: print(m.role.value, "|", m.content[:110].replace("\n", " / "))
```
Observed:
```text
user | answer as JSON
assistant | {"answer": "x", "confidence": 7}
user | Your JSON was invalid: 1 validation error for StructuredAnswer / confidence /   Input should be less than or equal
```
The repair prompt is the whole conversation so far: the original request, the model's invalid reply as an `assistant` message, and a `user` message quoting the validation error (`confidence ... Input should be less than or equal to 1`).

## Debugging practice
**BREAK IT:**
```python
import sys; sys.path.insert(0, '..')

from ragkit.gateway import *
gw = Gateway({"small": [ScriptedProvider("p", ["hi"])]})
gw.complete(LLMRequest.user("q", model="medium"))
```
Real output:
```text
Traceback (most recent call last):
  File "<string>", line 5, in <module>
  File "/Users/aman/Desktop/Project_resume/RAP_interview/20_llm_system_lld/08_llm_gateway_and_streaming/../ragkit/gateway.py", line 212, in complete
    chain = FallbackChain([(p.name, p.complete) for p in self.routes[req.model]])
                                                         ~~~~~~~~~~~^^^^^^^^^^^
KeyError: 'medium'
```
`KeyError: 'medium'` raised in `Gateway.complete` at `self.routes[req.model]`: the alias `medium` is not in the route table (`small` only). The traceback's last line is the missing key; the line above is where the lookup happened. A production gateway should convert this to a clear configuration error listing the valid aliases, like `Registry.create` does (step 05).

## Rebuild checklist
- [ ] Write the unified `LLMRequest`/`LLMResponse`/`Usage` types and say why they are frozen.
- [ ] Write a `Provider` Protocol and an adapter for a vendor client that has `messages.create(...)`.
- [ ] Write `complete_structured` with a repair loop; say what you put in the repair message.
- [ ] Write a sync and an async streaming generator; explain why streaming cannot fall back mid-way.
- [ ] Render one pydantic model as an Anthropic tool and as an OpenAI tool.
- [ ] Write a `CostMeter` with a budget guard and explain why `BudgetExceeded` must be a permanent error.

## The new problem (cliffhanger)
The service is pluggable, resilient, cached and metered, but when something goes wrong we can only guess: which stage was slow, how many tokens did *this* request use, did the cache help, did users like the answer? [Step 09](../09_observability_and_eval_hooks/WALKTHROUGH.md) adds traces, logs, metrics and online eval hooks.

> **Say it aloud (interview box):** "My gateway has frozen request and response types, a two-method Provider Protocol, and routes from a model alias to an ordered list of providers with fallback. It meters tokens and cost with a budget guard, streams through a generator or async iterator, validates structured output with pydantic and a bounded repair loop, and renders one tool schema to each vendor's format. An adapter turns it back into the Generator that the RAG service expects."
