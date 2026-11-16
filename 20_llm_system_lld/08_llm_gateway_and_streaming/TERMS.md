# Terms — The LLM gateway: provider abstraction, streaming, structured output, tools and cost

Folder: `20_llm_system_lld/08_llm_gateway_and_streaming` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [LLM gateway (router, proxy)](#llm-gateway-router-proxy--concept) · [Provider abstraction](#provider-abstraction--concept) · [Unified request/response types](#unified-requestresponse-types--concept) · [Model alias (route key)](#model-alias-route-key--concept) · [Streaming (sync generator, async iterator)](#streaming-sync-generator-async-iterator--concept) · [Server-sent events (SSE)](#server-sent-events-sse--concept) · [Structured output (repair loop)](#structured-output-repair-loop--concept) · [Tool-call schema (JSON Schema)](#tool-call-schema-json-schema--concept) · [Token and cost accounting (budget guard)](#token-and-cost-accounting-budget-guard--concept) · [Estimated vs reported usage](#estimated-vs-reported-usage--concept) · [asyncio and async generators](#asyncio-and-async-generators--syntax) · [KeyError: 'medium' (unknown route)](#keyerror-medium-unknown-route--error)

---

## LLM gateway (router, proxy)  `(concept)`
**1. Definition:** A single entry point that maps a model alias to providers and applies policy: fallback, budget, logging, auth.
**2. In plain words:** A switchboard for model calls.
**3. Example to quote:** > "A model upgrade becomes a config change."
**Seen in:** `ragkit/gateway.py`

## Provider abstraction  `(concept)`
**1. Definition:** A minimal interface (complete, stream) that hides each vendor's request format, errors and SDK.
**2. In plain words:** A universal remote.
**3. Example to quote:** > "Vendor details live in adapters, not in the application."
**Seen in:** `ragkit/gateway.py`

## Unified request/response types  `(concept)`
**1. Definition:** Immutable value objects that every provider adapter translates to and from.
**2. In plain words:** A shared form everyone fills in the same way.
**3. Example to quote:** > "Usage, finish reason and tool calls travel with the text."
**Seen in:** `ragkit/gateway.py`

## Model alias (route key)  `(concept)`
**1. Definition:** A logical model name used by application code, mapped by the gateway to concrete providers/models.
**2. In plain words:** A nickname that can point at different people over time.
**3. Example to quote:** > "Lets you swap a model without touching callers."
**Seen in:** `ragkit/gateway.py`

## Streaming (sync generator, async iterator)  `(concept)`
**1. Definition:** Delivering output piece by piece instead of one final string.
**2. In plain words:** Water from a tap rather than a bottle.
**3. Example to quote:** > "Perceived latency is time to first token; usage arrives only at the end."
```python
for ev in gw.stream(req): ...
async for p in provider.astream(req): ...
```
**Seen in:** `ragkit/gateway.py`

## Server-sent events (SSE)  `(concept)`
**1. Definition:** A one-way HTTP streaming protocol (text/event-stream) used by hosted LLM APIs.
**2. In plain words:** A radio broadcast over HTTP.
**3. Example to quote:** > "Errors can arrive after a 200 status, mid-stream."
**Seen in:** `WALKTHROUGH.md`, theory

## Structured output (repair loop)  `(concept)`
**1. Definition:** Requiring model output in a schema: extract JSON, validate with pydantic, and on failure re-ask with the validation error.
**2. In plain words:** Returning a form to be corrected.
**3. Example to quote:** > "Bound the repairs and track the repair rate."
```python
schema.model_validate(extract_json(resp.text))
```
**Seen in:** `ragkit/gateway.py`

## Tool-call schema (JSON Schema)  `(concept)`
**1. Definition:** A description of a callable function (name, description, parameter schema) given to the model.
**2. In plain words:** A menu of actions with their order forms.
**3. Example to quote:** > "One pydantic model yields both the schema and the argument validator."
```python
ToolSpec.from_model('search_docs', 'Search the KB', SearchArgs)
```
**Seen in:** `ragkit/gateway.py`

## Token and cost accounting (budget guard)  `(concept)`
**1. Definition:** Counting input/output tokens per call, converting to money with a price table, and refusing calls once a budget is spent.
**2. In plain words:** A prepaid meter.
**3. Example to quote:** > "Prices in this module are placeholders; use the provider's price page."
```python
cost = (in * p_in + out * p_out) / 1e6
```
**Seen in:** `ragkit/gateway.py`

## Estimated vs reported usage  `(concept)`
**1. Definition:** Counting words yourself versus reading the token counts the provider returns.
**2. In plain words:** A guess versus the receipt.
**3. Example to quote:** > "Prefer the receipt; flag estimates with `estimated=True`."
**Seen in:** `ragkit/gateway.py`

## asyncio and async generators  `(syntax)`
**1. Definition:** Python's cooperative concurrency: coroutines that `await`, and generators that `async for` consumes.
**2. In plain words:** Many conversations juggled by one waiter.
**3. Example to quote:** > "Always `await` something inside an async generator or it blocks the loop."
```python
async def astream(self, req):
    for w in ...:
        await asyncio.sleep(0)
        yield w
```
**Seen in:** `ragkit/gateway.py`

## KeyError: 'medium' (unknown route)  `(error)`
**1. Definition:** A dict lookup for a model alias that is not configured.
**2. In plain words:** Asking for a room that does not exist.
**3. Example to quote:** > "Convert to a configuration error listing valid aliases."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
