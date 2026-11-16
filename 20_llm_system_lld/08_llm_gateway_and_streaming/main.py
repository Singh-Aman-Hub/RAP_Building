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

# ---- 1. Providers behind one interface; a route is an ALIAS -> ordered list of providers ----------------------------
mock = MockLLM(rules=[(r"refund", "Refunds take 9 days after delivery [1]."), (r"", "I am the default mock reply.")])
primary = ScriptedProvider("mock-a", [TransientError("503 overloaded")])        # fails once, then replies
backup = SharedClientProvider(BACKEND, mock=mock, name=f"{BACKEND}-b")
prices = {"small": (0.25, 1.25), "large": (3.0, 15.0)}                           # ILLUSTRATIVE USD per 1M tokens
events = []
gw = Gateway({"small": [primary, backup]}, CostMeter(prices, budget_usd=0.05), bus=lambda n, p: events.append(p))

resp = gw.complete(LLMRequest.user("How many days for a refund?", model="small", system="Be brief."))
print(f"complete: provider={resp.provider} text={resp.text!r}")
print(f"          fell back: {gw.last_chain.failures} | tokens in/out={resp.usage.input_tokens}/{resp.usage.output_tokens} "
      f"(estimated={resp.usage.estimated}) | cost=${gw.meter.spent:.8f} | latency={resp.latency_s * 1000:.2f} ms")

# ---- 2. Streaming: a generator of deltas, then one final event with usage -----------------------------------------------
pieces = []
gw_s = Gateway({"small": [backup]}, CostMeter(prices))              # streaming uses ONE provider: no mid-stream fallback
for ev in gw_s.stream(LLMRequest.user("How many days for a refund?", model="small")):
    if ev.kind == "delta":
        pieces.append(ev.text)
    else:
        print(f"stream  : {pieces} then final usage={ev.response.usage}")

async def consume():
    return [p async for p in backup.astream(LLMRequest.user("refund?", model="small"))]
print("async   :", asyncio.run(consume()), "(async iterator: await between pieces keeps the event loop free)")

# ---- 3. Structured output: validate with pydantic, repair loop on failure ------------------------------------------------
reply_bad = '{"answer": "9 days", "citations": [1], "confidence": 7}'                       # confidence must be <= 1
reply_good = 'Here you go: {"answer": "9 days", "citations": [1], "confidence": 0.9}'       # chatter around the JSON
gw2 = Gateway({"small": [ScriptedProvider("scripted", [reply_bad, reply_good])]})
obj, repairs = gw2.complete_structured(LLMRequest.user("answer as JSON", model="small"), StructuredAnswer)
print(f"structured: repairs used={repairs} -> {obj!r}")

# ---- 4. Tool-call schema: one spec, two vendor formats ---------------------------------------------------------------------------
class SearchArgs(BaseModel):
    query: str
    top_k: int = 3


tool = ToolSpec.from_model("search_docs", "Search the knowledge base", SearchArgs)
print("anthropic format:", json.dumps(tool.to_anthropic())[:150], "...")
print("openai format   :", json.dumps(tool.to_openai())[:150], "...")
print("validate args   :", tool.validate_args({"query": "refund", "top_k": "5"}), "(top_k '5' coerced to int)")
tc = ScriptedProvider("t", [("", [ToolCall("call_1", "search_docs", {"query": "refund"})])]).complete(LLMRequest.user("x"))
print("tool-call reply : finish_reason =", tc.finish_reason, "| calls =", tc.tool_calls)

# ---- 5. The gateway under the RAG service: Adapter makes it a Generator ------------------------------------------------------------
system = build_system({}, DirectoryLoader(KB))
system.pipeline.run()
gw3 = Gateway({"small": [SharedClientProvider(BACKEND, mock=MockLLM(rules=[(r"", "Refunds are issued within 9 days of delivery. [1]")]))]},
              CostMeter(prices), bus=lambda n, p: events.append(p))
system.service.generator = GatewayGenerator(gw3, model="small")
a = system.service.answer("How many days do I have to get a refund?")
print(f"RAG on gateway: {a.text} | citations ok={a.meta['citations_ok']} | gateway requests={gw3.meter.requests} spend=${gw3.meter.spent:.8f}")
print("events seen by the bus:", [(e['provider'], e['fell_back']) for e in events])

# TRY IT: set the budget to 0.0000001 on gw and call complete() twice: read the BudgetExceeded error.
# TRY IT: LLM_BACKEND=local python main.py (needs the cached Qwen model); only the SharedClientProvider changes.
# BREAK IT: make complete_structured(max_repairs=0) with the bad reply: PermanentError after one call.
