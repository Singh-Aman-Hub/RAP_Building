import asyncio

import pytest
from pydantic import BaseModel

from ragkit.gateway import (BudgetExceeded, CostMeter, Gateway, GatewayGenerator, LLMRequest, Message, Role,
                            ScriptedProvider, SharedClientProvider, ToolCall, ToolSpec, Usage, LLMResponse, extract_json)
from ragkit.resilience import PermanentError, TransientError
from ragkit.schemas import StructuredAnswer


def test_request_is_immutable_and_has_id():
    r = LLMRequest.user("hi")
    assert r.request_id.startswith("req-") and r.messages[0].role is Role.USER
    with pytest.raises(Exception):
        r.model = "x"


def test_complete_and_cost():
    gw = Gateway({"small": [ScriptedProvider("p", ["hello world"])]}, CostMeter({"small": (1.0, 2.0)}))
    resp = gw.complete(LLMRequest.user("a b c d", model="small"))
    assert resp.text == "hello world" and resp.usage.input_tokens > 0
    assert gw.meter.spent == pytest.approx((resp.usage.input_tokens * 1 + resp.usage.output_tokens * 2) / 1e6)
    assert gw.meter.requests == 1


def test_fallback_between_providers_and_event():
    events = []
    gw = Gateway({"default": [ScriptedProvider("a", [TransientError]), ScriptedProvider("b", ["from b"])]},
                 bus=lambda n, p: events.append((n, p)))
    assert gw.complete(LLMRequest.user("q")).provider == "b"
    assert events[0][1]["fell_back"] is True and gw.last_chain.failures == [("a", "TransientError")]


def test_budget_blocks_next_call():
    gw = Gateway({"large": [ScriptedProvider("p", ["x " * 50, "y"])]}, CostMeter({"large": (1000.0, 1000.0)}, budget_usd=0.01))
    gw.complete(LLMRequest.user("hello there", model="large"))
    with pytest.raises(BudgetExceeded):
        gw.complete(LLMRequest.user("again", model="large"))


def test_streaming_yields_deltas_then_final():
    gw = Gateway({"default": [ScriptedProvider("p", ["one two three"])]})
    events = list(gw.stream(LLMRequest.user("go")))
    assert [e.kind for e in events] == ["delta", "delta", "delta", "final"]
    assert "".join(e.text for e in events[:-1]) == "one two three" and events[-1].response.text == "one two three"
    assert gw.meter.requests == 1


def test_async_streaming_with_shared_client_mock():
    from ragkit.adapters import _use_module14
    _use_module14()
    from llm_client import MockLLM
    prov = SharedClientProvider("mock", mock=MockLLM(rules=[("", "alpha beta gamma")]))

    async def run():
        return [p async for p in prov.astream(LLMRequest.user("x"))]
    assert "".join(asyncio.run(run())) == "alpha beta gamma"


def test_structured_ok_and_repair_and_failure():
    good = '{"answer": "9 days", "citations": [1], "confidence": 0.9}'
    gw = Gateway({"default": [ScriptedProvider("p", [f"Sure! {good} done"])]})
    obj, repairs = gw.complete_structured(LLMRequest.user("q"), StructuredAnswer)
    assert obj.answer == "9 days" and repairs == 0

    bad = '{"answer": "9 days", "confidence": 7}'          # confidence > 1 violates the schema
    gw = Gateway({"default": [ScriptedProvider("p", [bad, good])]})
    obj, repairs = gw.complete_structured(LLMRequest.user("q"), StructuredAnswer)
    assert repairs == 1 and obj.confidence == 0.9

    gw = Gateway({"default": [ScriptedProvider("p", ["not json", "still not"])]})
    with pytest.raises(PermanentError):
        gw.complete_structured(LLMRequest.user("q"), StructuredAnswer, max_repairs=1)


def test_repair_message_contains_the_error():
    prov = ScriptedProvider("p", ['{"answer": 1', '{"answer":"a","confidence":0.5}'])
    seen = []
    orig = prov.complete
    prov.complete = lambda req: (seen.append(req), orig(req))[1]
    Gateway({"default": [prov]}).complete_structured(LLMRequest.user("q"), StructuredAnswer)
    assert "invalid" in seen[1].messages[-1].content and len(seen[1].messages) == 3


class SearchArgs(BaseModel):
    query: str
    top_k: int = 3


def test_tool_spec_schema_and_validation():
    t = ToolSpec.from_model("search_docs", "Search the KB", SearchArgs)
    assert t.to_anthropic()["input_schema"]["required"] == ["query"]
    assert t.to_openai()["function"]["name"] == "search_docs"
    assert t.validate_args({"query": "refund", "top_k": "5"}) == {"query": "refund", "top_k": 5}
    with pytest.raises(Exception):
        t.validate_args({"top_k": 1})


def test_extract_json():
    assert extract_json('x {"a": 1} y') == {"a": 1}
    with pytest.raises(ValueError):
        extract_json("none")


def test_gateway_generator_adapter():
    gw = Gateway({"default": [ScriptedProvider("p", ["adapted"])]})
    assert GatewayGenerator(gw).generate("sys", "user") == "adapted"


def test_tool_call_reply():
    p = ScriptedProvider("p", [("", [ToolCall("t1", "search_docs", {"query": "x"})])])
    r = p.complete(LLMRequest.user("q"))
    assert r.finish_reason == "tool_use" and r.tool_calls[0].name == "search_docs"
