import pytest

from ragkit.domain import Chunk, SearchHit
from ragkit.ingest import HashingEmbedder
from ragkit.query import (BM25Retriever, CitationVerifier, DenseRetriever, ExtractiveGenerator, HybridRetriever,
                          OverlapReranker, PromptBuilder, RagService, ScoreThresholdPolicy, rrf)
from helpers import DOC_A, DOC_B, make_pipeline


def hit(text, score=1.0, section="s"):
    return SearchHit(Chunk.create("d", text, section), score)


@pytest.fixture
def service_parts():
    p, store = make_pipeline([DOC_A, DOC_B])
    p.run()
    emb = HashingEmbedder(64)
    return store, emb


def make_service(store, emb, min_score=0.3, **kw):
    ret = HybridRetriever([DenseRetriever(emb, store), BM25Retriever(store)])
    return RagService(ret, OverlapReranker(), PromptBuilder(budget=200), ExtractiveGenerator(), CitationVerifier(),
                      ScoreThresholdPolicy(min_score), store, "v1", **kw)


def test_bm25_finds_exact_term(service_parts):
    store, _ = service_parts
    assert BM25Retriever(store).retrieve("tracking link")[0].chunk.section == "Tracking"


def test_bm25_rebuilds_after_store_change(service_parts):
    store, _ = service_parts
    bm = BM25Retriever(store)
    bm.retrieve("tracking")
    store.delete_doc(DOC_B.doc_id)
    assert bm.retrieve("tracking") == []


def test_rrf_prefers_agreement():
    a, b, c = hit("a"), hit("b"), hit("c")
    fused = rrf([[a, b], [c, b]])
    assert fused[0].chunk.text == "b"                      # b is in both lists


def test_prompt_budget_skips_oversized_but_keeps_smaller():
    big, small = hit("word " * 300), hit("tiny fact here")
    bp = PromptBuilder(budget=120).build("q?", [big, small])
    assert len(bp.used) == 1 and bp.used[0].chunk.text == "tiny fact here" and len(bp.dropped) == 1
    assert bp.tokens <= 120


def test_prompt_numbers_sources_from_one():
    bp = PromptBuilder().build("q", [hit("alpha fact"), hit("beta fact")])
    assert "[1] (" in bp.user and "[2] (" in bp.user and bp.user.endswith("Question: q")


def test_end_to_end_answer_with_valid_citation(service_parts):
    store, emb = service_parts
    ans = make_service(store, emb).answer("How many days for a refund after delivery?")
    assert not ans.refused and "9 days" in ans.text and ans.meta["citations_ok"] and ans.citations[0].supported


def test_refusal_when_nothing_matches(service_parts):
    store, emb = service_parts
    ans = make_service(store, emb, min_score=0.5).answer("Do you sell gift cards for dragons?")
    assert ans.refused and ans.refusal_reason and not ans.citations


def test_refusal_does_not_call_generator(service_parts):
    store, emb = service_parts

    class Spy:
        calls = 0
        def generate(self, s, u):
            Spy.calls += 1
            return "x"
    svc = make_service(store, emb, min_score=0.99)
    svc.generator = Spy()
    svc.answer("zzz qqq")
    assert Spy.calls == 0


def test_events_fire_in_order(service_parts):
    store, emb = service_parts
    seen = []
    make_service(store, emb, on_event=lambda n, p: seen.append(n)).answer("refund days delivery")
    assert seen == ["retrieved", "reranked", "generated", "verified"]


def test_verifier_flags_bad_marker_and_unsupported():
    used = [hit("Refunds are issued within 9 days.")]
    v = CitationVerifier().verify("Refunds take 9 days [1]. Pigs can fly [2].", used)
    assert v.invalid_markers == (2,) and v.citations[0].supported
    v2 = CitationVerifier().verify("Bananas are yellow [1].", used)
    assert not v2.ok
