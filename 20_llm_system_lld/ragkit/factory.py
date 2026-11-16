"""factory.py -- build a whole RAG system from a plain dict (JSON/YAML-friendly config) (step 05)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chunking import FixedWordChunker, HeadingChunker, SentenceChunker
from .ingest import HashingEmbedder, InMemoryVectorStore, IngestionPipeline, Manifest, MarkdownParser
from .patterns import EventBus, Registry
from .qdrant_store import QdrantStore
from .query import (BM25Retriever, CitationVerifier, DenseRetriever, ExtractiveGenerator, HybridRetriever,
                    NoopReranker, OverlapReranker, PromptBuilder, RagService, ScoreThresholdPolicy)

EMBEDDERS: Registry = Registry("embedder")
STORES: Registry = Registry("store")
CHUNKERS: Registry = Registry("chunker")
RETRIEVERS: Registry = Registry("retriever")
RERANKERS: Registry = Registry("reranker")
GENERATORS: Registry = Registry("generator")

# Registration is just "name -> constructor". Adding a plugin never edits build_system().
EMBEDDERS.register("hashing")(lambda dim=128, **_: HashingEmbedder(dim))
STORES.register("memory")(lambda **_: InMemoryVectorStore())
STORES.register("qdrant")(lambda dim=128, **_: QdrantStore(dim))
CHUNKERS.register("heading")(lambda max_words=120, **_: HeadingChunker(max_words))
CHUNKERS.register("fixed")(lambda size=50, overlap=10, **_: FixedWordChunker(size, overlap))
CHUNKERS.register("sentence")(lambda max_words=40, **_: SentenceChunker(max_words))
RETRIEVERS.register("dense")(lambda embedder, store, **_: DenseRetriever(embedder, store))
RETRIEVERS.register("sparse")(lambda embedder, store, **_: BM25Retriever(store))
RETRIEVERS.register("hybrid")(lambda embedder, store, **_: HybridRetriever(
    [DenseRetriever(embedder, store), BM25Retriever(store)]))
RERANKERS.register("none")(lambda **_: NoopReranker())
RERANKERS.register("overlap")(lambda **_: OverlapReranker())
GENERATORS.register("extractive")(lambda **_: ExtractiveGenerator())


@dataclass
class System:
    pipeline: IngestionPipeline
    service: RagService
    store: Any
    bus: EventBus


DEFAULT_CONFIG = {
    "embedder": {"type": "hashing", "dim": 128},
    "store": {"type": "memory"},
    "chunker": {"type": "heading", "max_words": 80},
    "retriever": {"type": "hybrid"},
    "reranker": {"type": "overlap"},
    "generator": {"type": "extractive"},
    "refusal_min_score": 0.3,
    "token_budget": 300,
}


def _make(registry: Registry, spec: dict, **extra):
    spec = dict(spec)
    return registry.create(spec.pop("type"), **spec, **extra)


def build_system(config: dict, loader) -> System:
    cfg = {**DEFAULT_CONFIG, **config}
    emb = _make(EMBEDDERS, cfg["embedder"])
    store = _make(STORES, cfg["store"], dim=emb.dim)
    chunker = _make(CHUNKERS, cfg["chunker"])
    pipeline = IngestionPipeline(loader, MarkdownParser(), chunker, emb, store, Manifest())
    bus = EventBus()
    retriever = _make(RETRIEVERS, cfg["retriever"], embedder=emb, store=store)
    service = RagService(retriever, _make(RERANKERS, cfg["reranker"]), PromptBuilder(budget=cfg["token_budget"]),
                         _make(GENERATORS, cfg["generator"]), CitationVerifier(),
                         ScoreThresholdPolicy(cfg["refusal_min_score"]), store, pipeline.index_version, on_event=bus)
    return System(pipeline, service, store, bus)
