"""
GOAL:  The online half as small, swappable parts: Retriever (dense / sparse / hybrid) -> Reranker -> PromptBuilder
       (token budget) -> Generator -> CitationVerifier, with a refusal policy before generation. Real MiniLM embeddings
       and the real cross-encoder from module 14 are plugged in through ADAPTERS.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 20_llm_system_lld/04_query_pipeline/main.py
EXPECTED OUTPUT (shortened):
    indexed 36 chunks with all-MiniLM-L6-v2 ...
    Q: How many days do I have to get a refund?  -> A: Refunds ... within 9 days ... [1]   citations ok: True
    Q: Can you send tea to my cousin in London?  -> REFUSED (best_score_-10.33_below_-6.5)
    strategy swap: dense / sparse / hybrid top-1 for an exact code and for a paraphrase
NEEDS: torch + transformers + cached MiniLM and ms-marco cross-encoder (HF_HUB_OFFLINE=1). ~10 s on CPU.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragkit.adapters import Module14CrossEncoder, Module14Embedder
from ragkit.chunking import HeadingChunker
from ragkit.ingest import DirectoryLoader, InMemoryVectorStore, IngestionPipeline, Manifest, MarkdownParser
from ragkit.query import (BM25Retriever, CitationVerifier, DenseRetriever, ExtractiveGenerator, HybridRetriever,
                          PromptBuilder, RagService, ScoreThresholdPolicy)

KB = Path(__file__).resolve().parents[2] / "14_rag_and_retrieval" / "data" / "kb"

# ---- offline half (step 03), now with a REAL embedder behind the Embedder Protocol ----------------------------
embedder = Module14Embedder()                                       # adapter around module 14's MiniLM Embedder
store = InMemoryVectorStore()
pipe = IngestionPipeline(DirectoryLoader(KB), MarkdownParser(), HeadingChunker(), embedder, store, Manifest())
print("indexed", pipe.run().chunks_embedded, "chunks with", embedder.name, "| dim", embedder.dim, "| version", pipe.index_version)

# ---- the online half, assembled from Protocol-shaped parts -----------------------------------------------------
dense, sparse = DenseRetriever(embedder, store), BM25Retriever(store)
hybrid = HybridRetriever([dense, sparse])                           # Strategy of Strategies, fused with RRF
service = RagService(hybrid, Module14CrossEncoder(), PromptBuilder(budget=300), ExtractiveGenerator(),
                     CitationVerifier(), ScoreThresholdPolicy(min_score=-6.5), store, pipe.index_version)

for q in ["How many days do I have to get a refund?", "Is anyone there to help me on Sunday?",
          "Do you sell gift cards?", "Can you send tea to my cousin in London?"]:
    t0 = time.perf_counter()
    a = service.answer(q)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\nQ: {q}")
    if a.refused:
        print(f"   REFUSED ({a.refusal_reason})")
    else:
        print(f"   A: {a.text}")
        print(f"   citations: {[(c.marker, c.chunk_id[:8], c.supported) for c in a.citations]} ok={a.meta['citations_ok']} "
              f"prompt_tokens={a.meta['prompt_tokens']} dropped={a.meta['dropped']} best_rerank_score={a.meta['best_score']}")
    print(f"   {ms:.0f} ms total (retrieve+rerank+generate+verify)")

# ---- Strategy swap: same question, three retrievers, only the object injected changes ------------------------------
for swap_q in ["ck-masala-250", "What is the money back window after the tea arrives?"]:
    print(f"\nstrategy swap for {swap_q!r}:")
    for name, r in [("dense", dense), ("sparse", sparse), ("hybrid", hybrid)]:
        top = r.retrieve(swap_q, 3)[0]
        print(f"   {name:6s} top-1: {top.chunk.section!r:32s} score={top.score:.3f} source={top.source}")

# ---- the prompt budget at work: a tiny budget drops chunks instead of overflowing -------------------------------------
small = PromptBuilder(budget=150).build("How many days for a refund?", hybrid.retrieve("refund days", 5))
print(f"\nbudget=150 tokens -> used {len(small.used)} chunk(s), dropped {len(small.dropped)}, total {small.tokens} tokens")
print(small.user[:200].replace("\n", " | "), "...")

# TRY IT: set ScoreThresholdPolicy(min_score=-5): the gift-card question (best score -5.69) now refuses. Is -5 safe for the refund question (1.33)?
# TRY IT: swap hybrid for dense in RagService(...): which question loses its answer?
# BREAK IT: pass an object without .rerank() as the reranker and read the AttributeError (Protocols are not checked at runtime).
