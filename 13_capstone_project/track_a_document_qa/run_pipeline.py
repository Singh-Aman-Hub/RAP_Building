"""
GOAL:  Run the whole Track A document-intelligence pipeline once, from the CLI, and print what every stage produced:
       pixels / PDF -> OCR+routing (18) -> parsed_document (18/10 contract) -> layout-aware chunks (19/09) -> MiniLM embeddings + retrieval (14)
       -> optional cross-encoder rerank (14/07) -> grounded answer with page+bbox citations (mock extractive LLM) -> eval (hit@k, citations, latency).

RUN (from this step folder):  python run_pipeline.py            (flags: --k 3  --skip-rerank  --no-save)
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 13_capstone_project/track_a_document_qa/run_pipeline.py
EXPECTED OUTPUT (shortened):
    STAGE 1 OCR/parse: 3 documents ...    STAGE 2 chunks ...    STAGE 3 index ...    STAGE 4 three example questions with citations
    STAGE 5 eval table (dense vs dense+rerank): hit@1, hit@k, MRR, citation acc, answer acc, p50/p95 ms
NEEDS: numpy, scipy, Pillow, torch, transformers, pydantic (cached all-MiniLM-L6-v2 + ms-marco-MiniLM; no downloads, no paid API).  ~20 s on CPU.
Writes artifacts/parsed_*.json, artifacts/chunks.jsonl, artifacts/eval_report.json (the stage-to-stage data contracts, inspectable).
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import docqa

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--skip-rerank", action="store_true", help="skip the cross-encoder pass of the eval")
ap.add_argument("--no-save", action="store_true", help="do not write artifacts/")
args = ap.parse_args()

# ---- STAGE 1: OCR / parse. Pixels and PDF bytes become validated parsed_document dicts (module 18: steps 07, 08, 10) -------------------------
t0 = time.perf_counter()
corpus = docqa.build_default_corpus()                     # {doc_id: (parsed_document, ocr_meta)}; validation by 18/10's pydantic model happens inside
print(f"STAGE 1  OCR + routing + parse: {len(corpus)} documents in {time.perf_counter() - t0:.1f}s")
for doc_id, (doc, meta) in corpus.items():
    types = dict(Counter(e["type"] for e in doc["elements"]))
    routes = dict(Counter(m["route"] for m in meta.values() if "route" in m))
    print(f"   {doc_id:<20} pages={len(doc['pages'])} elements={len(doc['elements']):<3} types={types}" + (f" page routes={routes}" if routes else ""))
mixed = corpus["mixed_pdf"][0]
print("   mixed.pdf page 2 was an image-only scan; the OCR text we now hold:", [e["text"] for e in mixed["elements"] if e["page"] == 2])

# ---- STAGE 2: chunking. 19/09's LayoutChunker keeps tables whole, binds captions, merges cross-page tables, keeps page+bbox ---------------------
print("\nSTAGE 2  layout-aware chunking (19/09 LayoutChunker, max 140 tokens)")
all_chunks = {}
for doc_id, (doc, _) in corpus.items():
    all_chunks[doc_id] = docqa.chunk_document(doc)
    kinds = dict(Counter(c["kind"] for c in all_chunks[doc_id]))
    print(f"   {doc_id:<20} {len(all_chunks[doc_id])} chunks {kinds}")
sample = all_chunks["lumen_q3_report"][2]
print("   one chunk (the contract between chunking and retrieval):")
print("   " + json.dumps({k: sample[k] for k in ("chunk_id", "kind", "section_path", "pages", "boxes", "element_ids")}))
print("   embed_text (what MiniLM sees): " + repr(sample["embed_text"][:110]))

# ---- STAGE 3: index (embed every chunk with the cached all-MiniLM-L6-v2) + the engine with cache / metrics from module 20 ----------------------
engine = docqa.Engine()
t0 = time.perf_counter()
for doc_id, (doc, meta) in corpus.items():
    info = engine.ingest(doc, meta)
print(f"\nSTAGE 3  index: {len(engine.index.chunks)} chunks, vectors {engine.index.vectors.shape}, version {engine.index.version}, "
      f"built in {time.perf_counter() - t0:.1f}s (first call includes loading MiniLM)")

# ---- STAGE 4: ask. retrieve -> (rerank) -> mock extractive LLM -> answer + citations ------------------------------------------------------------
print(f"\nSTAGE 4  grounded answers (LLM backend: {engine.llm.name}; set DOCQA_LLM_BASE_URL to use an OpenAI-compatible server)")
for question, rerank in [("What is the price of Green Tea?", False), ("What is the total due on the invoice?", False),
                         ("Who is the CEO of Lumen?", False)]:
    out = engine.ask(question, k=args.k, rerank=rerank)
    print(f"   Q: {question}\n   A: {out['answer']}   (grounded={out['grounded']}, {out['latency_ms']} ms)")
    for c in out["citations"][:2]:
        print(f"      [{c['rank']}] {'USED ' if c['used'] else '     '}{c['doc_id']} p.{c['page']} bbox={c['bbox']} score={c['score']} {c['kind']}")
again = engine.ask("What is the price of Green Tea?", k=args.k)
print(f"   same question again -> cached={again['cached']} in {again['latency_ms']} ms (module 20 TTLLRUCache, key scoped by index version)")

# ---- STAGE 5: evaluation -----------------------------------------------------------------------------------------------------------------
qs = docqa.load_questions()
print(f"\nSTAGE 5  eval: {len(qs)} questions with gold (doc, page) + answer string; hit@k needs the right page AND the answer text in a top-k chunk")
reports = {"dense": docqa.evaluate(engine, args.k, False)}
if not args.skip_rerank:
    reports["dense+rerank"] = docqa.evaluate(engine, args.k, True)
print(f"   {'setup':<14} {'hit@1':>6} {'hit@'+str(args.k):>6} {'MRR':>6} {'cite acc':>9} {'answer acc':>11} {'p50 ms':>8} {'p95 ms':>8}")
for name, r in reports.items():
    print(f"   {name:<14} {r['hit_at_1']:>6.2f} {r['hit_at_k']:>6.2f} {r['mrr']:>6.2f} {r['citation_acc']:>9.2f} {r['answer_acc']:>11.2f} {r['p50_ms']:>8.1f} {r['p95_ms']:>8.1f}")
worst = reports.get("dense+rerank", reports["dense"])
fails = [r["id"] for r in worst["rows"] if not (r["hit_at_1"] and r["cite_ok"] and r["answer_ok"])]
print("   questions that still miss something:", fails, "(see README 'failure modes' for why)")
print("   engine metrics:", {k: round(v, 1) for k, v in engine.metrics.snapshot().items()})

if not args.no_save:
    out_dir = HERE / "artifacts"
    out_dir.mkdir(exist_ok=True)
    for doc_id, (doc, _) in corpus.items():
        (out_dir / f"parsed_{doc_id}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    (out_dir / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in engine.index.chunks))
    (out_dir / "eval_report.json").write_text(json.dumps(reports, indent=1))
    print(f"\nwrote {out_dir.relative_to(HERE.parent)}/ (parsed_*.json, chunks.jsonl, eval_report.json)")

# TRY IT:   run with DOCQA_LLM_BASE_URL unset (mock) then point it at any OpenAI-compatible server (vLLM, llama.cpp, Ollama) and compare answer_acc;
#           change max_tokens in docqa.chunk_document from 140 to 60 and see which table chunks split (19/09 step).
# BREAK IT: edit artifacts/parsed_mixed_pdf.json (bbox [5, 5, 1, 1]) and call docqa.validate(): the handoff contract names the bad field;
#           delete mixed.pdf and watch the pipeline regenerate it by running 18/08 (the self-sufficient step pattern).
