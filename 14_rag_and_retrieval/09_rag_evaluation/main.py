"""
GOAL:  Evaluate RAG the way you would before shipping it, on 29 labelled questions (24 answerable, 5 not):
         A) RETRIEVAL metrics for four retrievers: hit@1, recall@3, recall@5, MRR, nDCG@5
         B) REFUSAL calibration: sweep the re-rank threshold, trade false refusals against false answers
         C) GENERATION metrics on the full pipeline: correctness, faithfulness (is every answer sentence
            supported by its cited chunk?), citation validity, and an LLM-as-judge prompt template

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/09_rag_evaluation/main.py
EXPECTED OUTPUT (shortened):
    retriever            hit@1  recall@3  recall@5   MRR  nDCG@5
    BM25 ... dense ... hybrid RRF ... hybrid + re-rank ...
    threshold  answered(of 24)  refused(of 5)
    end-to-end: correct x/24 | faithful x/x | citations valid x/x | correct refusals x/5
NEEDS: torch, transformers (cached MiniLM + cross-encoder).  ~30 s.
"""

import importlib.util
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_eval_set, tokenize   # noqa: E402

# Re-use the pipeline from step 08 (loaded by file path, because folder names start with digits).
spec = importlib.util.spec_from_file_location("rag08", HERE.parent / "08_rag_pipeline_end_to_end" / "main.py")
rag08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag08)

rag = rag08.RAGPipeline(rag08.build_chunks())
ids = [c["chunk_id"] for c in rag.chunks]
eval_set = load_eval_set()
answerable = [x for x in eval_set if x["gold"]]
unanswerable = [x for x in eval_set if not x["gold"]]


# =========================================================================================================
# Metric functions. `ranked` = list of chunk ids, best first. `gold` = set of relevant chunk ids.
# =========================================================================================================
def hit_at_k(ranked, gold, k):
    return float(any(r in gold for r in ranked[:k]))


def recall_at_k(ranked, gold, k):
    """Share of ALL relevant chunks found in the top k (equals hit@k when there is one gold chunk)."""
    return len(set(ranked[:k]) & gold) / len(gold)


def reciprocal_rank(ranked, gold):
    """1/rank of the FIRST relevant chunk, 0 if none: rewards putting the answer at the very top."""
    for i, r in enumerate(ranked, 1):
        if r in gold:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked, gold, k):
    """Discounted cumulative gain: a hit at rank i is worth 1/log2(i+1); divided by the best possible."""
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1) if r in gold)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal


# =========================================================================================================
# A) Retrieval
# =========================================================================================================
def dense_only(q, k=20):
    s = rag.matrix @ rag.embedder.encode([q])[0]
    return [(int(j), float(s[j])) for j in np.argsort(-s)[:k]]


retrievers = {
    "BM25": lambda q: rag.bm25.search(q, 20),
    "dense": dense_only,
    "hybrid RRF": rag.retrieve,
    "hybrid + re-rank": lambda q: rag.rerank(q, rag.retrieve(q)),
}
print(f"A) RETRIEVAL on {len(answerable)} answerable questions")
print(f"   {'retriever':<18}{'hit@1':>7}{'recall@3':>10}{'recall@5':>10}{'MRR':>7}{'nDCG@5':>8}")
for name, fn in retrievers.items():
    rows = []
    for x in answerable:
        ranked, gold = [ids[j] for j, _ in fn(x["question"])], set(x["gold"])
        rows.append((hit_at_k(ranked, gold, 1), recall_at_k(ranked, gold, 3), recall_at_k(ranked, gold, 5),
                     reciprocal_rank(ranked, gold), ndcg_at_k(ranked, gold, 5)))
    m = np.mean(rows, axis=0)
    print(f"   {name:<18}{m[0]:>7.2f}{m[1]:>10.2f}{m[2]:>10.2f}{m[3]:>7.2f}{m[4]:>8.2f}")
print("   With 24 questions one question = 0.04. Differences smaller than ~0.1 are noise, not progress.")

# =========================================================================================================
# B) Refusal threshold: the best re-rank score decides "answer" or "I don't know"
# =========================================================================================================
best = {x["id"]: rag.rerank(x["question"], rag.retrieve(x["question"]))[0][1] for x in eval_set}
print(f"\nB) REFUSAL threshold on the best re-rank logit ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")
print(f"   {'threshold':>9} {'answered (of 24)':>17} {'refused (of 5)':>15}  verdict")
for th in (-12, -9, -7.5, -6.5, -5, -3, 0):
    answered = sum(best[x["id"]] >= th for x in answerable)
    refused = sum(best[x["id"]] < th for x in unanswerable)
    verdict = "answers everything (hallucination risk)" if refused == 0 else \
        "refuses everything unknown, but loses real answers" if refused == len(unanswerable) and answered < 22 else ""
    print(f"   {th:>9} {answered:>17} {refused:>15}  {verdict}")
print("   unanswerable scores:", sorted(round(best[x["id"]], 1) for x in unanswerable))
print("   lowest answerable  :", sorted(round(best[x["id"]], 1) for x in answerable)[:4], "<- they overlap: no perfect threshold")

# =========================================================================================================
# C) End-to-end generation metrics
# =========================================================================================================
def supported(sentence, source_text, min_overlap=0.6):
    """Faithfulness proxy: share of the sentence's content words that also appear in the cited chunk."""
    words = set(tokenize(re.sub(r"\[\d+\]", "", sentence)))
    return bool(words) and len(words & set(tokenize(source_text))) / len(words) >= min_overlap


chunk_text = {c["chunk_id"]: c["text"] for c in rag.chunks}
correct = faithful = cited_ok = answered = right_refusal = 0
failures = []
for x in eval_set:
    out = rag.answer(x["question"])
    if not x["gold"]:
        right_refusal += out["refused"]
        if not out["refused"]:
            failures.append(("answered an unanswerable question", x["question"], out["answer"]))
        continue
    if out["refused"]:
        failures.append(("refused an answerable question", x["question"], out["answer"]))
        continue
    answered += 1
    is_correct = x["answer"].lower() in out["answer"].lower()
    correct += is_correct
    cited_ok += out["citations_valid"]
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", out["answer"])]
    sources = [chunk_text[out["sources"][n - 1]] for n in nums if 1 <= n <= len(out["sources"])]
    faithful += bool(sources) and supported(out["answer"], " ".join(sources))
    if not is_correct:
        where = "gold chunk WAS in the context -> generation problem" if x["gold"][0] in out["sources"] \
            else "gold chunk NOT in the context -> retrieval problem"
        failures.append((f"wrong answer ({where})", x["question"], out["answer"]))

print(f"\nC) END-TO-END ({rag08.os.getenv('LLM_BACKEND', 'mock simulator')})")
print(f"   answered {answered}/{len(answerable)} | correct {correct}/{len(answerable)} | faithful {faithful}/{answered} "
      f"| citations valid {cited_ok}/{answered} | correct refusals {right_refusal}/{len(unanswerable)}")
print("   failures to read one by one (error analysis beats staring at averages):")
for kind, q, a in failures:
    print(f"   - {kind}: {q!r}\n       -> {a[:110]}")
print("   Faithful is not the same as correct: a true sentence from the context can still miss the question.")
print("   Here most errors are generation errors: the word-overlap simulator picks the wrong sentence of the right chunk.")

JUDGE_PROMPT = """You are grading a RAG answer.
Question: {question}
Retrieved context: {context}
Answer: {answer}
Score each from 1 to 5 and reply as JSON {{"faithfulness": n, "answer_relevance": n, "context_relevance": n}}:
- faithfulness: is every claim in the answer supported by the context?
- answer_relevance: does the answer address the question?
- context_relevance: is the retrieved context about the question?"""
print("\n   LLM-as-judge template (run it with a STRONG model; validate it against ~50 human labels first):")
print("   " + JUDGE_PROMPT.splitlines()[0], "...", JUDGE_PROMPT.splitlines()[-3].strip())

# TRY IT: add 5 new questions to data/eval_questions.json (including 2 unanswerable) and re-run. Do the tables move?
# TRY IT: set min_overlap in supported() to 0.9 and 0.3. How sensitive is this faithfulness proxy?
# TRY IT: re-run with rag08.RAGPipeline(..., top_k=1). Which metric changes: retrieval or end-to-end?
# BREAK IT: compute MRR with ranks starting at 0 (enumerate(ranked)) -> ZeroDivisionError for a top-1 hit.
