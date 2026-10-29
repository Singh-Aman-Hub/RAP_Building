# Module 14 — Exercises

Rules: change ONE thing at a time, predict the result before running, and write down what happened. Worked answers are in [solutions/solutions.md](solutions/solutions.md). Do not look until you have tried.

## Step 01 — documents and why RAG
- **Modify:** add `data/kb/gift_cards.md` with a `# Gift Cards` title and two `## ` sections. Re-run step 01, then step 08, and ask "Do you sell gift cards?". What changed, and why did you not need to retrain anything?
- **Modify:** set `pages = 500` and `price_per_m = 0.5`. Is "paste everything" now affordable? Name two reasons it is still a bad idea.
- **Break it:** remove `sorted(...)` around `glob`. Why can unsorted loading break citations and upserts later?
- **Explain aloud:** RAG vs fine-tuning vs long context, in three sentences.

## Step 02 — chunking
- **Modify:** run `fixed_chars` with size 80 and 1000. Record "cut sentences" and "max words" for each.
- **Modify:** run `solutions/chunk_size_sweep.py`. Why does the biggest chunk size look best on hit@1, and why is that misleading?
- **Break it:** set `overlap=40, size=40`. Read the error and explain it.
- **Rebuild:** write `sentence_pack(text, max_words)` from memory.
- **Explain aloud:** internal trade-off of chunk size: precision of the vector vs context for the LLM.

## Step 03 — embeddings
- **Modify:** replace mean pooling with the first token (`hidden[:, 0]`). Do the pair scores change? Which one does the model card recommend?
- **Modify:** remove the `normalize` line and compute dot products between a short and a very long text. What goes wrong?
- **Break it:** compare a MiniLM vector with an e5 vector. Why is there no error, and why is the number meaningless?
- **Rebuild:** write `embed(texts)` with tokenizer, model, mask-aware mean pooling and normalisation from memory.
- **Explain aloud:** bi-encoder, and why its document vectors can be computed once, offline.

## Step 04 — vector index
- **Modify:** search "cheapest tea" with `where={"doc_id": "product_catalog"}` and print all four results with scores.
- **Modify:** set `N = 1_000_000`. Is the time per query about 10× the 100k time? Why might it not be exactly 10×?
- **Break it:** skip normalising `big`, then compare cosine and dot rankings.
- **Rebuild:** `FlatIndex` with `upsert`, `search(k, where)` and `save/load` from memory.
- **Explain aloud:** pre-filtering vs post-filtering, with the 0-vs-3 example.

## Step 05 — ANN
- **Modify:** `NLIST = 64` and `NLIST = 1024`. Make a table of recall@10 and % scanned at nprobe 1 and 8.
- **Modify:** change the 4 long-range edges in `nsw` to 0, 1 and 8.
- **Modify:** use uniform random data. What happens to IVF recall at nprobe = 1, and why?
- **Explain aloud:** IVF, HNSW and PQ in one sentence each, and the dial each one exposes.

## Step 06 — BM25 and hybrid
- **Modify:** set `b=0`, then `b=1` in `BM25`. Which questions change rank?
- **Modify:** change `alpha` in `blend_rank` to 0.2 and 0.8, and RRF `k` to 1 and 1000.
- **Break it:** start RRF ranks at 0 with `k=0`.
- **Rebuild:** BM25 `scores()` and `rrf_fuse()` from memory.
- **Explain aloud:** one query where BM25 wins and one where dense wins, and why.

## Step 07 — re-ranking
- **Modify:** re-rank only the top 5 stage-1 candidates. Measure hit@1 and latency.
- **Modify:** re-rank dense-only candidates instead of hybrid ones.
- **Break it:** shuffle the passages before scoring but keep the old indices. What silently goes wrong?
- **Explain aloud:** why two stages, with the latency numbers.

## Step 08 — pipeline
- **Modify:** `refuse_below=-12`. Ask the gift-card question. What does the simulator answer?
- **Modify:** `max_context_tokens=60`. Which questions lose their answer?
- **Modify (optional, needs a ~1 GB download):** `LLM_BACKEND=local python main.py`. Compare real-model answers with the simulator's.
- **Break it:** number sources from 0. What does `citations_valid` say?
- **Rebuild:** the prompt template with delimiters, numbered sources and the three rules.
- **Explain aloud:** four failure modes shown in the run, and the fix for each.

## Step 09 — evaluation
- **Modify:** add 5 questions (2 unanswerable) to `data/eval_questions.json`. Do the tables move?
- **Modify:** `supported(..., min_overlap=0.9)` and `0.3`. How fragile is this faithfulness proxy?
- **Break it:** compute MRR with 0-based ranks.
- **Rebuild:** `reciprocal_rank` and `ndcg_at_k` from memory.
- **Explain aloud:** how to tell a retrieval failure from a generation failure.

## Step 10 — advanced RAG
- **Modify:** fuse only the rewrites (drop the original question) in A.
- **Modify:** write a deliberately wrong HyDE answer ("We ship to London in 5 days.").
- **Modify:** semantic cache threshold 0.6, then ask "What are your delivery hours?".
- **Explain aloud:** which advanced technique you would try first for (a) vocabulary mismatch, (b) answers needing surrounding conditions, (c) repeated FAQ traffic.

## Capstone exercise (no AI help)
Write `ask(question) -> {"answer", "sources"}` in under 80 lines, using only `common.py`: heading chunks, MiniLM, hybrid RRF, a refusal threshold, a numbered-sources prompt and the mock LLM. Then evaluate it with the step 09 metric functions.
