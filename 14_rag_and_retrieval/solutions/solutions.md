# Module 14 — Worked answers

Only read after trying. Numbers come from real runs on an M1 CPU. Timings will differ on your machine; ranks and scores are deterministic.

## Step 01
- **gift_cards.md:** the document becomes chunks at the next index build, and the question is now answerable with a citation. No weights change: knowledge lives outside the model. That is the main operational win of RAG.
- **Cheap tokens:** even when it is affordable, pasting everything (1) adds latency on every request, (2) lets the model get distracted by irrelevant text (models use the start and end of long contexts more reliably than the middle), and (3) cannot scale once the corpus grows past the context window. It also gives no clean way to cite or to enforce per-user permissions.
- **Unsorted glob:** file order depends on the filesystem. If positions are used as ids, a re-run can give the same chunk a different id, so citations and upserts point at the wrong text. Use stable ids such as `doc_id#section`.

## Step 02
- **fixed_chars 80 vs 1000:** small sizes cut almost every sentence. Large sizes cut few, but each chunk holds many topics (a blurry vector) and costs more prompt tokens.
- **chunk_size_sweep.py** (real output):
```
max words  chunks  avg words  hit@1  hit@3    MRR
       12      80       12.1   0.58   0.79   0.70
       25      52       18.5   0.50   0.79   0.67
       50      26       37.1   0.46   0.92   0.68
      100      14       68.9   0.75   0.92   0.83
      200       9      107.1   0.83   0.92   0.89
```
  At 200 words each document is one chunk, so there are only 9 candidates and the task becomes "pick the document" (random guessing gets 0.11). The metric got easier while the LLM's job got harder and every prompt more expensive. Heading-aware chunks (36 chunks, ~29 words) reach dense hit@1 0.83 (step 06). Boundaries that follow the document's structure beat any single fixed size.
- **overlap = size:** `step = size - overlap = 0`, and `range(0, n, 0)` raises `ValueError: range() arg 3 must not be zero`. Overlap must be smaller than the chunk size, and it is typically 10–20% of it.

## Step 03
- **CLS vs mean pooling:** the scores change. all-MiniLM-L6-v2 was trained with mean pooling, so use mean pooling. Always pool the way the model was trained.
- **No normalisation:** raw dot products grow with vector length, so the ranking mixes "how similar" with "how long". Normalise, or use cosine.
- **MiniLM vs e5:** both are 384-dimensional, so the shapes match and nothing complains. But the two models' axes mean different things; it is like comparing coordinates from two different maps.

## Step 04
- **1M vectors:** the time is roughly 10×, because brute force is O(n·d). It is not exactly 10×: at 1.5 GB the matrix no longer fits in CPU caches, so memory bandwidth dominates and BLAS threading behaves differently.
- **Pre vs post filter:** post-filtering takes the global top-3 first and then filters. None of the top-3 came from `brewing_guide`, so 0 results. Pre-filtering ranks only the allowed rows, giving 3 results. Vector databases solve this with filtered HNSW or by over-fetching.

## Step 05
- **NLIST:** fewer, larger cells give higher recall at nprobe = 1 but scan more vectors. Many small cells scan less but miss neighbours near cell borders, so they need a larger nprobe. A common starting point is nlist ≈ √n to 4√n.
- **Long-range edges:** 0 edges gives the plain kNN graph, where the walk gets stuck. A few random edges make the graph a small world. More edges raise recall per step but cost memory (HNSW stores M links per node per layer).
- **Uniform data:** there are no clusters, so a query's neighbours are spread over many cells and recall at small nprobe falls. ANN indexes exploit structure in real embeddings.

## Step 06
- **b:** `b=0` ignores length, so long chunks that repeat a word win. `b=1` normalises fully. 0.75 is the classic default.
- **alpha / RRF k:** weighted fusion is sensitive to alpha and to score scales that change from query to query. RRF depends only on ranks, which is why it is the robust default; `k=60` damps the influence of any single top rank.
- **RRF from 0 with k=0:** `1/(0+0)` gives `ZeroDivisionError: division by zero`.

## Step 07
- **Top 5 only:** faster, but if the answer sits at stage-1 rank 6–20 (e.g. "Is anyone there to help me on Sunday?" was at rank 12) the re-ranker never sees it. Re-ranking cannot fix recall.
- **Shuffled passages:** scores are attached to the wrong indices. There is no error and the results look plausible, but they are wrong. Keep `(index, text, score)` together.

## Step 08
- **refuse_below = -12:** the gift-card question now goes to the LLM with irrelevant chunks. The simulator returns a sentence that merely shares words ("gift"), which is the confident wrong answer the threshold existed to prevent.
- **max_context_tokens = 60:** the packer stops at the first chunk that does not fit. For "How many days do I have to get a refund?" the top chunk alone is ~70 tokens, so **nothing** is packed and the simulator answers "I don't know." Short chunks (the Sunday opening-hours chunk) still fit and are still answered. The budget must hold at least one typical chunk, and a smarter packer would skip an oversized chunk or trim it rather than stop.
- **Sources from 0:** the answer cites `[0]`, which fails the `1 <= n <= len(picked)` check, so `citations_valid` is False. Numbering must be consistent between the prompt and the checker.
- **Four failure modes:**
  - Retrieval miss / false refusal (London). Fix: query rewriting or HyDE (step 10).
  - Distractor chunk. Fix: re-ranking, better chunk boundaries, more chunks plus a stronger LLM.
  - Generation picking the wrong sentence. Fix: a real LLM, or better prompts.
  - Prompt injection. Fix: the untrusted-data rule, delimiters, ingest filtering, output checks.

## Step 09
- **min_overlap:** 0.9 rejects true paraphrases, while 0.3 accepts unsupported claims that share a few words. Lexical faithfulness is a cheap proxy. Production systems use an NLI model or an LLM judge, validated against human labels.
- **MRR from 0:** `1 / 0` for a top-1 hit gives `ZeroDivisionError`. Ranks are 1-based.
- **Retrieval vs generation failure:** check whether the gold chunk was in the context. If it was not, the failure is retrieval (fix chunking, the retriever or the re-ranker). If it was, the failure is generation (fix the prompt or the model, or reduce distractors).

## Step 10
- **Rewrites only:** the rank stays 1 here. In general, keep the original question in the fusion; rewrites can drift away from what the user asked.
- **Wrong HyDE answer:** surprisingly, the gold chunk is *still* rank 1 (real run). "We ship to London in 5 days." is false, but it is still *about* shipping abroad, and HyDE only needs topical similarity, not truth. A guess on the wrong topic would pull the wrong chunks. That is the real risk, along with ever showing the guess to the user.
- **Cache threshold 0.6:** "What are your delivery hours?" (cosine 0.66) gets the support-hours answer, which is a wrong answer served instantly. Tune the threshold on labelled pairs, keep a margin, and invalidate entries on document updates.
- **First technique to try:** (a) vocabulary mismatch → hybrid search, then multi-query or HyDE; (b) answers needing surrounding conditions → small-to-big or parent-document retrieval; (c) repeated FAQ traffic → semantic cache plus exact-match cache.
