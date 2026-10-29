# Terms — Re-ranking with a Cross-Encoder

Folder: `14_rag_and_retrieval/07_reranking_cross_encoder` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [Cross-encoder](#cross-encoder) · [Bi-encoder (recap)](#bi-encoder-recap) · [Two-stage retrieval (retrieve then rerank)](#two-stage-retrieval-retrieve-then-rerank) · [Recall stage vs precision stage](#recall-stage-vs-precision-stage) · [Candidate count N](#candidate-count-n) · [Sentence-pair tokenisation](#sentence-pair-tokenisation) · [AutoModelForSequenceClassification](#automodelforsequenceclassification) · [Relevance logit](#relevance-logit) · [MS MARCO](#ms-marco) · [hit@1](#hit1) · [MRR (mean reciprocal rank)](#mrr-mean-reciprocal-rank) · [recall@20](#recall20) · [Latency per query](#latency-per-query) · [time.perf_counter](#timeperf_counter) · [ColBERT (late interaction)](#colbert-late-interaction) · [LLM reranker (pointwise, listwise)](#llm-reranker-pointwise-listwise) · [Abstention threshold and calibration](#abstention-threshold-and-calibration) · [Lost in the middle](#lost-in-the-middle) · [Index and score drift (silent bug)](#index-and-score-drift-silent-bug)

---

## Cross-encoder  `(concept)`
**1. Definition:** A cross-encoder is a transformer that takes a query and a passage concatenated as one input and outputs a single relevance score, with self-attention running across both texts.
**2. In plain words:** It reads the question and the chunk side by side, word against word, like a person checking an answer. Much more accurate than comparing two pre-made vectors, but it must run once per (question, chunk) pair: about 92 ms for 20 chunks on this CPU.
**3. Example to quote:** > "The cross-encoder lifted the Sunday question from rank 12 to rank 1 because it can attend from 'Sunday' straight to 'closed on Sundays'."
**Seen in:** `common.py` (`CrossEncoderReranker`), `main.py` `stage2`

## Bi-encoder (recap)  `(concept)`
**1. Definition:** A bi-encoder encodes queries and documents independently into fixed vectors and scores them with a dot product or cosine similarity.
**2. In plain words:** Documents are embedded once, offline; a query costs only one embedding (about 6 ms here) plus a vector search. Fast and scalable, but less precise because the document vector never saw the question.
**3. Example to quote:** > "Bi-encoders give me recall at scale; cross-encoders give me precision on a short list."
**Seen in:** `common.py` (`Embedder`), `main.py` `stage1`

## Two-stage retrieval (retrieve then rerank)  `(concept)`
**1. Definition:** A retrieval architecture where a cheap first stage returns N candidates from the whole corpus and an expensive second stage re-scores only those N.
**2. In plain words:** Hybrid search picks 20 of 36 chunks; the cross-encoder re-orders those 20. You get most of the cross-encoder's accuracy at a tiny fraction of its full-corpus cost.
**3. Example to quote:** > "Stage 1 had recall@20 of 1.00, so stage 2 always had the gold chunk available, and it raised hit@1 from 0.83 to 0.88."
**Seen in:** `main.py`, sections 1 and 2

## Recall stage vs precision stage  `(concept)`
**1. Definition:** The recall stage aims to include every relevant item among the candidates; the precision stage aims to put the relevant items at the top.
**2. In plain words:** Stage 1 must not miss; order is secondary. Stage 2 must order well; it cannot add anything stage 1 missed.
**3. Example to quote:** > "A re-ranker can never recover a chunk the first stage did not retrieve, so I measure recall@N before tuning the re-ranker."
**Seen in:** `main.py` docstrings of `stage1` and `stage2`

## Candidate count N  `(concept)`
**1. Definition:** The number of first-stage results passed to the re-ranker.
**2. In plain words:** Bigger N means better recall but linearly more re-rank cost. Re-ranking only the top 5 here gave hit@1 0.83 and MRR 0.88 (no gain), because the Sunday question's gold chunk was at rank 12.
**3. Example to quote:** > "I chose N by plotting recall@N and re-rank latency, and took the knee of the curve."
**Seen in:** `main.py` (`k=20`), TRY IT

## Sentence-pair tokenisation  `(third-party: transformers)`
**1. Definition:** Calling a Hugging Face tokenizer with two lists, `tok(list_a, list_b, ...)`, encodes each pair as one sequence `[CLS] a [SEP] b [SEP]` with segment ids.
**2. In plain words:** It glues the question and the passage into one input so the model can attend across them. `[query] * len(passages)` repeats the question once per passage.
**3. Example to quote:** > "`tok([q]*n, passages, truncation=True, max_length=512)` builds n question-passage pairs in one batch."
```python
batch = tok(["refund days?"] * 2, ["Refunds within 9 days.", "Closed on Sundays."],
            padding=True, truncation=True, return_tensors="pt")
```
**Seen in:** `common.py`, `CrossEncoderReranker.score`

## AutoModelForSequenceClassification  `(third-party: transformers)`
**1. Definition:** A transformers class that loads an encoder with a classification head on the pooled `[CLS]` representation, returning `logits` of shape `(batch, num_labels)`.
**2. In plain words:** For this re-ranker `num_labels` is 1, so each pair gets one number; `.squeeze(-1)` turns `(20, 1)` into `(20,)`.
**3. Example to quote:** > "The re-ranker is just a sequence classifier with one output: the relevance logit."
**Seen in:** `common.py`

## Relevance logit  `(concept)`
**1. Definition:** The raw, unbounded output score of the classifier before any sigmoid; higher means more relevant.
**2. In plain words:** Scores like +3.1, -2.79, -9.77 only mean something relative to each other and to labelled examples; they are not probabilities.
**3. Example to quote:** > "For the refund-window question the gold chunk scored 3.1 and the runner-up -2.79, a clear margin that RRF scores never show."
**Seen in:** `main.py`, section 1 and 4

## MS MARCO  `(concept)`
**1. Definition:** MS MARCO is a large Microsoft dataset of real Bing search queries with passages labelled as relevant, widely used to train and benchmark rankers.
**2. In plain words:** The re-ranker `cross-encoder/ms-marco-MiniLM-L-6-v2` learnt "is this passage relevant to this web query" from it. Our tea-shop questions are a different domain, which is one reason it misjudged the London question.
**3. Example to quote:** > "An off-the-shelf MS MARCO re-ranker is a strong default, but domain shift is real: it pushed 'cousin in London' from rank 8 to 20."
**Seen in:** `common.py` (`RERANK_MODEL`)

## hit@1  `(concept)`
**1. Definition:** The share of questions whose top-ranked result is a relevant (gold) chunk.
**2. In plain words:** "Was the first chunk right?" 0.83 = 20 of 24; 0.88 = 21 of 24.
**3. Example to quote:** > "Re-ranking moved hit@1 from 0.83 to 0.88, which is one question out of 24."
**Seen in:** `main.py` (`summ`)

## MRR (mean reciprocal rank)  `(concept)`
**1. Definition:** The average over questions of 1 / (rank of the first relevant result), with 0 if it is not found.
**2. In plain words:** Rank 1 is worth 1, rank 2 is worth 0.5, rank 12 is worth 0.083. It rewards getting the answer near the top, not only at the top.
**3. Example to quote:** > "MRR went from 0.88 to 0.92 after re-ranking."
**Seen in:** `main.py` (`summ`)

## recall@20  `(concept)`
**1. Definition:** The share of relevant items that appear anywhere in the top 20 results.
**2. In plain words:** Here it is 1.00: for every answerable question the gold chunk is among the 20 candidates, so the re-ranker's ceiling is perfect.
**3. Example to quote:** > "Stage-1 recall@20 was 1.00, so any remaining error is an ordering error."
**Seen in:** `main.py`, section 2

## Latency per query  `(concept)`
**1. Definition:** The wall-clock time one query spends in a stage, averaged over many queries.
**2. In plain words:** About 6 ms to embed a question versus about 92 ms to cross-encode 20 candidates on this CPU (your numbers will differ). Per pair that is ~4.6 ms, so 1M chunks would be ~1.3 hours per query.
**3. Example to quote:** > "Re-ranking is linear in the candidate count, so I budget it in milliseconds per candidate."
**Seen in:** `main.py`, section 3

## time.perf_counter  `(standard library)`
**1. Definition:** `time.perf_counter()` returns a monotonic, high-resolution clock value in seconds, suitable for measuring short durations.
**2. In plain words:** Take one reading before and one after, subtract, multiply by 1000 for milliseconds.
**3. Example to quote:** > "I time stages with `perf_counter`, never `time.time`, because it is monotonic and high resolution."
**Seen in:** `main.py`, section 3

## ColBERT (late interaction)  `(concept)`
**1. Definition:** ColBERT is a retrieval model that stores one embedding per document token and scores a query by summing, over query tokens, the maximum similarity with any document token (MaxSim).
**2. In plain words:** A middle ground: token-level matching like a cross-encoder, but the document side is precomputed like a bi-encoder. The cost is storage, one vector per token instead of one per chunk.
**3. Example to quote:** > "If cross-encoder latency is too high, late interaction such as ColBERT keeps token-level matching while precomputing documents."
**Seen in:** `WALKTHROUGH.md`, theory

## LLM reranker (pointwise, listwise)  `(concept)`
**1. Definition:** Using a general LLM to judge relevance, either one passage at a time (pointwise), two at a time (pairwise) or by outputting a permutation of a whole list (listwise).
**2. In plain words:** Smarter about intent but far slower and costlier than a small cross-encoder, and listwise prompts are sensitive to the order you list passages in.
**3. Example to quote:** > "I'd try an LLM reranker only on the top 10 after a cross-encoder, and only if evaluation shows it pays."
**Seen in:** `WALKTHROUGH.md`, theory

## Abstention threshold and calibration  `(concept)`
**1. Definition:** A rule that refuses to answer when the best relevance score is below a threshold; calibration means choosing that threshold, or mapping scores to probabilities, using labelled data.
**2. In plain words:** Gift cards (unanswerable) scored -7.06, but London (answerable) scored -10.94. A "below 0 means refuse" guess would refuse London and the refund question would pass at +1.58. Pick the threshold from data and accept some errors.
**3. Example to quote:** > "Scores of answerable and unanswerable questions overlap, so the threshold is a trade-off chosen on a labelled set, not a constant."
**Seen in:** `main.py`, section 4

## Lost in the middle  `(concept)`
**1. Definition:** The finding (Liu et al., 2023) that language models use information placed at the beginning or end of a long context more reliably than information in the middle.
**2. In plain words:** Good ordering matters even after retrieval: put the best chunk first and do not stuff the prompt with weak ones.
**3. Example to quote:** > "Re-ranking also decides prompt order, and models attend best to the start and end of the context."
**Seen in:** `WALKTHROUGH.md`, theory

## Index and score drift (silent bug)  `(concept)`
**1. Definition:** A bug where scores computed for one ordering of items are paired with the indices of a different ordering.
**2. In plain words:** Reversing the passages sent to the re-ranker produced the same numbers (3.1, -2.79, -9.77) attached to wrong chunks, and hit@1 dropped from 0.88 to 0.00 with no error message.
**3. Example to quote:** > "I keep `(index, score)` pairs together and keep an eval in CI, because this kind of bug never raises an exception."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
