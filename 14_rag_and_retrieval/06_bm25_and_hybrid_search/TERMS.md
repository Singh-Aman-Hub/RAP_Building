# Terms — BM25 and Hybrid Search

Folder: `14_rag_and_retrieval/06_bm25_and_hybrid_search` · Files covered: `main.py`, `WALKTHROUGH.md`, `../common.py` (`tokenize`, `BM25`, `rrf_fuse`)
Back to module: [14 README](../README.md)

**Terms in this folder:** [Sparse retrieval](#sparse-retrieval) · [Inverted index](#inverted-index) · [Tokenisation](#tokenisation) · [Stopwords](#stopwords) · [Stemming](#stemming) · [Term frequency (tf)](#term-frequency-tf) · [IDF (inverse document frequency)](#idf-inverse-document-frequency) · [TF-IDF](#tf-idf) · [BM25](#bm25) · [k1 (term-frequency saturation)](#k1-term-frequency-saturation) · [b (length normalisation)](#b-length-normalisation) · [Vocabulary mismatch](#vocabulary-mismatch) · [Dense retrieval](#dense-retrieval) · [Hybrid search](#hybrid-search) · [Reciprocal Rank Fusion (RRF)](#reciprocal-rank-fusion-rrf) · [Weighted score fusion (min-max)](#weighted-score-fusion-min-max) · [SPLADE (learned sparse)](#splade-learned-sparse) · [hit@k](#hitk) · [MRR (mean reciprocal rank)](#mrr-mean-reciprocal-rank) · [Labelled eval set](#labelled-eval-set) · [collections.Counter](#collectionscounter) · [ZeroDivisionError](#zerodivisionerror)

---

## Sparse retrieval  `(concept)`
**1. Definition:** Sparse retrieval represents texts as high-dimensional vectors over the vocabulary in which almost all weights are zero, and scores documents by the query terms they share.
**2. In plain words:** Keyword search. A chunk only "lights up" for words it actually contains. BM25 is the classic example; it scored hit@1 0.71 here.
**3. Example to quote:** > "Sparse retrieval matches words; dense retrieval matches meaning."
**Seen in:** `main.py`, `"BM25 (sparse)"`

## Inverted index  `(concept)`
**1. Definition:** An inverted index maps each term to a posting list of the documents containing it, with term frequencies and optionally positions.
**2. In plain words:** The index at the back of a book. A query only reads the lists for its few terms, so BM25 stays fast over billions of documents. Our `BM25` class loops over all 36 chunks for clarity instead.
**3. Example to quote:** > "BM25 scales because of the inverted index: a query touches only the posting lists of its terms."
**Seen in:** `WALKTHROUGH.md`, theory

## Tokenisation  `(concept)`
**1. Definition:** Tokenisation splits text into the units (tokens) that a retriever compares.
**2. In plain words:** `tokenize` lower-cases and keeps hyphenated codes whole: "What is in CK-MASALA-250? How many days do I have for a refund?" became `['ck-masala-250', 'many', 'days', 'refund']`. Split the code at hyphens and an exact-code search gets much weaker.
**3. Example to quote:** > "My tokenizer keeps product codes like ck-masala-250 as one token so BM25 can match them exactly."
```python
re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
```
**Seen in:** `common.py`, `tokenize`; `main.py`, section 1

## Stopwords  `(concept)`
**1. Definition:** Stopwords are very frequent function words (the, is, do, a, what) removed before keyword scoring because they carry little meaning.
**2. In plain words:** They would match almost every chunk and add noise. Of the 13 words in the example sentence, only 4 tokens survived.
**3. Example to quote:** > "I drop stopwords so 'what' and 'is' don't add noise to BM25 scores."
**Seen in:** `common.py`, `STOPWORDS`

## Stemming  `(concept)`
**1. Definition:** Stemming reduces words to a common root form (for example "refunds" → "refund") so morphological variants match; lemmatisation does the same using a dictionary.
**2. In plain words:** Our tokenizer does not stem, so "refunds" and "refund" are different tokens to BM25. Search engines usually stem per language.
**3. Example to quote:** > "Without stemming, BM25 treats refunds and refund as different words."
**Seen in:** `WALKTHROUGH.md`, theory and experiments

## Term frequency (tf)  `(concept)`
**1. Definition:** Term frequency is how many times a term occurs in a document.
**2. In plain words:** `docs[i].get(term, 0)`; a chunk that says "refund" twice has tf = 2 for that term.
**3. Example to quote:** > "tf rewards documents that mention the query term, but BM25 saturates it so repetition can't dominate."
**Seen in:** `main.py`, `bm25_one`

## IDF (inverse document frequency)  `(concept)`
**1. Definition:** IDF weights a term by how rare it is across the collection; BM25 commonly uses `ln(1 + (N − df + 0.5)/(df + 0.5))`.
**2. In plain words:** Rare words are informative. With 36 chunks: 'ck-tulsi-100' (in 1 chunk) 3.21, 'refund' (3 chunks) 2.36, 'days' and 'tea' (8 chunks) 1.47.
**3. Example to quote:** > "A product code in one chunk gets idf 3.21; a common word like tea only 1.47."
```python
math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
```
**Seen in:** `main.py`, `idf`; `common.py`, `BM25.__init__`

## TF-IDF  `(concept)`
**1. Definition:** TF-IDF scores a term in a document as its term frequency multiplied by its inverse document frequency.
**2. In plain words:** BM25's ancestor. It has no saturation (ten mentions count ten times) and no length normalisation.
**3. Example to quote:** > "BM25 is TF-IDF plus tf saturation and document-length normalisation."
**Seen in:** `WALKTHROUGH.md`, theory; step 03 used TF-IDF for comparison

## BM25  `(concept)`
**1. Definition:** Okapi BM25 scores a document for a query as `Σ idf(t) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · |d|/avgdl))` over the query terms.
**2. In plain words:** Count matching words, weight rare ones more, stop rewarding repeats, and slightly penalise long chunks. Our hand version scored 3.735 for the refund question vs the refund-window chunk, exactly matching the class.
**3. Example to quote:** > "I wrote BM25 by hand and got 3.735, the same as the library class, so I know what each term does."
**Seen in:** `main.py`, `bm25_one`; `common.py`, `BM25`

## k1 (term-frequency saturation)  `(concept)`
**1. Definition:** k1 is the BM25 parameter controlling how quickly additional occurrences of a term stop increasing the score; the term's contribution approaches `idf · (k1 + 1)` as tf grows.
**2. In plain words:** The 5th "refund" adds much less than the 1st. k1 = 0 means only presence counts. Typical 1.2–2.0; we use 1.5.
**3. Example to quote:** > "k1 caps the reward for repeating a word, so keyword stuffing doesn't win."
**Seen in:** `common.py`, `BM25(k1=1.5)`; `main.py`, comments in `bm25_one`

## b (length normalisation)  `(concept)`
**1. Definition:** b is the BM25 parameter (0 to 1) that controls how much a document's score is normalised by its length relative to the average.
**2. In plain words:** Long chunks naturally contain more words, so their matches count a little less. b = 0.75 here; b = 0 ignores length, b = 1 normalises fully.
**3. Example to quote:** > "b = 0.75 is the usual default; I'd lower it if long documents are genuinely more relevant."
**Seen in:** `common.py`, `BM25(b=0.75)`

## Vocabulary mismatch  `(concept)`
**1. Definition:** Vocabulary mismatch occurs when a query and a relevant document express the same meaning with different words.
**2. In plain words:** BM25's main weakness: paraphrase hit@1 was only 0.50, vs 0.70 for dense. "How many days do I have to get a refund?" put a loyalty chunk first for BM25.
**3. Example to quote:** > "BM25 can't know that 'money back' means 'refund'; that's vocabulary mismatch."
**Seen in:** `main.py`, results table

## Dense retrieval  `(concept)`
**1. Definition:** Dense retrieval embeds queries and documents into continuous vectors with a neural encoder and ranks by vector similarity.
**2. In plain words:** The MiniLM search from steps 03–05. It won overall here: hit@1 0.83, hit@3 0.96, MRR 0.90, but lost "pin codes" and the "money back window" question to BM25.
**3. Example to quote:** > "Dense was strongest on our eval, but it missed short jargon queries that BM25 matched literally."
**Seen in:** `main.py`, `dense_rank`

## Hybrid search  `(concept)`
**1. Definition:** Hybrid search runs a sparse and a dense retriever and fuses their rankings into one list.
**2. In plain words:** A hedge: each covers the other's blind spots. Honestly, here RRF only tied dense on hit@1 (0.83) because the knowledge base is tiny and clean; it pays off on large corpora full of codes, names and jargon.
**3. Example to quote:** > "On our small eval hybrid only tied dense, but in production with SKUs and error strings it's the robust default."
**Seen in:** `main.py`, `methods`, closing prints

## Reciprocal Rank Fusion (RRF)  `(concept)`
**1. Definition:** RRF fuses rankings by scoring each document as `Σ 1/(k + rank)` over the input rankings, with ranks starting at 1 and k usually 60.
**2. In plain words:** Uses only positions, so BM25 scores and cosines never need a common scale. k = 60 keeps one retriever's #1 from dominating, so agreement between retrievers wins.
**3. Example to quote:** > "RRF needs no score calibration: a chunk ranked 1st and 3rd gets 1/61 + 1/63."
```python
for rank, (idx, _) in enumerate(ranking, start=1):
    fused[idx] += 1.0 / (k + rank)
```
**Seen in:** `common.py`, `rrf_fuse`; `main.py`, `"hybrid RRF k=60"`

## Weighted score fusion (min-max)  `(concept)`
**1. Definition:** Weighted score fusion normalises each retriever's scores per query (for example min-max to [0, 1]) and combines them as `alpha · dense + (1 − alpha) · sparse`.
**2. In plain words:** With alpha = 0.5 it got keyword hit@1 1.00 but paraphrase 0.50, overall hit@1 0.79. It needs labelled data to tune alpha and min-max is outlier-sensitive.
**3. Example to quote:** > "A weighted blend can beat RRF once alpha is tuned, but RRF works out of the box."
**Seen in:** `main.py`, `blend_rank`

## SPLADE (learned sparse)  `(concept)`
**1. Definition:** SPLADE is a learned sparse retrieval model that uses a transformer's masked-language-model head to assign weights to vocabulary terms, including expansion terms absent from the text.
**2. In plain words:** Still an inverted-index-friendly sparse vector, but "money back" can activate "refund", softening vocabulary mismatch.
**3. Example to quote:** > "SPLADE keeps the inverted index but learns term weights and expansions, a middle ground between BM25 and dense."
**Seen in:** `WALKTHROUGH.md`, theory

## hit@k  `(concept)`
**1. Definition:** hit@k is the fraction of questions for which a gold (correct) chunk appears in the top k results.
**2. In plain words:** hit@1 = "was the first result right?". Dense: hit@1 0.83, hit@3 0.96; for RAG, hit@k for the k chunks you pass to the LLM is what matters.
**3. Example to quote:** > "Dense hit@3 is 0.96, so if I give the LLM three chunks the answer is almost always there."
**Seen in:** `main.py`, eval loop

## MRR (mean reciprocal rank)  `(concept)`
**1. Definition:** MRR averages 1/rank of the first relevant result over all queries, counting 0 when no relevant result is returned.
**2. In plain words:** Rank 1 → 1, rank 2 → 0.5, rank 3 → 0.33. It gives partial credit for near misses: dense 0.90, RRF 0.88, BM25 0.79.
**3. Example to quote:** > "MRR rewards getting the gold chunk near the top even when it isn't first."
**Seen in:** `main.py`, `mrr = np.mean(...)`

## Labelled eval set  `(concept)`
**1. Definition:** A labelled evaluation set is a list of questions each annotated with the chunk(s) that answer them, used to measure retrieval quality.
**2. In plain words:** `eval_questions.json` has 29 questions; 24 have a gold chunk (11 keyword, 10 paraphrase, 3 exact_code), 5 are unanswerable. Small sets like this are noisy: one question is about 4 percentage points of hit@1.
**3. Example to quote:** > "I choose BM25, dense or hybrid by measuring on labelled questions, broken down by question type."
**Seen in:** `main.py`, `load_eval_set()`

## collections.Counter  `(standard library)`
**1. Definition:** `collections.Counter` is a dict subclass that counts hashable items and offers helpers like `most_common(n)`.
**2. In plain words:** One Counter per chunk gives term frequencies; another gives document frequencies; `rrf_fuse` uses one to add up fused scores.
**3. Example to quote:** > "`Counter(tokenize(text))` is a bag of words in one line."
**Seen in:** `main.py`, section 2; `common.py`, `BM25`, `rrf_fuse`

## ZeroDivisionError  `(standard library)`
**1. Definition:** Python raises `ZeroDivisionError` when dividing by zero.
**2. In plain words:** If RRF ranks start at 0 and k = 0, the top item computes `1.0 / (0 + 0)`: `ZeroDivisionError: float division by zero`. With k = 60 the same off-by-one is silent and the table did not change.
**3. Example to quote:** > "Ranks in RRF start at 1; starting at 0 with k = 0 crashes, and with k = 60 it's a silent off-by-one."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
