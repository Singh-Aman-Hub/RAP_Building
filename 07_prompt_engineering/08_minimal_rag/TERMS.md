# Terms — Step 08: Minimal RAG

Folder: `07_prompt_engineering/08_minimal_rag` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [07 Prompt engineering README](../README.md)

**Terms in this folder:** [Hallucination](#hallucination) · [Knowledge cutoff / frozen model](#knowledge-cutoff--frozen-model) · [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation) · [Knowledge base (document store)](#knowledge-base-document-store) · [TF-IDF](#tf-idf) · [Cosine similarity search](#cosine-similarity-search) · [Top-k retrieval](#top-k-retrieval) · [Context injection](#context-injection) · [Retrieval miss](#retrieval-miss) · [Similarity threshold](#similarity-threshold)

---

## Hallucination
*Type: concept*

**1. Definition:** A fluent, confident model answer that is not supported by facts.
**2. In plain words:** A model that has never seen your refund policy still answers "usually 30 days", because that is the most typical continuation. It does not know that it does not know.
**3. Example to quote:**
> "Without retrieval the model guessed 30 days; the real policy is 9."

**Seen in:** `main.py`, `WITHOUT retrieval` lines.

## Knowledge cutoff / frozen model
*Type: concept*

**1. Definition:** A trained model's weights do not change after training, so it knows nothing created later or kept private.
**2. In plain words:** The model is a snapshot. New or private facts must arrive through the prompt.
**3. Example to quote:**
> "Weights are frozen, so fresh facts go in the context window, not into the model."

**Seen in:** `WALKTHROUGH.md`, pain point.

## RAG (Retrieval-Augmented Generation)
*Type: concept / pattern*

**1. Definition:** Retrieve relevant documents for a question, add them to the prompt, then generate the answer from them.
**2. In plain words:** An open-book exam for the model. Search first, then answer with the pages open.
**3. Example to quote:**
> "RAG fixes missing knowledge without retraining by putting the right pages in the prompt."

**Seen in:** `main.py`, `ask(..., use_rag=True)`.

## Knowledge base (document store)
*Type: concept*

**1. Definition:** The collection of documents (or chunks) that retrieval searches. Here a plain Python list plus a matrix of their vectors.
**2. In plain words:** Your private facts, split into pieces small enough to paste into a prompt.
**3. Example to quote:**
> "My knowledge base is six sentences and a `(6, 49)` matrix."

**Seen in:** `main.py`, `DOCS`, `doc_vectors`.

## TF-IDF
*Type: third-party: scikit-learn (`TfidfVectorizer`)*

**1. Definition:** Term Frequency x Inverse Document Frequency: a word-count vector where rare words weigh more than common ones.
**2. In plain words:** A cheap way to turn text into numbers using only word overlap. It cannot know that "money back" means "refund".
**3. Example to quote:**
> "TF-IDF matches words, not meaning, so the money-back question missed the refund document."

**Seen in:** `main.py`, `build_embedder`.

## Cosine similarity search
*Type: concept / NumPy code*

**1. Definition:** Rank documents by the cosine of the angle between each document vector and the query vector: `dot(a, b) / (|a| * |b|)`.
**2. In plain words:** Two vectors pointing the same way are similar, whatever their length. One matrix product scores all documents at once.
**3. Example to quote:**
> "`doc_vectors @ q / norms` gives one cosine score per document."

```python
scores = doc_vectors @ q / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q))
```

**Seen in:** `main.py`, `retrieve`.

## Top-k retrieval
*Type: concept*

**1. Definition:** Keep only the k highest-scoring documents.
**2. In plain words:** Small k = short prompt but you may miss the answer. Large k = more tokens and more distracting text.
**3. Example to quote:**
> "I retrieve the top 2 chunks and paste them into the prompt."

**Seen in:** `main.py`, `np.argsort(scores)[::-1][:k]`.

## Context injection
*Type: concept / pattern*

**1. Definition:** Placing retrieved text into the prompt (usually under a `Context:` label) with an instruction to answer only from it.
**2. In plain words:** The model cannot tell retrieved text from instructions unless you label it; retrieved text is also untrusted (see [step 06](../06_prompt_injection/)).
**3. Example to quote:**
> "The prompt is `Context: ... Question: ...` and the system rule says answer only from the context."

**Seen in:** `main.py`, `ask`.

## Retrieval miss
*Type: concept / failure mode*

**1. Definition:** Retrieval returns documents that do not contain the answer, either because the right document uses different words or because no document has the answer.
**2. In plain words:** The model downstream is only as good as what retrieval hands it. Junk context gives a confident wrong answer.
**3. Example to quote:**
> "All scores were 0.0 and the model still answered from an irrelevant compost chunk."

**Seen in:** `main.py`, failure modes 1 and 2.

## Similarity threshold
*Type: concept / guard*

**1. Definition:** A minimum cosine score a chunk must reach to be used; below it the system answers "I don't know".
**2. In plain words:** It turns "best of a bad lot" into "nothing relevant". The value (0.2 here) depends on the embedder and must be tuned on real questions.
**3. Example to quote:**
> "With `min_score=0.2` the gift-card question gets an honest 'I don't know'."

**Seen in:** `main.py`, `retrieve(min_score=...)`.
