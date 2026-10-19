# Step 08 — Minimal RAG: giving a frozen model the facts it never saw

## The story so far
In [step 07](../07_prompt_testing/) you learned to instruct a model precisely, force valid JSON and test prompts. But a prompt can only rearrange what the model already knows. Its knowledge stopped on the day training ended, and it never saw your company's private documents.

## The pain point
Ask a frozen model "How many days do I have for a refund?" about a shop it has never heard of. It does not say "I don't know". It produces the most typical answer ("usually 30 days"). That confident wrong answer is a hallucination. Retraining the model for every new fact would be slow and expensive.

## The spark
Do not change the model. Change the prompt. Keep the facts in a small searchable store, find the few pieces that relate to the question, and paste them into the prompt as context. The model then reads the answer instead of guessing it. Finding "related pieces" is the cosine-similarity idea from [04_llm_fundamentals/02_vectors_and_similarity](../../04_llm_fundamentals/02_vectors_and_similarity/) applied to documents.

## Goal
Build a 20-line in-memory vector search with NumPy, retrieve the top-k documents, inject them into the prompt, and compare the answer before and after. Then break it on purpose.

## Run it
```bash
cd 07_prompt_engineering/08_minimal_rag && ../../.venv/bin/python main.py
# optional real embeddings (cached model, no network): RAG_EMBEDDER=minilm ../../.venv/bin/python main.py
```
Real output (TF-IDF embedder):
```
Embedder: TF-IDF (word-weight vectors) | 6 documents -> vectors (6, 49)

Q: How many days do I have for a refund?
   WITHOUT retrieval: Refunds are usually accepted within 30 days.
   retrieved: [('Refunds at Chai Kitchen are ...', 0.3), ('Orders ship from the Pune wa...', 0.26)]
   WITH retrieval   : Refunds at Chai Kitchen are issued within 9 days of delivery if the seal is unbroken.

Q: What are the support hours?
   WITHOUT retrieval: Support is available 24/7 by phone.
   retrieved: [('Customer support is open Mon...', 0.32), ('Tea is packed in compostable...', 0.0)]
   WITH retrieval   : Customer support is open Monday to Saturday, 10am to 6pm IST.

--- Failure mode 1: retrieval MISS (question uses different words than the document) ---
Q: What is the money back window?
   retrieved: [('Tea is packed in compostable...', 0.0), ('Every blend is nut free, but...', 0.0)]
   WITH retrieval   : Tea is packed in compostable pouches that can go into a home compost bin.

--- Failure mode 2: irrelevant chunk (the answer is NOT in the knowledge base) ---
Q: Do you sell gift cards?
   retrieved: [('Tea is packed in compostable...', 0.0), ('Every blend is nut free, but...', 0.0)]
   WITH retrieval   : Tea is packed in compostable pouches that can go into a home compost bin.
   same, but with a similarity threshold (min_score=0.2):
   retrieved: []
   -> I don't know: nothing relevant in the knowledge base.
```
The "model" here is a scripted simulator (it guesses when it has no context and repeats the best-matching context sentence when it has one). It proves the *pipeline*; a real model would also paraphrase.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `doc_vectors @ q` | matrix times vector = one dot product per document | `(6, 49) @ (49,)` -> `(6,)` |
| `np.linalg.norm(x, axis=1)` | length of each row vector | `norm([3, 4])` -> 5.0 |
| `np.argsort(scores)[::-1][:k]` | indices of the k largest scores | `argsort([.1,.9,.5])[::-1][:2]` -> `[1, 2]` |
| `lambda texts: vec.transform(texts).toarray()` | tiny function: text list -> dense array | – |
| `prompt = f"Context:
{context}

Question: ..."` | injecting retrieved text into the prompt | – |
| `os.getenv("RAG_EMBEDDER")` | read an environment variable | `os.getenv("HOME")` |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `TfidfVectorizer(stop_words="english").fit(corpus)` | scikit-learn | learns the vocabulary and word weights | fitted vectorizer |
| `vec.transform(texts)` | scikit-learn | texts -> sparse TF-IDF matrix | matrix `(n_texts, vocab)` |
| `AutoModel` + mean pooling | transformers | MiniLM sentence embeddings (optional path) | array `(n_texts, 384)` |
| `retrieve(question, k, min_score)` | our code | cosine search over `doc_vectors` | list of `(text, score)` |
| `ask(question, use_rag)` | our code | builds the prompt with or without context | answer `str` |
| `MockLLM(rules=[...])` | shared helper | the scripted frozen model | – |

## Execution trace
1. Six documents become a `(6, 49)` matrix (49 distinct words after stop-word removal).
2. The question becomes one vector `(49,)`. `doc_vectors @ q / norms` gives six cosine scores. The top two are kept.
3. Their text is pasted into the prompt under `Context:`. The model now sees the 9-day rule and repeats it.
4. For the miss, no question word matches any document word, so every score is 0.0 and the "top-2" are arbitrary. The simulator still answers from them.
5. With `min_score=0.2` nothing passes the filter, so the pipeline answers "I don't know" instead of using junk.

## Experiments
- `k=1` then `k=6`: more chunks cost more prompt tokens and add distractors.
- Add a gift-card document, ask again: the same question is now answered correctly. The fix was data, not model.
- `RAG_EMBEDDER=minilm`: scores become 0.5 for the right document (meaning-based vectors). The money-back question still scores low (0.19 / 0.15), so the threshold is what protects you.

## Debugging practice
Embed documents with one method and the query with another (e.g. TF-IDF docs, MiniLM query). The shapes differ and NumPy refuses:
`ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0`. Rule: one embedder for both sides, always. The silent version of this bug is worse: same shape, different meaning, and the scores are garbage.

## Rebuild checklist
- [ ] Embed documents once; embed the query with the same function.
- [ ] Cosine = dot / (norm * norm); take top-k with `argsort`.
- [ ] Paste hits under `Context:` and tell the model to answer only from it.
- [ ] A similarity threshold and an explicit "I don't know" path.

## The new problem (cliffhanger)
The model can now answer from your facts. But it can only *talk*. Ask it "cancel order 1234" or "save this note" and the best it can do is describe how. It has no hands: nothing in this pipeline can run code, read a file or call a service. Giving the brain hands means a loop that lets the model request a tool, runs it safely, and feeds the result back. Next: [08_agent_harness](../../08_agent_harness/).

> **Say it aloud (interview box):** RAG means I retrieve the few documents most similar to the question, using embeddings and cosine similarity, and paste them into the prompt so a frozen model answers from facts instead of guessing. Its failure modes are retrieval misses and irrelevant chunks, so I set a similarity threshold and allow an explicit "I don't know". It fixes missing knowledge without retraining, but it only lets the model talk, not act.
