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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| RAG | retrieval-augmented generation, grounded generation | Yes | Fetch relevant text first, then put it in the prompt |
| Embedding | vector, dense representation | Near | A list of numbers representing meaning (or, for TF-IDF, word weights) |
| TF-IDF | sparse vectors, bag-of-words weighting | No | Counts word overlap, not meaning; MiniLM embeds meaning |
| Cosine similarity | normalised dot product | Yes | Angle between two vectors; 1 = same direction |
| top-k | k nearest neighbours, k best hits | Near | Keep the k highest-scoring chunks |
| Hallucination | confabulation, making things up | Near | Confident answer not supported by facts |

## Code, step by step

### 1. Header docstring  (`main.py` lines 1–10)

```python
"""
GOAL   : Show why a frozen model "hallucinates" about private facts, then fix it with a minimal RAG
         pipeline: embed documents -> cosine-similarity search (NumPy) -> inject top-k into the prompt.
RUN    : cd 07_prompt_engineering/08_minimal_rag && python main.py
         (optional: RAG_EMBEDDER=minilm python main.py   -> real sentence embeddings, needs the cached
          all-MiniLM-L6-v2 model; falls back to TF-IDF automatically if it cannot be loaded offline)
EXPECTED OUTPUT: the same questions answered WITHOUT and WITH retrieved context, then two failure modes
         (retrieval miss, irrelevant chunk) and the similarity-threshold fix.
NEEDS  : numpy, scikit-learn (installed in .venv). No network. The "model" is a scripted simulator.
"""
```

- **In simple words:** Says: show hallucination, then fix it with a tiny RAG pipeline.
- **Exactly:** Default embedder is TF-IDF; `RAG_EMBEDDER=minilm` optionally uses MiniLM. The model is a scripted simulator.
- **Why it is here:** Tells the reader what is real (retrieval maths) and what is toy (the answerer).

### 2. Imports and shared client  (`main.py` lines 11–19)

```python
import os
import sys
from pathlib import Path

import numpy as np                                    # all vector maths below is plain NumPy
from sklearn.feature_extraction.text import TfidfVectorizer   # turns text into word-weight vectors

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared"))
from llm_client import generate, MockLLM
```

- **In simple words:** NumPy for the maths, scikit-learn for TF-IDF, our LLM helper.
- **Exactly:** `TfidfVectorizer` turns text into weighted word-count vectors. `sys.path` adds `<repo>/shared`.
- **Why it is here:** Missing path insert makes `llm_client` unimportable.

### 3. The private knowledge base  (`main.py` lines 24–31)

```python
DOCS = [
    "Refunds at Chai Kitchen are issued within 9 days of delivery if the seal is unbroken.",
    "Orders ship from the Pune warehouse and arrive in 4 working days. Shipping is free above 799 rupees.",
    "Customer support is open Monday to Saturday, 10am to 6pm IST. It is closed on Sunday.",
    "Loyalty points: customers earn 3 points per 100 rupees. 500 points give one free 250 g pack.",
    "Every blend is nut free, but the packing facility also handles sesame.",
    "Tea is packed in compostable pouches that can go into a home compost bin.",
]
```

- **In simple words:** Six invented facts about "Chai Kitchen" that no public model could know.
- **Exactly:** `DOCS` is a `list[str]` with 6 entries.
- **Why it is here:** Facts the model cannot have learned show that retrieval is needed.

### 4. Build an embedder (TF-IDF or MiniLM)  (`main.py` lines 36–55)

```python
def build_embedder(corpus):
    if os.getenv("RAG_EMBEDDER") == "minilm":
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")             # never touch the network
            import torch
            from transformers import AutoModel, AutoTokenizer
            name = "sentence-transformers/all-MiniLM-L6-v2"
            tok, net = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name)
            def embed(texts):                                        # returns shape: (n_texts, 384)
                batch = tok(texts, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    hidden = net(**batch).last_hidden_state          # shape: (n_texts, n_tokens, 384)
                mask = batch["attention_mask"].unsqueeze(-1)         # shape: (n_texts, n_tokens, 1)
                return ((hidden * mask).sum(1) / mask.sum(1)).numpy()  # mean-pool -> (n_texts, 384)
            embed(["warm up"])
            return embed, "MiniLM (sentence embeddings)"
        except Exception as err:                                     # model not cached, etc.
            print(f"[minilm unavailable ({type(err).__name__}); falling back to TF-IDF]")
    vec = TfidfVectorizer(stop_words="english").fit(corpus)
    return (lambda texts: vec.transform(texts).toarray()), "TF-IDF (word-weight vectors)"   # (n_texts, vocab)
```

- **In simple words:** A function that returns an "embed" function turning texts into vectors.
- **Exactly:** Default path: `TfidfVectorizer(stop_words="english").fit(corpus)` learns the vocabulary; `embed(texts)` returns a dense array of shape `(n_texts, vocab)`. In this run vocab is 49, so docs are `(6, 49)`. MiniLM path: tokenise, run `AutoModel`, take `last_hidden_state` `(n, tokens, 384)`, mean-pool with the attention mask to `(n, 384)`; any failure falls back to TF-IDF with a message.
- **Why it is here:** Search needs vectors; the fallback keeps the script runnable offline.

### 5. Embed all documents once  (`main.py` lines 57–58)

```python
embed, EMBEDDER_NAME = build_embedder(DOCS)
doc_vectors = embed(DOCS)                                  # shape: (n_docs, dim)
```

- **In simple words:** Convert every document to a vector one time up front.
- **Exactly:** `embed, EMBEDDER_NAME = build_embedder(DOCS)`; `doc_vectors.shape == (6, 49)` in the run.
- **Why it is here:** Embedding documents at question time would be slow.

### 6. Vector search by cosine similarity  (`main.py` lines 63–68)

```python
def retrieve(question, k=2, min_score=0.0):
    q = embed([question])[0]                               # shape: (dim,)
    norms = np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(q) + 1e-9   # shape: (n_docs,)
    scores = doc_vectors @ q / norms                       # cosine per document, shape: (n_docs,)
    best = np.argsort(scores)[::-1][:k]                    # indices of the k highest scores
    return [(DOCS[i], float(scores[i])) for i in best if scores[i] >= min_score]
```

- **In simple words:** Score every document against the question, keep the best few above a threshold.
- **Exactly:** `q` shape `(dim,)`; `norms` shape `(n_docs,)`; `scores = doc_vectors @ q / norms` is the cosine; `np.argsort(scores)[::-1][:k]` gives top-k indices; `+ 1e-9` avoids division by zero (all-zero vectors). Returns `[(doc, float score)]` with `score >= min_score`.
- **Why it is here:** This is the whole retrieval step; no vector database needed for 6 documents.

### 7. Canned hallucinations  (`main.py` lines 75–77)

```python
GUESSES = {"refund": "Refunds are usually accepted within 30 days.",
           "support": "Support is available 24/7 by phone.",
           "ship": "Delivery usually takes 2 days."}
```

- **In simple words:** What the toy model "believes" when it has no context.
- **Exactly:** Dict keyword to guess: refund 30 days, support 24/7, ship 2 days.
- **Why it is here:** Gives deterministic, wrong, confident answers to contrast with grounded ones.

### 8. The toy frozen model  (`main.py` lines 79–93)

```python
def frozen_model(messages, system):
    prompt = messages[-1]["content"]
    question = prompt.rsplit("Question:", 1)[1].strip().lower()
    if "Context:" not in prompt:
        for key, guess in GUESSES.items():
            if key in question:
                return guess
        return "Yes, that is generally possible."
    context = prompt.split("Context:")[1].split("Question:")[0]
    stop = {"the", "a", "an", "are", "is", "what", "do", "i", "for", "of", "how", "you", "to", "have", "many", "can"}
    words = set(question.replace("?", "").split()) - stop        # content words only
    sentences = [s.strip(" -") for s in context.replace("\n", " ").split(".") if s.strip(" -")]
    if not sentences:
        return "I don't know based on the provided context."
    return max(sentences, key=lambda s: len(words & set(s.lower().split()))) + "."
```

- **In simple words:** Without context it guesses; with context it repeats the best-matching sentence.
- **Exactly:** `frozen_model(messages, system)`: no `Context:` in prompt gives a guess from `GUESSES` (or `"Yes, that is generally possible."`). With context it removes stop words, splits context into sentences and returns the sentence with most word overlap with the question. This is a toy, not an LLM.
- **Why it is here:** Lets you see the effect of the prompt without a real model.

### 9. Wrap the toy as a MockLLM  (`main.py` lines 95–95)

```python
LLM = MockLLM(rules=[(r"", frozen_model)])
```

- **In simple words:** Plug the function into our shared mock so `generate` can call it.
- **Exactly:** `MockLLM(rules=[(r"", fn)])`: empty regex matches any prompt, and the reply may be a callable.
- **Why it is here:** Keeps the same `generate()` interface as a real backend.

### 10. The ask function: retrieve, build prompt, answer  (`main.py` lines 97–108)

```python
def ask(question, use_rag, k=2, min_score=0.0):
    if not use_rag:
        prompt = f"Question: {question}"
    else:
        hits = retrieve(question, k, min_score)
        context = "\n".join(f"- {text}" for text, _ in hits)      # inject retrieved chunks into the prompt
        prompt = f"Context:\n{context}\n\nQuestion: {question}" if hits else f"Question: {question}"
        print(f"   retrieved: {[(t[:28] + '...', round(s, 2)) for t, s in hits]}")
        if not hits:
            return "I don't know: nothing relevant in the knowledge base."
    system = "Answer ONLY from the context. If it is not there, say you don't know."
    return generate([{"role": "user", "content": prompt}], system=system, mock=LLM)
```

- **In simple words:** With RAG on: search, paste the hits into the prompt as context, and ask. Otherwise ask directly.
- **Exactly:** `context` is a bullet list `"- chunk"`. Prompt is `Context:\n...\n\nQuestion: ...`. If no hits above `min_score`, return "I don't know" without calling the model. The system prompt says answer ONLY from the context. Debug line prints top hits with rounded scores.
- **Why it is here:** Injecting context is the "augmented" in RAG; the threshold prevents answering from junk.

### 11. Compare without and with retrieval  (`main.py` lines 110–117)

```python
if __name__ == "__main__":
    print(f"Embedder: {EMBEDDER_NAME} | {len(DOCS)} documents -> vectors {doc_vectors.shape}\n")

    for q in ["How many days do I have for a refund?", "What are the support hours?"]:
        print(f"Q: {q}")
        print("   WITHOUT retrieval:", ask(q, use_rag=False))
        print("   WITH retrieval   :", ask(q, use_rag=True))
        print()
```

- **In simple words:** Ask two questions both ways.
- **Exactly:** Real output: refund question: without `Refunds are usually accepted within 30 days.`; with: the 9-day sentence (top score 0.3). Support hours: `24/7` vs `Monday to Saturday, 10am to 6pm IST` (score 0.32).
- **Why it is here:** Shows RAG fixing a hallucination.

### 12. Failure mode 1: retrieval miss  (`main.py` lines 119–122)

```python
    print("--- Failure mode 1: retrieval MISS (question uses different words than the document) ---")
    q = "What is the money back window?"
    print(f"Q: {q}")
    print("   WITH retrieval   :", ask(q, use_rag=True))
```

- **In simple words:** Ask with different words than the document ("money back" vs "refund").
- **Exactly:** All scores are 0.0 because TF-IDF matches only identical words; the top-2 are arbitrary and the answer is the compost sentence.
- **Why it is here:** Why dense embeddings, query rewriting or hybrid search exist.

### 13. Failure mode 2: irrelevant chunk and threshold fix  (`main.py` lines 124–129)

```python
    print("\n--- Failure mode 2: irrelevant chunk (the answer is NOT in the knowledge base) ---")
    q = "Do you sell gift cards?"
    print(f"Q: {q}")
    print("   WITH retrieval   :", ask(q, use_rag=True))
    print("   same, but with a similarity threshold (min_score=0.2):")
    print("   ->", ask(q, use_rag=True, min_score=0.2))
```

- **In simple words:** Ask something not in the knowledge base; with `min_score=0.2` nothing is retrieved and it admits ignorance.
- **Exactly:** Without threshold it returns an unrelated compost answer (scores 0.0). With `min_score=0.2`: `retrieved: []` then `I don't know: nothing relevant in the knowledge base.`
- **Why it is here:** Retrieval always returns something unless you set a threshold.

### 14. TRY IT / BREAK IT  (`main.py` lines 131–135)

```python
# TRY IT: change k=2 to k=1, then k=6 (everything) - watch the answer quality and prompt size change.
# TRY IT: add a document "Gift cards are sold in 500 and 1000 rupee values." and re-ask the gift card question.
# TRY IT: RAG_EMBEDDER=minilm python main.py - the money-back question may now find the refund document.
# BREAK IT: embed with one vectorizer and query with another -> shapes differ:
#           doc_vectors @ q  ->  ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
```

- **In simple words:** Experiments.
- **Exactly:** Mixing vectorizers makes shapes differ and `@` raises `ValueError`.
- **Why it is here:** Doc and query must be embedded with the same model.

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

**Deep dive:** every stage of this 20-line demo (chunking, embeddings, vector indexes, BM25 and hybrid search, re-ranking, evaluation) is expanded step by step in [14_rag_and_retrieval](../../14_rag_and_retrieval/README.md).

> **Say it aloud (interview box):** RAG means I retrieve the few documents most similar to the question, using embeddings and cosine similarity, and paste them into the prompt so a frozen model answers from facts instead of guessing. Its failure modes are retrieval misses and irrelevant chunks, so I set a similarity threshold and allow an explicit "I don't know". It fixes missing knowledge without retraining, but it only lets the model talk, not act.
