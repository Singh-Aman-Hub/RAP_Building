# Terms — Vectors and Similarity

Folder: `04_llm_fundamentals/02_vectors_and_similarity` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [Vector](#vector) · [Sparse vector](#sparse-vector) · [Dense vector](#dense-vector) · [TF-IDF](#tf-idf) · [Cosine similarity](#cosine-similarity) · [Co-occurrence matrix](#co-occurrence-matrix) · [SVD](#svd) · [Distributional hypothesis](#distributional-hypothesis) · [Stop words](#stop-words) · [Out-of-vocabulary](#out-of-vocabulary) · [Context-free vector](#context-free-vector)

---

## Vector
**1. Definition:** An ordered list of numbers, treated as a point or arrow in space.
**2. In plain words:** The only form of text a neural network can compute with. `[-1.96, 0, -1.47, 0]` is a 4-dimensional vector.
**3. Example to quote:** > "I treat each word as a vector like `[-1.96, 0, -1.47, 0]` because a network can only compute with numbers."

## Sparse vector
**1. Definition:** A vector with one slot per vocabulary word, almost all zeros.
**2. In plain words:** TF-IDF vectors. Cheap and interpretable (each slot is a word) but huge and blind to meaning: *car* and *automobile* are different slots.
**3. Example to quote:** > "TF-IDF vectors are sparse, one slot per word, so car and automobile land in different slots and look unrelated."

## Dense vector
**1. Definition:** A short vector (tens to thousands of numbers) where every slot is generally non-zero and slots have no individual meaning.
**2. In plain words:** The format of embeddings. Related words get similar numbers. Limitation: you cannot read what slot 17 "means".
**3. Example to quote:** > "A dense vector is a short list of numbers with no readable slots, but related words get similar numbers."

## TF-IDF
**1. Definition:** Term frequency x inverse document frequency: a word's weight rises with how often it appears in a document and falls with how many documents contain it.
**2. In plain words:** Rare, informative words count more than "the". You used it in `03_ml_foundations/08`.
**3. Example to quote:** > "TF-IDF weights a word up for frequency in a document and down for appearing in many documents, so 'the' counts for little."

## Cosine similarity
**1. Definition:** `a . b / (|a| |b|)`, the cosine of the angle between two vectors.
**2. In plain words:** 1 = same direction, 0 = unrelated, -1 = opposite; ignores length. Used for embedding search and (as a scaled dot product) inside attention.
**4. Interview angle:** "Why cosine and not Euclidean distance?" -> it compares direction/topic regardless of vector length.
**3. Example to quote:** > "I compare vectors with cosine similarity because it measures direction, so a long and a short document on the same topic still match."

## Co-occurrence matrix
**1. Definition:** A table counting how often each pair of words appears together (here: in the same sentence).
**2. In plain words:** Raw material for "company a word keeps". Each row is already a (sparse, large) vector for that word.
**3. Example to quote:** > "I counted which words share a sentence in a co-occurrence matrix, so each row is a first vector for that word."

## SVD
**1. Definition:** Singular value decomposition: factors a matrix into directions ordered by importance, so you can keep only the top k.
**2. In plain words:** A compressor. Keeping the top 4 directions of the co-occurrence table gives each word 4 dense numbers. Real embeddings are learned by training instead, but serve the same purpose.
**3. Example to quote:** > "I used SVD to squeeze the co-occurrence table down to 4 numbers per word, which is what embeddings do after training."

## Distributional hypothesis
**1. Definition:** Words that occur in similar contexts have similar meanings.
**2. In plain words:** "You shall know a word by the company it keeps." The idea behind both this toy and real embeddings.
**3. Example to quote:** > "Words that keep the same company get similar vectors, which is why car and automobile ended up identical in my toy."

## Stop words
**1. Definition:** Very common filler words ("the", "a") that carry little topic information.
**2. In plain words:** Removed here so they do not make every word look similar. TF-IDF down-weights them automatically.
**3. Example to quote:** > "I dropped 'the', 'a' and 'an' because they appear everywhere and would make every word look similar."

## Out-of-vocabulary
**1. Definition:** A word the model has no vector for.
**2. In plain words:** Word-level vocabularies fail on new words; sub-word tokenisation (step 01) avoids it.
**3. Example to quote:** > "A word-level vocabulary fails on unseen words like 'zebra', which sub-word tokenisation avoids."

## Context-free vector
**1. Definition:** A word vector that is the same wherever the word appears.
**2. In plain words:** *bank* has one vector for river and money. Attention (step 04) is how a transformer makes vectors depend on context.
**3. Example to quote:** > "A context-free vector gives 'bank' the same numbers for river and money, which attention fixes."
