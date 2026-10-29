# Terms — Embeddings with a Bi-encoder

Folder: `14_rag_and_retrieval/03_embeddings_bi_encoder` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [14 README](../README.md)

**Terms in this folder:** [Embedding](#embedding) · [Bi-encoder](#bi-encoder) · [Cross-encoder (contrast)](#cross-encoder-contrast) · [all-MiniLM-L6-v2](#all-minilm-l6-v2) · [AutoModel and last_hidden_state](#automodel-and-last_hidden_state) · [Tokenizer call: padding, truncation, max_length](#tokenizer-call-padding-truncation-max_length) · [Attention mask](#attention-mask) · [Mean pooling](#mean-pooling) · [CLS pooling](#cls-pooling) · [unsqueeze and broadcasting](#unsqueeze-and-broadcasting) · [L2 normalisation (F.normalize)](#l2-normalisation-fnormalize) · [Cosine similarity and dot product](#cosine-similarity-and-dot-product) · [torch.no_grad](#torchno_grad) · [Contrastive learning and InfoNCE](#contrastive-learning-and-infonce) · [In-batch and hard negatives](#in-batch-and-hard-negatives) · [Dense vs sparse retrieval](#dense-vs-sparse-retrieval) · [TfidfVectorizer](#tfidfvectorizer) · [Truncation](#truncation) · [Symmetric vs asymmetric embedding](#symmetric-vs-asymmetric-embedding) · [Query and passage prefixes (e5)](#query-and-passage-prefixes-e5) · [Embedding dimension and memory](#embedding-dimension-and-memory) · [Matryoshka embeddings](#matryoshka-embeddings) · [MTEB and choosing a model](#mteb-and-choosing-a-model) · [Same model for index and query](#same-model-for-index-and-query) · [np.argsort for top-k](#npargsort-for-top-k)

---

## Embedding  `(concept)`
**1. Definition:** An embedding is a fixed-length vector of real numbers produced by a model so that semantically similar inputs map to nearby vectors.
**2. In plain words:** Each chunk becomes 384 numbers. "How long can I wait before asking for my money back?" and the refund rule share no words but score 0.46 with MiniLM, while TF-IDF gives 0.00.
**3. Example to quote:** > "An embedding turns text into a point in space where distance means difference in meaning."
**Seen in:** `main.py`, sections 1 and 2

## Bi-encoder  `(concept)`
**1. Definition:** A bi-encoder encodes the query and each document independently with the same (or twin) encoder and scores them by vector similarity.
**2. In plain words:** Because documents never need the query, all 36 chunk vectors are computed once, offline. At query time: one encoder pass plus `matrix @ qv`.
**3. Example to quote:** > "Bi-encoders make retrieval scale: precompute document vectors once, then each query is one encode and a dot product per document."
**Seen in:** `main.py`, `embed()`; `common.Embedder`

## Cross-encoder (contrast)  `(concept)`
**1. Definition:** A cross-encoder feeds the query and passage together into one transformer and outputs a single relevance score.
**2. In plain words:** More accurate (it compares words directly) but must run once per (query, passage) pair, so it cannot search millions of documents; it re-ranks a short list. Covered in step 07.
**3. Example to quote:** > "Bi-encoder to retrieve the top 50, cross-encoder to re-rank them."
**Seen in:** `WALKTHROUGH.md`, theory

## all-MiniLM-L6-v2  `(third-party: sentence-transformers model)`
**1. Definition:** all-MiniLM-L6-v2 is a 6-layer MiniLM sentence-embedding model from sentence-transformers that outputs 384-dimensional mean-pooled vectors.
**2. In plain words:** Small (22.7M parameters), fast on CPU, English-focused. Truncates at 256 tokens by default and was trained on sequences of up to 128. It scored a Hindi refund question against the English rule at -0.031.
**3. Example to quote:** > "MiniLM is my CPU-friendly baseline: 384 dimensions, 22.7 million parameters, mean pooling."
**Seen in:** `common.EMBED_MODEL`, `main.py` first line of output

## AutoModel and last_hidden_state  `(third-party: transformers)`
**1. Definition:** `AutoModel.from_pretrained(name)` loads the bare transformer encoder without a task head; its output's `last_hidden_state` is the final layer's vector for every token, shape `(batch, tokens, hidden)`.
**2. In plain words:** For two short sentences we got `(2, 15, 384)`: one 384-vector per token. Pooling turns that into one vector per sentence. `.eval()` switches off dropout.
**3. Example to quote:** > "The encoder gives me token vectors; the sentence embedding is a pooling choice on top."
**Seen in:** `main.py`, `embed()`

## Tokenizer call: padding, truncation, max_length  `(third-party: transformers)`
**1. Definition:** `tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")` converts a list of texts into equal-length PyTorch id tensors plus an attention mask, cutting anything past `max_length`.
**2. In plain words:** Padding makes the batch rectangular; truncation enforces the model's limit. Both are silent.
**3. Example to quote:** > "I set truncation explicitly and keep chunks shorter than max_length so nothing is cut."
**Seen in:** `main.py`, `embed()`, `e5_embed()`

## Attention mask  `(concept)`
**1. Definition:** The attention mask is a 0/1 tensor marking which positions are real tokens (1) and which are padding (0).
**2. In plain words:** Without it, mean pooling would average in padding vectors and a short sentence's embedding would depend on how long its batch neighbours were.
**3. Example to quote:** > "I multiply token vectors by the mask before averaging so padding counts for nothing."
**Seen in:** `main.py`, `embed()`

## Mean pooling  `(concept)`
**1. Definition:** Mean pooling averages the token vectors of a text, weighted by the attention mask, to produce one sentence vector.
**2. In plain words:** `(hidden * mask).sum(1) / mask.sum(1)`: sum of real-token vectors divided by the number of real tokens. It is how all-MiniLM-L6-v2 and e5 were trained.
**3. Example to quote:** > "I use the pooling the model was trained with; for MiniLM that is masked mean pooling."
```python
mask = batch["attention_mask"].unsqueeze(-1).float()
pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
```
**Seen in:** `main.py`, `embed()`

## CLS pooling  `(concept)`
**1. Definition:** CLS pooling uses the vector of the first token (`[CLS]`) as the sentence embedding.
**2. In plain words:** Some models (many bge variants) are trained for it. Used on MiniLM it gave 0.77, 0.82, 0.91, 0.69, 0.60 for the five pairs; even the unrelated pair looked similar. Wrong pooling does not crash, it squashes the scale.
**3. Example to quote:** > "Switching MiniLM to CLS pooling made an unrelated pair score 0.60 instead of 0.07."
**Seen in:** `main.py`, TRY IT; `WALKTHROUGH.md`, Experiments

## unsqueeze and broadcasting  `(syntax)`
**1. Definition:** `tensor.unsqueeze(-1)` adds a size-1 axis at the end; broadcasting then stretches size-1 axes to match the other operand in element-wise operations.
**2. In plain words:** The mask `(2, 15)` becomes `(2, 15, 1)` so it can multiply hidden states `(2, 15, 384)`, zeroing all 384 numbers of every padding token.
**3. Example to quote:** > "`unsqueeze(-1)` lets a per-token mask multiply per-token vectors."
**Seen in:** `main.py`, `embed()`

## L2 normalisation (F.normalize)  `(third-party: torch)`
**1. Definition:** `torch.nn.functional.normalize(x, dim=1)` divides each row by its Euclidean (L2) norm so every row has length 1.
**2. In plain words:** The run printed `norms [1. 1.]`. Without it, raw vectors had lengths 4.65 to 7.28 and dot products would mix length with direction.
**3. Example to quote:** > "I normalise once at indexing time, so cosine similarity becomes a plain dot product."
**Seen in:** `main.py`, `embed()`

## Cosine similarity and dot product  `(concept)`
**1. Definition:** Cosine similarity is `a·b / (|a||b|)`, the cosine of the angle between two vectors, from -1 to 1; for unit vectors it equals the dot product.
**2. In plain words:** Scores in this step range from -0.031 to 0.60 for MiniLM. Only relative order matters for retrieval; a 0.46 is not "46% relevant".
**3. Example to quote:** > "With unit vectors, `matrix @ query` scores every chunk in one matrix multiplication."
**Seen in:** `main.py`, sections 2 and 5

## torch.no_grad  `(third-party: torch)`
**1. Definition:** `torch.no_grad()` is a context manager that disables gradient tracking for the operations inside it.
**2. In plain words:** We only run the model forward, so storing information for backpropagation would waste memory and time.
**3. Example to quote:** > "All embedding is inference, so it runs under `torch.no_grad()`."
**Seen in:** `main.py`, `embed()`, `e5_embed()`

## Contrastive learning and InfoNCE  `(concept)`
**1. Definition:** Contrastive learning trains an encoder to give matching pairs high similarity and non-matching pairs low similarity; InfoNCE is the softmax cross-entropy loss `L = -log( exp(sim(q,d+)/τ) / Σ_j exp(sim(q,d_j)/τ) )` with temperature τ.
**2. In plain words:** For each question, pick out the correct passage from a line-up. The model learns by being punished whenever a wrong passage scores higher than the right one.
**3. Example to quote:** > "Bi-encoders are trained contrastively: pull the query towards its positive passage, push it away from negatives, with an InfoNCE loss."
**Seen in:** `WALKTHROUGH.md`, theory

## In-batch and hard negatives  `(concept)`
**1. Definition:** In-batch negatives are the other examples' positives in the same training batch, used as negatives; hard negatives are deliberately chosen passages that look relevant but are not.
**2. In plain words:** In-batch negatives are free (bigger batch, more negatives). Hard negatives, such as "Shipping is issued within 9 days" for a refund question, teach the fine distinctions this model still misses (that pair scored 0.56).
**3. Example to quote:** > "Hard negatives mined with BM25 are what teach an embedding model the difference between refunds and shipping."
**Seen in:** `WALKTHROUGH.md`, theory

## Dense vs sparse retrieval  `(concept)`
**1. Definition:** Dense retrieval uses learned low-dimensional vectors with mostly non-zero values; sparse retrieval (TF-IDF, BM25) uses vocabulary-sized vectors that are non-zero only for words present in the text.
**2. In plain words:** Dense: 0.46 on a paraphrase with no shared words (TF-IDF 0.00). Sparse: better at exact codes and names. Each covers the other's blind spot, so production systems often combine them (step 06).
**3. Example to quote:** > "Dense finds meaning, sparse finds exact words; hybrid search uses both."
**Seen in:** `main.py`, section 2

## TfidfVectorizer  `(third-party: scikit-learn)`
**1. Definition:** `sklearn.feature_extraction.text.TfidfVectorizer` converts texts into TF-IDF vectors: term frequency weighted by inverse document frequency.
**2. In plain words:** `.fit()` learns the vocabulary from the 10 pair texts; `.transform()` turns a text into a sparse vector; `.toarray()` makes it a normal array for the cosine.
**3. Example to quote:** > "TF-IDF scored 'Which teas will not keep me awake?' vs 'Chamomile Calm is caffeine free.' at 0.00 because they share no words."
**Seen in:** `main.py`, section 2

## Truncation  `(concept)`
**1. Definition:** Truncation drops all tokens beyond the encoder's maximum input length before encoding.
**2. In plain words:** With ~600 tokens of filler and the answer at the end, cosine(with answer, without answer) = 1.0000: the model never saw the last sentence. No warning was printed.
**3. Example to quote:** > "Embedding models silently ignore text past their limit, so I chunk before I embed."
**Seen in:** `main.py`, section 3

## Symmetric vs asymmetric embedding  `(concept)`
**1. Definition:** Symmetric tasks compare texts of the same kind (sentence to sentence); asymmetric tasks compare different kinds, typically a short query with a longer passage.
**2. In plain words:** RAG is asymmetric. Models built for it may mark which side is which so they can embed a question and an answer passage close together.
**3. Example to quote:** > "Duplicate detection is symmetric; question-to-passage retrieval is asymmetric."
**Seen in:** `WALKTHROUGH.md`, theory

## Query and passage prefixes (e5)  `(concept)`
**1. Definition:** Some embedding models are trained with role prefixes (e5: `query: ` and `passage: `; others use instructions or `search_query:` / `search_document:`) and expect the same prefixes at inference.
**2. In plain words:** With multilingual-e5-small the gap between the refund and the shipping passage went from 0.027 without prefixes to 0.032 with them, even though both raw scores fell. Small single example, but the direction is the point: follow the model card.
**3. Example to quote:** > "I use the prefix format the model was trained with, identically at indexing and query time."
**Seen in:** `main.py`, section 4

## Embedding dimension and memory  `(concept)`
**1. Definition:** Index memory is number of vectors x dimension x bytes per value.
**2. In plain words:** 36 x 384 x 4 bytes = 55,296 bytes. 1M chunks would need 1.54 GB in float32. Halve it with float16; quantise further with int8 or binary codes.
**3. Example to quote:** > "At 384 dimensions in float32, a million chunks is about 1.5 GB before any index overhead."
**Seen in:** `main.py`, section 5

## Matryoshka embeddings  `(concept)`
**1. Definition:** Matryoshka representation learning trains an embedding so that its leading dimensions (for example the first 64, 128 or 256) also form usable embeddings on their own.
**2. In plain words:** You can store a shorter prefix of each vector to save memory, trading a little accuracy, without retraining. Only works for models trained this way.
**3. Example to quote:** > "With a Matryoshka model I can truncate vectors to a quarter of their size for a first-pass search."
**Seen in:** `WALKTHROUGH.md`, theory

## MTEB and choosing a model  `(concept)`
**1. Definition:** MTEB (Massive Text Embedding Benchmark) is a public benchmark and leaderboard that scores embedding models on many tasks, including retrieval.
**2. In plain words:** Use the retrieval columns to shortlist, then check domain, languages, dimension, max length, latency, licence and prefix requirements, and finally test the shortlist on your own evaluation set.
**3. Example to quote:** > "MTEB shortlists models; my own labelled questions pick the winner."
**Seen in:** `WALKTHROUGH.md`, theory

## Same model for index and query  `(concept)`
**1. Definition:** Query vectors are only comparable with document vectors produced by the same model (and preprocessing); vectors from different models live in unrelated spaces.
**2. In plain words:** A MiniLM query against e5 passages gave 0.088 and 0.080, with no error, because both are 384-dimensional. Changing the model means re-embedding the whole corpus.
**3. Example to quote:** > "I store the model name and version with the index and refuse to query it with any other model."
**Seen in:** `main.py`, BREAK IT; `WALKTHROUGH.md`, Debugging practice

## np.argsort for top-k  `(third-party: numpy)`
**1. Definition:** `np.argsort(a)` returns the indices that would sort `a` in ascending order; `np.argsort(-scores)[:k]` gives the indices of the k largest scores.
**2. In plain words:** It sorts all 36 scores to pick the best 3. Fine for 36; for millions you want `argpartition` or an ANN index (steps 04 and 05).
**3. Example to quote:** > "`np.argsort(-(matrix @ qv))[:3]` returned `allergens_and_ingredients#Caffeine` first for the caffeine question."
**Seen in:** `main.py`, section 5
