"""
common.py -- small helpers shared by the steps of module 14 (RAG and retrieval).

Every helper here is TAUGHT first, line by line, inside one step folder. Later steps import the
finished version from here so they can focus on the next idea:

    load_documents, split_by_heading   -> taught in 01 and 02
    Embedder                           -> taught in 03
    dense_search                       -> taught in 04
    BM25, rrf_fuse                     -> taught in 06
    CrossEncoderReranker               -> taught in 07
    load_eval_set                      -> used in 06, 07, 09

Everything runs offline on a CPU. Models come from the local Hugging Face cache.
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")          # never touch the network; use cached models only
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODULE_DIR = Path(__file__).resolve().parent
KB_DIR = MODULE_DIR / "data" / "kb"                   # the knowledge base: 9 markdown files
EVAL_FILE = MODULE_DIR / "data" / "eval_questions.json"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim bi-encoder, 22M parameters
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # cross-encoder trained on MS MARCO


# ---------------------------------------------------------------------------------------------
# Loading and splitting (steps 01, 02)
# ---------------------------------------------------------------------------------------------
def load_documents(kb_dir=KB_DIR):
    """Read every .md file -> list of {"doc_id", "title", "text"} sorted by doc_id."""
    docs = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip()   # first line is "# Title"
        docs.append({"doc_id": path.stem, "title": title, "text": text})
    return docs


def split_by_heading(doc):
    """One chunk per '## Section'. chunk_id = 'doc_id#Section'. The doc title is kept as metadata."""
    chunks = []
    for block in re.split(r"\n(?=## )", doc["text"]):         # split right before every '## '
        lines = block.strip().splitlines()
        if not lines or not lines[0].startswith("## "):
            continue                                           # skip the '# Title' block
        section = lines[0][3:].strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        chunks.append({"chunk_id": f"{doc['doc_id']}#{section}", "doc_id": doc["doc_id"],
                       "title": doc["title"], "section": section, "text": body})
    return chunks


def build_chunks():
    """All section chunks of the whole knowledge base."""
    return [c for d in load_documents() for c in split_by_heading(d)]


def chunk_for_embedding(chunk):
    """Prepend the document title and section heading ("contextual header") before embedding."""
    return f"{chunk['title']} - {chunk['section']}: {chunk['text']}"


# ---------------------------------------------------------------------------------------------
# Dense embeddings (step 03) and search (step 04)
# ---------------------------------------------------------------------------------------------
class Embedder:
    """Bi-encoder: text -> one L2-normalised vector. Mean pooling over tokens, like sentence-transformers."""

    def __init__(self, model_name=EMBED_MODEL, max_length=256):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()
        self.max_length = max_length
        self.dim = self.model.config.hidden_size

    def encode(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
            with self.torch.no_grad():
                hidden = self.model(**batch).last_hidden_state          # (b, tokens, dim)
            mask = batch["attention_mask"].unsqueeze(-1).float()       # (b, tokens, 1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)              # mean over real tokens -> (b, dim)
            pooled = self.torch.nn.functional.normalize(pooled, dim=1)  # unit length -> dot = cosine
            out.append(pooled.numpy())
        return np.vstack(out).astype(np.float32)                       # (n_texts, dim)


def dense_search(query_vec, matrix, k=5):
    """Exact (brute-force) top-k by dot product. Rows of `matrix` and `query_vec` are unit vectors."""
    scores = matrix @ query_vec                                        # (n,)
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]                          # k best, unordered, O(n)
    top = top[np.argsort(-scores[top])]                                # order only those k
    return [(int(i), float(scores[i])) for i in top]


# ---------------------------------------------------------------------------------------------
# Sparse retrieval and fusion (step 06)
# ---------------------------------------------------------------------------------------------
STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
my of on or our the this to was we what when where which who will with you your me am""".split())


def tokenize(text):
    """Lower-case word tokens; keeps codes like 'ck-masala-250' as one token; drops stopwords."""
    return [t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a list of texts. k1 = term-frequency saturation, b = length normalisation."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = self.lengths.mean()
        n = len(texts)
        df = Counter(term for d in self.docs for term in d)            # in how many docs each term appears
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        s = np.zeros(len(self.docs))
        for term in tokenize(query):
            if term not in self.idf:
                continue
            for i, d in enumerate(self.docs):
                tf = d.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    s[i] += self.idf[term] * tf * (self.k1 + 1) / norm
        return s

    def search(self, query, k=5):
        s = self.scores(query)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top]


def rrf_fuse(rankings, k=60, top_n=5):
    """Reciprocal Rank Fusion: score(doc) = sum over rankings of 1 / (k + rank). Ranks start at 1."""
    fused = Counter()
    for ranking in rankings:                                           # each ranking: list of (index, score)
        for rank, (idx, _) in enumerate(ranking, start=1):
            fused[idx] += 1.0 / (k + rank)
    return [(idx, score) for idx, score in fused.most_common(top_n)]


# ---------------------------------------------------------------------------------------------
# Re-ranking (step 07)
# ---------------------------------------------------------------------------------------------
class CrossEncoderReranker:
    """Reads (query, passage) TOGETHER and outputs one relevance logit. Slower but more accurate."""

    def __init__(self, model_name=RERANK_MODEL):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    def score(self, query, passages):
        batch = self.tok([query] * len(passages), passages, padding=True, truncation=True,
                         max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            return self.model(**batch).logits.squeeze(-1).numpy()      # (n_passages,)


# ---------------------------------------------------------------------------------------------
# Evaluation data (steps 06, 07, 09)
# ---------------------------------------------------------------------------------------------
def load_eval_set():
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))
