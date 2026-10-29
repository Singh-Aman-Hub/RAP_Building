"""
GOAL:  Turn text into meaning vectors with a bi-encoder (all-MiniLM-L6-v2): tokens -> hidden states ->
       mean pooling -> L2 normalisation.  Compare with TF-IDF on paraphrases, see truncation, and see why
       some models need "query:" / "passage:" prefixes.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 14_rag_and_retrieval/03_embeddings_bi_encoder/main.py
EXPECTED OUTPUT (shortened):
    hidden states (2, 15, 384) -> pooled (2, 384) -> norms [1. 1.]
    pair                                         MiniLM  TF-IDF
    paraphrase, no shared words                     0.46    0.0x
NEEDS: torch, transformers, scikit-learn; cached all-MiniLM-L6-v2 and multilingual-e5-small.  ~10 s.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EMBED_MODEL, build_chunks, chunk_for_embedding   # noqa: E402  (also sets HF offline)

tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
model = AutoModel.from_pretrained(EMBED_MODEL).eval()
print(f"model: {EMBED_MODEL} | {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | dim {model.config.hidden_size}")


# ---- 1. The pipeline, one tensor at a time ------------------------------------------------------------
def embed(texts, max_length=256, verbose=False):
    batch = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state                  # (n_texts, n_tokens, 384): one vector PER TOKEN
    mask = batch["attention_mask"].unsqueeze(-1).float()           # (n_texts, n_tokens, 1): 0 on padding
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)          # mean over REAL tokens -> (n_texts, 384)
    unit = torch.nn.functional.normalize(pooled, dim=1)            # length 1 -> dot product == cosine
    if verbose:
        print(f"hidden states {tuple(hidden.shape)} -> pooled {tuple(pooled.shape)} -> norms {unit.norm(dim=1).numpy().round(3)}")
    return unit.numpy()


embed(["Refunds are issued within 9 days.", "Shipping is free above 799 rupees for every order."], verbose=True)

# ---- 2. Meaning vs words: MiniLM against TF-IDF ---------------------------------------------------------
refund = "Refunds are issued within 9 days of delivery if the seal is unbroken."
pairs = [
    ("How long can I wait before asking for my money back?", refund, "paraphrase, no shared words"),
    ("How many days do I have for a refund?", refund, "paraphrase, shared words"),
    ("Refunds are issued within 9 days.", "Shipping is issued within 9 days.", "same words, different meaning"),
    ("Which teas will not keep me awake?", "Chamomile Calm is caffeine free.", "needs world knowledge"),
    ("How do I brew green tea?", "Our office is closed on Sundays.", "unrelated"),
]
tfidf = TfidfVectorizer().fit([t for p in pairs for t in p[:2]])
print(f"\n{'pair':<48}{'MiniLM':>8}{'TF-IDF':>8}")
for a, b, label in pairs:
    dense = float(embed([a])[0] @ embed([b])[0])
    va, vb = tfidf.transform([a]).toarray()[0], tfidf.transform([b]).toarray()[0]
    sparse = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
    print(f"{label:<48}{dense:>8.2f}{sparse:>8.2f}")
print("Dense vectors score paraphrases and world knowledge; TF-IDF needs shared words (rows 1 and 4 score 0).")
print("Row 3: dense similarity stays high for 'refunds' vs 'shipping' -> embeddings blur exact details.")

# ---- 3. Truncation: text beyond max_length is silently ignored --------------------------------------
filler = "Our teas are packed with care in Pune. " * 60                  # ~600 tokens of filler
tail = " The refund window is 9 days."
v_full = embed([filler + tail], max_length=256)[0]
v_without = embed([filler], max_length=256)[0]
print(f"\ntruncation: cosine(text WITH the answer at the end, text WITHOUT it) = {float(v_full @ v_without):.4f}")
print("1.0000 means the model never saw the last sentence. Chunk BEFORE embedding; keep chunks short.")

# ---- 4. Asymmetric models want prefixes (e5, bge, nomic ...) -----------------------------------------
e5_name = "intfloat/multilingual-e5-small"
e5_tok, e5 = AutoTokenizer.from_pretrained(e5_name), AutoModel.from_pretrained(e5_name).eval()


def e5_embed(texts):
    b = e5_tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        h = e5(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1).numpy()


q = "What is the money back window?"
passages = [refund, "Shipping is free on orders above 799 rupees."]
plain = e5_embed([q] + passages)
prefixed = e5_embed(["query: " + q] + ["passage: " + p for p in passages])
print(f"\ne5 without prefixes: refund {plain[0] @ plain[1]:.3f} | shipping {plain[0] @ plain[2]:.3f} "
      f"| gap {plain[0] @ plain[1] - plain[0] @ plain[2]:.3f}")
print(f"e5 with prefixes   : refund {prefixed[0] @ prefixed[1]:.3f} | shipping {prefixed[0] @ prefixed[2]:.3f} "
      f"| gap {prefixed[0] @ prefixed[1] - prefixed[0] @ prefixed[2]:.3f}")
print("Use the format the model was trained with; mixing formats between indexing and querying hurts.")

# ---- 5. Embed the whole knowledge base -----------------------------------------------------------------
chunks = build_chunks()
matrix = embed([chunk_for_embedding(c) for c in chunks])               # (36, 384)
print(f"\nknowledge base: {len(chunks)} chunks -> matrix {matrix.shape}, {matrix.nbytes:,} bytes (float32)")
print(f"1M chunks would need {1_000_000 * matrix.shape[1] * 4 / 1e9:.2f} GB of float32 vectors")
for question in ["Which teas will not keep me awake at night?", "Can you send tea to my cousin in London?"]:
    qv = embed([question])[0]                                            # (384,)
    print(f"Q: {question}")
    for i in np.argsort(-(matrix @ qv))[:3]:
        print(f"  {float(matrix[i] @ qv):.3f}  {chunks[i]['chunk_id']}")
print("The second question misses 'International shipping' (the chunk says 'outside India', the query says 'London').")
print("Fixes come later: hybrid search (06), re-ranking (07), query rewriting (10).")

# TRY IT: in embed(), replace mean pooling by the first token only: pooled = hidden[:, 0]. Do the scores change?
# TRY IT: remove the normalize line and print the vector norms: they differ (4.65 to 7.28 in one run), so raw dot
#         products mix 'how similar' with 'how long the vector is'. Normalising removes that.
# TRY IT: add your own pair to `pairs`, e.g. a question in Hindi vs the English refund rule (use e5_embed: it is multilingual).
# BREAK IT: compare a MiniLM vector with an e5 vector (both 384-dim!). No error, but the score is meaningless:
#           different models live in different vector spaces, so the index and the query MUST use the same model.
