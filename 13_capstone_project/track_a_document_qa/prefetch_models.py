"""Docker build helper: download the two small models into HF_HOME once, so the running container can stay offline (HF_HUB_OFFLINE=1).
Not used by the local course run (the models are already cached there)."""
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

EMBED, RERANK = "sentence-transformers/all-MiniLM-L6-v2", "cross-encoder/ms-marco-MiniLM-L-6-v2"
AutoTokenizer.from_pretrained(EMBED)
AutoModel.from_pretrained(EMBED)
AutoTokenizer.from_pretrained(RERANK)
AutoModelForSequenceClassification.from_pretrained(RERANK)
print("models cached")
