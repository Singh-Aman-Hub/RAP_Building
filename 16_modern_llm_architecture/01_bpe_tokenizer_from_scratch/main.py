"""
GOAL:  Train a byte-level BPE tokenizer by hand (about 40 lines), encode/decode with it, and compare it with the
       real gpt2 and Qwen2.5 tokenizers.  See why vocabulary size is a trade-off and what special/chat tokens are.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 16_modern_llm_architecture/01_bpe_tokenizer_from_scratch/main.py
EXPECTED OUTPUT (shortened):
    corpus: 11,xxx bytes ...
    merge 1: (101, 32) -> 256 ...
    round trip ok: True
    text                      mine  gpt2  qwen
NEEDS: transformers; cached gpt2 and Qwen/Qwen2.5-0.5B tokenizers.  ~3 s.
"""
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer

# ---- 1. A small training corpus: this repo's own module README ------------------------------------------
corpus_path = Path(__file__).resolve().parent / "corpus_sample.txt"      # 20,000 characters of this repo's markdown
text = corpus_path.read_text(encoding="utf-8")
data = list(text.encode("utf-8"))                      # bytes 0..255: the base vocabulary of byte-level BPE
print(f"corpus: {len(text):,} characters = {len(data):,} bytes")


# ---- 2. Count adjacent pairs and merge the most frequent one ---------------------------------------------
def pair_counts(ids):
    return Counter(zip(ids, ids[1:]))                  # {(a, b): how often b follows a}


def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:   # found the pair: replace two ids with one
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ---- 3. Training loop: repeat merge until we have the vocabulary size we want ---------------------------
NUM_MERGES = 300                                       # final vocab = 256 bytes + 300 merges = 556
merges = {}                                            # (a, b) -> new id, in the order learned
vocab = {i: bytes([i]) for i in range(256)}            # id -> the bytes it stands for
ids = data[:]
for step in range(NUM_MERGES):
    counts = pair_counts(ids)
    pair = counts.most_common(1)[0][0]                 # most frequent adjacent pair
    new_id = 256 + step
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]    # a merged token is the concatenation of its parts
    if step < 8 or step in (99, 299):
        print(f"merge {step + 1:>3}: {pair} -> {new_id}  = {vocab[new_id]!r}  (count {counts[pair]})")
print(f"training bytes {len(data):,} -> {len(ids):,} tokens  (compression {len(data) / len(ids):.2f} bytes/token)")


# ---- 4. Encode and decode ---------------------------------------------------------------------------
def encode(s):
    ids = list(s.encode("utf-8"))
    while len(ids) >= 2:
        counts = pair_counts(ids)
        # apply the EARLIEST-learned merge present (lowest new id): the same order as in training
        pair = min(counts, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                                      # nothing left to merge
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


sample = "The attention layer reads the tokens."
enc = encode(sample)
print("encode:", enc)
print("pieces:", [vocab[i].decode("utf-8", "replace") for i in enc])
print("round trip ok:", decode(enc) == sample)
emoji = "naïve café 🙂"
print("unicode round trip ok:", decode(encode(emoji)) == emoji, "| bytes:", len(emoji.encode()), "| tokens:", len(encode(emoji)))

# ---- 5. Compare with real tokenizers -------------------------------------------------------------------
gpt2 = AutoTokenizer.from_pretrained("gpt2")
qwen = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
print(f"\nvocab sizes: mine {len(vocab)} | gpt2 {len(gpt2)} | qwen2.5 {len(qwen)}")
samples = {
    "english prose": "The attention layer reads the tokens and decides what matters.",
    "python code": "def add(a, b):\n    return a + b\n",
    "numbers": "Total: 1234567.89 rupees",
    "hindi": "मुझे चाय पसंद है",
    "emoji": "great job 🙂🙂",
}
print(f"{'text':<15}{'bytes':>6}{'mine':>6}{'gpt2':>6}{'qwen':>6}")
for name, s in samples.items():
    print(f"{name:<15}{len(s.encode()):>6}{len(encode(s)):>6}{len(gpt2.encode(s)):>6}{len(qwen.encode(s)):>6}")
print("gpt2 pieces for numbers:", gpt2.tokenize(samples["numbers"]))
print("qwen pieces for numbers:", qwen.tokenize(samples["numbers"]))

# ---- 6. Special tokens and chat template -----------------------------------------------------------------------
print("\ngpt2 special:", gpt2.all_special_tokens, "| qwen special:", qwen.all_special_tokens)
msgs = [{"role": "user", "content": "Hi"}]
print("qwen chat template ->", repr(qwen.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
print("'<|im_start|>' is ONE id:", qwen.convert_tokens_to_ids("<|im_start|>"))

# ---- 7. Embedding table cost of a bigger vocabulary ------------------------------------------------------------
for name, v, d in [("gpt2", 50257, 768), ("qwen2.5-0.5b", 151936, 896)]:
    print(f"{name}: embedding table {v:,} x {d} = {v * d / 1e6:.1f}M params")

# TRY IT: change NUM_MERGES to 50 and 1000 and watch the compression (bytes/token) and the pieces change.
# BREAK IT: in decode() replace vocab[i] with i. Python raises a TypeError (can't join ints): ids are not bytes.
