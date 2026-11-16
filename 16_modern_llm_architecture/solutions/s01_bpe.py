"""
GOAL:  Solutions for step 01 exercises: byte-level BPE (train/encode/decode from memory), merges vs bytes per token,
       a word-boundary pre-tokenizer, and the decode bug.

RUN (from this folder):  python s01_bpe.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s01_bpe.py
EXPECTED OUTPUT (shortened):
    merges  bytes/token ...
NEEDS: python stdlib only (reads ../01_bpe_tokenizer_from_scratch/corpus_sample.txt). A few seconds on CPU.
"""
import re
from collections import Counter
from pathlib import Path

text = (Path(__file__).parent.parent / "01_bpe_tokenizer_from_scratch" / "corpus_sample.txt").read_text()[:20000]


def merge_pair(seq, pair, new_id):
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(seq[i]); i += 1
    return out


def train_bpe(text, n_merges, pretokenize=False):
    """Return (merges, vocab). merges: {(a, b): new_id} in learned order; vocab: id -> bytes."""
    # with pre-tokenization each word is its own sequence, so pairs never span word boundaries
    chunks = [list(w.encode()) for w in re.findall(r"\s?\S+|\s", text)] if pretokenize else [list(text.encode())]
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    for k in range(n_merges):
        counts = Counter()
        for ch in chunks:
            counts.update(zip(ch, ch[1:]))
        if not counts:
            break
        pair = counts.most_common(1)[0][0]
        new_id = 256 + k
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        chunks = [merge_pair(ch, pair, new_id) for ch in chunks]
    return merges, vocab


def encode(s, merges, pretokenize=False):
    pieces = [list(w.encode()) for w in re.findall(r"\s?\S+|\s", s)] if pretokenize else [list(s.encode())]
    ids = []
    for seq in pieces:
        for pair, new_id in merges.items():              # apply merges in the order they were learned
            seq = merge_pair(seq, pair, new_id)
        ids += seq
    return ids


def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


print(f"{'merges':>7}{'bytes/token':>13}{'vocab':>7}   (20 KB sample)")
for n in (50, 300, 1000):
    merges, vocab = train_bpe(text, n)
    ids = encode(text, merges)
    print(f"{n:>7}{len(text.encode()) / len(ids):>13.2f}{len(vocab):>7}")
    if n == 300:
        assert decode(ids, vocab) == text                 # round trip is lossless

merges, vocab = train_bpe(text, 300)
sample = "héllo wörld 🙂 Привет"
assert decode(encode(sample, merges), vocab) == sample
print("round trip incl. accents, emoji, Cyrillic: OK (no <unk> needed: any string is bytes)")

m2, v2 = train_bpe(text, 300, pretokenize=True)
cross = [v2[i] for i in m2.values() if b" " in v2[i].strip(b" ") or b"\n" in v2[i][1:]]
print(f"pre-tokenized: merged tokens containing an inner space/newline: {len(cross)} (plain BPE: "
      f"{sum(1 for i in merges.values() if b' ' in vocab[i].strip(b' '))})")
print("ids/bytes/token with pre-tokenizer:", f"{len(text.encode()) / len(encode(text, m2, True)):.2f}")

try:
    b"".join(i for i in [104, 105])                       # BUG: ids instead of vocab[i]
except TypeError as e:
    print("break-it error (using i instead of vocab[i]):", e)
