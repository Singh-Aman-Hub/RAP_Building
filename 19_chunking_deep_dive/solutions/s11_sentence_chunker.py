"""Step 11 exercise: a fourth strategy behind the same Protocol, plus the registry guard and a test.
RUN: HF_HUB_OFFLINE=1 ../../.venv/bin/python s11_sentence_chunker.py"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "11_chunker_lld"))
from chunkerlib import ChunkConfig, available, create, register  # noqa: E402
from chunkerlib.strategies import _Base  # noqa: E402
from chunkerlib.model import sha  # noqa: E402


@register("sentence_pack")
class SentencePackChunker(_Base):
    """Pack whole sentences up to max_tokens (never cuts a sentence unless one sentence alone is too long)."""

    def chunk(self, doc_id, text, base_offset=0):
        sents = [(m.start(), m.end()) for m in re.finditer(r"[^.!?\n]+[.!?]?", text) if m.group().strip()]
        spans, cur = [], []
        for s in sents:
            if cur and self.count(text[cur[0][0]:s[1]]) > self.config.max_tokens:
                spans.append((cur[0][0], cur[-1][1]))
                cur = []
            cur.append(s)
        if cur:
            spans.append((cur[0][0], cur[-1][1]))
        return self._emit(doc_id, text, spans, base_offset, sha(text))


words = lambda t: len(t.split())
text = "One two three. Four five six seven. Eight nine. Ten eleven twelve thirteen fourteen. Fifteen."
chunks = create("sentence_pack", ChunkConfig(max_tokens=8), words).chunk("d", text)
print("registered:", available())
print("chunks:", [c.text for c in chunks])
assert all(text[c.char_start:c.char_end] == c.text for c in chunks), "offsets must reproduce the text"
assert chunks[0].chunk_id == create("sentence_pack", ChunkConfig(max_tokens=8), words).chunk("d", text)[0].chunk_id
try:
    register("sentence_pack")(SentencePackChunker)
except ValueError as e:
    print("duplicate registration ->", e)
print("offsets and ids OK")
