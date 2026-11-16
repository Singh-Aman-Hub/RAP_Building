"""strategies.py -- three chunkers behind the same Protocol: token windows, recursive, markdown sections."""
from __future__ import annotations

import re
from collections import Counter

from .model import Chunk, ChunkConfig, TokenCounter, make_chunk_id, sha
from .registry import register


class _Base:
    version = "1.0.0"

    def __init__(self, config: ChunkConfig, count_tokens: TokenCounter):
        self.config, self.count = config, count_tokens

    def _emit(self, doc_id, text, spans, base_offset, doc_version, section_path=(), parent_id=None, embed_prefix="", metadata=None):
        """spans: [(start, end)] relative to `text`  ->  list[Chunk] with stable ids and offsets."""
        seen, out = Counter(), []
        for pos, (s, e) in enumerate(spans):
            raw = text[s:e]
            lead = len(raw) - len(raw.lstrip())
            body = raw.strip()
            if not body:
                continue
            embed = f"{embed_prefix}\n{body}" if embed_prefix else body
            occ = seen[body]
            seen[body] += 1
            out.append(Chunk(
                chunk_id=make_chunk_id(doc_id, self.chunker_id, self.version, self.config.config_hash, body, occ),
                doc_id=doc_id, doc_version=doc_version, parent_id=parent_id, text=body, embed_text=embed, display_text=body,
                token_count=self.count(embed), position=pos, char_start=base_offset + s + lead, char_end=base_offset + s + lead + len(body),
                section_path=tuple(section_path), metadata=dict(metadata or {}), chunker_id=self.chunker_id, chunker_version=self.version))
        return out


@register("token_window")
class TokenWindowChunker(_Base):
    """Fixed windows by tokens, found by binary search on the injected counter (works with ANY tokenizer)."""

    def chunk(self, doc_id, text, base_offset=0):
        spans, i, n = [], 0, len(text)
        while i < n:
            lo, hi = i + 1, n
            while lo < hi:                                           # largest j with count(text[i:j]) <= max_tokens
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if self.count(text[i:mid]) <= self.config.max_tokens else (lo, mid - 1)
            spans.append((i, lo))
            if lo >= n:
                break
            back = lo
            while back > i + 1 and self.count(text[back:lo]) < self.config.overlap_tokens:   # step back to create the overlap
                back -= 1
            i = max(back, i + 1)
        return self._emit(doc_id, text, spans, base_offset, sha(text))


@register("recursive")
class RecursiveChunker(_Base):
    """Separator hierarchy; merges small pieces; hard-cuts only when no separator is left."""

    def _pieces(self, text, start, end, seps):
        """Split text[start:end] into atomic spans <= max_tokens, using the first separator that occurs."""
        if self.count(text[start:end]) <= self.config.max_tokens:
            return [(start, end)]
        sep, rest = seps[0], seps[1:]
        if sep == "":                                                # last resort: character cut by token ratio
            step = max(1, int((end - start) * self.config.max_tokens / max(self.count(text[start:end]), 1)))
            return [(i, min(i + step, end)) for i in range(start, end, step)]
        cuts, pos = [], start
        while True:
            k = text.find(sep, pos, end)
            if k == -1:
                break
            cuts.append(k + len(sep))                                # the separator stays at the END of the left piece
            pos = k + len(sep)
        if not cuts:
            return self._pieces(text, start, end, rest)
        edges = [start] + [c for c in cuts if c < end] + [end]
        out = []
        for a, b in zip(edges, edges[1:]):
            out += self._pieces(text, a, b, rest) if self.count(text[a:b]) > self.config.max_tokens else [(a, b)]
        return out

    def _merge(self, text, pieces):
        """Greedy pack; carry the tail of each chunk into the next until `overlap_tokens` is reached."""
        cfg, groups, cur = self.config, [], []
        for p in pieces:
            if cur and self.count(text[cur[0][0]:p[1]]) > cfg.max_tokens:
                groups.append(cur)
                tail, t = [], 0
                for q in reversed(cur):
                    t += self.count(text[q[0]:q[1]])
                    if t > cfg.overlap_tokens:
                        break
                    tail.insert(0, q)
                cur = tail
                while cur and self.count(text[cur[0][0]:p[1]]) > cfg.max_tokens:
                    cur.pop(0)
            cur.append(p)
        if cur:
            groups.append(cur)
        return [(g[0][0], g[-1][1]) for g in groups]

    def chunk(self, doc_id, text, base_offset=0):
        spans = self._merge(text, self._pieces(text, 0, len(text), tuple(self.config.separators)))
        return self._emit(doc_id, text, spans, base_offset, sha(text))


@register("markdown_sections")
class MarkdownSectionChunker(RecursiveChunker):
    """Heading tree -> one chunk per section body (recursive split if too long); breadcrumb in embed_text."""
    HEAD = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)

    def chunk(self, doc_id, text, base_offset=0):
        heads, fence = [], False
        for m in self.HEAD.finditer(text):
            if text[:m.start()].count("```") % 2 == 0:               # ignore '# comment' lines inside code fences
                heads.append((m.start(), m.end(), len(m.group(1)), m.group(2)))
        bounds = [(h[1], heads[i + 1][0] if i + 1 < len(heads) else len(text)) for i, h in enumerate(heads)]
        path, out, dv = [], [], sha(text)
        for h, (s, e) in zip(heads, bounds):
            path = path[:h[2] - 1] + [h[3]]
            body = text[s:e]
            if not body.strip():
                continue
            crumb = " > ".join(path)
            sub = self._merge(body, self._pieces(body, 0, len(body), tuple(self.config.separators)))
            out += self._emit(doc_id, body, sub, base_offset + s, dv, tuple(path), parent_id=f"{doc_id}#{sha(crumb, 8)}",
                              embed_prefix=crumb if self.config.breadcrumb else "")
        for i, c in enumerate(out):                                  # renumber positions over the whole document
            object.__setattr__(c, "position", i)
        return out
