"""memory.py -- conversation memory store: per-session window with a token budget + optional summary (step 10, problem 5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .cache import MISS, TTLLRUCache
from .query import WordTokenCounter


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    content: str
    tokens: int


class ConversationMemory:
    """Sessions live in a TTL+LRU cache (idle sessions expire; at most `max_sessions` in memory).
    Each session keeps the newest turns that fit `max_tokens`; evicted turns are folded into a summary if a summariser is given."""

    def __init__(self, max_tokens: int = 200, max_sessions: int = 1000, ttl: float | None = 3600.0,
                 summarizer: Callable[[str, list[Turn]], str] | None = None, clock=None):
        self.max_tokens, self.summarizer = max_tokens, summarizer
        self.count = WordTokenCounter().count
        self._sessions = TTLLRUCache(max_sessions, ttl, clock)

    def _get(self, sid: str) -> dict:
        s = self._sessions.get(sid)
        return {"turns": [], "summary": ""} if s is MISS else s

    def add(self, sid: str, role: str, content: str) -> None:
        s = self._get(sid)
        s["turns"].append(Turn(role, content, self.count(content)))
        evicted: list[Turn] = []
        while len(s["turns"]) > 1 and sum(t.tokens for t in s["turns"]) > self.max_tokens:
            evicted.append(s["turns"].pop(0))                         # oldest first
        if evicted and self.summarizer:
            s["summary"] = self.summarizer(s["summary"], evicted)
        self._sessions.put(sid, s)                                    # put = refresh TTL + mark recently used

    def window(self, sid: str) -> list[dict]:
        s = self._get(sid)
        head = [{"role": "system", "content": f"Summary of earlier conversation: {s['summary']}"}] if s["summary"] else []
        return head + [{"role": t.role, "content": t.content} for t in s["turns"]]

    def clear(self, sid: str) -> None:
        self._sessions._data.pop(sid, None)
