"""evalkit.py -- a tiny LLM/RAG eval harness: cases, scorers, runner, regression compare (step 10, problem 4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .domain import Answer


@dataclass(frozen=True, slots=True)
class EvalCase:
    id: str
    question: str
    expected: str | None = None                  # a string the answer should contain
    gold_chunk_ids: tuple[str, ...] = ()
    answerable: bool = True


class Scorer(Protocol):
    name: str
    def score(self, case: EvalCase, answer: Answer) -> float | None: ...      # None = not applicable


class ContainsScorer:
    name = "contains"
    def score(self, case, answer):
        if not case.answerable or case.expected is None:
            return None
        return float(case.expected.casefold() in answer.text.casefold())


class RefusalScorer:
    name = "refusal_correct"
    def score(self, case, answer):
        return float(answer.refused == (not case.answerable))


class RetrievalHitScorer:
    name = "gold_in_context"
    def score(self, case, answer):
        if not case.gold_chunk_ids:
            return None
        return float(any(g in answer.context_chunk_ids for g in case.gold_chunk_ids))


class CitedGoldScorer:
    name = "cites_gold"
    def score(self, case, answer):
        if not case.gold_chunk_ids:
            return None
        return float(any(c.chunk_id in case.gold_chunk_ids for c in answer.citations))


@dataclass
class EvalReport:
    rows: list[dict] = field(default_factory=list)

    def mean(self, scorer: str) -> float | None:
        vals = [r[scorer] for r in self.rows if r.get(scorer) is not None]
        return sum(vals) / len(vals) if vals else None

    def means(self) -> dict[str, float | None]:
        names = sorted({k for r in self.rows for k in r if k not in ("id",)})
        return {n: self.mean(n) for n in names}

    def failures(self, scorer: str) -> list[str]:
        return [r["id"] for r in self.rows if r.get(scorer) == 0.0]


class EvalRunner:
    def __init__(self, answerer, scorers: Sequence[Scorer]):
        self.answerer, self.scorers = answerer, list(scorers)

    def run(self, cases: Sequence[EvalCase]) -> EvalReport:
        rep = EvalReport()
        for c in cases:
            try:
                ans = self.answerer.answer(c.question)
            except Exception as exc:                                  # one crashing case must not abort the run
                rep.rows.append({"id": c.id, "error": 1.0, **{s.name: 0.0 for s in self.scorers}})
                continue
            rep.rows.append({"id": c.id, **{s.name: s.score(c, ans) for s in self.scorers}})
        return rep


def regressions(base: EvalReport, new: EvalReport, scorer: str) -> list[str]:
    """Case ids that passed in `base` but fail in `new` for this scorer -- what a CI gate should list."""
    old_ok = {r["id"] for r in base.rows if r.get(scorer) == 1.0}
    return [i for i in new.failures(scorer) if i in old_ok]
