"""governance.py -- versioning, guardrails and budgets around the RAG service (step 11).

Everything here is built from ragkit parts that already exist:
  ids.content_hash         -> prompt content hashes, deterministic user buckets
  patterns.Handler chain   -> the INPUT guard pipeline (step 05 Chain of Responsibility)
  gateway.extract_json + schemas.StructuredAnswer + query.CitationVerifier -> the OUTPUT guard
  query.PromptBuilder      -> chunk packing inside the window we compute
  ratelimit.TenantLimiters + gateway.CostMeter -> per-tenant admission and spend
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from pydantic import ValidationError

from .gateway import (BudgetExceeded, CostMeter, Gateway, LLMRequest, LLMResponse, Message, Role, Usage,
                      extract_json)
from .ids import content_hash
from .patterns import Handler, Normalize, MaxLength, QueryContext, RedactPII, build_chain
from .query import CitationVerifier, PromptBuilder, RagService, WordTokenCounter
from .ratelimit import TenantLimiters
from .resilience import PermanentError, RateLimitedError, SystemClock
from .schemas import StructuredAnswer

# =====================================================================================================================
# A. VERSIONING: prompts are code, so they get versions, hashes, pins, staged rollout and rollback
# =====================================================================================================================
_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_semver(v: str) -> tuple[int, int, int]:
    m = _SEMVER.match(v)
    if not m:
        raise ValueError(f"not a semantic version MAJOR.MINOR.PATCH: {v!r}")
    return int(m[1]), int(m[2]), int(m[3])


@dataclass(frozen=True, slots=True)
class PromptVersion:
    name: str
    version: str
    template: str
    content_hash: str            # fingerprint of (name, version, template): logs/traces carry it, so "which prompt?" is never a guess


class PromptRegistry:
    """name -> {version -> PromptVersion}. Published versions are IMMUTABLE; 'active' is a movable pointer with history."""

    def __init__(self):
        self._versions: dict[str, dict[tuple, PromptVersion]] = {}
        self._history: dict[str, list[str]] = {}               # active-version stack; last item is live

    def register(self, name: str, version: str, template: str) -> PromptVersion:
        key = parse_semver(version)
        pv = PromptVersion(name, version, template, content_hash(name, version, template, length=12))
        existing = self._versions.setdefault(name, {}).get(key)
        if existing is not None:
            if existing.content_hash != pv.content_hash:
                raise ValueError(f"{name}@{version} already published with different content: bump the version")
            return existing                                    # same content again = idempotent no-op
        self._versions[name][key] = pv
        return pv

    def get(self, name: str, version: str | None = None) -> PromptVersion:
        vs = self._versions[name]
        return vs[parse_semver(version)] if version else vs[max(vs)]      # no version -> highest semver

    def activate(self, name: str, version: str) -> PromptVersion:
        pv = self.get(name, version)
        self._history.setdefault(name, []).append(version)
        return pv

    def active(self, name: str) -> PromptVersion:
        return self.get(name, self._history[name][-1])

    def rollback(self, name: str) -> PromptVersion:
        h = self._history[name]
        if len(h) < 2:
            raise ValueError(f"{name}: nothing to roll back to")
        h.pop()
        return self.active(name)


@dataclass(frozen=True, slots=True)
class ReleaseConfig:
    """Everything that changes behaviour, pinned together. Never 'latest': a pin is reproducible, 'latest' is not."""
    name: str                    # human label: "stable" / "canary"
    prompt_name: str
    prompt_version: str
    model: str                   # gateway route alias (step 08)
    model_snapshot: str          # the dated provider model id the alias currently points to (pin it, log it)
    temperature: float = 0.0
    max_tokens: int = 200

    @property
    def fingerprint(self) -> str:
        return content_hash(self.prompt_name, self.prompt_version, self.model, self.model_snapshot,
                            str(self.temperature), str(self.max_tokens), length=10)


def bucket(user_id: str, salt: str) -> float:
    """Deterministic number in [0, 1) for (salt, user): the same user always lands in the same bucket."""
    return int(content_hash(salt, user_id, length=8), 16) / 16 ** 8


class ABRouter:
    """Weighted variants by hash of user id. Order matters: variant 1 owns [0, w1), variant 2 owns [w1, w1+w2), ..."""

    def __init__(self, variants: Sequence[tuple[ReleaseConfig, float]], salt: str = "exp-1"):
        self.salt = salt
        self.set_weights(variants)

    def set_weights(self, variants: Sequence[tuple[ReleaseConfig, float]]) -> None:
        total = sum(w for _, w in variants)
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        edge, self._edges = 0.0, []
        for cfg, w in variants:
            edge += w / total
            self._edges.append((edge, cfg))

    def route(self, user_id: str) -> ReleaseConfig:
        b = bucket(user_id, self.salt)
        for edge, cfg in self._edges:
            if b < edge:
                return cfg
        return self._edges[-1][1]


class CanaryRouter(ABRouter):
    """Canary = an A/B test with a tiny, adjustable percentage. The candidate owns the LOWEST buckets, so raising the
    percentage only ADDS users (nobody flips back and forth); percent=0 is the instant rollback."""

    def __init__(self, stable: ReleaseConfig, candidate: ReleaseConfig, percent: float = 5, salt: str = "canary-1"):
        self.stable, self.candidate = stable, candidate
        super().__init__([(candidate, max(percent, 0.0)), (stable, 100 - percent)], salt)
        self.percent = percent

    def set_percent(self, percent: float) -> None:
        self.percent = percent
        self.set_weights([(self.candidate, max(percent, 0.0)), (self.stable, 100 - percent)] if percent > 0
                         else [(self.stable, 1.0)])


# =====================================================================================================================
# B. GUARDRAILS: input chain (before the model) and output guard (after it)
# =====================================================================================================================
def luhn_ok(digits: str) -> bool:
    """Credit-card checksum: cuts false positives, so order numbers are not redacted as cards."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


class RedactCards(Handler):
    def process(self, ctx: QueryContext) -> QueryContext:
        def sub(m: re.Match) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            return "[CARD]" if 13 <= len(digits) <= 19 and luhn_ok(digits) else m.group(0)
        q = re.sub(r"\b\d(?:[ -]?\d){12,18}\b", sub, ctx.question)
        if q != ctx.question:
            ctx.notes.append("card_redacted")
        ctx.question = q
        return ctx


INJECTION_PATTERNS: list[tuple[str, float]] = [          # (regex, suspicion weight); score >= 1.0 blocks
    (r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules|prompts?)", 1.0),
    (r"disregard .{0,30}(instructions|rules|system prompt)", 1.0),
    (r"(reveal|print|show|repeat|leak) .{0,30}(system prompt|hidden instructions|your instructions)", 0.6),
    (r"</?(system|documents)>", 0.7),                    # spoofing our own prompt delimiters
    (r"\b(developer mode|jailbreak|DAN mode)\b", 0.6),
    (r"\b(you are now|pretend (to be|you are))\b", 0.5),
]


class InjectionHeuristics(Handler):
    """Cheap, explainable and EASY TO BYPASS (paraphrase, other languages, encodings). It is one layer of defence in
    depth; the real mitigations are least privilege and treating retrieved text as data (08_agent_harness/07)."""

    def process(self, ctx: QueryContext) -> QueryContext:
        score = sum(w for pat, w in INJECTION_PATTERNS if re.search(pat, ctx.question, re.I))
        if score >= 1.0:
            ctx.blocked = f"prompt_injection(score={score:.1f})"
        elif score > 0:
            ctx.notes.append(f"injection_suspect(score={score:.1f})")
        return ctx


class TokenLimit(Handler):
    def __init__(self, limit: int = 150, counter=None):
        self.limit, self.counter = limit, counter or WordTokenCounter()

    def process(self, ctx: QueryContext) -> QueryContext:
        n = self.counter.count(ctx.question)
        if n > self.limit:
            ctx.blocked = f"too_many_tokens({n}>{self.limit})"
        return ctx


def build_input_guard(max_chars: int = 500, max_tokens: int = 150) -> Handler:
    """Order matters: cheap length checks FIRST (never run regexes over a 10 MB string), redact BEFORE logging/inject-scan."""
    return build_chain(MaxLength(max_chars), Normalize(), TokenLimit(max_tokens), RedactPII(), RedactCards(),
                       InjectionHeuristics())


REFUSAL_TEXT = "Sorry, I cannot give a reliable answer to that from the knowledge base."


@dataclass(frozen=True, slots=True)
class OutputVerdict:
    action: str                                  # "pass" | "repair" (ask the model again) | "refuse" (show REFUSAL_TEXT)
    answer: StructuredAnswer | None
    reasons: tuple[str, ...] = ()


class OutputGuard:
    """Validate the model's JSON (pydantic), check every citation exists and is supported by the retrieved chunks,
    apply the refusal policy, scrub PII that leaked into the answer."""

    def __init__(self, verifier: CitationVerifier | None = None, min_confidence: float = 0.3):
        self.verifier, self.min_confidence = verifier or CitationVerifier(), min_confidence

    def check(self, raw: str, used) -> OutputVerdict:
        try:
            ans = StructuredAnswer.model_validate(extract_json(raw))
        except ValidationError as exc:                       # valid JSON, wrong shape: the model can fix this
            return OutputVerdict("repair", None, ("schema:" + "; ".join(f"{'.'.join(map(str, e['loc']))} {e['msg']}"
                                                                       for e in exc.errors()),))
        except ValueError as exc:                            # no JSON at all (json.JSONDecodeError is a ValueError too)
            return OutputVerdict("repair", None, (f"not_json:{str(exc)[:40]}",))
        text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", ans.answer)
        reasons = ["pii_scrubbed"] if text != ans.answer else []
        if text != ans.answer:
            ans = ans.model_copy(update={"answer": text})
        if ans.answer.strip().lower().startswith("i don't know"):
            return OutputVerdict("pass", ans, tuple(reasons + ["model_declined"]))     # an honest "no" is a good answer
        if ans.confidence < self.min_confidence:
            return OutputVerdict("refuse", None, tuple(reasons + [f"low_confidence({ans.confidence:.2f})"]))
        v = self.verifier.verify(ans.answer, used)
        if v.invalid_markers:
            return OutputVerdict("refuse", None, tuple(reasons + [f"invalid_citation_markers{list(v.invalid_markers)}"]))
        if not v.ok:
            return OutputVerdict("refuse", None, tuple(reasons + ["ungrounded:no_supported_citation"]))
        return OutputVerdict("pass", ans, tuple(reasons))


# =====================================================================================================================
# C. BUDGETS: context window, per-request output cap, per-tenant rate and spend, cost ledger
# =====================================================================================================================
@dataclass(frozen=True, slots=True)
class ContextPlan:
    window: int
    system_tokens: int
    question_tokens: int
    answer_reserve: int
    history_kept: tuple[Message, ...]
    history_tokens: int
    history_dropped: int
    prompt_budget: int           # what PromptBuilder may spend on system + question + chunks


class ContextBudget:
    """window = system + history + retrieved chunks + question + ANSWER RESERVE. The reserve is subtracted first:
    a prompt that fills the window leaves the model no room to answer (it gets cut off mid-sentence)."""

    def __init__(self, window: int = 1024, answer_reserve: int = 200, history_share: float = 0.3, counter=None):
        self.window, self.reserve, self.history_share = window, answer_reserve, history_share
        self.counter = counter or WordTokenCounter()

    def plan(self, system: str, question: str, history: Sequence[Message]) -> ContextPlan:
        sys_t, q_t = self.counter.count(system), self.counter.count(question)
        if sys_t + q_t + self.reserve > self.window:
            raise PermanentError(f"context_window_too_small: system+question+reserve = {sys_t + q_t + self.reserve} > {self.window}")
        cap = int((self.window - sys_t - q_t - self.reserve) * self.history_share)
        kept, used = [], 0
        for m in reversed(history):                          # newest turns matter most: keep from the end
            c = self.counter.count(m.content) + 4            # +4: per-message role/format overhead
            if used + c > cap:
                break
            kept.append(m)
            used += c
        kept.reverse()
        return ContextPlan(self.window, sys_t, q_t, self.reserve, tuple(kept), used, len(history) - len(kept),
                           self.window - self.reserve - used)

    def output_limit(self, prompt_tokens: int, requested: int, hard_cap: int = 400) -> int:
        """Per-request max_tokens: what the caller asked, clamped by a hard cap and by the room left in the window."""
        return max(0, min(requested, hard_cap, self.window - prompt_tokens))


class TenantGate:
    """Admission control per tenant: tokens-per-minute bucket (ratelimit.TenantLimiters) + a spend ceiling (CostMeter)."""

    def __init__(self, rpm: int, tpm: int, budget_usd: float, clock=None, prices=None):
        self.clock = clock or SystemClock
        self.limiters, self.budget_usd, self.prices = TenantLimiters(rpm, tpm, self.clock), budget_usd, prices
        self.meters: dict[str, CostMeter] = {}

    def meter(self, tenant: str) -> CostMeter:
        if tenant not in self.meters:
            self.meters[tenant] = CostMeter(self.prices, self.budget_usd)
        return self.meters[tenant]

    def admit(self, tenant: str, est_tokens: int) -> None:
        self.meter(tenant).check()                           # raises BudgetExceeded (permanent: do not retry)
        lim = self.limiters.for_tenant(tenant)
        if est_tokens > lim.tokens.capacity:
            raise PermanentError(f"request needs {est_tokens} tokens but the per-minute budget is {int(lim.tokens.capacity)}")
        if not lim.try_acquire(est_tokens):
            wait = max(lim.requests.wait_time(1), lim.tokens.wait_time(est_tokens))
            raise RateLimitedError(f"tenant {tenant} over its rate limit", retry_after=round(wait, 1))

    def settle(self, tenant: str, est_tokens: int, resp: LLMResponse) -> float:
        self.limiters.for_tenant(tenant).settle(est_tokens, resp.usage.input_tokens + resp.usage.output_tokens)
        return self.meter(tenant).record(resp)


@dataclass(frozen=True, slots=True)
class LedgerRow:
    request_id: str
    tenant: str
    release: str
    fingerprint: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    outcome: str


@dataclass
class Ledger:
    """One row per request: who, which release, how many tokens, what it cost, how it ended. Billing and debugging read this."""
    rows: list[LedgerRow] = field(default_factory=list)

    def add(self, row: LedgerRow) -> None:
        self.rows.append(row)

    def by_tenant(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.rows:
            out[r.tenant] = out.get(r.tenant, 0.0) + r.cost_usd
        return out

    def by_release(self) -> dict[str, tuple[int, float]]:
        out: dict[str, tuple[int, float]] = {}
        for r in self.rows:
            n, c = out.get(r.release, (0, 0.0))
            out[r.release] = (n + 1, c + r.cost_usd)
        return out


# =====================================================================================================================
# D. THE GOVERNED SERVICE: route -> guard -> budget -> call -> guard -> account
# =====================================================================================================================
@dataclass(frozen=True, slots=True)
class GovernedResult:
    status: str                  # ok | blocked_input | refused_retrieval | refused_output | budget_exceeded | rate_limited | too_large
    text: str
    release: str = ""
    prompt_version: str = ""
    reasons: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class GovernedService:
    def __init__(self, rag: RagService, gateway: Gateway, router: ABRouter, prompts: PromptRegistry,
                 input_guard: Handler, output_guard: OutputGuard, context: ContextBudget, gate: TenantGate,
                 ledger: Ledger | None = None):
        self.rag, self.gateway, self.router, self.prompts = rag, gateway, router, prompts
        self.input_guard, self.output_guard, self.context, self.gate = input_guard, output_guard, context, gate
        self.ledger = ledger or Ledger()

    def _row(self, rid, tenant, rel, status, tin=0, tout=0, cost=0.0):
        self.ledger.add(LedgerRow(rid, tenant, rel.name, rel.fingerprint, tin, tout, cost, status))

    def ask(self, user_id: str, tenant: str, question: str, history: Sequence[Message] = ()) -> GovernedResult:
        rel = self.router.route(user_id)                                      # 1. which release serves this user
        prompt = self.prompts.get(rel.prompt_name, rel.prompt_version)        #    pinned version, never "latest"
        base = dict(release=rel.name, prompt_version=prompt.version)
        ctx = self.input_guard.handle(QueryContext(question))                 # 2. input guard (cheap, before any model/retrieval)
        if ctx.blocked:
            self._row("-", tenant, rel, "blocked_input")
            return GovernedResult("blocked_input", REFUSAL_TEXT, reasons=(ctx.blocked, *ctx.notes), **base)
        q = ctx.question
        ranked = self.rag.reranker.rerank(q, self.rag.retriever.retrieve(q, self.rag.retrieve_k), self.rag.top_n)
        why = self.rag.refusal.check(ranked)                                  # 3. retrieval refusal (no paid call)
        if why:
            self._row("-", tenant, rel, "refused_retrieval")
            return GovernedResult("refused_retrieval", REFUSAL_TEXT, reasons=(why,), **base)
        plan = self.context.plan(prompt.template, q, history)                 # 4. context-window budget
        built = PromptBuilder(self.context.counter, plan.prompt_budget, prompt.template).build(q, ranked)
        out_cap = self.context.output_limit(built.tokens + plan.history_tokens, rel.max_tokens)
        msgs = (*plan.history_kept, Message(Role.USER, built.user))
        req = LLMRequest(msgs, rel.model, built.system, out_cap, rel.temperature)
        est = built.tokens + plan.history_tokens + out_cap                    # worst-case reservation, settled after the call
        try:
            self.gate.admit(tenant, est)                                      # 5. tenant rate + spend
        except BudgetExceeded as exc:
            self._row(req.request_id, tenant, rel, "budget_exceeded")
            return GovernedResult("budget_exceeded", "Your usage budget is exhausted.", reasons=(str(exc),), **base)
        except PermanentError as exc:                                         # bigger than the bucket: retrying never helps
            self._row(req.request_id, tenant, rel, "too_large")
            return GovernedResult("too_large", "Request too large for your plan.", reasons=(str(exc),), **base)
        except RateLimitedError as exc:
            self._row(req.request_id, tenant, rel, "rate_limited")
            return GovernedResult("rate_limited", "Too many requests, retry later.",
                                  reasons=(f"retry_after={exc.retry_after}s",), **base)
        tin = tout = 0
        cost, verdict, reasons = 0.0, None, []
        for attempt in range(2):                                              # 6. call + output guard (one repair allowed)
            resp = self.gateway.complete(req)
            cost += self.gate.settle(tenant, est if attempt == 0 else 0, resp)
            tin, tout = tin + resp.usage.input_tokens, tout + resp.usage.output_tokens
            verdict = self.output_guard.check(resp.text, built.used)
            reasons += verdict.reasons
            if verdict.action != "repair":
                break
            req = LLMRequest((*req.messages, Message(Role.ASSISTANT, resp.text),
                              Message(Role.USER, "Return ONLY valid JSON: " + verdict.reasons[0][:120])),
                             req.model, req.system, req.max_tokens, req.temperature)
        ok = verdict.action == "pass"
        status = "ok" if ok else "refused_output"
        self._row(req.request_id, tenant, rel, status, tin, tout, cost)       # 7. account
        return GovernedResult(status, verdict.answer.answer if ok else REFUSAL_TEXT, reasons=tuple(reasons),
                              input_tokens=tin, output_tokens=tout, cost_usd=cost, **base)
