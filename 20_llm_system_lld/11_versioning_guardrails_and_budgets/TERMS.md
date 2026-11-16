# Terms — Versioning, guardrails and budgets

Folder: `20_llm_system_lld/11_versioning_guardrails_and_budgets` · Files covered: `main.py`, `WALKTHROUGH.md`, `ragkit/governance.py`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Semantic versioning (semver, MAJOR.MINOR.PATCH)](#semantic-versioning-semver-majorminorpatch--concept) · [Prompt registry (prompt store, prompt hub)](#prompt-registry-prompt-store-prompt-hub--class) · [Content hash (fingerprint, digest)](#content-hash-fingerprint-digest--concept) · [Pinning (version pinning, snapshot pinning)](#pinning-version-pinning-snapshot-pinning--concept) · [Canary release (staged rollout, progressive delivery)](#canary-release-staged-rollout-progressive-delivery--concept) · [A/B test (split test, experiment)](#ab-test-split-test-experiment--concept) · [Hash-based bucketing (sticky assignment)](#hash-based-bucketing-sticky-assignment--function) · [Guardrails (input/output validation, safety filters)](#guardrails-inputoutput-validation-safety-filters--concept) · [PII redaction (masking, scrubbing)](#pii-redaction-masking-scrubbing--concept) · [Luhn checksum](#luhn-checksum--function) · [Prompt-injection heuristics](#prompt-injection-heuristics--concept) · [Output validation (schema validation, structured output check)](#output-validation-schema-validation-structured-output-check--concept) · [Groundedness check (faithfulness, citation support)](#groundedness-check-faithfulness-citation-support--concept) · [Repair vs refuse](#repair-vs-refuse--concept) · [Context-window budget (token budget, prompt budget)](#context-window-budget-token-budget-prompt-budget--concept) · [Answer reserve (output headroom)](#answer-reserve-output-headroom--concept) · [Tenant gate (per-tenant admission control)](#tenant-gate-per-tenant-admission-control--class) · [Reserve and settle (pre-authorization)](#reserve-and-settle-pre-authorization--concept) · [Cost ledger (usage log, metering table)](#cost-ledger-usage-log-metering-table--class) · [pydantic ValidationError](#pydantic-validationerror--error) · [PermanentError: context_window_too_small](#permanenterror-contextwindowtoosmall--error)

---

## Semantic versioning (semver, MAJOR.MINOR.PATCH)  `(concept)`
**1. Definition:** A version scheme MAJOR.MINOR.PATCH compared numerically part by part; for prompts, MAJOR = output contract changed, MINOR = behaviour or wording improved, PATCH = typo.
**2. In plain words:** A label that says how big a change is.
**3. Example to quote:** > "Compare as integers: 1.10.0 is newer than 1.9.0, a string sort gets it wrong."
```python
parse_semver('1.10.2')  # (1, 10, 2)
```
**Seen in:** `ragkit/governance.py`

## Prompt registry (prompt store, prompt hub)  `(class)`
**1. Definition:** A store of named, versioned, immutable prompt texts with an active pointer and history for rollback.
**2. In plain words:** A library where each prompt edition is stamped, never edited, and one edition is marked live.
**3. Example to quote:** > "Publishing different text under an existing version raises an error: bump the version."
```python
reg.register('customer_rag', '1.1.0', text); reg.activate('customer_rag', '1.1.0'); reg.rollback('customer_rag')
```
**Seen in:** `ragkit/governance.py`

## Content hash (fingerprint, digest)  `(concept)`
**1. Definition:** A hash of the exact bytes of an artifact (here sha256 of name, version, template, first 12 hex chars) that proves identity.
**2. In plain words:** A fingerprint: change one character and it changes.
**3. Example to quote:** > "Log the version and the hash with every answer so you can tell which prompt produced it."
```python
content_hash(name, version, template, length=12)
```
**Seen in:** `ragkit/ids.py`, `ragkit/governance.py`

## Pinning (version pinning, snapshot pinning)  `(concept)`
**1. Definition:** Naming an exact prompt version and an exact dated model snapshot in a release, instead of 'latest' or a floating alias.
**2. In plain words:** Locking the recipe and the oven model so Tuesday's cake equals Monday's.
**3. Example to quote:** > "A provider can repoint an alias; only a pinned snapshot is reproducible."
```python
ReleaseConfig('stable','customer_rag','1.0.0','small','mock-small-2026-01-15')
```
**Seen in:** `ragkit/governance.py`

## Canary release (staged rollout, progressive delivery)  `(concept)`
**1. Definition:** Serving a small, growing percentage of users with a new version while watching error, refusal, latency, cost and feedback metrics, and rolling back on regression.
**2. In plain words:** Test the new dish on a few tables before the whole restaurant.
**3. Example to quote:** > "Hash the user id so the same user always gets the same variant; percent 0 is the rollback."
```python
router.set_percent(10) -> 50 -> 100  # or 0 to roll back
```
**Seen in:** `ragkit/governance.py`

## A/B test (split test, experiment)  `(concept)`
**1. Definition:** Serving fixed-weight variants to comparable users to measure a metric difference with enough sample size.
**2. In plain words:** Two menus, same customers, count which sells.
**3. Example to quote:** > "The salt is the experiment name; reusing it across experiments correlates their groups."
```python
ABRouter([(A, 1), (B, 1)], salt='exp-7')
```
**Seen in:** `ragkit/governance.py`

## Hash-based bucketing (sticky assignment)  `(function)`
**1. Definition:** Mapping an id to a pseudo-random number in [0, 1) via a hash and comparing it with thresholds, so assignment is deterministic and needs no table.
**2. In plain words:** Every user gets a fixed lottery number; the percentage decides the cut-off.
**3. Example to quote:** > "The candidate owns the lowest buckets, so ramping up only adds users."
```python
bucket('user-7', 'canary-1')  # same value every time
```
**Seen in:** `ragkit/governance.py`

## Guardrails (input/output validation, safety filters)  `(concept)`
**1. Definition:** Deterministic or model-based checks wrapped around an LLM call: limits, redaction, injection detection before; schema, grounding, policy checks after.
**2. In plain words:** Bumpers on both sides of the bowling lane.
**3. Example to quote:** > "Heuristic guards are bypassable; combine them with least privilege and output checks."
```python
build_input_guard(max_chars=500, max_tokens=150)
```
**Seen in:** `ragkit/governance.py`

## PII redaction (masking, scrubbing)  `(concept)`
**1. Definition:** Replacing personal data (email, phone, card number) with placeholders before the text is logged or sent to a third party; cards are confirmed with the Luhn checksum to limit false positives.
**2. In plain words:** Black out names and numbers before photocopying.
**3. Example to quote:** > "Redact before logging, not after."
```python
luhn_ok('4111111111111111')  # True -> [CARD]
```
**Seen in:** `ragkit/governance.py`, `ragkit/patterns.py`

## Luhn checksum  `(function)`
**1. Definition:** A mod-10 checksum over the digits of a card number: double every second digit from the right, subtract 9 when above 9, sum; valid if the total is divisible by 10.
**2. In plain words:** A typo detector for card numbers.
**3. Example to quote:** > "An order number that fails Luhn is not redacted as a card."
```python
luhn_ok('4111111111111112')  # False
```
**Seen in:** `ragkit/governance.py`

## Prompt-injection heuristics  `(concept)`
**1. Definition:** Weighted regex patterns that flag or block inputs that try to override instructions, leak the prompt or spoof delimiters.
**2. In plain words:** A bouncer who knows the usual fake IDs.
**3. Example to quote:** > "One layer only; paraphrases and other languages get through (see 08_agent_harness/07)."
```python
INJECTION_PATTERNS = [(r'ignore (all )?previous instructions', 1.0), ...]
```
**Seen in:** `ragkit/governance.py`

## Output validation (schema validation, structured output check)  `(concept)`
**1. Definition:** Parsing the model's reply as JSON and validating types, ranges and required fields with a schema, producing precise errors that can be shown back to the model.
**2. In plain words:** Checking the form is filled in correctly before accepting it.
**3. Example to quote:** > "`ValidationError` is a `ValueError` subclass: catch it first."
```python
StructuredAnswer.model_validate(extract_json(raw))
```
**Seen in:** `ragkit/governance.py`, `ragkit/schemas.py`

## Groundedness check (faithfulness, citation support)  `(concept)`
**1. Definition:** Verifying that every citation marker points to a chunk that was in the prompt and that the cited sentence is supported by that chunk's text.
**2. In plain words:** Checking the footnote really says what the sentence claims.
**3. Example to quote:** > "Lexical overlap here; an NLI model or LLM judge in production."
```python
CitationVerifier().verify(answer.answer, built.used)
```
**Seen in:** `ragkit/query.py`, `ragkit/governance.py`

## Repair vs refuse  `(concept)`
**1. Definition:** A format failure triggers one more call with the error shown to the model; a content failure (ungrounded, low confidence) triggers a fixed safe refusal.
**2. In plain words:** Fix a typo, but do not argue a made-up fact into truth.
**3. Example to quote:** > "An honest 'I don't know' is a pass, not a failure."
```python
OutputVerdict(action='repair' | 'refuse' | 'pass', ...)
```
**Seen in:** `ragkit/governance.py`

## Context-window budget (token budget, prompt budget)  `(concept)`
**1. Definition:** Dividing the model window among system prompt, history, retrieved chunks, the question and a reserved answer: window = system + history + chunks + question + reserve.
**2. In plain words:** Packing a suitcase: reserve the space you need on the way home first.
**3. Example to quote:** > "The reserve is subtracted first; drop oldest history and lowest-ranked chunks before anything else."
```python
ContextBudget(window=800, answer_reserve=150).plan(system, q, history)
```
**Seen in:** `ragkit/governance.py`

## Answer reserve (output headroom)  `(concept)`
**1. Definition:** Tokens deliberately left free in the window so the model has room to finish its reply.
**2. In plain words:** A cut-off answer is usually a prompt that ate the window.
**3. Example to quote:** > "`max_tokens = min(requested, cap, window - prompt_tokens)`."
```python
cb.output_limit(prompt_tokens, requested=300)
```
**Seen in:** `ragkit/governance.py`

## Tenant gate (per-tenant admission control)  `(class)`
**1. Definition:** Admission check per tenant combining a tokens-per-minute token bucket with a spend ceiling; reserves a worst-case estimate, then settles to real usage.
**2. In plain words:** A bouncer who knows each customer's tab and rate.
**3. Example to quote:** > "Rate exhausted = transient (`retry_after`); budget exhausted = permanent for the period."
```python
gate.admit('acme', est_tokens); ...; gate.settle('acme', est, resp)
```
**Seen in:** `ragkit/governance.py`, `ragkit/ratelimit.py`

## Reserve and settle (pre-authorization)  `(concept)`
**1. Definition:** Reserve the worst-case cost before a call and refund the unused part after it finishes.
**2. In plain words:** A hotel holds the maximum on your card and charges the real bill at checkout.
**3. Example to quote:** > "We reserve input + max output tokens; the model usually uses less."
```python
LLMRateLimiter.settle(estimated, actual)
```
**Seen in:** `ragkit/ratelimit.py`

## Cost ledger (usage log, metering table)  `(class)`
**1. Definition:** One row per request with tenant, release fingerprint, token counts, cost and outcome, including blocked and refused requests at zero cost.
**2. In plain words:** The till roll, one line per sale.
**3. Example to quote:** > "Billing, quota and quality reports read the same table."
```python
LedgerRow(request_id, tenant, release, fingerprint, in, out, cost_usd, outcome)
```
**Seen in:** `ragkit/governance.py`

## pydantic ValidationError  `(error)`
**1. Definition:** Raised by `model_validate` when input does not match the model; `.errors()` lists `{loc, msg, type}` per failing field.
**2. In plain words:** Pydantic's list of everything wrong with the form.
**3. Example to quote:** > "1 validation error: confidence Input should be less than or equal to 1"
```python
schema:confidence Input should be less than or equal to 1
```
**Seen in:** `ragkit/governance.py`

## PermanentError: context_window_too_small  `(error)`
**1. Definition:** Raised by `ContextBudget.plan` when system + question + answer reserve already exceed the window, so no history or chunk can fit.
**2. In plain words:** The suitcase is full before you pack.
**3. Example to quote:** > "Shorten the system prompt, lower the reserve, or choose a larger window."
```python
PermanentError: context_window_too_small: system+question+reserve = 163 > 150
```
**Seen in:** `ragkit/governance.py`
