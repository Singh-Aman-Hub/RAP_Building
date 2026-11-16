# Terms — LLD interview problems: six worked designs on top of ragkit

Folder: `20_llm_system_lld/10_lld_interview_problems` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [LLD interview frame (requirements -> entities -> interfaces -> diagrams -> edge cases -> tests)](#lld-interview-frame-requirements---entities---interfaces---diagrams---edge-cases---tests--concept) · [Composition root](#composition-root--concept) · [Token bucket with reserve and settle](#token-bucket-with-reserve-and-settle--concept) · [RPM and TPM limits](#rpm-and-tpm-limits--concept) · [Per-tenant isolation](#per-tenant-isolation--concept) · [Eval harness (scorers, cases, report)](#eval-harness-scorers-cases-report--concept) · [Regression gate](#regression-gate--concept) · [Conversation memory (window, summary, TTL)](#conversation-memory-window-summary-ttl--concept) · [Tool registry (errors as observations)](#tool-registry-errors-as-observations--concept) · [Loop guard and step limit](#loop-guard-and-step-limit--concept) · [Indirect prompt injection via tool output](#indirect-prompt-injection-via-tool-output--concept) · [AttributeError: 'NoneType' object has no attribute 'split'](#attributeerror-nonetype-object-has-no-attribute-split--error)

---

## LLD interview frame (requirements -> entities -> interfaces -> diagrams -> edge cases -> tests)  `(concept)`
**1. Definition:** A fixed order for answering an object-design question so no part is forgotten.
**2. In plain words:** A checklist before take-off.
**3. Example to quote:** > "State the frame first; it shows structure."
**Seen in:** `WALKTHROUGH.md`, theory

## Composition root  `(concept)`
**1. Definition:** The single place where concrete classes are chosen and connected, e.g. build_system(config).
**2. In plain words:** The switchboard.
**3. Example to quote:** > "No other module should name every implementation."
**Seen in:** `ragkit/factory.py`

## Token bucket with reserve and settle  `(concept)`
**1. Definition:** Reserve an estimated number of tokens before a call, then refund the difference once the real usage is known.
**2. In plain words:** Pre-authorising a payment then charging the exact amount.
**3. Example to quote:** > "Reserve worst case (input + max output) so the limit is never exceeded."
```python
limiter.settle(estimated, actual)
```
**Seen in:** `ragkit/ratelimit.py`

## RPM and TPM limits  `(concept)`
**1. Definition:** Requests per minute and tokens per minute, the two limits LLM providers apply.
**2. In plain words:** A speed limit and a weight limit.
**3. Example to quote:** > "A limiter must check both before spending either."
**Seen in:** `ragkit/ratelimit.py`

## Per-tenant isolation  `(concept)`
**1. Definition:** Separate limiter, cache scope and memory key per tenant.
**2. In plain words:** Separate lanes on a road.
**3. Example to quote:** > "A noisy neighbour cannot starve or read another tenant."
**Seen in:** `ragkit/ratelimit.py`

## Eval harness (scorers, cases, report)  `(concept)`
**1. Definition:** Code that runs labelled cases through an answerer and aggregates metric scores.
**2. In plain words:** A test suite for answer quality.
**3. Example to quote:** > "Scorers return None when not applicable; crashes become rows, not aborts."
**Seen in:** `ragkit/evalkit.py`

## Regression gate  `(concept)`
**1. Definition:** A CI check that fails when cases that used to pass now fail.
**2. In plain words:** A turnstile that stops you re-introducing old bugs.
**3. Example to quote:** > "regressions(base, new, scorer) lists the case ids."
**Seen in:** `ragkit/evalkit.py`

## Conversation memory (window, summary, TTL)  `(concept)`
**1. Definition:** Per-session history trimmed to a token budget, with evicted turns summarised and idle sessions expired.
**2. In plain words:** A notepad with a page limit and a one-line recap on top.
**3. Example to quote:** > "Never return an empty window; keep names and ids verbatim in summaries."
**Seen in:** `ragkit/memory.py`

## Tool registry (errors as observations)  `(concept)`
**1. Definition:** A name-to-function table whose call() never raises: unknown tools, bad arguments and tool exceptions come back as results the model can read.
**2. In plain words:** A vending machine that displays an error message instead of jamming.
**3. Example to quote:** > "Lets the model self-correct without crashing the loop."
**Seen in:** `ragkit/agent.py`

## Loop guard and step limit  `(concept)`
**1. Definition:** Stopping an agent that repeats the same tool call or exceeds a maximum number of steps.
**2. In plain words:** A circuit breaker for agents.
**3. Example to quote:** > "Bounds cost and latency."
```python
sig = (call.name, json.dumps(call.arguments, sort_keys=True))
```
**Seen in:** `ragkit/agent.py`

## Indirect prompt injection via tool output  `(concept)`
**1. Definition:** Instructions hidden in data a tool returns (a web page, a document) that try to steer the model.
**2. In plain words:** A forged note slipped into the files.
**3. Example to quote:** > "Treat tool output as data; gate side-effect tools."
**Seen in:** `WALKTHROUGH.md`, theory

## AttributeError: 'NoneType' object has no attribute 'split'  `(error)`
**1. Definition:** A None reached code that expects a string.
**2. In plain words:** Handing over an empty envelope.
**3. Example to quote:** > "Validate input types at the boundary, near the cause."
**Seen in:** `WALKTHROUGH.md`, Debugging practice
