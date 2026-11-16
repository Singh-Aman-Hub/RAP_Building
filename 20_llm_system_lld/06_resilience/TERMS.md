# Terms — Resilience: retries, timeouts, circuit breaker, rate limiter, fallbacks, idempotency

Folder: `20_llm_system_lld/06_resilience` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [20 README](../README.md)

**Terms in this folder:** [Transient vs permanent error](#transient-vs-permanent-error--concept) · [Exponential backoff](#exponential-backoff--concept) · [Jitter (full jitter)](#jitter-full-jitter--concept) · [Thundering herd (retry storm)](#thundering-herd-retry-storm--concept) · [Retry-After header](#retry-after-header--concept) · [Timeout and deadline](#timeout-and-deadline--concept) · [Circuit breaker (closed, open, half-open)](#circuit-breaker-closed-open-half-open--concept) · [Token bucket](#token-bucket--concept) · [Fallback chain (provider failover)](#fallback-chain-provider-failover--concept) · [Idempotency key](#idempotency-key--concept) · [Injectable clock / fake clock](#injectable-clock--fake-clock--concept) · [time.monotonic](#timemonotonic--library) · [CallTimeout / CircuitOpenError](#calltimeout--circuitopenerror--error)

---

## Transient vs permanent error  `(concept)`
**1. Definition:** Transient errors may succeed if retried (500, 504, 529, connection reset, 429); permanent ones will not (400, 401, 403, 404).
**2. In plain words:** A busy line versus a wrong number.
**3. Example to quote:** > "Only transient errors enter retry and fallback."
**Seen in:** `ragkit/resilience.py`

## Exponential backoff  `(concept)`
**1. Definition:** Waiting longer after each failed attempt, doubling the delay up to a cap: base * 2**attempt.
**2. In plain words:** Knocking, then waiting twice as long before each next knock.
**3. Example to quote:** > "Backoff reduces load on a struggling server."
**Seen in:** `ragkit/resilience.py`

## Jitter (full jitter)  `(concept)`
**1. Definition:** Randomising the delay, here uniform(0, cap), so many clients do not retry at the same moment.
**2. In plain words:** Not everyone leaving the stadium through one gate at once.
**3. Example to quote:** > "Without jitter 1000 clients retried in one slot; with it about 100 per slot."
```python
rng.uniform(0, min(max_delay, base * 2**attempt))
```
**Seen in:** `ragkit/resilience.py`

## Thundering herd (retry storm)  `(concept)`
**1. Definition:** Many clients retrying simultaneously after a failure, overwhelming the recovering service.
**2. In plain words:** Everyone rushing the door when it reopens.
**3. Example to quote:** > "Jitter, caps on attempts and circuit breakers prevent it."
**Seen in:** `WALKTHROUGH.md`, Experiments

## Retry-After header  `(concept)`
**1. Definition:** A response header telling the client how long to wait before retrying (usually on 429).
**2. In plain words:** 'Come back in 2 seconds.'
**3. Example to quote:** > "Take the larger of your backoff and retry-after."
```python
delay = max(delay, exc.retry_after)
```
**Seen in:** `ragkit/resilience.py`

## Timeout and deadline  `(concept)`
**1. Definition:** A timeout limits one wait; a deadline limits the whole operation including retries.
**2. In plain words:** A kitchen timer versus the dinner reservation time.
**3. Example to quote:** > "Python cannot kill a thread, so also set socket timeouts in the HTTP client."
```python
future.result(timeout=seconds)
```
**Seen in:** `ragkit/resilience.py`

## Circuit breaker (closed, open, half-open)  `(concept)`
**1. Definition:** A guard that stops calls after repeated failures (open), then lets one probe through (half-open) to test recovery.
**2. In plain words:** A fuse box.
**3. Example to quote:** > "Permanent errors must not count as failures."
**Seen in:** `ragkit/resilience.py`

## Token bucket  `(concept)`
**1. Definition:** A rate limiter: a bucket of capacity tokens refilled at rate per second; requests spend tokens; allows bursts then a steady rate.
**2. In plain words:** A bucket of tickets that refills slowly.
**3. Example to quote:** > "Lazy refill from timestamps needs no background thread."
```python
tokens = min(capacity, tokens + elapsed * rate)
```
**Seen in:** `ragkit/resilience.py`

## Fallback chain (provider failover)  `(concept)`
**1. Definition:** Trying a list of providers in order, moving on only for infrastructure errors.
**2. In plain words:** A backup generator.
**3. Example to quote:** > "Record which provider answered; permanent errors do not fall back."
**Seen in:** `ragkit/resilience.py`

## Idempotency key  `(concept)`
**1. Definition:** A client-chosen key identifying one logical operation so repeated requests have the effect of one.
**2. In plain words:** A cheque number: cashing it twice does nothing.
**3. Example to quote:** > "Derive the key from the operation, not randomly, so a retry recomputes it."
```python
idempotency_key('charge-user-42', 'order-1001')
```
**Seen in:** `ragkit/resilience.py`

## Injectable clock / fake clock  `(concept)`
**1. Definition:** Passing a time source as a parameter so tests can control time.
**2. In plain words:** A TV remote for time.
**3. Example to quote:** > "FakeClock.sleep advances time instantly, so backoff tests are exact and fast."
```python
clock = FakeClock(); clock.advance(30)
```
**Seen in:** `ragkit/testing.py`

## time.monotonic  `(library)`
**1. Definition:** A clock that never goes backwards, suitable for measuring durations.
**2. In plain words:** A stopwatch rather than a wall clock.
**3. Example to quote:** > "Wall clocks can jump (NTP, daylight saving)."
**Seen in:** `ragkit/resilience.py`

## CallTimeout / CircuitOpenError  `(error)`
**1. Definition:** CallTimeout: the wait exceeded the limit; CircuitOpenError: the breaker rejected the call without trying.
**2. In plain words:** 'No answer in time' and 'not even trying'.
**3. Example to quote:** > "CallTimeout is retryable; CircuitOpenError triggers fallback."
**Seen in:** `ragkit/resilience.py`
