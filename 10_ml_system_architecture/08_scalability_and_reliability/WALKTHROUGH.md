# Step 08 — Scalability, latency and reliability

## The story so far
Step 07 told us when the service is slow or failing. The natural next question is what to do when more users arrive than one process can handle.

## The pain point
Traffic doubles and latency climbs, queues grow without bound, retries pile on, and a small overload becomes an outage.

## The spark
Climb a ladder of answers: measure, cache, batch, add workers and replicas, then protect with timeouts, bounded queues and load shedding. Size with Little's Law.

## Goal
Learn the standard answers to "what happens when requests increase or latency rises?": cache, scale out, queue and shed load, time out and retry, and size capacity with arithmetic — then confirm with a load test.

## Run it
```bash
cd 10_ml_system_architecture/08_scalability_and_reliability
python main.py          # four small demos
python load_test.py     # ~20 s; starts the module-09 Flask API under Gunicorn with 1/2/4 workers
```
Real output of `main.py`:
```
1. cache: 300 requests -> 2 model runs, hits=298, misses=2, total 25 ms (without cache ~3000 ms)
2. retry with backoff:
   attempt 1: connection reset -> retrying
   attempt 2: connection reset -> retrying
   result: ok
   timeout: TIMEOUT -> return fallback answer / 504 to the client
3. burst of 40 into queue(max 5) with 2 workers: accepted=23, rejected (HTTP 429/503)=17, processed=23
4. capacity planning (Little's Law ...)
        50 ms          |    1    |       20 req/s  |    10
        50 ms          |    4    |       80 req/s  |    10
       200 ms          |    4    |       20 req/s  |    40
      1000 ms          |    4    |        4 req/s  |   200
```
(Step 3's accepted/rejected split depends on thread timing; arrivals every 10 ms outrun two workers that finish 40 jobs/s, so the queue fills and the excess is rejected.)
Real output of `load_test.py` (16 clients, `/slow?seconds=0.05`):
```
workers=1  clients=16  throughput=  24.3 req/s  p50= 833.7 ms  p95= 906.5 ms  errors=0
workers=2  clients=16  throughput=  40.7 req/s  p50= 419.6 ms  p95= 578.6 ms  errors=0
workers=4  clients=16  throughput=  79.0 req/s  p50= 210.7 ms  p95= 216.7 ms  errors=0
```
Throughput scales with workers (each does ~20 req/s at 50 ms/request). Latency per request is 50 ms of work **plus queueing**: with 16 clients and 1 worker each request waits behind ~15 others → ~800 ms. **Latency rises long before you "run out" of throughput.** (Slightly high throughput numbers: requests finishing just after the deadline are still counted.)

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Scaling up vs scaling out | vertical vs horizontal scaling | No | Up = bigger machine; out = more copies behind a load balancer. |
| Caching vs memoization | lru_cache, result cache, memo | Mostly | Memoization is caching within one process by function arguments; caching can also be shared (Redis). |
| Timeout vs retry vs circuit breaker | resilience patterns | No | Timeout: stop waiting; retry: try again; circuit breaker: stop trying for a while after repeated failures. |
| Load shedding vs backpressure | 429/503 rejection, bounded queue | Related | Backpressure tells producers to slow down; load shedding drops excess work. |
| Throughput vs latency vs concurrency | req/s, response time, in-flight requests | No | Little's Law links them: concurrency = throughput x latency. |
| Workers | processes, threads, replicas | No | A worker is one handler (process/thread); a replica is a whole copy of the service. |
| Load test vs benchmark | stress test, soak test | Related | Load test: realistic traffic; stress: beyond limits; soak: long duration. |

## Code, step by step

### 1. Imports (all standard library)  (`main.py` lines 14–18)

```python
import queue                                            # built-in: thread-safe FIFO queues
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import lru_cache                         # built-in: memoize a function (remember results)
```

- **In simple words:** Load the built-in tools for queues, threads, timing, thread pools and caching.
- **Exactly:** `queue.Queue` is a thread-safe FIFO; `ThreadPoolExecutor` runs functions in worker threads and returns `Future`s; `TimeoutError as FutureTimeout` renames the exception raised by `future.result(timeout=...)` (in Python 3.11+ it is the builtin `TimeoutError`); `lru_cache` memoizes function results.
- **Why it is here:** No pip installs: every reliability tool here ships with Python.

### 2. Cache: a counter and a memoized fake model  (`main.py` lines 20–28)

```python
# ----------------------------- 1. caching ---------------------------------------
CALLS = {"model": 0}


@lru_cache(maxsize=1024)                                # decorator: same arguments -> return the saved result
def cached_predict(text: str) -> str:
    CALLS["model"] += 1                                 # counts REAL model executions
    time.sleep(0.01)                                    # pretend inference takes 10 ms
    return "positive" if "good" in text else "negative"
```

- **In simple words:** Pretend the model takes 10 ms and remember answers so repeated texts are free.
- **Exactly:** `@lru_cache(maxsize=1024)` stores up to 1024 results keyed by the argument (`text` must be hashable); least-recently-used entries are evicted. `CALLS['model']` increments only when the body really runs. `time.sleep(0.01)` fakes 10 ms of inference.
- **Why it is here:** Real traffic repeats inputs; paying inference once per distinct input cuts cost and latency.

### 3. demo_cache: send 300 requests with repeats  (`main.py` lines 31–40)

```python
def demo_cache():
    traffic = ["good film", "bad film", "good film", "good film", "bad film", "good film"] * 50   # repeats are common
    start = time.perf_counter()
    for t in traffic:
        cached_predict(t)
    ms = (time.perf_counter() - start) * 1000
    info = cached_predict.cache_info()                  # namedtuple: hits, misses, maxsize, currsize
    print(f"1. cache: {len(traffic)} requests -> {CALLS['model']} model runs, hits={info.hits}, misses={info.misses}, "
          f"total {ms:.0f} ms (without cache ~{len(traffic) * 10} ms)")
    print("   Limits: only for repeated INPUTS, stale if the model changes (include model_version in the key!), uses memory.")
```

- **In simple words:** Fire 300 requests made of only two distinct texts and see how few reach the model.
- **Exactly:** `traffic` has 6 items x 50 = 300 requests, 2 distinct. `cache_info()` is a namedtuple (hits, misses, maxsize, currsize). Real output: `300 requests -> 2 model runs, hits=298, misses=2, total 25 ms (without cache ~3000 ms)`.
- **Why it is here:** Shows the best-case gain (120x) and the caveats printed on the last line: only exact repeats, staleness when the model changes, memory use.

### 4. Flaky downstream service  (`main.py` lines 43–52)

```python
# ----------------------------- 2. timeout + retry -------------------------------
attempts = {"n": 0}


def flaky_service() -> str:
    """Pretend downstream call (DB, another API) failing the first 2 times -- typical transient error."""
    attempts["n"] += 1
    if attempts["n"] < 3:
        raise ConnectionError(f"attempt {attempts['n']}: connection reset")
    return "ok"
```

- **In simple words:** A function that fails twice with a connection error and then succeeds, like a hiccuping database.
- **Exactly:** `attempts` is a mutable dict so the function can count calls without `global`. Calls 1 and 2 raise `ConnectionError`; call 3 returns `"ok"`.
- **Why it is here:** Gives the retry logic something realistic (a transient failure) to recover from.

### 5. with_retries: exponential backoff  (`main.py` lines 55–64)

```python
def with_retries(fn, retries: int = 4, base_delay: float = 0.05):
    """Retry with EXPONENTIAL BACKOFF: wait 0.05s, 0.1s, 0.2s ... so a struggling service is not hammered."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except ConnectionError as err:                  # retry ONLY errors that can be transient
            print(f"   {err} -> retrying" if attempt < retries else f"   {err} -> giving up")
            if attempt == retries:
                raise                                   # re-raise the original error after the last attempt
            time.sleep(base_delay * 2 ** (attempt - 1))  # ** is the power operator
```

- **In simple words:** Try again after a growing pause (0.05 s, 0.1 s, 0.2 s...) and give up after a fixed number of tries.
- **Exactly:** Loop `attempt` 1..4. On `ConnectionError` it prints, and if it was the last attempt, `raise` re-throws the original exception. Otherwise `time.sleep(base_delay * 2 ** (attempt-1))`. Only `ConnectionError` is caught, so bugs like `ValueError` are not retried. Real: two 'retrying' lines, then `result: ok`.
- **Why it is here:** Immediate hammering makes an overloaded service worse; unlimited retries never stop; retrying non-transient errors wastes time.

### 6. call_with_timeout: stop waiting after N seconds  (`main.py` lines 67–74)

```python
def call_with_timeout(fn, seconds: float):
    """Never wait forever: a hung dependency would pile up requests until the whole service falls over."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=seconds)
        except FutureTimeout:
            return "TIMEOUT -> return fallback answer / 504 to the client"
```

- **In simple words:** Run the call in a helper thread and give up waiting if it takes too long.
- **Exactly:** `pool.submit(fn)` returns a `Future`; `future.result(timeout=seconds)` waits at most that long and raises `TimeoutError` otherwise. Caveat: the thread is not killed, and leaving the `with` block still waits for it to finish (here 0.5 s). Real: returns the TIMEOUT fallback string.
- **Why it is here:** A hung dependency would otherwise pile up waiting requests until the whole service falls over.

### 7. demo_retry_timeout: run both  (`main.py` lines 77–81)

```python
def demo_retry_timeout():
    print("2. retry with backoff:")
    print("   result:", with_retries(flaky_service))
    print("   timeout:", call_with_timeout(lambda: time.sleep(0.5) or "slow result", seconds=0.1))
    print("   Rule: retry only idempotent operations (safe to repeat); always add timeouts AND a max retry count.")
```

- **In simple words:** Show a retry that succeeds on attempt 3 and a 0.5 s call cut off at 0.1 s.
- **Exactly:** `lambda: time.sleep(0.5) or "slow result"`: `sleep` returns `None`, so `or` yields the string. Printed: `result: ok` and `timeout: TIMEOUT -> return fallback answer / 504 to the client`. Rule printed: retry only idempotent operations.
- **Why it is here:** Retrying a non-idempotent call (e.g. charging a card) could do the action twice.

### 8. Bounded queue and worker threads  (`main.py` lines 84–98)

```python
# ----------------------------- 3. bounded queue / load shedding ----------------
def demo_queue():
    work: queue.Queue = queue.Queue(maxsize=5)          # BOUNDED: at most 5 waiting jobs
    done = []

    def worker():
        while True:
            item = work.get()                           # blocks until a job is available
            if item is None:                            # sentinel value = "stop"
                return
            time.sleep(0.05)                            # processing time per job
            done.append(item)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
```

- **In simple words:** Make a waiting room with only 5 seats and two workers who take jobs from it.
- **Exactly:** `Queue(maxsize=5)`: `put` blocks and `put_nowait` raises `queue.Full` when 5 items are waiting. `work.get()` blocks until a job exists. `None` is a sentinel meaning stop. Each job takes 0.05 s, so 2 workers finish 40 jobs/s. Threads are started with `t.start()`.
- **Why it is here:** Bounding the queue limits both memory and waiting time.

### 9. Send a burst of 40 and shed what does not fit  (`main.py` lines 99–107)

```python
        t.start()
    accepted = rejected = 0
    for i in range(40):                                 # a burst of 40 requests arrives at once
        try:
            work.put_nowait(i)                          # raises queue.Full instead of waiting
            accepted += 1
        except queue.Full:
            rejected += 1                               # LOAD SHEDDING: answer 429/503 fast instead of timing out slowly
        time.sleep(0.01)                                # requests arrive every 10 ms (100/s) but 2 workers only finish 40/s
```

- **In simple words:** Requests arrive faster than they can be served; extra ones are turned away immediately.
- **Exactly:** Requests arrive every 10 ms (100/s) while capacity is 40/s. `put_nowait` raises `queue.Full` -> `rejected += 1` (in a web app: HTTP 429/503). Load shedding means failing fast instead of failing slowly.
- **Why it is here:** An unbounded queue would accept everything and every user would wait longer and longer.

### 10. Shut down workers and print the counts  (`main.py` lines 108–113)

```python
    for _ in threads:
        work.put(None)
    for t in threads:
        t.join()
    print(f"3. burst of 40 into queue(max 5) with 2 workers: accepted={accepted}, rejected (HTTP 429/503)={rejected}, processed={len(done)}")
    print("   An UNBOUNDED queue never rejects but latency grows without limit and memory runs out -- bounded + backpressure is safer.")
```

- **In simple words:** Tell each worker to stop, wait for them, and report who was accepted or rejected.
- **Exactly:** One `None` per worker via blocking `put`; `join()` waits for the thread to exit. Real output (varies with timing): `accepted=23, rejected (HTTP 429/503)=17, processed=23`.
- **Why it is here:** Without the sentinel the workers loop forever and the program never exits.

### 11. Capacity maths with Little's Law  (`main.py` lines 116–124)

```python
# ----------------------------- 4. capacity maths --------------------------------
def demo_capacity():
    print("4. capacity planning (Little's Law: concurrent requests = arrival rate x latency)")
    print("   latency per request | workers | max throughput | workers needed for 200 req/s")
    for latency_s, workers in [(0.05, 1), (0.05, 4), (0.2, 4), (1.0, 4)]:
        per_worker = 1 / latency_s                      # a sync worker does one request at a time
        needed = 200 * latency_s                        # = arrival rate x latency
        print(f"   {latency_s*1000:>7.0f} ms          | {workers:^7} | {per_worker*workers:>8.0f} req/s  | {needed:>5.0f}")
    print("   Slow model => you need MORE workers/replicas, OR faster inference (smaller model, quantization, batching, GPU).")
```

- **In simple words:** Work out how many workers you need for a target load.
- **Exactly:** Little's Law: concurrency L = arrival rate x latency. A sync worker handles `1/latency` req/s. For 200 req/s at 50 ms you need 200 x 0.05 = 10 workers; at 200 ms, 40; at 1000 ms, 200 (real table). `{x:>7.0f}`, `{workers:^7}` are right/centre alignment formats.
- **Why it is here:** Turns 'will it hold?' into arithmetic before you pay for hardware.

### 12. Run all four demos  (`main.py` lines 127–133)

```python
if __name__ == "__main__":
    demo_cache()
    demo_retry_timeout()
    demo_queue()
    demo_capacity()
    # TRY IT: change maxsize=5 to 50 and 1 -- how do accepted/rejected change?
    # TRY IT: make flaky_service fail 5 times and see "giving up" with the re-raised ConnectionError traceback.
```

- **In simple words:** When run as a script, execute each demo in order.
- **Exactly:** `if __name__ == "__main__":` runs only when executed directly, not when imported. The `# TRY IT:` lines are exercises.
- **Why it is here:** Keeps the file importable without side effects.

### 13. load_test.py: imports  (`load_test.py` lines 12–19)

```python
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
```

- **In simple words:** Tools for stats, starting a server process, threads, timing and HTTP calls.
- **Exactly:** `statistics.quantiles` computes percentiles; `subprocess.Popen` starts Gunicorn; `httpx` is the HTTP client (needs the venv).
- **Why it is here:** A load test needs real concurrent clients against a real server.

### 14. Constants: which app and port  (`load_test.py` lines 21–23)

```python
APP_DIR = Path(__file__).resolve().parents[2] / "09_api_development" / "06_model_api_flask_vs_fastapi"
PORT = 8125
BASE = f"http://127.0.0.1:{PORT}"
```

- **In simple words:** Point at the module-09 Flask app and a local port.
- **Exactly:** `APP_DIR` is `09_api_development/06_model_api_flask_vs_fastapi`; `BASE` is `http://127.0.0.1:8125`.
- **Why it is here:** Reuses the earlier API as the system under test.

### 15. client_loop: one simulated user  (`load_test.py` lines 26–35)

```python
def client_loop(stop_at: float, latencies: list, errors: list) -> None:
    """One simulated user: send requests back-to-back until the deadline."""
    with httpx.Client(base_url=BASE, timeout=30) as client:
        while time.perf_counter() < stop_at:
            start = time.perf_counter()
            try:
                ok = client.get("/slow", params={"seconds": 0.05}).status_code == 200
            except httpx.HTTPError:
                ok = False
            (latencies if ok else errors).append((time.perf_counter() - start) * 1000)   # list.append is thread-safe in CPython
```

- **In simple words:** Each user sends requests back to back until time is up and records how long each took.
- **Exactly:** `GET /slow?seconds=0.05` sleeps 50 ms server-side (stand-in for a 50 ms model). Successful latencies go in `latencies`, failures in `errors`. `list.append` is thread-safe in CPython.
- **Why it is here:** Closed-loop clients measure what the server can sustain.

### 16. run(): start Gunicorn and wait until healthy  (`load_test.py` lines 38–47)

```python
def run(workers: int, clients: int = 16, duration: float = 3.0) -> None:
    server = subprocess.Popen([sys.executable, "-m", "gunicorn", "flask_app:app", "--workers", str(workers),
                               "--bind", f"127.0.0.1:{PORT}", "--log-level", "error"], cwd=APP_DIR)
    try:
        for _ in range(100):
            try:
                httpx.get(BASE + "/health", timeout=1)
                break
            except httpx.TransportError:
                time.sleep(0.2)
```

- **In simple words:** Launch the server with N workers and poll `/health` until it answers.
- **Exactly:** `sys.executable -m gunicorn flask_app:app --workers N` starts N processes; polling up to 100 times, 0.2 s apart, avoids measuring startup.
- **Why it is here:** Without the wait the first requests would fail with connection errors.

### 17. run(): spawn clients, measure, print  (`load_test.py` lines 48–58)

```python
        latencies: list = []
        errors: list = []
        stop_at = time.perf_counter() + duration
        threads = [threading.Thread(target=client_loop, args=(stop_at, latencies, errors)) for _ in range(clients)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        q = statistics.quantiles(latencies, n=100)       # 99 cut points -> q[49]=p50, q[94]=p95
        print(f"workers={workers}  clients={clients}  throughput={len(latencies)/duration:6.1f} req/s  "
              f"p50={q[49]:6.1f} ms  p95={q[94]:6.1f} ms  errors={len(errors)}")
```

- **In simple words:** Start 16 client threads for 3 seconds, then compute throughput and percentiles.
- **Exactly:** `statistics.quantiles(latencies, n=100)` gives 99 cut points: `q[49]` is p50, `q[94]` is p95. Throughput = successful requests / duration. Real: workers=1 24.0 req/s p50=842 ms; workers=2 41.0 req/s p50=428 ms; workers=4 61.0 req/s p50=252 ms.
- **Why it is here:** Shows throughput scaling with workers, and latency shrinking because 16 clients queue less.

### 18. run(): always stop the server; main loop  (`load_test.py` lines 59–68)

```python
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    for w in (1, 2, 4):
        run(w)
    # TRY IT: raise clients to 64 and watch p95 explode on workers=1 (requests queue up waiting for a free worker).
    # TRY IT: point it at /predict instead (POST) -- a microsecond model makes HTTP overhead the bottleneck.
```

- **In simple words:** Stop Gunicorn even on error, and test 1, 2 and 4 workers.
- **Exactly:** `finally` runs `terminate()` then `wait(timeout=10)`, so no stray server holds port 8125.
- **Why it is here:** A leaked server would break the next run with 'address in use'.


## When requests increase — the ladder of answers
1. **Measure first** (p95 latency, error rate, CPU, RAM, queue depth).
2. **Cheap wins:** cache repeated inputs, batch requests, trim input size, smaller/quantized/distilled model.
3. **Scale up:** more workers per machine (RAM = workers × model size!).
4. **Scale out:** more replicas behind a load balancer; autoscale on CPU/queue depth/latency; keep services stateless.
5. **Protect:** timeouts, bounded queues, load shedding (429/503), rate limits, circuit breakers, graceful degradation (fallback model or cached answer).
6. **Decouple:** put slow work behind a queue (async job + polling/webhook) when users needn't wait.
When **latency rises** instead: profile (model vs preprocessing vs network vs DB), check cold starts, warm up the model, check noisy neighbours, check payload size, look at p99 not average.

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `@lru_cache(maxsize=1024)` | Memoize: same args → saved result | — |
| `cached_predict.cache_info()` | hits/misses/size | — |
| `base_delay * 2 ** (attempt - 1)` | Exponential backoff (`**` = power) | 0.05, 0.1, 0.2 … |
| `raise` (bare) inside `except` | Re-raise the same error | — |
| `queue.Queue(maxsize=5)`, `put_nowait`, `queue.Full` | Bounded FIFO between threads; `put_nowait` fails instead of blocking | — |
| `None` as a **sentinel** | A special value meaning "stop" | `if item is None: return` |
| `threading.Thread(target=fn)`, `.start()`, `.join()` | Run in background; wait to finish | — |
| `fn() or "x"` trick: `time.sleep(0.5) or "slow result"` | `sleep` returns `None`, so `or` yields the string | — |
| `statistics.quantiles(data, n=100)` | 99 cut points → `[49]`=p50, `[94]`=p95 | — |

## Libraries & functions
| Name | Kind | Does |
|---|---|---|
| `functools.lru_cache` | stdlib | In-memory cache keyed by arguments (must be hashable) |
| `concurrent.futures.ThreadPoolExecutor.submit(...).result(timeout=)` | stdlib | Run in a thread, wait at most `timeout` s (the thread itself can't be killed) |
| `httpx.Client` | 3rd-party | Reused connections in the load tester |
| `subprocess.Popen([... "gunicorn" ...])` | stdlib | Start the server under test |

## Little's Law (the one formula to remember)
`concurrent requests in the system = arrival rate × time each spends in the system`.
Need 200 req/s at 0.2 s each → 40 requests in flight → ≥40 sync workers (or async/threads/batching). A slow model multiplies the infrastructure bill — which is why inference optimisation (module 06) is also a *scalability* topic.

## Reliability vocabulary
- **Timeout:** never wait forever. **Retry with backoff + jitter:** only for idempotent operations, with a cap. **Circuit breaker:** stop calling a failing dependency for a while. **Idempotency:** repeating a request has the same effect as doing it once. **Graceful degradation:** return something useful (cached/fallback) instead of an error. **Health vs readiness checks:** "process alive" vs "model loaded, ready for traffic" (used by Docker/Kubernetes, module 11). **SLO:** a target such as "p95 < 300 ms, 99.9% success".

## Experiments
1. `maxsize=5` → `50` and `1` in the queue demo.
2. `load_test.py`: `clients=64` → p95 for workers=1.
3. Point the load test at `/predict` (POST) and see that a microsecond model makes HTTP overhead the limit.

## Debugging practice
Load test shows `errors=...` > 0? Read the exception class: `ReadTimeout` (server too slow/queueing), `ConnectError` (server down or port wrong), 5xx (app crash — check the **server's** logs). Reproduce with one `curl` before blaming load.

## Rebuild checklist
- [ ] Write a retry loop with exponential backoff.
- [ ] Explain why a bounded queue is better than an unbounded one.
- [ ] Calculate workers needed for 100 req/s at 150 ms.
- [ ] List 5 things you'd do if p95 latency doubled.

## The new problem (cliffhanger)
Scaling handles more of the same traffic, but not different traffic: the data itself can change under a model that scales perfectly. Next: [09_drift_monitoring](../09_drift_monitoring/).

## Say it aloud
> **Say it aloud (interview box):** If traffic grows I first measure, then cache, batch and shrink the model, then add workers and replicas behind a load balancer with autoscaling. I add timeouts, bounded queues and load shedding so overload degrades gracefully, and I size capacity with Little's Law, where in-flight requests equal arrival rate times latency. The costs are that caching can serve stale answers, retries can amplify an outage, and every replica costs memory and money.
