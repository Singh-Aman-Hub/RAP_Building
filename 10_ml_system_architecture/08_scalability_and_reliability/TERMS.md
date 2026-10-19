# Terms — Step 08: Scalability and reliability

Folder: `10_ml_system_architecture/08_scalability_and_reliability` · Files covered: `main.py`, `load_test.py`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Scalability](#scalability-concept) · [Caching (lru_cache)](#caching-lru_cache-standard-library) · [Decorator](#decorator-syntax) · [cache_info()](#cache_info-standard-library) · [Hashable arguments for caching](#hashable-arguments-for-caching-concept) · [Cache staleness and keys](#cache-staleness-and-keys-concept) · [Timeout](#timeout-concept) · [ThreadPoolExecutor.submit and result(timeout)](#threadpoolexecutorsubmit-and-resulttimeout-standard-library) · [Retry with exponential backoff](#retry-with-exponential-backoff-concept) · [Idempotency](#idempotency-concept) · [Retrying only transient errors](#retrying-only-transient-errors-concept) · [Exponent operator (**) and bare raise](#exponent-operator-and-bare-raise-syntax) · [time.sleep(...) or "x" trick](#timesleep-or-x-trick-syntax) · [Bounded queue](#bounded-queue-concept) · [Load shedding and backpressure](#load-shedding-and-backpressure-concept) · [queue.Queue, put_nowait, queue.Full](#queuequeue-put_nowait-queuefull-standard-library) · [Sentinel value](#sentinel-value-concept) · [threading.Thread, start, join](#threadingthread-start-join-standard-library) · [Little's Law](#littles-law-concept) · [Capacity planning](#capacity-planning-concept) · [Scale up vs scale out](#scale-up-vs-scale-out-concept) · [Load balancer and autoscaling](#load-balancer-and-autoscaling-concept) · [Load test](#load-test-concept) · [Queueing delay](#queueing-delay-concept) · [Throughput](#throughput-concept) · [statistics.quantiles](#statisticsquantiles-standard-library) · [Simulated client loop](#simulated-client-loop-concept) · [httpx.Client and httpx.HTTPError](#httpxclient-and-httpxhttperror-third-party-httpx) · [Starting and stopping a server under test](#starting-and-stopping-a-server-under-test-standard-library) · [Reliability vocabulary (circuit breaker, graceful degradation, SLO)](#reliability-vocabulary-circuit-breaker-graceful-degradation-slo-concept) · [Health vs readiness checks](#health-vs-readiness-checks-concept) · [When latency rises: what to check](#when-latency-rises-what-to-check-concept) · [Multiple assignment and defaults in demos](#multiple-assignment-and-defaults-in-demos-syntax) · [Re-raise and exception types in demos](#re-raise-and-exception-types-in-demos-syntax)

---

## Scalability  `(concept)`
**1. Definition:** A system's ability to handle more load (requests, data) by adding resources or becoming more efficient.
**2. In plain words:** When traffic grows you first measure, then take cheap wins (cache, batch, smaller model), then add workers and replicas, and protect the system so overload degrades gracefully. It always costs memory and money.
**3. Example to quote:**
> "If traffic doubles I check p95 latency, add caching and batching, then more workers and replicas behind a load balancer."

**Seen in:** whole step

## Caching (lru_cache)  `(standard library)`
**1. Definition:** Storing the result of a computation so that repeated calls with the same inputs return the saved result; `functools.lru_cache(maxsize)` does this for a function and evicts the least recently used entries.
**2. In plain words:** The demo had 300 requests but only 2 distinct texts, so the model ran twice and the other 298 were hits. Limits: it helps only for repeated inputs, it can serve stale answers when the model changes (include the model version in the key), and it uses memory.
**3. Example to quote:**
> "`@lru_cache(maxsize=1024)` took 300 requests down to 2 model runs, 25 ms instead of about 3000 ms."

```python
from functools import lru_cache
@lru_cache(maxsize=128)
def f(x): return x * 2
```

**Seen in:** `cached_predict`

## Decorator  `(syntax)`
**1. Definition:** A function applied to another function using `@name` above the definition, returning a modified version of it.
**2. In plain words:** `@lru_cache` wraps `cached_predict` so it first checks a saved-results table. You get caching without changing the function body.
**3. Example to quote:**
> "The `@lru_cache(maxsize=1024)` line above `cached_predict` adds memoization with no other code change."

**Seen in:** `cached_predict`

## cache_info()  `(standard library)`
**1. Definition:** A method on an `lru_cache`-decorated function returning a named tuple with `hits`, `misses`, `maxsize` and `currsize`.
**2. In plain words:** The hit rate shows whether the cache is working. Here hits=298 and misses=2.
**3. Example to quote:**
> "`cached_predict.cache_info()` says hits=298, misses=2, so the cache avoided almost all model runs."

**Seen in:** `demo_cache`

## Hashable arguments for caching  `(concept)`
**1. Definition:** `lru_cache` uses the function's arguments as dictionary keys, so they must be hashable (strings, numbers, tuples), not lists or dicts.
**2. In plain words:** That is why the cached function takes a `str`. Passing a list raises `TypeError: unhashable type`.
**3. Example to quote:**
> "My cached function takes the text string, which is hashable; a list of texts would need converting to a tuple first."

**Seen in:** `WALKTHROUGH.md`

## Cache staleness and keys  `(concept)`
**1. Definition:** A cached answer can become wrong if the model or data it came from changes.
**2. In plain words:** Include `model_version` in the cache key, or clear the cache on a new model, otherwise users could get answers from an old version.
**3. Example to quote:**
> "After promoting v2 I must not serve v1's cached answers, so the cache key includes the model version."

**Seen in:** `demo_cache` print

## Timeout  `(concept)`
**1. Definition:** A maximum time to wait for an operation before giving up.
**2. In plain words:** A hung dependency can pile up waiting requests until the whole service falls over. Always set a timeout and decide what to do on expiry: return a fallback answer or a 504.
**3. Example to quote:**
> "`future.result(timeout=0.1)` gave up on a 0.5 s call and returned a fallback instead of waiting."

**Seen in:** `call_with_timeout`

## ThreadPoolExecutor.submit and result(timeout)  `(standard library)`
**1. Definition:** `pool.submit(fn)` runs `fn` in a worker thread and returns a `Future`; `future.result(timeout=s)` waits at most `s` seconds and raises `TimeoutError` otherwise.
**2. In plain words:** It adds a timeout to a function that has none. Note the background thread itself cannot be killed; the caller just stops waiting.
**3. Example to quote:**
> "I submit the slow call to a pool and wait 0.1 seconds; on `FutureTimeout` I return the fallback message."

**Seen in:** `call_with_timeout`

## Retry with exponential backoff  `(concept)`
**1. Definition:** Repeating a failed operation after waits that double each time (for example 0.05 s, 0.1 s, 0.2 s), up to a maximum number of attempts.
**2. In plain words:** Transient errors often clear on their own. Waiting longer each time avoids hammering a struggling service. Add jitter (random variation) in real systems, and cap the attempts.
**3. Example to quote:**
> "My flaky call failed twice then succeeded on attempt 3; the waits were 0.05 s and 0.1 s."

**Seen in:** `with_retries`

## Idempotency  `(concept)`
**1. Definition:** An operation is idempotent if repeating it has the same effect as doing it once.
**2. In plain words:** Only retry idempotent operations. Reading a prediction is safe to repeat; charging a card is not unless it is made idempotent with a unique key.
**3. Example to quote:**
> "Retrying a prediction is harmless, but retrying a payment could charge twice, so I'd add an idempotency key."

**Seen in:** `demo_retry_timeout` print, `WALKTHROUGH.md`

## Retrying only transient errors  `(concept)`
**1. Definition:** Catching specific exception types (here `ConnectionError`) in the retry loop, instead of every exception.
**2. In plain words:** A programming bug or invalid input will fail again every time, so retrying only wastes time. Retries are for conditions that may clear on their own.
**3. Example to quote:**
> "`except ConnectionError` retries network blips, but a `ValueError` from bad input passes straight through."

**Seen in:** `with_retries`

## Exponent operator (**) and bare raise  `(syntax)`
**1. Definition:** `2 ** n` is two to the power n; a bare `raise` inside `except` re-raises the original exception with its traceback.
**2. In plain words:** The power operator builds the doubling delay. After the final attempt the original error is re-raised so the caller sees what really failed.
**3. Example to quote:**
> "`base_delay * 2 ** (attempt - 1)` gives 0.05, 0.1, 0.2, and `raise` after the last attempt passes the original `ConnectionError` up."

```python
print(0.05 * 2 ** 2)   # 0.2
```

**Seen in:** `with_retries`

## time.sleep(...) or "x" trick  `(syntax)`
**1. Definition:** `time.sleep(0.5) or "slow result"` evaluates the sleep (returns `None`, which is falsy), so the `or` expression yields the string.
**2. In plain words:** A one-line lambda that sleeps and then returns a value. It is a compact trick for demos; in real code you would write a normal function.
**3. Example to quote:**
> "`lambda: time.sleep(0.5) or 'slow result'` sleeps half a second and returns the text."

**Seen in:** `demo_retry_timeout`

## Bounded queue  `(concept)`
**1. Definition:** A FIFO queue with a maximum size (`queue.Queue(maxsize=5)`), so only a limited number of jobs can wait.
**2. In plain words:** An unbounded queue never rejects work but latency grows without limit and memory runs out. A bounded queue forces a decision when full: reject quickly.
**3. Example to quote:**
> "My queue holds at most 5 waiting jobs; once it is full new requests are rejected immediately."

**Seen in:** `demo_queue`

## Load shedding and backpressure  `(concept)`
**1. Definition:** Deliberately rejecting excess requests (HTTP 429 or 503) when the system is overloaded, so that accepted requests still finish quickly.
**2. In plain words:** It is better to say "busy, try later" fast than to time out slowly for everyone. In the burst demo 23 requests were accepted and 17 rejected.
**3. Example to quote:**
> "A burst of 40 hit a queue of 5 with 2 workers: 23 accepted, 17 rejected with 429/503, and the 23 finished fine."

**Seen in:** `demo_queue`

## queue.Queue, put_nowait, queue.Full  `(standard library)`
**1. Definition:** `queue.Queue` is a thread-safe FIFO; `put_nowait(item)` adds without waiting and raises `queue.Full` if there is no room; `get()` blocks until an item is available.
**2. In plain words:** `put_nowait` with `except queue.Full` is the shedding mechanism: instead of blocking the caller, you count a rejection.
**3. Example to quote:**
> "`work.put_nowait(i)` raises `queue.Full` when 5 jobs are waiting, so I increment `rejected`."

**Seen in:** `demo_queue`

## Sentinel value  `(concept)`
**1. Definition:** A special value placed in a queue to signal a consumer to stop, here `None`.
**2. In plain words:** Workers loop forever on `work.get()`; sending one `None` per worker tells each to return, so threads can be joined cleanly.
**3. Example to quote:**
> "After the burst I `put(None)` once per worker, and each worker exits when it reads `None`."

**Seen in:** `demo_queue`

## threading.Thread, start, join  `(standard library)`
**1. Definition:** `Thread(target=fn)` creates a thread, `.start()` runs it, and `.join()` waits for it to finish.
**2. In plain words:** Workers and simulated clients run as threads. Joining ensures all work is done before results are printed.
**3. Example to quote:**
> "I start 2 worker threads, send the burst, then `join` them so every accepted job has finished."

**Seen in:** `demo_queue`, `load_test.py`

## Little's Law  `(concept)`
**1. Definition:** The average number of requests in the system equals the arrival rate multiplied by the average time each spends in the system (L = λ × W).
**2. In plain words:** To handle 200 requests per second at 0.2 s each, you need about 40 requests in flight, so at least 40 synchronous workers. A slower model multiplies the infrastructure needed, which is why inference optimisation is also a scalability topic.
**3. Example to quote:**
> "200 req/s at 200 ms means 40 concurrent requests, so with sync workers I need roughly 40 of them, or faster inference."

**Seen in:** `demo_capacity`, `WALKTHROUGH.md`

## Capacity planning  `(concept)`
**1. Definition:** Estimating the workers, replicas or hardware needed to meet a target load and latency.
**2. In plain words:** One sync worker does `1 / latency` requests per second (20 req/s at 50 ms). Multiply by workers for capacity, then verify with a load test.
**3. Example to quote:**
> "At 50 ms per request, one worker does 20 req/s and four workers do 80 req/s."

**Seen in:** `demo_capacity`

## Scale up vs scale out  `(concept)`
**1. Definition:** Scaling up means giving one machine more resources (more workers, more RAM); scaling out means running more machines or replicas behind a load balancer.
**2. In plain words:** Workers multiply memory use, so you eventually scale out with stateless replicas and autoscaling on CPU, queue depth or latency.
**3. Example to quote:**
> "I first raise workers per machine, but RAM equals workers times model size, so past a point I add replicas."

**Seen in:** `WALKTHROUGH.md` ladder

## Load balancer and autoscaling  `(concept)`
**1. Definition:** A load balancer distributes incoming requests across replicas; autoscaling adds or removes replicas based on metrics such as CPU, queue depth or latency.
**2. In plain words:** Together they let a stateless service grow with traffic. Statelessness matters: any replica must be able to answer any request.
**3. Example to quote:**
> "Replicas sit behind a load balancer, and autoscaling adds more when p95 latency crosses my threshold."

**Seen in:** `WALKTHROUGH.md`

## Load test  `(concept)`
**1. Definition:** A test that simulates many concurrent clients against a service for a period and measures throughput, latency percentiles and errors.
**2. In plain words:** Real tools are locust, k6, wrk and ab; writing a tiny one shows what they measure. With 16 clients and one worker, each request waited behind about 15 others, so p50 was about 834 ms even though the work took 50 ms.
**3. Example to quote:**
> "With 1, 2 and 4 workers throughput went 24, 41 and 79 req/s and p50 latency fell from 834 ms to 211 ms."

**Seen in:** `load_test.py`

## Queueing delay  `(concept)`
**1. Definition:** Time a request spends waiting for a free worker before processing starts.
**2. In plain words:** Latency is the work time plus the queueing time. Latency rises long before you run out of throughput, because requests pile up once all workers are busy.
**3. Example to quote:**
> "The 50 ms request took 834 ms with one worker, because it waited behind other requests."

**Seen in:** `WALKTHROUGH.md`

## Throughput  `(concept)`
**1. Definition:** The number of requests completed per second.
**2. In plain words:** It scales roughly with workers until saturation. The tester counts successes divided by the test duration.
**3. Example to quote:**
> "Four workers reached about 79 requests per second in the load test."

**Seen in:** `load_test.py`

## statistics.quantiles  `(standard library)`
**1. Definition:** `statistics.quantiles(data, n=100)` returns 99 cut points dividing the data into 100 groups, so index 49 is the median (p50) and 94 is p95.
**2. In plain words:** It computes percentiles without numpy. Remember the off-by-one: percentile p is at index p minus 1.
**3. Example to quote:**
> "`q = statistics.quantiles(latencies, n=100)`; then `q[49]` is p50 and `q[94]` is p95."

**Seen in:** `load_test.py`

## Simulated client loop  `(concept)`
**1. Definition:** Each virtual user is a thread that sends requests back to back until a deadline, recording the latency of successes and of failures separately.
**2. In plain words:** This models many users hitting the service at once. `list.append` is thread-safe in CPython, so the shared lists need no lock here.
**3. Example to quote:**
> "16 threads each loop `client.get('/slow', params={'seconds': 0.05})` for 3 seconds and record how long each took."

**Seen in:** `client_loop`

## httpx.Client and httpx.HTTPError  `(third-party: httpx)`
**1. Definition:** `httpx.Client(base_url=..., timeout=...)` is a connection-reusing HTTP client; `httpx.HTTPError` is the base class of its request errors.
**2. In plain words:** Reusing one client per virtual user is efficient and realistic. Catching `httpx.HTTPError` counts failures instead of crashing the test.
**3. Example to quote:**
> "If a request raises `httpx.HTTPError` I count it as an error and keep going."

**Seen in:** `client_loop`

## Starting and stopping a server under test  `(standard library)`
**1. Definition:** `subprocess.Popen([...gunicorn...])` launches the server; `try/finally` with `terminate()` and `wait()` guarantees it is stopped.
**2. In plain words:** The test runs the module-09 Flask API under Gunicorn with 1, 2 and 4 workers, polls `/health` until ready, and always cleans up.
**3. Example to quote:**
> "`run(workers)` starts Gunicorn with N workers, runs 16 clients for 3 seconds, prints results, and terminates the server in `finally`."

**Seen in:** `load_test.py`

## Reliability vocabulary (circuit breaker, graceful degradation, SLO)  `(concept)`
**1. Definition:** A circuit breaker stops calling a failing dependency for a while; graceful degradation returns something useful (cached or fallback) instead of an error; an SLO is a target such as "p95 under 300 ms and 99.9% success".
**2. In plain words:** These patterns prevent one failing part from cascading. Retries can amplify an outage, so combine them with caps, backoff and breakers.
**3. Example to quote:**
> "If the model service is down I serve a cached or fallback answer, and my SLO is p95 under 300 ms with 99.9% success."

**Seen in:** `WALKTHROUGH.md`

## Health vs readiness checks  `(concept)`
**1. Definition:** A health (liveness) check says the process is alive; a readiness check says it has loaded the model and is ready for traffic.
**2. In plain words:** Orchestrators such as Docker and Kubernetes use them to restart dead containers and to avoid sending requests to one that is still loading. Module 11 uses health checks in Compose.
**3. Example to quote:**
> "A replica is 'alive' as soon as the process starts but only 'ready' after the model has loaded."

**Seen in:** `WALKTHROUGH.md`

## When latency rises: what to check  `(concept)`
**1. Definition:** A diagnosis checklist: profile where time goes (model, preprocessing, network, database), check cold starts and warm-up, noisy neighbours, payload size, and look at p99 not the average.
**2. In plain words:** "It got slower" is vague. Splitting the request path into stages tells you which stage to fix.
**3. Example to quote:**
> "If p95 doubled I'd compare model time with network time, check for cold starts, and look at payload sizes."

**Seen in:** `WALKTHROUGH.md`

## Multiple assignment and defaults in demos  `(syntax)`
**1. Definition:** `accepted = rejected = 0` assigns the same value to two names; function parameters like `clients: int = 16` provide defaults.
**2. In plain words:** Small conveniences used throughout the demos to keep counters and test settings compact.
**3. Example to quote:**
> "`accepted = rejected = 0` starts both counters at zero before the burst loop."

**Seen in:** `demo_queue`, `run`

## Re-raise and exception types in demos  `(syntax)`
**1. Definition:** `ConnectionError` is a built-in exception for network failures and `raise ConnectionError(msg)` creates one deliberately.
**2. In plain words:** The flaky service raises it on the first two calls to simulate transient failure; the retry loop catches it.
**3. Example to quote:**
> "`flaky_service` raises `ConnectionError('attempt 1: connection reset')` twice, then returns 'ok'."

**Seen in:** `flaky_service`
