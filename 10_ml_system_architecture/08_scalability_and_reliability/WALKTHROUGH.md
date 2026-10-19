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
