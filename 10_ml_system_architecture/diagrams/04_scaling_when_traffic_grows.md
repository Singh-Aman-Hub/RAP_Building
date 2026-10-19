# Diagram 4 — What changes when requests increase or latency rises

```
Stage 0: laptop              Stage 1: one server          Stage 2: scale out              Stage 3: protect + decouple
 client ─► 1 process          client ─► Gunicorn            clients ─► load balancer         clients ─► LB ─► API replicas
                                       4 workers                      ├─► replica A (4 w)             │  rate limit, auth
                                       (RAM = 4 × model)              ├─► replica B (4 w)             ├─► cache (Redis) ◄─ hit
                                                                      └─► replica C (autoscaled)      ├─► queue ─► GPU workers
                                                                                                      └─► timeouts, 429/503
Symptoms that trigger the next stage:
  p95 latency ↑ with CPU ~100%   → more workers/replicas, faster model (quantize/distill), batching
  p95 ↑ but CPU low              → waiting on something: DB, downstream API, lock, cold start; use async/caching
  RAM full                       → fewer workers per box, share weights (threads, --preload), smaller model
  error spikes during bursts     → bounded queue + load shedding + retries with backoff
  one dependency failing         → timeout + circuit breaker + fallback answer
```
Capacity rule (Little's Law): in-flight requests = arrival rate × latency. See [step 08](../08_scalability_and_reliability/).

## Latency budget example (target p95 = 300 ms)
```
network in 20 ms | validation 2 ms | preprocessing 5 ms | model 150 ms | postprocess 3 ms | logging (async) 1 ms | network out 20 ms
                                  └─ model is 60% of the budget → optimise there first (profile before guessing)
```
