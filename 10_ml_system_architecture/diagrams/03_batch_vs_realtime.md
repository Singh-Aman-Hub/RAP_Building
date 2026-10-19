# Diagram 3 — Batch vs real-time vs micro-batch

```
REAL-TIME           request ─► [predict 1 item] ─► response          latency matters (p95)
                    request ─► [predict 1 item] ─► response

BATCH               ┌ table of 10M rows ┐
                    │ chunk 1 (500)     ├─► predict(500) ─┐
                    │ chunk 2 (500)     ├─► predict(500) ─┼─► results table     throughput matters
                    └ ...               ┘                 ┘

MICRO-BATCH         requests arrive: r1 r2 r3 r4 ... within a 10-50 ms window
(LLM servers)       └────── collect ──────┘ ─► predict([r1..r4]) ─► respond to each
                    (continuous batching in vLLM adds/removes sequences every token step)
```
Measured in [step 06](../06_batch_vs_realtime/): 2000 items one-by-one 497 ms vs batched 5.5 ms (91×).
Trade-off: bigger batch ⇒ better throughput, worse latency for the first item in the batch.
