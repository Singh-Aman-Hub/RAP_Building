# Template — answering "Design an ML service for X" in an interview (≈ 5–8 minutes)

Use this order. Say the headings out loud; it shows structure.

1. **Clarify requirements (30 s).** Input/output? Who calls it? Latency target (p95)? Traffic (req/s now and later)? Accuracy vs cost priority? Online or batch? Data sensitivity?
2. **Data & training pipeline (1 min).** Data sources → cleaning → features → train → evaluate on held-out data → artifact + metadata in a registry. Mention reproducibility (seeds, data fingerprint, library versions).
3. **Inference path (1 min).** Request → validate → preprocess (same code as training) → model → postprocess → response with `model_version`. Load model once per worker at startup.
4. **Serving architecture (1 min).** Pick A/B/C/D from [diagram 2](02_serving_architecture.md) and justify it with the requirements. Name the framework + server (FastAPI + Uvicorn / Flask + Gunicorn), container (Docker), and where config comes from (env vars).
5. **Storage (30 s).** Artifacts in object storage + registry; predictions/feedback in a DB; features in a cache if online lookup is needed.
6. **Scale & reliability (1 min).** Workers/replicas + autoscaling; cache, batching; timeouts, bounded queues, retries with backoff; health/readiness checks; capacity estimate via Little's Law.
7. **Monitoring & retraining (1 min).** Structured logs, latency percentiles, error rate, input/prediction drift, delayed true-label metrics; alert thresholds; retrain trigger; canary or shadow deployment; instant rollback via alias.
8. **Trade-offs & next steps (30 s).** Name at least two: accuracy vs latency, cost vs freshness, simplicity vs flexibility; what you'd build second.

## Mini worked example: "sentiment API for 100 req/s, p95 < 200 ms"
- Model: TF-IDF + logistic regression (~0.3 ms) → one machine easily handles 100 req/s; HTTP overhead dominates. Start with FastAPI + 2 Uvicorn workers in one container; add a second replica for availability (not only capacity).
- Logging every prediction (async), p95 alert at 150 ms, drift check on text length daily.
- If the model were a 7B LLM (2 s/request): 100 req/s ⇒ ~200 in flight ⇒ separate GPU model server with continuous batching (vLLM), queue + 429 when full, streaming responses.
- Trade-off: simple model is cheaper and faster, but may be less accurate on sarcasm — decide with an evaluation set, not opinion.

## Phrases that score well
"I'd measure first." · "Let me state the assumptions." · "The failure mode is… so I'd add…" · "That trades X for Y." · "I'd start simple and add complexity only when the metrics demand it."
