# Terms — Architecture diagrams and the design-interview template

Folder: `10_ml_system_architecture/diagrams` · Files covered: `01_end_to_end_pipeline.md`, `02_serving_architecture.md`, `03_batch_vs_realtime.md`, `04_scaling_when_traffic_grows.md`, `05_design_interview_answer_template.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Load balancer](#load-balancer-concept) · [Training side vs serving side](#training-side-vs-serving-side-concept) · [Retraining loop](#retraining-loop-concept) · [Model registry and alias](#model-registry-and-alias-concept) · [Embedded model (option A)](#embedded-model-option-a-concept) · [Separate model server (option B)](#separate-model-server-option-b-concept) · [Async job via queue (option C)](#async-job-via-queue-option-c-concept) · [202 Accepted](#202-accepted-concept) · [Batch scoring (option D)](#batch-scoring-option-d-concept) · [Real-time, batch and micro-batch](#real-time-batch-and-micro-batch-concept) · [Continuous batching](#continuous-batching-concept) · [Scaling stages](#scaling-stages-concept) · [Symptom-to-fix mapping](#symptom-to-fix-mapping-concept) · [Rate limiting](#rate-limiting-concept) · [Redis cache](#redis-cache-concept) · [Little's Law](#littles-law-concept) · [Latency budget](#latency-budget-concept) · [p95 latency](#p95-latency-concept) · [Async logging](#async-logging-concept) · [Design-interview answer template](#design-interview-answer-template-concept) · [Clarify requirements](#clarify-requirements-concept) · [Reproducibility in the pipeline](#reproducibility-in-the-pipeline-concept) · [Canary and shadow deployment](#canary-and-shadow-deployment-concept) · [Delayed true-label metrics](#delayed-true-label-metrics-concept) · [Trade-off statements](#trade-off-statements-concept) · [Streaming responses](#streaming-responses-concept) · [Start simple](#start-simple-concept)

---

## Load balancer  `(concept)`
**1. Definition:** A component that distributes incoming requests across several server replicas and usually performs health checks to avoid sending traffic to unhealthy ones.
**2. In plain words:** It is the front door of a scaled service. Clients talk to one address; the balancer picks a replica for each request, so you can add replicas without clients noticing.
**3. Example to quote:**
> "In my diagram the load balancer spreads requests across replicas A, B and C and stops sending to one that fails its health check."

**Seen in:** `01_end_to_end_pipeline.md`, `04_scaling_when_traffic_grows.md`

## Training side vs serving side  `(concept)`
**1. Definition:** The training side is the offline, scheduled path (collect, clean, features, train, evaluate, save artifact); the serving side is the online, per-request path (validate, preprocess, model, postprocess, respond).
**2. In plain words:** They are different programs joined by an artifact in a registry. The key rule marked "SAME CODE" in the diagram is that preprocessing and features must be identical on both sides.
**3. Example to quote:**
> "Training runs nightly on a big machine and publishes a model; serving loads it at startup and answers each request."

**Seen in:** `01_end_to_end_pipeline.md`

## Retraining loop  `(concept)`
**1. Definition:** The feedback path in which monitoring detects drift or low accuracy, new labelled data from logs and feedback is collected, and the model is retrained and redeployed.
**2. In plain words:** A deployed model decays as the world changes, so the pipeline is a loop, not a line. The loop closes through the prediction log, evaluation against production and the registry.
**3. Example to quote:**
> "My drift check fired, so I labelled recent low-confidence rows, retrained, and promoted the new version only because it beat production."

**Seen in:** `01_end_to_end_pipeline.md`

## Model registry and alias  `(concept)`
**1. Definition:** A versioned store of model artifacts with movable labels such as `production`, from which serving loads at startup.
**2. In plain words:** It is the handoff point between training and serving, and the place to roll back by moving a label.
**3. Example to quote:**
> "The API loads whatever the `production` alias points to; promoting v0002 is a label change, not a redeploy."

**Seen in:** `01_end_to_end_pipeline.md`

## Embedded model (option A)  `(concept)`
**1. Definition:** Serving architecture where the model lives inside each API worker process, so RAM use equals workers times model size.
**2. In plain words:** It is the simplest option with no extra network hop, but the API and model scale together and a bad model release means redeploying the API.
**3. Example to quote:**
> "For a small team and small traffic I'd start with the model embedded in FastAPI; it's simple and fast."

**Seen in:** `02_serving_architecture.md`

## Separate model server (option B)  `(concept)`
**1. Definition:** Serving architecture where the API calls a dedicated model server (Triton, vLLM, TorchServe) over gRPC or HTTP.
**2. In plain words:** It lets you scale and upgrade the model independently, put GPUs only where needed, and batch requests, at the cost of an extra network hop and more to operate.
**3. Example to quote:**
> "A large GPU model shared by several apps belongs in its own server with continuous batching."

**Seen in:** `02_serving_architecture.md`

## Async job via queue (option C)  `(concept)`
**1. Definition:** Architecture where the API accepts a job, returns `202 Accepted` with a job id, a queue feeds a worker pool, and the client polls or receives a webhook for the result.
**2. In plain words:** It suits inference that takes minutes and bursts of traffic, because the queue absorbs load and retries are easy. The client must poll and there are more moving parts.
**3. Example to quote:**
> "For a 3-minute video model I return a job id immediately, and the client polls `GET /jobs/{id}`."

**Seen in:** `02_serving_architecture.md`

## 202 Accepted  `(concept)`
**1. Definition:** An HTTP status meaning the request was accepted for processing but is not complete yet.
**2. In plain words:** It is the correct reply to "I've queued your job", usually with a job id and a way to check status.
**3. Example to quote:**
> "`POST /jobs` returns 202 with `{'job_id': 'abc123'}` while the worker pool processes it."

**Seen in:** `02_serving_architecture.md`

## Batch scoring (option D)  `(concept)`
**1. Definition:** A scheduled job reads a table or files in chunks, calls `model.predict(chunk)` and writes a results table.
**2. In plain words:** Use it when nobody is waiting and there are millions of rows; a scheduler such as cron or Airflow triggers it.
**3. Example to quote:**
> "Nightly recommendations are batch scoring: read users in chunks, predict, write the results table."

**Seen in:** `02_serving_architecture.md`, `03_batch_vs_realtime.md`

## Real-time, batch and micro-batch  `(concept)`
**1. Definition:** Real-time predicts one item per request (latency matters), batch scores many items in chunks (throughput matters), and micro-batch collects requests for 10 to 50 ms and predicts them together.
**2. In plain words:** Bigger batches give better throughput but worse latency for the first item in the batch. LLM servers use continuous batching, adding and removing sequences at every token step.
**3. Example to quote:**
> "In step 06, 2000 items one-by-one took 497 ms and batched took 5.5 ms, a 91 times difference."

**Seen in:** `03_batch_vs_realtime.md`

## Continuous batching  `(concept)`
**1. Definition:** An LLM-serving technique (used in vLLM) that adds and removes sequences from the running batch at every token step rather than waiting for a whole batch to finish.
**2. In plain words:** Short requests no longer wait for long ones, so GPU utilisation and throughput improve.
**3. Example to quote:**
> "vLLM's continuous batching lets a new request join the batch on the next token step instead of waiting for others to finish."

**Seen in:** `03_batch_vs_realtime.md`

## Scaling stages  `(concept)`
**1. Definition:** The usual progression: laptop, one server with several workers, scale out with a load balancer and autoscaled replicas, then add protection and decoupling (rate limits, cache, queue, timeouts, load shedding).
**2. In plain words:** Each stage is triggered by symptoms. Move to the next only when metrics demand it, because every stage adds complexity and cost.
**3. Example to quote:**
> "RAM full means fewer workers per box or a smaller model; CPU at 100% with rising p95 means more workers or replicas."

**Seen in:** `04_scaling_when_traffic_grows.md`

## Symptom-to-fix mapping  `(concept)`
**1. Definition:** A table linking observed symptoms to likely remedies: p95 up with high CPU, p95 up with low CPU, RAM full, error spikes during bursts, a failing dependency.
**2. In plain words:** High CPU suggests compute limits; low CPU with high latency suggests waiting on a database or downstream call. Diagnose first; do not guess.
**3. Example to quote:**
> "p95 is up but CPU is low, so I suspect something is waiting: a database, a downstream API, a lock or a cold start."

**Seen in:** `04_scaling_when_traffic_grows.md`

## Rate limiting  `(concept)`
**1. Definition:** Restricting how many requests a client may make in a time window, rejecting extra requests (often with HTTP 429).
**2. In plain words:** It protects a service from being overwhelmed by one client or a runaway script.
**3. Example to quote:**
> "I limit each API key to 10 requests per second and return 429 beyond that."

**Seen in:** `04_scaling_when_traffic_grows.md`

## Redis cache  `(concept)`
**1. Definition:** Redis is an in-memory key-value store commonly used as a shared cache or queue.
**2. In plain words:** A shared cache lets all replicas reuse answers for repeated inputs, unlike an in-process cache that each replica has to fill separately.
**3. Example to quote:**
> "Replicas check Redis first; a hit returns instantly without running the model."

**Seen in:** `04_scaling_when_traffic_grows.md`

## Little's Law  `(concept)`
**1. Definition:** In-flight requests equal arrival rate multiplied by latency.
**2. In plain words:** It converts a throughput target and a per-request time into the concurrency you must support, and so into workers or replicas.
**3. Example to quote:**
> "100 req/s at 2 s per request means about 200 in flight, so a 7B LLM needs a GPU server with batching, not a few workers."

**Seen in:** `04_scaling_when_traffic_grows.md`, `05_design_interview_answer_template.md`

## Latency budget  `(concept)`
**1. Definition:** A breakdown of the target latency into the time allowed for each stage of a request (network, validation, preprocessing, model, postprocessing, logging).
**2. In plain words:** It shows where the time goes. In the example the model uses 150 of 300 ms, so optimise there first, after profiling rather than guessing.
**3. Example to quote:**
> "With a p95 target of 300 ms, the model takes 60% of the budget, so I'd profile and optimise it before anything else."

**Seen in:** `04_scaling_when_traffic_grows.md`

## p95 latency  `(concept)`
**1. Definition:** The latency below which 95% of requests complete.
**2. In plain words:** It reflects the experience of slower requests, so targets such as "p95 under 200 ms" are more meaningful than averages.
**3. Example to quote:**
> "My service-level target is p95 under 200 ms."

**Seen in:** `03_batch_vs_realtime.md`, `05_design_interview_answer_template.md`

## Async logging  `(concept)`
**1. Definition:** Writing logs without blocking the request path, for example through a background thread or queue.
**2. In plain words:** Logging should add about a millisecond, not delay the response. Slow log writes must not make users wait.
**3. Example to quote:**
> "Logging is 1 ms of my 300 ms budget because it happens asynchronously."

**Seen in:** `04_scaling_when_traffic_grows.md`, `05_design_interview_answer_template.md`

## Design-interview answer template  `(concept)`
**1. Definition:** A fixed order for answering "design an ML service": clarify requirements, data and training pipeline, inference path, serving architecture, storage, scale and reliability, monitoring and retraining, trade-offs and next steps.
**2. In plain words:** Saying the headings out loud shows structure. It keeps you from jumping into details and makes sure you cover each area in 5 to 8 minutes.
**3. Example to quote:**
> "First I'll clarify requirements, then walk the training pipeline, the inference path, serving, storage, scaling, monitoring, and finish with trade-offs."

**Seen in:** `05_design_interview_answer_template.md`

## Clarify requirements  `(concept)`
**1. Definition:** Asking about inputs and outputs, callers, latency target, traffic, accuracy versus cost, online versus batch and data sensitivity before designing.
**2. In plain words:** The right design depends on the answers: a 50 ms target and 1000 req/s leads to different choices than a nightly report.
**3. Example to quote:**
> "Before designing, I'd like to know the p95 latency target and expected requests per second."

**Seen in:** `05_design_interview_answer_template.md`

## Reproducibility in the pipeline  `(concept)`
**1. Definition:** Recording seeds, a data fingerprint and library versions so a model can be recreated.
**2. In plain words:** Without them you cannot explain or reproduce why a model behaves as it does.
**3. Example to quote:**
> "Each registry version stores the data fingerprint and library versions, so I can retrain the same model."

**Seen in:** `05_design_interview_answer_template.md`

## Canary and shadow deployment  `(concept)`
**1. Definition:** A canary sends a small share of real traffic to a new model; a shadow deployment runs it on copies of traffic without returning its answers to users.
**2. In plain words:** Both reduce risk by testing a new model on real data before full rollout, and instant rollback via an alias remains available.
**3. Example to quote:**
> "I'd send 5% of traffic to v0002 as a canary, compare metrics with v0001, then promote or roll back."

**Seen in:** `05_design_interview_answer_template.md`

## Delayed true-label metrics  `(concept)`
**1. Definition:** Accuracy measured after the correct answers become known, perhaps days later, in contrast to immediate proxies such as drift and confidence.
**2. In plain words:** For many applications labels arrive late, so you monitor proxies in the meantime and compute real accuracy when ground truth arrives.
**3. Example to quote:**
> "Fraud labels arrive after weeks, so I watch drift and confidence now and compute true accuracy later."

**Seen in:** `05_design_interview_answer_template.md`

## Trade-off statements  `(concept)`
**1. Definition:** Explicit "X versus Y" decisions, such as accuracy vs latency, cost vs freshness, simplicity vs flexibility.
**2. In plain words:** Naming at least two trade-offs shows you understand there is no free option. Decide with an evaluation set, not opinion.
**3. Example to quote:**
> "A TF-IDF model is cheaper and faster but may be weaker on sarcasm; I'd decide using an evaluation set."

**Seen in:** `05_design_interview_answer_template.md`

## Streaming responses  `(concept)`
**1. Definition:** Sending output to the client incrementally as it is produced instead of waiting for the full result.
**2. In plain words:** For LLMs generating tokens over seconds, streaming makes the first text appear immediately, improving perceived latency.
**3. Example to quote:**
> "For a 7B LLM I'd stream tokens so the user sees text after the first token, not after two seconds."

**Seen in:** `05_design_interview_answer_template.md`

## Start simple  `(concept)`
**1. Definition:** The principle of adding architectural complexity only when metrics demand it.
**2. In plain words:** A TF-IDF model on 100 req/s fits on one machine; the HTTP overhead dominates. Do not build a GPU cluster for a problem a small model solves.
**3. Example to quote:**
> "I'd start with FastAPI and two Uvicorn workers in one container and add complexity only when the metrics demand it."

**Seen in:** `05_design_interview_answer_template.md`
