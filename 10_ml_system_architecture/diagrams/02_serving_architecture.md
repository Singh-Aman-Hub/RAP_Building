# Diagram 2 — Model-serving architecture options

## A. Model embedded in the API process (what we built; simplest)
```
client ──HTTP──► [ API worker 1: app + model ]
                 [ API worker 2: app + model ]     RAM = workers × model size
```
Pros: simple, no network hop. Cons: scale API and model together; one bad model release = redeploy the API.

## B. Separate model server behind the API
```
client ──► API (validation, auth, business logic) ──gRPC/HTTP──► model server (Triton, vLLM, TorchServe)
                                                                  batching, GPU, many models
```
Pros: scale and upgrade independently, GPU only where needed, request batching. Cons: extra hop (latency), more to operate.

## C. Async job via queue (slow models, nobody waiting)
```
client ──POST /jobs──► API ──► queue (Redis/SQS) ──► worker pool ──► result store
   ▲                     │ 202 Accepted + job_id                          │
   └──── GET /jobs/{id} (poll) or webhook ◄─────────────────────────────────┘
```
Pros: absorbs bursts, retries are easy. Cons: client must poll; more moving parts.

## D. Batch scoring
```
scheduler (cron/Airflow) ──► job reads table/files in chunks ──► model.predict(chunk) ──► writes results table
```

## Choosing
| Question | Leans toward |
|---|---|
| Sub-second answer needed? | A or B |
| Large GPU model shared by several apps? | B |
| Inference takes minutes? | C |
| Nobody waiting, millions of rows? | D |
| Small team, small traffic? | A (don't over-engineer) |
