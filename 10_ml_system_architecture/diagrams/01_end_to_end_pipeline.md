# Diagram 1 — The path from user input to model output (and back into training)

```
 TRAINING SIDE (offline, scheduled)                       SERVING SIDE (online, per request)
 ──────────────────────────────────                       ───────────────────────────────────
  raw data sources                                          user / client app
  (files, DB, APIs, user feedback)                           │  POST /predict {"text": "..."}
        │ collect                                            ▼
        ▼                                                  ┌──────────────┐
  ┌────────────┐                                           │ load balancer│  (spreads traffic, health checks)
  │ clean/     │  preprocess ◄──── SAME CODE ────►         └──────┬───────┘
  │ validate   │                                                  ▼
  └─────┬──────┘                                           ┌──────────────┐  Gunicorn/Uvicorn workers
        ▼                                                  │  API (09)    │  each holds the model in RAM
  ┌────────────┐                                           │ 1 validate   │  <- Pydantic / manual checks
  │ features   │  TF-IDF / embeddings  ◄── SAME CODE ──►   │ 2 preprocess │
  └─────┬──────┘                                           │ 3 features   │
        ▼                                                  │ 4 model      │  predict_proba
  ┌────────────┐                                           │ 5 postprocess│  threshold, label mapping
  │ train      │                                           │ 6 respond    │  JSON + model_version
  └─────┬──────┘                                           └──┬────────┬──┘
        ▼                                                     │        │ log
  ┌────────────┐                                              │        ▼
  │ evaluate   │ metrics must beat production                 │   ┌──────────────┐  prediction log (05)
  └─────┬──────┘                                              │   │ DB / logs    │  metrics, drift (07)
        ▼ save artifact                                       │   └──────┬───────┘
  ┌────────────────┐   load at startup   ┌──────────────────┐ │          │ monitoring/alerts
  │ model registry │────────────────────►│ model in memory  │◄┘          │
  │ v0001, v0002...│   (alias: production)└──────────────────┘            │
  └────────────────┘                                                      ▼
        ▲                                                          drift or low accuracy?
        └──────────────── new labelled data from logs/feedback ◄──── yes → retrain (loop closes)
```
Where each box lives in this repo: collect→train→evaluate [`01`/`02`](../01_pipeline_end_to_end/); artifact → registry [`04`](../04_model_registry_versioning/); API → [module 09](../../09_api_development/); log → [`05`](../05_storage_and_prediction_log/); monitoring → [`07`](../07_logging_and_monitoring/); containers → [module 11](../../11_docker/).
