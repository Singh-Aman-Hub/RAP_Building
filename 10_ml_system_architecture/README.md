# Module 10 — Basic ML System Architecture

*(Syllabus topic 5.)*

## 1. Why this matters
Interviewers for ML Engineer roles rarely stop at "I trained a model". They ask: *how does user input reach the model and come back? Where is the model stored? How do you know it still works? What if traffic doubles?* This module turns those questions into small runnable code and diagrams, and ends with a template for answering "design an ML service" questions.

## 2. Where this fits
**Builds on**
- [03_ml_foundations](../03_ml_foundations/) — train/validation/test split, metrics, the sklearn text classifier and its saved artifact. Module 10 repackages the same stages as *system components*.
- [09_api_development](../09_api_development/) — the API is the "serving" box; `MODEL_PATH` loads artifacts produced here (verified with this module's `02_.../artifacts/model.joblib` and module 03's artifact); `/slow` + Gunicorn power the load test; workers/memory ideas feed scalability.
- [06_llm_training_and_fast_inference](../06_llm_training_and_fast_inference/) — latency/throughput/GPU-memory vocabulary; the same batching and capacity ideas apply to LLM serving.
- [01_debugging_and_environment](../01_debugging_and_environment/) — version mismatches, tracebacks, environment issues reappear as serialization and skew bugs.

**Leads to**
- [11_docker](../11_docker/) — the unit you deploy: image = code + dependencies + (a pinned model version); logs go to stdout; env vars configure it; health checks = readiness.
- [12_streamlit_gradio](../12_streamlit_gradio/) — the client in the diagram.
- [13_capstone_project](../13_capstone_project/) — build the whole diagram: train → registry → API → UI → container → logs.

## 3. Concept map
```
 collect ─► preprocess ─► features ─► train ─► evaluate ─► SERIALIZE ─► REGISTRY (versions, aliases)
 (01)        (01)          (01)       (01)     (01/02)      (03)           (04)
    └──────────── TRAINING PIPELINE (02, offline) ───────────────┘            │ load by alias/version
                                                                              ▼
 user ─► API (module 09) ─► validate ─► [preprocess+features+model] ─► postprocess ─► response
            │                (same code as training: avoid skew, 02)          │
            ▼ log every prediction (05)                                       ▼
        DB / logs ─► metrics + drift (07) ─► alert ─► retrain ─► new version (04)
            │
        batch (06) vs real-time (06)      scale: cache · workers · replicas · queue · timeouts (08)
```

## 4. Study path
| # | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 01 | [01_pipeline_end_to_end](01_pipeline_end_to_end/) | every pipeline stage in code; fit vs transform; leakage | `python main.py` | 90 min |
| 02 | [02_training_vs_inference_pipelines](02_training_vs_inference_pipelines/) | two pipelines, one artifact; training/serving skew | `python main.py` | 90 min |
| 03 | [03_serialization](03_serialization/) | pickle/joblib/safetensors; pickle security; custom-code gotcha | `python main.py` | 60 min |
| 04 | [04_model_registry_versioning](04_model_registry_versioning/) | versions, metadata, aliases, rollback | `python main.py` | 75 min |
| 05 | [05_storage_and_prediction_log](05_storage_and_prediction_log/) | SQLite prediction log, SQL, injection, storage map | `python main.py` | 75 min |
| 06 | [06_batch_vs_realtime](06_batch_vs_realtime/) | batch vs real-time, generators, throughput vs latency | `python main.py` | 60 min |
| 07 | [07_logging_and_monitoring](07_logging_and_monitoring/) | JSON logs, percentiles, PSI drift | `python main.py` | 90 min |
| 08 | [08_scalability_and_reliability](08_scalability_and_reliability/) | cache, retries, queues, Little's Law, load test | `python main.py`, `python load_test.py` | 90 min |
| — | [diagrams/](diagrams/) | 5 diagrams + design answer template | read | 45 min |

Total ≈ **10–11 hours** + [exercises](exercises.md). Step 01 first; steps 02–04 share the same data/pipeline; 05–08 reuse step 02's artifact.

## 5. The syllabus topics in plain English

**End-to-end ML pipeline.** Data → clean → features → train → evaluate → save → serve → monitor → retrain. A pipeline is a repeatable program, not a notebook you re-run by hand. *Limitation:* more structure than a one-off analysis needs.

**Data collection and preprocessing.** Models are only as good as the data's relevance and cleanliness; apply identical preprocessing at serving time. *Trade-off:* aggressive cleaning can remove signal (e.g. emoji, casing).

**Feature engineering.** Turn raw input into numbers: TF-IDF for text, embeddings (module 04), scaling for numbers. Fit feature extractors on training data only (leakage otherwise).

**Model training and evaluation.** Fit on training data, evaluate on held-out data with a metric that matches the business need (accuracy can mislead on imbalanced classes — module 03). A tiny test set gives noisy estimates.

**Model serialization and loading.** Save the fitted object (joblib/pickle for sklearn, safetensors for weights). Pickle executes code on load → trust only your own files; match library versions.

**Training vs inference pipelines.** Offline/heavy/occasional vs online/light/constant; connected only by the artifact and shared preprocessing. Skew = silent accuracy loss.

**API and model-serving architecture.** Embedded model in an API process (simple), separate model server (GPU/batching), async job queue (slow models), batch job (no one waiting). Choose from requirements, not fashion.

**Database and model storage.** Artifacts → object storage + registry; predictions/feedback → relational DB; online features → key-value cache; metrics → time-series DB.

**Batch vs real-time inference.** Throughput vs latency; micro-batching blends them. Batching gave ~90× speed-up in our demo.

**Logging and monitoring.** Structured logs with request ids; latency percentiles; error rate; input/prediction drift; delayed accuracy. Log lengths, not private text.

**Model versioning.** Immutable numbered artifacts + metadata + aliases → reproducibility, A/B tests, rollback.

**Scalability, latency and reliability.** Measure; cache; batch; scale workers/replicas; bound queues; shed load; time out; retry with backoff; health checks; capacity via Little's Law.

## 6. Common mistakes and gotchas
- Fitting the vectorizer/scaler on all data (leakage) or at inference (shape crash).
- Different preprocessing in training and serving (silent skew).
- Unpickling untrusted files; loading with a different scikit-learn version.
- Pickling a function defined in `__main__` then loading from another script.
- Using `latest` in production; overwriting model files in place (no rollback).
- Reporting only average latency; ignoring p95/p99.
- Unbounded queues/retries without backoff → outages that amplify themselves.
- Logging raw user text (privacy) or secrets.
- Over-engineering: Kubernetes + feature store for a model that runs in 0.3 ms.

## 7. Self-check
- [ ] I can draw the path from user input to model output and back into retraining, naming every component.
- [ ] I can explain why preprocessing must be identical in training and serving and show the failure.
- [ ] I can save/reload a model and name two risks of pickle.
- [ ] I can explain what a registry stores and how rollback works.
- [ ] I can log predictions to a database with parameterized SQL.
- [ ] I can choose batch vs real-time and justify it.
- [ ] I can compute p50/p95 and explain a drift score.
- [ ] I can answer "traffic doubled / latency rose — what do you do?" with an ordered list.
- [ ] I can use the design template to answer an open design question in ≤ 8 minutes.

## 8. Interview questions
1. **Walk me through an ML system end to end.** Data collection → preprocessing → features → train/evaluate → artifact in a registry → API loads it → validate/preprocess/predict/postprocess → log → monitor → retrain. Mention that preprocessing code is shared. *Trade-off:* each box adds operational cost; start with the minimum.
2. **Training vs inference pipeline?** Offline, heavy, scheduled vs online, light, per request. They share an artifact and preprocessing.
3. **What is training/serving skew and how do you prevent it?** Different data processing between training and serving → silent degradation. Prevent: one shared pipeline object/code, version pinning, tests on the loaded artifact, monitoring of input distributions.
4. **Why is `pickle.load` dangerous?** It can execute arbitrary code from the file. Only load trusted files; prefer safetensors/ONNX for sharing.
5. **How do you version models?** Immutable `vN` artifacts with metadata (metrics, params, data fingerprint, library versions); alias `production`; pin versions in deploys; log `model_version` with every prediction.
6. **Batch or real-time?** Real-time if a user waits; batch if not (cheaper, retry-able); micro-batch for GPU efficiency. Latency vs throughput.
7. **What do you monitor in production?** Latency percentiles, error rate, throughput, resource use, input/prediction drift, and true performance once labels arrive.
8. **What's drift and what do you do about it?** Data distribution change (data drift) or changed input→label relation (concept drift). Detect with stats like PSI; investigate, relabel, retrain, canary, promote, rollback if needed.
9. **Traffic doubles — what do you do?** Measure; cache/batch/optimise model; add workers/replicas + autoscaling; bounded queue + load shedding + timeouts; capacity via Little's Law.
10. **Why do average latencies mislead?** Tail latency (p95/p99) is what users feel and what causes timeouts; averages hide it.
11. **Where does the model live and how is it loaded?** Artifact in object storage/registry; loaded once per worker at startup; memory scales with workers.
12. **Where would you store predictions and why?** Relational DB with model version, input reference, output, confidence, latency — debugging, monitoring, future training data. Mind privacy.
13. **Design a sentiment API for 100 req/s.** Use [the template](diagrams/05_design_interview_answer_template.md): clarify → pipeline → inference path → serving option → storage → scale/reliability → monitoring → trade-offs.
