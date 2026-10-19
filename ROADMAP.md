# ROADMAP — ML Engineer Internship Interview

Your guide to **what to learn, in what order, how to work through this repo, and how every topic connects.**
The syllabus is still [ML_Engineer_Interview_Preparation.md](ML_Engineer_Interview_Preparation.md) (unchanged — it is the master checklist). Every syllabus bullet is covered by a module below.

**Contents**
1. [How to follow this repo](#1-how-to-follow-this-repo)
2. [The big picture](#2-the-big-picture)
3. [Modules at a glance (priority and effort)](#3-modules-at-a-glance)
4. [The five phases — every module and sub-topic in detail](#4-the-five-phases)
5. [How the topics connect](#5-how-the-topics-connect)
6. [Pacing — flexible, not a timetable](#6-pacing)
7. [Readiness checks](#7-readiness-checks)
8. [Syllabus coverage map](#8-syllabus-coverage-map)

---

## 1. How to follow this repo

### 1.1 What lives where

```
RAP_interview/
├── ROADMAP.md                      ← you are here (the plan and the map)
├── README.md                       ← quick start, folder map, progress tracker
├── ML_Engineer_Interview_Preparation.md   ← original syllabus checklist
├── requirements.txt                ← all libraries
├── shared/llm_client.py            ← one tiny "call an LLM" helper (mock / local / Anthropic)
├── 00_ … 13_ modules               ← the course, in learning order
├── vocabulary/                     ← searchable dictionaries (python, ml, llm, api) + index
└── mock_interviews/                ← question bank, live-coding drills, debugging drills, scorecard
```

### 1.2 What every module contains

| File / folder | What it is | When to use it |
|---------------|-----------|----------------|
| `README.md` | **The lesson.** Why it matters → builds-on / leads-to → concept map → one explanation per syllabus bullet → gotchas → self-check → interview Q&A | Read first, before touching code |
| `NN_step/main.py` | The smallest runnable example for that step: goal, run command, expected output, commented code, `TRY IT` and `BREAK IT` hints | Predict the output, then run |
| `NN_step/WALKTHROUGH.md` | Line-by-line explanation: syntax table, libraries/functions table, execution trace with real output, experiments, debugging drill, rebuild list, "say it aloud" | After you ran it |
| `exercises.md` | Per step: modify one thing, break it, rebuild from memory, explain aloud | After the walkthrough |
| `solutions/` | Worked answers | Only after you attempted the exercise |
| `broken_*` files | Code that fails on purpose | Debug these **without AI** |

### 1.3 The loop for every step (the syllabus's 10-step practice)

```
 read module README section ─► predict output ─► run main.py ─► read WALKTHROUGH.md
        ▲                                                              │
        │                                                              ▼
 tick self-check  ◄─ say it aloud ◄─ rebuild from memory ◄─ BREAK IT drill ◄─ change ONE thing (TRY IT)
        │
        └─► add/refresh terms in vocabulary/ (your own words)
```

### 1.4 Ground rules
- Do not move on until you can explain the current step in your own words, including **one limitation or trade-off**.
- Read tracebacks yourself first; AI is the last resort, not the first (module 01 trains this).
- Use AI only with the syllabus prompt ("Rule for every AI-assisted coding session"); never keep code you cannot explain.
- Keep a short "stuck log" (what failed, what you tried, root cause) — the template is in `01_debugging_and_environment/DEBUG_JOURNAL_TEMPLATE.md`.
- Do **DSA alongside everything else** (see [§6](#6-pacing)); it is a habit, not a phase.

### 1.5 Effort legend used below
| Symbol | Meaning |
|:------:|---------|
| **P0** | Must be solid. Likely to be coded or tested live. Never cut. |
| **P1** | Know it, demo it, name a trade-off. |
| **P2** | Awareness plus one working demo. |
| **L1–L4** | Depth: L1 recognise · L2 run & explain · L3 write from scratch · L4 debug cold |

---

## 2. The big picture

The ten syllabus topics are **one pipeline** viewed from both ends. The running example is a *text service*:

`text → tokenizer (04) → model (03/04/06) → sampling (05) → prompt / agent wrapper (07/08) → HTTP API (09) → logged & versioned (10) → container (11) → UI (12)`

```
                         YOU WRITE CODE                       Phase 1: 00 Python · 01 Debugging · 02 DSA
                               │
        ┌──────────────────────┴───────────────────────┐
        ▼                                              ▼
   THE MODEL                                    THE PRODUCT AROUND IT
   03 ML foundations     (loss, gradients, training loop)               Phase 2
   04 LLM fundamentals   (tokens, embeddings, attention, logits)
   05 LLM parameters     (lr, batch, temperature, top-p, quantization)
   06 Training & fast inference (datasets, LoRA, KV cache, vLLM)
        │
        ├── 07 Prompt engineering   (talk to the model)               Phase 3
        ├── 08 Agent harness        (loop + tools, no frameworks)
        │
        └────────────► 09 API (Flask+Gunicorn · FastAPI+Uvicorn)      Phase 4
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      10 ML architecture   11 Docker      12 Streamlit/Gradio
              └───────────────┼───────────────┘
                              ▼
                       13 Capstone: everything together             Phase 5
```

If you can narrate that one line and drill into any arrow, you can answer most interview questions.

---

## 3. Modules at a glance

| Phase | Module | Syllabus topic | Priority | Effort | What you will be able to do |
|:-----:|--------|----------------|:--------:|:------:|-----------------------------|
| 1 | [00 Python foundations](00_python_foundations/) | prerequisite | **P0** | M | Write and explain the Python the other modules use |
| 1 | [01 Debugging & environment](01_debugging_and_environment/) | 9 | **P0** | M | Diagnose errors without AI; keep a clean environment |
| 1 | [02 DSA: arrays & hash maps](02_dsa_arrays_hashmaps/) | 10 | **P0** | L (spread out) | Solve array/dict problems and state complexity |
| 2 | [03 ML foundations](03_ml_foundations/) | prerequisite | P1 | M | Train a model by gradient descent; save/load a classifier |
| 2 | [04 LLM fundamentals](04_llm_fundamentals/) | 1 | **P0** | L | Go from text to tokens to logits to a generated token |
| 2 | [05 LLM parameters](05_llm_parameters/) | 2 | P1 | M | Explain and tune training and generation parameters |
| 2 | [06 Training & fast inference](06_llm_training_and_fast_inference/) | 1 | P1 | L | Fine-tune with LoRA; reason about speed and GPU memory |
| 3 | [07 Prompt engineering](07_prompt_engineering/) | 4 | P1 | M | Write, structure, test and defend prompts |
| 3 | [08 Agent harness](08_agent_harness/) | 3 | **P0** | L | Build a tool-calling agent loop with no framework |
| 4 | [09 API development](09_api_development/) | 6 | **P0** | L | Build and serve Flask and FastAPI model APIs |
| 4 | [10 ML system architecture](10_ml_system_architecture/) | 5 | P1 | M | Design and explain an ML system end to end |
| 4 | [11 Docker & Compose](11_docker/) | 7 | P1 | M | Containerise and orchestrate the API |
| 4 | [12 Streamlit / Gradio](12_streamlit_gradio/) | 8 | P2 | S | Build a UI that talks to the API |
| 5 | [13 Capstone](13_capstone_project/) | integration | P2 | M | Run and explain the whole stack |
| 5 | [mock_interviews](mock_interviews/) + [vocabulary](vocabulary/index.md) | readiness | — | S | Turn knowledge into spoken answers |

**Prerequisites beat priority:** you cannot understand temperature (P1, module 05) without logits (P0, module 04), and logits need the ML basics (P1, module 03).

---

## 4. The five phases

Each module below lists **every step folder**, **what the sub-topic teaches**, and **why it matters elsewhere** (relevance). Step folders are the exact folder names, so you can jump straight to them.

### Phase 1 — Foundations: write code, debug code, think in dictionaries

*Goal: be fluent enough that syntax never blocks you, and be able to find your own bugs. DSA starts here and continues through every phase.*

#### 00 Python foundations — P0 · [open](00_python_foundations/)
Needs: nothing. Leads to: everything.

| Step | Sub-topics it contains | Relevance (where you will use it) |
|------|-----------------------|-----------------------------------|
| `01_variables_and_types` | variables, int/float/str/bool/None, type(), f-strings | Reading any code; f-strings in prompts (07) and logs (01) |
| `02_control_flow_and_loops` | if/elif/else, for, while, range, break/continue | Training loop (03), agent loop (08), DSA (02) |
| `03_functions_and_type_hints` | def, arguments/defaults, return, `*args/**kwargs`, type hints | Tool functions (08), FastAPI endpoints and Pydantic (09) |
| `04_collections_and_comprehensions` | list, tuple, dict, set, slicing, comprehensions | Dicts = JSON = tool args (08) = request bodies (09); DSA (02) |
| `05_classes_and_dataclasses` | class, `__init__`, methods, `@dataclass`, inheritance basics | PyTorch `nn.Module` (03), Pydantic models (09), tool objects (08) |
| `06_exceptions_files_json` | try/except/else/finally, raise, `open`, `with`, `json`, `pathlib` | Retries (08), HTTP errors (09), datasets (06), configs (05) |
| `07_modules_decorators_generators` | import, `if __name__ == "__main__"`, decorators, generators | Flask/FastAPI route decorators (09), DataLoader-style iteration (03/06) |

#### 01 Debugging & environment — P0 · [open](01_debugging_and_environment/)
Needs: 00. Extras: `CHEATSHEET.md`, `DEBUG_JOURNAL_TEMPLATE.md`.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_reading_tracebacks` | read bottom-up, call stack, locating the failing line | Every error you will ever hit |
| `02_syntax_and_runtime_errors` | SyntaxError vs runtime error, NameError, IndexError, KeyError | Fast triage in live coding (02) |
| `03_import_and_environment` | ModuleNotFoundError, `sys.path`, venv, `pip list`, version conflicts | Setting up torch/transformers (04), servers (09), images (11) |
| `04_paths_and_cwd` | relative vs absolute paths, working directory, `Path(__file__)` | Model/artifact loading (03, 09), Docker `COPY`/`WORKDIR` (11) |
| `05_types_and_validation` | TypeError, ValueError, validating input | Tool args (08), Pydantic (09) |
| `06_http_and_network` | status codes, connection refused, timeouts, requests/httpx | Calling LLM APIs (08), your own API (09), UI→API (12) |
| `07_env_vars_and_config` | `os.environ`, `.env`, defaults, missing-config errors | Config (05), API settings (09), Docker `ENV` (11) |
| `08_logging_vs_print` | `logging` levels, formatting, when print is enough | Agent traces (08), monitoring (10), `docker logs` (11) |
| `09_minimal_repro_and_rca` | minimal reproducible example, bisecting, root-cause write-up | Debugging round of the interview; capstone drills |
| `10_docker_errors` | reading build/run failures (offline lookup; deepened in 11) | Docker module (11) |

#### 02 DSA: arrays & hash maps — P0 · [open](02_dsa_arrays_hashmaps/)
Needs: 00. Extras: `DSA_PRACTICE_PLAN.md` (suggested daily practice), `DRY_RUN_TEMPLATE.md`.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_complexity_and_dry_runs` | Big-O, brute force vs optimized, dry runs, edge cases | Reasoning about attention cost (04), KV cache (06), caching (10) |
| `02_hashmap_basics` | key-value ops, frequency counting, complement lookup (**Two Sum**), sets (**Contains Duplicate**), **Valid Anagram** | Vocab token→id (04), JSON (07/08/09) |
| `03_array_scans` | traversal, searching, min/max, duplicates | Everyday data handling |
| `04_two_pointers` | opposing/same-direction pointers, sorted arrays | Interview staple |
| `05_sliding_window` | fixed and variable windows | Mirrors context windows (05/08) |
| `06_prefix_sums` | running totals, subarray sums | Cumulative sums behind top-p (05) |
| `07_in_place_and_sorting` | in-place edits, sorting basics | Memory-aware thinking (06) |
| `08_grouping_and_strings` | grouping with dicts, string problems | Text processing for tokenization (04) |

---

### Phase 2 — The model: from gradients to generated tokens

*Goal: understand what a model is, how it is trained, how an LLM turns text into the next token, and what the knobs do.*

#### 03 ML foundations — P1 · [open](03_ml_foundations/)
Needs: 00. Prerequisite for 04–06. The sklearn model from step 08 is reused by 09, 10 and 13.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_numpy_and_data` | arrays, shapes, features/labels | All tensors later |
| `02_linear_regression_numpy` | gradient descent from scratch, loss, learning rate, epochs | The training loop in 05/06 |
| `03_pytorch_tensors_autograd` | tensors, autograd, `.backward()` | Same engine as LLM training (06) |
| `04_pytorch_training_loop` | `nn.Module`, optimizer, DataLoader, batch size, train loop | Fine-tuning (06), parameter tuning (05) |
| `05_overfitting_train_val` | train/val split, overfitting | Why validation loss/perplexity is tracked (06) |
| `06_softmax_cross_entropy` | softmax, logits, cross-entropy | **Temperature and sampling (05), next-token prediction (04)** |
| `07_metrics` | accuracy, precision, recall, F1 | Evaluating prompts (07), monitoring (10) |
| `08_sklearn_text_classifier` | pipeline, `train.py`, `predict.py`, joblib save/load, pytest | Served by the API (09), registry (10), capstone (13) |

#### 04 LLM fundamentals — P0 · [open](04_llm_fundamentals/)
Needs: 03.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_attention_from_scratch` | queries/keys/values, attention scores, shapes | Why long context is costly (06), transformer block |
| `02_tokenizer` | BPE intuition, token ids, special tokens, odd splits | Context window and cost (05, 07, 08), dataset prep (06) |
| `03_embeddings` | `nn.Embedding`, vectors, cosine similarity, position info | How text becomes numbers; semantic search ideas |
| `04_load_pretrained_model` | `AutoTokenizer`/`AutoModelForCausalLM`, parameter count, dtype, memory | Quantization (05), memory calculator (06), API model loading (09) |
| `05_forward_pass_logits` | forward pass, logits → softmax → next-token distribution | **Temperature/top-k/top-p (05)** |
| `06_generation_loop` | hand-written generation loop vs `model.generate` | Sampling parameters (05), KV cache (06) |
| `07_pretrain_vs_finetune_vs_inference` | the three modes and what changes | Fine-tuning/LoRA (06), serving (09/10) |

#### 05 LLM parameters — P1 · [open](05_llm_parameters/)
Needs: 03 (training params), 04 (logits).

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_config_file` | a config file and what each parameter means | Env vars/config in 09 and 11 |
| `02_sampling_from_scratch` | temperature, top-k, top-p, repetition penalty on real logits | Prompt reliability (07), agent determinism (08) |
| `03_hf_generate_compare` | the same settings through `generate()` | Verifies you understand the library |
| `04_max_new_tokens_and_context_window` | length limits, context window, truncation | Context management in the agent (08) |
| `05_quantization` | symmetric quantization, memory vs error | LoRA/QLoRA, vLLM, memory math (06) |
| `06_training_parameters` | learning rate, batch size, epochs, optimizers, losses | Fine-tuning runs (06) |
| `07_experiment_logger` | table of length, latency, memory, quality notes | The "record what changed" habit; monitoring (10) |

#### 06 LLM training & fast inference — P1 · [open](06_llm_training_and_fast_inference/)
Needs: 03, 04, 05. Reference-only (not run): QLoRA and vLLM need a GPU.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_dataset_preparation` | tokenize → chunk → labels shifted → DataLoader | All fine-tuning; data pipelines (10) |
| `02_training_loop` | causal-LM loop, loss going down, gradient clipping | Ties back to 03 step 04 |
| `03_eval_perplexity_checkpoints` | validation loss, perplexity, save/reload checkpoints | Model registry and versioning (10) |
| `04_lora_math_numpy` | low-rank update W + BA, parameter counts | Why LoRA is cheap |
| `05_lora_peft_finetune` | real `peft` fine-tune, adapter size, merge | Adapters as versioned artifacts (10) |
| `06_qlora_reference` | 4-bit base + LoRA (read only) | Quantization (05) |
| `07_hf_inference_batching_kvcache` | KV cache on/off, padding side, batching, dtype | Latency/throughput; API concurrency (09) |
| `08_gpu_memory_calculator` | weights + KV cache memory for a 7B model at fp16/int8/int4 | Workers × model size (09), container limits (11) |
| `09_throughput_benchmark` | tokens/sec vs batch size | Scaling decisions (10) |
| `10_vllm_reference` | PagedAttention, continuous batching (read only) | Serving at scale (10) |

---

### Phase 3 — Talking to the model and building agents

*Goal: control model behaviour with prompts, then wrap the model in a loop that can use tools, safely.*

#### 07 Prompt engineering — P1 · [open](07_prompt_engineering/)
Needs: 04, 05. Runs offline with a simulated model; works with a real model through `shared/llm_client.py`.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_roles_and_instructions` | role, task, instructions, system vs user prompts | Agent system prompts (08) |
| `02_few_shot` | zero-shot vs few-shot examples | Cheap accuracy gains |
| `03_prompt_templates` | f-strings / `str.format` / `Template` | Reusable prompts; 00 string skills |
| `04_structured_json` | JSON output, schemas, validation (hand-written and Pydantic) | Tool calls (08), Pydantic (09) |
| `05_decomposition` | task decomposition, chain-of-thought alternatives | Multi-step agent behaviour (08) |
| `06_prompt_injection` | injection demo and mitigations | Tool safety (08), API hardening (09) |
| `07_prompt_testing` | test set (normal, ambiguous, missing, adversarial), v1/v2/v3 comparison | Metrics (03), monitoring (10) |

#### 08 Agent harness without frameworks — P0 · [open](08_agent_harness/)
Needs: 00, 07. Extras: `broken_examples/` (debug drills), `sandbox_files/`, `fake_llm.py`.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_llm_api_call` | calling an LLM, messages format, backends | `shared/llm_client.py`; API calls (09) |
| `02_agent_loop` | the loop, stop condition, max steps | Core of the topic |
| `03_tool_registry_and_validation` | tool registry, dispatch, argument validation | Pydantic (09), validation thread |
| `04_history_and_context` | conversation history, context limit, truncation/summarisation | Tokens and context window (04/05) |
| `05_memory` | short-term vs long-term memory store | State in services (10) |
| `06_errors_and_retries` | error handling, retry with backoff | Reliability (10), HTTP errors (09) |
| `07_safety_and_permissions` | allow-lists, read-only vs write tools, path sandboxing, confirmation | Injection (07), non-root and least privilege (11) |
| `08_trace_and_full_agent` | JSONL trace of every call; the combined `agent.py` | Logging/monitoring (10); debugging (01) |
| `09_native_tool_calling` | provider-native tool-use APIs vs your JSON protocol | Knowing what frameworks hide |

---

### Phase 4 — Shipping it: API, architecture, containers, UI

*Goal: put the model/agent behind an HTTP API, explain the system around it, package it, and give it a face.*

#### 09 API development — P0 · [open](09_api_development/)
Needs: 00, 01, 03 (saved model). Flask+Gunicorn and FastAPI+Uvicorn, built side by side.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_http_basics` | HTTP, REST, GET vs POST, status codes, curl | UI→API (12), debugging (01) |
| `02_wsgi_asgi_by_hand` | a tiny WSGI app and ASGI app by hand | Explains Gunicorn vs Uvicorn |
| `03_flask_basics` | routes, methods, request/response | |
| `04_fastapi_basics` | routes, path/query params, `/docs` | |
| `05_validation_and_errors` | Pydantic models, 422, error handlers | Validation thread (07, 08) |
| `06_model_api_flask_vs_fastapi` | same `/health` and `/predict` in both frameworks, shared model loader | Serves 03's model; containerised in 11; called by 12 |
| `07_servers_workers_concurrency` | Gunicorn/Uvicorn workers, sync vs async, a model copy per worker | Memory (04/06), scaling (10) |
| `08_config_and_env` | env vars, `.env`, settings | Docker `ENV` (11) |
| `09_testing_and_docs` | pytest + TestClient, httpx, OpenAPI docs | Debugging (01), CI habits |

#### 10 ML system architecture — P1 · [open](10_ml_system_architecture/)
Needs: 03, 09. Extras: `diagrams/` including a design-interview answer template.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_pipeline_end_to_end` | collect → preprocess → features → train → evaluate → serve | The narrative for any design question |
| `02_training_vs_inference_pipelines` | train/serve skew (with a deliberately broken example) | Why preprocessing code must be shared |
| `03_serialization` | joblib/pickle/safetensors, pickle security | Artifacts in 03, 06, 11 |
| `04_model_registry_versioning` | versioned folders, `metadata.json`, pinned vs latest | Capstone retraining (13) |
| `05_storage_and_prediction_log` | sqlite prediction log, model storage | Monitoring |
| `06_batch_vs_realtime` | batch vs real-time inference | Throughput (06), API (09) |
| `07_logging_and_monitoring` | structured logs, latency, simple drift check | Logging thread |
| `08_scalability_and_reliability` | queues, caching, timeouts, retries, load test | Replicas in Compose (11) |

#### 11 Docker & Compose — P1 · [open](11_docker/)
Needs: 09. Note: the Docker daemon was not running when this was written, so builds were not executed; Compose files were validated. If a build fails, that is a debugging drill.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_first_image` | containers vs VMs, image vs container, Dockerfile, build/run | Packages 09 |
| `02_layers_and_cache` | layer caching, base images, requirements ordering | Fast rebuilds |
| `03_ports_env_volumes` | port mapping, env vars, volumes vs bind mounts | Config (05, 09) |
| `04_compose_multi_container` | Compose YAML, networking by service name, multi-container | UI + API + redis; capstone (13) |
| `05_healthchecks_logs_debugging` | health checks, logs, five `broken_*` configs with diagnosis guide | Debugging (01) |
| `06_multistage_and_security` | multi-stage builds, non-root user, secrets, `.dockerignore` | Safety thread (08) |

#### 12 Streamlit / Gradio — P2 · [open](12_streamlit_gradio/)
Needs: 09 (API contract). A mock API is included so it runs standalone.

| Step | Sub-topics it contains | Relevance |
|------|-----------------------|-----------|
| `01_streamlit_basics` | inputs/outputs, buttons, rerun model | |
| `02_streamlit_model_state_uploads` | cached model loading, session state, uploads | Load-once principle (09) |
| `03_streamlit_api_client` | calling the API, spinners, error states | The same API as 09; Compose `API_URL` (11) |
| `04_streamlit_layout_style` | columns, tabs, styling | |
| `05_gradio_basics` | components and callbacks | Callbacks vs rerun comparison |
| `06_gradio_blocks_api_share` | Blocks, API, sharing and deployment | |

---

### Phase 5 — Integrate and rehearse

*Goal: show it works as one system and practise explaining it.*

#### 13 Capstone — P2 · [open](13_capstone_project/)
Needs: 09, 11, 12 (+10). Contains `api/`, `ui/`, `compose.yaml`, `run_local.sh`, and six tasks (retrain, version, log, add `/generate`, expose the agent, break it three ways).

#### Mock interviews and vocabulary — [mock_interviews](mock_interviews/) · [vocabulary](vocabulary/index.md)

| Resource | What it contains | Use |
|----------|------------------|-----|
| `question_bank.md` | 15 cross-module "bridge" questions with answers | Answer aloud, then compare |
| `live_coding_drills.md` | 13 timed tasks (Python, DSA, agent, API, Docker, UI) | Closed-book practice |
| `debugging_drills.md` | the repo's `broken_*` files as a drill list | Debug without AI |
| `scorecard.md` | self-scoring table | Track weak areas |
| `vocabulary/` | dictionaries for Python, ML, LLM, API/deployment + index | Look up, then re-explain in your own words |

---

## 5. How the topics connect

### 5.1 Dependency graph (arrow = "needed first")

```
00 Python ───► 01 Debugging ───────────────────────────────────────────┐ (used by all)
   │  └──► 02 DSA (daily, alongside everything)                        │
   ▼                                                                    ▼
03 ML foundations ─► 04 LLM fundamentals ─► 05 LLM parameters ─► 06 Training & fast inference
   │                       │   │                  │
   │                       │   └──────────────────┴─► 07 Prompting ─► 08 Agent harness
   │                       │                               │               │
   └── sklearn model ──────┼───────────────────────────────┴───────────────┤
                           ▼                                               ▼
                      09 API (Flask / FastAPI) ◄───────────────────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     10 Architecture   11 Docker    12 Streamlit/Gradio
            └──────────────┼──────────────┘
                           ▼
                      13 Capstone
```

### 5.2 Concept threads — one idea, many places

Learn each thread once, deeply, then recognise it everywhere.

| Thread | First appears | Reappears |
|--------|--------------|-----------|
| **Dictionary / JSON** | 00, 02 | 04 vocab · 05 config · 07 JSON output · 08 tool calls · 09 bodies · 11 YAML/env · 12 session state |
| **Loss → gradient → optimizer step** | 03 | 05 (lr, batch, epochs) · 06 (causal-LM loss; LoRA trains small matrices) |
| **Logits → softmax → probabilities** | 03 | 04 next-token distribution · 05 temperature/top-k/top-p · 07 why answers vary · 08 determinism for tests |
| **Tokens** | 04 | 05 limits · 06 chunking, KV cache per token · 07 prompt cost · 08 history truncation · 10 cost/latency |
| **Memory (RAM/GPU)** | 04 params × bytes | 05 quantization · 06 KV cache, LoRA/QLoRA, calculator · 09 one model per worker · 11 container limits |
| **Latency vs throughput** | 06 | 09 workers/async · 10 batch vs real-time · 11 replicas · 12 spinners |
| **Validation** | 00, 01 | 07 LLM JSON · 08 tool args · 09 Pydantic/422 · 12 UI errors |
| **Errors, retries, timeouts** | 00, 01 | 08 backoff · 09 4xx vs 5xx · 10 reliability · 11 health checks |
| **Config & environment** | 01 | 05 config · 09 env/.env · 11 `ENV`/Compose · 12 `API_URL` |
| **Save & load an artifact** | 03 joblib | 06 checkpoints/adapters · 09 load once at startup · 10 registry · 11 bake vs mount |
| **Logging / traces** | 01 | 08 JSONL trace · 09 request logs · 10 monitoring · 11 `docker logs` |
| **Least privilege & safety** | 08 | 07 injection · 09 input limits · 11 non-root, no secrets in images |
| **Big-O thinking** | 02 | 04 attention cost · 06 KV cache trade-off · 10 caching/batching |

### 5.3 "Bridge" questions interviewers like
Full answers are in [mock_interviews/question_bank.md](mock_interviews/question_bank.md).

| Question | Connects |
|----------|----------|
| Why does temperature 0 give the same answer every time? | 03 → 04 → 05 |
| Why does the API run out of memory with 4 workers? | 04 → 05/06 → 09 |
| Your agent loops forever — what do you check? | 08 → 01 |
| The model returned bad JSON and the endpoint crashed | 07 → 08 → 09 → 01 |
| The container exits immediately | 11 → 01 → 09 |
| Why is generation slower for long outputs? | 06 → 04 → 02 |
| How would you serve 100 requests/second? | 10 → 09 → 06 → 11 |
| Explain the code you wrote with AI | every WALKTHROUGH "Say it aloud" |

---

## 6. Pacing

There is no fixed timetable — adapt to your interview date. The numbers below are **rough guides**, not requirements.

### 6.1 Rough effort per module
| Module | Typical hours | Where the time goes |
|--------|:------------:|---------------------|
| 00 Python | 10–14 | collections/comprehensions, classes, exceptions/files |
| 01 Debugging | 6–9 | tracebacks, imports/venv, minimal repro |
| 02 DSA | 15–25 (spread out) | hash maps, two pointers, sliding window, prefix sums |
| 03 ML foundations | 7–10 | gradient descent, PyTorch loop |
| 04 LLM fundamentals | 9–13 | attention, tokenizer, logits → generation |
| 05 LLM parameters | 5–8 | sampling from scratch |
| 06 Training & fast inference | 7–10 | training loop, LoRA, KV cache/memory |
| 07 Prompt engineering | 5–8 | structured JSON, testing |
| 08 Agent harness | 10–14 | tools/validation, context, retries, safety |
| 09 API development | 10–14 | FastAPI + Pydantic, workers |
| 10 Architecture | 4–7 | diagrams, registry, monitoring |
| 11 Docker | 7–10 | Dockerfile, ports/env, Compose |
| 12 UI | 4–6 | Streamlit first, Gradio second |
| 13 Capstone + mocks | 8–12 | run, break, explain |

### 6.2 If you have less time
| Situation | What to do |
|-----------|-----------|
| Short on time overall | Keep all **P0**. For P1/P2 read the README, run each `main.py` once, write one trade-off per topic. |
| Interview is tomorrow | Python basics, tracebacks, Two Sum/Contains Duplicate/Valid Anagram, tokenizer → logits → sample, the agent loop (08 steps 02–03), one FastAPI endpoint. |
| Lots of time | Redo every `broken_*` drill, do the capstone tasks, add your own vocabulary terms and problems. |

### 6.3 Daily habits (any pace)
- **DSA, a little every day** (see `02_dsa_arrays_hashmaps/DSA_PRACTICE_PLAN.md`).
- **Vocabulary**: add or review a few terms after each step.
- **Once a week**: redo two broken-code drills without AI and one mock interview.

---

## 7. Readiness checks

| Check | After | You pass when you can… (without AI) |
|-------|-------|-------------------------------------|
| **A** | Phase 1 | Write a typed function, a class and a `try/except`; read/write JSON; read a traceback and name the cause; solve Two Sum with a dict and state Big-O |
| **B** | Phase 2 | Sketch the training loop (forward → loss → backward → step); tokenize, get logits, apply temperature/top-p, pick a token; explain LoRA and the KV cache; estimate 7B memory at fp16 vs int4 |
| **C** | Phase 3 | Write a prompt that returns validated JSON; build an agent loop with tool registry, validation, max steps, retry and a trace |
| **D** | Phase 4 | Build a validated FastAPI POST endpoint; start it with Uvicorn (and Flask with Gunicorn); write a Dockerfile and Compose file; explain WSGI vs ASGI; build a small UI |
| **E** | Phase 5 | Run the stack end to end, narrate a request, fix three injected bugs |

Track these in [README.md](README.md#progress-tracker).

---

## 8. Syllabus coverage map

| Syllabus topic | Primary module | Also reinforced in |
|----------------|---------------|--------------------|
| 1. Basic LLM training & fast inference | 04 + 06 | 03, 05, 09, 10 |
| 2. LLM parameters | 05 | 03, 04, 06 |
| 3. Agent harness (no frameworks) | 08 | 00, 07, 09 |
| 4. Prompt engineering | 07 | 05, 08 |
| 5. ML system architecture | 10 | 03, 06, 09, 11 |
| 6. Flask+Gunicorn / FastAPI+Uvicorn | 09 | 01, 10, 11, 12 |
| 7. Docker & Compose | 11 | 01, 09, 13 |
| 8. Streamlit / Gradio | 12 | 09, 11, 13 |
| 9. Debugging without AI | 01 | every `broken_*` drill |
| 10. LeetCode: arrays & dictionaries | 02 | 00, 04 |
| Prerequisite: Python syntax | 00 | all |
| Prerequisite: ML basics | 03 | 04–06 |
