# ML Engineer Internship — Interview Study Pack

A step-by-step, hands-on course built around the syllabus in [ML_Engineer_Interview_Preparation.md](ML_Engineer_Interview_Preparation.md).
Goal: **understand every line of code you use** — explain it, change it, debug it, and rebuild a basic version without AI.

**Start here → [ROADMAP.md](ROADMAP.md)** (what to learn, how to follow the repo, every module and sub-topic with relevance, connections, readiness checks).

---

## Quick start

```bash
cd RAP_interview

# 1. Environment (Python 3.10+; developed with 3.12). uv is fastest, plain pip works too.
uv venv --python 3.12 .venv && source .venv/bin/activate      # or: python3 -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt                            # or: pip install -r requirements.txt

# 2. Sanity check
python shared/llm_client.py                                   # should print: Hi there! / 4 / {'label': 'positive', 'score': 0.9}

# 3. Begin with module 00
open 00_python_foundations/README.md
```

Everything runs on a laptop CPU (tested on an Apple M1). A few scripts download a small model on first run (noted in each script's `NEEDS:` line).
Anything that really needs a GPU (vLLM, QLoRA) is a clearly named `*_reference.py` that is **read, not run**.

### LLM backends (modules 07 and 08)
Prompt and agent examples run **offline by default** with a scripted mock model, which is great for testing your harness. To try real models:

```bash
LLM_BACKEND=local     python 07_prompt_engineering/<step>/main.py    # small open model via transformers (~1 GB download)
LLM_BACKEND=anthropic ANTHROPIC_API_KEY=sk-... python 08_agent_harness/<step>/main.py   # hosted model
```
See [shared/llm_client.py](shared/llm_client.py) — it is short and worth reading once.

---

## Folder map (in learning order)

| Folder | Syllabus topic | What's inside |
|--------|----------------|---------------|
| [00_python_foundations](00_python_foundations/) | prerequisite (Python syntax) | syntax tour → classes, exceptions, files, JSON |
| [01_debugging_and_environment](01_debugging_and_environment/) | 9 Debugging without AI | tracebacks, venv, imports, HTTP/Docker errors, broken examples |
| [02_dsa_arrays_hashmaps](02_dsa_arrays_hashmaps/) | 10 LeetCode | arrays, dicts, strings, patterns, 4-week daily plan |
| [03_ml_foundations](03_ml_foundations/) | prerequisite (ML basics) | gradient descent, PyTorch loop, sklearn pipeline |
| [04_llm_fundamentals](04_llm_fundamentals/) | 1 LLM basics | tokenizer, embeddings, attention, logits, generation |
| [05_llm_parameters](05_llm_parameters/) | 2 LLM parameters | lr/batch/epochs, temperature/top-k/top-p, quantization |
| [06_llm_training_and_fast_inference](06_llm_training_and_fast_inference/) | 1 (training + inference) | datasets, training loop, LoRA/QLoRA, KV cache, vLLM |
| [07_prompt_engineering](07_prompt_engineering/) | 4 Prompt engineering | roles, few-shot, JSON, injection, prompt tests |
| [08_agent_harness](08_agent_harness/) | 3 Agent harness | loop, tools, validation, memory, safety — no frameworks |
| [09_api_development](09_api_development/) | 6 Flask/FastAPI | routes, Pydantic, WSGI/ASGI, Gunicorn/Uvicorn |
| [10_ml_system_architecture](10_ml_system_architecture/) | 5 ML architecture | pipeline, serving, registry, monitoring |
| [11_docker](11_docker/) | 7 Docker | Dockerfile, Compose, networking, health checks |
| [12_streamlit_gradio](12_streamlit_gradio/) | 8 UI | Streamlit and Gradio apps calling the API |
| [13_capstone_project](13_capstone_project/) | integration | whole stack, explained end to end |
| [vocabulary](vocabulary/index.md) | vocabulary files | searchable dictionary: definition, plain English, analogy, snippet |
| [mock_interviews](mock_interviews/) | interview readiness | question banks, live-coding & debugging drills, scorecard |
| [shared](shared/) | helper | `llm_client.py` used by modules 07/08 |

### What every module contains
```
NN_module/
├── README.md          the lesson: why it matters, builds-on / leads-to links, concept map, explanations, interview Q&A, self-check
├── 01_step/ main.py + WALKTHROUGH.md     smallest runnable example, then the line-by-line explanation
├── 02_step/ ...                          each step builds on the previous
├── exercises.md       modify / break / rebuild / explain-aloud tasks per step
└── solutions/         worked answers (try first!)
```
`WALKTHROUGH.md` follows the syllabus's 10-step loop: goal → syntax → libraries → execution trace → experiments → debugging → rebuild → say it aloud.

### Build / refresh the vocabulary dictionaries
Terms are written in `vocabulary/_drafts/*.md` and compiled into the four dictionaries plus the index:
```bash
python vocabulary/build_vocabulary.py
```
Add your own terms to a draft file (e.g. `vocabulary/_drafts/mine.md`) in the same format, then re-run.

---

## Progress tracker

Tick as you pass each gate from [ROADMAP.md §7](ROADMAP.md#7-readiness-checks). Update the date to see your pace.

| Done | Module | Priority | Rough hours | Started | Finished | Rebuilt from memory? | Explained aloud? |
|:----:|--------|:--------:|-------:|---------|----------|:--------------------:|:----------------:|
| [ ] | 00 Python foundations | P0 | 12 h | | | [ ] | [ ] |
| [ ] | 01 Debugging & environment | P0 | 8 h | | | [ ] | [ ] |
| [ ] | 02 DSA (daily) | P0 | 20 h | | | [ ] | [ ] |
| [ ] | 03 ML foundations | P1 | 9 h | | | [ ] | [ ] |
| [ ] | 04 LLM fundamentals | P0 | 11 h | | | [ ] | [ ] |
| [ ] | 05 LLM parameters | P1 | 7 h | | | [ ] | [ ] |
| [ ] | 06 Training & fast inference | P1 | 9 h | | | [ ] | [ ] |
| [ ] | 07 Prompt engineering | P1 | 7 h | | | [ ] | [ ] |
| [ ] | 08 Agent harness | P0 | 12 h | | | [ ] | [ ] |
| [ ] | 09 API development | P0 | 13 h | | | [ ] | [ ] |
| [ ] | 10 ML system architecture | P1 | 6 h | | | [ ] | [ ] |
| [ ] | 11 Docker | P1 | 9 h | | | [ ] | [ ] |
| [ ] | 12 Streamlit / Gradio | P2 | 5 h | | | [ ] | [ ] |
| [ ] | 13 Capstone | P2 | 7 h | | | [ ] | [ ] |
| [ ] | Mock interviews | — | 5 h | | | | |

Gates: [ ] A (00–01) · [ ] B (03–06) · [ ] C (07–08) · [ ] D (09–12) · [ ] E (13 + mocks)

---

## The rules (from the syllabus)

1. Use AI as a partner, but **never paste code you cannot explain.**
2. Small steps; run the code; change **one** thing at a time.
3. Read the traceback yourself before asking anyone.
4. Rebuild each example from memory, then explain it aloud with one trade-off.
5. Keep the vocabulary files growing in your own words.

Default prompt for AI sessions: copy the "Rule for every AI-assisted coding session" block from the [syllabus file](ML_Engineer_Interview_Preparation.md#rule-for-every-ai-assisted-coding-session).
