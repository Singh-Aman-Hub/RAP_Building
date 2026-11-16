# ML Engineer Internship — Interview Study Pack

A step-by-step, hands-on course built around the syllabus in [ML_Engineer_Interview_Preparation.md](ML_Engineer_Interview_Preparation.md).
Goal: **understand every line of code you use** — explain it, change it, debug it, and rebuild a basic version without AI.

**Read the story → [MASTER_READING_GUIDE.md](MASTER_READING_GUIDE.md)** (25 chapters in 5 acts: every topic is the next chapter of one engineering story).

**Plan and map → [ROADMAP.md](ROADMAP.md)** (what to learn, how to follow the repo, every module and sub-topic with relevance, connections, readiness checks).

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

## Recommended reading order (Nov 2026 curriculum)

Folder numbers are kept stable (so links and notes never break); this is the order to **read** them in. Full rationale and dependency graph: [ROADMAP.md](ROADMAP.md).

```
00 Python → 01 Debugging            (02 DSA daily, in parallel)
03 ML foundations → 15 Neural networks → 04 LLM fundamentals → 05 LLM parameters
→ 16 Modern LLM architecture → 21 LLM inference engineering → 06 Training & LoRA
→ 17 Fine-tuning, alignment & benchmarks → 07 Prompting → 14 RAG
→ 18 OCR & document parsing → 19 Chunking → 08 Agents → 09 API → 20 LLM system LLD
→ 10 ML architecture → 11 Docker → 12 UI → 13 Capstone (+ Track A document QA, Track B inference report) → mock interviews
```

## Folder map (by folder number)

| Folder | Syllabus topic | What's inside |
|--------|----------------|---------------|
| [00_python_foundations](00_python_foundations/) | prerequisite (Python syntax) | syntax tour → classes, exceptions, files, JSON |
| [01_debugging_and_environment](01_debugging_and_environment/) | 9 Debugging without AI | tracebacks, venv, imports, HTTP/Docker errors, broken examples |
| [02_dsa_arrays_hashmaps](02_dsa_arrays_hashmaps/) | 10 LeetCode | arrays, dicts, strings, patterns, 4-week daily plan |
| [03_ml_foundations](03_ml_foundations/) | prerequisite (ML basics) | gradient descent, PyTorch loop, sklearn pipeline |
| [04_llm_fundamentals](04_llm_fundamentals/) | 1 LLM basics | tokenizer, vectors, embeddings, attention, transformer block, logits, generation |
| [05_llm_parameters](05_llm_parameters/) | 2 LLM parameters | lr/batch/epochs, temperature/top-k/top-p, quantization |
| [06_llm_training_and_lora](06_llm_training_and_lora/) | 1 (training) | datasets, training loop, perplexity & checkpoints, LoRA (NumPy + peft), QLoRA reference. *Inference steps moved to 21.* |
| [07_prompt_engineering](07_prompt_engineering/) | 4 Prompt engineering | roles, few-shot, JSON, injection, prompt tests |
| [08_agent_harness](08_agent_harness/) | 3 Agent harness | loop, tools, validation, memory, safety — no frameworks |
| [09_api_development](09_api_development/) | 6 Flask/FastAPI | routes, Pydantic, WSGI/ASGI, Gunicorn/Uvicorn |
| [10_ml_system_architecture](10_ml_system_architecture/) | 5 ML architecture | pipeline, serving, registry, monitoring |
| [11_docker](11_docker/) | 7 Docker | Dockerfile, Compose, networking, health checks |
| [12_streamlit_gradio](12_streamlit_gradio/) | 8 UI | Streamlit and Gradio apps calling the API |
| [13_capstone_project](13_capstone_project/) | integration | whole stack, explained end to end; Track A: document-intelligence QA (18 → 19 → 14/20 → API); Track B: inference report (21) |
| [14_rag_and_retrieval](14_rag_and_retrieval/) | RAG deep dive (extends 07/08_minimal_rag) | chunking, embeddings, vector index, IVF/HNSW/PQ, BM25 + hybrid, re-ranking, full pipeline, evaluation, vector DBs |
| [15_neural_networks](15_neural_networks/) | NN deep dive (bridges 03 → 04) | neuron, MLP + backprop by hand, tiny autograd, activations, init & norms, optimizers, regularization, CNN, RNN/LSTM, precision & stability |
| [16_modern_llm_architecture](16_modern_llm_architecture/) | how a 2026 LLM differs from GPT-2 + its LLD | BPE from scratch, RoPE/ALiBi/YaRN, RMSNorm + SwiGLU, MHA/MQA/GQA/MLA + KV cache, FlashAttention, MoE, a mini Llama, real config, 2026 landscape (speculative decoding moved to 21) |
| [17_finetuning_alignment_and_benchmarks](17_finetuning_alignment_and_benchmarks/) | post-training → benchmarks | chat templates, SFT with loss masking, PEFT family, reward model, DPO, PPO vs GRPO (RLVR), distillation & merging, eval from scratch, LLM-as-judge, 2026 benchmarks, TRL / lm-eval references |
| [18_ocr_and_document_parsing](18_ocr_and_document_parsing/) | OCR + document AI | pixels, Otsu, deskew, segmentation, CNN + CRNN/CTC, CER/WER, reading order (XY-cut), tables, PDF routing, 2026 VLM parsers (PaddleOCR-VL, Docling, MinerU, olmOCR, DeepSeek-OCR) |
| [19_chunking_deep_dive](19_chunking_deep_dive/) | detailed RAG chunking | token-based, recursive rebuilt, structure-aware, semantic, late, contextual retrieval, parent-child/RAPTOR, propositions, layout-aware, evaluation, chunker LLD |
| [20_llm_system_lld](20_llm_system_lld/) | low-level design of LLM/RAG systems | SOLID, domain model, ingestion & query pipelines, design patterns, resilience, caching, gateway & streaming, observability, LLD interview problems, versioning/guardrails/budgets (tested `ragkit` package) |
| [21_llm_inference_engineering](21_llm_inference_engineering/) | **inference & optimization (top priority)** | safetensors/mmap/sharding/HF cache, GGUF, memory calculator, quantization math, calibration/SmoothQuant/QAT, GPTQ/AWQ, KV-cache quantization, prefill vs decode roofline, TTFT/ITL benchmarking, PagedAttention, continuous batching & prefix caching, speculative decoding, parallelism, serving engines (vLLM, SGLang, TensorRT-LLM, llama.cpp, Ollama), cost per token, inference report |
| [CONFUSABLES.md](CONFUSABLES.md) | terms side by side | 380+ "same thing, different names" and look-alike pairs across every topic |
| [TECH_RADAR_2026.md](TECH_RADAR_2026.md) | current tools (Nov 2026) | what is used today, latest versions seen, sources |
| [vocabulary](vocabulary/index.md) | vocabulary files | searchable dictionary: definition, plain English, analogy, snippet |
| [mock_interviews](mock_interviews/) | interview readiness | question banks (00–14 and 15–21), live-coding & debugging drills, 45-min mock scripts, scorecard |
| [shared](shared/) | helper | `llm_client.py` used by modules 07/08 |

### What every module contains
```
NN_module/
├── README.md          the lesson: why it matters, builds-on / leads-to links, concept map, explanations, interview Q&A, self-check
├── 01_step/ main.py + WALKTHROUGH.md + TERMS.md   runnable example, line-by-line explanation, and every term (definition / plain words / quotable example)
├── 02_step/ ...                          each step builds on the previous
├── exercises.md       modify / break / rebuild / explain-aloud tasks per step
└── solutions/         worked answers (try first!)
```
`WALKTHROUGH.md` follows the syllabus's 10-step loop: goal → syntax → libraries → execution trace → experiments → debugging → rebuild → say it aloud.

**How to read a WALKTHROUGH (every step, modules 00–20):**
1. **Story** — the pain point, the spark, the goal.
2. **Theory** — exact definitions and formulas, plus what is used in practice in 2026.
3. **Same thing, different names** — a table of synonyms and look-alikes that are easy to confuse (the full cross-topic list is in [CONFUSABLES.md](CONFUSABLES.md)).
4. **Code, step by step** — the code in small blocks, in order. Each block has three bullets:
   - **In simple words** — what it does, in plain language.
   - **Exactly** — the precise meaning: types, shapes and real values from a real run.
   - **Why it is here** — what breaks without it.
5. **Syntax / libraries tables**, the **real execution trace**, **experiments**, a **debugging drill**, a **rebuild checklist** and a **say-it-aloud** interview box.

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
| [ ] | 06 Training & LoRA | P1 | 7 h | | | [ ] | [ ] |
| [ ] | 07 Prompt engineering | P1 | 7 h | | | [ ] | [ ] |
| [ ] | 08 Agent harness | P0 | 12 h | | | [ ] | [ ] |
| [ ] | 09 API development | P0 | 13 h | | | [ ] | [ ] |
| [ ] | 10 ML system architecture | P1 | 6 h | | | [ ] | [ ] |
| [ ] | 11 Docker | P1 | 9 h | | | [ ] | [ ] |
| [ ] | 12 Streamlit / Gradio | P2 | 5 h | | | [ ] | [ ] |
| [ ] | 13 Capstone | P2 | 7 h | | | [ ] | [ ] |
| [ ] | 14 RAG & retrieval | P0 | 10 h | | | [ ] | [ ] |
| [ ] | 15 Neural networks | P1 | 10 h | | | [ ] | [ ] |
| [ ] | 16 Modern LLM architecture | P0 | 12 h | | | [ ] | [ ] |
| [ ] | 17 Fine-tuning, alignment & benchmarks | P1 | 12 h | | | [ ] | [ ] |
| [ ] | 18 OCR & document parsing | P2 | 9 h | | | [ ] | [ ] |
| [ ] | 19 Chunking deep dive | P1 | 9 h | | | [ ] | [ ] |
| [ ] | 20 LLM system LLD | P1 | 11 h | | | [ ] | [ ] |
| [ ] | 21 LLM inference engineering | P0 | 18 h | | | [ ] | [ ] |
| [ ] | Mock interviews | — | 5 h | | | | |

Gates: [ ] A (00–01) · [ ] B (03–06) · [ ] C (07–08) · [ ] D (09–12) · [ ] E (13 + mocks) · [ ] F (14–21 deep dives)

---

## The rules (from the syllabus)

1. Use AI as a partner, but **never paste code you cannot explain.**
2. Small steps; run the code; change **one** thing at a time.
3. Read the traceback yourself before asking anyone.
4. Rebuild each example from memory, then explain it aloud with one trade-off.
5. Keep the vocabulary files growing in your own words.

Default prompt for AI sessions: copy the "Rule for every AI-assisted coding session" block from the [syllabus file](ML_Engineer_Interview_Preparation.md#rule-for-every-ai-assisted-coding-session).
