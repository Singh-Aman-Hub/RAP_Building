# Master Reading Guide - the story of building an LLM system

One narrative, 25 chapters, 5 acts. Every chapter follows the same arc:

1. **Pain point** - what broke before this existed.
2. **Spark** - the common-sense idea that fixes it.
3. **Minimal code proof** - the smallest runnable script.
4. **Try it & break it** - one tweak, one deliberate bug.
5. **Cliffhanger** - the new problem this creates, which summons the next chapter.

**Status:** every chapter below is DONE: its WALKTHROUGH.md has `The story so far`, `The pain point`, `The spark`, `The new problem (cliffhanger)` and a 3-sentence `Say it aloud (interview box)`. Items marked NEW were added in this pass.

> Folder numbers now match the reading order. `04_llm_fundamentals` was renumbered to data-flow order and every link was updated.

---

## Act I - The Numerical Awakening (`03_ml_foundations`)
| Ch | Title | Read | Status | Cliffhanger (why the next chapter exists) |
|---|---|---|---|---|
| 1 | The Language of Space (NumPy, tensors) | `03/01_numpy_and_data` | DONE | We can hold data as matrices - but how does a machine *find* the right numbers? |
| 2 | The Art of Learning (gradient descent) | `03/02_linear_regression_numpy` | DONE | Writing gradients by hand for one formula is fine; for a 100-layer network it is impossible. |
| 3 | The Modern Engine (autograd, 5-line loop) | `03/03_pytorch_tensors_autograd`, `03/04_pytorch_training_loop` | DONE | The loop drives loss to zero... on data it has seen. Does it understand or memorise? |
| 4 | The Overfitting Dilemma | `03/05_overfitting_train_val` | DONE | We can detect memorising. Now: how do we turn raw scores into a *decision with confidence*? |
| 5 | Decisions & Confidence (logits, softmax, cross-entropy) + metrics | `03/06_softmax_cross_entropy`, `03/07_metrics` | DONE | We can classify anything numeric. Text is not numeric. |
| 6 | The Traditional Text Baseline & Its Fatal Flaw | `03/08_sklearn_text_classifier` (+ new `flaw_demo.py`) | DONE | TF-IDF works, but "king" and "queen" are unrelated columns and word order is lost. We need a new representation, and first a way to feed raw text in at all. -> Act II |

## Act II - Anatomy of an LLM (`04_llm_fundamentals`) - data flows text -> output
| Ch | Title | Read | Status | Cliffhanger |
|---|---|---|---|---|
| prologue | What is an LLM? (next-token prediction, the loop, the map) | `04/00_what_is_an_llm` | DONE | A counting toy sees one word. Before fixing that, how does text become numbers? |
| 7 | The Bridge of Babylon (tokenization, BPE) | `04/01_tokenizer` | DONE | Ids are arbitrary integers: id 464 is not "closer" to 465 in meaning. Need geometry. |
| 8 | Giving Words Geometry (embeddings + positions) | `04/02_vectors_and_similarity` (warm-up) then `04/03_embeddings` | DONE | A word has one fixed vector, and averaging loses order ("bank", "driver parks car"). Need context. |
| 9 | The Great Council of Tokens (self-attention, causal mask) | `04/04_attention_from_scratch` | DONE | One attention layer is a single mixing step. How do we make it deep and trainable? |
| 10 | Assembling the Beast (transformer block, pretrained model, memory) | `04/05_transformer_block` (NEW) then `04/06_load_pretrained_model` | DONE | The model exists. What does it actually output? |
| 11 | The Oracle Speaks (forward pass, logits, loss) | `04/07_forward_pass_logits` | DONE | One forward pass gives ONE next token. We want a paragraph. |
| 12 | The Endless Loop (generation, KV cache) | `04/08_generation_loop` | DONE | Greedy loops and repeats; and the cache eats memory. -> Act III |
| 13 | The Life Cycle (pretrain / fine-tune / inference) | `04/09_pretrain_vs_finetune_vs_inference` | DONE | We can run a model; can we control it, shrink it, specialise it? |

## Act III - Taming the Model (`05_llm_parameters`, `06_llm_training_and_fast_inference`)
| Ch | Title | Read | Status | Cliffhanger |
|---|---|---|---|---|
| 14 | Rolling the Dice (temperature, top-k, top-p) | `05/01_config_file`, `05/02_sampling_from_scratch`, `05/03`, `05/04` | DONE | Great output, but the model will not fit on our GPU. |
| 15 | Squeezing into VRAM (quantization) | `05/05_quantization`, `06/08_gpu_memory_calculator` | DONE | Inference fits. Fine-tuning needs gradients and optimizer state: 4-16x more memory. |
| 16 | The Surgical Fine-Tune (LoRA, QLoRA) | `06/01-06` (data, loop, eval, `04_lora_math_numpy`, `05_lora_peft`, `06_qlora`) | DONE | Tuned model works; one user at a time is too slow. |
| 17 | Production Speed (KV/batching, vLLM, PagedAttention) | `06/07`, `06/09`, `06/10`, `06/11_paged_attention_fragmentation` | DONE (fragmentation demo NEW) | The model is fast, but a raw model only completes text. How do we instruct it? -> Act IV |

## Act IV - Expanding the Mind (`07_prompt_engineering`, `08_agent_harness`)
| Ch | Title | Read | Status | Cliffhanger |
|---|---|---|---|---|
| 18 | The Art of Instruction (system/user prompts, few-shot, JSON) | `07/01-07` | DONE (Pydantic `ValidationError` + retry added in `04_structured_json`) | Prompts cannot add facts the model never saw; it hallucinates. |
| 19 | The Memory Externalizer (minimal RAG) | `07/08_minimal_rag` | NEW, DONE | The model can now answer from facts but still only *talks*. It cannot act. |
| 20 | Giving the Brain Hands (agent loop, tools, permissions) | `08/01-09` | DONE | The agent works in a notebook. Others need to call it. -> Act V |

## Act V - Shipping to Production (`09`-`13`)
| Ch | Title | Read | Status | Cliffhanger |
|---|---|---|---|---|
| 21 | The Castle Gate (FastAPI, load once, SSE streaming) | `09/01-09` + `09/10_streaming_sse` | DONE (SSE step NEW) | It serves one request; how do we keep it correct over months? |
| 22 | The Industrial Blueprint (train/serve skew, prediction log, drift) | `10/01-08` + `10/09_drift_monitoring` | DONE (drift demo NEW) | Works on my machine only. |
| 23 | The Shipping Container (Docker, Compose) | `11/01-06` | DONE | Humans cannot test it through curl. |
| 24 | The Face of AI (Streamlit/Gradio) | `12/01-06` | DONE | Everything exists as pieces. |
| 25 | The Grand Synthesis (capstone) | `13_capstone_project` | DONE | - |

---

## What changed in this pass
| Gap | Resolution |
|---|---|
| Attention before tokenizer/embeddings | `04_llm_fundamentals` renumbered to data-flow order (00 prologue, 01 tokenizer, 02 vectors, 03 embeddings, 04 attention, 05 transformer block, 06 load model, 07 forward pass, 08 generation, 09 life cycle); all links fixed |
| No "fatal flaw" ending the TF-IDF chapter | `03_ml_foundations/08_sklearn_text_classifier/flaw_demo.py` (king/queen cosine 0.0, "dog bites man" = "man bites dog") |
| Transformer block had no code | `04/05_transformer_block` (multi-head attention, residual, pre-LayerNorm, MLP, stacking; parameter formula checked against a real distilgpt2 block) |
| No RAG | `07/08_minimal_rag` |
| No SSE streaming | `09/10_streaming_sse` |
| Drift was only mentioned | `10/09_drift_monitoring` (PSI/KS on logged inputs) |
| No PagedAttention demo | `06/11_paged_attention_fragmentation` |
| Pydantic errors not shown | `07/04_structured_json` prints a real `ValidationError` and a retry message built from it |
| 5-part arc, shape comments, interview box | applied to every step in modules 03-13; capstone README has a chapter-to-file map |

## Known limits
- Docker commands were not run (daemon off here); the existing "typical output" blocks in 11 and 13 are unchanged and labelled as such.
- Timing numbers (KV-cache speed-up, throughput) vary by machine; walkthroughs claim trends, not exact seconds.
- The "frozen model" in the RAG step is a scripted simulator: it proves the retrieval pipeline, not real-model behaviour.
- Agent-written walkthroughs in modules 03-13 should be skimmed once by a human.
