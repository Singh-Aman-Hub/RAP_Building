# Confusables — same thing, different names (and look-alikes that are NOT the same)

When you read papers, docs, and interview questions, one idea often has several names, and some pairs *look* the same but are not. This sheet puts them side by side, by topic.

**Same thing?** column: **Yes** = pure synonyms. **Nearly** = same idea, small scope or usage difference. **No** = different things that get mixed up.

Module links go to the study-pack folders (some are still being written). See also [TECH_RADAR_2026.md](TECH_RADAR_2026.md) for tools, [TERMS_INDEX.md](TERMS_INDEX.md) and [vocabulary/index.md](vocabulary/index.md) for definitions, [ROADMAP.md](ROADMAP.md). Alphabetical index at the [end](#alphabetical-index).

## Topics
1. [Python and tooling](#python-and-tooling)
2. [Classic ML](#classic-ml)
3. [Neural networks](#neural-networks)
4. [Transformers and LLM architecture](#transformers-and-llm-architecture)
5. [Tokenization](#tokenization)
6. [Training and fine-tuning](#training-and-fine-tuning)
7. [Alignment and RL](#alignment-and-rl)
8. [Inference and serving](#inference-and-serving)
9. [Quantization](#quantization)
10. [Evaluation and benchmarks](#evaluation-and-benchmarks)
11. [Prompting and agents](#prompting-and-agents)
12. [RAG and retrieval](#rag-and-retrieval)
13. [Chunking](#chunking)
14. [Embeddings and vector DBs](#embeddings-and-vector-dbs)
15. [OCR and document parsing](#ocr-and-document-parsing)
16. [APIs and deployment](#apis-and-deployment)
17. [Docker](#docker)
18. [MLOps](#mlops)
19. [LLD and system design](#lld-and-system-design)

---

## Python and tooling
Modules: [00](00_python_foundations/), [01](01_debugging_and_environment/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| parameter / argument (function) | No | Parameter = name in the `def`; argument = value you pass in the call | Parameter is the parking space, argument is the car |
| `==` / `is` | No | `==` compares values; `is` compares object identity | `==` same content, `is` same object |
| list / tuple | No | List is mutable; tuple is immutable and hashable if its items are | List edits, tuple is sealed |
| list / array (NumPy) | No | Python list holds any objects; ndarray holds one dtype in contiguous memory and does vectorized math | List is a bag, array is a grid |
| dict / set | No | Dict maps key to value; set holds unique keys only | Set is a dict with no values |
| shallow copy / deep copy | No | Shallow copies the outer container only; deep copies nested objects too | Shallow shares the insides |
| generator / iterator / iterable | No | Iterable can be looped; iterator has `__next__`; generator is an iterator made by `yield` | Iterable is the book, iterator is the bookmark |
| decorator / context manager | No | Decorator wraps a function at definition; context manager wraps a block with `with` | `@` wraps a function, `with` wraps a block |
| `__str__` / `__repr__` | No | `__str__` is for readers; `__repr__` is for developers and ideally rebuilds the object | repr is for debugging |
| module / package / library | Nearly | Module = one `.py` file; package = folder of modules; library = informal "a bunch of code you import" | File, folder, product |
| venv / virtualenv / conda env / uv venv | Nearly | All isolate packages; venv is stdlib, virtualenv is the older third-party tool, conda also manages non-Python binaries, uv venv is the fast creator | Different makers of the same sandbox |
| pip / uv pip / conda install | Nearly | pip and `uv pip` install from PyPI (uv is much faster); conda installs from conda channels including non-Python libs | Same job, different engine |
| requirements.txt / pyproject.toml / lock file | No | requirements lists loose pins; pyproject declares project metadata and deps; lock file freezes the exact resolved set | Wish list vs recipe vs receipt |
| traceback / stack trace / exception | Nearly | Exception is the error object; traceback (stack trace) is the printed call chain that led to it | Exception is the crash, traceback is the footprints |
| logging / print | No | Logging has levels, handlers, timestamps; print just writes to stdout | Print is a shout, logging is a record |
| unit test / integration test | No | Unit tests one function in isolation; integration tests parts working together | One brick vs the wall |
| sync / async (Python) | No | Sync blocks while waiting; async yields to the event loop during I/O waits | Async is cooperative waiting |
| thread / process (Python) | No | Threads share memory (and the GIL); processes have separate memory and bypass the GIL | Threads share a kitchen, processes each have one |
| GIL / free-threading | No | GIL lets one thread run Python bytecode at a time; free-threaded builds (3.13+ optional, supported in 3.14) remove it | Free-threading is the GIL-off build |
| `dataclass` / `pydantic.BaseModel` | No | Dataclass is plain structure with no validation; Pydantic validates and coerces types at runtime | Dataclass trusts, Pydantic checks |
| `typing.Optional[X]` / `X or None` / `X \| None` | Yes | All mean "X or None"; `X \| None` is the modern spelling | Same type, shorter syntax |
| type hint / type check | No | Hints are annotations ignored at runtime; a checker (mypy, pyright) or Pydantic enforces them | Hints are labels, not locks |
| linter / formatter | No | Linter finds likely bugs and style issues; formatter rewrites layout only (Ruff does both) | Linter warns, formatter tidies |
| `*args` / `**kwargs` | No | `*args` collects extra positional args as a tuple; `**kwargs` collects keyword args as a dict | Star = list, double star = dict |
| relative path / absolute path / cwd | No | Relative resolves from the current working directory; absolute starts at root; cwd is where the process was launched | cwd is "where am I standing" |

## Classic ML
Module: [03](03_ml_foundations/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| feature / predictor / independent variable / input variable / attribute | Yes | Different fields' names for the input columns `X` | X by many names |
| label / target / response / dependent variable / ground truth | Yes | The thing you predict, `y` | y by many names |
| sample / example / instance / observation / row / data point | Yes | One record in the dataset | One row |
| weights / parameters / params | Yes | Values the model learns during training | What training changes |
| parameters / hyperparameters | No | Parameters are learned; hyperparameters (learning rate, depth) are set by you | Learned vs chosen |
| loss function / cost function / objective / error function | Nearly | Loss is per-example, cost is often its average, objective is the whole thing optimized (may include regularization) | Loss is one miss, cost is the average |
| training / validation / test set | No | Train fits parameters; validation tunes hyperparameters; test is touched once for the final score | Study, practice exam, real exam |
| overfitting / underfitting | No | Overfit = great train, poor test (memorized); underfit = poor on both (too simple) | Memorizer vs lazy learner |
| bias (error) / bias (intercept) / bias (fairness) | No | Three meanings: systematic error from simple model; the `b` in `wx+b`; unfair group behaviour | Ask which "bias" is meant |
| bias-variance / overfit-underfit | Nearly | Same trade-off seen from error decomposition vs from train/test behaviour | High bias = underfit, high variance = overfit |
| L1 / L2 regularization (lasso / ridge) | No | L1 penalizes absolute weights and makes some exactly zero; L2 penalizes squares and shrinks all | L1 deletes, L2 shrinks |
| regularization / weight decay | Nearly | Equal for plain SGD with L2; with Adam they differ, which is why AdamW decouples decay | AdamW fixes the mismatch |
| precision / recall / accuracy / F1 | No | Precision = correct among predicted positives; recall = found among actual positives; accuracy = all correct; F1 = harmonic mean of P and R | Precision is purity, recall is coverage |
| recall / sensitivity / true positive rate (TPR) | Yes | Same formula TP/(TP+FN) | Medical word for recall |
| specificity / true negative rate | Yes | TN/(TN+FP) | Recall for the negatives |
| ROC-AUC / PR-AUC | No | ROC-AUC uses TPR vs FPR (fine for balanced data); PR-AUC uses precision vs recall (better for rare positives) | Imbalanced data, use PR |
| classification / regression | No | Classification predicts a category; regression predicts a number | Label vs number |
| supervised / unsupervised / self-supervised | No | Supervised needs human labels; unsupervised finds structure with none; self-supervised creates labels from the data itself (next-token) | LLM pretraining is self-supervised |
| clustering / classification | No | Clustering groups unlabeled data; classification assigns known classes | Unknown groups vs known classes |
| gradient descent / SGD / mini-batch GD / batch GD | Nearly | Same update rule, different batch size: all data, one sample, or a small batch. In deep learning "SGD" usually means mini-batch | Batch size is the dial |
| epoch / iteration / step / batch | No | Epoch = one pass through the dataset; step/iteration = one parameter update; batch = samples per step | Many steps per epoch |
| data leakage / contamination | Nearly | Leakage = test info reaches training; contamination is the LLM-era word for benchmark text in the training set | Leakage is the general term |
| cross-validation / k-fold | Nearly | K-fold is the common type of cross-validation | K-fold is a flavour of CV |
| normalization / standardization | No | Normalization usually rescales to [0,1]; standardization gives mean 0, std 1 (naming is used loosely, check context) | Min-max vs z-score |
| one-hot / label encoding / ordinal encoding | No | One-hot makes a binary column per class; label/ordinal assigns an integer per class | Columns vs integers |
| bagging / boosting | No | Bagging trains models in parallel on resamples (random forest); boosting trains sequentially to fix earlier errors (XGBoost) | Parallel crowd vs sequential coach |
| random forest / decision tree | No | Forest is many randomized trees averaged | Forest = many trees |
| probability / likelihood | No | Probability: data given fixed parameters; likelihood: parameters given fixed data | Same formula, different question |
| logit / log-odds / sigmoid output | Nearly | Logit = log-odds = raw score before sigmoid; sigmoid turns it into a probability | Logit goes in, probability comes out |
| softmax / sigmoid | No | Sigmoid squashes one score to 0-1; softmax turns a vector into probabilities summing to 1 | Binary vs multi-class |

## Neural networks
Module: [15](15_neural_networks/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| MLP / feed-forward network / dense network / fully connected network / FFN | Nearly | Same stack of linear layers plus nonlinearity; "FFN" is the name inside a transformer block | Dense layers by many names |
| linear layer / dense layer / fully connected layer / `nn.Linear` | Yes | `y = xW + b` | One matrix multiply plus bias |
| epoch / step / batch / mini-batch | No | See Classic ML: pass vs update vs samples-per-update | Same trio |
| activation function / non-linearity | Yes | Function applied after a linear layer so the net can model curves | What makes depth useful |
| ReLU / GELU / SiLU (Swish) / SwiGLU | No | ReLU clips negatives; GELU and SiLU are smooth; SwiGLU is a gated FFN using SiLU, used in Llama-style models | Smooth, gated successors of ReLU |
| backpropagation / autograd / automatic differentiation | Nearly | Autograd is the engine; backprop is the algorithm it runs on a computation graph (reverse-mode) | Backprop is the recipe, autograd the cook |
| gradient / derivative / Jacobian | Nearly | Gradient = vector of partial derivatives of a scalar loss; Jacobian = matrix when output is a vector | Gradient is the loss's slope |
| vanishing gradient / exploding gradient | No | Gradients shrink to ~0 vs blow up through many layers | Too faint vs too loud |
| gradient clipping / gradient accumulation | No | Clipping caps gradient size for stability; accumulation sums gradients over micro-batches to fake a bigger batch | Clip = safety, accumulate = bigger batch |
| BatchNorm / LayerNorm / RMSNorm / GroupNorm | No | BatchNorm normalizes across the batch; LayerNorm across features per token; RMSNorm skips mean subtraction (cheaper); GroupNorm across channel groups | Batch vs token vs cheaper token |
| pre-LN / post-LN | No | Norm before the sublayer (stable, modern) vs after (original transformer, harder to train deep) | Pre-norm is the modern default |
| dropout / weight decay / early stopping | No | Three different regularizers: random zeroing of activations, shrinking weights, stopping training early | Different ways to fight overfitting |
| SGD / Adam / AdamW | No | SGD uses raw gradients (with momentum); Adam adds per-parameter adaptive rates; AdamW decouples weight decay | AdamW is Adam done right |
| learning rate / step size | Yes | Size of the update step | Speed dial |
| learning-rate schedule / warmup / cosine decay | Nearly | Schedule is the overall plan; warmup ramps up at the start, cosine decays smoothly after | Warm start then glide down |
| initialization (Xavier / He / Kaiming) | Nearly | Xavier (Glorot) suits tanh/sigmoid; He (Kaiming) suits ReLU; Kaiming and He are the same person's init | He = Kaiming |
| residual connection / skip connection / shortcut | Yes | Add the input to a layer's output: `x + f(x)` | The highway |
| CNN / RNN / LSTM / GRU / Transformer | No | Convolution for local grids; recurrent nets carry state step by step (LSTM/GRU add gates); transformers use attention in parallel | Local, sequential, global |
| encoder / decoder / autoencoder | No | Encoder maps input to representation; decoder generates output; autoencoder trains an encoder-decoder to reconstruct input | Compress then rebuild |
| VAE / GAN / diffusion | No | VAE: probabilistic encode-decode; GAN: generator vs discriminator game; diffusion: learn to denoise step by step | Three generators |
| float32 / float16 / bfloat16 / TF32 | No | fp32 = 8 exp bits, 23 mantissa; fp16 = 5 exp, 10 mantissa (narrow range); bf16 = 8 exp, 7 mantissa (fp32 range, less precision); TF32 = fp32 range with 10-bit mantissa on tensor cores | bf16 = range, fp16 = precision |
| mixed precision / AMP | Yes | Compute in low precision, keep master weights and sensitive ops in fp32 | Speed with safety net |
| `model.train()` / `model.eval()` / `torch.no_grad()` | No | train/eval switch layer behaviour (dropout, BatchNorm); `no_grad` stops gradient tracking; use both for inference | Mode vs memory |
| tensor `view` / `reshape` / `permute` / `transpose` | No | view needs contiguous memory and shares storage; reshape copies if needed; permute/transpose reorder axes | Reshape reshapes, permute reorders |
| `detach()` / `.item()` / `.cpu()` | No | detach cuts from the graph; item gives a Python number; cpu moves device | Graph, number, device |

## Transformers and LLM architecture
Module: [16](16_modern_llm_architecture/), also [04](04_llm_fundamentals/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| weights / parameters / params | Yes | The learned numbers; "7B" means 7 billion of them | The model's memory |
| activations / hidden states / intermediate representations | Nearly | Layer outputs for a given input (not learned); "hidden states" is the HF name | Per-input, not stored in the model |
| embedding / vector / representation / latent | Nearly | A list of numbers standing for meaning; "embedding" is a learned lookup or model output, "representation" is the abstract idea | Embedding is the concrete form |
| token embedding / positional embedding | No | Token embedding encodes which token; positional encodes where | What vs where |
| context window / context length / max_position_embeddings / sequence length / max_seq_len | Nearly | All mean tokens a model can attend to at once; `max_position_embeddings` is the config field, "sequence length" is also the training/batch length, which can be shorter than the model's max | Window size in tokens |
| prompt tokens / input tokens / prefill tokens | Nearly | What you send; prefill is the compute phase that processes them | Input side |
| completion tokens / output tokens / generated tokens / decode tokens | Nearly | What the model produces; decode is the phase producing them | Output side |
| self-attention / cross-attention | No | Self: tokens attend to the same sequence; cross: queries from one sequence attend to another (encoder output) | Look at myself vs look at the source |
| causal attention / masked attention / autoregressive mask | Yes | Each token sees only earlier tokens | No peeking at the future |
| bidirectional attention / full attention | Nearly | Tokens see both sides (BERT-style); "full" can also mean non-sparse | BERT sees all |
| MHA / MQA / GQA / MLA | No | Multi-head: separate K,V per head; multi-query: one shared K,V; grouped-query: K,V shared per group (Llama-style); multi-head latent: compress K,V into a latent (DeepSeek) | Fewer K,V = smaller KV cache |
| attention head / head dimension / `d_model` / `d_ff` | No | Head = one attention subspace; head_dim = its width; d_model = hidden size; d_ff = FFN inner width | Sizes of different boxes |
| encoder-only / decoder-only / encoder-decoder | No | BERT-style understanding; GPT/Llama-style generation; T5-style seq2seq | BERT, GPT, T5 |
| LLM / foundation model / base model / pretrained model | Nearly | Base = pretrained, not instruction-tuned; foundation model is the broad term (may be non-text) | Base is the raw material |
| base model / instruct model / chat model / aligned model | Nearly | Base predicts text; instruct/chat is tuned to follow instructions and a chat template; "aligned" adds preference tuning | Raw vs polite |
| SLM / LLM | Nearly | Small language models (roughly under 10B) vs large; no formal cutoff | Size label, not a type |
| dense model / MoE (mixture of experts) | No | Dense uses all parameters per token; MoE routes each token to a few experts | Total vs active params |
| total params / active params | No | MoE has many total but uses few per token; speed follows active, memory follows total | Memory vs compute |
| sinusoidal / learned / RoPE / ALiBi positions | No | Fixed sine waves, trained table, rotary on Q/K (modern default), linear distance bias | RoPE rotates |
| FlashAttention / PagedAttention | No | FlashAttention is a fused exact kernel that avoids materializing the attention matrix; PagedAttention manages KV-cache memory in pages (vLLM) | Flash = speed of attention, Paged = memory of cache |
| FlashAttention / sparse attention / sliding-window attention | No | Flash is exact, just faster; sparse and sliding window change which tokens are attended | Exact vs approximate pattern |
| KV cache / `past_key_values` | Yes | Saved keys and values of earlier tokens so decode does not recompute them; `past_key_values` is the Hugging Face argument name | Same cache, API name |
| KV cache / prompt cache / prefix cache | Nearly | KV cache is the mechanism; prompt/prefix caching reuses it across requests with an identical prefix | Reuse across calls |
| prefill / decode | No | Prefill processes the prompt in parallel (compute-bound); decode makes one token at a time (memory-bandwidth-bound) | Read then write |
| logits / probabilities / log-probs | No | Logits are raw scores; softmax gives probabilities; log-probs are their logs | Score, chance, log-chance |
| perplexity / loss / cross-entropy | Nearly | Perplexity = exp(average cross-entropy loss) | Loss in friendlier units |
| scaling laws / Chinchilla | Nearly | Scaling laws relate loss to size, data, compute; Chinchilla is the specific compute-optimal result (about balanced tokens and params) | Chinchilla is one law |
| reasoning model / thinking model / chain-of-thought model | Nearly | Trained (often with RL) to emit long reasoning tokens before answering | Thinks before speaking |
| multimodal / VLM / vision-language model | Nearly | VLM = image+text in, text out; multimodal also includes audio/video | VLM is a type of multimodal |
| open weights / open source (models) | No | Open weights: downloadable weights under some license; open source in the strict sense also releases data, code, permissive license | Weights alone aren't the recipe |

## Tokenization
Modules: [04](04_llm_fundamentals/), [16](16_modern_llm_architecture/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| token / word / subword / character | No | Token is the unit the model sees, usually a subword piece, not a word | About 3/4 of an English word |
| tokenizer / vocabulary / vocab | No | Tokenizer is the algorithm plus rules; vocab is its list of tokens | Machine vs menu |
| BPE / byte-level BPE / WordPiece / Unigram / SentencePiece | No | BPE merges frequent pairs; byte-level BPE starts from bytes (no unknowns); WordPiece (BERT) picks merges by likelihood; Unigram prunes a big vocab; SentencePiece is a library that implements BPE or Unigram | SentencePiece is a toolbox, not an algorithm |
| tiktoken / HF tokenizers / SentencePiece | No | Three libraries: OpenAI's fast BPE, Hugging Face's Rust tokenizers, Google's SentencePiece | Libraries, not algorithms |
| special tokens / control tokens / BOS / EOS / PAD | Nearly | Reserved tokens such as BOS (start), EOS (end), PAD (fill) | Punctuation for the model |
| EOS / stop token / stop sequence | No | EOS is a token the model emits; a stop sequence is a string you ask the API to halt on | Model's own end vs your end |
| padding / truncation | No | Padding lengthens short sequences to a batch length; truncation cuts long ones | Fill vs cut |
| attention mask / padding mask / causal mask | No | Attention mask marks real vs pad tokens; causal mask blocks future tokens | Which tokens exist vs which are allowed |
| chat template / prompt template / system prompt | No | Chat template formats role messages into model tokens; prompt template is your own text with slots; system prompt is one role's content | Format vs fill-in vs instruction |
| `vocab_size` / embedding size / `d_model` | No | Rows in the embedding table vs width of each row | Rows vs columns |
| token count (API) / character count / word count | No | Billing and limits use tokens; character and word counts only estimate | Count in tokens |

## Training and fine-tuning
Modules: [06](06_llm_training_and_lora/), [17](17_finetuning_alignment_and_benchmarks/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| pretraining / continued pretraining / mid-training | Nearly | Pretraining starts from scratch on huge text; continued pretraining resumes on new domain text; mid-training is a later pretraining stage on curated data | Same objective, different stage |
| fine-tuning / SFT / instruction tuning | Nearly | Fine-tuning is the umbrella; SFT is fine-tuning on labeled (prompt, response) pairs; instruction tuning is SFT specifically on instruction-following data | Umbrella, method, flavour |
| full fine-tuning / PEFT / LoRA / QLoRA / adapters | No | Full updates all weights; PEFT is the family; LoRA trains low-rank deltas; QLoRA = LoRA on a 4-bit base; adapters add small modules | Fewer trainable params |
| LoRA rank `r` / `lora_alpha` / `lora_dropout` | No | r = size of the low-rank update; alpha = scaling (effective scale = alpha/r); dropout regularizes the adapter | Size, scale, noise |
| LoRA / DoRA / rsLoRA / LoRA+ | No | Variants of LoRA: weight-decomposed, rank-stabilized scaling, different learning rates for A and B | LoRA with tweaks |
| prompt tuning / prefix tuning / P-tuning / soft prompts | Nearly | All learn continuous vectors instead of weights; prompt tuning only at input, prefix tuning at every layer | Trainable invisible prompt |
| prompt engineering / fine-tuning / RAG | No | Change the input vs change the weights vs fetch outside facts at query time | Ask better, teach, look up |
| distillation / synthetic data / self-training | No | Distillation trains a student on a teacher's outputs or logits; synthetic data is any generated data; self-training uses a model's own labels | Teacher-student vs any generated |
| catastrophic forgetting / overfitting | No | Forgetting = new training erases old skills; overfitting = memorizing training data | Old lost vs data memorized |
| model merging / ensembling | No | Merging averages or combines weights into one model; ensembling runs several models and combines outputs | One model vs many |
| batch size / micro-batch size / global batch size / gradient accumulation | No | Micro-batch fits in GPU memory; accumulation steps multiply it; global = micro x accumulation x devices | Global is what the optimizer sees |
| DP / DDP / FSDP / ZeRO / DeepSpeed | No | DP is old single-process replica; DDP replicates the model per GPU and syncs gradients; FSDP and ZeRO shard parameters/gradients/optimizer states; DeepSpeed is the library that implements ZeRO | Replicate vs shard |
| data parallel / tensor parallel / pipeline parallel / expert parallel | No | Split data, split each layer's matrices, split layers into stages, split MoE experts | What gets split |
| gradient checkpointing / activation checkpointing / rematerialization | Yes | Drop activations in forward and recompute in backward to save memory | Trade compute for memory |
| checkpoint (training) / checkpoint (gradient) | No | Saved model/optimizer state on disk vs recompute trick | Same word, two jobs |
| `bf16` / `fp16` loss scaling | No | fp16 needs loss scaling to avoid underflow; bf16 usually does not | bf16 is easier |
| LoRA merge / unmerge / adapter swap | No | Merging folds the adapter into base weights; adapter swap keeps them separate | Baked in vs plugged in |

## Alignment and RL
Module: [17](17_finetuning_alignment_and_benchmarks/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| alignment / RLHF / preference tuning | Nearly | Alignment is the goal; RLHF and preference tuning are methods toward it | Goal vs methods |
| RLHF / RLAIF / RLVR | No | Reward from human preferences; from AI-judged preferences; from verifiable checks (unit tests, math answers) | Human, AI, verifier as the judge |
| RLHF / DPO | No | RLHF trains a reward model then runs RL (PPO); DPO optimizes preferences directly with no reward model or RL loop | Two-step RL vs one-step loss |
| PPO / GRPO / REINFORCE / RLOO | No | PPO uses a learned value critic; GRPO scores a group of samples and uses group-relative advantage (no critic); REINFORCE and RLOO are simpler policy gradients with baselines | GRPO = relative to the group |
| DPO / IPO / KTO / ORPO / SimPO | No | All preference-optimization losses: DPO log-ratio vs a reference; IPO adds a regularized squared loss; KTO uses single good/bad labels (no pairs); ORPO merges SFT with an odds-ratio term (no reference model); SimPO uses length-normalized log-prob, no reference | Reference model, pairs, or neither |
| reward model / preference model / verifier / judge | Nearly | Reward and preference models score outputs (learned); a verifier checks correctness (rule-based or learned); an LLM judge scores by prompting | Scorers of different kinds |
| outcome reward / process reward (ORM / PRM) | No | Score only the final answer vs score each reasoning step | Result vs steps |
| policy / reference model / value model | No | Policy is the model being trained; reference is a frozen copy for the KL penalty; value model estimates future reward | Trainee, anchor, forecaster |
| KL penalty / KL divergence / entropy bonus | No | KL keeps the policy near the reference; entropy bonus encourages exploration | Leash vs curiosity |
| reward hacking / specification gaming / Goodhart | Nearly | The model exploits the reward without doing the intended thing | Gaming the score |
| sycophancy / hallucination | No | Agreeing with the user against the truth vs inventing unsupported content | Pleasing vs inventing |
| Constitutional AI / RLAIF | Nearly | Constitutional AI is a specific recipe (self-critique against principles) that uses AI feedback | One recipe of RLAIF |
| rejection sampling / best-of-N | Nearly | Sample many and keep the best (by reward); used at inference or to build SFT data | Pick the winner |
| on-policy / off-policy | No | Data comes from the current model vs from another or older one | Fresh vs stale samples |
| safety training / red teaming / jailbreak | No | Training to refuse harm vs adversarial testing vs prompts that bypass refusals | Defend, probe, attack |

## Inference and serving
Module: [06](06_llm_training_and_lora/), [10](10_ml_system_architecture/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| inference / prediction / serving / deployment | Nearly | Inference runs the model; serving makes it available over a network; deployment is shipping it | Run, offer, ship |
| latency / throughput / TTFT / TPOT / ITL | No | Latency = time for one request; throughput = work per second; TTFT = time to first token; TPOT = time per output token; ITL = inter-token latency (about the same as TPOT) | Wait vs volume |
| TPOT / ITL | Nearly | Time per output token vs gap between consecutive tokens | Averages vs gaps |
| temperature / top-k / top-p / min-p | No | Temperature rescales logits; top-k keeps the k best; top-p keeps the smallest set reaching probability p; min-p keeps tokens above a fraction of the top probability | Four sampling dials |
| top-p / nucleus sampling | Yes | Same method, two names | Nucleus = top-p |
| greedy / beam search / sampling | No | Greedy takes the top token; beam keeps several candidate sequences; sampling draws randomly | Deterministic vs random |
| deterministic / `temperature=0` | Nearly | Temperature 0 is greedy, but GPU batching and floating-point can still change output | Not a guarantee |
| frequency penalty / presence penalty / repetition penalty | No | Frequency scales with count of repeats; presence is a flat penalty for any repeat; repetition penalty (HF) divides repeated tokens' logits | Count vs yes/no vs multiplicative |
| max_tokens / max_new_tokens / max_completion_tokens | Nearly | Cap on generated tokens; the name varies by API (HF uses `max_new_tokens`) | Output cap |
| max_tokens / context window | No | Output cap vs total input+output limit | Per-answer vs whole room |
| streaming / batching / continuous batching | No | Streaming sends tokens as produced; static batching waits for a full batch; continuous batching adds/removes requests each step | Send early vs pack smart |
| dynamic batching / continuous batching | Nearly | Dynamic groups requests within a time window; continuous schedules at the token step (iteration-level) | Window vs per-token |
| speculative decoding / draft model / Medusa / EAGLE | Nearly | A cheap proposer guesses tokens and the big model verifies them in one pass; draft model, Medusa heads, and EAGLE are different proposers | Guess and check |
| vLLM / SGLang / TGI / TensorRT-LLM / llama.cpp / Ollama | No | Serving engines: open GPU default; RadixAttention-based; HF's (maintenance mode); NVIDIA-optimized; CPU/Metal C++ engine; a friendly runner around such an engine | Engines vs runner |
| llama.cpp / GGUF / Ollama | No | Engine / file format / user-friendly wrapper | Engine, format, app |
| ONNX / TorchScript / `torch.compile` / TensorRT | No | Portable model format; legacy graph capture; PyTorch 2 JIT compiler; NVIDIA optimizer/runtime | Format, legacy, compiler, runtime |
| model parallel serving / replicas | No | Splitting one model over GPUs vs running copies to handle more traffic | Split vs copy |
| online / batch / offline inference | No | Real-time per request vs scheduled bulk jobs (vs sometimes "offline" meaning on-device) | Now vs later |
| cold start / warm start | No | First request pays model load time vs model already in memory | Ready or not |
| autoscaling / load balancing | No | Change the number of replicas vs spread requests across them | More servers vs fair split |
| OpenAI-compatible API / OpenAI API | No | Same request shape served by other engines vs OpenAI's own service | Look-alike interface |

## Quantization
Module: [06](06_llm_training_and_lora/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| quantization / compression / pruning / distillation | No | Quantization lowers number precision; pruning removes weights; distillation trains a smaller model; "compression" is the umbrella | Shrink bits, shrink weights, shrink model |
| PTQ / QAT | No | Post-training quantization after training (no retraining); quantization-aware training simulates low precision during training | After vs during |
| GPTQ / AWQ / bitsandbytes / GGUF / FP8 | No | Two PTQ methods for 4-bit weights; a library (LLM.int8, NF4); a llama.cpp file format with K-quants; an 8-bit float format | Methods, library, file, format |
| INT8 / INT4 / FP8 / FP4 / NF4 | No | Integer vs float 8/4-bit; NF4 is the QLoRA 4-bit "normal float" tailored to weights | Different 4-bit/8-bit grids |
| MXFP4 / NVFP4 | No | Both 4-bit floats with block scales; MXFP4 is the OCP microscaling standard (32-element blocks), NVFP4 is NVIDIA's variant with smaller blocks and finer scales | Standard vs NVIDIA flavour |
| weight-only / weight-and-activation quantization | No | Only weights are low precision (memory saved) vs weights and activations (compute also faster) | W4A16 vs W8A8 |
| W4A16 / W8A8 notation | Nearly | Bits for weights then activations: W4A16 = 4-bit weights, 16-bit activations | Weights first |
| per-tensor / per-channel / per-group quantization | No | One scale for the whole tensor, per row/column, or per block of values; finer = more accurate, more overhead | Finer scale, better quality |
| quantized model / dequantization | No | Stored in low precision; dequantize converts back to compute (often inside the kernel) | Store small, compute big |
| KV-cache quantization / weight quantization | No | Quantize cache entries to save serving memory vs quantize the model weights | Cache vs weights |
| QLoRA / quantized LoRA model | Nearly | QLoRA trains LoRA on a frozen 4-bit base; afterwards you may merge or serve with the 4-bit base | Training recipe |
| calibration data / training data | No | Small sample used to measure activation ranges for PTQ vs data used to learn weights | Measure ranges, not learn |

## Evaluation and benchmarks
Module: [17](17_finetuning_alignment_and_benchmarks/), [05](05_llm_parameters/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| benchmark / eval / evaluation / test set / leaderboard | Nearly | Benchmark = fixed dataset+metric; eval = running any measurement; leaderboard = public ranking | Exam, grading, scoreboard |
| metric / score / measure | Nearly | The number computed, e.g. accuracy | Formula vs value |
| offline eval / online eval / A/B test | No | Fixed dataset before launch vs measured on live traffic vs controlled live experiment | Lab, field, controlled field |
| LLM-as-judge / human eval / reward model | No | A model grades outputs by prompt vs people grade vs a trained scoring model | Who grades |
| pass@k / pass@1 / exact match | No | Probability at least one of k samples passes tests vs first try vs string equality | Tries counted |
| BLEU / ROUGE / METEOR / BERTScore | No | N-gram precision (translation); n-gram recall (summarization); alignment with synonyms; embedding similarity | Overlap vs meaning |
| perplexity / accuracy | No | How surprised the LM is vs fraction correct | Likelihood vs correctness |
| MMLU / MMLU-Pro / GPQA / HLE | No | Broad multiple-choice knowledge (saturated); harder 10-option version; graduate-level science Q&A; very hard expert questions | Easy to hard |
| SWE-bench / SWE-bench Verified / HumanEval / MBPP | No | Real GitHub issues; human-validated subset; short function writing; basic Python problems | Real repo vs toy function |
| LMArena / Chatbot Arena | Yes | Human pairwise voting Elo ranking site (renamed) | Crowd preference |
| lm-evaluation-harness / lighteval / Inspect | No | Three eval frameworks (EleutherAI, Hugging Face, UK AISI) | Toolkits |
| RAGAS / DeepEval / TruLens | No | Libraries for RAG/LLM app metrics | App-level metric kits |
| faithfulness / groundedness / answer relevancy / correctness | No | Answer supported by the retrieved context; same idea in other tools; answers the question asked; matches ground truth | Supported, on-topic, true |
| faithfulness / factuality | No | Consistent with the given context vs true in the world | Source vs world |
| context precision / context recall | No | Share of retrieved chunks that are relevant vs share of needed facts that were retrieved | Purity vs coverage |
| hit rate / recall@k / MRR / nDCG / MAP | No | Any relevant in top-k; share of relevant found in top-k; reciprocal rank of first hit; rank-weighted gain; mean average precision | Retrieval metrics by what they reward |
| recall@k / precision@k | No | Of all relevant docs, how many in top k vs of top k, how many relevant | Coverage vs purity |
| CER / WER | No | Character error rate vs word error rate, both edit distance over reference length | Characters vs words |
| contamination / leakage / benchmark saturation | No | Test text seen in training; same as leakage; all models score near the ceiling | Cheating vs ceiling |
| hallucination / confabulation / unfaithfulness / fabrication | Nearly | Hallucination is the general term; confabulation stresses plausible invention; unfaithfulness means contradicting the given source; fabrication is made-up facts or citations | Invented vs unsupported |
| calibration / confidence | No | Whether stated confidence matches real accuracy vs the number itself | Honest confidence |
| regression test / eval suite / golden set | Nearly | Re-run checks to catch quality drops; golden set is its labeled data | Safety net |

## Prompting and agents
Modules: [07](07_prompt_engineering/), [08](08_agent_harness/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| system prompt / system message / developer message / instructions | Nearly | Top-priority role message setting behaviour; OpenAI calls the newer role "developer" | Rules before the chat |
| zero-shot / one-shot / few-shot / in-context learning | Nearly | Zero examples, one, several in the prompt; all are in-context learning (no weight change) | Examples in the prompt |
| prompt / context / context engineering | Nearly | Prompt is the instruction text; context is everything in the window (history, retrieved docs, tool output); context engineering is designing it all | Prompt is part of context |
| chain-of-thought / reasoning tokens / scratchpad / thinking | Nearly | Intermediate reasoning text; "thinking" and reasoning tokens are the model-native form | Show your work |
| CoT / ReAct / Tree of Thoughts / self-consistency | No | Linear reasoning; reasoning interleaved with actions; branching search; vote across many samples | Think, think+act, branch, vote |
| structured output / JSON mode / schema-constrained decoding / grammar | Nearly | JSON mode makes valid JSON; schema/grammar-constrained decoding enforces a specific schema token by token | Valid vs matches schema |
| tool calling / function calling | Yes | Model emits a structured request to run a named function; "function calling" is the older OpenAI name | Same mechanism |
| tool calling / MCP | No | Tool calling is the model-API feature; MCP is an open protocol that standardizes how tools/resources are exposed to any host | Ability vs plug standard |
| MCP server / MCP client / MCP host | No | Server exposes tools; client is the connector; host is the app (e.g. an IDE or chat app) running clients | Provider, plug, app |
| MCP tool / resource / prompt | No | Action the model may call; readable data; reusable template | Do, read, template |
| agent / workflow / chain | No | Chain/workflow: fixed code path of LLM calls; agent: model decides next steps in a loop | Who picks the path |
| agent / assistant / copilot / chatbot | Nearly | Chatbot talks; assistant helps; copilot works beside a human; agent acts autonomously with tools. Marketing blurs them | Autonomy ladder |
| single agent / multi-agent / orchestrator-worker | No | One loop vs several cooperating agents vs a lead that delegates subtasks | Count of loops |
| agent harness / scaffold / framework / runtime | Nearly | Code around the model: loop, tools, memory, guardrails | Everything but the model |
| LangChain / LangGraph / LlamaIndex / DSPy | No | Component library; stateful graph runtime; data/RAG framework; programmatic prompt optimizer | Different jobs |
| short-term memory / long-term memory / context | No | History inside the window vs persisted store retrieved later | Window vs database |
| guardrails / moderation / validation | No | Policy and safety checks on inputs/outputs vs content classification vs schema checks | Safety, content, shape |
| prompt injection / jailbreak | No | Untrusted content hijacks the model's instructions vs the user bypasses safety policy | Attack from data vs from user |
| temperature / top-p in prompting | No | Sampling parameters, not prompt text | Settings, not words |
| human-in-the-loop / human-on-the-loop | No | Human approves each step vs human supervises and can intervene | Approve vs monitor |

## RAG and retrieval
Modules: [14](14_rag_and_retrieval/), [19](19_chunking_deep_dive/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| RAG / retrieval-augmented generation / grounding | Nearly | RAG is the technique; grounding is the goal of tying answers to sources | Method vs aim |
| RAG / fine-tuning / long context | No | Look up facts at query time vs bake behaviour into weights vs paste everything into the window | Lookup, learn, paste |
| RAG / semantic search | No | Semantic search only retrieves; RAG then generates an answer from the results | Search plus answer |
| dense retrieval / semantic search / vector search / embedding search | Nearly | Retrieve by embedding similarity | Meaning match |
| sparse retrieval / lexical search / keyword search / BM25 | Nearly | Retrieve by term overlap; BM25 is the standard scoring function (TF-IDF family) | Word match |
| dense vs sparse retrieval | No | Dense = learned vectors that capture meaning; sparse = term-based (BM25, or learned sparse like SPLADE) | Meaning vs exact words |
| BM25 / TF-IDF | No | BM25 adds term-frequency saturation and length normalization on top of the TF-IDF idea | Better TF-IDF |
| SPLADE / BM25 | No | SPLADE is learned sparse (neural term expansion) vs BM25's pure statistics | Neural vs statistical sparse |
| hybrid search / hybrid retrieval | Yes | Combine dense and sparse results | Both signals |
| RRF / weighted score fusion | No | Reciprocal Rank Fusion merges by ranks; weighted fusion merges normalized scores | Rank vs score |
| retriever / ranker / re-ranker | No | Retriever fetches candidates fast; re-ranker re-orders the shortlist more accurately | Fetch wide, sort narrow |
| re-ranker / cross-encoder | Nearly | A cross-encoder (query and doc in one pass) is the most common re-ranker; rerankers can also be LLM-based or late-interaction | Cross-encoder is a kind of re-ranker |
| bi-encoder / dual encoder / two-tower | Yes | Query and doc encoded separately; compare by dot product or cosine, so doc vectors can be precomputed | Two towers, precompute |
| bi-encoder / cross-encoder | No | Bi: separate encodings, fast, less precise; cross: joint encoding, slow, more precise | Fast first pass vs slow second pass |
| ColBERT / late interaction / multi-vector | Nearly | One vector per token and sum of best-match similarities (MaxSim) instead of one vector per doc | Many vectors per doc |
| top-k / top-p (retrieval vs sampling) | No | Number of chunks retrieved vs probability mass in sampling | Same word, different stage |
| query rewriting / query expansion / HyDE / multi-query | No | Rephrase the query; add terms; embed a hypothetical answer instead of the query; run several variants | Improve what you search for |
| HyDE / query expansion | No | HyDE generates a fake answer document to embed; expansion adds related terms | Hypothetical doc vs extra words |
| parent-document / small-to-big / sentence-window retrieval | Nearly | Search small pieces, then return the larger surrounding text | Match small, show big |
| contextual retrieval / contextual chunk headers | Nearly | Prepend an LLM-written (or metadata) context blurb to each chunk before embedding | Give chunks their backstory |
| GraphRAG / knowledge graph RAG / vector RAG | No | Retrieve via entity-relationship graph and summaries vs via vector similarity | Links vs similarity |
| agentic RAG / naive RAG / advanced RAG / modular RAG | No | Single retrieve-then-generate; add pre/post-processing; swappable modules; an agent decides when and what to retrieve | Fixed pipeline vs decision-making |
| ANN / exact (brute-force) / kNN search | No | Approximate nearest neighbours trade a little recall for speed; exact scans everything | Fast guess vs full scan |
| metadata filter / pre-filter / post-filter | No | Restrict by fields; pre-filter before vector search vs post-filter after (may return too few) | Filter first or last |
| citation / attribution / source linking | Nearly | Pointing the answer to the supporting passage | Show the receipt |
| reranking / re-scoring / MMR | No | MMR re-selects for relevance plus diversity; reranking models re-score relevance | Diversity vs precision |
| lost in the middle / context rot / context overflow | No | Models miss facts in the middle of long context; accuracy degrades as context grows; input exceeds the window | Position, length, limit |
| hallucination (RAG) / retrieval failure | No | Model invents despite good context vs the right chunk was never retrieved | Generator vs retriever fault |

## Chunking
Module: [19](19_chunking_deep_dive/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| chunk / passage / node / split / segment / document piece | Nearly | A piece of source text that gets embedded; LlamaIndex calls it a node (with metadata and links), LangChain a split or Document, search folks a passage | Same piece, different libraries |
| chunk / document / page | No | Chunk is a sub-document unit you choose; document is the source file; page is a physical PDF page | Piece, whole, sheet |
| chunk size / chunk length / max_tokens | Nearly | Size of each chunk, in characters or tokens; check which unit | Characters vs tokens trap |
| chunk overlap / stride / window overlap | Nearly | Repeated text at chunk boundaries; stride = chunk size minus overlap | Overlap is the shared part |
| fixed-size / recursive / semantic / structure-aware / agentic chunking | No | Cut by length; cut by separators in priority order; cut where embedding similarity drops; cut by headings/tables; LLM decides boundaries | Dumb to smart, cheap to costly |
| `RecursiveCharacterTextSplitter` / `CharacterTextSplitter` | No | Recursive tries a list of separators in order; Character splits on one separator | One rule vs fallback ladder |
| late chunking / chunking then embedding | No | Late chunking embeds the whole long text first and pools per chunk; normal chunking cuts first | Context before cutting |
| late chunking / late interaction | No | Pooling token embeddings per chunk vs ColBERT-style per-token matching | Different "late" |
| hierarchical chunking / parent-child chunking | Nearly | Chunks at several levels; retrieve children, return parents | Tree of chunks |
| proposition chunking / sentence chunking | No | Atomic self-contained facts vs sentence boundaries | Facts vs sentences |
| tokenizer-aware chunk size / embedding model max length | No | Chunk limit you set vs the model's input limit; exceeding it silently truncates | Don't exceed the model |
| chunk metadata / payload / attributes | Nearly | Fields stored with the chunk (source, page, section); vector DBs call it payload | Chunk's label |

## Embeddings and vector DBs
Module: [14](14_rag_and_retrieval/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| embedding / vector / representation | Nearly | Embedding is the vector produced for text; "vector" is just the math object; "representation" the concept | Product, container, idea |
| embedding model / encoder / sentence encoder | Nearly | Model that turns text into a vector | Text to numbers |
| embedding / one-hot / TF-IDF vector | No | Dense learned meaning vs sparse indicator vs sparse weighted term counts | Dense meaning vs sparse words |
| cosine similarity / dot product / Euclidean (L2) / inner product | No | Cosine ignores length; dot product includes length; L2 is distance; for normalized vectors cosine = dot product and ranks the same as L2 | Normalize and they agree |
| similarity / distance | No | Higher is closer vs lower is closer | Opposite directions |
| dimension / `dim` / embedding size | Yes | Length of the vector | Numbers per vector |
| Matryoshka embeddings / truncated embeddings | Nearly | Trained so the first N dimensions still work; truncating them saves storage | Nested dolls |
| vector database / vector index / vector store / vector library | No | DB stores, filters, persists; index is the search structure (HNSW); store is the LangChain word for a DB wrapper; library (FAISS) has no server or persistence | Library < index < database |
| HNSW / IVF / PQ / flat / DiskANN | No | Graph index; clustered lists; product quantization (compression); exhaustive scan; disk-based graph | Index families |
| FAISS / Qdrant / Milvus / Weaviate / pgvector / Chroma / LanceDB | No | A library, then six databases with different deployment styles | Library vs databases |
| collection / index / table / namespace / class | Nearly | The container for vectors: Qdrant collection, Pinecone index/namespace, pgvector table, Weaviate class/collection | Container, many names |
| payload / metadata / attributes | Yes | Extra fields stored beside the vector | Qdrant says payload |
| ef_search / `ef_construction` / M | No | HNSW knobs: query beam width, build beam width, links per node | Query, build, degree |
| recall (ANN) / recall (RAG retrieval) | No | Share of true nearest neighbours the index found vs share of relevant docs retrieved | Index recall vs search recall |
| upsert / insert / update | No | Upsert inserts or replaces by ID | Insert-or-replace |
| sentence-transformers / transformers / HF Hub | No | Embedding library / model library / model host | Library, library, host |
| text-embedding / multimodal embedding (CLIP) | No | Text only vs shared text-image space | One modality vs shared |

## OCR and document parsing
Module: [18](18_ocr_and_document_parsing/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| OCR / ICR / HTR / OMR | No | Printed text; handwritten characters (older term); handwritten text recognition (modern); mark/checkbox reading | Print, hand, hand, tick |
| ICR / HTR | Nearly | Both read handwriting; HTR is the research name for line/word-level handwritten text | Same goal |
| OCR / document AI / IDP / document parsing / document understanding | No | OCR reads characters; document parsing adds layout and structure; document AI is the broad cloud product term; IDP (intelligent document processing) is the business workflow (ingest, extract, validate); document understanding is the model capability | Read, structure, product, workflow |
| text-layer PDF / scanned PDF / born-digital | No | Born-digital PDF has real text to extract; scanned PDF is an image and needs OCR | Extract vs recognize |
| PDF text extraction / OCR | No | Reading embedded text vs recognizing pixels | pdfplumber vs Tesseract |
| layout analysis / document layout detection / page segmentation | Nearly | Finding regions such as titles, tables, figures, reading order | Where things are |
| table detection / table structure recognition (TSR) / table extraction | No | Find the table; recover rows, columns, spans; the whole job to usable data | Locate, structure, extract |
| KIE / NER / form extraction / key-value extraction | No | Key information extraction from documents (uses layout); NER tags entities in text; form extraction reads fields; key-value pairs are one output | Doc-aware vs text-only |
| CER / WER | No | Character error rate vs word error rate, edit distance divided by reference length | Letters vs words |
| edit distance / Levenshtein | Yes | Minimum insert, delete, substitute operations | One metric, two names |
| TEDS / OmniDocBench / DocVQA | No | Table similarity metric; broad document-parsing benchmark; question answering over document images | Table score, parse benchmark, QA |
| detection / recognition (OCR stages) | No | Find text boxes vs read the text in each box | Where vs what |
| pipeline OCR / end-to-end VLM OCR | No | Separate detection+recognition+layout stages vs one vision-language model produces Markdown/JSON | Many steps vs one model |
| Tesseract / PaddleOCR / EasyOCR | No | Classic engine; deep-learning toolkit with VL models; PyTorch lightweight OCR | Different engines |
| Docling / Unstructured / MinerU / Marker | No | Document converters producing structured output: IBM open library, ETL toolkit, PDF-to-Markdown with VLM backend, PDF-to-Markdown tool | Parsers, not recognizers |
| olmOCR / DeepSeek-OCR / dots.ocr / GLM-OCR | No | Different open VLM OCR models; compare on your documents | Models, not tools |
| Markdown output / JSON output / HTML table output | No | Parser output formats; Markdown loses complex tables, HTML keeps spans | Format affects tables |
| DPI / resolution / scale | Nearly | Image sharpness; 200-300 DPI is typical for scans | Dots per inch |
| deskew / binarize / denoise | No | Straighten tilt; make black-white; clean speckles (preprocessing) | Pre-clean the image |

## APIs and deployment
Modules: [09](09_api_development/), [10](10_ml_system_architecture/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| API / endpoint / route / URL path | Nearly | API is the interface; endpoint/route is one URL+method pair | One door of the building |
| REST / HTTP API / web service | Nearly | HTTP-based API; REST is a style using resources and verbs | REST is a style |
| REST / gRPC / GraphQL / WebSocket / SSE | No | Resource HTTP; binary RPC with protobuf; client-shaped queries; two-way persistent socket; one-way server stream (used for LLM token streaming) | SSE streams tokens |
| GET / POST / PUT / PATCH / DELETE | No | Read; create or run; replace; partial update; remove | CRUD verbs |
| PUT / PATCH | No | Replace the whole resource vs change part of it | Whole vs part |
| 401 / 403 | No | Not authenticated vs authenticated but not allowed | Who are you vs not you |
| 400 / 422 | No | Malformed request vs well-formed but invalid (FastAPI uses 422 for validation) | Broken vs invalid |
| 500 / 502 / 503 / 504 | No | App crash; bad upstream reply; unavailable/overloaded; upstream timeout | Whose fault |
| authentication / authorization | No | Who you are vs what you may do | Login vs permission |
| API key / OAuth / JWT | No | Static secret; delegated-access flow; signed token carrying claims | Key, flow, token |
| WSGI / ASGI | No | Sync server-app interface (Flask, Django classic) vs async interface supporting WebSockets and streaming (FastAPI, Starlette) | A for async |
| Flask / FastAPI / Django | No | Sync micro-framework; async typed framework with auto docs; full-stack batteries-included | Small, modern, big |
| Uvicorn / Gunicorn / Hypercorn | No | ASGI server; WSGI process manager (can run Uvicorn workers); another ASGI server | Server vs manager |
| Pydantic model / schema / dataclass / `TypedDict` | No | Pydantic validates; schema describes (JSON Schema); dataclass stores; TypedDict is typing only | Validate, describe, store, hint |
| request body / query param / path param / header | No | JSON payload; `?x=1`; `/items/{id}`; metadata | Where the data rides |
| sync endpoint / async endpoint | No | `def` runs in a threadpool; `async def` runs on the event loop and must not block | Don't block the loop |
| middleware / dependency injection / decorator | No | Wraps every request; supplies shared resources to endpoints (`Depends`); wraps one function | Global vs per-route vs per-function |
| rate limiting / throttling / backpressure | Nearly | Cap request rate; throttling slows; backpressure signals upstream to slow down | Limit, slow, signal |
| timeout / retry / backoff / circuit breaker | No | Give up after a time; try again; wait longer each time; stop calling a failing service | Resilience toolbox |
| idempotent / safe (HTTP) | No | Repeating gives the same effect vs does not change state at all | GET is safe, PUT is idempotent |
| streamlit / gradio | No | Script-driven dashboards vs ML demo UIs with input/output components | Dashboards vs demos |
| model server / API server / inference server | Nearly | A process exposing a model over HTTP | Same pattern |
| health check / readiness / liveness | No | Is it up; is it ready for traffic; should it be restarted | Alive vs ready |

## Docker
Module: [11](11_docker/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| image / container | No | Image is the read-only template; container is a running instance of it | Class vs object |
| Dockerfile / image / docker-compose.yml | No | Build recipe; built artifact; multi-container run config | Recipe, dish, menu |
| container / virtual machine | No | Containers share the host kernel; VMs run their own OS | Light vs heavy |
| Docker / containerd / runc / OCI | No | Tool; runtime daemon; low-level runtime; open image/runtime standards | Tool to standard |
| Docker / Podman | Nearly | Podman is daemonless and rootless-friendly with a compatible CLI | Docker without the daemon |
| volume / bind mount / tmpfs | No | Docker-managed persistent storage; host folder mapped in; RAM only | Managed, mapped, memory |
| `COPY` / `ADD` | No | COPY copies files; ADD also unpacks archives and fetches URLs (prefer COPY) | Prefer COPY |
| `CMD` / `ENTRYPOINT` | No | CMD is default arguments (overridable); ENTRYPOINT is the fixed executable | Default vs fixed |
| `RUN` / `CMD` | No | Executes at build time vs at container start | Build vs run |
| `EXPOSE` / `-p` publish | No | EXPOSE documents a port; `-p` actually maps it to the host | Docs vs wiring |
| build context / `.dockerignore` | No | Folder sent to the builder vs list of files to exclude | What to send vs what to skip |
| layer / cache | Nearly | Each instruction makes a layer; unchanged layers are reused from cache | Order matters |
| multi-stage build / single-stage | No | Build in one stage, copy only artifacts into a slim final stage | Leave the tools behind |
| tag / digest / `latest` | No | Mutable label; immutable content hash; just a default tag name, not "newest" | Pin by digest |
| registry / repository / Docker Hub | No | Server storing images; named image collection; the public registry | Warehouse, shelf, big warehouse |
| `docker compose` / `docker-compose` | Nearly | v2 plugin (`docker compose`) replaced the old v1 Python binary | Use the space |
| `docker run` / `docker exec` / `docker start` | No | Create and start new container; run a command in a running one; restart a stopped one | New, inside, resume |
| `stop` / `kill` | No | SIGTERM then SIGKILL after grace period vs immediate SIGKILL | Polite vs force |
| CPU image / GPU image / NVIDIA Container Toolkit | No | Plain image; image with CUDA libs; host runtime hook that exposes GPUs | Libs inside, hook outside |
| slim / alpine / distroless | No | Debian-slim base; musl-based tiny base (can break Python wheels); no shell, minimal runtime | Size vs compatibility |

## MLOps
Module: [10](10_ml_system_architecture/), [13](13_capstone_project/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| MLOps / LLMOps / DevOps / AIOps | No | DevOps for software; MLOps adds data/model lifecycle; LLMOps adds prompts, evals, guardrails; AIOps is AI for IT operations (reversed meaning) | AIOps is not MLOps |
| experiment tracking / model registry / artifact store | No | Log runs and metrics; versioned approved models; storage for files | Log, shelf, disk |
| MLflow / Weights & Biases / DVC | No | Tracking + registry; hosted tracking; data and pipeline versioning with git | Different layers |
| CI / CD / continuous training (CT) | No | Test on every change; deploy automatically; retrain automatically | Test, ship, retrain |
| data drift / concept drift / model drift / prompt drift | No | Input distribution changes; input-to-output relationship changes; performance degrades; provider changes a model under the same name | What shifted |
| feature store / vector store | No | Serves ML features online/offline vs stores embeddings | Tabular features vs vectors |
| batch / streaming pipeline | No | Process in chunks on a schedule vs per event | Scheduled vs live |
| canary / blue-green / shadow deployment | No | Small share of traffic first; two environments switched at once; copy of traffic with no user impact | Few, swap, mirror |
| rollback / roll forward | No | Revert to old version vs ship a fix | Back vs on |
| logging / metrics / tracing / observability | No | Events; numbers over time; per-request path; the ability to ask questions from all three | Three pillars |
| trace / span | No | Whole request path vs one step inside it | Journey vs leg |
| OpenTelemetry / Langfuse / LangSmith / Phoenix | No | Open standard and SDK; LLM tracing products | Standard vs products |
| SLA / SLO / SLI | No | Contract with consequences; internal target; the measured number | Promise, goal, reading |
| p50 / p95 / p99 latency | No | Median and tail percentiles | Tail hurts users |
| model card / data sheet / system card | No | Model documentation; dataset documentation; documentation of a whole system with safeguards | Model, data, system |
| reproducibility / determinism | No | Same results given recorded environment and seeds vs bit-identical output | Recorded vs identical |

## LLD and system design
Modules: [20](20_llm_system_lld/), [10](10_ml_system_architecture/)

| Terms side by side | Same thing? | Exact difference in one line | Simple way to remember |
|---|---|---|---|
| LLD / HLD / system design | No | LLD = classes, interfaces, code structure; HLD = components, services, data flow; "system design" usually means HLD (scale, storage, queues) | Rooms vs floor plan vs city |
| LLD / OOD / OOP | Nearly | LLD interviews test object-oriented design (using OOP) | Design with objects |
| class diagram / sequence diagram / ER diagram / architecture diagram | No | Static structure; time-ordered calls; data entities; service boxes | Shape, order, data, services |
| SOLID / design patterns | No | Five design principles vs reusable solutions (Factory, Strategy, ...) | Principles vs recipes |
| interface / abstract class / protocol | Nearly | Contract of methods; abstract class may hold shared code; Python Protocol is structural typing | Contract kinds |
| inheritance / composition | No | Is-a reuse vs has-a reuse; prefer composition | Is vs has |
| Strategy / Factory / Adapter / Facade / Decorator (pattern) | No | Swap algorithms; create objects; convert interfaces; simplify a subsystem; add behaviour by wrapping | Used to swap LLM providers |
| dependency injection / dependency inversion | No | Passing dependencies in vs depending on abstractions | Mechanism vs principle |
| cache / CDN / queue / message broker | No | Fast copy of data; edge cache; buffer of work; the service running queues (Kafka, RabbitMQ) | Remember, deliver, wait, route |
| semantic cache / exact cache / prompt cache | No | Reuse an answer for similar queries (embedding match); same exact request; provider reuse of a prefix's KV | Answer reuse vs compute reuse |
| gateway / proxy / load balancer / router | Nearly | API gateway handles auth/limits; LLM gateway/router picks model/provider; load balancer spreads traffic | Front desk roles |
| synchronous / asynchronous / event-driven | No | Wait for reply; respond later; react to events | Wait or not |
| horizontal / vertical scaling | No | More machines vs bigger machine | Out vs up |
| stateless / stateful | No | No memory between requests vs keeps session state | Stateless scales easily |
| consistency / availability / partition tolerance (CAP) | No | Three properties; you cannot guarantee all under partitions | Pick two when split |
| SQL / NoSQL / vector DB | No | Relational tables; flexible stores; similarity search | Rows, documents, vectors |
| token bucket / leaky bucket / sliding window | No | Rate-limit algorithms: burst-friendly; smooth output; counted window | Limit shapes |
| monolith / microservices | No | One deployable vs many small services | One vs many |
| LLM gateway / orchestrator / agent runtime | No | Routes/limits calls; coordinates multi-step flows; runs the agent loop | Route, coordinate, loop |

---

## Alphabetical index

1018 terms. Each links to the topic section holding the comparison row.


**#**

- \*\*kwargs — [Python and tooling](#python-and-tooling)
- \*args — [Python and tooling](#python-and-tooling)
- -p publish — [Docker](#docker)
- .cpu() — [Neural networks](#neural-networks)
- .dockerignore — [Docker](#docker)
- .item() — [Neural networks](#neural-networks)
- 400 — [APIs and deployment](#apis-and-deployment)
- 401 — [APIs and deployment](#apis-and-deployment)
- 403 — [APIs and deployment](#apis-and-deployment)
- 422 — [APIs and deployment](#apis-and-deployment)
- 500 — [APIs and deployment](#apis-and-deployment)
- 502 — [APIs and deployment](#apis-and-deployment)
- 503 — [APIs and deployment](#apis-and-deployment)
- 504 — [APIs and deployment](#apis-and-deployment)
- == — [Python and tooling](#python-and-tooling)
- __repr__ — [Python and tooling](#python-and-tooling)
- __str__ — [Python and tooling](#python-and-tooling)

**A**

- A/B test — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- absolute path — [Python and tooling](#python-and-tooling)
- abstract class — [LLD and system design](#lld-and-system-design)
- accuracy — [Classic ML](#classic-ml), [Evaluation and benchmarks](#evaluation-and-benchmarks)
- activation checkpointing — [Training and fine-tuning](#training-and-fine-tuning)
- activation function — [Neural networks](#neural-networks)
- activations — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- active params — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- Adam — [Neural networks](#neural-networks)
- AdamW — [Neural networks](#neural-networks)
- Adapter — [LLD and system design](#lld-and-system-design)
- adapter swap — [Training and fine-tuning](#training-and-fine-tuning)
- adapters — [Training and fine-tuning](#training-and-fine-tuning)
- ADD — [Docker](#docker)
- advanced RAG — [RAG and retrieval](#rag-and-retrieval)
- agent — [Prompting and agents](#prompting-and-agents)
- agent harness — [Prompting and agents](#prompting-and-agents)
- agent runtime — [LLD and system design](#lld-and-system-design)
- agentic chunking — [Chunking](#chunking)
- agentic RAG — [RAG and retrieval](#rag-and-retrieval)
- AIOps — [MLOps](#mlops)
- ALiBi positions — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- aligned model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- alignment — [Alignment and RL](#alignment-and-rl)
- alpine — [Docker](#docker)
- AMP — [Neural networks](#neural-networks)
- ANN — [RAG and retrieval](#rag-and-retrieval)
- answer relevancy — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- API — [APIs and deployment](#apis-and-deployment)
- API key — [APIs and deployment](#apis-and-deployment)
- API server — [APIs and deployment](#apis-and-deployment)
- architecture diagram — [LLD and system design](#lld-and-system-design)
- argument (function) — [Python and tooling](#python-and-tooling)
- array (NumPy) — [Python and tooling](#python-and-tooling)
- artifact store — [MLOps](#mlops)
- ASGI — [APIs and deployment](#apis-and-deployment)
- assistant — [Prompting and agents](#prompting-and-agents)
- async (Python) — [Python and tooling](#python-and-tooling)
- async endpoint — [APIs and deployment](#apis-and-deployment)
- asynchronous — [LLD and system design](#lld-and-system-design)
- attention head — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- attention mask — [Tokenization](#tokenization)
- attribute — [Classic ML](#classic-ml)
- attributes — [Chunking](#chunking), [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- attribution — [RAG and retrieval](#rag-and-retrieval)
- authentication — [APIs and deployment](#apis-and-deployment)
- authorization — [APIs and deployment](#apis-and-deployment)
- autoencoder — [Neural networks](#neural-networks)
- autograd — [Neural networks](#neural-networks)
- automatic differentiation — [Neural networks](#neural-networks)
- autoregressive mask — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- autoscaling — [Inference and serving](#inference-and-serving)
- availability — [LLD and system design](#lld-and-system-design)
- AWQ — [Quantization](#quantization)

**B**

- backoff — [APIs and deployment](#apis-and-deployment)
- backpressure — [APIs and deployment](#apis-and-deployment)
- backpropagation — [Neural networks](#neural-networks)
- bagging — [Classic ML](#classic-ml)
- base model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- batch — [Classic ML](#classic-ml), [Inference and serving](#inference-and-serving), [MLOps](#mlops), [Neural networks](#neural-networks)
- batch GD — [Classic ML](#classic-ml)
- batch size — [Training and fine-tuning](#training-and-fine-tuning)
- batching — [Inference and serving](#inference-and-serving)
- BatchNorm — [Neural networks](#neural-networks)
- beam search — [Inference and serving](#inference-and-serving)
- benchmark — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- benchmark saturation — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- BERTScore — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- best-of-N — [Alignment and RL](#alignment-and-rl)
- bf16 — [Training and fine-tuning](#training-and-fine-tuning)
- bfloat16 — [Neural networks](#neural-networks)
- bi-encoder — [RAG and retrieval](#rag-and-retrieval)
- bias (error) — [Classic ML](#classic-ml)
- bias (fairness) — [Classic ML](#classic-ml)
- bias (intercept) — [Classic ML](#classic-ml)
- bias-variance — [Classic ML](#classic-ml)
- bidirectional attention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- binarize — [OCR and document parsing](#ocr-and-document-parsing)
- bind mount — [Docker](#docker)
- bitsandbytes — [Quantization](#quantization)
- BLEU — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- blue-green — [MLOps](#mlops)
- BM25 — [RAG and retrieval](#rag-and-retrieval)
- boosting — [Classic ML](#classic-ml)
- born-digital — [OCR and document parsing](#ocr-and-document-parsing)
- BOS — [Tokenization](#tokenization)
- BPE — [Tokenization](#tokenization)
- build context — [Docker](#docker)
- byte-level BPE — [Tokenization](#tokenization)

**C**

- cache — [Docker](#docker), [LLD and system design](#lld-and-system-design)
- calibration — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- calibration data — [Quantization](#quantization)
- canary — [MLOps](#mlops)
- catastrophic forgetting — [Training and fine-tuning](#training-and-fine-tuning)
- causal attention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- causal mask — [Tokenization](#tokenization)
- CD — [MLOps](#mlops)
- CDN — [LLD and system design](#lld-and-system-design)
- CER — [Evaluation and benchmarks](#evaluation-and-benchmarks), [OCR and document parsing](#ocr-and-document-parsing)
- chain — [Prompting and agents](#prompting-and-agents)
- chain-of-thought — [Prompting and agents](#prompting-and-agents)
- chain-of-thought model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- character — [Tokenization](#tokenization)
- character count — [Tokenization](#tokenization)
- CharacterTextSplitter — [Chunking](#chunking)
- chat model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- chat template — [Tokenization](#tokenization)
- chatbot — [Prompting and agents](#prompting-and-agents)
- Chatbot Arena — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- checkpoint (gradient) — [Training and fine-tuning](#training-and-fine-tuning)
- checkpoint (training) — [Training and fine-tuning](#training-and-fine-tuning)
- Chinchilla — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- Chroma — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- chunk — [Chunking](#chunking)
- chunk length — [Chunking](#chunking)
- chunk metadata — [Chunking](#chunking)
- chunk overlap — [Chunking](#chunking)
- chunk size — [Chunking](#chunking)
- chunking then embedding — [Chunking](#chunking)
- CI — [MLOps](#mlops)
- circuit breaker — [APIs and deployment](#apis-and-deployment)
- citation — [RAG and retrieval](#rag-and-retrieval)
- class — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- class diagram — [LLD and system design](#lld-and-system-design)
- classification — [Classic ML](#classic-ml)
- clustering — [Classic ML](#classic-ml)
- CMD — [Docker](#docker)
- CNN — [Neural networks](#neural-networks)
- ColBERT — [RAG and retrieval](#rag-and-retrieval)
- cold start — [Inference and serving](#inference-and-serving)
- collection — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- completion tokens — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- composition — [LLD and system design](#lld-and-system-design)
- compression — [Quantization](#quantization)
- concept drift — [MLOps](#mlops)
- conda env — [Python and tooling](#python-and-tooling)
- conda install — [Python and tooling](#python-and-tooling)
- confabulation — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- confidence — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- consistency — [LLD and system design](#lld-and-system-design)
- Constitutional AI — [Alignment and RL](#alignment-and-rl)
- container — [Docker](#docker)
- containerd — [Docker](#docker)
- contamination — [Classic ML](#classic-ml), [Evaluation and benchmarks](#evaluation-and-benchmarks)
- context — [Prompting and agents](#prompting-and-agents)
- context engineering — [Prompting and agents](#prompting-and-agents)
- context length — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- context manager — [Python and tooling](#python-and-tooling)
- context overflow — [RAG and retrieval](#rag-and-retrieval)
- context precision — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- context recall — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- context rot — [RAG and retrieval](#rag-and-retrieval)
- context window — [Inference and serving](#inference-and-serving), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- contextual chunk headers — [RAG and retrieval](#rag-and-retrieval)
- contextual retrieval — [RAG and retrieval](#rag-and-retrieval)
- continued pretraining — [Training and fine-tuning](#training-and-fine-tuning)
- continuous batching — [Inference and serving](#inference-and-serving)
- continuous training (CT) — [MLOps](#mlops)
- control tokens — [Tokenization](#tokenization)
- copilot — [Prompting and agents](#prompting-and-agents)
- COPY — [Docker](#docker)
- correctness — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- cosine decay — [Neural networks](#neural-networks)
- cosine similarity — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- cost function — [Classic ML](#classic-ml)
- CoT — [Prompting and agents](#prompting-and-agents)
- CPU image — [Docker](#docker)
- cross-attention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- cross-encoder — [RAG and retrieval](#rag-and-retrieval)
- cross-entropy — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- cross-validation — [Classic ML](#classic-ml)
- cwd — [Python and tooling](#python-and-tooling)

**D**

- d_ff — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- d_model — [Tokenization](#tokenization), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- data drift — [MLOps](#mlops)
- data leakage — [Classic ML](#classic-ml)
- data parallel — [Training and fine-tuning](#training-and-fine-tuning)
- data point — [Classic ML](#classic-ml)
- data sheet — [MLOps](#mlops)
- dataclass — [APIs and deployment](#apis-and-deployment), [Python and tooling](#python-and-tooling)
- DDP — [Training and fine-tuning](#training-and-fine-tuning)
- decision tree — [Classic ML](#classic-ml)
- decode — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- decode tokens — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- decoder — [Neural networks](#neural-networks)
- decoder-only — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- decorator — [APIs and deployment](#apis-and-deployment), [Python and tooling](#python-and-tooling)
- Decorator (pattern) — [LLD and system design](#lld-and-system-design)
- deep copy — [Python and tooling](#python-and-tooling)
- DeepEval — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- DeepSeek-OCR — [OCR and document parsing](#ocr-and-document-parsing)
- DeepSpeed — [Training and fine-tuning](#training-and-fine-tuning)
- DELETE — [APIs and deployment](#apis-and-deployment)
- denoise — [OCR and document parsing](#ocr-and-document-parsing)
- dense — [RAG and retrieval](#rag-and-retrieval)
- dense layer — [Neural networks](#neural-networks)
- dense model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- dense network — [Neural networks](#neural-networks)
- dense retrieval — [RAG and retrieval](#rag-and-retrieval)
- dependency injection — [APIs and deployment](#apis-and-deployment), [LLD and system design](#lld-and-system-design)
- dependency inversion — [LLD and system design](#lld-and-system-design)
- dependent variable — [Classic ML](#classic-ml)
- deployment — [Inference and serving](#inference-and-serving)
- dequantization — [Quantization](#quantization)
- derivative — [Neural networks](#neural-networks)
- design patterns — [LLD and system design](#lld-and-system-design)
- deskew — [OCR and document parsing](#ocr-and-document-parsing)
- detach() — [Neural networks](#neural-networks)
- detection — [OCR and document parsing](#ocr-and-document-parsing)
- determinism — [MLOps](#mlops)
- deterministic — [Inference and serving](#inference-and-serving)
- developer message — [Prompting and agents](#prompting-and-agents)
- DevOps — [MLOps](#mlops)
- dict — [Python and tooling](#python-and-tooling)
- diffusion — [Neural networks](#neural-networks)
- digest — [Docker](#docker)
- dim — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- dimension — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- DiskANN — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- distance — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- distillation — [Quantization](#quantization), [Training and fine-tuning](#training-and-fine-tuning)
- distroless — [Docker](#docker)
- Django — [APIs and deployment](#apis-and-deployment)
- Docker — [Docker](#docker)
- docker compose — [Docker](#docker)
- docker exec — [Docker](#docker)
- Docker Hub — [Docker](#docker)
- docker run — [Docker](#docker)
- docker start — [Docker](#docker)
- docker-compose — [Docker](#docker)
- docker-compose.yml — [Docker](#docker)
- Dockerfile — [Docker](#docker)
- Docling — [OCR and document parsing](#ocr-and-document-parsing)
- document — [Chunking](#chunking)
- document AI — [OCR and document parsing](#ocr-and-document-parsing)
- document layout detection — [OCR and document parsing](#ocr-and-document-parsing)
- document parsing — [OCR and document parsing](#ocr-and-document-parsing)
- document piece — [Chunking](#chunking)
- document understanding — [OCR and document parsing](#ocr-and-document-parsing)
- DocVQA — [OCR and document parsing](#ocr-and-document-parsing)
- DoRA — [Training and fine-tuning](#training-and-fine-tuning)
- dot product — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- dots.ocr — [OCR and document parsing](#ocr-and-document-parsing)
- DP — [Training and fine-tuning](#training-and-fine-tuning)
- DPI — [OCR and document parsing](#ocr-and-document-parsing)
- DPO — [Alignment and RL](#alignment-and-rl)
- draft model — [Inference and serving](#inference-and-serving)
- dropout — [Neural networks](#neural-networks)
- DSPy — [Prompting and agents](#prompting-and-agents)
- dual encoder — [RAG and retrieval](#rag-and-retrieval)
- DVC — [MLOps](#mlops)
- dynamic batching — [Inference and serving](#inference-and-serving)

**E**

- EAGLE — [Inference and serving](#inference-and-serving)
- early stopping — [Neural networks](#neural-networks)
- EasyOCR — [OCR and document parsing](#ocr-and-document-parsing)
- edit distance — [OCR and document parsing](#ocr-and-document-parsing)
- ef_construction — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- ef_search — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- embedding — [Embeddings and vector DBs](#embeddings-and-vector-dbs), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- embedding model — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- embedding model max length — [Chunking](#chunking)
- embedding search — [RAG and retrieval](#rag-and-retrieval)
- embedding size — [Embeddings and vector DBs](#embeddings-and-vector-dbs), [Tokenization](#tokenization)
- encoder — [Embeddings and vector DBs](#embeddings-and-vector-dbs), [Neural networks](#neural-networks)
- encoder-decoder — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- encoder-only — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- end-to-end VLM OCR — [OCR and document parsing](#ocr-and-document-parsing)
- endpoint — [APIs and deployment](#apis-and-deployment)
- ensembling — [Training and fine-tuning](#training-and-fine-tuning)
- entropy bonus — [Alignment and RL](#alignment-and-rl)
- ENTRYPOINT — [Docker](#docker)
- EOS — [Tokenization](#tokenization)
- epoch — [Classic ML](#classic-ml), [Neural networks](#neural-networks)
- ER diagram — [LLD and system design](#lld-and-system-design)
- error function — [Classic ML](#classic-ml)
- Euclidean (L2) — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- eval — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- eval suite — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- evaluation — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- event-driven — [LLD and system design](#lld-and-system-design)
- exact (brute-force) — [RAG and retrieval](#rag-and-retrieval)
- exact cache — [LLD and system design](#lld-and-system-design)
- exact match — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- example — [Classic ML](#classic-ml)
- exception — [Python and tooling](#python-and-tooling)
- experiment tracking — [MLOps](#mlops)
- expert parallel — [Training and fine-tuning](#training-and-fine-tuning)
- exploding gradient — [Neural networks](#neural-networks)
- EXPOSE — [Docker](#docker)

**F**

- F1 — [Classic ML](#classic-ml)
- fabrication — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- Facade — [LLD and system design](#lld-and-system-design)
- Factory — [LLD and system design](#lld-and-system-design)
- factuality — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- FAISS — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- faithfulness — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- FastAPI — [APIs and deployment](#apis-and-deployment)
- feature — [Classic ML](#classic-ml)
- feature store — [MLOps](#mlops)
- feed-forward network — [Neural networks](#neural-networks)
- few-shot — [Prompting and agents](#prompting-and-agents)
- FFN — [Neural networks](#neural-networks)
- fine-tuning — [RAG and retrieval](#rag-and-retrieval), [Training and fine-tuning](#training-and-fine-tuning)
- fixed-size — [Chunking](#chunking)
- FlashAttention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- Flask — [APIs and deployment](#apis-and-deployment)
- flat — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- float16 — [Neural networks](#neural-networks)
- float32 — [Neural networks](#neural-networks)
- form extraction — [OCR and document parsing](#ocr-and-document-parsing)
- formatter — [Python and tooling](#python-and-tooling)
- foundation model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- fp16 loss scaling — [Training and fine-tuning](#training-and-fine-tuning)
- FP4 — [Quantization](#quantization)
- FP8 — [Quantization](#quantization)
- framework — [Prompting and agents](#prompting-and-agents)
- free-threading — [Python and tooling](#python-and-tooling)
- frequency penalty — [Inference and serving](#inference-and-serving)
- FSDP — [Training and fine-tuning](#training-and-fine-tuning)
- full attention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- full fine-tuning — [Training and fine-tuning](#training-and-fine-tuning)
- fully connected layer — [Neural networks](#neural-networks)
- fully connected network — [Neural networks](#neural-networks)
- function calling — [Prompting and agents](#prompting-and-agents)

**G**

- GAN — [Neural networks](#neural-networks)
- gateway — [LLD and system design](#lld-and-system-design)
- GELU — [Neural networks](#neural-networks)
- generated tokens — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- generator — [Python and tooling](#python-and-tooling)
- GET — [APIs and deployment](#apis-and-deployment)
- GGUF — [Inference and serving](#inference-and-serving), [Quantization](#quantization)
- GIL — [Python and tooling](#python-and-tooling)
- GLM-OCR — [OCR and document parsing](#ocr-and-document-parsing)
- global batch size — [Training and fine-tuning](#training-and-fine-tuning)
- golden set — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- Goodhart — [Alignment and RL](#alignment-and-rl)
- GPQA — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- GPTQ — [Quantization](#quantization)
- GPU image — [Docker](#docker)
- GQA — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- gradient — [Neural networks](#neural-networks)
- gradient accumulation — [Neural networks](#neural-networks), [Training and fine-tuning](#training-and-fine-tuning)
- gradient checkpointing — [Training and fine-tuning](#training-and-fine-tuning)
- gradient clipping — [Neural networks](#neural-networks)
- gradient descent — [Classic ML](#classic-ml)
- gradio — [APIs and deployment](#apis-and-deployment)
- grammar — [Prompting and agents](#prompting-and-agents)
- GraphQL — [APIs and deployment](#apis-and-deployment)
- GraphRAG — [RAG and retrieval](#rag-and-retrieval)
- greedy — [Inference and serving](#inference-and-serving)
- ground truth — [Classic ML](#classic-ml)
- groundedness — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- grounding — [RAG and retrieval](#rag-and-retrieval)
- GroupNorm — [Neural networks](#neural-networks)
- gRPC — [APIs and deployment](#apis-and-deployment)
- GRPO — [Alignment and RL](#alignment-and-rl)
- GRU — [Neural networks](#neural-networks)
- guardrails — [Prompting and agents](#prompting-and-agents)
- Gunicorn — [APIs and deployment](#apis-and-deployment)

**H**

- hallucination — [Alignment and RL](#alignment-and-rl), [Evaluation and benchmarks](#evaluation-and-benchmarks)
- hallucination (RAG) — [RAG and retrieval](#rag-and-retrieval)
- He — [Neural networks](#neural-networks)
- head dimension — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- header — [APIs and deployment](#apis-and-deployment)
- health check — [APIs and deployment](#apis-and-deployment)
- HF Hub — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- HF tokenizers — [Tokenization](#tokenization)
- hidden states — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- hierarchical chunking — [Chunking](#chunking)
- hit rate — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- HLD — [LLD and system design](#lld-and-system-design)
- HLE — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- HNSW — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- horizontal — [LLD and system design](#lld-and-system-design)
- HTML table output — [OCR and document parsing](#ocr-and-document-parsing)
- HTR — [OCR and document parsing](#ocr-and-document-parsing)
- HTTP API — [APIs and deployment](#apis-and-deployment)
- human eval — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- human-in-the-loop — [Prompting and agents](#prompting-and-agents)
- human-on-the-loop — [Prompting and agents](#prompting-and-agents)
- HumanEval — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- hybrid retrieval — [RAG and retrieval](#rag-and-retrieval)
- hybrid search — [RAG and retrieval](#rag-and-retrieval)
- HyDE — [RAG and retrieval](#rag-and-retrieval)
- Hypercorn — [APIs and deployment](#apis-and-deployment)
- hyperparameters — [Classic ML](#classic-ml)

**I**

- ICR — [OCR and document parsing](#ocr-and-document-parsing)
- idempotent — [APIs and deployment](#apis-and-deployment)
- IDP — [OCR and document parsing](#ocr-and-document-parsing)
- image — [Docker](#docker)
- in-context learning — [Prompting and agents](#prompting-and-agents)
- independent variable — [Classic ML](#classic-ml)
- index — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- inference — [Inference and serving](#inference-and-serving)
- inference server — [APIs and deployment](#apis-and-deployment)
- inheritance — [LLD and system design](#lld-and-system-design)
- initialization (Xavier — [Neural networks](#neural-networks)
- inner product — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- input tokens — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- input variable — [Classic ML](#classic-ml)
- insert — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- Inspect — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- instance — [Classic ML](#classic-ml)
- instruct model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- instruction tuning — [Training and fine-tuning](#training-and-fine-tuning)
- instructions — [Prompting and agents](#prompting-and-agents)
- INT4 — [Quantization](#quantization)
- INT8 — [Quantization](#quantization)
- integration test — [Python and tooling](#python-and-tooling)
- interface — [LLD and system design](#lld-and-system-design)
- intermediate representations — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- IPO — [Alignment and RL](#alignment-and-rl)
- is — [Python and tooling](#python-and-tooling)
- iterable — [Python and tooling](#python-and-tooling)
- iteration — [Classic ML](#classic-ml)
- iterator — [Python and tooling](#python-and-tooling)
- ITL — [Inference and serving](#inference-and-serving)
- IVF — [Embeddings and vector DBs](#embeddings-and-vector-dbs)

**J**

- Jacobian — [Neural networks](#neural-networks)
- jailbreak — [Alignment and RL](#alignment-and-rl), [Prompting and agents](#prompting-and-agents)
- JSON mode — [Prompting and agents](#prompting-and-agents)
- JSON output — [OCR and document parsing](#ocr-and-document-parsing)
- judge — [Alignment and RL](#alignment-and-rl)
- JWT — [APIs and deployment](#apis-and-deployment)

**K**

- k-fold — [Classic ML](#classic-ml)
- Kaiming) — [Neural networks](#neural-networks)
- key-value extraction — [OCR and document parsing](#ocr-and-document-parsing)
- keyword search — [RAG and retrieval](#rag-and-retrieval)
- KIE — [OCR and document parsing](#ocr-and-document-parsing)
- kill — [Docker](#docker)
- KL divergence — [Alignment and RL](#alignment-and-rl)
- KL penalty — [Alignment and RL](#alignment-and-rl)
- kNN search — [RAG and retrieval](#rag-and-retrieval)
- knowledge graph RAG — [RAG and retrieval](#rag-and-retrieval)
- KTO — [Alignment and RL](#alignment-and-rl)
- KV cache — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- KV-cache quantization — [Quantization](#quantization)

**L**

- L1 — [Classic ML](#classic-ml)
- L2 regularization (lasso — [Classic ML](#classic-ml)
- label — [Classic ML](#classic-ml)
- label encoding — [Classic ML](#classic-ml)
- LanceDB — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- LangChain — [Prompting and agents](#prompting-and-agents)
- Langfuse — [MLOps](#mlops)
- LangGraph — [Prompting and agents](#prompting-and-agents)
- LangSmith — [MLOps](#mlops)
- late chunking — [Chunking](#chunking)
- late interaction — [Chunking](#chunking), [RAG and retrieval](#rag-and-retrieval)
- latency — [Inference and serving](#inference-and-serving)
- latent — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- latest — [Docker](#docker)
- layer — [Docker](#docker)
- LayerNorm — [Neural networks](#neural-networks)
- layout analysis — [OCR and document parsing](#ocr-and-document-parsing)
- leaderboard — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- leakage — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- leaky bucket — [LLD and system design](#lld-and-system-design)
- learned — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- learning rate — [Neural networks](#neural-networks)
- learning-rate schedule — [Neural networks](#neural-networks)
- Levenshtein — [OCR and document parsing](#ocr-and-document-parsing)
- lexical search — [RAG and retrieval](#rag-and-retrieval)
- library — [Python and tooling](#python-and-tooling)
- lighteval — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- likelihood — [Classic ML](#classic-ml)
- linear layer — [Neural networks](#neural-networks)
- linter — [Python and tooling](#python-and-tooling)
- list — [Python and tooling](#python-and-tooling)
- liveness — [APIs and deployment](#apis-and-deployment)
- llama.cpp — [Inference and serving](#inference-and-serving)
- LlamaIndex — [Prompting and agents](#prompting-and-agents)
- LLD — [LLD and system design](#lld-and-system-design)
- LLM — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- LLM gateway — [LLD and system design](#lld-and-system-design)
- LLM-as-judge — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- LLMOps — [MLOps](#mlops)
- lm-evaluation-harness — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- LMArena — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- load balancer — [LLD and system design](#lld-and-system-design)
- load balancing — [Inference and serving](#inference-and-serving)
- lock file — [Python and tooling](#python-and-tooling)
- log-odds — [Classic ML](#classic-ml)
- log-probs — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- logging — [MLOps](#mlops), [Python and tooling](#python-and-tooling)
- logit — [Classic ML](#classic-ml)
- logits — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- long context — [RAG and retrieval](#rag-and-retrieval)
- long-term memory — [Prompting and agents](#prompting-and-agents)
- LoRA — [Training and fine-tuning](#training-and-fine-tuning)
- LoRA merge — [Training and fine-tuning](#training-and-fine-tuning)
- LoRA rank r — [Training and fine-tuning](#training-and-fine-tuning)
- LoRA+ — [Training and fine-tuning](#training-and-fine-tuning)
- lora_alpha — [Training and fine-tuning](#training-and-fine-tuning)
- lora_dropout — [Training and fine-tuning](#training-and-fine-tuning)
- loss — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- loss function — [Classic ML](#classic-ml)
- lost in the middle — [RAG and retrieval](#rag-and-retrieval)
- LSTM — [Neural networks](#neural-networks)

**M**

- M — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- MAP — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- Markdown output — [OCR and document parsing](#ocr-and-document-parsing)
- Marker — [OCR and document parsing](#ocr-and-document-parsing)
- masked attention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- Matryoshka embeddings — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- max_completion_tokens — [Inference and serving](#inference-and-serving)
- max_new_tokens — [Inference and serving](#inference-and-serving)
- max_position_embeddings — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- max_seq_len — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- max_tokens — [Chunking](#chunking), [Inference and serving](#inference-and-serving)
- MBPP — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- MCP — [Prompting and agents](#prompting-and-agents)
- MCP client — [Prompting and agents](#prompting-and-agents)
- MCP host — [Prompting and agents](#prompting-and-agents)
- MCP server — [Prompting and agents](#prompting-and-agents)
- MCP tool — [Prompting and agents](#prompting-and-agents)
- measure — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- Medusa — [Inference and serving](#inference-and-serving)
- message broker — [LLD and system design](#lld-and-system-design)
- metadata — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- metadata filter — [RAG and retrieval](#rag-and-retrieval)
- METEOR — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- metric — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- metrics — [MLOps](#mlops)
- MHA — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- micro-batch size — [Training and fine-tuning](#training-and-fine-tuning)
- microservices — [LLD and system design](#lld-and-system-design)
- mid-training — [Training and fine-tuning](#training-and-fine-tuning)
- middleware — [APIs and deployment](#apis-and-deployment)
- Milvus — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- min-p — [Inference and serving](#inference-and-serving)
- MinerU — [OCR and document parsing](#ocr-and-document-parsing)
- mini-batch — [Neural networks](#neural-networks)
- mini-batch GD — [Classic ML](#classic-ml)
- mixed precision — [Neural networks](#neural-networks)
- MLA — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- MLflow — [MLOps](#mlops)
- MLOps — [MLOps](#mlops)
- MLP — [Neural networks](#neural-networks)
- MMLU — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- MMLU-Pro — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- MMR — [RAG and retrieval](#rag-and-retrieval)
- model card — [MLOps](#mlops)
- model drift — [MLOps](#mlops)
- model merging — [Training and fine-tuning](#training-and-fine-tuning)
- model parallel serving — [Inference and serving](#inference-and-serving)
- model registry — [MLOps](#mlops)
- model server — [APIs and deployment](#apis-and-deployment)
- model.eval() — [Neural networks](#neural-networks)
- model.train() — [Neural networks](#neural-networks)
- moderation — [Prompting and agents](#prompting-and-agents)
- modular RAG — [RAG and retrieval](#rag-and-retrieval)
- module — [Python and tooling](#python-and-tooling)
- MoE (mixture of experts) — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- monolith — [LLD and system design](#lld-and-system-design)
- MQA — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- MRR — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- multi-agent — [Prompting and agents](#prompting-and-agents)
- multi-query — [RAG and retrieval](#rag-and-retrieval)
- multi-stage build — [Docker](#docker)
- multi-vector — [RAG and retrieval](#rag-and-retrieval)
- multimodal — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- multimodal embedding (CLIP) — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- MXFP4 — [Quantization](#quantization)

**N**

- naive RAG — [RAG and retrieval](#rag-and-retrieval)
- namespace — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- nDCG — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- NER — [OCR and document parsing](#ocr-and-document-parsing)
- NF4 — [Quantization](#quantization)
- nn.Linear — [Neural networks](#neural-networks)
- node — [Chunking](#chunking)
- non-linearity — [Neural networks](#neural-networks)
- normalization — [Classic ML](#classic-ml)
- NoSQL — [LLD and system design](#lld-and-system-design)
- nucleus sampling — [Inference and serving](#inference-and-serving)
- NVFP4 — [Quantization](#quantization)
- NVIDIA Container Toolkit — [Docker](#docker)

**O**

- OAuth — [APIs and deployment](#apis-and-deployment)
- objective — [Classic ML](#classic-ml)
- observability — [MLOps](#mlops)
- observation — [Classic ML](#classic-ml)
- OCI — [Docker](#docker)
- OCR — [OCR and document parsing](#ocr-and-document-parsing)
- off-policy — [Alignment and RL](#alignment-and-rl)
- offline eval — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- offline inference — [Inference and serving](#inference-and-serving)
- Ollama — [Inference and serving](#inference-and-serving)
- olmOCR — [OCR and document parsing](#ocr-and-document-parsing)
- OmniDocBench — [OCR and document parsing](#ocr-and-document-parsing)
- OMR — [OCR and document parsing](#ocr-and-document-parsing)
- on-policy — [Alignment and RL](#alignment-and-rl)
- one-hot — [Classic ML](#classic-ml), [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- one-shot — [Prompting and agents](#prompting-and-agents)
- online — [Inference and serving](#inference-and-serving)
- online eval — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- ONNX — [Inference and serving](#inference-and-serving)
- OOD — [LLD and system design](#lld-and-system-design)
- OOP — [LLD and system design](#lld-and-system-design)
- open source (models) — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- open weights — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- OpenAI API — [Inference and serving](#inference-and-serving)
- OpenAI-compatible API — [Inference and serving](#inference-and-serving)
- OpenTelemetry — [MLOps](#mlops)
- orchestrator — [LLD and system design](#lld-and-system-design)
- orchestrator-worker — [Prompting and agents](#prompting-and-agents)
- ordinal encoding — [Classic ML](#classic-ml)
- ORPO — [Alignment and RL](#alignment-and-rl)
- outcome reward — [Alignment and RL](#alignment-and-rl)
- output tokens — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- overfit-underfit — [Classic ML](#classic-ml)
- overfitting — [Classic ML](#classic-ml), [Training and fine-tuning](#training-and-fine-tuning)

**P**

- P-tuning — [Training and fine-tuning](#training-and-fine-tuning)
- p50 — [MLOps](#mlops)
- p95 — [MLOps](#mlops)
- p99 latency — [MLOps](#mlops)
- package — [Python and tooling](#python-and-tooling)
- PAD — [Tokenization](#tokenization)
- padding — [Tokenization](#tokenization)
- padding mask — [Tokenization](#tokenization)
- PaddleOCR — [OCR and document parsing](#ocr-and-document-parsing)
- page — [Chunking](#chunking)
- page segmentation — [OCR and document parsing](#ocr-and-document-parsing)
- PagedAttention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- parameter — [Python and tooling](#python-and-tooling)
- parameters — [Classic ML](#classic-ml), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- params — [Classic ML](#classic-ml), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- parent-child chunking — [Chunking](#chunking)
- parent-document — [RAG and retrieval](#rag-and-retrieval)
- partition tolerance (CAP) — [LLD and system design](#lld-and-system-design)
- pass@1 — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- pass@k — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- passage — [Chunking](#chunking)
- past_key_values — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- PATCH — [APIs and deployment](#apis-and-deployment)
- path param — [APIs and deployment](#apis-and-deployment)
- payload — [Chunking](#chunking), [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- PDF text extraction — [OCR and document parsing](#ocr-and-document-parsing)
- PEFT — [Training and fine-tuning](#training-and-fine-tuning)
- per-channel — [Quantization](#quantization)
- per-group quantization — [Quantization](#quantization)
- per-tensor — [Quantization](#quantization)
- permute — [Neural networks](#neural-networks)
- perplexity — [Evaluation and benchmarks](#evaluation-and-benchmarks), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- pgvector — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- Phoenix — [MLOps](#mlops)
- pip — [Python and tooling](#python-and-tooling)
- pipeline OCR — [OCR and document parsing](#ocr-and-document-parsing)
- pipeline parallel — [Training and fine-tuning](#training-and-fine-tuning)
- Podman — [Docker](#docker)
- policy — [Alignment and RL](#alignment-and-rl)
- positional embedding — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- POST — [APIs and deployment](#apis-and-deployment)
- post-filter — [RAG and retrieval](#rag-and-retrieval)
- post-LN — [Neural networks](#neural-networks)
- PPO — [Alignment and RL](#alignment-and-rl)
- PQ — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- PR-AUC — [Classic ML](#classic-ml)
- pre-filter — [RAG and retrieval](#rag-and-retrieval)
- pre-LN — [Neural networks](#neural-networks)
- precision — [Classic ML](#classic-ml)
- precision@k — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- prediction — [Inference and serving](#inference-and-serving)
- predictor — [Classic ML](#classic-ml)
- preference model — [Alignment and RL](#alignment-and-rl)
- preference tuning — [Alignment and RL](#alignment-and-rl)
- prefill — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- prefill tokens — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- prefix cache — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- prefix tuning — [Training and fine-tuning](#training-and-fine-tuning)
- presence penalty — [Inference and serving](#inference-and-serving)
- pretrained model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- pretraining — [Training and fine-tuning](#training-and-fine-tuning)
- print — [Python and tooling](#python-and-tooling)
- PRM) — [Alignment and RL](#alignment-and-rl)
- probabilities — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- probability — [Classic ML](#classic-ml)
- process (Python) — [Python and tooling](#python-and-tooling)
- process reward (ORM — [Alignment and RL](#alignment-and-rl)
- prompt — [Prompting and agents](#prompting-and-agents)
- prompt cache — [LLD and system design](#lld-and-system-design), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- prompt drift — [MLOps](#mlops)
- prompt engineering — [Training and fine-tuning](#training-and-fine-tuning)
- prompt injection — [Prompting and agents](#prompting-and-agents)
- prompt template — [Tokenization](#tokenization)
- prompt tokens — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- prompt tuning — [Training and fine-tuning](#training-and-fine-tuning)
- proposition chunking — [Chunking](#chunking)
- protocol — [LLD and system design](#lld-and-system-design)
- proxy — [LLD and system design](#lld-and-system-design)
- pruning — [Quantization](#quantization)
- PTQ — [Quantization](#quantization)
- PUT — [APIs and deployment](#apis-and-deployment)
- Pydantic model — [APIs and deployment](#apis-and-deployment)
- pydantic.BaseModel — [Python and tooling](#python-and-tooling)
- pyproject.toml — [Python and tooling](#python-and-tooling)

**Q**

- QAT — [Quantization](#quantization)
- Qdrant — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- QLoRA — [Quantization](#quantization), [Training and fine-tuning](#training-and-fine-tuning)
- quantization — [Quantization](#quantization)
- quantized LoRA model — [Quantization](#quantization)
- quantized model — [Quantization](#quantization)
- query expansion — [RAG and retrieval](#rag-and-retrieval)
- query param — [APIs and deployment](#apis-and-deployment)
- query rewriting — [RAG and retrieval](#rag-and-retrieval)
- queue — [LLD and system design](#lld-and-system-design)

**R**

- RAG — [RAG and retrieval](#rag-and-retrieval), [Training and fine-tuning](#training-and-fine-tuning)
- RAGAS — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- random forest — [Classic ML](#classic-ml)
- ranker — [RAG and retrieval](#rag-and-retrieval)
- rate limiting — [APIs and deployment](#apis-and-deployment)
- re-ranker — [RAG and retrieval](#rag-and-retrieval)
- re-scoring — [RAG and retrieval](#rag-and-retrieval)
- ReAct — [Prompting and agents](#prompting-and-agents)
- readiness — [APIs and deployment](#apis-and-deployment)
- reasoning model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- reasoning tokens — [Prompting and agents](#prompting-and-agents)
- recall — [Classic ML](#classic-ml)
- recall (ANN) — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- recall (RAG retrieval) — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- recall@k — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- recognition (OCR stages) — [OCR and document parsing](#ocr-and-document-parsing)
- recursive — [Chunking](#chunking)
- RecursiveCharacterTextSplitter — [Chunking](#chunking)
- red teaming — [Alignment and RL](#alignment-and-rl)
- reference model — [Alignment and RL](#alignment-and-rl)
- registry — [Docker](#docker)
- regression — [Classic ML](#classic-ml)
- regression test — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- regularization — [Classic ML](#classic-ml)
- REINFORCE — [Alignment and RL](#alignment-and-rl)
- rejection sampling — [Alignment and RL](#alignment-and-rl)
- relative path — [Python and tooling](#python-and-tooling)
- ReLU — [Neural networks](#neural-networks)
- rematerialization — [Training and fine-tuning](#training-and-fine-tuning)
- repetition penalty — [Inference and serving](#inference-and-serving)
- replicas — [Inference and serving](#inference-and-serving)
- repository — [Docker](#docker)
- representation — [Embeddings and vector DBs](#embeddings-and-vector-dbs), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- reproducibility — [MLOps](#mlops)
- request body — [APIs and deployment](#apis-and-deployment)
- requirements.txt — [Python and tooling](#python-and-tooling)
- reranking — [RAG and retrieval](#rag-and-retrieval)
- reshape — [Neural networks](#neural-networks)
- residual connection — [Neural networks](#neural-networks)
- resolution — [OCR and document parsing](#ocr-and-document-parsing)
- resource — [Prompting and agents](#prompting-and-agents)
- response — [Classic ML](#classic-ml)
- REST — [APIs and deployment](#apis-and-deployment)
- retrieval failure — [RAG and retrieval](#rag-and-retrieval)
- retrieval-augmented generation — [RAG and retrieval](#rag-and-retrieval)
- retriever — [RAG and retrieval](#rag-and-retrieval)
- retry — [APIs and deployment](#apis-and-deployment)
- reward hacking — [Alignment and RL](#alignment-and-rl)
- reward model — [Alignment and RL](#alignment-and-rl), [Evaluation and benchmarks](#evaluation-and-benchmarks)
- ridge) — [Classic ML](#classic-ml)
- RLAIF — [Alignment and RL](#alignment-and-rl)
- RLHF — [Alignment and RL](#alignment-and-rl)
- RLOO — [Alignment and RL](#alignment-and-rl)
- RLVR — [Alignment and RL](#alignment-and-rl)
- RMSNorm — [Neural networks](#neural-networks)
- RNN — [Neural networks](#neural-networks)
- ROC-AUC — [Classic ML](#classic-ml)
- roll forward — [MLOps](#mlops)
- rollback — [MLOps](#mlops)
- RoPE — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- ROUGE — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- route — [APIs and deployment](#apis-and-deployment)
- router — [LLD and system design](#lld-and-system-design)
- row — [Classic ML](#classic-ml)
- RRF — [RAG and retrieval](#rag-and-retrieval)
- rsLoRA — [Training and fine-tuning](#training-and-fine-tuning)
- RUN — [Docker](#docker)
- runc — [Docker](#docker)
- runtime — [Prompting and agents](#prompting-and-agents)

**S**

- safe (HTTP) — [APIs and deployment](#apis-and-deployment)
- safety training — [Alignment and RL](#alignment-and-rl)
- sample — [Classic ML](#classic-ml)
- sampling — [Inference and serving](#inference-and-serving)
- sampling) — [RAG and retrieval](#rag-and-retrieval)
- scaffold — [Prompting and agents](#prompting-and-agents)
- scale — [OCR and document parsing](#ocr-and-document-parsing)
- scaling laws — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- scanned PDF — [OCR and document parsing](#ocr-and-document-parsing)
- schema — [APIs and deployment](#apis-and-deployment)
- schema-constrained decoding — [Prompting and agents](#prompting-and-agents)
- score — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- scratchpad — [Prompting and agents](#prompting-and-agents)
- segment — [Chunking](#chunking)
- self-attention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- self-consistency — [Prompting and agents](#prompting-and-agents)
- self-supervised — [Classic ML](#classic-ml)
- self-training — [Training and fine-tuning](#training-and-fine-tuning)
- semantic — [Chunking](#chunking)
- semantic cache — [LLD and system design](#lld-and-system-design)
- semantic search — [RAG and retrieval](#rag-and-retrieval)
- sensitivity — [Classic ML](#classic-ml)
- sentence chunking — [Chunking](#chunking)
- sentence encoder — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- sentence-transformers — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- sentence-window retrieval — [RAG and retrieval](#rag-and-retrieval)
- SentencePiece — [Tokenization](#tokenization)
- sequence diagram — [LLD and system design](#lld-and-system-design)
- sequence length — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- serving — [Inference and serving](#inference-and-serving)
- set — [Python and tooling](#python-and-tooling)
- SFT — [Training and fine-tuning](#training-and-fine-tuning)
- SGD — [Classic ML](#classic-ml), [Neural networks](#neural-networks)
- SGLang — [Inference and serving](#inference-and-serving)
- shadow deployment — [MLOps](#mlops)
- shallow copy — [Python and tooling](#python-and-tooling)
- short-term memory — [Prompting and agents](#prompting-and-agents)
- shortcut — [Neural networks](#neural-networks)
- sigmoid — [Classic ML](#classic-ml)
- sigmoid output — [Classic ML](#classic-ml)
- SiLU (Swish) — [Neural networks](#neural-networks)
- similarity — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- SimPO — [Alignment and RL](#alignment-and-rl)
- single agent — [Prompting and agents](#prompting-and-agents)
- single-stage — [Docker](#docker)
- sinusoidal — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- skip connection — [Neural networks](#neural-networks)
- SLA — [MLOps](#mlops)
- SLI — [MLOps](#mlops)
- sliding window — [LLD and system design](#lld-and-system-design)
- sliding-window attention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- slim — [Docker](#docker)
- SLM — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- SLO — [MLOps](#mlops)
- small-to-big — [RAG and retrieval](#rag-and-retrieval)
- soft prompts — [Training and fine-tuning](#training-and-fine-tuning)
- softmax — [Classic ML](#classic-ml)
- SOLID — [LLD and system design](#lld-and-system-design)
- source linking — [RAG and retrieval](#rag-and-retrieval)
- span — [MLOps](#mlops)
- sparse attention — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- sparse retrieval — [RAG and retrieval](#rag-and-retrieval)
- special tokens — [Tokenization](#tokenization)
- specification gaming — [Alignment and RL](#alignment-and-rl)
- specificity — [Classic ML](#classic-ml)
- speculative decoding — [Inference and serving](#inference-and-serving)
- SPLADE — [RAG and retrieval](#rag-and-retrieval)
- split — [Chunking](#chunking)
- SQL — [LLD and system design](#lld-and-system-design)
- SSE — [APIs and deployment](#apis-and-deployment)
- stack trace — [Python and tooling](#python-and-tooling)
- standardization — [Classic ML](#classic-ml)
- stateful — [LLD and system design](#lld-and-system-design)
- stateless — [LLD and system design](#lld-and-system-design)
- step — [Classic ML](#classic-ml), [Neural networks](#neural-networks)
- step size — [Neural networks](#neural-networks)
- stop — [Docker](#docker)
- stop sequence — [Tokenization](#tokenization)
- stop token — [Tokenization](#tokenization)
- Strategy — [LLD and system design](#lld-and-system-design)
- streaming — [Inference and serving](#inference-and-serving)
- streaming pipeline — [MLOps](#mlops)
- streamlit — [APIs and deployment](#apis-and-deployment)
- stride — [Chunking](#chunking)
- structure-aware — [Chunking](#chunking)
- structured output — [Prompting and agents](#prompting-and-agents)
- subword — [Tokenization](#tokenization)
- supervised — [Classic ML](#classic-ml)
- SWE-bench — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- SWE-bench Verified — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- SwiGLU — [Neural networks](#neural-networks)
- sycophancy — [Alignment and RL](#alignment-and-rl)
- sync — [Python and tooling](#python-and-tooling)
- sync endpoint — [APIs and deployment](#apis-and-deployment)
- synchronous — [LLD and system design](#lld-and-system-design)
- synthetic data — [Training and fine-tuning](#training-and-fine-tuning)
- system card — [MLOps](#mlops)
- system design — [LLD and system design](#lld-and-system-design)
- system message — [Prompting and agents](#prompting-and-agents)
- system prompt — [Prompting and agents](#prompting-and-agents), [Tokenization](#tokenization)

**T**

- table — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- table detection — [OCR and document parsing](#ocr-and-document-parsing)
- table extraction — [OCR and document parsing](#ocr-and-document-parsing)
- table structure recognition (TSR) — [OCR and document parsing](#ocr-and-document-parsing)
- tag — [Docker](#docker)
- target — [Classic ML](#classic-ml)
- TEDS — [OCR and document parsing](#ocr-and-document-parsing)
- temperature — [Inference and serving](#inference-and-serving), [Prompting and agents](#prompting-and-agents)
- temperature=0 — [Inference and serving](#inference-and-serving)
- tensor parallel — [Training and fine-tuning](#training-and-fine-tuning)
- tensor view — [Neural networks](#neural-networks)
- TensorRT — [Inference and serving](#inference-and-serving)
- TensorRT-LLM — [Inference and serving](#inference-and-serving)
- Tesseract — [OCR and document parsing](#ocr-and-document-parsing)
- test set — [Classic ML](#classic-ml), [Evaluation and benchmarks](#evaluation-and-benchmarks)
- text-embedding — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- text-layer PDF — [OCR and document parsing](#ocr-and-document-parsing)
- TF-IDF — [RAG and retrieval](#rag-and-retrieval)
- TF-IDF vector — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- TF32 — [Neural networks](#neural-networks)
- TGI — [Inference and serving](#inference-and-serving)
- thinking — [Prompting and agents](#prompting-and-agents)
- thinking model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- thread — [Python and tooling](#python-and-tooling)
- throttling — [APIs and deployment](#apis-and-deployment)
- throughput — [Inference and serving](#inference-and-serving)
- tiktoken — [Tokenization](#tokenization)
- timeout — [APIs and deployment](#apis-and-deployment)
- tmpfs — [Docker](#docker)
- token — [Tokenization](#tokenization)
- token bucket — [LLD and system design](#lld-and-system-design)
- token count (API) — [Tokenization](#tokenization)
- token embedding — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- tokenizer — [Tokenization](#tokenization)
- tokenizer-aware chunk size — [Chunking](#chunking)
- tool calling — [Prompting and agents](#prompting-and-agents)
- top-k — [Inference and serving](#inference-and-serving), [RAG and retrieval](#rag-and-retrieval)
- top-p — [Inference and serving](#inference-and-serving)
- top-p (retrieval — [RAG and retrieval](#rag-and-retrieval)
- top-p in prompting — [Prompting and agents](#prompting-and-agents)
- torch.compile — [Inference and serving](#inference-and-serving)
- torch.no_grad() — [Neural networks](#neural-networks)
- TorchScript — [Inference and serving](#inference-and-serving)
- total params — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- TPOT — [Inference and serving](#inference-and-serving)
- trace — [MLOps](#mlops)
- traceback — [Python and tooling](#python-and-tooling)
- tracing — [MLOps](#mlops)
- training — [Classic ML](#classic-ml)
- training data — [Quantization](#quantization)
- Transformer — [Neural networks](#neural-networks)
- transformers — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- transpose — [Neural networks](#neural-networks)
- Tree of Thoughts — [Prompting and agents](#prompting-and-agents)
- true negative rate — [Classic ML](#classic-ml)
- true positive rate (TPR) — [Classic ML](#classic-ml)
- TruLens — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- truncated embeddings — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- truncation — [Tokenization](#tokenization)
- TTFT — [Inference and serving](#inference-and-serving)
- tuple — [Python and tooling](#python-and-tooling)
- two-tower — [RAG and retrieval](#rag-and-retrieval)
- type check — [Python and tooling](#python-and-tooling)
- type hint — [Python and tooling](#python-and-tooling)
- TypedDict — [APIs and deployment](#apis-and-deployment)
- typing.Optional[X] — [Python and tooling](#python-and-tooling)

**U**

- underfitting — [Classic ML](#classic-ml)
- unfaithfulness — [Evaluation and benchmarks](#evaluation-and-benchmarks)
- Unigram — [Tokenization](#tokenization)
- unit test — [Python and tooling](#python-and-tooling)
- unmerge — [Training and fine-tuning](#training-and-fine-tuning)
- Unstructured — [OCR and document parsing](#ocr-and-document-parsing)
- unsupervised — [Classic ML](#classic-ml)
- update — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- upsert — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- URL path — [APIs and deployment](#apis-and-deployment)
- uv pip — [Python and tooling](#python-and-tooling)
- uv venv — [Python and tooling](#python-and-tooling)
- Uvicorn — [APIs and deployment](#apis-and-deployment)

**V**

- VAE — [Neural networks](#neural-networks)
- validation — [Classic ML](#classic-ml), [Prompting and agents](#prompting-and-agents)
- value model — [Alignment and RL](#alignment-and-rl)
- vanishing gradient — [Neural networks](#neural-networks)
- vector — [Embeddings and vector DBs](#embeddings-and-vector-dbs), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- vector database — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- vector DB — [LLD and system design](#lld-and-system-design)
- vector index — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- vector library — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- vector RAG — [RAG and retrieval](#rag-and-retrieval)
- vector search — [RAG and retrieval](#rag-and-retrieval)
- vector store — [Embeddings and vector DBs](#embeddings-and-vector-dbs), [MLOps](#mlops)
- venv — [Python and tooling](#python-and-tooling)
- verifier — [Alignment and RL](#alignment-and-rl)
- vertical scaling — [LLD and system design](#lld-and-system-design)
- virtual machine — [Docker](#docker)
- virtualenv — [Python and tooling](#python-and-tooling)
- vision-language model — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- vLLM — [Inference and serving](#inference-and-serving)
- VLM — [Transformers and LLM architecture](#transformers-and-llm-architecture)
- vocab — [Tokenization](#tokenization)
- vocab_size — [Tokenization](#tokenization)
- vocabulary — [Tokenization](#tokenization)
- volume — [Docker](#docker)

**W**

- W4A16 — [Quantization](#quantization)
- W8A8 notation — [Quantization](#quantization)
- warm start — [Inference and serving](#inference-and-serving)
- warmup — [Neural networks](#neural-networks)
- Weaviate — [Embeddings and vector DBs](#embeddings-and-vector-dbs)
- web service — [APIs and deployment](#apis-and-deployment)
- WebSocket — [APIs and deployment](#apis-and-deployment)
- weight decay — [Classic ML](#classic-ml), [Neural networks](#neural-networks)
- weight quantization — [Quantization](#quantization)
- weight-and-activation quantization — [Quantization](#quantization)
- weight-only — [Quantization](#quantization)
- weighted score fusion — [RAG and retrieval](#rag-and-retrieval)
- weights — [Classic ML](#classic-ml), [Transformers and LLM architecture](#transformers-and-llm-architecture)
- Weights & Biases — [MLOps](#mlops)
- WER — [Evaluation and benchmarks](#evaluation-and-benchmarks), [OCR and document parsing](#ocr-and-document-parsing)
- window overlap — [Chunking](#chunking)
- word — [Tokenization](#tokenization)
- word count — [Tokenization](#tokenization)
- WordPiece — [Tokenization](#tokenization)
- workflow — [Prompting and agents](#prompting-and-agents)
- WSGI — [APIs and deployment](#apis-and-deployment)

**X**

- X or None — [Python and tooling](#python-and-tooling)
- X | None — [Python and tooling](#python-and-tooling)

**Z**

- ZeRO — [Training and fine-tuning](#training-and-fine-tuning)
- zero-shot — [Prompting and agents](#prompting-and-agents)
