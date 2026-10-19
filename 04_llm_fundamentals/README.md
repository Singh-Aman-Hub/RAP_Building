# 04 - LLM Fundamentals (syllabus topic 1, first half)

## Why this matters
Every other topic - parameters, prompting, agents, APIs, Docker - is "plumbing around an LLM". Interviewers want proof you understand what the model actually does: text -> tokens -> vectors -> logits -> next token. If you can run, inspect and explain these 7 small scripts, you can answer most "how does an LLM work?" questions with code, not buzzwords.

## Where this fits
**Builds on**
- [00_python_foundations](../00_python_foundations/) - functions, lists/dicts, f-strings, `with`, comprehensions appear in every script here.
- [01_debugging_and_environment](../01_debugging_and_environment/) - virtual environment + reading tracebacks (you will hit `OSError`, `IndexError`, shape errors in the "BREAK IT" drills).
- [03_ml_foundations](../03_ml_foundations/) - the **training loop** (`forward -> loss -> backward -> step -> zero_grad`), softmax and cross-entropy. Step 05 and 07 here reuse them; the loss numbers (11 -> 6 -> 0.5) only make sense once you know cross-entropy.

**Leads to**
- [05_llm_parameters](../05_llm_parameters/) - temperature/top-k/top-p act on the **logits you print in step 05**; `max_new_tokens`/context window come from the **position table in step 03**; the LR/batch/epoch experiments extend the **loop in step 07**.
- [06_llm_training_and_fast_inference](../06_llm_training_and_fast_inference/) - the real fine-tune (datasets, LoRA/QLoRA) generalises step 07; vLLM's KV-cache management generalises step 06; memory arithmetic extends step 04.
- [07_prompt_engineering](../07_prompt_engineering/) and [08_agent_harness](../08_agent_harness/) - prompts are just tokens in the context window (step 02); a harness repeatedly calls the generation loop of step 06.
- [09_api_development](../09_api_development/) - you will wrap this tokenizer + model + generate in an HTTP endpoint (load the model ONCE at startup, as in step 04).

## Concept map
```
 raw text
    |  02 tokenizer (BPE, ids, special tokens, padding+mask)
    v
 token ids ----------------------------.
    |  03 embeddings: id -> vector      |  context window limit = number of
    |     + position vector             |  position vectors (05_llm_parameters/04)
    v                                   |
 [ transformer block ] x N   <- 01 attention (Q,K,V, causal mask) + MLP
    |  04 the weights live here: layers, hidden size, parameters, bytes
    v
 logits (vocab-sized scores)  <- 05 forward pass
    |  softmax -> probabilities
    v
 choose next token  <- greedy (06) | sampling (module 05)
    |  append, repeat (KV cache makes it fast)   <- 06 generation loop
    v
 text out
 
 life cycle:  random weights --pretrain--> base model --fine-tune--> specialised model --inference--> product   (07)
```

## Study path
| # | Folder | You learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_attention_from_scratch](01_attention_from_scratch/) | Q/K/V, softmax weights, causal mask, shapes | `.venv/bin/python 04_llm_fundamentals/01_attention_from_scratch/main.py` | 60-90 min |
| 2 | [02_tokenizer](02_tokenizer/) | BPE, ids, `Ġ`, special tokens, padding, attention mask | `.../02_tokenizer/main.py` | 45-60 min |
| 3 | [03_embeddings](03_embeddings/) | embedding table, cosine similarity, position embeddings | `.../03_embeddings/main.py` | 45-60 min |
| 4 | [04_load_pretrained_model](04_load_pretrained_model/) | config vs weights, parameters, dtype, memory | `.../04_load_pretrained_model/main.py` | 45 min |
| 5 | [05_forward_pass_logits](05_forward_pass_logits/) | forward pass, logits, softmax, top-k tokens, loss | `.../05_forward_pass_logits/main.py` | 45 min |
| 6 | [06_generation_loop](06_generation_loop/) | autoregressive loop by hand, `generate`, KV cache | `.../06_generation_loop/main.py` | 60 min |
| 7 | [07_pretrain_vs_finetune_vs_inference](07_pretrain_vs_finetune_vs_inference/) | the three stages through loss numbers, a full training loop | `.../07_pretrain_vs_finetune_vs_inference/main.py` | 60 min |

Each step folder has `main.py` (commented, runnable) and `WALKTHROUGH.md` (syntax tables, real output, experiments, debugging drill, say-it-aloud). Then do [exercises.md](exercises.md). Models used are small (`distilgpt2`, 330 MB; first run downloads it).

> From the repo root use `.venv/bin/python ...`; inside a step folder plain `python main.py` works once the venv is activated (`source .venv/bin/activate`).

## Syllabus coverage (topic 1)
| Syllabus bullet | Where |
|---|---|
| LLM fundamentals and transformer architecture | this README + step 01, 04, 05 |
| Tokenization and embeddings | steps 02, 03 |
| Pretraining vs fine-tuning vs inference | step 07 + section below |
| Loading pretrained models | step 04 |
| Training loop, loss and optimizer | step 07 here (taste) -> [03](../03_ml_foundations/) (basics) -> [05/06](../05_llm_parameters/06_training_parameters/) (knobs) |
| Dataset preparation and preprocessing | [06_llm_training_and_fast_inference](../06_llm_training_and_fast_inference/) |
| LoRA and QLoRA basics | [06_llm_training_and_fast_inference](../06_llm_training_and_fast_inference/) |
| Fast inference frameworks (vLLM, Transformers) | [06_llm_training_and_fast_inference](../06_llm_training_and_fast_inference/); Transformers `generate` in step 06 here |
| Latency, throughput and GPU memory | step 04/06 here + [05 step 04, 07](../05_llm_parameters/) + module 06 |

## The ideas in plain English

### LLM and the transformer
A large language model is a neural network trained to predict the next token. "Large" = many parameters (82 M for distilgpt2, billions for chat models). A **transformer** is the architecture: stacked blocks, each with **self-attention** (every token gathers information from other tokens) and an **MLP** (processes each token further). Residual connections and layer norm keep deep stacks trainable. *Analogy:* attention is a group chat where each word decides whose messages to read; the MLP is each word thinking privately before the next round. *Trade-off:* attention cost grows with the square of sequence length.

### Tokenization
Text is split into sub-word tokens (BPE). Common words = 1 token, rare words/other scripts = many. *Consequences:* cost and context budgeting are in tokens, not words; spelling quirks and numbers are harder; spaces matter (`world` vs ` world`).

### Embeddings
A table mapping token id -> learned vector. Similar tokens end up nearby (cosine similarity). A position embedding is added so word order is known. *Limitation:* the input embedding for "bank" is the same in every sentence; context makes it specific only in later layers.

### Pretraining vs fine-tuning vs inference
- **Pretraining**: next-token prediction on a huge corpus from random weights. Expensive; you almost never do it.
- **Fine-tuning**: continue training on your small dataset (full or LoRA) for a style/task; low learning rate.
- **Inference**: frozen weights, forward passes only; where latency and cost matter.
*Trade-off:* fine-tuning improves task fit but can overfit and forget; prompting needs no training but uses context window and can be less consistent (compare module 07).

### Loading pretrained models
Config (architecture) + weights (numbers) from the Hugging Face Hub, cached in `~/.cache/huggingface`. `AutoTokenizer` and `AutoModelForCausalLM` pick the right classes by name. Always pair the matching tokenizer and model.

### Logits -> probabilities -> token
The model outputs one score per vocabulary entry for each position; softmax makes probabilities; a decoding rule picks the next token.

### Memory and speed basics
`memory ~ parameters x bytes per parameter` (+ KV cache + activations). One forward pass per generated token -> output length drives latency; KV cache trades memory for speed. These ideas are the base of module 06's inference engineering.

## Common mistakes & gotchas
- Using a tokenizer from a different model than the weights (ids mean different things) -> gibberish, no error.
- Forgetting `.eval()` / `torch.no_grad()` at inference (slower, dropout noise, more memory).
- Counting words instead of tokens when estimating cost or context fit.
- Printing logits and treating them as probabilities.
- Prompt + `max_new_tokens` > context window -> `IndexError` on GPT-2.
- Padding on the wrong side/no attention mask for batched generation.
- Assuming a small model "knows" facts (distilgpt2 says "The capital of France is the ..."): fluent is not factual.
- Naming every script `main.py` and importing `main` from another folder -> wrong file (name collision). Use distinct module names for shared helpers.

## Self-check
- [ ] I can draw the pipeline text -> tokens -> embeddings -> blocks -> logits -> token.
- [ ] I can explain Q, K, V and the causal mask using my own numpy code.
- [ ] I can explain why `world` and ` world` have different ids.
- [ ] I can compute parameters x bytes for any model and dtype.
- [ ] I can write the greedy generation loop from memory.
- [ ] I can state the loss of a random model for vocabulary size V.
- [ ] I can explain pretraining, fine-tuning and inference in 30 seconds each.
- [ ] I have run every `main.py`, done a "TRY IT" and a "BREAK IT" for each.

## Interview questions
1. **What does an LLM do, in one sentence?** It predicts a probability distribution over the next token given the previous tokens; text is produced by repeatedly sampling/choosing a token and appending it.
2. **What is a token and why not use words or characters?** A sub-word unit from a fixed vocabulary. Words give a huge vocabulary and can't handle unseen words; characters give very long sequences. BPE balances both. Limitation: odd splits for numbers and non-English text.
3. **Explain self-attention.** Each token creates a query, key and value. Query-key dot products (scaled, softmaxed) give weights over other tokens; the output is the weighted sum of values. Causal masking prevents looking at future tokens. Cost is quadratic in sequence length.
4. **Why do transformers need position information?** Attention treats tokens as an unordered set; position embeddings (learned or rotary) inject order.
5. **What is an embedding?** A learned vector for each token (a row in a vocab x hidden table); similar meanings -> similar vectors, comparable with cosine similarity.
6. **Pretraining vs fine-tuning vs inference?** See above; give the data size, cost and learning-rate differences.
7. **How much memory does a 7B model need in float16? int4?** 7e9 x 2 B = 14 GB; 7e9 x 0.5 B = 3.5 GB, plus KV cache and activations.
8. **What are logits? Why softmax?** Raw scores per vocabulary token; softmax converts them to non-negative numbers summing to 1.
9. **What is the KV cache and its trade-off?** Stores keys/values of previous tokens so each new token costs one step; speeds decoding but memory grows with context length and batch size.
10. **Why is generation slower than a single forward pass?** One forward pass per new token, sequentially.
11. **What's a causal LM vs masked LM?** Causal (GPT) predicts the next token left-to-right; masked (BERT) predicts hidden tokens using both sides - better for classification/embeddings, not for text generation.
12. **Why might a model's output differ between runs?** Sampling randomness (temperature/top-k/top-p) - module 05; also non-deterministic GPU kernels.
13. **What does `model.eval()` do?** Disables dropout (and uses running statistics for batch norm); not the same as `torch.no_grad()`, which disables gradient tracking.
14. **Why is the loss of an untrained model about ln(V)?** Uniform guessing over V tokens gives -ln(1/V) = ln V; ln 50257 = 10.8.
15. **What is a distilled model?** A smaller student trained to mimic a bigger teacher; faster/smaller, a bit less accurate (distilgpt2 vs gpt2).
