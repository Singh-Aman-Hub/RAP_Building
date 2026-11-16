# Terms — Build a mini-Llama

Folder: `16_modern_llm_architecture/07_build_mini_llama` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [16 README](../README.md)

**Terms in this folder:** [Decoder-only transformer](#decoder-only-transformer) · [ModelConfig dataclass](#modelconfig-dataclass) · [DecoderBlock](#decoderblock) · [head_dim](#head_dim) · [Rotate-half RoPE layout](#rotate-half-rope-layout) · [start_pos (position offset)](#start_pos-position-offset) · [KVCache class](#kvcache-class) · [Cached vs uncached equality test](#cached-vs-uncached-equality-test) · [Weight tying](#weight-tying) · [Teacher forcing and shifted targets](#teacher-forcing-and-shifted-targets) · [Cross-entropy loss](#cross-entropy-loss) · [Perplexity](#perplexity) · [AdamW with warm-up and cosine decay](#adamw-with-warm-up-and-cosine-decay) · [Greedy decoding](#greedy-decoding) · [Causality test](#causality-test)

---

## Decoder-only transformer  `(concept)`
**1. Definition:** A stack of causal self-attention + MLP blocks that predicts the next token from the previous ones.
**2. In plain words:** The architecture behind GPT, Llama, Qwen, DeepSeek.
**3. Example to quote:** > "738,432 parameters in our version."
**Seen in:** `main.py`, `Transformer`

## ModelConfig dataclass  `(concept)`
**1. Definition:** A dataclass collecting every architecture hyperparameter in one place.
**2. In plain words:** The settings sheet for the model.
**3. Example to quote:** > "vocab_size, d_model, n_layers, n_heads, n_kv_heads, ffn_hidden, ..."
**Seen in:** `main.py`, `ModelConfig`

## DecoderBlock  `(concept)`
**1. Definition:** One layer: x = x + attn(norm(x)); x = x + mlp(norm(x)).
**2. In plain words:** The repeated unit.
**3. Example to quote:** > "176,384 parameters per block here."
**Seen in:** `main.py`, `DecoderBlock`

## head_dim  `(concept)`
**1. Definition:** Size of each attention head, d_model / n_heads (16 here).
**2. In plain words:** How wide each head is.
**3. Example to quote:** > "head_dim is a derived property."
**Seen in:** `main.py`, `ModelConfig`

## Rotate-half RoPE layout  `(concept)`
**1. Definition:** RoPE implementation that rotates the first half of the head dimension against the second half using rotate_half, as in Hugging Face Llama.
**2. In plain words:** Equivalent to interleaved pairs up to a permutation of dimensions.
**3. Example to quote:** > "(a, b) becomes (-b, a)."
**Seen in:** `main.py`, `RotaryEmbedding`

## start_pos (position offset)  `(concept)`
**1. Definition:** Number of tokens already in the cache; new tokens get absolute positions start_pos...start_pos+t-1.
**2. In plain words:** Where we are in the story.
**3. Example to quote:** > "Must be read once per forward, not per layer."
**Seen in:** `main.py`, `Transformer.forward`

## KVCache class  `(concept)`
**1. Definition:** An object holding per-layer K and V tensors, appended to at each step.
**2. In plain words:** The model's notebook.
**3. Example to quote:** > "20,480 bytes for 20 tokens here."
**Seen in:** `main.py`, `KVCache`

## Cached vs uncached equality test  `(concept)`
**1. Definition:** Greedy generation with and without the cache must produce identical ids, and logits must agree to float rounding.
**2. In plain words:** The proof the cache is correct.
**3. Example to quote:** > "torch.equal True; max logit diff 4.3e-06."
**Seen in:** `main.py`, section 11

## Weight tying  `(concept)`
**1. Definition:** The output head shares the embedding matrix.
**2. In plain words:** One table, two jobs.
**3. Example to quote:** > "Saves 32,768 parameters here; 136M in Qwen2.5-0.5B."
**Seen in:** `main.py`, `Transformer.__init__`

## Teacher forcing and shifted targets  `(concept)`
**1. Definition:** Training feeds the true previous tokens and asks for the next one; y is x shifted by one.
**2. In plain words:** Predict the next byte at every position at once.
**3. Example to quote:** > "x[i, j+1] == y[i, j]."
**Seen in:** `main.py`, `get_batch`

## Cross-entropy loss  `(concept)`
**1. Definition:** Negative log-probability the model assigns to the true next token, averaged.
**2. In plain words:** How surprised the model is.
**3. Example to quote:** > "Initial loss 5.537 = ln 256."
**Seen in:** `main.py`, training

## Perplexity  `(concept)`
**1. Definition:** exp(loss): the effective number of equally likely choices.
**2. In plain words:** 8.15 means about as unsure as choosing among 8 bytes.
**3. Example to quote:** > "A random guess has perplexity 256."
**Seen in:** `main.py`, validation

## AdamW with warm-up and cosine decay  `(concept)`
**1. Definition:** Adam with decoupled weight decay, a short linear warm-up and cosine learning-rate decay.
**2. In plain words:** Gentle start, fast middle, careful end.
**3. Example to quote:** > "betas (0.9, 0.95), weight decay 0.1."
**Seen in:** `main.py`, training loop

## Greedy decoding  `(concept)`
**1. Definition:** Always pick the highest-probability next token (argmax).
**2. In plain words:** Deterministic but can loop.
**3. Example to quote:** > "Our output repeats 'the the the'."
**Seen in:** `main.py`, `generate`

## Causality test  `(concept)`
**1. Definition:** Changing future tokens must not change past logits.
**2. In plain words:** The model cannot peek ahead.
**3. Example to quote:** > "True for positions < 20."
**Seen in:** `main.py`, section 12

