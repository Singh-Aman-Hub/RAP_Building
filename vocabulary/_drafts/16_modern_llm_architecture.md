<!-- vocabulary draft: 16_modern_llm_architecture -->

### Byte-level BPE (byte-pair encoding)
<!-- cat: llm -->
- **Technical definition:** Subword tokenizer that starts from the 256 byte values and repeatedly merges the most frequent adjacent pair into a new token; encoding applies merges in learned order.
- **Plain English:** Glue the most common letter pairs into bigger pieces until the vocabulary is full.
- **Example / analogy:** 300 merges on a 20 KB text gave 2.21 bytes per token.
- **Snippet:**
```python
ids = list(text.encode())  # start: one token per byte
```
- **Snippet explained:** Every string becomes a list of byte values in 0..255 before any merge.
- **Related terms:** [[Vocabulary size trade-off]], [[Pre-tokenizer]], [[Special tokens]]

### Vocabulary size trade-off
<!-- cat: llm -->
- **Technical definition:** A bigger vocabulary gives shorter sequences and better multilingual coverage but a larger embedding table (and output head) and under-trained rare tokens.
- **Plain English:** More dictionary entries: shorter sentences, heavier book.
- **Example / analogy:** Embedding is 27.6% of Qwen2.5-0.5B (151,936 x 896).
- **Snippet:**
```python
emb_params = vocab_size * d_model
```
- **Snippet explained:** Embedding parameter count.
- **Related terms:** [[Byte-level BPE (byte-pair encoding)]], [[Tied embeddings]], [[Pre-tokenizer]]

### Pre-tokenizer
<!-- cat: llm -->
- **Technical definition:** A regex or rule-based split of text (words, digits, spaces) applied before BPE so merges never cross those boundaries.
- **Plain English:** Cut the text into rough words first.
- **Example / analogy:** Qwen splits digits one by one; gpt2 groups them.
- **Snippet:**
```python
re.findall(r"\s?\S+|\s", text)
```
- **Snippet explained:** Splits into words with their leading space.
- **Related terms:** [[Byte-level BPE (byte-pair encoding)]], [[Special tokens]], [[Vocabulary size trade-off]]

### Special tokens
<!-- cat: llm -->
- **Technical definition:** Reserved vocabulary entries such as <|im_start|> or <|endoftext|> that mark roles and boundaries; each is a single id that is never produced by merging text.
- **Plain English:** Control codes in the token stream.
- **Example / analogy:** Chat models use them to wrap each turn.
- **Snippet:**
```python
tok.convert_tokens_to_ids('<|im_start|>')
```
- **Snippet explained:** Returns one integer id for the marker.
- **Related terms:** [[Byte-level BPE (byte-pair encoding)]], [[Pre-tokenizer]]

### Rotary position embedding (RoPE)
<!-- cat: llm -->
- **Technical definition:** Rotates each (even, odd) pair of query and key components by an angle pos * theta_i with theta_i = base^(-2i/d), so q.k depends only on the relative distance; no parameters; applied to q and k, not v.
- **Plain English:** Encode position by turning the vectors; the angle between them encodes distance.
- **Example / analogy:** Score is -0.960123 at offsets (5,2), (105,102) and (1005,1002).
- **Snippet:**
```python
q_rot = rope(q, pos); k_rot = rope(k, pos)
```
- **Snippet explained:** Rotate before the dot product (and before caching keys).
- **Related terms:** [[RoPE context extension (PI, NTK-aware, YaRN)]]

### RoPE context extension (PI, NTK-aware, YaRN)
<!-- cat: llm -->
- **Technical definition:** Ways to run a RoPE model beyond its training length: PI divides positions by the scale, NTK-aware raises the base, YaRN blends per frequency and rescales attention logits (0.1 ln s + 1).
- **Plain English:** Stretch the rotation clock so longer texts fit.
- **Example / analogy:** YaRN is used by gpt-oss (factor 32) and DeepSeek-V3 (40).
- **Snippet:**
```python
base_new = base * s ** (d / (d - 2))
```
- **Snippet explained:** NTK-aware base for scale s on head_dim d.
- **Related terms:** [[Rotary position embedding (RoPE)]], [[Context window]]

### ALiBi (attention with linear biases)
<!-- cat: llm -->
- **Technical definition:** Adds a head-specific slope times distance as a negative bias to attention scores instead of position embeddings; extrapolates to longer sequences.
- **Plain English:** Penalise far-away tokens a bit more for every step of distance.
- **Example / analogy:** Used in BLOOM and MPT.
- **Snippet:**
```python
scores = q @ k.T - slope * distance
```
- **Snippet explained:** Linear penalty added before softmax.
- **Related terms:** [[Rotary position embedding (RoPE)]], [[RoPE context extension (PI, NTK-aware, YaRN)]]

### Pre-norm vs post-norm
<!-- cat: llm -->
- **Technical definition:** Pre-norm computes x + f(norm(x)) keeping an identity residual path; post-norm computes norm(x + f(x)). Modern LLMs use pre-norm with a final norm.
- **Plain English:** Normalise before the sub-layer instead of after the sum.
- **Example / analogy:** Pre-norm trains deep stacks stably without long warm-up.
- **Snippet:**
```python
x = x + attn(rms_norm(x))
```
- **Snippet explained:** Residual add outside the normalised branch.
- **Related terms:** [[RMSNorm in decoders]], [[Residual stream]], [[QK-norm]]

### RMSNorm in decoders
<!-- cat: llm -->
- **Technical definition:** Scale-only normalisation x / sqrt(mean(x^2) + eps) * g with d parameters; in Llama-style blocks it appears twice per layer plus a final norm. See also the module 15 treatment.
- **Plain English:** LayerNorm without the centring.
- **Example / analogy:** Saves the mean computation and the bias vector.
- **Snippet:**
```python
y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * g
```
- **Snippet explained:** Root-mean-square scaling.
- **Related terms:** [[Pre-norm vs post-norm]], [[QK-norm]], [[SwiGLU feed-forward]]

### Residual stream
<!-- cat: llm -->
- **Technical definition:** The running sum x that every attention and FFN block reads from and adds to; width d_model (hidden_size).
- **Plain English:** The shared notebook all layers write into.
- **Example / analogy:** Final RMSNorm then lm_head read the stream at the end.
- **Snippet:**
```python
x = x + block_output
```
- **Snippet explained:** Each block adds its contribution.
- **Related terms:** [[Pre-norm vs post-norm]], [[RMSNorm in decoders]]

### QK-norm
<!-- cat: llm -->
- **Technical definition:** RMSNorm applied to queries and keys before the dot product to bound attention logits and stabilise training.
- **Plain English:** Keep the dot products from getting huge.
- **Example / analogy:** Max logit fell from 194.8 to 2.9 in the step-03 test.
- **Snippet:**
```python
q = rms_norm(q); k = rms_norm(k)
```
- **Snippet explained:** Normalise per head before scores.
- **Related terms:** [[RMSNorm in decoders]], [[Grouped-query attention (GQA)]], [[Pre-norm vs post-norm]]

### SwiGLU feed-forward
<!-- cat: llm -->
- **Technical definition:** FFN down(silu(gate(x)) * up(x)) with three matrices; hidden size about 8/3 d keeps parameters equal to a 4d GELU MLP (4,718,592 at d=768); Llama-3 uses 3.5d.
- **Plain English:** A gated MLP: one path decides how much of the other passes.
- **Example / analogy:** gate_proj, up_proj, down_proj in HF configs.
- **Snippet:**
```python
y = down(F.silu(gate(x)) * up(x))
```
- **Snippet explained:** Elementwise gate then projection.
- **Related terms:** [[RMSNorm in decoders]], [[Mixture of experts (MoE)]]

### KV cache (key-value cache, past_key_values)
<!-- cat: llm -->
- **Technical definition:** Stored keys and values of earlier tokens per layer so decoding computes one new token per step; bytes per token = 2 x layers x kv_heads x head_dim x bytes.
- **Plain English:** Remember what you already computed instead of rereading the whole text.
- **Example / analogy:** Llama-3-8B: 128 KB per token, 17.2 GB at 128k tokens in bf16.
- **Snippet:**
```python
bytes_per_token = 2 * layers * kv_heads * head_dim * 2
```
- **Snippet explained:** Factor 2 for K and V, 2 bytes for bf16.
- **Related terms:** [[Grouped-query attention (GQA)]], [[Multi-head latent attention (MLA)]], [[Sliding-window attention]]

### Grouped-query attention (GQA)
<!-- cat: llm -->
- **Technical definition:** h query heads share g key/value heads (g divides h); g = h is MHA, g = 1 is MQA. Cuts KV cache by h/g with small quality loss.
- **Plain English:** Several question-askers share one set of notes.
- **Example / analogy:** Llama-3-8B: 32 query heads, 8 KV heads.
- **Snippet:**
```python
k = k.repeat_interleave(h // g, dim=1)
```
- **Snippet explained:** Expands the g KV heads to h before attention; the cache stores only g.
- **Related terms:** [[KV cache (key-value cache, past_key_values)]], [[Multi-head latent attention (MLA)]], [[QK-norm]]

### Multi-head latent attention (MLA)
<!-- cat: llm -->
- **Technical definition:** DeepSeek attention that caches a low-rank latent (512) plus a small decoupled RoPE key (64) per token and rebuilds K and V by up-projections that can be absorbed into other matrices.
- **Plain English:** Store a compressed summary instead of all keys and values.
- **Example / analogy:** DeepSeek-V3 caches 576 values per layer per token.
- **Snippet:**
```python
c = W_down @ x   # cached; K = W_up_k @ c on demand
```
- **Snippet explained:** The latent is what goes in the cache.
- **Related terms:** [[Grouped-query attention (GQA)]], [[KV cache (key-value cache, past_key_values)]], [[Rotary position embedding (RoPE)]]

### Sliding-window attention
<!-- cat: llm -->
- **Technical definition:** A layer attends only to the last W tokens, capping compute and cache; information still spreads through depth. Often mixed with global layers (Gemma 3 5:1, gpt-oss alternating).
- **Plain English:** Only look at the last few pages.
- **Example / analogy:** gpt-oss uses window 128 in alternating layers.
- **Snippet:**
```python
mask = (i - j >= 0) & (i - j < W)
```
- **Snippet explained:** Causal band mask of width W.
- **Related terms:** [[KV cache (key-value cache, past_key_values)]], [[Attention sink]]

### Attention sink
<!-- cat: llm -->
- **Technical definition:** Early tokens receive large attention from all later tokens even when uninformative; evicting them breaks generation, so StreamingLLM keeps a few sink tokens plus a recent window.
- **Plain English:** The first token is a parking spot for spare attention.
- **Example / analogy:** GPT-2 puts 70-79% of attention on token 0 in middle layers.
- **Snippet:**
```python
keep = list(range(n_sink)) + list(range(t - window, t))
```
- **Snippet explained:** Positions retained in the cache.
- **Related terms:** [[Sliding-window attention]], [[KV cache (key-value cache, past_key_values)]]

### Online softmax
<!-- cat: llm -->
- **Technical definition:** Computes softmax over chunks keeping a running max m and sum l: m_new = max(m, chunk max); l = l exp(m - m_new) + sum exp(chunk - m_new); accumulator rescaled by the same factor.
- **Plain English:** Update the softmax as new pieces arrive without seeing everything.
- **Example / analogy:** Basis of FlashAttention.
- **Snippet:**
```python
l = l * np.exp(m - m_new) + np.exp(chunk - m_new).sum()
```
- **Snippet explained:** Rescales the old sum to the new maximum.
- **Related terms:** [[FlashAttention (tiled IO-aware attention)]], [[SDPA (scaled dot-product attention)]]

### FlashAttention (tiled IO-aware attention)
<!-- cat: llm -->
- **Technical definition:** Exact attention computed in tiles with online softmax so the N x N score matrix is never written to GPU HBM; same FLOPs, far less memory traffic, O(N) memory.
- **Plain English:** Same answer, fewer trips to slow memory.
- **Example / analogy:** Tiled vs naive error 5.6e-16.
- **Snippet:**
```python
F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
- **Snippet explained:** PyTorch dispatches to a fused flash kernel when available.
- **Related terms:** [[Online softmax]], [[SDPA (scaled dot-product attention)]], [[KV cache (key-value cache, past_key_values)]]

### SDPA (scaled dot-product attention)
<!-- cat: llm -->
- **Technical definition:** PyTorch API torch.nn.functional.scaled_dot_product_attention that picks a FlashAttention, memory-efficient, cuDNN or plain math backend.
- **Plain English:** One function; PyTorch chooses the fastest kernel.
- **Example / analogy:** Not the same as PagedAttention, which manages KV memory.
- **Snippet:**
```python
o = F.scaled_dot_product_attention(q, k, v)
```
- **Snippet explained:** Computes softmax(qk^T/sqrt(d)) v.
- **Related terms:** [[FlashAttention (tiled IO-aware attention)]], [[KV cache (key-value cache, past_key_values)]]

### Mixture of experts (MoE)
<!-- cat: llm -->
- **Technical definition:** FFN replaced by E expert MLPs plus a router that picks top-k per token (and optional shared experts); total parameters set memory, active parameters set compute.
- **Plain English:** Many specialists, only a few consulted per word.
- **Example / analogy:** DeepSeek-V3: 671B total, 37B active.
- **Snippet:**
```python
top_idx = probs.topk(k, dim=-1).indices
```
- **Snippet explained:** Chooses k experts per token.
- **Related terms:** [[Load balancing (auxiliary loss, bias)]], [[Total vs active parameters]], [[SwiGLU feed-forward]]

### Load balancing (auxiliary loss, bias)
<!-- cat: llm -->
- **Technical definition:** Keeping experts evenly used: Switch aux loss E sum f_e P_e (1.0 balanced) or DeepSeek-V3's selection-only bias nudged by -gamma sign(load - mean).
- **Plain English:** Stop a few experts from doing all the work.
- **Example / analogy:** Bias method reached max/min load 1.29x without changing the router.
- **Snippet:**
```python
bias -= gamma * torch.sign(load - load.mean())
```
- **Snippet explained:** Overloaded experts get lower selection scores.
- **Related terms:** [[Mixture of experts (MoE)]], [[Total vs active parameters]]

### Total vs active parameters
<!-- cat: llm -->
- **Technical definition:** Total counts all weights (memory); active counts the weights used per token (compute). 'A37B' in a name means 37B active.
- **Plain English:** Size of the whole library vs pages read per question.
- **Example / analogy:** Kimi K2: 1T total, 32B active.
- **Snippet:**
```python
active = router + k * expert + shared
```
- **Snippet explained:** Per-token active parameters of one MoE layer.
- **Related terms:** [[Mixture of experts (MoE)]], [[Load balancing (auxiliary loss, bias)]]

### Tied embeddings
<!-- cat: llm -->
- **Technical definition:** The output head reuses the input embedding matrix (tie_word_embeddings=True), saving vocab x d parameters.
- **Plain English:** One table for reading and writing tokens.
- **Example / analogy:** Qwen2.5-0.5B ties them; counted once in 494,032,768.
- **Snippet:**
```python
self.lm_head.weight = self.embed.weight
```
- **Snippet explained:** Both modules point to the same tensor.
- **Related terms:** [[Vocabulary size trade-off]], [[Decoder config fields]]

### Decoder config fields
<!-- cat: llm -->
- **Technical definition:** config.json entries of a Llama-family model: hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, intermediate_size, vocab_size, rope_theta, rms_norm_eps, tie_word_embeddings, max_position_embeddings.
- **Plain English:** The spec sheet that fully determines the architecture.
- **Example / analogy:** Parameters follow: V d + L(attn + 3 d ff + 2 d) + d.
- **Snippet:**
```python
json.load(open('config.json'))['num_key_value_heads']
```
- **Snippet explained:** Reads one switch from the config.
- **Related terms:** [[Tied embeddings]], [[Grouped-query attention (GQA)]], [[SwiGLU feed-forward]]

### Context window
<!-- cat: llm -->
- **Technical definition:** Maximum number of tokens configured (max_position_embeddings); usable quality may be lower than the configured length.
- **Plain English:** How much text the model can see at once.
- **Example / analogy:** Position scaling moves the limit; cache memory grows linearly with it.
- **Snippet:**
```python
cfg.max_position_embeddings
```
- **Snippet explained:** Configured maximum length.
- **Related terms:** [[RoPE context extension (PI, NTK-aware, YaRN)]], [[KV cache (key-value cache, past_key_values)]]

### Prefill vs decode
<!-- cat: llm -->
- **Technical definition:** Prefill processes the whole prompt in parallel (compute-bound); decode generates one token per step reading all weights and the whole cache (memory-bound).
- **Plain English:** Reading the question vs writing the answer word by word.
- **Example / analogy:** KV cache removes recomputation in decode but not the memory reads.
- **Snippet:**
```python
logits = model(prompt, cache); logits = model(next_token, cache)
```
- **Snippet explained:** First call prefill, later calls decode.
- **Related terms:** [[KV cache (key-value cache, past_key_values)]], [[FlashAttention (tiled IO-aware attention)]]

### Reasoning model (test-time compute)
<!-- cat: llm -->
- **Technical definition:** A model with an ordinary architecture post-trained with RL to produce long chains of thought; cost moves to inference tokens.
- **Plain English:** Same brain, trained to think longer before answering.
- **Example / analogy:** DeepSeek-R1 is the V3 architecture plus RL.
- **Snippet:**
```python
answer = model.generate(prompt, max_new_tokens=8192)
```
- **Snippet explained:** Long outputs raise decode time and cache size.
- **Related terms:** [[Prefill vs decode]], [[KV cache (key-value cache, past_key_values)]]

### Hybrid attention (linear attention / SSM layers)
<!-- cat: llm -->
- **Technical definition:** Mix of full-attention layers with layers keeping a fixed-size state (Mamba, Gated DeltaNet); only the attention layers grow a cache.
- **Plain English:** A few full-memory layers, many cheap ones.
- **Example / analogy:** Qwen3.5 uses 3 Gated DeltaNet layers per full-attention layer.
- **Snippet:**
```python
state = decay * state + k.T @ v   # fixed size
```
- **Snippet explained:** Constant memory per token.
- **Related terms:** [[Sliding-window attention]], [[KV cache (key-value cache, past_key_values)]]

### Multi-token prediction (MTP)
<!-- cat: llm -->
- **Technical definition:** Training objective and module that predicts several future tokens; at inference its extra head can serve as a draft for speculative decoding (see module 21 step 13).
- **Plain English:** Learn to guess a few tokens ahead.
- **Example / analogy:** Used in DeepSeek-V3.
- **Snippet:**
```python
loss = sum(ce(head_k(h), y_shift_k) for k in range(K))
```
- **Snippet explained:** Sum of shifted-target losses.
- **Related terms:** [[Mixture of experts (MoE)]], [[Reasoning model (test-time compute)]]
