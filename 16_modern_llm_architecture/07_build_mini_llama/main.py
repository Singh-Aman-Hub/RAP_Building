"""
GOAL:  Assemble a small Llama-style decoder from clean classes (ModelConfig, RMSNorm, RotaryEmbedding, Attention with
       GQA, SwiGLU MLP, DecoderBlock, Transformer, KVCache), train it for a few hundred steps on a bundled text
       file, then generate WITH and WITHOUT the KV cache and verify the outputs are identical.

RUN (from this step folder):  python main.py
RUN (from repo root):         .venv/bin/python 16_modern_llm_architecture/07_build_mini_llama/main.py
EXPECTED OUTPUT (shortened):
    parameters 738,432 (embedding counted once, tied)
    step   0 | loss 5.6 ... step 299 | loss ~2.0 ...
    with cache == without cache: True
NEEDS: torch only (the text is corpus.txt in this folder, 128 KB of this repo's markdown).  ~15-20 s on CPU (varies by machine).
"""
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ---- 1. ModelConfig: every architecture choice lives in one dataclass ---------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 256            # byte-level "tokenizer": each byte is a token (see step 01 for real BPE)
    d_model: int = 128               # also called hidden_size / n_embd
    n_layers: int = 4
    n_heads: int = 8                 # query heads
    n_kv_heads: int = 2              # GQA: 4 query heads share each K/V head
    ffn_hidden: int = 352            # ~ 8/3 * d_model rounded up to a multiple of 32
    max_seq: int = 256               # max_position_embeddings
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True      # output head shares the embedding matrix

    @property
    def head_dim(self):
        return self.d_model // self.n_heads


# ---- 2. RMSNorm (step 03) ------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ---- 3. RotaryEmbedding (step 02), "rotate-half" layout as in Hugging Face Llama --------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq, base):
        super().__init__()
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)     # (head_dim/2,)
        ang = torch.arange(max_seq).float()[:, None] * inv_freq[None, :]          # (max_seq, head_dim/2)
        ang = torch.cat([ang, ang], dim=-1)                                       # (max_seq, head_dim): halves share angles
        self.register_buffer("cos", ang.cos(), persistent=False)                  # buffers move with .to(device), not trained
        self.register_buffer("sin", ang.sin(), persistent=False)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, start_pos):                                           # q: (b, heads, t, hd)  k: (b, kv_heads, t, hd)
        t = q.shape[2]
        cos = self.cos[start_pos:start_pos + t]                                   # absolute positions of THESE tokens
        sin = self.sin[start_pos:start_pos + t]
        return q * cos + self.rotate_half(q) * sin, k * cos + self.rotate_half(k) * sin


# ---- 4. KVCache: per layer, K and V of all tokens seen so far ---------------------------------------------------------------
class KVCache:
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def update(self, layer, k_new, v_new):                                        # k_new: (b, kv_heads, t_new, hd)
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)              # append along the sequence axis
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)
        return self.k[layer], self.v[layer]

    def seq_len(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


# ---- 5. Attention with grouped-query attention (step 04) -----------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.cfg, self.layer_idx = cfg, layer_idx
        hd = cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * hd, cfg.d_model, bias=False)

    def forward(self, x, rope, start, cache=None):                                # x: (b, t, d_model); start = tokens already cached
        b, t, _ = x.shape
        c = self.cfg
        q = self.wq(x).view(b, t, c.n_heads, c.head_dim).transpose(1, 2)          # (b, heads, t, hd)
        k = self.wk(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)       # (b, kv_heads, t, hd)
        v = self.wv(x).view(b, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
        q, k = rope(q, k, start)                                                  # rotate BEFORE caching: cached keys keep their positions
        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)                             # (b, kv_heads, start + t, hd)
        k = k.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)                 # GQA: share each K/V head with its query group
        v = v.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
        # prefill / training: causal mask. Decoding one token: it may see everything cached, so no mask.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=(start == 0 and t > 1))
        return self.wo(o.transpose(1, 2).reshape(b, t, -1))                       # merge heads: (b, t, d_model)


# ---- 6. SwiGLU MLP (step 03) ------------------------------------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---- 7. DecoderBlock: pre-norm residual stream ----------------------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(cfg.d_model, cfg.norm_eps), Attention(cfg, layer_idx)
        self.mlp_norm, self.mlp = RMSNorm(cfg.d_model, cfg.norm_eps), MLP(cfg)

    def forward(self, x, rope, start, cache=None):
        x = x + self.attn(self.attn_norm(x), rope, start, cache)                         # x + f(norm(x))
        return x + self.mlp(self.mlp_norm(x))


# ---- 8. Transformer: embeddings -> blocks -> final norm -> logits -------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq, cfg.rope_base)
        self.blocks = nn.ModuleList(DecoderBlock(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight                               # one matrix, two roles
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, cache=None):                                           # ids: (b, t)
        x = self.embed(ids)                                                       # (b, t, d_model)
        start = cache.seq_len() if cache is not None else 0                       # read ONCE: layer 0 updates the cache before layer 1 runs
        for blk in self.blocks:
            x = blk(x, self.rope, start, cache)
        return self.lm_head(self.final_norm(x))                                   # (b, t, vocab)

    @torch.no_grad()
    def generate(self, ids, n_new, use_cache=True):                               # greedy decoding
        cache = KVCache(self.cfg.n_layers) if use_cache else None
        out = ids.clone()
        inp = ids
        for _ in range(n_new):
            logits = self(inp, cache)                                             # with cache: inp is only the NEW token(s)
            nxt = logits[:, -1].argmax(-1, keepdim=True)                          # (b, 1)
            out = torch.cat([out, nxt], dim=1)
            inp = nxt if use_cache else out                                       # no cache: re-feed the WHOLE sequence
        return out


def main():
    """Train, generate, verify. Wrapped in a function so step 08 can import the classes without training."""
    # ---- 9. Data: bytes of corpus.txt ------------------------------------------------------------------------------------------------------------
    cfg = ModelConfig()
    raw = (Path(__file__).parent / "corpus.txt").read_bytes()
    data = torch.tensor(list(raw), dtype=torch.long)                                  # (n_bytes,) values 0..255
    split = int(0.95 * len(data))
    train_data, val_data = data[:split], data[split:]
    print(f"corpus {len(raw):,} bytes -> train {len(train_data):,} | val {len(val_data):,} tokens (vocab {cfg.vocab_size})")


    def get_batch(d, batch=16, seq=128):
        ix = torch.randint(0, len(d) - seq - 1, (batch,))
        x = torch.stack([d[i:i + seq] for i in ix])                                   # (batch, seq) inputs
        y = torch.stack([d[i + 1:i + seq + 1] for i in ix])                           # targets: inputs shifted by one
        return x, y


    model = Transformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())                             # parameters() de-duplicates the tied matrix
    print(f"parameters {n_params:,} (embedding {cfg.vocab_size * cfg.d_model:,} counted once because tied)")
    print("per block:", sum(p.numel() for p in model.blocks[0].parameters()), "| attention", sum(p.numel() for p in model.blocks[0].attn.parameters()),
          "| mlp", sum(p.numel() for p in model.blocks[0].mlp.parameters()))
    print(f"initial loss {F.cross_entropy(model(get_batch(val_data)[0]).flatten(0, 1), get_batch(val_data)[1].flatten()).item():.3f} (ln 256 = {math.log(256):.3f})")

    # ---- 10. Training loop (see module 06 step 02 for the anatomy) ---------------------------------------------------------------------------------
    STEPS, LR = 300, 3e-3
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    t0 = time.time()
    for step in range(STEPS):
        lr = LR * min(1, (step + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))   # warm-up then cosine decay
        for g in opt.param_groups:
            g["lr"] = lr
        xb, yb = get_batch(train_data)
        loss = F.cross_entropy(model(xb).flatten(0, 1), yb.flatten())                 # (b*t, vocab) vs (b*t,)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:>3} | lr {lr:.4f} | loss {loss.item():.3f}")
    model.eval()
    with torch.no_grad():
        vx, vy = get_batch(val_data, batch=64)
        vloss = F.cross_entropy(model(vx).flatten(0, 1), vy.flatten()).item()
    print(f"train time {time.time() - t0:.0f}s | val loss {vloss:.3f} nats/byte = {vloss / math.log(2):.2f} bits/byte | perplexity {math.exp(vloss):.2f}")

    # ---- 11. Generate with and without the KV cache and compare -------------------------------------------------------------------------------------------
    prompt = torch.tensor([list("The attention".encode())])                           # (1, 13) bytes
    N_NEW = 200
    t0 = time.time()
    out_cache = model.generate(prompt, N_NEW, use_cache=True)
    t_cache = time.time() - t0
    t0 = time.time()
    out_nocache = model.generate(prompt, N_NEW, use_cache=False)
    t_nocache = time.time() - t0
    print("\n--- generated text (greedy, with cache) ---")
    print(bytes(out_cache[0].tolist()).decode("utf-8", errors="replace"))
    print("---")
    print("with cache == without cache:", torch.equal(out_cache, out_nocache))
    print(f"time: cache {t_cache:.2f}s | no cache {t_nocache:.2f}s | speedup {t_nocache / t_cache:.1f}x")

    # also compare the LOGITS of one step: full forward vs prefill + one cached step
    with torch.no_grad():
        full = model(out_cache[:, :20])[:, -1]
        cache = KVCache(cfg.n_layers)
        model(out_cache[:, :19], cache)
        step_logits = model(out_cache[:, 19:20], cache)[:, -1]
    print("max |logit diff| full vs cached step:", (full - step_logits).abs().max().item())
    print(f"cache after 20 tokens: {cache.nbytes()} bytes = {cache.nbytes() / 20:.0f} bytes/token (formula 2*layers*kv_heads*hd*4 = {2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 4})")

    # ---- 12. Sanity checks that prove the pieces are wired correctly ------------------------------------------------------------------------------------------
    with torch.no_grad():
        a = torch.randint(0, 256, (1, 30))
        b = a.clone()
        b[0, 20:] = torch.randint(0, 256, (10,))                                      # change only the LAST 10 tokens
        la, lb = model(a), model(b)
    print("causality: logits at positions < 20 unchanged when later tokens change:", torch.allclose(la[:, :20], lb[:, :20], atol=1e-5))
    print("tied weights share memory:", model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr())


if __name__ == "__main__":
    main()

# TRY IT: set n_kv_heads=8 (MHA) or 1 (MQA) in ModelConfig and compare the parameter count and the cache bytes/token.
# BREAK IT: in Attention.forward move `q, k = rope(q, k, start)` AFTER cache.update(...). The "with cache == without cache"
#           check becomes False: cached keys would be re-rotated at the wrong positions.
