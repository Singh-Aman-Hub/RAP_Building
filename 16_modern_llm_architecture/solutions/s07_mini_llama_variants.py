"""
GOAL:  Solutions for step 07: n_kv_heads 8/2/1 (params and cache bytes), temperature sampling with cached ==
       uncached under a fixed seed, and the two BREAK IT bugs (RoPE after cache update; `start` read per layer).
       Uses the UNTRAINED model: these properties do not depend on training. (Training 600 vs 300 steps is not run
       here: run step 07 with STEPS edited, about 1-2 minutes each; expect a lower validation loss and more
       coherent but still byte-level, repo-flavoured text.)

RUN (from this folder):  python s07_mini_llama_variants.py
RUN (from repo root):    .venv/bin/python 16_modern_llm_architecture/solutions/s07_mini_llama_variants.py
NEEDS: torch (CPU), imports the classes of ../07_build_mini_llama/main.py. ~5 s.
"""
import importlib.util
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("mini", Path(__file__).parent.parent / "07_build_mini_llama" / "main.py")
mini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mini)                 # defines classes; main() is guarded by __name__ so nothing trains
ModelConfig, Transformer, KVCache = mini.ModelConfig, mini.Transformer, mini.KVCache

# 1. n_kv_heads
base = ModelConfig()
print(f"{'n_kv_heads':>10}{'params':>10}{'cache bytes/token (fp32)':>26}")
for kv in (8, 2, 1):
    cfg = replace(base, n_kv_heads=kv)
    m = Transformer(cfg)
    print(f"{kv:>10}{sum(p.numel() for p in m.parameters()):>10,}{2 * cfg.n_layers * kv * cfg.head_dim * 4:>26,}")

# 2. temperature sampling, cached vs uncached
torch.manual_seed(0)
model = Transformer(base).eval()
prompt = torch.tensor([list(b"The attention")])


@torch.no_grad()
def sample(model, ids, n_new, T, use_cache, seed):
    g = torch.Generator().manual_seed(seed)
    cache = KVCache(model.cfg.n_layers) if use_cache else None
    out, inp = ids.clone(), ids
    for _ in range(n_new):
        logits = model(inp, cache)[:, -1] / T
        nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=g)
        out = torch.cat([out, nxt], 1)
        inp = nxt if use_cache else out
    return out


a, b = sample(model, prompt, 60, 1.0, True, 7), sample(model, prompt, 60, 1.0, False, 7)
print("sampled cached == uncached with the same seed:", torch.equal(a, b),
      "(true as long as the logits agree to float precision: the random numbers are drawn identically)")

# 3. BREAK IT: rotate AFTER cache.update (cached keys get re-rotated every step)
Attn = mini.Attention
orig = Attn.forward


def bad_forward(self, x, rope, start, cache=None):
    b_, t, _ = x.shape
    c = self.cfg
    q = self.wq(x).view(b_, t, c.n_heads, c.head_dim).transpose(1, 2)
    k = self.wk(x).view(b_, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
    v = self.wv(x).view(b_, t, c.n_kv_heads, c.head_dim).transpose(1, 2)
    if cache is not None:
        k, v = cache.update(self.layer_idx, k, v)        # cache stores UNROTATED keys ...
    q, k = rope(q, k, start)                             # ... and rope() assumes k starts at `start`, so old keys get wrong positions
    k = k.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
    v = v.repeat_interleave(c.n_heads // c.n_kv_heads, dim=1)
    o = F.scaled_dot_product_attention(q, k, v, is_causal=(start == 0 and t > 1))
    return self.wo(o.transpose(1, 2).reshape(b_, t, -1))


def cached_vs_full(model, ids):
    with torch.no_grad():
        full = model(ids)[:, -1]
        cache = KVCache(model.cfg.n_layers)
        model(ids[:, :-1], cache)
        step = model(ids[:, -1:], cache)[:, -1]
    return (full - step).abs().max().item()


ids = torch.randint(0, 256, (1, 20))
print(f"correct code : max |logit diff| full vs cached step = {cached_vs_full(model, ids):.2e}")
Attn.forward = bad_forward
print(f"rope after cache.update: max |logit diff| = {cached_vs_full(model, ids):.2e}  (prefill is OK, decode is wrong)")
Attn.forward = orig
print("BREAK IT 2: reading start from the cache length INSIDE each layer fails because layer 0 has already appended the "
      "new token when layer 1 runs, so layers >= 1 would see start = old_len + t. The cached-vs-uncached equality test fails; that is why Transformer.forward reads it once.")
