"""Exercise A3 solution: plan a sharded checkpoint (greedy fill) and build the weight_map index for a Llama-70B-like model."""
import json

def plan(sizes, max_bytes):
    shards, cur, cur_b = [], {}, 0
    for name, b in sizes.items():
        if cur and cur_b + b > max_bytes:               # start a new shard, never split a tensor
            shards.append(cur); cur, cur_b = {}, 0
        cur[name] = b; cur_b += b
    if cur: shards.append(cur)
    return shards

layers, d, ffn, vocab, kvh, hd, heads = 80, 8192, 28672, 128256, 8, 128, 64
bp = 2                                                   # bf16
sizes = {"model.embed_tokens.weight": vocab * d * bp}
for i in range(layers):
    p = f"model.layers.{i}."
    sizes[p + "attn.q_proj.weight"] = heads * hd * d * bp
    sizes[p + "attn.k_proj.weight"] = kvh * hd * d * bp
    sizes[p + "attn.v_proj.weight"] = kvh * hd * d * bp
    sizes[p + "attn.o_proj.weight"] = d * heads * hd * bp
    for n in ("gate", "up", "down"):
        sizes[p + f"mlp.{n}_proj.weight"] = d * ffn * bp
    sizes[p + "norms"] = 2 * d * bp
sizes["lm_head.weight"] = vocab * d * bp
total = sum(sizes.values())
shards = plan(sizes, 5 * 10**9)
wm = {t: f"model-{i+1:05d}-of-{len(shards):05d}.safetensors" for i, s in enumerate(shards) for t in s}
print(f"{len(sizes)} tensors, {total/1e9:.1f} GB, {len(shards)} shards of <= 5 GB; biggest {max(sum(s.values()) for s in shards)/1e9:.2f} GB")
print("index.json sample:", json.dumps(dict(list(wm.items())[:2]), indent=0)[:140].replace("\n", " "), "...")
assert sum(sum(s.values()) for s in shards) == total and len(wm) == len(sizes)
assert max(sum(s.values()) for s in shards) <= 5 * 10**9
print("every tensor appears in exactly one shard and the index covers all of them: OK")
