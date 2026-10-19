"""Solution ex04: where do a model's parameters live? (embeddings vs attention vs MLP vs the rest)
RUN: .venv/bin/python 04_llm_fundamentals/solutions/ex06_param_breakdown.py"""
from collections import defaultdict
from transformers import AutoModelForCausalLM


def breakdown(model):
    groups = defaultdict(int)                               # defaultdict(int): missing keys start at 0
    for name, p in model.named_parameters():
        if "wte" in name or "wpe" in name:
            groups["embeddings (token+position)"] += p.numel()
        elif ".attn." in name:
            groups["attention"] += p.numel()
        elif ".mlp." in name:
            groups["mlp (feed-forward)"] += p.numel()
        else:
            groups["layer norms / other"] += p.numel()
    return groups


if __name__ == "__main__":
    for name in ("distilgpt2", "gpt2"):
        m = AutoModelForCausalLM.from_pretrained(name)
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{name}: {total:,} parameters")
        for g, n in sorted(breakdown(m).items(), key=lambda kv: -kv[1]):
            print(f"  {g:30} {n:>12,}  {100 * n / total:5.1f}%")
