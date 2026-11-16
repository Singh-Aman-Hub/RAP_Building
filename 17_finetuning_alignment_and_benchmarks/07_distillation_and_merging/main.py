"""
GOAL:  (A) Logit knowledge distillation: teacher gpt2 -> student distilgpt2 with KL at temperature T, compared with
       hard-label training. (B) Model merging on small tensors: linear average, task arithmetic, SLERP, TIES, written by hand
       and compared on a toy where two "task vectors" conflict. (C) Merge two real LoRA adapters with peft.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/07_distillation_and_merging/main.py
EXPECTED OUTPUT (shortened):
    teacher entropy at T=1 vs T=2, student held-out CE / KL-to-teacher before and after three recipes
    merging table on a toy: cosine of merged vector with each task vector for linear / task-arith / SLERP / TIES
    LoRA adapters A and B: loss on both tasks for A, B, linear merge, TIES merge
NEEDS: torch, transformers, peft; cached gpt2 + distilgpt2; reads ../../06_llm_training_and_lora/data/tiny_corpus.txt.  ~1-2 min.
"""
import copy
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import make_preference_data, make_sentiment_data   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
TEMP = float(os.environ.get("TEMP", 2.0))
STEPS = int(os.environ.get("STEPS", 24))
torch.manual_seed(SEED)

# ## 1. Data: tiny corpus split into train and held-out lines
CORPUS = Path(__file__).resolve().parents[2] / "06_llm_training_and_lora" / "data" / "tiny_corpus.txt"
lines = [l.strip() for l in CORPUS.read_text().splitlines() if l.strip()]
train_lines, test_lines = lines[:30], lines[30:]
tok = AutoTokenizer.from_pretrained("gpt2")                # gpt2 and distilgpt2 share one tokenizer (50257 tokens)
tok.pad_token = tok.eos_token

def batch(texts):
    enc = tok(texts, return_tensors="pt", padding=True)    # input_ids, attention_mask: (B, L)
    return enc.input_ids, enc.attention_mask

# ## 2. Teacher and the soft-target idea
teacher = AutoModelForCausalLM.from_pretrained("gpt2").eval()
x, m = batch(test_lines)
with torch.no_grad():
    tl = teacher(x, attention_mask=m).logits[:, :-1]       # (B, L-1, vocab)
def entropy(logits, T):
    p = F.softmax(logits / T, -1)
    return -(p * torch.log(p + 1e-12)).sum(-1)             # (B, L-1)
mask = m[:, 1:].bool()
print(f"teacher mean entropy of next-token distribution: T=1 {entropy(tl, 1.0)[mask].mean():.2f} nats | T={TEMP} {entropy(tl, TEMP)[mask].mean():.2f} nats")

# ## 3. The distillation loss
def kd_loss(student_logits, teacher_logits, labels, mask, T, alpha):
    """alpha * T^2 * KL(teacher_T || student_T) + (1 - alpha) * hard cross-entropy."""
    s = F.log_softmax(student_logits / T, -1)
    t = F.softmax(teacher_logits / T, -1)
    kl = (t * (torch.log(t + 1e-12) - s)).sum(-1)          # (B, L-1) KL per position
    kl = (kl * mask).sum() / mask.sum()
    ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1), reduction="none")
    ce = (ce * mask.reshape(-1)).sum() / mask.sum()
    return alpha * (T ** 2) * kl + (1 - alpha) * ce       # T^2 keeps gradient magnitude comparable across temperatures

# ## 4. Evaluate a student: held-out CE and KL to the teacher at T=1
@torch.no_grad()
def evaluate(student):
    student.eval()
    sl = student(x, attention_mask=m).logits[:, :-1]
    ce = F.cross_entropy(sl.reshape(-1, sl.size(-1)), x[:, 1:].reshape(-1), reduction="none")
    ce = (ce * mask.reshape(-1)).sum() / mask.sum()
    kl = (F.softmax(tl, -1) * (F.log_softmax(tl, -1) - F.log_softmax(sl, -1))).sum(-1)[mask].mean()
    return ce.item(), kl.item()

def distill(T, alpha, label):
    torch.manual_seed(SEED)
    student = AutoModelForCausalLM.from_pretrained("distilgpt2")
    opt = torch.optim.AdamW(student.parameters(), lr=2e-5)
    rng = random.Random(SEED)
    for step in range(STEPS):
        texts = rng.sample(train_lines, 6)
        xb, mb = batch(texts)
        with torch.no_grad():
            tlogits = teacher(xb, attention_mask=mb).logits[:, :-1]
        student.train()
        slogits = student(xb, attention_mask=mb).logits[:, :-1]
        loss = kd_loss(slogits, tlogits, xb[:, 1:], mb[:, 1:].float(), T, alpha)
        opt.zero_grad(); loss.backward(); opt.step()
    ce, kl = evaluate(student)
    print(f"{label:<34} held-out CE {ce:.3f}  KL(teacher||student) {kl:.3f}")

print(f"\nstudent before training: held-out CE/KL = {tuple(round(v, 3) for v in evaluate(AutoModelForCausalLM.from_pretrained('distilgpt2')))}")
print(f"teacher itself:          held-out CE = {evaluate(teacher)[0]:.3f}")
distill(1.0, 0.0, "hard labels only (alpha=0)")
distill(1.0, 1.0, "soft targets T=1 (alpha=1)")
distill(TEMP, 1.0, f"soft targets T={TEMP} (alpha=1)")
distill(TEMP, 0.5, f"mixed T={TEMP} (alpha=0.5)")

# ## 5. Merging on small tensors: four recipes
def slerp(a, b, t):
    """Spherical interpolation between two flat vectors a and b."""
    an, bn = a / a.norm(), b / b.norm()
    omega = torch.acos(torch.clamp((an * bn).sum(), -1 + 1e-7, 1 - 1e-7))      # angle between them
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)

def ties(taus, density=0.5):
    """TIES: trim small values per task, elect a sign per coordinate, average only agreeing entries."""
    trimmed = []
    for t in taus:
        k = max(1, int(len(t) * density))
        thr = t.abs().topk(k).values.min()                                    # keep the top-`density` by magnitude
        trimmed.append(torch.where(t.abs() >= thr, t, torch.zeros_like(t)))
    stack = torch.stack(trimmed)                                              # (n_tasks, D)
    sign = torch.sign(stack.sum(0))                                           # elected sign per coordinate (by total mass)
    agree = (torch.sign(stack) == sign) & (stack != 0)                        # entries that agree with the elected sign
    return (stack * agree).sum(0) / agree.sum(0).clamp(min=1)                 # disjoint mean over agreeing entries

D = 1000
base = torch.zeros(D)
tau1 = torch.randn(D) * (torch.rand(D) < 0.3)                                 # task 1 edits ~30% of the weights
tau2 = torch.randn(D) * (torch.rand(D) < 0.3)                                 # task 2 edits a different ~30%
tau2[:150] = -tau1[:150] + 0.05 * torch.randn(150)                            # ...and conflicts with task 1 on 150 coordinates
cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
print("\nmerge toy: D=1000, tau1/tau2 conflict on the first 150 coordinates")
print(f"{'method':<28}{'cos(merged,tau1)':>18}{'cos(merged,tau2)':>18}")
for name, delta in {
    "linear average":          (tau1 + tau2) / 2,
    "task arithmetic (lam=1)": tau1 + tau2,
    "SLERP t=0.5":             slerp(tau1, tau2, 0.5),
    "TIES density 0.5":        ties([tau1, tau2], 0.5),
}.items():
    merged = base + delta
    print(f"{name:<28}{cos(merged, tau1):>18.3f}{cos(merged, tau2):>18.3f}")
print(f"{'(reference) cos(tau1,tau2)':<28}{cos(tau1, tau2):>18.3f}")

# ## 6. Merge two real LoRA adapters trained on different tasks
from peft import LoraConfig, get_peft_model       # noqa: E402
ptok = AutoTokenizer.from_pretrained("distilgpt2"); ptok.pad_token = ptok.eos_token
sent, _ = make_sentiment_data(); pref, _ = make_preference_data()
task_a = [r["prompt"] + r["completion"] for r in sent]                         # task A: sentiment words
task_b = [p["prompt"] + p["chosen"] for p in pref]                             # task B: polite replies

def lm_loss(model, texts):
    enc = ptok(texts, return_tensors="pt", padding=True)
    labels = enc.input_ids.masked_fill(enc.attention_mask == 0, -100)
    return model(**enc, labels=labels).loss

cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True, task_type="CAUSAL_LM")
pm = get_peft_model(AutoModelForCausalLM.from_pretrained("distilgpt2"), cfg, adapter_name="a")
pm.add_adapter("b", cfg)
for name, texts in (("a", task_a), ("b", task_b)):
    pm.set_adapter(name)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-3)
    for _ in range(STEPS):
        loss = lm_loss(pm, random.Random(_).sample(texts, 8)); opt.zero_grad(); loss.backward(); opt.step()

def report(tag):
    pm.eval()
    with torch.no_grad():
        print(f"{tag:<22} loss on A (sentiment) {lm_loss(pm, task_a[:30]).item():.3f}   loss on B (polite) {lm_loss(pm, task_b[:30]).item():.3f}")
with pm.disable_adapter():
    print("\nLoRA adapters (same rank 8):"); report("base, no adapter")
for name in ("a", "b"):
    pm.set_adapter(name); report(f"adapter {name} alone")
pm.add_weighted_adapter(["a", "b"], [0.5, 0.5], adapter_name="lin", combination_type="linear")
pm.set_adapter("lin"); report("linear merge 0.5/0.5")
pm.add_weighted_adapter(["a", "b"], [1.0, 1.0], adapter_name="ties", combination_type="ties", density=0.5)
pm.set_adapter("ties"); report("TIES merge density .5")

# TRY IT: TEMP=1 python main.py vs TEMP=4 python main.py; change the TIES density to 0.2 in the toy.
# BREAK IT: call pm.add_weighted_adapter with adapters of DIFFERENT ranks and combination_type="linear" and read the error.
