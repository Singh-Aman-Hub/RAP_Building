"""
GOAL:  The full loop on a laptop: take distilgpt2, fine-tune it with LoRA (SFT, completion-only loss) on a tiny sentiment task,
       evaluate BEFORE and AFTER with the step-08 harness (evalkit.py) and print a report card: metric, before, after, delta, 95% CI.
       Then check for regressions: did general knowledge (a 30-question MCQ probe) and general-text perplexity get worse?
       Run a gentle and an aggressive recipe to see catastrophic forgetting appear.

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/11_end_to_end_report_card/main.py
EXPECTED OUTPUT (shortened):
    two report cards (gentle LoRA, aggressive LoRA): task accuracy up a lot, CI shown; general-MCQ and perplexity rows show the cost
NEEDS: torch, transformers, peft; cached distilgpt2.  ~1-2 minutes on CPU.
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import make_sentiment_data   # noqa: E402
from evalkit import GENERAL_MCQ, bootstrap_ci, mcq_accuracy, paired_bootstrap, perplexity   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
torch.manual_seed(SEED)
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token
train, test = make_sentiment_data()

# ## 1. Evaluation items: the task (held-out reviews) and two regression probes (general MCQ, general text)
general_text = [l.strip() for l in (ROOT.parent / "06_llm_training_and_lora" / "data" / "tiny_corpus.txt").read_text().splitlines() if l.strip()]

def task_correct(model):
    """1/0 per held-out review: is log P(' positive') vs log P(' negative') ranked correctly given the SFT prompt format?"""
    from evalkit import cont_logprob
    out = []
    for r in test:
        lp = {w: cont_logprob(model, tok, r["prompt"], " " + w)[0] for w in ("positive", "negative")}
        out.append(int(max(lp, key=lp.get) == r["label"]))
    return np.array(out)

@torch.no_grad()
def task_generated(model):
    """1/0 per held-out review: does greedy decoding START with the right label word? (what a user would see)"""
    out = []
    for r in test:
        enc = tok(r["prompt"], return_tensors="pt")
        g = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
        words = tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True).split()
        out.append(int(bool(words) and words[0] == r["label"]))
    return np.array(out)

def measure(model):
    model.eval()
    return {"task": task_correct(model), "gen": task_generated(model), "general": mcq_accuracy(model, tok, GENERAL_MCQ), "ppl": perplexity(model, tok, general_text)}

# ## 2. Fine-tune with LoRA (completion-only loss, as in step 02)
def lora_sft(lr, epochs):
    torch.manual_seed(SEED)
    base = AutoModelForCausalLM.from_pretrained("distilgpt2")
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"], fan_in_fan_out=True, task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = random.Random(SEED)
    for _ in range(epochs):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        for s in range(0, len(order), 8):
            rows = [train[i] for i in order[s:s + 8]]
            ids, labs = [], []
            for r in rows:
                p, c = tok(r["prompt"]).input_ids, tok(r["completion"]).input_ids + [tok.eos_token_id]
                ids.append(p + c); labs.append([-100] * len(p) + c)
            L = max(map(len, ids))
            x = torch.tensor([i + [tok.pad_token_id] * (L - len(i)) for i in ids])
            y = torch.tensor([l + [-100] * (L - len(l)) for l in labs])
            m = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i in ids])
            logits = model(input_ids=x, attention_mask=m).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    return model

# ## 3. The report card
def card(title, before, after):
    print(f"\n=== {title} ===")
    print(f"{'metric':<28}{'before':>8}{'after':>8}{'delta':>8}   {'95% CI of delta':<18}{'verdict'}")
    for name, key in (("task ranked-choice (n=40)", "task"), ("task generated EM (n=40)", "gen"), ("general MCQ acc (n=30)", "general")):
        d, (lo, hi) = paired_bootstrap(after[key], before[key])
        verdict = "real gain" if lo > 0 else "real regression" if hi < 0 else "within noise"
        print(f"{name:<28}{before[key].mean():>8.3f}{after[key].mean():>8.3f}{d:>+8.3f}   [{lo:+.2f}, {hi:+.2f}]{'':<5}{verdict}")
    print(f"{'general-text perplexity':<28}{before['ppl']:>8.1f}{after['ppl']:>8.1f}{after['ppl'] - before['ppl']:>+8.1f}   (lower is better; no CI: one number)")
    lo, hi = bootstrap_ci(after["gen"])
    print(f"generated exact-match after, bootstrap 95% CI: [{lo:.2f}, {hi:.2f}]")

base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
before = measure(base_model)
print(f"BEFORE: task {before['task'].mean():.3f} generated {before['gen'].mean():.3f}  general {before['general'].mean():.3f}  ppl {before['ppl']:.1f}")

gentle = lora_sft(lr=5e-4, epochs=3)
card("gentle LoRA: lr 5e-4, 3 epochs", before, measure(gentle))
aggressive = lora_sft(lr=5e-3, epochs=10)
card("aggressive LoRA: lr 5e-3, 10 epochs", before, measure(aggressive))

# TRY IT: add a third recipe with r=2 or with lr=1e-4; add the "mix in general text" replay idea from the README and re-measure.
# BREAK IT: evaluate on the TRAIN reviews instead of the test reviews. Compare. That is why held-out data exists.
