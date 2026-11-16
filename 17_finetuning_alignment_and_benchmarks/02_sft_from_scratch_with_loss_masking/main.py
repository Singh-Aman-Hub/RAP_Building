"""
GOAL:  Supervised fine-tuning (SFT) from scratch with a hand-written loop: tokenise (prompt, completion), put
       -100 on prompt tokens (completion-only loss), pad a batch, train distilgpt2 on 80 tiny sentiment examples,
       and measure BEFORE/AFTER accuracy on held-out reviews (including adjectives never seen in training).
       Compare completion-only loss with full-sequence loss, and padding with packing (token arithmetic).

RUN (from this step folder):  python main.py
RUN (from repo root):         HF_HUB_OFFLINE=1 .venv/bin/python 17_finetuning_alignment_and_benchmarks/02_sft_from_scratch_with_loss_masking/main.py
EXPECTED OUTPUT (shortened):
    example row, tokens and labels with -100 on the prompt
    before: ranked-choice acc already high (gpt-style models know the words) but generated exact-match low;
    after SFT: generated exact-match ~1.0 and held-out loss much lower; full-sequence-loss variant reported next to it
    padding wastes X% of positions; packing wastes ~0%
NEEDS: torch, transformers; cached distilgpt2.  ~1 minute on CPU.
"""
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import POS_HELD, NEG_HELD, make_sentiment_data, seq_logprob   # noqa: E402

SEED = int(os.environ.get("SEED", 0))
EPOCHS = int(os.environ.get("EPOCHS", 3))
LR = float(os.environ.get("LR", 5e-5))
BATCH = int(os.environ.get("BATCH", 8))
MODEL = "distilgpt2"
torch.manual_seed(SEED)
random.seed(SEED)

# ## 1. Load tokenizer and the data
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token                      # GPT-2 has no pad token; reuse EOS (masked out of the loss below)
train, test = make_sentiment_data()
print(f"train={len(train)} test={len(test)}  example: {train[0]['prompt']!r} -> {train[0]['completion']!r}")

# ## 2. Turn one example into input_ids and labels
def encode(row, mask_prompt=True):
    p = tok(row["prompt"]).input_ids                          # prompt token ids
    c = tok(row["completion"]).input_ids + [tok.eos_token_id]  # completion ids + EOS so the model learns to STOP
    ids = p + c                                               # one sequence, (Lp + Lc,)
    labels = ([-100] * len(p) if mask_prompt else p) + c      # -100 = ignored by cross_entropy
    return ids, labels

ids, labels = encode(train[0])
print("\ninput_ids:", ids)
print("labels   :", labels)

# ## 3. Pad a batch (right padding) and build the attention mask
def collate(rows, mask_prompt=True):
    enc = [encode(r, mask_prompt) for r in rows]
    L = max(len(i) for i, _ in enc)                           # longest sequence in this batch
    x = torch.full((len(enc), L), tok.pad_token_id)           # (B, L) filled with pad
    y = torch.full((len(enc), L), -100)                       # (B, L) pad positions also ignored in the loss
    m = torch.zeros((len(enc), L), dtype=torch.long)          # (B, L) 1 = real token
    for b, (i, l) in enumerate(enc):
        x[b, :len(i)], y[b, :len(l)], m[b, :len(i)] = torch.tensor(i), torch.tensor(l), 1
    return x, y, m

# ## 4. The loss: shift by one, cross-entropy, ignore -100
def sft_loss(model, x, y, m):
    logits = model(input_ids=x, attention_mask=m).logits      # (B, L, vocab)
    # position t predicts token t+1, so drop the last logit and the first label
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100)

# ## 5. Evaluation: accuracy by comparing two candidate completions, plus held-out loss
@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    correct = 0
    for r in rows:
        lp = {w: seq_logprob(model, tok, r["prompt"], " " + w).item() for w in ("positive", "negative")}
        correct += max(lp, key=lp.get) == r["label"]
    x, y, m = collate(rows)
    return correct / len(rows), sft_loss(model, x, y, m).item()

@torch.no_grad()
def gen_exact_match(model, rows):
    """Generative check: greedy-decode 3 tokens and require the first word to equal the label (this is what a user would see)."""
    model.eval()
    hits = 0
    for r in rows:
        enc = tok(r["prompt"], return_tensors="pt")
        out = model.generate(**enc, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
        words = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).split()
        hits += bool(words) and words[0] == r["label"]
    return hits / len(rows)

def split_acc(model):
    held = POS_HELD + NEG_HELD                                # adjectives the model never saw in training
    unseen = [r for r in test if any(a in r["prompt"] for a in held)]
    seen = [r for r in test if r not in unseen]
    return evaluate(model, seen)[0], evaluate(model, unseen)[0]

# ## 6. The training loop (AdamW, shuffled mini-batches)
def finetune(mask_prompt):
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    rng = random.Random(SEED)
    for ep in range(EPOCHS):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        tot = 0.0
        for s in range(0, len(order), BATCH):
            x, y, m = collate([train[i] for i in order[s:s + BATCH]], mask_prompt)
            loss = sft_loss(model, x, y, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # keeps rare big gradients from wrecking the weights
            opt.step()
            tot += loss.item()
        print(f"  epoch {ep + 1}  mean train loss {tot / (len(order) / BATCH):.3f}")
    return model

# ## 7. Before / after
base = AutoModelForCausalLM.from_pretrained(MODEL)
acc0, loss0 = evaluate(base, test)
s0, u0 = split_acc(base)
print(f"\nBEFORE  ranked-choice acc={acc0:.3f} (seen-adj {s0:.3f}, unseen-adj {u0:.3f})  generated exact-match={gen_exact_match(base, test):.3f}  held-out loss={loss0:.3f}")
print("training with completion-only loss (prompt labels = -100):")
m_c = finetune(True)
acc1, loss1 = evaluate(m_c, test)
s1, u1 = split_acc(m_c)
print(f"AFTER   ranked-choice acc={acc1:.3f} (seen-adj {s1:.3f}, unseen-adj {u1:.3f})  generated exact-match={gen_exact_match(m_c, test):.3f}  held-out loss={loss1:.3f}")
print("training with full-sequence loss (prompt tokens also trained):")
m_f = finetune(False)
acc2, _ = evaluate(m_f, test)
s2, u2 = split_acc(m_f)
print(f"AFTER   ranked-choice acc={acc2:.3f} (seen-adj {s2:.3f}, unseen-adj {u2:.3f})  generated exact-match={gen_exact_match(m_f, test):.3f}   <- full-sequence variant")

# ## 8. A generated answer before and after
def generate(model, prompt):
    enc = tok(prompt, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:])
q = test[-1]["prompt"]
print(f"\nprompt {q!r}\n  base     -> {generate(base, q)!r}\n  SFT      -> {generate(m_c, q)!r}")

# ## 9. Padding vs packing: how many positions are wasted?
lens = [len(encode(r)[0]) for r in train]                     # real token counts per example
pad_total = sum(max(lens[i:i + BATCH]) * len(lens[i:i + BATCH]) for i in range(0, len(lens), BATCH))   # padded batch area
real = sum(lens)
BLOCK = 64
packed_blocks = -(-real // BLOCK)                             # ceil: greedy concatenation into fixed 64-token blocks
print(f"\nreal tokens={real}  padded positions={pad_total} (waste {1 - real / pad_total:.1%})  "
      f"packed into {packed_blocks} blocks of {BLOCK} (waste {1 - real / (packed_blocks * BLOCK):.1%})")

# TRY IT: EPOCHS=1 python main.py   (how much does one epoch buy?)   LR=5e-4 python main.py   (too hot?)
# BREAK IT: delete the `labels[...] = -100` mask by calling finetune(False) only, then look at what the prompt tokens cost.
