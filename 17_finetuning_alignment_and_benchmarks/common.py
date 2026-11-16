"""
Shared helpers for module 17: tiny, seeded, hand-built datasets and small utilities.
Nothing here downloads data. Everything is generated from word lists so you can read it in one minute.

  make_sentiment_data()  -> (train, test) lists of {"prompt", "completion", "label"}   (steps 02, 08, 11)
  make_preference_data() -> (train, test) lists of {"prompt", "chosen", "rejected"}     (steps 04, 05)
  seq_logprob(...)       -> sum of log p(completion tokens | prompt) for a causal LM   (steps 05, 08)
"""
import random
import torch
import torch.nn.functional as F

POS_TRAIN = ["great", "wonderful", "delicious", "amazing", "lovely", "fantastic"]
NEG_TRAIN = ["awful", "terrible", "horrible", "disgusting", "bad", "dreadful"]
POS_HELD = ["excellent", "perfect"]          # adjectives NEVER seen in training: tests generalisation
NEG_HELD = ["poor", "boring"]
NOUNS = ["pizza", "movie", "hotel", "service", "book", "phone", "concert", "soup", "game", "taxi"]
TEMPLATES = ["The {n} was {a}.", "I found the {n} {a}.", "What a {a} {n}!", "Honestly, the {n} seemed {a}."]


def _make(adjs_pos, adjs_neg, count, rng, seen=None):
    seen = set() if seen is None else seen
    rows = []
    while len(rows) < count:
        label = rng.choice(["positive", "negative"])
        a = rng.choice(adjs_pos if label == "positive" else adjs_neg)
        text = rng.choice(TEMPLATES).format(n=rng.choice(NOUNS), a=a)
        if text in seen:                       # no duplicates, and no train/test leakage
            continue
        seen.add(text)
        rows.append({"prompt": f"Review: {text}\nSentiment:", "completion": f" {label}", "label": label})
    return rows


def make_sentiment_data(n_train=80, n_test=40, seed=0):
    rng = random.Random(seed)
    seen = set()
    train = _make(POS_TRAIN, NEG_TRAIN, n_train, rng, seen)
    # test: half seen adjectives in NEW sentences, half UNSEEN adjectives
    half = n_test // 2
    test = _make(POS_TRAIN, NEG_TRAIN, half, rng, seen) + _make(POS_HELD, NEG_HELD, n_test - half, rng, seen)
    return train, test


QUESTIONS = ["How do I reset my password?", "Where is my order?", "Can I change my email address?",
             "How do I cancel my plan?", "What are your opening hours?", "Can I get a receipt?",
             "How do I contact support?", "Is there a student discount?", "Why was my card declined?",
             "How do I update my address?", "Can I pause my account?", "Do you ship abroad?"]
GOOD = ["Sure, happy to help. Here is how.", "Of course! Let me walk you through it.", "Great question, here are the steps.",
        "Thanks for asking. I can help with that.", "Glad to help, here is what to do.", "Absolutely, it only takes a minute.",
        "Happy to help you with that today.", "Certainly, I will explain it step by step."]
BAD = ["No. Figure it out yourself.", "Why are you asking me this?", "Read the manual, I am busy.",
       "That is a silly question.", "I do not care, go away.", "Not my problem.",
       "Stop wasting my time.", "Ask someone else, not me."]


def make_preference_data(n_train=60, n_test=30, seed=0):
    """Chosen = polite helpful reply, rejected = rude reply. The LAST 2 phrases of each pool only appear in test."""
    rng = random.Random(seed)
    def build(n, good, bad):
        return [{"prompt": f"User: {q}\nAssistant:", "chosen": " " + rng.choice(good), "rejected": " " + rng.choice(bad)}
                for q in (rng.choice(QUESTIONS) for _ in range(n))]
    train = build(n_train, GOOD[:6], BAD[:6])
    test = build(n_test // 2, GOOD[:6], BAD[:6]) + build(n_test - n_test // 2, GOOD[6:], BAD[6:])
    return train, test


def seq_logprob(model, tok, prompt, completion, grad=False):
    """Sum over completion tokens of log p(token | everything before it).  Returns a 0-d tensor."""
    p_ids = tok(prompt, return_tensors="pt").input_ids                     # (1, Lp)
    c_ids = tok(completion, return_tensors="pt").input_ids                 # (1, Lc)
    ids = torch.cat([p_ids, c_ids], dim=1)                                 # (1, Lp+Lc)
    with torch.set_grad_enabled(grad):
        logits = model(ids).logits                                         # (1, L, vocab)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)                   # predictions for tokens 1..L-1
    tgt = ids[:, 1:]                                                       # (1, L-1)
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)                # (1, L-1) log p of the true next token
    return tok_lp[:, p_ids.shape[1] - 1:].sum()                            # keep only completion positions
