"""
GOAL:   See, in ~60 lines of plain Python, what a language model IS before any
        transformer maths: a machine that turns "text so far" into a probability
        for every possible next token, then picks one and repeats.

RUN:    from this folder:  python main.py
        from repo root:    .venv/bin/python 04_llm_fundamentals/00_what_is_an_llm/main.py
EXPECTED OUTPUT (short): a token list, a table of next-word counts, probabilities
        that sum to 1.0, and a short generated sentence.
NEEDS:  Python standard library only. No internet, no numpy.

Big idea
--------
Every LLM, from this toy to GPT-4, does the same loop:
    text -> tokens -> (some function) -> probability per next token -> pick one -> append -> repeat
This toy's "function" is just COUNTING which word followed which in a tiny corpus.
A transformer replaces the counting with attention + MLP layers (steps 04-05) that can
use the WHOLE context, not just the previous word, and learns from billions of tokens.
"""

import random                      # built-in: random choice with a seed for repeatable output
from collections import Counter, defaultdict  # built-in: Counter = dict that counts things

# ---------------------------------------------------------------------------
# 1. Training data: a tiny "internet".  Real models read trillions of tokens.
# ---------------------------------------------------------------------------
corpus = "the cat sat on the mat . the cat ate the fish . the dog sat on the rug ."

# ---------------------------------------------------------------------------
# 2. Tokenize: cut text into pieces and give each piece an integer id.
#    Here a token = a word (real models use sub-words; step 01).
# ---------------------------------------------------------------------------
tokens = corpus.split()                                # list[str], shape: (n_tokens,) = (20,)
vocab = sorted(set(tokens))                            # list[str], shape: (vocab_size,) = (10,)
token_to_id = {t: i for i, t in enumerate(vocab)}      # dict comprehension: word -> id; size (vocab_size,)
print("tokens :", tokens[:8], "...")
print("vocab  :", vocab, f"(size {len(vocab)})")
print("ids    :", [token_to_id[t] for t in tokens[:8]], "...")

# ---------------------------------------------------------------------------
# 3. "Training": count which token follows which.  This table IS the model.
#    (A transformer stores what it learned in weight matrices instead.)
# ---------------------------------------------------------------------------
follows = defaultdict(Counter)                         # follows["the"] -> Counter of next words
for current, nxt in zip(tokens, tokens[1:]):           # consecutive pairs: (the,cat),(cat,sat),...
    follows[current][nxt] += 1

print("\nwords seen after 'the':", dict(follows["the"]))

# ---------------------------------------------------------------------------
# 4. Next-token probabilities: counts / total.  Same shape of answer an LLM gives:
#    one probability per vocabulary entry, summing to 1 (step 07 and module 05).
# ---------------------------------------------------------------------------
def next_token_probs(word):
    counts = follows[word]
    total = sum(counts.values())
    return {t: c / total for t, c in counts.items()}   # dict: next-token -> prob, size (n_distinct_next,)

probs = next_token_probs("the")                        # dict, shape: (5,) = 5 distinct words seen after "the"
print("P(next | 'the') =", {t: round(p, 2) for t, p in probs.items()})
print("sum of probabilities:", round(sum(probs.values()), 6))

# ---------------------------------------------------------------------------
# 5. Generation = predict, pick, append, repeat (step 08 does this with a real model).
#    Greedy = always take the most likely token.  Sampling = roll weighted dice.
# ---------------------------------------------------------------------------
def generate(start, n_tokens, greedy=True, seed=0):
    rng = random.Random(seed)
    out = [start]
    for _ in range(n_tokens):
        p = next_token_probs(out[-1])                  # the model only sees the LAST word
        if greedy:
            out.append(max(p, key=p.get))              # highest probability wins
        else:
            out.append(rng.choices(list(p), weights=list(p.values()))[0])
    return " ".join(out)

print("\ngreedy  :", generate("the", 8, greedy=True))
print("sampled :", generate("the", 8, greedy=False, seed=1))
print("sampled :", generate("the", 8, greedy=False, seed=2))

# ---------------------------------------------------------------------------
# 6. Why this toy is NOT enough -> why transformers exist.
#    It sees only ONE previous word, so "the cat sat on the" and "the dog ate the"
#    get the same prediction after "the".  It can't use context.  Attention (step 04)
#    is the mechanism that lets every position read every earlier position.
# ---------------------------------------------------------------------------
def predict_from_context(context):
    return next_token_probs(context.split()[-1])       # toy model: throws away everything but the last word

a = predict_from_context("the cat sat on the")
b = predict_from_context("the dog ate the")
print("\nsame prediction for two different contexts:", a == b)
