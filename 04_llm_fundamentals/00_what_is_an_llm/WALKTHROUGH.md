# Step 00 - What is an LLM? (the map before the details)

## The story so far
This is the prologue. Module 03 taught you how models learn from data (features, loss, gradient descent, TF-IDF + a classifier). Now we ask a bigger question: how does a machine write text at all? The next nine steps are one pipeline, and each step exists because the one before it left a problem open.

## The pain point
A classifier answers "which label?". Writing text needs an answer to "what comes next, and then next again?", over a vocabulary of tens of thousands of options, using everything written so far. Nobody can write rules for that.

## The spark
Turn writing into a repeated multiple-choice question: *given the text so far, how likely is each possible next token?* Pick one, append it, ask again. Everything else in this module is how to compute that probability well. The toy in `main.py` computes it by counting which word followed which. It works a little, and fails in a way that tells us what to build next.

## Goal
Understand the one job of a language model (predict the next token), the loop that turns that into text, and how the next 9 steps fit together. No transformer maths yet - just a counting toy that behaves like a tiny LLM.

## Run it
```bash
.venv/bin/python 04_llm_fundamentals/00_what_is_an_llm/main.py
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Token | word piece, sub-word, vocabulary item | Not always a word | Here a token is a whole word; real LLMs use sub-words (step 01) |
| Vocabulary | vocab, token inventory | Yes | The fixed list of all tokens the model knows; here 10, GPT-2 has 50,257 |
| Corpus | training data, training set, dataset | Roughly | The text the model learns from |
| Language model | LM, next-token predictor, autoregressive model | Yes | A function from context to a probability distribution over the next token |
| Bigram model | 2-gram, Markov chain of order 1 | Yes | Predicts from only the previous token, like this toy |
| Greedy decoding | argmax decoding | Yes | Always take the most likely next token |
| Sampling | stochastic decoding | Yes | Pick randomly according to the probabilities |
| Inference vs training | serving vs learning | No | Training changes the stored numbers; inference only uses them |

## Code, step by step

### 1. Module docstring: the big idea  (`main.py` lines 1–19)

```python
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
```

- **In simple words:** The header explains that every LLM runs the same loop, and this file builds the simplest version.
- **Exactly:** Docstring only; no runtime effect. The loop: text -> tokens -> function -> probability per next token -> pick one -> append -> repeat. This toy's function is counting; a transformer's is attention plus MLP layers.
- **Why it is here:** Gives the mental model that everything else in the module refines.

### 2. Imports from the standard library  (`main.py` lines 21–22)

```python
import random                      # built-in: random choice with a seed for repeatable output
from collections import Counter, defaultdict  # built-in: Counter = dict that counts things
```

- **In simple words:** A random number tool and a counting dictionary. No third-party packages.
- **Exactly:** `random.Random(seed)` makes a private seeded generator. `Counter` is a dict subclass mapping item -> count; `defaultdict(factory)` creates a missing value on first access.
- **Why it is here:** Keeps the first step dependency-free so you see the idea without libraries.

### 3. The training data (corpus)  (`main.py` lines 24–27)

```python
# ---------------------------------------------------------------------------
# 1. Training data: a tiny "internet".  Real models read trillions of tokens.
# ---------------------------------------------------------------------------
corpus = "the cat sat on the mat . the cat ate the fish . the dog sat on the rug ."
```

- **In simple words:** A tiny "internet" of three sentences.
- **Exactly:** `corpus` is one `str` with spaces around the full stops, so `split()` will make `.` a token. 20 tokens in total.
- **Why it is here:** Without a corpus there is nothing to learn from; real models read trillions of tokens.

### 4. Tokenise: text to words to integer ids  (`main.py` lines 29–38)

```python
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
```

- **In simple words:** Cut the text into words and give each distinct word a number.
- **Exactly:** `corpus.split()` gives a `list[str]` of 20 tokens. `sorted(set(tokens))` is the vocabulary of 10: `['.', 'ate', 'cat', 'dog', 'fish', 'mat', 'on', 'rug', 'sat', 'the']`. The dict comprehension maps word -> position, e.g. `the -> 9`. Real output: first eight ids `[9, 2, 8, 6, 9, 5, 0, 9]`.
- **Why it is here:** Neural networks and counting tables need integers, not strings. Step 01 shows how real tokenisers do this with sub-words.

### 5. "Train": count which word follows which  (`main.py` lines 40–48)

```python
# ---------------------------------------------------------------------------
# 3. "Training": count which token follows which.  This table IS the model.
#    (A transformer stores what it learned in weight matrices instead.)
# ---------------------------------------------------------------------------
follows = defaultdict(Counter)                         # follows["the"] -> Counter of next words
for current, nxt in zip(tokens, tokens[1:]):           # consecutive pairs: (the,cat),(cat,sat),...
    follows[current][nxt] += 1

print("\nwords seen after 'the':", dict(follows["the"]))
```

- **In simple words:** Look at every neighbouring pair of words and tally them. That tally table is the model.
- **Exactly:** `follows` is `dict[str, Counter]`. `zip(tokens, tokens[1:])` yields 19 pairs `(current, next)`. Real result for `'the'`: `{'cat': 2, 'mat': 1, 'fish': 1, 'dog': 1, 'rug': 1}`, a total of 6 occurrences of "the" followed by something.
- **Why it is here:** This is the whole "training" of the toy. A transformer stores what it learns in weight matrices rather than a count table.

### 6. Turn counts into next-word probabilities  (`main.py` lines 50–61)

```python
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
```

- **In simple words:** Divide each count by the total to get a chance for each possible next word.
- **Exactly:** `next_token_probs(word)` returns `{token: count/total}` with `total = 6` for "the": `cat 0.33, mat 0.17, fish 0.17, dog 0.17, rug 0.17`. The probabilities sum to 1.0. Words never seen after "the" are absent (probability 0), unlike an LLM which gives every vocabulary entry a probability.
- **Why it is here:** This is the same shape of answer as an LLM's softmax output (step 07), the thing generation depends on.

### 7. Generate: predict, pick, append, repeat  (`main.py` lines 63–76)

```python
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
```

- **In simple words:** Start from a word, ask for the next-word chances, choose one, and repeat.
- **Exactly:** `out[-1]` is the last word only. With `greedy=True`, `max(p, key=p.get)` picks the highest-probability key; otherwise `rng.choices(list(p), weights=list(p.values()))[0]` draws one word in proportion to its weight (`choices` returns a list, hence `[0]`). `" ".join(out)` produces a sentence. Real greedy output: `the cat sat on the cat sat on the`; a loop, because greedy always follows the same path.
- **Why it is here:** Greedy repeats itself and sampling varies. This is the same autoregressive loop as GPT, only with a counting table inside.

### 8. Run greedy and two sampled generations  (`main.py` lines 78–80)

```python
print("\ngreedy  :", generate("the", 8, greedy=True))
print("sampled :", generate("the", 8, greedy=False, seed=1))
print("sampled :", generate("the", 8, greedy=False, seed=3))
```

- **In simple words:** Print one greedy and two sampled sentences.
- **Exactly:** Real output: greedy `the cat sat on the cat sat on the`; seed 1 `the cat ate the cat sat on the dog`; seed 3 `the cat ate the fish . the cat ate`. Same seed gives the same sentence each run.
- **Why it is here:** Seeds make sampling reproducible for debugging; module 05 covers temperature, top-k and top-p.

### 9. Why the toy fails: it ignores context  (`main.py` lines 82–93)

```python
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
```

- **In simple words:** Two different sentences end with "the", and the toy gives the same answer for both.
- **Exactly:** `predict_from_context` keeps only `context.split()[-1]`, here `"the"` in both calls, so the two returned dicts are equal: real output `same prediction for two different contexts: True`. A transformer's attention lets every position read all earlier positions.
- **Why it is here:** This limitation is the reason attention (step 04) and transformers (step 05) exist.

## The story in six sentences
1. A **language model** answers one question: *given the text so far, how likely is each possible next token?*
2. Text is first cut into **tokens** and each gets an integer **id** (step 01).
3. Each id becomes a **vector** of numbers, the only thing a neural network can work with (steps 02 and 03).
4. Layers of **attention + MLP** mix information between tokens so each position "understands" its context (steps 04 and 05).
5. The last layer outputs one score (**logit**) per vocabulary entry; softmax turns them into probabilities (step 07).
6. We pick a token, append it to the text, and run the whole thing again - that is **generation** (step 08). Everything the model knows was set during **training** (step 09).

```
text -> tokens -> ids -> vectors -> [attention + MLP] x N -> logits -> probabilities -> pick -> append -> repeat
         01        01     02, 03        04 / 05                07           07           08       08
```

## The order of the steps (data-flow order)
We read the pipeline left to right, the way data flows through it:
- **01 tokenizer** - text becomes integer ids.
- **02 vectors and similarity** - why ids are not enough, and what a good vector looks like (building on the TF-IDF you already know).
- **03 embeddings** - the learned table that gives each id a vector, plus positions.
- **04 attention** - each vector looks at the others to get context.
- **05 transformer block** - attention + MLP, stacked deep.
- **06-08** load a real model, read its logits, run the generation loop. **09** how it was trained.

The toy's weakness (one word of context) is first attacked in step 04, but steps 01-03 come first because attention needs vectors to work on.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `text.split()` | cut a string at whitespace into a list | `"a b".split() == ["a", "b"]` |
| `{k: v for ...}` | dict comprehension: build a dict in one line | `{t: i for i, t in enumerate(["a","b"])}` -> `{"a": 0, "b": 1}` |
| `zip(a, b)` | walk two lists side by side | `zip(tokens, tokens[1:])` gives consecutive pairs |
| `defaultdict(Counter)` | dict that creates a fresh counter for a missing key | `follows["the"]["cat"] += 1` just works |
| `max(d, key=d.get)` | the key whose value is largest | `max({"a":1,"b":3}, key=...)` -> `"b"` |
| `rng.choices(items, weights=w)` | weighted random pick | likelier items are picked more often |

## Libraries & functions
| Name | Built-in or 3rd-party | What it does | Important args | Returns |
|---|---|---|---|---|
| `random.Random(seed)` | built-in | repeatable random generator | `seed` | generator object |
| `collections.Counter` | built-in | dict that counts items | iterable (optional) | `Counter` |
| `collections.defaultdict` | built-in | dict with automatic default values | factory such as `Counter` | `defaultdict` |
| `sorted(set(x))` | built-in | unique items in order | iterable | list |

## Execution trace (input -> processing -> output)
Real output from a run:
```
tokens : ['the', 'cat', 'sat', 'on', 'the', 'mat', '.', 'the'] ...
vocab  : ['.', 'ate', 'cat', 'dog', 'fish', 'mat', 'on', 'rug', 'sat', 'the'] (size 10)
ids    : [9, 2, 8, 6, 9, 5, 0, 9] ...

words seen after 'the': {'cat': 2, 'mat': 1, 'fish': 1, 'dog': 1, 'rug': 1}
P(next | 'the') = {'cat': 0.33, 'mat': 0.17, 'fish': 0.17, 'dog': 0.17, 'rug': 0.17}
sum of probabilities: 1.0

greedy  : the cat sat on the cat sat on the
sampled : the cat ate the cat sat on the dog
sampled : the rug . the cat ate the dog sat

same prediction for two different contexts: True
```
- **Tokens/ids**: the toy uses whole words; a real tokenizer uses sub-words and a ~50,000-entry vocabulary (step 01).
- **Counting = training**: 'the' was followed by 'cat' twice out of six times -> 33 %. A real model learns such statistics (and far subtler ones) into its weights.
- **Greedy** always repeats the same loop ("the cat sat on the cat sat on the") - a first look at why pure greedy decoding gets repetitive; sampling (module 05) adds variety.
- **Last line = the weakness**: the toy only sees one previous word, so it cannot tell a cat story from a dog story. Giving every position access to the whole context is exactly what **attention** adds - that is step 04.

## Experiments (change ONE thing)
- Add a sentence to `corpus` ("the cat sat on the rug .") and re-run: how do the probabilities after "on" and "the" move?
- Change `greedy=True` to a different `seed` for sampling: same prompt, different text - the source of non-determinism in chat models.
- Make the toy look at the last TWO words (key = a tuple of two words). Output gets more coherent - and the table gets much bigger. That trade-off (more context vs. exploding table) is why we learn compact weights instead of counting.

## Debugging practice
Call `next_token_probs("elephant")`. A `defaultdict` returns an empty counter, so `total = 0` and you get `ZeroDivisionError: division by zero`. Meaning: the model has never seen that word. Real LLMs avoid this because sub-word tokenisation can spell any text from known pieces (step 01).

## Rebuild checklist
- [ ] Count next-word frequencies from a corpus without looking.
- [ ] Turn counts into probabilities that sum to 1.
- [ ] Write the generate loop: predict -> pick -> append -> repeat.
- [ ] Say which line of the loop a transformer replaces (the "predict" line).

## The new problem (cliffhanger)
The toy keeps a table keyed by *words*. A real vocabulary has endless words, typos and other languages, and `"elephant"` already crashed it. Before any network can touch text, we need a way to cut any string into pieces from a fixed list and give each piece an id. Next: [Step 01 - Tokenizer](../01_tokenizer/WALKTHROUGH.md).

> **Say it aloud (interview box):** A language model maps the text so far to a probability for every possible next token, and to produce text I pick one token, append it and repeat. My toy does the prediction by counting which word followed which, so it sees only one word of context and cannot handle unseen words. A transformer runs the same loop but computes the probabilities with attention and MLP layers that use the whole context, with weights learned from huge amounts of text.

## Next
[Step 01 - Tokenizer](../01_tokenizer/WALKTHROUGH.md): how any text becomes a list of integer ids.
