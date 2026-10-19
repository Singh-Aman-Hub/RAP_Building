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
