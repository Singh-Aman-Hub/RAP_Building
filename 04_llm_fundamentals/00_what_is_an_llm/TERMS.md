# Terms — What Is an LLM?

Folder: `04_llm_fundamentals/00_what_is_an_llm` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [Language model](#language-model) · [Token](#token) · [Vocabulary](#vocabulary) · [Next-token prediction](#next-token-prediction) · [Autoregressive generation](#autoregressive-generation) · [Greedy decoding](#greedy-decoding) · [Sampling](#sampling) · [Context](#context) · [collections.Counter](#collectionscounter) · [collections.defaultdict](#collectionsdefaultdict)

---

## Language model
**1. Definition:** A function that, given a sequence of tokens, returns a probability distribution over the next token.
**2. In plain words:** It is an autocomplete that gives odds instead of one answer. "The cat sat on the ..." -> mat 30 %, rug 20 %, ... A large language model (LLM) is the same idea with a neural network that has billions of parameters.
**4. Interview angle:** "What does an LLM do?" -> predicts the next-token distribution; text is made by repeating that.
**3. Example to quote:** > "A language model maps the text so far to a probability for every possible next token; generation just repeats that."

## Token
**1. Definition:** The unit of text a model reads and writes, represented by an integer id from a fixed vocabulary.
**2. In plain words:** Usually a word or a piece of a word. Cost, speed and context limits are all counted in tokens. The toy uses whole words; real tokenizers use sub-words (step 01).
**3. Example to quote:** > "Cost and context limits are counted in tokens, not words; my toy used whole words but real tokenizers use sub-words."

## Vocabulary
**1. Definition:** The complete set of tokens a model knows, each with an id.
**2. In plain words:** The model's alphabet. The toy has 10; distilgpt2 has 50,257. The model outputs one score per vocabulary entry at every step.
**3. Example to quote:** > "The vocabulary is the model's alphabet: my toy had 10 entries, distilgpt2 has 50,257, and the model scores every one at each step."

## Next-token prediction
**1. Definition:** The training and inference task of predicting token *t+1* from tokens *1..t*.
**2. In plain words:** The only thing a base LLM is trained to do. Answering questions, writing code and translating all emerge from doing this well on enough text.
**3. Example to quote:** > "A base LLM is trained on one task, next-token prediction, and everything else emerges from doing it well."

## Autoregressive generation
**1. Definition:** Producing a sequence one token at a time, feeding each output back as part of the next input.
**2. In plain words:** Predict, pick, append, repeat. It is why long answers take longer: one model run per new token (step 08).
**3. Example to quote:** > "Generation is predict, pick, append, repeat, which is why a longer answer needs more model runs."

## Greedy decoding
**1. Definition:** Always choosing the single most probable next token.
**2. In plain words:** Deterministic and safe, but can loop ("the cat sat on the cat sat on the") and sounds dull. See `generate(..., greedy=True)`.
**3. Example to quote:** > "Greedy decoding always takes the top token, so my toy looped on 'the cat sat on the cat sat on the'."

## Sampling
**1. Definition:** Choosing the next token at random, weighted by its probability.
**2. In plain words:** Weighted dice. Gives varied text; temperature, top-k and top-p (module 05) control how adventurous the dice are.
**3. Example to quote:** > "Sampling picks the next token by weighted dice, so the same prompt gives different text on different seeds."

## Context
**1. Definition:** The tokens the model can see when predicting the next one.
**2. In plain words:** The toy's context is one word; a transformer's context is the whole prompt up to its context window (thousands of tokens). Using context well is what attention is for.
**3. Example to quote:** > "My toy's context was one word; a transformer's context is the whole prompt, and attention is how it uses it."

## collections.Counter  `(built-in: collections)`
**1. Definition:** A dict subclass that counts hashable items.
**2. In plain words:** `Counter("abca")` -> `{'a': 2, 'b': 1, 'c': 1}`. Used to count which word follows which.
**3. Example to quote:** > "I used `Counter` to count how often each word followed another, e.g. `follows['the']['cat'] == 2`."

## collections.defaultdict  `(built-in: collections)`
**1. Definition:** A dict that builds a default value for a missing key using a factory function.
**2. In plain words:** `defaultdict(Counter)` lets you write `follows[word][next] += 1` without checking whether the key exists. Gotcha: merely *reading* a missing key creates it.
**3. Example to quote:** > "I used `defaultdict(Counter)` so I could write `follows[word][next] += 1` without checking the key exists."
