# Terms — Forward Pass, Logits, Softmax, Next Token

Folder: `04_llm_fundamentals/05_forward_pass_logits` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [04 LLM fundamentals README](../README.md)

**Terms in this folder:** [Causal language model](#causal-language-model) · [Prompt](#prompt) · [Forward pass](#forward-pass) · [torch.no_grad()](#torchno_grad) · [with statement (context manager)](#with-statement-context-manager) · [**inputs unpacking](#inputs-unpacking) · [Logits](#logits) · [Logits shape (batch, seq_len, vocab)](#logits-shape-batch-seq_len-vocab) · [Negative indexing and slicing](#negative-indexing-and-slicing) · [Softmax (torch.softmax)](#softmax-torchsoftmax) · [dim argument](#dim-argument) · [Probability distribution](#probability-distribution) · [torch.topk](#torchtopk) · [zip](#zip) · [torch.argmax](#torchargmax) · [Greedy decoding](#greedy-decoding) · [Cross-entropy loss](#cross-entropy-loss) · [labels argument](#labels-argument) · [Perplexity](#perplexity) · [Entropy](#entropy) · [Old-style % formatting](#old-style--formatting) · [Attribute access on a result object](#attribute-access-on-a-result-object)

---

## Causal language model  `(concept)`
**1. Definition:** A language model trained to predict the next token given only the tokens before it.
**2. In plain words:** It is a very strong autocomplete. "Causal" means it can only look backward, never at future words. GPT-style models are all causal. Everything else (chat, summaries) is built by repeating this next-token step.
**3. Example to quote:** > "Given 'The capital of France is', the model gives a probability for every token being next; it never sees what comes after."

**Seen in:** `main.py`, docstring.

## Prompt  `(concept)`
**1. Definition:** The input text given to a language model, which it continues.
**2. In plain words:** It is the starting text you provide. The model's whole job is to estimate what token comes after it.
**3. Example to quote:** > "My prompt was 'The capital of France is', which is 5 tokens."

**Seen in:** `main.py`, `prompt = ...`.

## Forward pass  `(concept)`
**1. Definition:** One run of input data through the network from input to output, without updating weights.
**2. In plain words:** You feed in token IDs, they flow through every layer, and you get scores out. Training does a forward pass, then a backward pass to adjust weights; inference only does the forward pass.
**3. Example to quote:** > "One forward pass took my five token IDs and produced logits for all 50,257 tokens at each position."
```python
out = model(**inputs)
```
**Seen in:** `main.py`, section 1.

## torch.no_grad()  `(third-party: torch)`
**1. Definition:** A context manager that turns off gradient tracking for the operations inside it.
**2. In plain words:** PyTorch normally records every operation so it can compute gradients later. When you are not training, that recording wastes memory. Wrapping inference in `no_grad` saves memory and a little time without changing the outputs.
**3. Example to quote:** > "I ran inference under `torch.no_grad()` because I am not training, so there is no need to store a gradient graph."
```python
with torch.no_grad():
    out = model(**inputs)
```
**Seen in:** `main.py`, sections 1 and 6.

## with statement (context manager)  `(syntax)`
**1. Definition:** `with obj:` runs a block with a temporary setup that is automatically undone when the block ends, even after an error.
**2. In plain words:** You see it with files (`with open(...)`) and here with `torch.no_grad()`. The setting applies only inside the indented block, then restores itself.
**3. Example to quote:** > "Gradients are disabled only inside the `with` block and come back automatically afterwards."

**Seen in:** `main.py`, section 1.

## **inputs unpacking  `(syntax)`
**1. Definition:** Putting `**` before a dict in a call passes each key-value pair as a keyword argument.
**2. In plain words:** The tokenizer returns a dict with `input_ids` and `attention_mask`. `model(**inputs)` is the same as `model(input_ids=..., attention_mask=...)` without typing them.
**3. Example to quote:** > "`model(**inputs)` unpacks my tokenizer dict into the arguments the model expects."
```python
def f(a): return a
f(**{"a": 1})  # same as f(a=1)
```
**Seen in:** `main.py`, section 1.

## Logits  `(concept)`
**1. Definition:** The raw, unnormalised scores a model outputs, one per vocabulary token; they can be negative and do not sum to 1.
**2. In plain words:** A higher logit means the model prefers that token more. Logits are not probabilities until softmax is applied. Sampling settings like temperature work on logits.
**3. Example to quote:** > "My last-position logits ranged from -85.5 to -57.4: scores, not probabilities."

**Seen in:** `main.py`, section 1.

## Logits shape (batch, seq_len, vocab)  `(concept)`
**1. Definition:** The logits tensor has three dimensions: number of sequences in the batch, number of positions, and vocabulary size.
**2. In plain words:** Every position outputs a score for every possible next token. For my prompt that is `(1, 5, 50257)`: one sentence, five positions, 50,257 candidate tokens each.
**3. Example to quote:** > "The shape is `(1, 5, 50257)` because each of the 5 positions predicts its own next token."

**Seen in:** `main.py`, section 1.

## Negative indexing and slicing  `(syntax)`
**1. Definition:** In Python indexes, `-1` means the last element; `x[0, -1, :]` selects batch 0, the last position, and all entries of the last axis.
**2. In plain words:** Only the last position matters when generating, because it is the one predicting the token after the prompt. The `:` means "take everything along this axis".
**3. Example to quote:** > "`logits[0, -1, :]` gives me the 50,257 scores for the token that follows my prompt."

**Seen in:** `main.py`, section 2.

## Softmax (torch.softmax)  `(third-party: torch)`
**1. Definition:** `torch.softmax(x, dim)` converts values along `dim` into probabilities that are positive and sum to 1.
**2. In plain words:** It is the same function you wrote in numpy in step 01. It exponentiates the scores and divides by their total. The sum printed 1.0000281 only because of floating-point rounding.
**3. Example to quote:** > "I applied `torch.softmax(last_logits, dim=-1)` and the probabilities summed to 1."
```python
probs = torch.softmax(last_logits, dim=-1)
```
**Seen in:** `main.py`, section 3.

## dim argument  `(third-party: torch)`
**1. Definition:** The axis along which a PyTorch operation is computed; `dim=-1` is the last axis.
**2. In plain words:** For softmax it tells PyTorch which axis should sum to 1. Using the wrong axis gives numbers that do not form probability distributions.
**3. Example to quote:** > "`dim=-1` means normalise across the vocabulary entries."

**Seen in:** `main.py`, section 3.

## Probability distribution  `(concept)`
**1. Definition:** A set of non-negative numbers that sum to 1, giving the probability of each possible outcome.
**2. In plain words:** After softmax, the 50,257 numbers say how likely each token is to come next. Choosing how to pick from this distribution is the subject of module 05.
**3. Example to quote:** > "The top guess ' the' had probability 0.118, ' home' 0.082."

**Seen in:** `main.py`, sections 3 and 4.

## torch.topk  `(third-party: torch)`
**1. Definition:** Returns the `k` largest values of a tensor and their indices as `.values` and `.indices`.
**2. In plain words:** It is how you list the model's best guesses. The indices are token IDs that you decode back to text.
**3. Example to quote:** > "`torch.topk(probs, k=5)` gave me the five most likely next tokens."
```python
top = torch.topk(probs, k=5)
```
**Seen in:** `main.py`, section 4.

## zip  `(Python built-in)`
**1. Definition:** Combines several iterables element by element into tuples.
**2. In plain words:** It lets the loop walk through `top.values` and `top.indices` together, so each probability is printed with its token ID.
**3. Example to quote:** > "I used `zip(top.values, top.indices)` to pair each probability with its token."
```python
list(zip([1, 2], [3, 4]))  # [(1, 3), (2, 4)]
```
**Seen in:** `main.py`, section 4.

## torch.argmax  `(third-party: torch)`
**1. Definition:** Returns the index of the largest value in a tensor.
**2. In plain words:** Applied to the probabilities it gives the most likely token ID, which is the "greedy" choice.
**3. Example to quote:** > "`torch.argmax(probs).item()` returned 262, the ID of ' the'."

**Seen in:** `main.py`, section 5.

## Greedy decoding  `(concept)`
**1. Definition:** Always choosing the token with the highest probability at each step.
**2. In plain words:** It is deterministic: the same prompt gives the same answer. The downside is that text can become repetitive or dull. Sampling (module 05) adds variety.
**3. Example to quote:** > "Greedy picked ' the' every time I ran the script."

**Seen in:** `main.py`, section 5.

## Cross-entropy loss  `(concept)`
**1. Definition:** The average negative log probability the model assigned to the true tokens; lower is better.
**2. In plain words:** It measures surprise. A loss of 5.72 means the real next tokens received on average about 0.3% probability. Training lowers this number.
**3. Example to quote:** > "The cross-entropy loss on my prompt was 5.72, so distilgpt2 found it fairly unpredictable."

**Seen in:** `main.py`, section 6.

## labels argument  `(third-party: transformers)`
**1. Definition:** An argument to a causal language model forward call; when given, the model computes the loss against these target IDs, shifting them internally by one position.
**2. In plain words:** Passing `labels=input_ids` says "the correct answers are the same text shifted by one". Without labels, `out.loss` is `None`.
**3. Example to quote:** > "I passed `labels=inputs['input_ids']` so the model returned `.loss` directly."

**Seen in:** `main.py`, section 6.

## Perplexity  `(concept)`
**1. Definition:** The exponential of the average cross-entropy loss, `e^loss`; roughly how many equally likely choices the model is torn between.
**2. In plain words:** Lower perplexity means the model finds the text more predictable. It is an easier number to picture than loss.
**3. Example to quote:** > "A loss of 5.72 corresponds to a perplexity of about 305."

**Seen in:** `WALKTHROUGH.md`, execution trace.

## Entropy  `(concept)`
**1. Definition:** A measure of uncertainty of a distribution: `-sum(p * log p)`; high when probabilities are spread out, low when one outcome dominates.
**2. In plain words:** A confident prompt gives low entropy; a vague one like "I like" gives high entropy. It tells you how risky sampling would be.
**3. Example to quote:** > "I compared the entropy of 'I like' and 'The capital of France is' to see which next-token distribution is flatter."

**Seen in:** `main.py`, TRY IT comment.

## Old-style % formatting  `(syntax)`
**1. Definition:** String formatting with the `%` operator, where `%.1f` formats a float with one decimal.
**2. In plain words:** It is the older alternative to f-strings; you may see it in older code. `"%.1f" % 3.14159` gives `'3.1'`.
**3. Example to quote:** > "The script prints min and max logits using `%.1f`, an older formatting style than f-strings."

**Seen in:** `main.py`, section 2.

## Attribute access on a result object  `(syntax)`
**1. Definition:** Reading a field of an object with dot notation, such as `out.logits`.
**2. In plain words:** The model returns an output object that holds several results (`logits`, `loss`) as named attributes rather than a plain tuple. This makes code easier to read.
**3. Example to quote:** > "`out.logits` is the tensor of scores; `out.loss` exists only when I pass labels."

**Seen in:** `main.py`, section 1.
