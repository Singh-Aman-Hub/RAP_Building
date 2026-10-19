<!-- vocabulary draft: 04_llm_fundamentals -->

### Large language model (LLM)
<!-- cat: llm -->
- **Technical definition:** A neural network with (typically) billions of parameters trained on large text corpora to predict the next token given the previous tokens.
- **Plain English:** A very big autocomplete: given the words so far it scores every possible next token.
- **Example / analogy:** Your phone keyboard suggests one next word; an LLM does the same with far more context and knowledge, one token at a time.
- **Snippet:**
```python
ids = tok('Hello', return_tensors='pt')['input_ids']
next_token_logits = model(ids).logits[0, -1]
```
- **Snippet explained:** Run the model on the prompt ids; the scores at the LAST position are the prediction for the next token.
- **Related terms:** [[Transformer]], [[Logits]], [[Autoregressive generation]]

### Transformer
<!-- cat: llm -->
- **Technical definition:** A neural-network architecture made of stacked blocks, each combining self-attention and a feed-forward (MLP) network with residual connections and layer normalisation.
- **Plain English:** The blueprint almost every modern LLM follows.
- **Example / analogy:** Each block lets tokens exchange information (attention) and then process it individually (MLP); distilgpt2 stacks 6 blocks, gpt2 stacks 12.
- **Snippet:**
```python
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('distilgpt2')
print(cfg.n_layer, cfg.n_embd, cfg.n_head)
```
- **Snippet explained:** Reads the architecture sizes: number of blocks (6), vector size per token (768) and attention heads (12).
- **Related terms:** [[Self-attention]], [[Model config]], [[Attention head (multi-head attention)]]

### Self-attention
<!-- cat: llm -->
- **Technical definition:** An operation where every token computes weights over all (allowed) tokens in the sequence using query-key dot products and returns a weighted sum of value vectors.
- **Plain English:** Each word decides which other words to pay attention to and blends their information into itself.
- **Example / analogy:** In 'The animal didn't cross the street because it was tired', attention lets 'it' pull information from 'animal'.
- **Snippet:**
```python
scores = Q @ K.T / np.sqrt(d_head)
weights = softmax(scores)
out = weights @ V
```
- **Snippet explained:** Score every token pair, turn scores into probabilities per row, then average the value vectors with those probabilities.
- **Related terms:** [[Query, key, value (Q/K/V)]], [[Causal mask]], [[Softmax]]

### Query, key, value (Q/K/V)
<!-- cat: llm -->
- **Technical definition:** Three learned linear projections of each token vector: the query says what the token looks for, the key says what it offers, the value is the content passed on if selected.
- **Plain English:** A search engine inside the model: query = search text, key = page title, value = page content.
- **Example / analogy:** Matching a query with keys gives relevance scores; those scores decide how much of each value is mixed into the output.
- **Snippet:**
```python
Q = x @ W_q
K = x @ W_k
V = x @ W_v
```
- **Snippet explained:** Three matrix multiplications of the same input with three different learned weight matrices.
- **Related terms:** [[Self-attention]], [[Attention head (multi-head attention)]], [[Parameter (weight)]]

### Causal mask
<!-- cat: llm -->
- **Technical definition:** A mask that sets attention scores for future positions to -infinity before softmax so token i can only attend to tokens 0..i.
- **Plain English:** A rule that says 'you cannot peek at the words that come after you'.
- **Example / analogy:** Without it, a model trained to predict the next token could simply copy it from the input.
- **Snippet:**
```python
mask = np.triu(np.ones((n, n), dtype=bool), k=1)
scores = np.where(mask, -np.inf, scores)
```
- **Snippet explained:** np.triu marks the area above the diagonal (the future); -inf becomes probability 0 after softmax.
- **Related terms:** [[Self-attention]], [[Causal language model]], [[Softmax]]

### Attention head (multi-head attention)
<!-- cat: llm -->
- **Technical definition:** One independent attention computation with its own Q/K/V projections; multi-head attention runs several heads in parallel and concatenates their outputs.
- **Plain English:** Several people reading the same sentence, each looking for a different kind of relationship (grammar, topic, position).
- **Example / analogy:** distilgpt2 has 12 heads of size 64 (768 / 12) per layer.
- **Snippet:**
```python
head_dim = cfg.n_embd // cfg.n_head   # 768 // 12 = 64
```
- **Snippet explained:** Splitting the hidden size equally across heads keeps the total cost the same while allowing different attention patterns.
- **Related terms:** [[Self-attention]], [[Transformer]], [[Model config]]

### Token
<!-- cat: llm -->
- **Technical definition:** The basic unit of text an LLM reads and writes: a sub-word piece from a fixed vocabulary, represented by an integer id.
- **Plain English:** A chunk of text, roughly three-quarters of an English word on average.
- **Example / analogy:** 'unbelievably' becomes ['un', 'bel', 'iev', 'ably'] in GPT-2; billing, context limits and speed are all counted in tokens.
- **Snippet:**
```python
tok.tokenize('unbelievably')
```
- **Snippet explained:** Returns the list of string pieces the tokenizer splits the word into.
- **Related terms:** [[Tokenizer]], [[Token ID]], [[Context window]]

### Tokenizer
<!-- cat: llm -->
- **Technical definition:** The component that converts text to token ids (encode) and back (decode) using a learned vocabulary and merge rules.
- **Plain English:** A translator from human text to the numbers a model understands - and back.
- **Example / analogy:** GPT-2's tokenizer turns 'Hello world' into [15496, 995].
- **Snippet:**
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('distilgpt2')
ids = tok.encode('Hello world')
text = tok.decode(ids)
```
- **Snippet explained:** Load the tokenizer once, encode to a list of ints, decode back to the original string.
- **Related terms:** [[Byte-pair encoding (BPE)]], [[Token ID]], [[Vocabulary (vocab size)]]

### Byte-pair encoding (BPE)
<!-- cat: llm -->
- **Technical definition:** A sub-word tokenisation algorithm that starts from bytes/characters and repeatedly merges the most frequent adjacent pair in the training corpus until the vocabulary has a target size.
- **Plain English:** Frequent letter combinations get glued into single pieces; rare words are built from smaller pieces.
- **Example / analogy:** 'the' is one token because it is extremely common; 'Pneumonoultramicroscopic' needs 8 pieces.
- **Snippet:**
```python
tok.tokenize('Pneumonoultramicroscopic')
```
- **Snippet explained:** Shows the rare word broken into 8 known sub-word pieces.
- **Related terms:** [[Tokenizer]], [[Token]], [[Vocabulary (vocab size)]]

### Vocabulary (vocab size)
<!-- cat: llm -->
- **Technical definition:** The fixed set of tokens a tokenizer can produce; its size also sets the number of rows in the embedding table and the width of the output logits.
- **Plain English:** The model's dictionary of allowed pieces.
- **Example / analogy:** GPT-2 has 50,257 tokens, so each prediction is a choice among 50,257 options.
- **Snippet:**
```python
print(tok.vocab_size)   # 50257
```
- **Snippet explained:** A property of the tokenizer object.
- **Related terms:** [[Token]], [[Embedding]], [[Logits]]

### Token ID
<!-- cat: llm -->
- **Technical definition:** The integer index of a token in the vocabulary; the model only ever sees ids.
- **Plain English:** A token's row number in the dictionary.
- **Example / analogy:** 'world' (no space) is 6894 but ' world' (with a leading space) is 995.
- **Snippet:**
```python
tok.encode('world'), tok.encode(' world')
```
- **Snippet explained:** Two different strings give two different ids because spaces are part of the token.
- **Related terms:** [[Token]], [[Tokenizer]], [[Embedding]]

### Special token (EOS)
<!-- cat: llm -->
- **Technical definition:** A reserved token with a control meaning rather than text, such as end-of-sequence, padding, or role markers in chat formats.
- **Plain English:** A punctuation mark for the model itself.
- **Example / analogy:** GPT-2's <|endoftext|> (id 50256) marks the end of a document; generation can stop when the model emits it.
- **Snippet:**
```python
tok.eos_token, tok.eos_token_id
```
- **Snippet explained:** Attributes holding the end-of-text string and its id.
- **Related terms:** [[Token ID]], [[Padding and attention mask]], [[model.generate()]]

### Padding and attention mask
<!-- cat: llm -->
- **Technical definition:** Padding adds filler tokens so sequences in a batch have equal length; the attention mask (1 = real, 0 = pad) tells the model to ignore the filler.
- **Plain English:** Filling shorter rows of a table with blanks and marking which cells are blanks.
- **Example / analogy:** Batching 'Hi' with a 7-token sentence pads 'Hi' with six <|endoftext|> ids and a mask of [1,0,0,0,0,0,0].
- **Snippet:**
```python
tok.pad_token = tok.eos_token
batch = tok(['Hi', 'Hello there, how are you?'], padding=True, return_tensors='pt')
```
- **Snippet explained:** GPT-2 has no pad token, so reuse EOS; padding=True pads to the longest sentence and the mask comes back in the result.
- **Related terms:** [[Special token (EOS)]], [[Tokenizer]], [[Label masking (-100)]]

### Embedding
<!-- cat: llm -->
- **Technical definition:** A learned dense vector assigned to each discrete item (here token id); stored as a lookup table of shape vocab_size x hidden_size.
- **Plain English:** A numeric 'meaning coordinate' for each token.
- **Example / analogy:** Monday and Tuesday have vectors with cosine similarity 0.89 in distilgpt2.
- **Snippet:**
```python
import torch.nn as nn
emb = nn.Embedding(5, 3)
print(emb(torch.tensor([4, 0])))
```
- **Snippet explained:** Looks up rows 4 and 0 of a 5x3 table; the numbers are trainable weights.
- **Related terms:** [[Position embedding]], [[Cosine similarity]], [[Vocabulary (vocab size)]]

### Cosine similarity
<!-- cat: ml -->
- **Technical definition:** The cosine of the angle between two vectors: dot(a, b) / (|a| |b|); 1 = same direction, 0 = orthogonal, -1 = opposite.
- **Plain English:** How much two arrows point the same way, ignoring their length.
- **Example / analogy:** cosine('cat', 'dog') = 0.57 but cosine('cat', 'economics') = 0.20 in GPT-2 embeddings.
- **Snippet:**
```python
import torch.nn.functional as F
F.cosine_similarity(a, b, dim=0).item()
```
- **Snippet explained:** Compares two 1-D vectors and converts the 0-d result to a Python float.
- **Related terms:** [[Embedding]], [[Position embedding]]

### Position embedding
<!-- cat: llm -->
- **Technical definition:** Vectors (learned or computed) added to token embeddings to tell attention where each token sits in the sequence.
- **Plain English:** Seat numbers added to each word so order is not lost.
- **Example / analogy:** GPT-2 learns one vector for each of its 1024 positions; beyond that there is no row (IndexError).
- **Snippet:**
```python
x = wte(ids) + wpe(torch.arange(ids.shape[1]))
```
- **Snippet explained:** Token vector plus position vector gives the input to the first transformer block.
- **Related terms:** [[Embedding]], [[Context window]], [[Self-attention]]

### Pretraining
<!-- cat: llm -->
- **Technical definition:** Training a model from random initial weights on a very large unlabelled corpus with the next-token prediction objective.
- **Plain English:** Reading most of the internet to learn language and general knowledge.
- **Example / analogy:** Costs millions of dollars and thousands of GPUs; you normally download the result instead.
- **Snippet:**
```python
model = AutoModelForCausalLM.from_config(cfg)   # random weights = start of pretraining
```
- **Snippet explained:** from_config builds the architecture without trained weights; its loss is about ln(vocab_size).
- **Related terms:** [[Fine-tuning]], [[Pretrained model]], [[Cross-entropy loss]]

### Fine-tuning
<!-- cat: llm -->
- **Technical definition:** Continuing to train a pretrained model on a smaller task- or domain-specific dataset, usually with a small learning rate.
- **Plain English:** Sending a graduate to a short specialised course.
- **Example / analogy:** 20 steps on four pirate sentences made distilgpt2 reproduce pirate text (and overfit).
- **Snippet:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss = model(**batch, labels=labels).loss
loss.backward(); opt.step(); opt.zero_grad()
```
- **Snippet explained:** The standard training step: forward + loss, gradients, update, reset gradients.
- **Related terms:** [[Pretraining]], [[Inference]], [[Backpropagation]]

### Inference
<!-- cat: llm -->
- **Technical definition:** Using a trained model to produce outputs: forward passes only, weights frozen.
- **Plain English:** The exam, as opposed to studying.
- **Example / analogy:** Typing a prompt into a chatbot triggers inference; latency and cost matter here.
- **Snippet:**
```python
model.eval()
with torch.no_grad():
    out = model(**inputs)
```
- **Snippet explained:** eval() switches off dropout; no_grad() skips gradient bookkeeping to save memory and time.
- **Related terms:** [[eval mode and no_grad]], [[Pretraining]], [[Fine-tuning]]

### Pretrained model
<!-- cat: llm -->
- **Technical definition:** A model whose weights were already trained by someone else and are distributed (e.g. on the Hugging Face Hub) for reuse.
- **Plain English:** A ready-made brain you can use or specialise.
- **Example / analogy:** distilgpt2 is a pretrained model of 82 million parameters.
- **Snippet:**
```python
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
```
- **Snippet explained:** Downloads (once) and loads architecture plus weights into memory.
- **Related terms:** [[AutoModelForCausalLM]], [[Model config]], [[Pretraining]]

### Hugging Face Transformers
<!-- cat: llm -->
- **Technical definition:** An open-source Python library providing tokenizers, model architectures, pretrained weights and utilities (generate, Trainer) behind a uniform API.
- **Plain English:** The standard toolbox and app store for open models.
- **Example / analogy:** from transformers import AutoTokenizer, AutoModelForCausalLM
- **Snippet:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```
- **Snippet explained:** Auto* classes choose the right concrete class from the model name.
- **Related terms:** [[AutoModelForCausalLM]], [[Tokenizer]], [[model.generate()]]

### AutoModelForCausalLM
<!-- cat: llm -->
- **Technical definition:** A Hugging Face factory class that loads the correct causal (next-token) language-model architecture for a given model name.
- **Plain English:** Say the model's name and it builds the right model for you.
- **Example / analogy:** Works for GPT-2, Llama, Qwen, Mistral ... with the same one-line call.
- **Snippet:**
```python
model = AutoModelForCausalLM.from_pretrained('gpt2')
```
- **Snippet explained:** Returns a torch nn.Module with a language-model head that outputs logits over the vocabulary.
- **Related terms:** [[Hugging Face Transformers]], [[Pretrained model]], [[Causal language model]]

### Model config
<!-- cat: llm -->
- **Technical definition:** A small JSON-like description of a model's architecture (layers, hidden size, heads, vocab size, context length) separate from its weights.
- **Plain English:** The blueprint without the furniture.
- **Example / analogy:** distilgpt2: 6 layers, hidden 768, 12 heads, 1024 positions, vocab 50257.
- **Snippet:**
```python
cfg = AutoConfig.from_pretrained('distilgpt2')
print(cfg.n_positions)
```
- **Snippet explained:** Downloads only the tiny config file and reads the context length.
- **Related terms:** [[Pretrained model]], [[Transformer]], [[Context window]]

### Parameter (weight)
<!-- cat: ml -->
- **Technical definition:** A number inside the model that is adjusted during training; collectively they store what the model has learned.
- **Plain English:** The knobs the training process turns.
- **Example / analogy:** distilgpt2 has 81,912,576 of them.
- **Snippet:**
```python
sum(p.numel() for p in model.parameters())
```
- **Snippet explained:** Counts every number in every weight tensor.
- **Related terms:** [[Parameter count]], [[Backpropagation]], [[Data type (dtype)]]

### Parameter count
<!-- cat: llm -->
- **Technical definition:** The total number of trainable numbers in a model; the usual measure of model size and the starting point for memory estimates.
- **Plain English:** How many knobs the model has.
- **Example / analogy:** 82 M parameters x 4 bytes (float32) = 0.33 GB; a 7B model in float16 needs about 14 GB just for weights.
- **Snippet:**
```python
n = sum(p.numel() for p in model.parameters())
print(f'{n:,} params, {n*4/1e9:.2f} GB in float32')
```
- **Snippet explained:** numel() counts elements per tensor; multiply by bytes per element for memory.
- **Related terms:** [[Parameter (weight)]], [[Data type (dtype)]], [[Distillation]]

### Data type (dtype)
<!-- cat: ml -->
- **Technical definition:** How each number is stored: float32 (4 bytes), float16/bfloat16 (2), int8 (1), int4 (0.5). Fewer bytes means less memory but less precision.
- **Plain English:** How many decimal digits of precision you keep per number.
- **Example / analogy:** Casting a model with model.half() halves its memory.
- **Snippet:**
```python
p = next(model.parameters())
print(p.dtype, p.element_size())
```
- **Snippet explained:** Shows the dtype (float32) and its size in bytes (4).
- **Related terms:** [[Parameter count]], [[Parameter (weight)]]

### Logits
<!-- cat: llm -->
- **Technical definition:** The raw, unnormalised scores the model outputs for each vocabulary token at each position; they can be negative and do not sum to 1.
- **Plain English:** Raw points each candidate next token scored before converting to percentages.
- **Example / analogy:** Shape (1, 5, 50257) for a 5-token prompt: a score for every token at every position.
- **Snippet:**
```python
logits = model(**inputs).logits
last = logits[0, -1, :]
```
- **Snippet explained:** Take the final position's 50,257 scores: the prediction for the next token.
- **Related terms:** [[Softmax]], [[Greedy decoding]], [[Vocabulary (vocab size)]]

### Softmax
<!-- cat: ml -->
- **Technical definition:** A function turning a vector of scores into probabilities: exp(x_i) / sum_j exp(x_j).
- **Plain English:** Converts scores into percentages that add up to 100%.
- **Example / analogy:** Logits [2, 1, 0] become about [0.67, 0.24, 0.09].
- **Snippet:**
```python
probs = torch.softmax(last, dim=-1)
```
- **Snippet explained:** dim=-1 applies it across the vocabulary dimension.
- **Related terms:** [[Logits]], [[Cross-entropy loss]], [[Self-attention]]

### Cross-entropy loss
<!-- cat: ml -->
- **Technical definition:** The negative log-probability the model assigned to the correct next token, averaged over positions; the standard training loss for classification and language modelling.
- **Plain English:** A 'surprise meter': low when the model expected what happened.
- **Example / analogy:** A random model on 50,257 tokens has loss ln(50257) = 10.8; pretrained distilgpt2 scored 5.7 on a simple prompt.
- **Snippet:**
```python
loss = model(**inputs, labels=inputs['input_ids']).loss
```
- **Snippet explained:** Passing labels makes the model shift them by one position and compute the average cross-entropy.
- **Related terms:** [[Perplexity]], [[Softmax]], [[Backpropagation]]

### Perplexity
<!-- cat: llm -->
- **Technical definition:** exp(average cross-entropy loss): the effective number of equally likely tokens the model is choosing between per position.
- **Plain English:** How confused the model is - lower is better.
- **Example / analogy:** Perplexity 244 on 'The cat sat on the mat.' means about 244 plausible choices per token for distilgpt2.
- **Snippet:**
```python
import math
ppl = math.exp(loss.item())
```
- **Snippet explained:** Converts loss from log-space into an intuitive 'branching factor'.
- **Related terms:** [[Cross-entropy loss]], [[Logits]]

### Greedy decoding
<!-- cat: llm -->
- **Technical definition:** Always choosing the highest-probability token at each step (argmax).
- **Plain English:** Always taking the most obvious next word.
- **Example / analogy:** Deterministic but prone to dull or repetitive text such as 'I like pizza. I like pizza...'.
- **Snippet:**
```python
next_id = logits[0, -1].argmax()
```
- **Snippet explained:** argmax returns the index (token id) of the largest score.
- **Related terms:** [[Autoregressive generation]], [[Logits]], [[model.generate()]]

### Autoregressive generation
<!-- cat: llm -->
- **Technical definition:** Producing a sequence one token at a time, feeding each new token back as input for the next prediction.
- **Plain English:** Writing a sentence word by word, re-reading everything so far each time.
- **Example / analogy:** Generating 100 tokens requires 100 sequential forward passes - why latency grows with output length.
- **Snippet:**
```python
for _ in range(n):
    nxt = model(ids).logits[0, -1].argmax().view(1, 1)
    ids = torch.cat([ids, nxt], dim=1)
```
- **Snippet explained:** Predict, append, repeat.
- **Related terms:** [[Greedy decoding]], [[KV cache]], [[model.generate()]]

### KV cache
<!-- cat: llm -->
- **Technical definition:** Stored key and value tensors of all previous tokens in every layer, reused at each decoding step so only the newest token is computed.
- **Plain English:** Keeping your notes from earlier so you do not re-read the whole book each time.
- **Example / analogy:** 60 tokens took 1.40 s without cache and 0.81 s with it on distilgpt2; the cache's memory grows with context length and batch size.
- **Snippet:**
```python
model.generate(ids, max_new_tokens=60, use_cache=True)
```
- **Snippet explained:** use_cache=True (the default) enables the cache.
- **Related terms:** [[Autoregressive generation]], [[Context window]], [[model.generate()]]

### model.generate()
<!-- cat: llm -->
- **Technical definition:** Hugging Face's complete decoding loop: handles caching, sampling options, stopping criteria and batching.
- **Plain English:** The library's ready-made 'write the next N tokens' button.
- **Example / analogy:** With do_sample=False it matches a hand-written greedy loop exactly.
- **Snippet:**
```python
out = model.generate(ids, max_new_tokens=15, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0]))
```
- **Snippet explained:** Returns prompt plus new ids; pad_token_id silences GPT-2's missing-pad warning.
- **Related terms:** [[Greedy decoding]], [[KV cache]], [[Special token (EOS)]]

### eval mode and no_grad
<!-- cat: ml -->
- **Technical definition:** model.eval() turns off training-only behaviour (dropout); torch.no_grad() stops gradient tracking. They are separate and normally used together for inference.
- **Plain English:** 'Exam mode' for behaviour and 'don't take notes' for memory.
- **Example / analogy:** Forgetting no_grad on large inputs wastes memory storing a computation graph you never use.
- **Snippet:**
```python
model.eval()
with torch.no_grad():
    logits = model(ids).logits
```
- **Snippet explained:** eval() changes layer behaviour; no_grad() is a context manager that restores tracking on exit.
- **Related terms:** [[Inference]], [[Dropout]], [[Backpropagation]]

### Dropout
<!-- cat: ml -->
- **Technical definition:** A regularisation technique that randomly zeroes some activations during training to reduce overfitting; disabled at inference.
- **Plain English:** Making students work with random teammates absent so nobody relies on one person.
- **Example / analogy:** model.train() enables it; model.eval() disables it.
- **Snippet:**
```python
model.train()   # dropout active
model.eval()    # dropout off
```
- **Snippet explained:** Switches the mode flag for every layer in the model.
- **Related terms:** [[eval mode and no_grad]], [[Fine-tuning]]

### Distillation
<!-- cat: llm -->
- **Technical definition:** Training a smaller 'student' model to imitate a larger 'teacher', yielding a faster, smaller model with slightly lower quality.
- **Plain English:** A student learning from a professor's answers rather than from raw textbooks.
- **Example / analogy:** distilgpt2 has 6 layers vs gpt2's 12 and 82 M vs 124 M parameters.
- **Snippet:**
```python
AutoConfig.from_pretrained('distilgpt2').n_layer   # 6
```
- **Snippet explained:** Reads the layer count to compare with gpt2's 12.
- **Related terms:** [[Parameter count]], [[Pretrained model]]

### Causal language model
<!-- cat: llm -->
- **Technical definition:** A model trained to predict the next token using only previous tokens (GPT, Llama, Qwen). Contrasts with masked language models (BERT).
- **Plain English:** A writer who only sees what is already written.
- **Example / analogy:** AutoModelForCausalLM loads this kind; used for text generation.
- **Snippet:**
```python
AutoModelForCausalLM.from_pretrained('gpt2')
```
- **Snippet explained:** The class name tells you the model predicts causally (left to right).
- **Related terms:** [[Causal mask]], [[AutoModelForCausalLM]], [[Autoregressive generation]]

### Label masking (-100)
<!-- cat: ml -->
- **Technical definition:** Setting label values to -100 so PyTorch's cross-entropy ignores those positions, commonly used for padding (or for prompt tokens in instruction tuning).
- **Plain English:** A 'do not grade this part' marker.
- **Example / analogy:** Without it, padding tokens contribute meaningless loss.
- **Snippet:**
```python
labels = batch['input_ids'].clone()
labels[batch['attention_mask'] == 0] = -100
```
- **Snippet explained:** Copy the ids as labels, then hide padded positions from the loss.
- **Related terms:** [[Padding and attention mask]], [[Cross-entropy loss]], [[Fine-tuning]]

### Backpropagation
<!-- cat: ml -->
- **Technical definition:** The algorithm that computes the gradient of the loss with respect to every parameter by applying the chain rule backwards through the network.
- **Plain English:** Working out how much each knob contributed to the error.
- **Example / analogy:** loss.backward() fills p.grad for each parameter; the optimizer then uses it.
- **Snippet:**
```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
- **Snippet explained:** Compute gradients, update weights, clear old gradients.
- **Related terms:** [[Parameter (weight)]], [[Cross-entropy loss]], [[Fine-tuning]]
