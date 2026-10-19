# Exercises - 04 LLM Fundamentals

Work in this order. For each step: (1) do the **Modify** task in a copy of `main.py`, (2) **Break it** and read the traceback before looking anything up, (3) **Rebuild** with the file closed, (4) **Explain aloud** (record yourself, 60-90 s). Worked answers: [solutions/](solutions/) - peek only after a real attempt.

Rule: change ONE thing at a time and predict the result before you run.

## Step 00 - What is an LLM?
- **Modify:** make the toy look at the last TWO words (use a tuple key). Compare generated text and table size with the one-word version.
- **Predict first:** after adding "the cat sat on the rug ." to the corpus, does P(rug | the) go up or down? Check.
- **Break it:** call `next_token_probs("elephant")`; read the `ZeroDivisionError` and say why a real tokenizer avoids it.
- **Rebuild:** count -> probabilities -> generate loop, with the file closed.
- **Explain aloud:** "What does an LLM do, and what is the one line a transformer replaces in my toy?"

## Step 01 - Tokenizer
- **Modify:** write `token_report(text)` returning characters, words, tokens, tokens-per-word, and a cost estimate. Compare English, a code snippet, and Hindi/your language. -> `solutions/ex01_token_report.py`
- **Predict first:** which of `"hello"`, `" hello"`, `"Hello"`, `"HELLO"` is 1 token? Check.
- **Break it:** `tok.decode("hello")`; then a misspelled model name; read both errors.
- **Rebuild:** encode -> pieces -> decode -> assert equality; batch with padding.
- **Explain aloud:** "Why do Indian-language prompts cost more with this tokenizer?"

## Step 02 - Vectors and similarity
- **Modify:** write `most_similar(word, k)` over the dense vectors; test it on `car` and `cat`. Then do the same for the TF-IDF word vectors and compare.
- **Experiment:** vary `k` (2, 4, 8) and remove `STOP`; record how car~kitten changes and explain why.
- **Break it:** `sentence_vec("zebra")`; read the warning and name the problem (out-of-vocabulary).
- **Rebuild:** `cosine` and the co-occurrence matrix, file closed.
- **Explain aloud:** "Why do `driver parks car` and `car parks driver` get the same averaged vector, and what do we need instead?"

## Step 03 - Embeddings
- **Modify:** write `nearest(word, k)` returning the k most similar vocabulary tokens by cosine similarity. -> `solutions/ex03_nearest_words.py`
- **Experiment:** is `cosine(" king", " queen")` bigger than `cosine(" king", " throne")`? Why can't you trust one number?
- **Break it:** use an id outside the table; use a multi-token word.
- **Rebuild:** `nn.Embedding(5, 3)` lookup and position addition.
- **Explain aloud:** "Where does the meaning in an embedding come from?"

## Step 04 - Attention
- **Modify:** turn the code into a function `attention(x, W_q, W_k, W_v, causal=True)` that returns `(output, weights)`. Make `d_head` (4) different from `d_model` (16). -> `solutions/ex04_attention_function.py`
- **Prove causality:** change a *future* token and assert that earlier outputs don't change.
- **Break it:** make `W_k` shape `(7, 8)`; write down which dimensions mismatch and which line failed.
- **Rebuild:** softmax -> scores -> mask -> weights -> output, from memory.
- **Explain aloud:** "What do Q, K, V mean and why divide by sqrt(d)?"

## Step 05 - Transformer block
- **Modify:** change `n_heads` from 4 to 2 and 8 with `d_model` fixed; confirm the parameter count of a block does not change (why?).
- **Experiment:** stack 2, 6 and 12 blocks and print the standard deviation of the activations after each; then remove LayerNorm and repeat.
- **Break it:** set `d_model=30, n_heads=4`; read the error and say which line failed.
- **Rebuild:** one block (LayerNorm -> attention -> residual -> LayerNorm -> MLP -> residual) from memory.
- **Explain aloud:** "Why do transformers need residual connections and LayerNorm?"

## Step 06 - Loading a model
- **Modify:** write `breakdown(model)` giving the parameter share of embeddings vs attention vs MLP, for distilgpt2 and gpt2. -> `solutions/ex06_param_breakdown.py`
- **Calculate by hand then verify:** memory of a 1.3B model in float32/float16/int8; the KV cache of gpt2 at 1024 tokens (see 05 step 04 formula).
- **Break it:** load a model name with a typo; then (optional) disconnect the network and load an un-cached model.
- **Rebuild:** `count_params`.
- **Explain aloud:** "What is the difference between config and weights?"

## Step 07 - Forward pass
- **Modify:** compute **perplexity** (`exp(loss)`) of several sentences; try a natural sentence, a scrambled one and gibberish. Explain any surprise (gibberish splits into many easy sub-word pieces!). -> `solutions/ex07_perplexity.py`
- **Experiment:** print the entropy of the next-token distribution for 4 different prompts; rank by uncertainty.
- **Break it:** pass a python list instead of tensors to the model.
- **Rebuild:** tokenise -> forward -> last logits -> softmax -> top-5.
- **Explain aloud:** "Why does the output have shape (1, seq_len, 50257)?"

## Step 08 - Generation loop
- **Modify:** `generate_until(prompt, stop, max_new_tokens)` with a stop string as well as EOS. -> `solutions/ex08_generate_until_stop.py`
- **Experiment:** time cached vs uncached generation for 20, 60, 120 new tokens and sketch the curves.
- **Break it:** remove `.view(1, 1)`; add a `print(shape)` before the failing line.
- **Rebuild:** the greedy loop in <10 lines.
- **Explain aloud:** "Why does the KV cache speed up decoding and what does it cost?"

## Step 09 - Pretrain / fine-tune / inference
- **Modify:** fine-tune on 4 sentences of a different style (formal legal English, or Hinglish); sample before/after.
- **Experiment:** `lr=5e-3` instead of `5e-5`; record the loss per step; then `range(3)` steps vs `range(30)` steps.
- **Break it:** remove `optimizer.zero_grad()`; describe how you would *notice* the bug without a traceback.
- **Rebuild:** the 5-line training loop with comments.
- **Explain aloud:** "How do I know a fine-tuned model has overfit?" (answer: held-out validation loss).

## Capstone for this module (30 min)
Write one script `my_llm_tour.py` from a blank file, no copy-paste: tokenise a sentence, print ids, load distilgpt2, print parameter count, print top-5 next tokens, then greedily generate 10 tokens with your own loop. If you get stuck, note exactly *what* you didn't know - that goes into your vocabulary file.
