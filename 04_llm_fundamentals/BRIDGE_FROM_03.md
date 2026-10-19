# Bridge: from 03 ML Foundations to 04 LLM Fundamentals

Read this between `03/08_sklearn_text_classifier` and the first step of 04. Then read the one-line **cliffhanger notes** at the end of each chapter in Act II to see why the next folder exists.

## The story so far (Act I in one minute)
You can hold data as arrays, write gradient descent by hand, let PyTorch do the gradients, spot overfitting with a validation set, and turn scores into probabilities with softmax and cross-entropy. In step 08 you applied all of it to text and got a classifier that *works*.

## The villain: what TF-IDF cannot do
TF-IDF gave every sentence one slot per vocabulary word. Two failures:
1. **No meaning.** `king` and `queen` are different columns, so their similarity is exactly 0 (run `../03_ml_foundations/08_sklearn_text_classifier/flaw_demo.py`; `02_vectors_and_similarity` shows `car` vs `automobile` = 0.0).
2. **No order.** `dog bites man` and `man bites dog` produce the same vector (`02_vectors_and_similarity`: `driver parks car` = `car parks driver`).

And it can only *classify*; it cannot *write*.

## The hero: learn the representation, then predict the next token
Instead of hand-designed columns, let a network **learn** a dense vector per token and learn to predict the next token from context. To do that, the whole path from raw text to output must be built, in the order data actually flows:

```
raw text -> tokens -> ids -> vectors (+ positions) -> attention x N blocks -> logits -> next token -> loop
  Ch 7        Ch 7    Ch 7      Ch 8                    Ch 9-10               Ch 11     Ch 11      Ch 12
```
Warm-up first: [00_what_is_an_llm](00_what_is_an_llm/) shows the full loop with a counting toy; [02_vectors_and_similarity](02_vectors_and_similarity/) shows the dense-vector fix and its remaining limits.

## Reading order for 04 (data-flow order; folder numbers match)
| Order | Folder | Chapter | Why it comes here |
|---|---|---|---|
| 0 | `00_what_is_an_llm` | prologue | the loop, with no maths |
| 1 | `01_tokenizer` | Ch 7 | raw text must become ids before anything else |
| 2 | `02_vectors_and_similarity` | Ch 8 warm-up | dense vectors fix TF-IDF's blindness to meaning |
| 3 | `03_embeddings` | Ch 8 | ids need learned geometry and a sense of position |
| 4 | `04_attention_from_scratch` | Ch 9 | the vectors are now real token vectors; attention gives them context |
| 5 | `05_transformer_block` | Ch 10 | stack attention into a deep, stable block |
| 6 | `06_load_pretrained_model` | Ch 10 | inspect a real model: weights, parameters, memory |
| 7 | `07_forward_pass_logits` | Ch 11 | see the output |
| 8 | `08_generation_loop` | Ch 12 | repeat it, then make it fast with the KV cache |
| 9 | `09_pretrain_vs_finetune_vs_inference` | Ch 13 | how the weights got there |

## Cliffhanger notes (add these as the closing section of each step)
- **Ch 7 Tokenizer ->** Ids are arbitrary integers; id 464 is no closer in meaning to 465 than to 40,000. *We need geometry.*
- **Ch 8 Embeddings ->** Each token has one fixed vector, so `bank` means the same everywhere, and a bag of vectors has no order. *We need tokens to read each other.*
- **Ch 9 Attention ->** One attention step mixes information once. Real understanding needs depth, stability and per-token thinking. *We need to stack and stabilise it: `05_transformer_block`.*
- **Ch 10 Blocks / pretrained model ->** The network exists and has millions of weights. *What does it output, and how do we read it?*
- **Ch 11 Forward pass ->** One pass yields one next-token distribution. *We want a whole answer: loop.*
- **Ch 12 Generation ->** Each step re-reads the full history (quadratic slowdown, solved by the KV cache), but greedy text is repetitive and the cache eats memory. *We need control (Act III) and a story for where these weights came from.*
- **Ch 13 Life cycle ->** We know pretraining, fine-tuning, inference. *Next: how to control output, shrink the model, and tune it cheaply (module 05 and 06).*
