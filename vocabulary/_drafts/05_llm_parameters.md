<!-- vocabulary draft: 05_llm_parameters -->

### Learning rate
<!-- cat: ml -->
- **Technical definition:** The scalar multiplying the gradient in each weight update; it sets the step size of optimisation.
- **Plain English:** How big a step you take downhill each time.
- **Example / analogy:** On a toy problem lr=1e-4 barely learned, lr=0.1 trained well, lr=10 diverged. LLM fine-tuning uses small values (1e-5 to 5e-4).
- **Snippet:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
```
- **Snippet explained:** lr is passed to the optimizer, which applies weight -= step_size_from_gradient on every opt.step().
- **Related terms:** [[Optimizer]], [[Learning-rate scheduler (warm-up)]], [[Hyperparameter]]

### Batch size
<!-- cat: ml -->
- **Technical definition:** The number of training examples processed together before one weight update.
- **Plain English:** How many exercises you mark before adjusting your study method.
- **Example / analogy:** Batch 4 gave 75 updates per epoch of 300 examples; batch 300 gave 1. Bigger batches need more GPU memory.
- **Snippet:**
```python
for i in range(0, len(X), batch_size):
    idx = perm[i:i + batch_size]
```
- **Snippet explained:** Steps through the shuffled data in slices of batch_size.
- **Related terms:** [[Epoch]], [[Iteration (step)]], [[Gradient accumulation]]

### Epoch
<!-- cat: ml -->
- **Technical definition:** One complete pass through the entire training dataset.
- **Plain English:** Reading the whole textbook once.
- **Example / analogy:** Fine-tuning LLMs typically uses 1-3 epochs; many more on small data overfits.
- **Snippet:**
```python
for epoch in range(30):
    ...  # one pass over all batches
```
- **Snippet explained:** The outer loop counts passes; the inner loop walks through batches.
- **Related terms:** [[Batch size]], [[Overfitting]], [[Early stopping]]

### Iteration (step)
<!-- cat: ml -->
- **Technical definition:** One weight update, i.e. processing one batch (forward, loss, backward, optimizer step).
- **Plain English:** One tweak of the knobs.
- **Example / analogy:** Steps per epoch = ceil(dataset size / batch size).
- **Snippet:**
```python
steps_per_epoch = -(-300 // 32)   # ceiling division = 10
```
- **Snippet explained:** Negative floor-division trick for rounding up.
- **Related terms:** [[Batch size]], [[Epoch]]

### Optimizer
<!-- cat: ml -->
- **Technical definition:** The algorithm that converts gradients into weight updates (SGD, momentum, Adam, AdamW).
- **Plain English:** The strategy a hiker uses to walk downhill using the slope information.
- **Example / analogy:** SGD 0.39 vs Adam 0.13 final loss on the same toy task and learning rate.
- **Snippet:**
```python
opt = torch.optim.Adam(model.parameters(), lr=0.01)
```
- **Snippet explained:** Constructed with the parameters to update and the learning rate.
- **Related terms:** [[SGD (stochastic gradient descent)]], [[Adam and AdamW]], [[Learning rate]]

### SGD (stochastic gradient descent)
<!-- cat: ml -->
- **Technical definition:** Update rule weight -= lr * gradient computed on a mini-batch; optional momentum smooths updates.
- **Plain English:** The simplest downhill rule: step against the slope.
- **Example / analogy:** Needs careful learning-rate tuning; momentum=0.9 improved the toy loss from 0.39 to 0.24.
- **Snippet:**
```python
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```
- **Snippet explained:** momentum keeps part of the previous step so updates are smoother.
- **Related terms:** [[Optimizer]], [[Adam and AdamW]], [[Learning rate]]

### Adam and AdamW
<!-- cat: ml -->
- **Technical definition:** Optimizers that track running averages of gradients and squared gradients to give each parameter its own adaptive step size; AdamW applies weight decay separately (decoupled).
- **Plain English:** A hiker with a personal pace per leg of the journey.
- **Example / analogy:** The default for LLM training; stores two extra numbers per parameter, so optimizer memory is about 2x the weights.
- **Snippet:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
```
- **Snippet explained:** weight_decay shrinks weights slightly each step to curb overfitting.
- **Related terms:** [[Optimizer]], [[Weight decay]], [[Learning rate]]

### Weight decay
<!-- cat: ml -->
- **Technical definition:** A regularisation penalty that gently pulls weights toward zero each update.
- **Plain English:** A small tax on big weights, discouraging over-complicated solutions.
- **Example / analogy:** AdamW(weight_decay=0.01) is a common LLM fine-tuning setting.
- **Snippet:**
```python
torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
```
- **Snippet explained:** Passed as an optimizer argument.
- **Related terms:** [[Adam and AdamW]], [[Overfitting]]

### Loss function
<!-- cat: ml -->
- **Technical definition:** A function that scores how wrong the model's predictions are; training minimises it.
- **Plain English:** The score the training process tries to lower.
- **Example / analogy:** Cross-entropy for classification and next-token prediction; mean squared error for regression.
- **Snippet:**
```python
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(model(X), y)
```
- **Snippet explained:** Takes raw logits and integer class labels; applies softmax internally.
- **Related terms:** [[Optimizer]], [[Overfitting]]

### Overfitting
<!-- cat: ml -->
- **Technical definition:** When a model fits the training data (including noise) so closely that it performs worse on unseen data; shown by falling training loss and rising validation loss.
- **Plain English:** Memorising the practice answers instead of learning the subject.
- **Example / analogy:** In the epoch experiment train loss dropped to 0.03 while validation loss climbed from 0.27 to 0.68.
- **Snippet:**
```python
print(train_loss, val_loss)   # watch the gap grow
```
- **Snippet explained:** Always evaluate on data the model did not train on.
- **Related terms:** [[Validation set]], [[Early stopping]], [[Epoch]]

### Underfitting
<!-- cat: ml -->
- **Technical definition:** When a model is too simple or trained too little to capture the pattern; both training and validation loss stay high.
- **Plain English:** Not studying enough.
- **Example / analogy:** lr=1e-4 for 30 epochs left training loss at 0.73 (almost no learning).
- **Snippet:**
```python
# loss stays near its starting value
```
- **Snippet explained:** Fix with more epochs, bigger model or a larger learning rate.
- **Related terms:** [[Overfitting]], [[Learning rate]], [[Epoch]]

### Validation set
<!-- cat: ml -->
- **Technical definition:** A portion of data held out from training and used to monitor generalisation and tune hyperparameters.
- **Plain English:** Practice exam questions you never trained on.
- **Example / analogy:** The first 300 points trained the model; the next 300 validated it.
- **Snippet:**
```python
X_train, X_val = X[:300], X[300:]
```
- **Snippet explained:** Slicing splits data into two non-overlapping parts.
- **Related terms:** [[Overfitting]], [[Early stopping]], [[Hyperparameter]]

### Gradient accumulation
<!-- cat: ml -->
- **Technical definition:** Summing gradients over several small batches before one optimizer step, emulating a larger batch with less memory.
- **Plain English:** Marking five small piles of homework before changing your study method once.
- **Example / analogy:** Effective batch 32 = batch 8 x 4 accumulation steps.
- **Snippet:**
```python
loss = loss / 4
loss.backward()
if (i + 1) % 4 == 0:
    opt.step(); opt.zero_grad()
```
- **Snippet explained:** Gradients add up in .grad until zero_grad is called after every 4th batch.
- **Related terms:** [[Batch size]], [[Iteration (step)]]

### Learning-rate scheduler (warm-up)
<!-- cat: ml -->
- **Technical definition:** A rule changing the learning rate during training, commonly a short warm-up from near zero followed by a gradual decay.
- **Plain English:** Starting a run slowly, then easing off toward the finish.
- **Example / analogy:** Warm-up avoids huge early updates that can destabilise a pretrained model.
- **Snippet:**
```python
sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=10)
sched.step()
```
- **Snippet explained:** Call sched.step() after each optimizer step to update the learning rate.
- **Related terms:** [[Learning rate]], [[Optimizer]]

### Early stopping
<!-- cat: ml -->
- **Technical definition:** Halting training when validation performance stops improving, keeping the best checkpoint.
- **Plain English:** Stopping revision when practice-test scores stop rising.
- **Example / analogy:** In the epoch table the best validation loss occurred around epoch 5.
- **Snippet:**
```python
if val_loss < best:
    best = val_loss; save(model)
else:
    patience -= 1
```
- **Snippet explained:** Track the best score and a patience counter.
- **Related terms:** [[Overfitting]], [[Validation set]], [[Epoch]]

### Hyperparameter
<!-- cat: ml -->
- **Technical definition:** A setting chosen by the engineer rather than learned by training (learning rate, batch size, epochs, temperature, top_p).
- **Plain English:** The dials you set before the model starts learning.
- **Example / analogy:** config.json holds the hyperparameters of a run.
- **Snippet:**
```python
cfg['training']['learning_rate']
```
- **Snippet explained:** Read a hyperparameter from a nested dict.
- **Related terms:** [[Configuration file]], [[Learning rate]], [[Reproducibility]]

### Temperature (sampling)
<!-- cat: llm -->
- **Technical definition:** A scalar T dividing logits before softmax: T<1 sharpens the distribution, T>1 flattens it, T->0 approaches greedy decoding.
- **Plain English:** A creativity dial: cold = predictable, hot = adventurous.
- **Example / analogy:** For 'The best way to learn programming is', p(' to') was 0.82 at T=0.7 but 0.16 at T=1.5.
- **Snippet:**
```python
probs = torch.softmax(logits / temperature, dim=-1)
```
- **Snippet explained:** Dividing before softmax scales gaps between scores; the order of tokens never changes.
- **Related terms:** [[Top-k sampling]], [[Top-p (nucleus) sampling]], [[Entropy (of a distribution)]]

### Top-k sampling
<!-- cat: llm -->
- **Technical definition:** Restricting sampling to the k highest-probability tokens and renormalising.
- **Plain English:** Choose only from the shortlist of the k best candidates.
- **Example / analogy:** top_k=5 left 5 tokens alive; top_k=1 equals greedy.
- **Snippet:**
```python
kth = torch.topk(logits, k).values[-1]
logits = torch.where(logits < kth, float('-inf'), logits)
```
- **Snippet explained:** Everything below the k-th largest score is set to -inf, so softmax gives it 0.
- **Related terms:** [[Temperature (sampling)]], [[Top-p (nucleus) sampling]], [[Sampling (decoding)]]

### Top-p (nucleus) sampling
<!-- cat: llm -->
- **Technical definition:** Restricting sampling to the smallest set of tokens whose cumulative probability is at least p.
- **Plain English:** Keep adding the best candidates until you cover p% of the probability.
- **Example / analogy:** top_p=0.9 kept 15 tokens for one prompt and thousands for an uncertain one.
- **Snippet:**
```python
cum = torch.cumsum(sorted_probs, -1)
remove = (cum - sorted_probs) >= p
```
- **Snippet explained:** A token is removed if the probability mass before it already reached p.
- **Related terms:** [[Top-k sampling]], [[Min-p sampling]], [[Temperature (sampling)]]

### Min-p sampling
<!-- cat: llm -->
- **Technical definition:** Keeping tokens whose probability is at least min_p times the top token's probability, so the cut-off scales with model confidence.
- **Plain English:** Ignore anything far less likely than the favourite.
- **Example / analogy:** With min_p=0.1 a confident prompt kept 11 tokens, adapting to the distribution.
- **Snippet:**
```python
keep = probs >= min_p * probs.max()
```
- **Snippet explained:** Boolean mask comparing every probability with a threshold derived from the maximum.
- **Related terms:** [[Top-p (nucleus) sampling]], [[Top-k sampling]]

### Sampling (decoding)
<!-- cat: llm -->
- **Technical definition:** Choosing the next token randomly according to the (processed) probability distribution instead of always taking the maximum.
- **Plain English:** Rolling a weighted die for each next word.
- **Example / analogy:** do_sample=True in generate(); the same prompt can give different outputs per run.
- **Snippet:**
```python
next_id = torch.multinomial(probs, num_samples=1)
```
- **Snippet explained:** Draws one index with chance proportional to its probability.
- **Related terms:** [[Greedy decoding]], [[Temperature (sampling)]], [[Seed (random seed)]]

### Beam search
<!-- cat: llm -->
- **Technical definition:** A decoding method that keeps the best num_beams partial sequences at every step and returns the highest-scoring complete one.
- **Plain English:** Exploring several promising sentence drafts at once.
- **Example / analogy:** model.generate(ids, num_beams=4, do_sample=False) explores 4 paths; slower but often more generic.
- **Snippet:**
```python
model.generate(ids, num_beams=4, do_sample=False)
```
- **Snippet explained:** num_beams sets how many candidate sequences are retained.
- **Related terms:** [[Greedy decoding]], [[Sampling (decoding)]]

### Repetition penalty
<!-- cat: llm -->
- **Technical definition:** A decoding adjustment that lowers the score of tokens already generated (positive logits divided by the penalty, negative ones multiplied).
- **Plain English:** A nudge saying 'try not to repeat yourself'.
- **Example / analogy:** Greedy 'I like pizza. I like pizza...' loops at penalty 1.0 but not at 1.3.
- **Snippet:**
```python
model.generate(ids, do_sample=False, repetition_penalty=1.3)
```
- **Snippet explained:** Values above 1 discourage repeats; too high avoids necessary words.
- **Related terms:** [[Greedy decoding]], [[Temperature (sampling)]]

### max_new_tokens
<!-- cat: llm -->
- **Technical definition:** The maximum number of tokens generate() may add after the prompt; generation can end earlier on the end-of-text token.
- **Plain English:** A word-limit for the answer.
- **Example / analogy:** max_new_tokens=10 took 0.18 s, 83 tokens took 1.44 s: latency grows with length.
- **Snippet:**
```python
model.generate(ids, max_new_tokens=20)
```
- **Snippet explained:** Counts only new tokens, unlike max_length which includes the prompt.
- **Related terms:** [[Context window]], [[Latency]], [[Special token (EOS)]]

### Context window
<!-- cat: llm -->
- **Technical definition:** The maximum number of tokens (prompt plus output) a model can attend to at once.
- **Plain English:** The model's short-term memory size.
- **Example / analogy:** GPT-2: 1024 tokens; a 1030-token input raised IndexError. Modern models range from 8k to over 1M.
- **Snippet:**
```python
budget = 1024 - max_new_tokens
prompt_ids = prompt_ids[:, -budget:]
```
- **Snippet explained:** Keep only the most recent tokens so prompt plus output fits.
- **Related terms:** [[max_new_tokens]], [[KV cache size]], [[Position embedding]]

### KV cache size
<!-- cat: llm -->
- **Technical definition:** Memory used by cached keys and values: 2 x layers x hidden size x bytes x tokens x batch.
- **Plain English:** How much notebook space the model needs for everything it has read so far.
- **Example / analogy:** A 7B-class model at 4096 tokens, batch 16, fp16 needs about 34 GB for the cache alone.
- **Snippet:**
```python
gb = 2 * layers * hidden * 2 * tokens * batch / 1e9
```
- **Snippet explained:** The first 2 is for keys and values, the second 2 is bytes per float16 number.
- **Related terms:** [[KV cache]], [[Context window]], [[Memory-bandwidth bound]]

### Entropy (of a distribution)
<!-- cat: ml -->
- **Technical definition:** -sum p log p: a measure of uncertainty; 0 for a certain outcome, log(N) for a uniform one over N options.
- **Plain English:** How surprised you expect to be by the next token.
- **Example / analogy:** Entropy was 0.01 at T=0.2 and 10.28 at T=3.0 (uniform over 50,257 would be 10.8).
- **Snippet:**
```python
H = -(p[p > 0] * p[p > 0].log()).sum()
```
- **Snippet explained:** Ignore zero probabilities (log 0 is undefined) and sum p*log p.
- **Related terms:** [[Temperature (sampling)]], [[Perplexity]]

### Quantization
<!-- cat: llm -->
- **Technical definition:** Storing model weights (and sometimes activations) with fewer bits, e.g. int8 or int4 instead of float32, using scale factors to map back to real values.
- **Plain English:** Rounding every number to fewer digits to save space.
- **Example / analogy:** int8 shrank distilgpt2's matrices 4x with almost unchanged loss (4.98 to 4.96); int4 per-tensor ruined it (9.7).
- **Snippet:**
```python
q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
w_hat = q * scale
```
- **Snippet explained:** Divide by the scale, round to integers, store as int8; multiply back to use.
- **Related terms:** [[Symmetric quantization and scale]], [[Group-wise quantization]], [[Memory-bandwidth bound]]

### Symmetric quantization and scale
<!-- cat: llm -->
- **Technical definition:** Quantization where the integer range is centred on zero and scale = max(|w|) / (2^(bits-1) - 1).
- **Plain English:** A ruler whose tick spacing is set by the biggest number you must represent.
- **Example / analogy:** For int8 and max weight 1.274 the scale is 0.01003 and the weight maps to integer 127.
- **Snippet:**
```python
scale = np.abs(w).max() / 127
```
- **Snippet explained:** The largest absolute weight maps exactly to the largest integer.
- **Related terms:** [[Quantization]], [[Outlier weights]], [[Group-wise quantization]]

### Outlier weights
<!-- cat: llm -->
- **Technical definition:** A few weights or activations much larger than typical ones; they stretch the quantization scale and wreck precision for all other values.
- **Plain English:** One giant in a crowd forcing everyone's measurements into coarse units.
- **Example / analogy:** One weight of 5.0 raised int8 mean error on the rest from 0.00019 to 0.0098.
- **Snippet:**
```python
W2[0, 0] = 5.0   # inject an outlier
```
- **Snippet explained:** A single large number changes max(|w|) and therefore the scale for the whole tensor.
- **Related terms:** [[Symmetric quantization and scale]], [[Group-wise quantization]]

### Group-wise quantization
<!-- cat: llm -->
- **Technical definition:** Quantizing small blocks (e.g. 64 weights) each with their own scale so an outlier only affects its own block.
- **Plain English:** Measuring with a separate ruler per small group instead of one ruler for everyone.
- **Example / analogy:** int4 loss: 9.69 per-tensor vs 5.13 group-wise (group 64); cost is one extra scale per group.
- **Snippet:**
```python
flat = w.reshape(-1, 64)
scales = np.abs(flat).max(axis=1, keepdims=True) / 7
```
- **Snippet explained:** Each row of 64 weights gets its own scale.
- **Related terms:** [[Quantization]], [[Outlier weights]]

### Memory-bandwidth bound
<!-- cat: llm -->
- **Technical definition:** A workload whose speed is limited by how fast data can be moved from memory to the compute units rather than by arithmetic speed; typical for LLM token generation.
- **Plain English:** A chef limited by how fast ingredients arrive, not by how fast they can chop.
- **Example / analogy:** Quantization speeds up decoding mainly because fewer bytes of weights must be read per token.
- **Snippet:** n/a
- **Snippet explained:** n/a
- **Related terms:** [[Quantization]], [[Throughput]], [[Latency]]

### Latency
<!-- cat: llm -->
- **Technical definition:** The time to complete one request (or to produce the first/each token).
- **Plain English:** How long you wait for your answer.
- **Example / analogy:** Median of three timed runs: 0.73 s for 40 tokens on distilgpt2 (CPU).
- **Snippet:**
```python
t0 = time.perf_counter()
model.generate(ids, max_new_tokens=40)
print(time.perf_counter() - t0)
```
- **Snippet explained:** perf_counter is a high-resolution stopwatch; subtract before and after.
- **Related terms:** [[Throughput]], [[Tokens per second]], [[max_new_tokens]]

### Throughput
<!-- cat: llm -->
- **Technical definition:** The amount of work completed per unit time, e.g. tokens or requests per second across all users.
- **Plain English:** How many customers the restaurant serves per hour.
- **Example / analogy:** Batching requests raises throughput but may increase each user's latency.
- **Snippet:** n/a
- **Snippet explained:** n/a
- **Related terms:** [[Latency]], [[Tokens per second]], [[Batch size]]

### Tokens per second
<!-- cat: llm -->
- **Technical definition:** A speed measure: generated tokens divided by elapsed time.
- **Plain English:** The model's typing speed.
- **Example / analogy:** distilgpt2 reached about 55 tokens/s on an M1 CPU in the experiment logger; gpt2 about 40.
- **Snippet:**
```python
tps = n_new_tokens / latency
```
- **Snippet explained:** Divide the count of new tokens by seconds.
- **Related terms:** [[Latency]], [[Throughput]]

### Seed (random seed)
<!-- cat: ml -->
- **Technical definition:** A number initialising a pseudo-random generator so that 'random' results can be repeated.
- **Plain English:** Starting a shuffled deck from the same arrangement every time.
- **Example / analogy:** torch.manual_seed(7) before each generate() call isolates the effect of the parameter you changed.
- **Snippet:**
```python
torch.manual_seed(7)
```
- **Snippet explained:** Fixes PyTorch's global random stream.
- **Related terms:** [[Sampling (decoding)]], [[Reproducibility]]

### Reproducibility
<!-- cat: ml -->
- **Technical definition:** Being able to get the same results by rerunning the experiment: fixed seeds, recorded configs, pinned library versions.
- **Plain English:** A recipe precise enough that anyone gets the same cake.
- **Example / analogy:** config.json plus fixed seeds let you compare runs fairly.
- **Snippet:**
```python
torch.manual_seed(0)
```
- **Snippet explained:** Seeding is one ingredient; recording config and versions are the others.
- **Related terms:** [[Seed (random seed)]], [[Configuration file]], [[Hyperparameter]]

### Configuration file
<!-- cat: python -->
- **Technical definition:** A file (JSON, YAML, .env) holding settings separate from code so they can change without editing code.
- **Plain English:** The settings menu of your program.
- **Example / analogy:** config.json holds training, generation and limit parameters validated by load_config.
- **Snippet:**
```python
import json
with open('config.json') as f:
    cfg = json.load(f)
```
- **Snippet explained:** Opens the file safely and parses JSON into nested dicts and lists.
- **Related terms:** [[Hyperparameter]], [[JSON (config format)]], [[Reproducibility]]

### JSON (config format)
<!-- cat: python -->
- **Technical definition:** A text format for nested dictionaries, lists, numbers, strings, true/false/null. No comments or trailing commas.
- **Plain English:** A universal way to write structured data as text.
- **Example / analogy:** {"temperature": 0.8, "top_k": 50, "do_sample": true}
- **Snippet:**
```python
import json
json.loads('{"a": 1}')   # {'a': 1}
```
- **Snippet explained:** loads parses a JSON string; json.load reads from a file object.
- **Related terms:** [[Configuration file]]

### Dictionary merge ({**a, **b})
<!-- cat: python -->
- **Technical definition:** Creating a new dict from several others using ** unpacking; later dicts override earlier keys.
- **Plain English:** Layering settings: defaults first, overrides on top.
- **Example / analogy:** {**base_settings, **overrides} is how each experiment row changes only one parameter.
- **Snippet:**
```python
base = {'temperature': 0.8, 'top_k': 50}
print({**base, **{'temperature': 0.2}})
```
- **Snippet explained:** Output {'temperature': 0.2, 'top_k': 50}: the right-hand value wins.
- **Related terms:** [[Configuration file]], [[Hyperparameter]]
