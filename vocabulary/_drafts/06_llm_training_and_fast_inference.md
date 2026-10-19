# Vocabulary draft — module 06: LLM training and fast inference

### Causal language model
<!-- cat: llm -->
- **Technical definition:** A model trained to predict token t+1 from tokens 1..t using a causal (left-to-right) attention mask.
- **Plain English:** A model that only looks at the past and guesses the next word.
- **Example / analogy:** Phone keyboard autocomplete: it sees what you typed so far, never the future.
- **Snippet:**
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('distilgpt2')
```
- **Snippet explained:** Loads a GPT-2 style causal LM; its output logits score every vocabulary token as the next token.
- **Related terms:** [[Next-token prediction]], [[Cross-entropy loss]], [[Logits (training)]]

### Next-token prediction
<!-- cat: llm -->
- **Technical definition:** The training objective of causal LMs: maximise the probability of the actual next token at every position.
- **Plain English:** Guess the next piece of text, over and over, millions of times.
- **Example / analogy:** Fill in the blank: 'The cat sat on the ___'.
- **Snippet:**
```python
x = ids[:-1]   # model input
y = ids[1:]    # what it must predict
```
- **Snippet explained:** Targets are the inputs shifted left by one position.
- **Related terms:** [[Causal language model]], [[Labels shift]], [[Cross-entropy loss]]

### Labels shift
<!-- cat: llm -->
- **Technical definition:** Targets for position t are the input token at t+1; Hugging Face performs the shift internally when `labels=input_ids` is passed.
- **Plain English:** The answer to each position is the token right after it, so the model shifts the labels by one for you.
- **Example / analogy:** Reading a sentence with a ruler under it: each word's 'answer' is the next word.
- **Snippet:**
```python
out = model(input_ids=batch, labels=batch)
print(out.loss)
```
- **Snippet explained:** labels is a copy of the inputs; the library shifts and averages the cross-entropy.
- **Related terms:** [[Next-token prediction]], [[Ignore index (-100)]]

### Ignore index (-100)
<!-- cat: llm -->
- **Technical definition:** The label value that cross-entropy skips, used to exclude padding positions from the loss.
- **Plain English:** A 'do not grade this' stamp on certain tokens.
- **Example / analogy:** A teacher skipping blank answer lines when marking.
- **Snippet:**
```python
labels = ids.masked_fill(mask == 0, -100)
```
- **Snippet explained:** Where the attention mask is 0 (padding) the label becomes -100 so it does not affect the loss.
- **Related terms:** [[Padding]], [[Attention mask (padding)]], [[Cross-entropy loss]]

### Padding
<!-- cat: llm -->
- **Technical definition:** Adding filler tokens so sequences in a batch share one length.
- **Plain English:** Stuffing short sentences with blanks so all rows are equally long.
- **Example / analogy:** Bus seats: empty seats keep every row the same width.
- **Snippet:**
```python
tok.pad_token = tok.eos_token
enc = tok(['hi', 'a longer text'], padding=True, return_tensors='pt')
```
- **Snippet explained:** GPT-2 has no pad token, so EOS is reused; padding=True pads to the longest item in the batch.
- **Related terms:** [[Attention mask (padding)]], [[Padding side]], [[Ignore index (-100)]]

### Attention mask (padding)
<!-- cat: llm -->
- **Technical definition:** A 0/1 tensor marking real tokens (1) versus padding (0) so attention ignores padding.
- **Plain English:** A list saying which tokens are real and which are filler.
- **Example / analogy:** A guest list at the door: pad tokens are not on it.
- **Snippet:**
```python
enc['attention_mask']  # tensor([[1,1,1,0,0]])
```
- **Snippet explained:** Zeros mark padded positions; pass the mask into the model so it ignores them.
- **Related terms:** [[Padding]], [[Padding side]]

### Padding side
<!-- cat: llm -->
- **Technical definition:** Whether pad tokens are added left or right of the real tokens; decoder-only generation requires left padding.
- **Plain English:** Where the blanks go; for text generation put them at the start.
- **Example / analogy:** Aligning receipts at the right edge so the newest line is at the same spot.
- **Snippet:**
```python
tok = AutoTokenizer.from_pretrained('distilgpt2', padding_side='left')
```
- **Snippet explained:** Ensures the last position of every row is a real token, which is where generation continues.
- **Related terms:** [[Padding]], [[Batching (inference)]]

### Block size (context chunk)
<!-- cat: llm -->
- **Technical definition:** The fixed number of tokens per training example after chunking a token stream.
- **Plain English:** How many tokens the model sees at once while training.
- **Example / analogy:** Cutting a long scroll into equal pages.
- **Snippet:**
```python
n = len(ids) // block_size
blocks = torch.tensor(ids[:n*block_size]).view(n, block_size)
```
- **Snippet explained:** Drops the incomplete tail and reshapes into (n_blocks, block_size).
- **Related terms:** [[Context window]], [[DataLoader]]

### DataLoader
<!-- cat: ml -->
- **Technical definition:** A PyTorch utility that serves a dataset in shuffled mini-batches.
- **Plain English:** A waiter that brings the data to the model a few items at a time.
- **Example / analogy:** A card dealer handing out hands of 4.
- **Snippet:**
```python
from torch.utils.data import DataLoader, TensorDataset
loader = DataLoader(TensorDataset(x), batch_size=4, shuffle=True)
```
- **Snippet explained:** Each loop iteration yields one batch; shuffle reorders every epoch.
- **Related terms:** [[Batch size (training)]], [[Epoch]], [[TensorDataset]]

### TensorDataset
<!-- cat: ml -->
- **Technical definition:** A dataset wrapping one or more tensors, indexed along the first dimension.
- **Plain English:** A table where row i holds the i-th example.
- **Example / analogy:** A spreadsheet with one example per row.
- **Snippet:**
```python
ds = TensorDataset(blocks, labels)
print(len(ds), ds[0][0].shape)
```
- **Snippet explained:** ds[i] returns the i-th row from each tensor as a tuple.
- **Related terms:** [[DataLoader]]

### Batch size (training)
<!-- cat: ml -->
- **Technical definition:** Number of examples processed per optimizer step.
- **Plain English:** How many examples the model looks at before updating itself once.
- **Example / analogy:** Marking 4 homework papers before adjusting how you teach.
- **Snippet:**
```python
loader = DataLoader(ds, batch_size=4)
```
- **Snippet explained:** Larger batches give smoother gradients but use more memory.
- **Related terms:** [[Epoch]], [[Learning rate (training)]], [[DataLoader]]

### Epoch
<!-- cat: ml -->
- **Technical definition:** One full pass of the training set through the model.
- **Plain English:** One complete read-through of the study material.
- **Example / analogy:** Reading the whole textbook once.
- **Snippet:**
```python
for epoch in range(6):
    for (batch,) in loader:
        ...
```
- **Snippet explained:** The outer loop counts epochs; the inner loop visits each batch.
- **Related terms:** [[Batch size (training)]], [[Overfitting (training)]]

### Training loop
<!-- cat: ml -->
- **Technical definition:** Repeated forward pass, loss computation, backward pass and parameter update over batches.
- **Plain English:** Try, measure the mistake, adjust, repeat.
- **Example / analogy:** Practising free throws and adjusting your aim after each miss.
- **Snippet:**
```python
loss = model(input_ids=b, labels=b).loss
opt.zero_grad(); loss.backward(); opt.step()
```
- **Snippet explained:** Forward gives the loss, backward gives gradients, step updates weights.
- **Related terms:** [[Optimizer (AdamW)]], [[Backward pass]], [[Gradient clipping]]

### Backward pass
<!-- cat: ml -->
- **Technical definition:** Backpropagation: computing the gradient of the loss with respect to every parameter via the chain rule.
- **Plain English:** Working out which weights caused the error and by how much.
- **Example / analogy:** Tracing back through a recipe to find which ingredient made the dish too salty.
- **Snippet:**
```python
loss.backward()
```
- **Snippet explained:** Fills each parameter's .grad with d(loss)/d(parameter).
- **Related terms:** [[Training loop]], [[Gradient accumulation (PyTorch default)]]

### Gradient accumulation (PyTorch default)
<!-- cat: ml -->
- **Technical definition:** PyTorch adds new gradients to existing .grad values, so they must be cleared with zero_grad() each step.
- **Plain English:** Old corrections pile up unless you wipe them.
- **Example / analogy:** A notepad you must erase before the next calculation.
- **Snippet:**
```python
opt.zero_grad()   # forget this and updates become wrong
loss.backward()
```
- **Snippet explained:** zero_grad clears stale gradients before computing new ones; the same behaviour enables intentional accumulation to simulate large batches.
- **Related terms:** [[Backward pass]], [[Training loop]]

### Optimizer (AdamW)
<!-- cat: ml -->
- **Technical definition:** An algorithm that updates weights from gradients; AdamW uses per-parameter adaptive step sizes plus decoupled weight decay.
- **Plain English:** The coach that decides how big a correction each weight gets.
- **Example / analogy:** A GPS recalculating the route using recent traffic history.
- **Snippet:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
```
- **Snippet explained:** Creates an optimizer over all parameters; call opt.step() after backward().
- **Related terms:** [[Learning rate (training)]], [[Weight decay]], [[Training loop]]

### Weight decay
<!-- cat: ml -->
- **Technical definition:** A regularisation term that shrinks weights slightly each step.
- **Plain English:** A gentle pull keeping weights from growing too large.
- **Example / analogy:** Pruning a plant a little each week.
- **Snippet:**
```python
AdamW(params, lr=5e-5, weight_decay=0.01)
```
- **Snippet explained:** Adds a small shrink to weights each update; helps limit overfitting.
- **Related terms:** [[Optimizer (AdamW)]], [[Overfitting (training)]]

### Learning rate (training)
<!-- cat: ml -->
- **Technical definition:** The scale of each weight update step.
- **Plain English:** How big a step you take after each correction.
- **Example / analogy:** Walking downhill in fog: tiny steps are slow, huge steps overshoot.
- **Snippet:**
```python
AdamW(params, lr=5e-4)
```
- **Snippet explained:** Measured: 5e-7 barely moves, 5e-5 steady, 5e-4 faster on this tiny model.
- **Related terms:** [[Optimizer (AdamW)]], [[Training loop]]

### Gradient clipping
<!-- cat: ml -->
- **Technical definition:** Rescaling gradients whose global norm exceeds a threshold before the optimizer step.
- **Plain English:** A speed limit on how hard the model can be corrected at once.
- **Example / analogy:** A governor on an engine.
- **Snippet:**
```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
- **Snippet explained:** Returns the norm before clipping; if it was above 1.0 the gradients are scaled down to 1.0.
- **Related terms:** [[Training loop]], [[Optimizer (AdamW)]]

### Cross-entropy loss
<!-- cat: ml -->
- **Technical definition:** The negative log-probability the model assigned to the correct token, averaged over positions.
- **Plain English:** A score that is low when the model was confident and right.
- **Example / analogy:** A quiz where confident wrong answers lose the most points.
- **Snippet:**
```python
import math
print(-math.log(0.25))   # 1.386
```
- **Snippet explained:** If the model gave the right token probability 0.25 the loss is about 1.39.
- **Related terms:** [[Perplexity]], [[Next-token prediction]], [[Logits (training)]]

### Logits (training)
<!-- cat: llm -->
- **Technical definition:** Raw unnormalised scores the model outputs for each vocabulary token at each position.
- **Plain English:** The model's raw votes before they become percentages.
- **Example / analogy:** Judges' raw scores before converting to a ranking.
- **Snippet:**
```python
logits = model(**enc).logits   # (batch, seq, vocab)
```
- **Snippet explained:** Shape (batch, sequence length, vocabulary size); softmax turns them into probabilities.
- **Related terms:** [[Cross-entropy loss]], [[Causal language model]]

### Perplexity
<!-- cat: llm -->
- **Technical definition:** exp(average cross-entropy loss); the effective number of equally likely choices per token.
- **Plain English:** How many tokens the model is torn between on average.
- **Example / analogy:** A multiple-choice test where perplexity 12 means about 12 equally likely options per blank.
- **Snippet:**
```python
import math
print(math.exp(2.52))   # 12.4
```
- **Snippet explained:** Our validation loss 2.52 corresponds to perplexity 12.4.
- **Related terms:** [[Cross-entropy loss]], [[Validation set]]

### Validation set
<!-- cat: ml -->
- **Technical definition:** Held-out data not used for training, used to estimate generalisation.
- **Plain English:** Exam questions the model never studied.
- **Example / analogy:** A practice test with new questions.
- **Snippet:**
```python
train, val = blocks[:-5], blocks[-5:]
```
- **Snippet explained:** Last 5 blocks are kept aside and never used for weight updates.
- **Related terms:** [[Overfitting (training)]], [[Perplexity]]

### Overfitting (training)
<!-- cat: ml -->
- **Technical definition:** The model fits training data (low train loss) but does worse on unseen data (high validation loss).
- **Plain English:** Memorising instead of understanding.
- **Example / analogy:** A student who memorises last year's exam answers.
- **Snippet:**
```python
print(train_loss, val_loss)   # train falls, val rises
```
- **Snippet explained:** Diverging train and validation loss is the warning sign.
- **Related terms:** [[Validation set]], [[Epoch]], [[Weight decay]]

### Checkpoint
<!-- cat: ml -->
- **Technical definition:** A saved snapshot of model weights (and config/tokenizer) that can be reloaded later.
- **Plain English:** A save-game for your model.
- **Example / analogy:** Saving a video game before a boss fight.
- **Snippet:**
```python
model.save_pretrained('ckpt'); tok.save_pretrained('ckpt')
```
- **Snippet explained:** Writes config + safetensors weights, plus tokenizer files to the folder.
- **Related terms:** [[safetensors]], [[state_dict]]

### state_dict
<!-- cat: ml -->
- **Technical definition:** A Python dict mapping each layer's parameter name to its tensor.
- **Plain English:** The model's weights as a labelled dictionary.
- **Example / analogy:** A parts list with a label on every part.
- **Snippet:**
```python
sd = model.state_dict()
print(next(iter(sd)))
```
- **Snippet explained:** Prints the first parameter name, e.g. transformer.wte.weight.
- **Related terms:** [[Checkpoint]], [[safetensors]]

### safetensors
<!-- cat: ml -->
- **Technical definition:** A safe, fast tensor file format used by Hugging Face that cannot execute code on load.
- **Plain English:** A weights file format that is quick and safe to open.
- **Example / analogy:** A sealed zip of numbers versus a pickle that could contain surprises.
- **Snippet:**
```python
from safetensors.torch import load_file
w = load_file('model.safetensors')
```
- **Snippet explained:** Loads a dict of tensors from disk.
- **Related terms:** [[Checkpoint]], [[state_dict]]

### Low-rank update
<!-- cat: ml -->
- **Technical definition:** Approximating a weight change ΔW (d_out x d_in) as a product B·A of two thin matrices of rank r.
- **Plain English:** Describing a big change with only a few numbers.
- **Example / analogy:** Describing a face with a few sliders instead of every pixel.
- **Snippet:**
```python
delta = B @ A   # (d_out, r) @ (r, d_in)
```
- **Snippet explained:** The product has full size but only r*(d_out+d_in) free numbers.
- **Related terms:** [[LoRA]], [[Rank (r)]]

### LoRA
<!-- cat: llm -->
- **Technical definition:** Low-Rank Adaptation: freeze pretrained weights and train small low-rank matrices B and A added to selected layers.
- **Plain English:** Fine-tune by training tiny add-on pieces instead of the whole model.
- **Example / analogy:** Clip-on lenses on glasses instead of re-grinding the glass.
- **Snippet:**
```python
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=['c_attn'], task_type='CAUSAL_LM')
model = get_peft_model(base, cfg)
```
- **Snippet explained:** Wraps the base model with trainable adapters on c_attn; base weights are frozen.
- **Related terms:** [[Low-rank update]], [[Rank (r)]], [[LoRA alpha]], [[Adapter]], [[QLoRA]]

### Rank (r)
<!-- cat: llm -->
- **Technical definition:** The inner dimension of LoRA's A and B matrices; controls adapter capacity and size.
- **Plain English:** How wide the add-on is: bigger can learn more but costs more.
- **Example / analogy:** The number of sliders in the face example.
- **Snippet:**
```python
LoraConfig(r=32, ...)
```
- **Snippet explained:** Measured: r=1/8/32 gave 18k/147k/590k trainable parameters for distilgpt2.
- **Related terms:** [[LoRA]], [[LoRA alpha]]

### LoRA alpha
<!-- cat: llm -->
- **Technical definition:** Scaling constant: the LoRA update is multiplied by lora_alpha / r.
- **Plain English:** A volume knob for the adapter's contribution.
- **Example / analogy:** Turning up the strength of a clip-on lens.
- **Snippet:**
```python
scale = lora_alpha / r   # 16 / 8 = 2.0
```
- **Snippet explained:** Update = scale * B @ A; doubling alpha doubles the update's strength.
- **Related terms:** [[LoRA]], [[Rank (r)]]

### target_modules
<!-- cat: llm -->
- **Technical definition:** The list of layer names where peft inserts LoRA adapters; names depend on the architecture.
- **Plain English:** Which layers get the add-ons.
- **Example / analogy:** Choosing which windows in a house get tinted film.
- **Snippet:**
```python
print([n for n, _ in model.named_modules()][:15])
```
- **Snippet explained:** Prints layer names to pick valid target_modules values (GPT-2 uses c_attn).
- **Related terms:** [[LoRA]], [[Adapter]]

### Adapter
<!-- cat: llm -->
- **Technical definition:** A small set of extra trainable weights stored separately from the frozen base model.
- **Plain English:** A small plug-in file that changes the model's behaviour.
- **Example / analogy:** A cartridge for a game console.
- **Snippet:**
```python
model.save_pretrained('adapter_out')   # ~0.6 MB
```
- **Snippet explained:** Saves only adapter weights and its config, not the base model.
- **Related terms:** [[LoRA]], [[Merge adapter]]

### Merge adapter
<!-- cat: llm -->
- **Technical definition:** Folding the adapter into the base weights (W + scale·B·A) so no extra layers remain at inference.
- **Plain English:** Gluing the add-on into the model so it runs at normal speed.
- **Example / analogy:** Permanently installing the lens in the frame.
- **Snippet:**
```python
merged = peft_model.merge_and_unload()
```
- **Snippet explained:** Returns a plain transformers model with updated weights.
- **Related terms:** [[Adapter]], [[LoRA]]

### Trainable parameters
<!-- cat: ml -->
- **Technical definition:** Parameters with requires_grad=True that the optimizer will update.
- **Plain English:** The weights that are allowed to change.
- **Example / analogy:** The unlocked dials on a mixing desk.
- **Snippet:**
```python
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
- **Snippet explained:** Counts elements in all trainable tensors.
- **Related terms:** [[LoRA]], [[Optimizer (AdamW)]]

### QLoRA
<!-- cat: llm -->
- **Technical definition:** LoRA trained on top of a frozen base model quantised to 4-bit NF4, with double quantisation and paged optimizers.
- **Plain English:** LoRA with the big model squeezed to 4-bit so it fits on a small GPU.
- **Example / analogy:** Teaching a compressed textbook with sticky-note corrections.
- **Snippet:**
```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4')
```
- **Snippet explained:** Config passed to from_pretrained(quantization_config=...) to load the base in 4-bit (CUDA only).
- **Related terms:** [[LoRA]], [[NF4]], [[Quantization (weights)]]

### NF4
<!-- cat: llm -->
- **Technical definition:** NormalFloat4: a 4-bit data type whose levels are tuned to normally distributed weights.
- **Plain English:** A 4-bit number format designed for how model weights are spread.
- **Example / analogy:** Ruler marks packed densely where most measurements fall.
- **Snippet:**
```python
bnb_4bit_quant_type='nf4'
```
- **Snippet explained:** Chooses the NF4 data type for the quantised base weights.
- **Related terms:** [[QLoRA]], [[Quantization (weights)]]

### Quantization (weights)
<!-- cat: llm -->
- **Technical definition:** Storing weights with fewer bits (int8, int4) to cut memory and sometimes speed up inference.
- **Plain English:** Rounding the model's numbers to save space.
- **Example / analogy:** Saving a photo at lower quality.
- **Snippet:**
```python
bytes_per_param = {'fp32': 4, 'fp16': 2, 'int8': 1, 'int4': 0.5}
```
- **Snippet explained:** 7B params x 2 bytes = ~14 GB; x 0.5 bytes = ~3.5 GB.
- **Related terms:** [[QLoRA]], [[NF4]], [[GPU memory estimate]]

### Mixed precision
<!-- cat: ml -->
- **Technical definition:** Training with 16-bit math for speed while keeping some state (master weights, optimizer) in fp32.
- **Plain English:** Doing most maths in a smaller number format to go faster.
- **Example / analogy:** Rough sketches for speed, a clean master copy for accuracy.
- **Snippet:**
```python
torch.autocast(device_type='cuda', dtype=torch.bfloat16)
```
- **Snippet explained:** Context manager running eligible ops in bf16.
- **Related terms:** [[Quantization (weights)]], [[GPU memory estimate]]

### Inference mode (no_grad)
<!-- cat: ml -->
- **Technical definition:** Running a model without gradient tracking using model.eval() and torch.no_grad().
- **Plain English:** Using the model, not teaching it, so it skips bookkeeping.
- **Example / analogy:** Driving instead of learning to drive: no instructor notes.
- **Snippet:**
```python
model.eval()
with torch.no_grad():
    out = model.generate(**enc)
```
- **Snippet explained:** eval turns off dropout; no_grad saves memory and time.
- **Related terms:** [[Training loop]], [[KV cache]]

### KV cache
<!-- cat: llm -->
- **Technical definition:** Stored attention keys and values of previous tokens so each new token only computes its own.
- **Plain English:** The model's short-term memory of everything already processed.
- **Example / analogy:** Keeping meeting notes so you don't re-read the whole transcript for each new sentence.
- **Snippet:**
```python
model.generate(**enc, use_cache=True)
```
- **Snippet explained:** Measured: 100 tokens 1.5 s with cache vs 4.8 s without.
- **Related terms:** [[Batching (inference)]], [[GPU memory estimate]], [[PagedAttention]]

### Batching (inference)
<!-- cat: llm -->
- **Technical definition:** Processing several prompts in one forward pass to increase hardware utilisation.
- **Plain English:** Serving many customers at the same time.
- **Example / analogy:** A bus carrying many passengers on one trip.
- **Snippet:**
```python
enc = tok(prompts, return_tensors='pt', padding=True)
out = model.generate(**enc, max_new_tokens=30)
```
- **Snippet explained:** One generate call handles all prompts in the list together.
- **Related terms:** [[Throughput]], [[Latency]], [[Continuous batching]]

### Latency
<!-- cat: llm -->
- **Technical definition:** Time for one request to complete (or time to first token).
- **Plain English:** How long one user waits.
- **Example / analogy:** The wait for your own coffee.
- **Snippet:**
```python
t0 = time.perf_counter()
model.generate(**enc)
print(time.perf_counter() - t0)
```
- **Snippet explained:** Subtracts two clock readings to get seconds elapsed.
- **Related terms:** [[Throughput]], [[Batching (inference)]]

### Throughput
<!-- cat: llm -->
- **Technical definition:** Total tokens (or requests) processed per unit time across all users.
- **Plain English:** How much work the whole system finishes per second.
- **Example / analogy:** Coffees per hour the shop produces.
- **Snippet:**
```python
tokens_per_s = batch_size * new_tokens / seconds
```
- **Snippet explained:** Measured: batch 16 gave 684 tokens/s vs 71 at batch 1.
- **Related terms:** [[Latency]], [[Batching (inference)]]

### GPU memory estimate
<!-- cat: llm -->
- **Technical definition:** Back-of-envelope: weights = params x bytes/param; KV cache = 2 x layers x kv_heads x head_dim x tokens x batch x bytes; training AdamW about 16 bytes/param.
- **Plain English:** A quick sum to see if the model fits on your graphics card.
- **Example / analogy:** Packing a suitcase: add the weights of everything first.
- **Snippet:**
```python
kv = 2 * layers * kv_heads * head_dim * tokens * batch * 2   # fp16
```
- **Snippet explained:** 7B, 32 layers, 32 kv heads, 128 dim, 4096 tokens, batch 8 gives ~17 GB.
- **Related terms:** [[KV cache]], [[Quantization (weights)]], [[Grouped-query attention]]

### Grouped-query attention
<!-- cat: llm -->
- **Technical definition:** An attention variant where several query heads share fewer key/value heads, shrinking the KV cache.
- **Plain English:** Heads share notebooks so the memory is smaller.
- **Example / analogy:** Colleagues sharing one set of meeting notes.
- **Snippet:**
```python
num_key_value_heads = 8   # instead of 32
```
- **Snippet explained:** With 8 KV heads the cache is 4x smaller than with 32.
- **Related terms:** [[KV cache]], [[GPU memory estimate]]

### PagedAttention
<!-- cat: llm -->
- **Technical definition:** vLLM's KV-cache manager that stores cache in fixed-size blocks allocated on demand, like virtual-memory pages.
- **Plain English:** Hand out memory in small pages instead of reserving a huge slab per user.
- **Example / analogy:** A hotel assigning rooms as guests arrive instead of reserving whole floors.
- **Snippet:**
```python
llm = LLM(model='Qwen/Qwen2.5-0.5B-Instruct', gpu_memory_utilization=0.9)
```
- **Snippet explained:** vLLM pre-allocates KV pages within 90% of GPU memory and hands them out per request.
- **Related terms:** [[KV cache]], [[Continuous batching]], [[vLLM]]

### Continuous batching
<!-- cat: llm -->
- **Technical definition:** Scheduling at every decoding step: finished sequences leave the batch and waiting ones join immediately.
- **Plain English:** Never let the GPU wait for the slowest request.
- **Example / analogy:** A restaurant seating new guests as soon as any table frees up.
- **Snippet:**
```python
outputs = llm.generate(prompts, params)   # vLLM schedules internally
```
- **Snippet explained:** vLLM batches the prompts continuously without extra code.
- **Related terms:** [[PagedAttention]], [[Batching (inference)]], [[vLLM]]

### vLLM
<!-- cat: llm -->
- **Technical definition:** A high-throughput LLM inference/serving engine using PagedAttention and continuous batching, with an OpenAI-compatible server.
- **Plain English:** The professional engine for serving a model to many users.
- **Example / analogy:** A busy kitchen run with an efficient head chef versus a home cook.
- **Snippet:**
```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
```
- **Snippet explained:** Starts an HTTP server exposing /v1/chat/completions.
- **Related terms:** [[PagedAttention]], [[Continuous batching]], [[Throughput]]
