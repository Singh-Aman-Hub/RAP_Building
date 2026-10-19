# Machine Learning Terms

92 terms, alphabetical. Back to [index](index.md).

- [Accuracy](#accuracy)
- [Adam and AdamW](#adam-and-adamw)
- [Autograd](#autograd)
- [Backpropagation](#backpropagation)
- [Backward pass](#backward-pass)
- [Batch size](#batch-size)
- [Batch size (training)](#batch-size-training)
- [Bias (parameter)](#bias-parameter)
- [Broadcasting](#broadcasting)
- [Checkpoint](#checkpoint)
- [Concept drift](#concept-drift)
- [Confidence](#confidence)
- [Confusion matrix](#confusion-matrix)
- [Cosine similarity](#cosine-similarity)
- [Cross-entropy loss](#cross-entropy-loss)
- [Data collection](#data-collection)
- [Data drift](#data-drift)
- [Data leakage](#data-leakage)
- [Data type (dtype)](#data-type-dtype)
- [DataLoader](#dataloader)
- [Dropout](#dropout)
- [Early stopping](#early-stopping)
- [Entropy (of a distribution)](#entropy-of-a-distribution)
- [Epoch](#epoch)
- [eval mode and no_grad](#eval-mode-and-no_grad)
- [Evaluation metric](#evaluation-metric)
- [F1 score](#f1-score)
- [Feature](#feature)
- [Feature engineering](#feature-engineering)
- [Feature scaling](#feature-scaling)
- [fit vs transform](#fit-vs-transform)
- [Gradient](#gradient)
- [Gradient accumulation](#gradient-accumulation)
- [Gradient accumulation (PyTorch default)](#gradient-accumulation-pytorch-default)
- [Gradient clipping](#gradient-clipping)
- [Gradient descent](#gradient-descent)
- [Hyperparameter](#hyperparameter)
- [Inference mode (no_grad)](#inference-mode-no_grad)
- [Inference pipeline](#inference-pipeline)
- [Iteration (step)](#iteration-step)
- [Label](#label)
- [Label masking (-100)](#label-masking--100)
- [Lazy loading](#lazy-loading)
- [Learning rate](#learning-rate)
- [Learning rate (training)](#learning-rate-training)
- [Learning-rate scheduler (warm-up)](#learning-rate-scheduler-warm-up)
- [Log loss](#log-loss)
- [Logits](#logits)
- [Loss function](#loss-function)
- [Low-rank update](#low-rank-update)
- [Machine learning](#machine-learning)
- [Mixed precision](#mixed-precision)
- [ML pipeline](#ml-pipeline)
- [Model artifact](#model-artifact)
- [Model monitoring](#model-monitoring)
- [Model serialization (joblib)](#model-serialization-joblib)
- [MSE](#mse)
- [nn.Module](#nnmodule)
- [Optimizer](#optimizer)
- [Optimizer (AdamW)](#optimizer-adamw)
- [Overfitting](#overfitting)
- [Overfitting (training)](#overfitting-training)
- [Parameter (model)](#parameter-model)
- [Parameter (weight)](#parameter-weight)
- [Pipeline (sklearn)](#pipeline-sklearn)
- [Postprocessing](#postprocessing)
- [Precision](#precision)
- [Preprocessing](#preprocessing)
- [PSI](#psi)
- [Recall](#recall)
- [Reproducibility](#reproducibility)
- [safetensors](#safetensors)
- [Seed (random seed)](#seed-random-seed)
- [SGD (stochastic gradient descent)](#sgd-stochastic-gradient-descent)
- [Softmax](#softmax)
- [state_dict](#state_dict)
- [Supervised learning](#supervised-learning)
- [Temperature (softmax)](#temperature-softmax)
- [Tensor](#tensor)
- [TensorDataset](#tensordataset)
- [TF-IDF](#tf-idf)
- [Train/test split](#traintest-split)
- [Train/validation split](#trainvalidation-split)
- [Trainable parameters](#trainable-parameters)
- [Training loop](#training-loop)
- [Training pipeline](#training-pipeline)
- [Training/serving skew](#trainingserving-skew)
- [Underfitting](#underfitting)
- [Validation set](#validation-set)
- [Vectorizer](#vectorizer)
- [Weight](#weight)
- [Weight decay](#weight-decay)

---

### Accuracy
*From module: 03_ml_foundations*

- **Technical definition:** Fraction of predictions that are correct.
- **Plain English:** How often the model is right.
- **Example / analogy:** Marks out of 100.
- **Snippet:**
```python
accuracy_score(y_true, y_pred)
```
- **Snippet explained:** Correct predictions / total.
- **Related terms:** [Precision](ml_terms.md#precision), [Recall](ml_terms.md#recall)

---

### Adam and AdamW
*From module: 05_llm_parameters*

- **Technical definition:** Optimizers that track running averages of gradients and squared gradients to give each parameter its own adaptive step size; AdamW applies weight decay separately (decoupled).
- **Plain English:** A hiker with a personal pace per leg of the journey.
- **Example / analogy:** The default for LLM training; stores two extra numbers per parameter, so optimizer memory is about 2x the weights.
- **Snippet:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
```
- **Snippet explained:** weight_decay shrinks weights slightly each step to curb overfitting.
- **Related terms:** [Optimizer](ml_terms.md#optimizer), [Weight decay](ml_terms.md#weight-decay), [Learning rate](ml_terms.md#learning-rate)

---

### Autograd
*From module: 03_ml_foundations*

- **Technical definition:** PyTorch's automatic differentiation: records tensor operations and computes gradients via backward().
- **Plain English:** PyTorch works out the derivatives for you.
- **Example / analogy:** A calculator that remembers every step so it can run them backwards.
- **Snippet:**
```python
w = torch.tensor(0., requires_grad=True)
loss = (w * 2 - 4) ** 2
loss.backward()
print(w.grad)
```
- **Snippet explained:** backward() fills w.grad with dloss/dw.
- **Related terms:** [Gradient](ml_terms.md#gradient), [Tensor](ml_terms.md#tensor), [Backpropagation](ml_terms.md#backpropagation)

---

### Backpropagation
*From module: 03_ml_foundations*

- **Technical definition:** Applying the chain rule backward through a network to compute gradients of the loss for every parameter.
- **Plain English:** How the blame for the error is passed back to each weight.
- **Example / analogy:** Tracing an error back through every step of a production line.
- **Snippet:**
```python
loss.backward()
```
- **Snippet explained:** Triggers backpropagation through the computation graph.
- **Related terms:** [Autograd](ml_terms.md#autograd), [Gradient](ml_terms.md#gradient)

---

### Backward pass
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** Backpropagation: computing the gradient of the loss with respect to every parameter via the chain rule.
- **Plain English:** Working out which weights caused the error and by how much.
- **Example / analogy:** Tracing back through a recipe to find which ingredient made the dish too salty.
- **Snippet:**
```python
loss.backward()
```
- **Snippet explained:** Fills each parameter's .grad with d(loss)/d(parameter).
- **Related terms:** [Training loop](ml_terms.md#training-loop), [Gradient accumulation (PyTorch default)](ml_terms.md#gradient-accumulation-pytorch-default)

---

### Batch size
*From module: 03_ml_foundations*

- **Technical definition:** Number of samples processed before each parameter update.
- **Plain English:** How many examples you look at before adjusting.
- **Example / analogy:** Grading papers in stacks of 32 before giving feedback.
- **Snippet:**
```python
DataLoader(train_set, batch_size=32, shuffle=True)
```
- **Snippet explained:** Loader yields 32 samples at a time.
- **Related terms:** [Epoch](ml_terms.md#epoch), [DataLoader](ml_terms.md#dataloader), [Hyperparameter](ml_terms.md#hyperparameter)

---

### Batch size (training)
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** Number of examples processed per optimizer step.
- **Plain English:** How many examples the model looks at before updating itself once.
- **Example / analogy:** Marking 4 homework papers before adjusting how you teach.
- **Snippet:**
```python
loader = DataLoader(ds, batch_size=4)
```
- **Snippet explained:** Larger batches give smoother gradients but use more memory.
- **Related terms:** [Epoch](ml_terms.md#epoch), [Learning rate (training)](ml_terms.md#learning-rate-training), [DataLoader](ml_terms.md#dataloader)

---

### Bias (parameter)
*From module: 03_ml_foundations*

- **Technical definition:** A learned constant added to a weighted sum, letting the model shift its output.
- **Plain English:** The baseline output when inputs are zero.
- **Example / analogy:** A starting offset.
- **Snippet:**
```python
y_hat = w * x + b
```
- **Snippet explained:** b shifts the line up/down.
- **Related terms:** [Weight](ml_terms.md#weight)

---

### Broadcasting
*From module: 03_ml_foundations*

- **Technical definition:** Numpy/PyTorch rule that expands smaller arrays across larger ones during element-wise operations.
- **Plain English:** Automatic repetition so shapes line up.
- **Example / analogy:** Adding the same tax to every price in a list.
- **Snippet:**
```python
(X - X.mean(axis=0)) / X.std(axis=0)
```
- **Snippet explained:** A (n,2) array minus a (2,) vector subtracts per column.
- **Related terms:** [Tensor](ml_terms.md#tensor), [Feature scaling](ml_terms.md#feature-scaling)

---

### Checkpoint
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** A saved snapshot of model weights (and config/tokenizer) that can be reloaded later.
- **Plain English:** A save-game for your model.
- **Example / analogy:** Saving a video game before a boss fight.
- **Snippet:**
```python
model.save_pretrained('ckpt'); tok.save_pretrained('ckpt')
```
- **Snippet explained:** Writes config + safetensors weights, plus tokenizer files to the folder.
- **Related terms:** [safetensors](ml_terms.md#safetensors), [state_dict](ml_terms.md#state_dict)

---

### Concept drift
*From module: 10_ml_system_architecture*

- **Technical definition:** A change in the relationship between inputs and the correct output over time (new slang, new fraud tactics) that needs fresh labels to detect.
- **Plain English:** The right answer for the same input has changed.
- **Example / analogy:** 'Sick' now meaning 'great'.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [Data drift](ml_terms.md#data-drift), [Model monitoring](ml_terms.md#model-monitoring)

---

### Confidence
*From module: 10_ml_system_architecture*

- **Technical definition:** The probability a model assigns to its predicted class (e.g. max of predict_proba); not guaranteed to be calibrated.
- **Plain English:** How sure the model says it is.
- **Example / analogy:** A weather forecast's '70% chance'.
- **Snippet:**
```python
probs = model.predict_proba([text])[0]
confidence = probs.max()
```
- **Snippet explained:** Maximum class probability.
- **Related terms:** [Postprocessing](ml_terms.md#postprocessing), [Log loss](ml_terms.md#log-loss)

---

### Confusion matrix
*From module: 03_ml_foundations*

- **Technical definition:** Table counting true/false positives and negatives.
- **Plain English:** A scoreboard of right and wrong guesses by class.
- **Example / analogy:** A 2x2 grid of hits and misses.
- **Snippet:**
```python
confusion_matrix(y_true, y_pred)   # rows: truth; cols: prediction
```
- **Snippet explained:** Rows are truth, columns predictions.
- **Related terms:** [Precision](ml_terms.md#precision), [Recall](ml_terms.md#recall)

---

### Cosine similarity
*From module: 04_llm_fundamentals*

- **Technical definition:** The cosine of the angle between two vectors: dot(a, b) / (|a| |b|); 1 = same direction, 0 = orthogonal, -1 = opposite.
- **Plain English:** How much two arrows point the same way, ignoring their length.
- **Example / analogy:** cosine('cat', 'dog') = 0.57 but cosine('cat', 'economics') = 0.20 in GPT-2 embeddings.
- **Snippet:**
```python
import torch.nn.functional as F
F.cosine_similarity(a, b, dim=0).item()
```
- **Snippet explained:** Compares two 1-D vectors and converts the 0-d result to a Python float.
- **Related terms:** [Embedding](llm_terms.md#embedding), [Position embedding](llm_terms.md#position-embedding)

---

### Cross-entropy loss
*From module: 03_ml_foundations*

- **Technical definition:** -log of the probability assigned to the correct class; standard classification and language-model loss.
- **Plain English:** Penalises the model heavily when it is confident and wrong.
- **Example / analogy:** A quiz where a confident wrong answer costs the most.
- **Snippet:**
```python
F.cross_entropy(logits, target)
```
- **Snippet explained:** Takes raw logits and class indices; applies softmax internally.
- **Related terms:** [Softmax](ml_terms.md#softmax), [Logits](ml_terms.md#logits), [Loss function](ml_terms.md#loss-function)

---

### Data collection
*From module: 10_ml_system_architecture*

- **Technical definition:** Gathering raw examples from files, databases, APIs or user feedback that represent the cases the model will face in production.
- **Plain English:** Getting the examples the model will learn from.
- **Example / analogy:** Gathering ingredients before cooking.
- **Snippet:**
```python
texts, labels = get_training_data(n=200, seed=0)
```
- **Snippet explained:** A deterministic generator stands in for real data sources.
- **Related terms:** [Preprocessing](ml_terms.md#preprocessing), [Data leakage](ml_terms.md#data-leakage)

---

### Data drift
*From module: 10_ml_system_architecture*

- **Technical definition:** A change in the distribution of model inputs between training and production, potentially lowering accuracy; detected with statistics such as PSI.
- **Plain English:** The kind of data coming in has changed.
- **Example / analogy:** A language model trained on formal text now seeing slang.
- **Snippet:**
```python
psi(reference_lengths, live_lengths) > 0.25
```
- **Snippet explained:** PSI above 0.25 signals significant drift.
- **Related terms:** [Concept drift](ml_terms.md#concept-drift), [PSI](ml_terms.md#psi)

---

### Data leakage
*From module: 03_ml_foundations*

- **Technical definition:** Information from validation/test data leaking into training or preprocessing, inflating metrics.
- **Plain English:** Peeking at the exam answers.
- **Example / analogy:** Fitting a scaler on all data before splitting.
- **Snippet:**
```python
pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
pipe.fit(X_train, y_train)
```
- **Snippet explained:** Pipeline fits the vectorizer on training data only.
- **Related terms:** [Train/validation split](ml_terms.md#trainvalidation-split), [Pipeline (sklearn)](ml_terms.md#pipeline-sklearn)

---

### Data type (dtype)
*From module: 04_llm_fundamentals*

- **Technical definition:** How each number is stored: float32 (4 bytes), float16/bfloat16 (2), int8 (1), int4 (0.5). Fewer bytes means less memory but less precision.
- **Plain English:** How many decimal digits of precision you keep per number.
- **Example / analogy:** Casting a model with model.half() halves its memory.
- **Snippet:**
```python
p = next(model.parameters())
print(p.dtype, p.element_size())
```
- **Snippet explained:** Shows the dtype (float32) and its size in bytes (4).
- **Related terms:** [Parameter count](llm_terms.md#parameter-count), [Parameter (weight)](ml_terms.md#parameter-weight)

---

### DataLoader
*From module: 03_ml_foundations*

- **Technical definition:** PyTorch utility that batches, shuffles and iterates over a dataset.
- **Plain English:** Hands the model data in neat batches.
- **Example / analogy:** A cafeteria server giving trays of fixed size.
- **Snippet:**
```python
for xb, yb in DataLoader(ds, batch_size=32):
    ...
```
- **Snippet explained:** Each iteration yields one batch of inputs and labels.
- **Related terms:** [Batch size](ml_terms.md#batch-size), [Epoch](ml_terms.md#epoch)

---

### Dropout
*From module: 04_llm_fundamentals*

- **Technical definition:** A regularisation technique that randomly zeroes some activations during training to reduce overfitting; disabled at inference.
- **Plain English:** Making students work with random teammates absent so nobody relies on one person.
- **Example / analogy:** model.train() enables it; model.eval() disables it.
- **Snippet:**
```python
model.train()   # dropout active
model.eval()    # dropout off
```
- **Snippet explained:** Switches the mode flag for every layer in the model.
- **Related terms:** [eval mode and no_grad](ml_terms.md#eval-mode-and-no_grad), [Fine-tuning](llm_terms.md#fine-tuning)

---

### Early stopping
*From module: 03_ml_foundations*

- **Technical definition:** Stop training when validation loss stops improving to avoid overfitting.
- **Plain English:** Quit while you're ahead.
- **Example / analogy:** Stop studying when practice-test scores stop improving.
- **Snippet:**
```python
if val_loss > best_val_loss: patience_left -= 1
```
- **Snippet explained:** Count epochs without improvement and stop at zero.
- **Related terms:** [Overfitting](ml_terms.md#overfitting), [Epoch](ml_terms.md#epoch)

---

### Entropy (of a distribution)
*From module: 05_llm_parameters*

- **Technical definition:** -sum p log p: a measure of uncertainty; 0 for a certain outcome, log(N) for a uniform one over N options.
- **Plain English:** How surprised you expect to be by the next token.
- **Example / analogy:** Entropy was 0.01 at T=0.2 and 10.28 at T=3.0 (uniform over 50,257 would be 10.8).
- **Snippet:**
```python
H = -(p[p > 0] * p[p > 0].log()).sum()
```
- **Snippet explained:** Ignore zero probabilities (log 0 is undefined) and sum p*log p.
- **Related terms:** [Temperature (sampling)](llm_terms.md#temperature-sampling), [Perplexity](llm_terms.md#perplexity)

---

### Epoch
*From module: 03_ml_foundations*

- **Technical definition:** One complete pass through the entire training dataset.
- **Plain English:** One full read-through of the training data.
- **Example / analogy:** Reading the whole textbook once.
- **Snippet:**
```python
for epoch in range(10):
    ...
```
- **Snippet explained:** Ten passes over the data.
- **Related terms:** [Batch size](ml_terms.md#batch-size), [Training loop](ml_terms.md#training-loop)

---

### eval mode and no_grad
*From module: 04_llm_fundamentals*

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
- **Related terms:** [Inference](llm_terms.md#inference), [Dropout](ml_terms.md#dropout), [Backpropagation](ml_terms.md#backpropagation)

---

### Evaluation metric
*From module: 10_ml_system_architecture*

- **Technical definition:** A number summarising model quality on held-out data (accuracy, precision/recall, F1, log loss...).
- **Plain English:** A score for how good the model is.
- **Example / analogy:** Exam marks.
- **Snippet:**
```python
accuracy_score(y_test, model.predict(X_test))
```
- **Snippet explained:** Fraction of correct predictions.
- **Related terms:** [Log loss](ml_terms.md#log-loss), [Train/test split](ml_terms.md#traintest-split)

---

### F1 score
*From module: 03_ml_foundations*

- **Technical definition:** Harmonic mean of precision and recall: 2PR/(P+R).
- **Plain English:** One number balancing precision and recall.
- **Example / analogy:** A team grade that's low if either teammate is weak.
- **Snippet:**
```python
f1_score(y_true, y_pred)
```
- **Snippet explained:** Returns the harmonic mean.
- **Related terms:** [Precision](ml_terms.md#precision), [Recall](ml_terms.md#recall)

---

### Feature
*From module: 03_ml_foundations*

- **Technical definition:** An input variable used by a model to make a prediction.
- **Plain English:** A fact about each example that the model may use.
- **Example / analogy:** For house price: size, rooms, location.
- **Snippet:**
```python
X = np.array([ [50, 2], [80, 3] ])   # columns = features
```
- **Snippet explained:** Each row is a house; each column is a feature.
- **Related terms:** [Label](ml_terms.md#label), [Feature scaling](ml_terms.md#feature-scaling)

---

### Feature engineering
*From module: 10_ml_system_architecture*

- **Technical definition:** Turning raw data into numeric features a model can use (TF-IDF vectors, embeddings, scaled numbers, counts).
- **Plain English:** Converting text or records into numbers the model understands.
- **Example / analogy:** Measuring ingredients into grams.
- **Snippet:**
```python
vec = TfidfVectorizer()
X = vec.fit_transform(texts)
```
- **Snippet explained:** Learns the vocabulary and returns a texts-by-words matrix.
- **Related terms:** [TF-IDF](ml_terms.md#tf-idf), [fit vs transform](ml_terms.md#fit-vs-transform), [Preprocessing](ml_terms.md#preprocessing)

---

### Feature scaling
*From module: 03_ml_foundations*

- **Technical definition:** Rescaling features (e.g. standardisation) so they have comparable ranges, which speeds up training.
- **Plain English:** Putting all columns on a similar scale.
- **Example / analogy:** Converting height in cm and weight in kg into 'how many standard deviations from average'.
- **Snippet:**
```python
X_scaled = (X - X.mean(0)) / X.std(0)
```
- **Snippet explained:** Each column gets mean 0 and std 1.
- **Related terms:** [Feature](ml_terms.md#feature), [Broadcasting](ml_terms.md#broadcasting)

---

### fit vs transform
*From module: 10_ml_system_architecture*

- **Technical definition:** fit learns parameters from data (vocabulary, scaling statistics, weights); transform applies learned parameters; fit_transform does both on the same data. Only fit on training data.
- **Plain English:** fit = learn the rules, transform = apply the rules.
- **Example / analogy:** Studying (fit) then taking the exam with what you learned (transform).
- **Snippet:**
```python
Xtr = vec.fit_transform(train)
Xte = vec.transform(test)   # never fit on test
```
- **Snippet explained:** Test data is transformed with the training vocabulary.
- **Related terms:** [Data leakage](ml_terms.md#data-leakage), [Vectorizer](ml_terms.md#vectorizer)

---

### Gradient
*From module: 03_ml_foundations*

- **Technical definition:** The vector of partial derivatives of the loss with respect to each parameter; points toward steepest increase.
- **Plain English:** Which way and how steeply the loss changes if you nudge each parameter.
- **Example / analogy:** The slope of a hill under your feet.
- **Snippet:**
```python
grad_w = 2 * np.mean(error * x)
```
- **Snippet explained:** Derivative of MSE with respect to w.
- **Related terms:** [Gradient descent](ml_terms.md#gradient-descent), [Autograd](ml_terms.md#autograd), [Backpropagation](ml_terms.md#backpropagation)

---

### Gradient accumulation
*From module: 05_llm_parameters*

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
- **Related terms:** [Batch size](ml_terms.md#batch-size), [Iteration (step)](ml_terms.md#iteration-step)

---

### Gradient accumulation (PyTorch default)
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** PyTorch adds new gradients to existing .grad values, so they must be cleared with zero_grad() each step.
- **Plain English:** Old corrections pile up unless you wipe them.
- **Example / analogy:** A notepad you must erase before the next calculation.
- **Snippet:**
```python
opt.zero_grad()   # forget this and updates become wrong
loss.backward()
```
- **Snippet explained:** zero_grad clears stale gradients before computing new ones; the same behaviour enables intentional accumulation to simulate large batches.
- **Related terms:** [Backward pass](ml_terms.md#backward-pass), [Training loop](ml_terms.md#training-loop)

---

### Gradient clipping
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** Rescaling gradients whose global norm exceeds a threshold before the optimizer step.
- **Plain English:** A speed limit on how hard the model can be corrected at once.
- **Example / analogy:** A governor on an engine.
- **Snippet:**
```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
- **Snippet explained:** Returns the norm before clipping; if it was above 1.0 the gradients are scaled down to 1.0.
- **Related terms:** [Training loop](ml_terms.md#training-loop), [Optimizer (AdamW)](ml_terms.md#optimizer-adamw)

---

### Gradient descent
*From module: 03_ml_foundations*

- **Technical definition:** Iteratively updating parameters opposite to the gradient: p = p - lr * grad.
- **Plain English:** Walk downhill step by step.
- **Example / analogy:** Hiking down a foggy mountain by feeling the slope.
- **Snippet:**
```python
w -= learning_rate * grad_w
```
- **Snippet explained:** Move w a small step against the slope.
- **Related terms:** [Learning rate](ml_terms.md#learning-rate), [Gradient](ml_terms.md#gradient), [Optimizer](ml_terms.md#optimizer)

---

### Hyperparameter
*From module: 03_ml_foundations*

- **Technical definition:** A setting chosen before/outside training (learning rate, batch size, epochs, number of layers).
- **Plain English:** A knob that you set, not the model.
- **Example / analogy:** Oven temperature and cooking time.
- **Snippet:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
```
- **Snippet explained:** lr is a hyperparameter; weights are parameters.
- **Related terms:** [Learning rate](ml_terms.md#learning-rate), [Batch size](ml_terms.md#batch-size), [Epoch](ml_terms.md#epoch)

---

### Inference mode (no_grad)
*From module: 06_llm_training_and_fast_inference*

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
- **Related terms:** [Training loop](ml_terms.md#training-loop), [KV cache](llm_terms.md#kv-cache)

---

### Inference pipeline
*From module: 10_ml_system_architecture*

- **Technical definition:** The online path that loads a saved artifact once and, per request, validates input, preprocesses, predicts and postprocesses.
- **Plain English:** The 'answering' program, run all the time.
- **Example / analogy:** A student answering exam questions with what they learned.
- **Snippet:**
```python
model, meta = load_artifact()
predict_text(model, 'What a superb hotel')
```
- **Snippet explained:** Loads once, predicts per request, never calls fit.
- **Related terms:** [Training pipeline](ml_terms.md#training-pipeline), [Postprocessing](ml_terms.md#postprocessing)

---

### Iteration (step)
*From module: 05_llm_parameters*

- **Technical definition:** One weight update, i.e. processing one batch (forward, loss, backward, optimizer step).
- **Plain English:** One tweak of the knobs.
- **Example / analogy:** Steps per epoch = ceil(dataset size / batch size).
- **Snippet:**
```python
steps_per_epoch = -(-300 // 32)   # ceiling division = 10
```
- **Snippet explained:** Negative floor-division trick for rounding up.
- **Related terms:** [Batch size](ml_terms.md#batch-size), [Epoch](ml_terms.md#epoch)

---

### Label
*From module: 03_ml_foundations*

- **Technical definition:** The target value the model should predict (the 'right answer' in supervised learning).
- **Plain English:** The answer key for each training example.
- **Example / analogy:** The house's real sale price.
- **Snippet:**
```python
y = np.array([200, 320])
```
- **Snippet explained:** One label per row of X.
- **Related terms:** [Feature](ml_terms.md#feature), [Supervised learning](ml_terms.md#supervised-learning)

---

### Label masking (-100)
*From module: 04_llm_fundamentals*

- **Technical definition:** Setting label values to -100 so PyTorch's cross-entropy ignores those positions, commonly used for padding (or for prompt tokens in instruction tuning).
- **Plain English:** A 'do not grade this part' marker.
- **Example / analogy:** Without it, padding tokens contribute meaningless loss.
- **Snippet:**
```python
labels = batch['input_ids'].clone()
labels[batch['attention_mask'] == 0] = -100
```
- **Snippet explained:** Copy the ids as labels, then hide padded positions from the loss.
- **Related terms:** [Padding and attention mask](llm_terms.md#padding-and-attention-mask), [Cross-entropy loss](ml_terms.md#cross-entropy-loss), [Fine-tuning](llm_terms.md#fine-tuning)

---

### Lazy loading
*From module: 09_api_development*

- **Technical definition:** Deferring expensive initialisation (like importing or loading a big model) until it is first needed, then caching it.
- **Plain English:** Don't unpack the heavy thing until someone actually asks for it.
- **Example / analogy:** Not starting the oven until an order for pizza arrives.
- **Snippet:**
```python
_GEN = None
def gen():
    global _GEN
    if _GEN is None:
        _GEN = load_big_model()
    return _GEN
```
- **Snippet explained:** First call pays the loading cost; later calls reuse the cached object.
- **Related terms:** Model loading, [Lifespan (startup/shutdown)](api_and_deployment_terms.md#lifespan-startupshutdown)

---

### Learning rate
*From module: 03_ml_foundations*

- **Technical definition:** Hyperparameter controlling the step size of each parameter update.
- **Plain English:** How big each step downhill is.
- **Example / analogy:** Stride length when walking down a hill.
- **Snippet:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
- **Snippet explained:** lr=0.01 scales every update.
- **Related terms:** [Gradient descent](ml_terms.md#gradient-descent), [Hyperparameter](ml_terms.md#hyperparameter)

---

### Learning rate (training)
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** The scale of each weight update step.
- **Plain English:** How big a step you take after each correction.
- **Example / analogy:** Walking downhill in fog: tiny steps are slow, huge steps overshoot.
- **Snippet:**
```python
AdamW(params, lr=5e-4)
```
- **Snippet explained:** Measured: 5e-7 barely moves, 5e-5 steady, 5e-4 faster on this tiny model.
- **Related terms:** [Optimizer (AdamW)](ml_terms.md#optimizer-adamw), [Training loop](ml_terms.md#training-loop)

---

### Learning-rate scheduler (warm-up)
*From module: 05_llm_parameters*

- **Technical definition:** A rule changing the learning rate during training, commonly a short warm-up from near zero followed by a gradual decay.
- **Plain English:** Starting a run slowly, then easing off toward the finish.
- **Example / analogy:** Warm-up avoids huge early updates that can destabilise a pretrained model.
- **Snippet:**
```python
sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=10)
sched.step()
```
- **Snippet explained:** Call sched.step() after each optimizer step to update the learning rate.
- **Related terms:** [Learning rate](ml_terms.md#learning-rate), [Optimizer](ml_terms.md#optimizer)

---

### Log loss
*From module: 10_ml_system_architecture*

- **Technical definition:** A probabilistic metric penalising confident wrong predictions: -mean(log p(true class)); lower is better.
- **Plain English:** Rewards being confident and right, punishes being confident and wrong.
- **Example / analogy:** A quiz where bold wrong answers lose more points.
- **Snippet:**
```python
log_loss(y_test, model.predict_proba(X_test), labels=list(model.classes_))
```
- **Snippet explained:** Computes cross-entropy from predicted probabilities.
- **Related terms:** [Evaluation metric](ml_terms.md#evaluation-metric)

---

### Logits
*From module: 03_ml_foundations*

- **Technical definition:** Raw, unnormalised model output scores before softmax.
- **Plain English:** The model's raw scores for each option.
- **Example / analogy:** Judges' scores before converting to percentages.
- **Snippet:**
```python
logits = np.array([2.0, 1.0, 0.1])
```
- **Snippet explained:** Scores for 3 classes.
- **Related terms:** [Softmax](ml_terms.md#softmax), [Cross-entropy loss](ml_terms.md#cross-entropy-loss)

---

### Loss function
*From module: 03_ml_foundations*

- **Technical definition:** A function mapping predictions and targets to a single number that training minimises.
- **Plain English:** A score for how wrong the model is.
- **Example / analogy:** A golf score: lower is better.
- **Snippet:**
```python
loss = np.mean((y_hat - y) ** 2)
```
- **Snippet explained:** Mean squared error.
- **Related terms:** [MSE](ml_terms.md#mse), [Cross-entropy loss](ml_terms.md#cross-entropy-loss), [Gradient descent](ml_terms.md#gradient-descent)

---

### Low-rank update
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** Approximating a weight change ΔW (d_out x d_in) as a product B·A of two thin matrices of rank r.
- **Plain English:** Describing a big change with only a few numbers.
- **Example / analogy:** Describing a face with a few sliders instead of every pixel.
- **Snippet:**
```python
delta = B @ A   # (d_out, r) @ (r, d_in)
```
- **Snippet explained:** The product has full size but only r*(d_out+d_in) free numbers.
- **Related terms:** [LoRA](llm_terms.md#lora), [Rank (r)](llm_terms.md#rank-r)

---

### Machine learning
*From module: 03_ml_foundations*

- **Technical definition:** Methods that learn parameters of a function from example data to make predictions on new data.
- **Plain English:** Teaching a program by examples instead of writing rules.
- **Example / analogy:** Learning to spot spam by seeing thousands of emails labelled spam / not spam.
- **Snippet:**
```python
model.fit(X_train, y_train)
model.predict(X_new)
```
- **Snippet explained:** fit learns from examples, predict applies what was learned.
- **Related terms:** [Feature](ml_terms.md#feature), [Label](ml_terms.md#label), [Parameter (model)](ml_terms.md#parameter-model)

---

### Mixed precision
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** Training with 16-bit math for speed while keeping some state (master weights, optimizer) in fp32.
- **Plain English:** Doing most maths in a smaller number format to go faster.
- **Example / analogy:** Rough sketches for speed, a clean master copy for accuracy.
- **Snippet:**
```python
torch.autocast(device_type='cuda', dtype=torch.bfloat16)
```
- **Snippet explained:** Context manager running eligible ops in bf16.
- **Related terms:** [Quantization (weights)](llm_terms.md#quantization-weights), [GPU memory estimate](llm_terms.md#gpu-memory-estimate)

---

### ML pipeline
*From module: 10_ml_system_architecture*

- **Technical definition:** An ordered, repeatable sequence of stages (collect, preprocess, feature extraction, train, evaluate, save, serve, monitor) implemented as code.
- **Plain English:** The assembly line that turns raw data into a working model and keeps it working.
- **Example / analogy:** A factory line: each station does one job in order.
- **Snippet:**
```python
from sklearn.pipeline import Pipeline
Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
```
- **Snippet explained:** Chains feature extraction and a model into one object with fit/predict.
- **Related terms:** [Training pipeline](ml_terms.md#training-pipeline), [Inference pipeline](ml_terms.md#inference-pipeline), [Feature engineering](ml_terms.md#feature-engineering)

---

### Model artifact
*From module: 10_ml_system_architecture*

- **Technical definition:** The saved output of training that serving loads: fitted model/weights plus any preprocessing, usually with metadata.
- **Plain English:** The saved file containing the trained model.
- **Example / analogy:** A printed textbook.
- **Snippet:**
```python
joblib.dump(pipeline, 'artifacts/model.joblib')
```
- **Snippet explained:** Serializes the whole fitted pipeline.
- **Related terms:** [Serialization](python_syntax.md#serialization), [Model registry](api_and_deployment_terms.md#model-registry)

---

### Model monitoring
*From module: 10_ml_system_architecture*

- **Technical definition:** Observing a deployed model's service health (latency, errors) and data/prediction quality (drift, accuracy when labels arrive) with alerts.
- **Plain English:** Watching the model after launch to see if it still works.
- **Example / analogy:** Dashboard in a car.
- **Snippet:**
```python
drift_report('normal traffic', ref_texts, live_texts)
```
- **Snippet explained:** Compares a live feature distribution to the training reference.
- **Related terms:** [Data drift](ml_terms.md#data-drift), [Structured logging](api_and_deployment_terms.md#structured-logging)

---

### Model serialization (joblib)
*From module: 03_ml_foundations*

- **Technical definition:** Saving a trained model object to disk (joblib.dump) and restoring it later (joblib.load).
- **Plain English:** Saving the trained model like a document so you can reopen it.
- **Example / analogy:** Saving a game so you don't replay from the start.
- **Snippet:**
```python
joblib.dump(model, 'model.joblib')
model = joblib.load('model.joblib')
```
- **Snippet explained:** Writes and reads the whole fitted pipeline.
- **Related terms:** [Pipeline (sklearn)](ml_terms.md#pipeline-sklearn)

---

### MSE
*From module: 03_ml_foundations*

- **Technical definition:** Mean squared error: average of squared differences between prediction and target; used for regression.
- **Plain English:** Average of the squared mistakes.
- **Example / analogy:** Squaring makes big mistakes count extra.
- **Snippet:**
```python
nn.MSELoss()(pred, target)
```
- **Snippet explained:** PyTorch built-in MSE.
- **Related terms:** [Loss function](ml_terms.md#loss-function)

---

### nn.Module
*From module: 03_ml_foundations*

- **Technical definition:** Base class of PyTorch models; subclasses define layers in __init__ and computation in forward.
- **Plain English:** The template every PyTorch model is built from.
- **Example / analogy:** A Lego baseplate for layers.
- **Snippet:**
```python
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)
```
- **Snippet explained:** Registers a linear layer and defines how input flows through it.
- **Related terms:** [Tensor](ml_terms.md#tensor), [Training loop](ml_terms.md#training-loop)

---

### Optimizer
*From module: 03_ml_foundations*

- **Technical definition:** Algorithm that updates parameters from gradients (SGD, Adam).
- **Plain English:** The rule that decides how to turn gradients into updates.
- **Example / analogy:** The driver steering down the hill.
- **Snippet:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```
- **Snippet explained:** Adam adapts step sizes per parameter.
- **Related terms:** [Gradient descent](ml_terms.md#gradient-descent), [Learning rate](ml_terms.md#learning-rate)

---

### Optimizer (AdamW)
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** An algorithm that updates weights from gradients; AdamW uses per-parameter adaptive step sizes plus decoupled weight decay.
- **Plain English:** The coach that decides how big a correction each weight gets.
- **Example / analogy:** A GPS recalculating the route using recent traffic history.
- **Snippet:**
```python
opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
```
- **Snippet explained:** Creates an optimizer over all parameters; call opt.step() after backward().
- **Related terms:** [Learning rate (training)](ml_terms.md#learning-rate-training), [Weight decay](ml_terms.md#weight-decay), [Training loop](ml_terms.md#training-loop)

---

### Overfitting
*From module: 03_ml_foundations*

- **Technical definition:** When a model fits training noise and performs poorly on unseen data; seen as low train loss and high validation loss.
- **Plain English:** Memorising instead of understanding.
- **Example / analogy:** A student who memorises past papers but can't solve new questions.
- **Snippet:**
```python
np.polyfit(x_train, y_train, 12)   # very flexible curve
```
- **Snippet explained:** A degree-12 polynomial on 15 points memorises noise.
- **Related terms:** [Underfitting](ml_terms.md#underfitting), [Train/validation split](ml_terms.md#trainvalidation-split), [Early stopping](ml_terms.md#early-stopping)

---

### Overfitting (training)
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** The model fits training data (low train loss) but does worse on unseen data (high validation loss).
- **Plain English:** Memorising instead of understanding.
- **Example / analogy:** A student who memorises last year's exam answers.
- **Snippet:**
```python
print(train_loss, val_loss)   # train falls, val rises
```
- **Snippet explained:** Diverging train and validation loss is the warning sign.
- **Related terms:** [Validation set](ml_terms.md#validation-set), [Epoch](ml_terms.md#epoch), [Weight decay](ml_terms.md#weight-decay)

---

### Parameter (model)
*From module: 03_ml_foundations*

- **Technical definition:** A number inside a model that is learned from data (e.g. weights and biases); distinct from hyperparameters set by the human.
- **Plain English:** The knobs the training process turns.
- **Example / analogy:** The tuning dials on a radio.
- **Snippet:**
```python
sum(p.numel() for p in model.parameters())
```
- **Snippet explained:** Counts all learnable numbers; LLMs have billions.
- **Related terms:** [Hyperparameter](ml_terms.md#hyperparameter), [Weight](ml_terms.md#weight), [Gradient descent](ml_terms.md#gradient-descent)

---

### Parameter (weight)
*From module: 04_llm_fundamentals*

- **Technical definition:** A number inside the model that is adjusted during training; collectively they store what the model has learned.
- **Plain English:** The knobs the training process turns.
- **Example / analogy:** distilgpt2 has 81,912,576 of them.
- **Snippet:**
```python
sum(p.numel() for p in model.parameters())
```
- **Snippet explained:** Counts every number in every weight tensor.
- **Related terms:** [Parameter count](llm_terms.md#parameter-count), [Backpropagation](ml_terms.md#backpropagation), [Data type (dtype)](ml_terms.md#data-type-dtype)

---

### Pipeline (sklearn)
*From module: 03_ml_foundations*

- **Technical definition:** sklearn object chaining preprocessing steps and a final estimator into one fit/predict unit.
- **Plain English:** Preprocessing and model packaged together.
- **Example / analogy:** An assembly line that always runs the same steps in the same order.
- **Snippet:**
```python
Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
```
- **Snippet explained:** Two named steps executed in sequence.
- **Related terms:** [Data leakage](ml_terms.md#data-leakage), [Model serialization (joblib)](ml_terms.md#model-serialization-joblib), [TF-IDF](ml_terms.md#tf-idf)

---

### Postprocessing
*From module: 10_ml_system_architecture*

- **Technical definition:** Turning raw model outputs into the final response: thresholds, label mapping, formatting, abstaining when unsure.
- **Plain English:** Final tidying and decision rules after the model speaks.
- **Example / analogy:** The editor polishing a draft.
- **Snippet:**
```python
label = classes[best] if confidence >= 0.6 else 'uncertain'
```
- **Snippet explained:** Low-confidence predictions become 'uncertain'.
- **Related terms:** [Inference pipeline](ml_terms.md#inference-pipeline), [Confidence](ml_terms.md#confidence)

---

### Precision
*From module: 03_ml_foundations*

- **Technical definition:** TP / (TP + FP): of items predicted positive, the fraction truly positive.
- **Plain English:** When I say yes, how often am I right?
- **Example / analogy:** A search engine returning mostly relevant results.
- **Snippet:**
```python
precision_score(y_true, y_pred)
```
- **Snippet explained:** Computes TP/(TP+FP).
- **Related terms:** [Recall](ml_terms.md#recall), [F1 score](ml_terms.md#f1-score), [Confusion matrix](ml_terms.md#confusion-matrix)

---

### Preprocessing
*From module: 10_ml_system_architecture*

- **Technical definition:** Cleaning and normalising raw inputs (lowercasing, removing punctuation, handling missing values) so training and serving see consistent data.
- **Plain English:** Tidying the data before the model sees it.
- **Example / analogy:** Washing and chopping vegetables.
- **Snippet:**
```python
re.sub(r'[^a-z0-9\s]', '', text.lower())
```
- **Snippet explained:** Lowercase then delete anything that is not a letter, digit or space.
- **Related terms:** [Feature engineering](ml_terms.md#feature-engineering), [Training/serving skew](ml_terms.md#trainingserving-skew)

---

### PSI
*From module: 10_ml_system_architecture*

- **Technical definition:** Population Stability Index: sum over bins of (live% - reference%) x ln(live%/reference%); under 0.1 stable, 0.1-0.25 watch, over 0.25 drift.
- **Plain English:** A single score for how different two distributions are.
- **Example / analogy:** Comparing today's crowd to a normal day's crowd.
- **Snippet:**
```python
np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct))
```
- **Snippet explained:** Per-bin differences weighted by the log ratio.
- **Related terms:** [Data drift](ml_terms.md#data-drift)

---

### Recall
*From module: 03_ml_foundations*

- **Technical definition:** TP / (TP + FN): of all true positives, the fraction found.
- **Plain English:** Of all real yes cases, how many did I catch?
- **Example / analogy:** A smoke alarm that catches every fire.
- **Snippet:**
```python
recall_score(y_true, y_pred)
```
- **Snippet explained:** Computes TP/(TP+FN).
- **Related terms:** [Precision](ml_terms.md#precision), [F1 score](ml_terms.md#f1-score)

---

### Reproducibility
*From module: 05_llm_parameters*

- **Technical definition:** Being able to get the same results by rerunning the experiment: fixed seeds, recorded configs, pinned library versions.
- **Plain English:** A recipe precise enough that anyone gets the same cake.
- **Example / analogy:** config.json plus fixed seeds let you compare runs fairly.
- **Snippet:**
```python
torch.manual_seed(0)
```
- **Snippet explained:** Seeding is one ingredient; recording config and versions are the others.
- **Related terms:** [Seed (random seed)](ml_terms.md#seed-random-seed), [Configuration file](python_syntax.md#configuration-file), [Hyperparameter](ml_terms.md#hyperparameter)

---

### safetensors
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** A safe, fast tensor file format used by Hugging Face that cannot execute code on load.
- **Plain English:** A weights file format that is quick and safe to open.
- **Example / analogy:** A sealed zip of numbers versus a pickle that could contain surprises.
- **Snippet:**
```python
from safetensors.torch import load_file
w = load_file('model.safetensors')
```
- **Snippet explained:** Loads a dict of tensors from disk.
- **Related terms:** [Checkpoint](ml_terms.md#checkpoint), [state_dict](ml_terms.md#state_dict)

---

### Seed (random seed)
*From module: 05_llm_parameters*

- **Technical definition:** A number initialising a pseudo-random generator so that 'random' results can be repeated.
- **Plain English:** Starting a shuffled deck from the same arrangement every time.
- **Example / analogy:** torch.manual_seed(7) before each generate() call isolates the effect of the parameter you changed.
- **Snippet:**
```python
torch.manual_seed(7)
```
- **Snippet explained:** Fixes PyTorch's global random stream.
- **Related terms:** [Sampling (decoding)](llm_terms.md#sampling-decoding), [Reproducibility](ml_terms.md#reproducibility)

---

### SGD (stochastic gradient descent)
*From module: 05_llm_parameters*

- **Technical definition:** Update rule weight -= lr * gradient computed on a mini-batch; optional momentum smooths updates.
- **Plain English:** The simplest downhill rule: step against the slope.
- **Example / analogy:** Needs careful learning-rate tuning; momentum=0.9 improved the toy loss from 0.39 to 0.24.
- **Snippet:**
```python
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```
- **Snippet explained:** momentum keeps part of the previous step so updates are smoother.
- **Related terms:** [Optimizer](ml_terms.md#optimizer), [Adam and AdamW](ml_terms.md#adam-and-adamw), [Learning rate](ml_terms.md#learning-rate)

---

### Softmax
*From module: 03_ml_foundations*

- **Technical definition:** Function exp(z_i)/sum(exp(z)) turning logits into probabilities that sum to 1.
- **Plain English:** Scores -> percentages.
- **Example / analogy:** Share of the pie each option gets.
- **Snippet:**
```python
e = np.exp(z - z.max()); probs = e / e.sum()
```
- **Snippet explained:** Subtract max for numerical stability, exponentiate, normalise.
- **Related terms:** [Logits](ml_terms.md#logits), [Cross-entropy loss](ml_terms.md#cross-entropy-loss), [Temperature (softmax)](ml_terms.md#temperature-softmax)

---

### state_dict
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** A Python dict mapping each layer's parameter name to its tensor.
- **Plain English:** The model's weights as a labelled dictionary.
- **Example / analogy:** A parts list with a label on every part.
- **Snippet:**
```python
sd = model.state_dict()
print(next(iter(sd)))
```
- **Snippet explained:** Prints the first parameter name, e.g. transformer.wte.weight.
- **Related terms:** [Checkpoint](ml_terms.md#checkpoint), [safetensors](ml_terms.md#safetensors)

---

### Supervised learning
*From module: 03_ml_foundations*

- **Technical definition:** Learning from input-output pairs (X, y) to predict y for new X.
- **Plain English:** Learning with an answer key.
- **Example / analogy:** Flashcards with the answer on the back.
- **Snippet:**
```python
model.fit(X, y)
```
- **Snippet explained:** Both X and y are given.
- **Related terms:** [Label](ml_terms.md#label), [Machine learning](ml_terms.md#machine-learning)

---

### Temperature (softmax)
*From module: 03_ml_foundations*

- **Technical definition:** Divisor applied to logits before softmax: <1 sharpens, >1 flattens the distribution.
- **Plain English:** How adventurous the choice is.
- **Example / analogy:** A low-temperature gas is calm, a high-temperature one is chaotic.
- **Snippet:**
```python
softmax(logits / 0.5)
```
- **Snippet explained:** Dividing by 0.5 doubles the logits, making the top choice more dominant.
- **Related terms:** [Softmax](ml_terms.md#softmax)

---

### Tensor
*From module: 03_ml_foundations*

- **Technical definition:** N-dimensional array (PyTorch) that can track gradients and run on GPUs.
- **Plain English:** A numpy array on steroids.
- **Example / analogy:** A spreadsheet that may have more than two dimensions.
- **Snippet:**
```python
t = torch.tensor([ [1., 2.], [3., 4.] ])
print(t.shape)
```
- **Snippet explained:** A 2x2 tensor; shape is torch.Size([2, 2]).
- **Related terms:** [Autograd](ml_terms.md#autograd), [Broadcasting](ml_terms.md#broadcasting)

---

### TensorDataset
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** A dataset wrapping one or more tensors, indexed along the first dimension.
- **Plain English:** A table where row i holds the i-th example.
- **Example / analogy:** A spreadsheet with one example per row.
- **Snippet:**
```python
ds = TensorDataset(blocks, labels)
print(len(ds), ds[0][0].shape)
```
- **Snippet explained:** ds[i] returns the i-th row from each tensor as a tuple.
- **Related terms:** [DataLoader](ml_terms.md#dataloader)

---

### TF-IDF
*From module: 03_ml_foundations*

- **Technical definition:** Term frequency x inverse document frequency: weights words that are frequent in a document but rare across documents.
- **Plain English:** Gives rare, informative words more weight than 'the'.
- **Example / analogy:** A rare word like 'refund' says more about a review than 'and'.
- **Snippet:**
```python
TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)
```
- **Snippet explained:** Converts texts into weighted word/bigram vectors.
- **Related terms:** [Feature](ml_terms.md#feature), [Pipeline (sklearn)](ml_terms.md#pipeline-sklearn)

---

### Train/test split
*From module: 10_ml_system_architecture*

- **Technical definition:** Separating labelled data into a part used to fit the model and a held-out part used only to estimate performance on unseen data.
- **Plain English:** Keep some examples hidden so you can test honestly.
- **Example / analogy:** Practice problems vs the real exam.
- **Snippet:**
```python
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
```
- **Snippet explained:** 80/20 split with a fixed seed for reproducibility.
- **Related terms:** [Data leakage](ml_terms.md#data-leakage), [Evaluation metric](ml_terms.md#evaluation-metric)

---

### Train/validation split
*From module: 03_ml_foundations*

- **Technical definition:** Dividing data into training data (used to fit) and held-out validation data (used to evaluate generalisation).
- **Plain English:** Hold some questions back to test the student on unseen material.
- **Example / analogy:** Practice exam vs real exam.
- **Snippet:**
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
```
- **Snippet explained:** 75/25 split with a fixed seed.
- **Related terms:** [Overfitting](ml_terms.md#overfitting), [Data leakage](ml_terms.md#data-leakage)

---

### Trainable parameters
*From module: 06_llm_training_and_fast_inference*

- **Technical definition:** Parameters with requires_grad=True that the optimizer will update.
- **Plain English:** The weights that are allowed to change.
- **Example / analogy:** The unlocked dials on a mixing desk.
- **Snippet:**
```python
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
- **Snippet explained:** Counts elements in all trainable tensors.
- **Related terms:** [LoRA](llm_terms.md#lora), [Optimizer (AdamW)](ml_terms.md#optimizer-adamw)

---

### Training loop
*From module: 03_ml_foundations*

- **Technical definition:** Repeated cycle of forward pass, loss, zero_grad, backward, optimizer step over batches and epochs.
- **Plain English:** The routine every model follows to learn.
- **Example / analogy:** Practice: attempt, get feedback, adjust, repeat.
- **Snippet:**
```python
pred = model(xb)
loss = loss_fn(pred, yb)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
- **Snippet explained:** The five lines of one training step.
- **Related terms:** [Epoch](ml_terms.md#epoch), [Batch size](ml_terms.md#batch-size), [Optimizer](ml_terms.md#optimizer)

---

### Training pipeline
*From module: 10_ml_system_architecture*

- **Technical definition:** The offline process that prepares data, fits and evaluates a model and writes an artifact plus metadata; runs on a schedule or on demand.
- **Plain English:** The 'teaching' program, run occasionally.
- **Example / analogy:** Writing a textbook.
- **Snippet:**
```python
metadata = train(n_samples=200, C=1.0)
```
- **Snippet explained:** Fits, evaluates and saves model.joblib and metadata.json.
- **Related terms:** [Inference pipeline](ml_terms.md#inference-pipeline), [Model artifact](ml_terms.md#model-artifact)

---

### Training/serving skew
*From module: 10_ml_system_architecture*

- **Technical definition:** A mismatch between how data is processed in training and in production (different cleaning, vocabulary, units), causing silent accuracy loss.
- **Plain English:** The model is taught one way and tested another.
- **Example / analogy:** Learning chess then being asked to play with checkers rules.
- **Snippet:**
```python
wrong = clf.predict(vec.transform(live_texts))   # forgot the lowercase step used in training
```
- **Snippet explained:** Unseen uppercase tokens produce empty vectors and wrong predictions without any error.
- **Related terms:** [Preprocessing](ml_terms.md#preprocessing), [Model monitoring](ml_terms.md#model-monitoring)

---

### Underfitting
*From module: 03_ml_foundations*

- **Technical definition:** Model too simple to capture the pattern; both training and validation error are high.
- **Plain English:** Not learning enough.
- **Example / analogy:** Drawing a straight line through a curve.
- **Snippet:**
```python
np.polyfit(x, y, 1)
```
- **Snippet explained:** A line can't fit a sine wave.
- **Related terms:** [Overfitting](ml_terms.md#overfitting)

---

### Validation set
*From module: 05_llm_parameters*

- **Technical definition:** A portion of data held out from training and used to monitor generalisation and tune hyperparameters.
- **Plain English:** Practice exam questions you never trained on.
- **Example / analogy:** The first 300 points trained the model; the next 300 validated it.
- **Snippet:**
```python
X_train, X_val = X[:300], X[300:]
```
- **Snippet explained:** Slicing splits data into two non-overlapping parts.
- **Related terms:** [Overfitting](ml_terms.md#overfitting), [Early stopping](ml_terms.md#early-stopping), [Hyperparameter](ml_terms.md#hyperparameter)

---

### Vectorizer
*From module: 10_ml_system_architecture*

- **Technical definition:** An object that converts text into numeric vectors (e.g. TfidfVectorizer); it must be fitted on training data and reused unchanged at inference.
- **Plain English:** The translator from words to numbers.
- **Example / analogy:** A dictionary mapping each word to a column number.
- **Snippet:**
```python
vec.transform(['great film'])
```
- **Snippet explained:** Uses the already learned vocabulary; unseen words are ignored.
- **Related terms:** [TF-IDF](ml_terms.md#tf-idf), [fit vs transform](ml_terms.md#fit-vs-transform)

---

### Weight
*From module: 03_ml_foundations*

- **Technical definition:** A learned multiplier applied to an input within a model.
- **Plain English:** How much a feature matters.
- **Example / analogy:** A recipe's amount for each ingredient.
- **Snippet:**
```python
y_hat = w * x + b
```
- **Snippet explained:** w is the weight, b the bias.
- **Related terms:** [Bias (parameter)](ml_terms.md#bias-parameter), [Parameter (model)](ml_terms.md#parameter-model)

---

### Weight decay
*From module: 05_llm_parameters*

- **Technical definition:** A regularisation penalty that gently pulls weights toward zero each update.
- **Plain English:** A small tax on big weights, discouraging over-complicated solutions.
- **Example / analogy:** AdamW(weight_decay=0.01) is a common LLM fine-tuning setting.
- **Snippet:**
```python
torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
```
- **Snippet explained:** Passed as an optimizer argument.
- **Related terms:** [Adam and AdamW](ml_terms.md#adam-and-adamw), [Overfitting](ml_terms.md#overfitting)

---
