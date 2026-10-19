### Machine learning
<!-- cat: ml -->
- **Technical definition:** Methods that learn parameters of a function from example data to make predictions on new data.
- **Plain English:** Teaching a program by examples instead of writing rules.
- **Example / analogy:** Learning to spot spam by seeing thousands of emails labelled spam / not spam.
- **Snippet:**
```python
model.fit(X_train, y_train)
model.predict(X_new)
```
- **Snippet explained:** fit learns from examples, predict applies what was learned.
- **Related terms:** [[Feature]], [[Label]], [[Parameter (model)]]

### Feature
<!-- cat: ml -->
- **Technical definition:** An input variable used by a model to make a prediction.
- **Plain English:** A fact about each example that the model may use.
- **Example / analogy:** For house price: size, rooms, location.
- **Snippet:**
```python
X = np.array([ [50, 2], [80, 3] ])   # columns = features
```
- **Snippet explained:** Each row is a house; each column is a feature.
- **Related terms:** [[Label]], [[Feature scaling]]

### Label
<!-- cat: ml -->
- **Technical definition:** The target value the model should predict (the 'right answer' in supervised learning).
- **Plain English:** The answer key for each training example.
- **Example / analogy:** The house's real sale price.
- **Snippet:**
```python
y = np.array([200, 320])
```
- **Snippet explained:** One label per row of X.
- **Related terms:** [[Feature]], [[Supervised learning]]

### Supervised learning
<!-- cat: ml -->
- **Technical definition:** Learning from input-output pairs (X, y) to predict y for new X.
- **Plain English:** Learning with an answer key.
- **Example / analogy:** Flashcards with the answer on the back.
- **Snippet:**
```python
model.fit(X, y)
```
- **Snippet explained:** Both X and y are given.
- **Related terms:** [[Label]], [[Machine learning]]

### Parameter (model)
<!-- cat: ml -->
- **Technical definition:** A number inside a model that is learned from data (e.g. weights and biases); distinct from hyperparameters set by the human.
- **Plain English:** The knobs the training process turns.
- **Example / analogy:** The tuning dials on a radio.
- **Snippet:**
```python
sum(p.numel() for p in model.parameters())
```
- **Snippet explained:** Counts all learnable numbers; LLMs have billions.
- **Related terms:** [[Hyperparameter]], [[Weight]], [[Gradient descent]]

### Hyperparameter
<!-- cat: ml -->
- **Technical definition:** A setting chosen before/outside training (learning rate, batch size, epochs, number of layers).
- **Plain English:** A knob that you set, not the model.
- **Example / analogy:** Oven temperature and cooking time.
- **Snippet:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
```
- **Snippet explained:** lr is a hyperparameter; weights are parameters.
- **Related terms:** [[Learning rate]], [[Batch size]], [[Epoch]]

### Weight
<!-- cat: ml -->
- **Technical definition:** A learned multiplier applied to an input within a model.
- **Plain English:** How much a feature matters.
- **Example / analogy:** A recipe's amount for each ingredient.
- **Snippet:**
```python
y_hat = w * x + b
```
- **Snippet explained:** w is the weight, b the bias.
- **Related terms:** [[Bias (parameter)]], [[Parameter (model)]]

### Bias (parameter)
<!-- cat: ml -->
- **Technical definition:** A learned constant added to a weighted sum, letting the model shift its output.
- **Plain English:** The baseline output when inputs are zero.
- **Example / analogy:** A starting offset.
- **Snippet:**
```python
y_hat = w * x + b
```
- **Snippet explained:** b shifts the line up/down.
- **Related terms:** [[Weight]]

### Loss function
<!-- cat: ml -->
- **Technical definition:** A function mapping predictions and targets to a single number that training minimises.
- **Plain English:** A score for how wrong the model is.
- **Example / analogy:** A golf score: lower is better.
- **Snippet:**
```python
loss = np.mean((y_hat - y) ** 2)
```
- **Snippet explained:** Mean squared error.
- **Related terms:** [[MSE]], [[Cross-entropy loss]], [[Gradient descent]]

### MSE
<!-- cat: ml -->
- **Technical definition:** Mean squared error: average of squared differences between prediction and target; used for regression.
- **Plain English:** Average of the squared mistakes.
- **Example / analogy:** Squaring makes big mistakes count extra.
- **Snippet:**
```python
nn.MSELoss()(pred, target)
```
- **Snippet explained:** PyTorch built-in MSE.
- **Related terms:** [[Loss function]]

### Cross-entropy loss
<!-- cat: ml -->
- **Technical definition:** -log of the probability assigned to the correct class; standard classification and language-model loss.
- **Plain English:** Penalises the model heavily when it is confident and wrong.
- **Example / analogy:** A quiz where a confident wrong answer costs the most.
- **Snippet:**
```python
F.cross_entropy(logits, target)
```
- **Snippet explained:** Takes raw logits and class indices; applies softmax internally.
- **Related terms:** [[Softmax]], [[Logits]], [[Loss function]]

### Logits
<!-- cat: ml -->
- **Technical definition:** Raw, unnormalised model output scores before softmax.
- **Plain English:** The model's raw scores for each option.
- **Example / analogy:** Judges' scores before converting to percentages.
- **Snippet:**
```python
logits = np.array([2.0, 1.0, 0.1])
```
- **Snippet explained:** Scores for 3 classes.
- **Related terms:** [[Softmax]], [[Cross-entropy loss]]

### Softmax
<!-- cat: ml -->
- **Technical definition:** Function exp(z_i)/sum(exp(z)) turning logits into probabilities that sum to 1.
- **Plain English:** Scores -> percentages.
- **Example / analogy:** Share of the pie each option gets.
- **Snippet:**
```python
e = np.exp(z - z.max()); probs = e / e.sum()
```
- **Snippet explained:** Subtract max for numerical stability, exponentiate, normalise.
- **Related terms:** [[Logits]], [[Cross-entropy loss]], [[Temperature (softmax)]]

### Temperature (softmax)
<!-- cat: ml -->
- **Technical definition:** Divisor applied to logits before softmax: <1 sharpens, >1 flattens the distribution.
- **Plain English:** How adventurous the choice is.
- **Example / analogy:** A low-temperature gas is calm, a high-temperature one is chaotic.
- **Snippet:**
```python
softmax(logits / 0.5)
```
- **Snippet explained:** Dividing by 0.5 doubles the logits, making the top choice more dominant.
- **Related terms:** [[Softmax]]

### Gradient
<!-- cat: ml -->
- **Technical definition:** The vector of partial derivatives of the loss with respect to each parameter; points toward steepest increase.
- **Plain English:** Which way and how steeply the loss changes if you nudge each parameter.
- **Example / analogy:** The slope of a hill under your feet.
- **Snippet:**
```python
grad_w = 2 * np.mean(error * x)
```
- **Snippet explained:** Derivative of MSE with respect to w.
- **Related terms:** [[Gradient descent]], [[Autograd]], [[Backpropagation]]

### Gradient descent
<!-- cat: ml -->
- **Technical definition:** Iteratively updating parameters opposite to the gradient: p = p - lr * grad.
- **Plain English:** Walk downhill step by step.
- **Example / analogy:** Hiking down a foggy mountain by feeling the slope.
- **Snippet:**
```python
w -= learning_rate * grad_w
```
- **Snippet explained:** Move w a small step against the slope.
- **Related terms:** [[Learning rate]], [[Gradient]], [[Optimizer]]

### Learning rate
<!-- cat: ml -->
- **Technical definition:** Hyperparameter controlling the step size of each parameter update.
- **Plain English:** How big each step downhill is.
- **Example / analogy:** Stride length when walking down a hill.
- **Snippet:**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
- **Snippet explained:** lr=0.01 scales every update.
- **Related terms:** [[Gradient descent]], [[Hyperparameter]]

### Epoch
<!-- cat: ml -->
- **Technical definition:** One complete pass through the entire training dataset.
- **Plain English:** One full read-through of the training data.
- **Example / analogy:** Reading the whole textbook once.
- **Snippet:**
```python
for epoch in range(10):
    ...
```
- **Snippet explained:** Ten passes over the data.
- **Related terms:** [[Batch size]], [[Training loop]]

### Batch size
<!-- cat: ml -->
- **Technical definition:** Number of samples processed before each parameter update.
- **Plain English:** How many examples you look at before adjusting.
- **Example / analogy:** Grading papers in stacks of 32 before giving feedback.
- **Snippet:**
```python
DataLoader(train_set, batch_size=32, shuffle=True)
```
- **Snippet explained:** Loader yields 32 samples at a time.
- **Related terms:** [[Epoch]], [[DataLoader]], [[Hyperparameter]]

### Optimizer
<!-- cat: ml -->
- **Technical definition:** Algorithm that updates parameters from gradients (SGD, Adam).
- **Plain English:** The rule that decides how to turn gradients into updates.
- **Example / analogy:** The driver steering down the hill.
- **Snippet:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```
- **Snippet explained:** Adam adapts step sizes per parameter.
- **Related terms:** [[Gradient descent]], [[Learning rate]]

### Autograd
<!-- cat: ml -->
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
- **Related terms:** [[Gradient]], [[Tensor]], [[Backpropagation]]

### Backpropagation
<!-- cat: ml -->
- **Technical definition:** Applying the chain rule backward through a network to compute gradients of the loss for every parameter.
- **Plain English:** How the blame for the error is passed back to each weight.
- **Example / analogy:** Tracing an error back through every step of a production line.
- **Snippet:**
```python
loss.backward()
```
- **Snippet explained:** Triggers backpropagation through the computation graph.
- **Related terms:** [[Autograd]], [[Gradient]]

### Tensor
<!-- cat: ml -->
- **Technical definition:** N-dimensional array (PyTorch) that can track gradients and run on GPUs.
- **Plain English:** A numpy array on steroids.
- **Example / analogy:** A spreadsheet that may have more than two dimensions.
- **Snippet:**
```python
t = torch.tensor([ [1., 2.], [3., 4.] ])
print(t.shape)
```
- **Snippet explained:** A 2x2 tensor; shape is torch.Size([2, 2]).
- **Related terms:** [[Autograd]], [[Broadcasting]]

### Broadcasting
<!-- cat: ml -->
- **Technical definition:** Numpy/PyTorch rule that expands smaller arrays across larger ones during element-wise operations.
- **Plain English:** Automatic repetition so shapes line up.
- **Example / analogy:** Adding the same tax to every price in a list.
- **Snippet:**
```python
(X - X.mean(axis=0)) / X.std(axis=0)
```
- **Snippet explained:** A (n,2) array minus a (2,) vector subtracts per column.
- **Related terms:** [[Tensor]], [[Feature scaling]]

### Feature scaling
<!-- cat: ml -->
- **Technical definition:** Rescaling features (e.g. standardisation) so they have comparable ranges, which speeds up training.
- **Plain English:** Putting all columns on a similar scale.
- **Example / analogy:** Converting height in cm and weight in kg into 'how many standard deviations from average'.
- **Snippet:**
```python
X_scaled = (X - X.mean(0)) / X.std(0)
```
- **Snippet explained:** Each column gets mean 0 and std 1.
- **Related terms:** [[Feature]], [[Broadcasting]]

### nn.Module
<!-- cat: ml -->
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
- **Related terms:** [[Tensor]], [[Training loop]]

### DataLoader
<!-- cat: ml -->
- **Technical definition:** PyTorch utility that batches, shuffles and iterates over a dataset.
- **Plain English:** Hands the model data in neat batches.
- **Example / analogy:** A cafeteria server giving trays of fixed size.
- **Snippet:**
```python
for xb, yb in DataLoader(ds, batch_size=32):
    ...
```
- **Snippet explained:** Each iteration yields one batch of inputs and labels.
- **Related terms:** [[Batch size]], [[Epoch]]

### Training loop
<!-- cat: ml -->
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
- **Related terms:** [[Epoch]], [[Batch size]], [[Optimizer]]

### Train/validation split
<!-- cat: ml -->
- **Technical definition:** Dividing data into training data (used to fit) and held-out validation data (used to evaluate generalisation).
- **Plain English:** Hold some questions back to test the student on unseen material.
- **Example / analogy:** Practice exam vs real exam.
- **Snippet:**
```python
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
```
- **Snippet explained:** 75/25 split with a fixed seed.
- **Related terms:** [[Overfitting]], [[Data leakage]]

### Overfitting
<!-- cat: ml -->
- **Technical definition:** When a model fits training noise and performs poorly on unseen data; seen as low train loss and high validation loss.
- **Plain English:** Memorising instead of understanding.
- **Example / analogy:** A student who memorises past papers but can't solve new questions.
- **Snippet:**
```python
np.polyfit(x_train, y_train, 12)   # very flexible curve
```
- **Snippet explained:** A degree-12 polynomial on 15 points memorises noise.
- **Related terms:** [[Underfitting]], [[Train/validation split]], [[Early stopping]]

### Underfitting
<!-- cat: ml -->
- **Technical definition:** Model too simple to capture the pattern; both training and validation error are high.
- **Plain English:** Not learning enough.
- **Example / analogy:** Drawing a straight line through a curve.
- **Snippet:**
```python
np.polyfit(x, y, 1)
```
- **Snippet explained:** A line can't fit a sine wave.
- **Related terms:** [[Overfitting]]

### Early stopping
<!-- cat: ml -->
- **Technical definition:** Stop training when validation loss stops improving to avoid overfitting.
- **Plain English:** Quit while you're ahead.
- **Example / analogy:** Stop studying when practice-test scores stop improving.
- **Snippet:**
```python
if val_loss > best_val_loss: patience_left -= 1
```
- **Snippet explained:** Count epochs without improvement and stop at zero.
- **Related terms:** [[Overfitting]], [[Epoch]]

### Data leakage
<!-- cat: ml -->
- **Technical definition:** Information from validation/test data leaking into training or preprocessing, inflating metrics.
- **Plain English:** Peeking at the exam answers.
- **Example / analogy:** Fitting a scaler on all data before splitting.
- **Snippet:**
```python
pipe = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
pipe.fit(X_train, y_train)
```
- **Snippet explained:** Pipeline fits the vectorizer on training data only.
- **Related terms:** [[Train/validation split]], [[Pipeline (sklearn)]]

### Accuracy
<!-- cat: ml -->
- **Technical definition:** Fraction of predictions that are correct.
- **Plain English:** How often the model is right.
- **Example / analogy:** Marks out of 100.
- **Snippet:**
```python
accuracy_score(y_true, y_pred)
```
- **Snippet explained:** Correct predictions / total.
- **Related terms:** [[Precision]], [[Recall]]

### Precision
<!-- cat: ml -->
- **Technical definition:** TP / (TP + FP): of items predicted positive, the fraction truly positive.
- **Plain English:** When I say yes, how often am I right?
- **Example / analogy:** A search engine returning mostly relevant results.
- **Snippet:**
```python
precision_score(y_true, y_pred)
```
- **Snippet explained:** Computes TP/(TP+FP).
- **Related terms:** [[Recall]], [[F1 score]], [[Confusion matrix]]

### Recall
<!-- cat: ml -->
- **Technical definition:** TP / (TP + FN): of all true positives, the fraction found.
- **Plain English:** Of all real yes cases, how many did I catch?
- **Example / analogy:** A smoke alarm that catches every fire.
- **Snippet:**
```python
recall_score(y_true, y_pred)
```
- **Snippet explained:** Computes TP/(TP+FN).
- **Related terms:** [[Precision]], [[F1 score]]

### F1 score
<!-- cat: ml -->
- **Technical definition:** Harmonic mean of precision and recall: 2PR/(P+R).
- **Plain English:** One number balancing precision and recall.
- **Example / analogy:** A team grade that's low if either teammate is weak.
- **Snippet:**
```python
f1_score(y_true, y_pred)
```
- **Snippet explained:** Returns the harmonic mean.
- **Related terms:** [[Precision]], [[Recall]]

### Confusion matrix
<!-- cat: ml -->
- **Technical definition:** Table counting true/false positives and negatives.
- **Plain English:** A scoreboard of right and wrong guesses by class.
- **Example / analogy:** A 2x2 grid of hits and misses.
- **Snippet:**
```python
confusion_matrix(y_true, y_pred)   # rows: truth; cols: prediction
```
- **Snippet explained:** Rows are truth, columns predictions.
- **Related terms:** [[Precision]], [[Recall]]

### TF-IDF
<!-- cat: ml -->
- **Technical definition:** Term frequency x inverse document frequency: weights words that are frequent in a document but rare across documents.
- **Plain English:** Gives rare, informative words more weight than 'the'.
- **Example / analogy:** A rare word like 'refund' says more about a review than 'and'.
- **Snippet:**
```python
TfidfVectorizer(ngram_range=(1, 2)).fit_transform(texts)
```
- **Snippet explained:** Converts texts into weighted word/bigram vectors.
- **Related terms:** [[Feature]], [[Pipeline (sklearn)]]

### Pipeline (sklearn)
<!-- cat: ml -->
- **Technical definition:** sklearn object chaining preprocessing steps and a final estimator into one fit/predict unit.
- **Plain English:** Preprocessing and model packaged together.
- **Example / analogy:** An assembly line that always runs the same steps in the same order.
- **Snippet:**
```python
Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
```
- **Snippet explained:** Two named steps executed in sequence.
- **Related terms:** [[Data leakage]], [[Model serialization (joblib)]], [[TF-IDF]]

### Model serialization (joblib)
<!-- cat: ml -->
- **Technical definition:** Saving a trained model object to disk (joblib.dump) and restoring it later (joblib.load).
- **Plain English:** Saving the trained model like a document so you can reopen it.
- **Example / analogy:** Saving a game so you don't replay from the start.
- **Snippet:**
```python
joblib.dump(model, 'model.joblib')
model = joblib.load('model.joblib')
```
- **Snippet explained:** Writes and reads the whole fitted pipeline.
- **Related terms:** [[Pipeline (sklearn)]]
