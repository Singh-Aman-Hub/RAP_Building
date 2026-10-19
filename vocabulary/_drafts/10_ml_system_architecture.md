### ML pipeline
<!-- cat: ml -->
- **Technical definition:** An ordered, repeatable sequence of stages (collect, preprocess, feature extraction, train, evaluate, save, serve, monitor) implemented as code.
- **Plain English:** The assembly line that turns raw data into a working model and keeps it working.
- **Example / analogy:** A factory line: each station does one job in order.
- **Snippet:**
```python
from sklearn.pipeline import Pipeline
Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])
```
- **Snippet explained:** Chains feature extraction and a model into one object with fit/predict.
- **Related terms:** [[Training pipeline]], [[Inference pipeline]], [[Feature engineering]]

### Data collection
<!-- cat: ml -->
- **Technical definition:** Gathering raw examples from files, databases, APIs or user feedback that represent the cases the model will face in production.
- **Plain English:** Getting the examples the model will learn from.
- **Example / analogy:** Gathering ingredients before cooking.
- **Snippet:**
```python
texts, labels = get_training_data(n=200, seed=0)
```
- **Snippet explained:** A deterministic generator stands in for real data sources.
- **Related terms:** [[Preprocessing]], [[Data leakage]]

### Preprocessing
<!-- cat: ml -->
- **Technical definition:** Cleaning and normalising raw inputs (lowercasing, removing punctuation, handling missing values) so training and serving see consistent data.
- **Plain English:** Tidying the data before the model sees it.
- **Example / analogy:** Washing and chopping vegetables.
- **Snippet:**
```python
re.sub(r'[^a-z0-9\s]', '', text.lower())
```
- **Snippet explained:** Lowercase then delete anything that is not a letter, digit or space.
- **Related terms:** [[Feature engineering]], [[Training/serving skew]]

### Feature engineering
<!-- cat: ml -->
- **Technical definition:** Turning raw data into numeric features a model can use (TF-IDF vectors, embeddings, scaled numbers, counts).
- **Plain English:** Converting text or records into numbers the model understands.
- **Example / analogy:** Measuring ingredients into grams.
- **Snippet:**
```python
vec = TfidfVectorizer()
X = vec.fit_transform(texts)
```
- **Snippet explained:** Learns the vocabulary and returns a texts-by-words matrix.
- **Related terms:** [[TF-IDF]], [[fit vs transform]], [[Preprocessing]]

### TF-IDF
<!-- cat: ml -->
- **Technical definition:** Term Frequency-Inverse Document Frequency: a text weighting where a word scores high if frequent in a document but rare across all documents.
- **Plain English:** A score for 'how special is this word for this text'.
- **Example / analogy:** 'the' appears everywhere (low score); 'dreadful' is distinctive (high score).
- **Snippet:**
```python
TfidfVectorizer().fit_transform(['good movie', 'bad movie']).shape
```
- **Snippet explained:** Returns (2, 3): 2 texts, 3 distinct words.
- **Related terms:** [[Feature engineering]], [[Vectorizer]]

### Vectorizer
<!-- cat: ml -->
- **Technical definition:** An object that converts text into numeric vectors (e.g. TfidfVectorizer); it must be fitted on training data and reused unchanged at inference.
- **Plain English:** The translator from words to numbers.
- **Example / analogy:** A dictionary mapping each word to a column number.
- **Snippet:**
```python
vec.transform(['great film'])
```
- **Snippet explained:** Uses the already learned vocabulary; unseen words are ignored.
- **Related terms:** [[TF-IDF]], [[fit vs transform]]

### fit vs transform
<!-- cat: ml -->
- **Technical definition:** fit learns parameters from data (vocabulary, scaling statistics, weights); transform applies learned parameters; fit_transform does both on the same data. Only fit on training data.
- **Plain English:** fit = learn the rules, transform = apply the rules.
- **Example / analogy:** Studying (fit) then taking the exam with what you learned (transform).
- **Snippet:**
```python
Xtr = vec.fit_transform(train)
Xte = vec.transform(test)   # never fit on test
```
- **Snippet explained:** Test data is transformed with the training vocabulary.
- **Related terms:** [[Data leakage]], [[Vectorizer]]

### Data leakage
<!-- cat: ml -->
- **Technical definition:** Information from outside the training set (e.g. test data statistics or the label) influencing training, producing overly optimistic evaluation.
- **Plain English:** Accidentally peeking at the answers.
- **Example / analogy:** Studying with the exam paper.
- **Snippet:**
```python
vec.fit(all_texts)   # BAD: includes test texts
```
- **Snippet explained:** Fitting preprocessing on all data leaks test information into training.
- **Related terms:** [[fit vs transform]], [[Train/test split]]

### Train/test split
<!-- cat: ml -->
- **Technical definition:** Separating labelled data into a part used to fit the model and a held-out part used only to estimate performance on unseen data.
- **Plain English:** Keep some examples hidden so you can test honestly.
- **Example / analogy:** Practice problems vs the real exam.
- **Snippet:**
```python
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
```
- **Snippet explained:** 80/20 split with a fixed seed for reproducibility.
- **Related terms:** [[Data leakage]], [[Evaluation metric]]

### Evaluation metric
<!-- cat: ml -->
- **Technical definition:** A number summarising model quality on held-out data (accuracy, precision/recall, F1, log loss...).
- **Plain English:** A score for how good the model is.
- **Example / analogy:** Exam marks.
- **Snippet:**
```python
accuracy_score(y_test, model.predict(X_test))
```
- **Snippet explained:** Fraction of correct predictions.
- **Related terms:** [[Log loss]], [[Train/test split]]

### Log loss
<!-- cat: ml -->
- **Technical definition:** A probabilistic metric penalising confident wrong predictions: -mean(log p(true class)); lower is better.
- **Plain English:** Rewards being confident and right, punishes being confident and wrong.
- **Example / analogy:** A quiz where bold wrong answers lose more points.
- **Snippet:**
```python
log_loss(y_test, model.predict_proba(X_test), labels=list(model.classes_))
```
- **Snippet explained:** Computes cross-entropy from predicted probabilities.
- **Related terms:** [[Evaluation metric]]

### Training pipeline
<!-- cat: ml -->
- **Technical definition:** The offline process that prepares data, fits and evaluates a model and writes an artifact plus metadata; runs on a schedule or on demand.
- **Plain English:** The 'teaching' program, run occasionally.
- **Example / analogy:** Writing a textbook.
- **Snippet:**
```python
metadata = train(n_samples=200, C=1.0)
```
- **Snippet explained:** Fits, evaluates and saves model.joblib and metadata.json.
- **Related terms:** [[Inference pipeline]], [[Model artifact]]

### Inference pipeline
<!-- cat: ml -->
- **Technical definition:** The online path that loads a saved artifact once and, per request, validates input, preprocesses, predicts and postprocesses.
- **Plain English:** The 'answering' program, run all the time.
- **Example / analogy:** A student answering exam questions with what they learned.
- **Snippet:**
```python
model, meta = load_artifact()
predict_text(model, 'What a superb hotel')
```
- **Snippet explained:** Loads once, predicts per request, never calls fit.
- **Related terms:** [[Training pipeline]], [[Postprocessing]]

### Postprocessing
<!-- cat: ml -->
- **Technical definition:** Turning raw model outputs into the final response: thresholds, label mapping, formatting, abstaining when unsure.
- **Plain English:** Final tidying and decision rules after the model speaks.
- **Example / analogy:** The editor polishing a draft.
- **Snippet:**
```python
label = classes[best] if confidence >= 0.6 else 'uncertain'
```
- **Snippet explained:** Low-confidence predictions become 'uncertain'.
- **Related terms:** [[Inference pipeline]], [[Confidence]]

### Confidence
<!-- cat: ml -->
- **Technical definition:** The probability a model assigns to its predicted class (e.g. max of predict_proba); not guaranteed to be calibrated.
- **Plain English:** How sure the model says it is.
- **Example / analogy:** A weather forecast's '70% chance'.
- **Snippet:**
```python
probs = model.predict_proba([text])[0]
confidence = probs.max()
```
- **Snippet explained:** Maximum class probability.
- **Related terms:** [[Postprocessing]], [[Log loss]]

### Training/serving skew
<!-- cat: ml -->
- **Technical definition:** A mismatch between how data is processed in training and in production (different cleaning, vocabulary, units), causing silent accuracy loss.
- **Plain English:** The model is taught one way and tested another.
- **Example / analogy:** Learning chess then being asked to play with checkers rules.
- **Snippet:**
```python
wrong = clf.predict(vec.transform(live_texts))   # forgot the lowercase step used in training
```
- **Snippet explained:** Unseen uppercase tokens produce empty vectors and wrong predictions without any error.
- **Related terms:** [[Preprocessing]], [[Model monitoring]]

### Model artifact
<!-- cat: ml -->
- **Technical definition:** The saved output of training that serving loads: fitted model/weights plus any preprocessing, usually with metadata.
- **Plain English:** The saved file containing the trained model.
- **Example / analogy:** A printed textbook.
- **Snippet:**
```python
joblib.dump(pipeline, 'artifacts/model.joblib')
```
- **Snippet explained:** Serializes the whole fitted pipeline.
- **Related terms:** [[Serialization]], [[Model registry]]

### Serialization
<!-- cat: ml -->
- **Technical definition:** Converting an in-memory object into bytes/files (pickle, joblib, safetensors, JSON) so it can be stored or transferred, and loading it back (deserialization).
- **Plain English:** Saving an object to disk so you can reload it later.
- **Example / analogy:** Freeze-drying food to rehydrate later.
- **Snippet:**
```python
joblib.dump(model, 'm.joblib')
model2 = joblib.load('m.joblib')
```
- **Snippet explained:** Round trip: save then load an equivalent object.
- **Related terms:** [[Pickle]], [[joblib]], [[safetensors]]

### Pickle
<!-- cat: python -->
- **Technical definition:** Python's built-in binary serialization format for arbitrary objects; loading a pickle can execute arbitrary code, so only load trusted files.
- **Plain English:** Python's own 'save anything' format - convenient but unsafe from strangers.
- **Example / analogy:** A sealed box that may contain a booby trap if you don't trust the sender.
- **Snippet:**
```python
with open('m.pkl', 'wb') as f:
    pickle.dump(model, f)
```
- **Snippet explained:** Writes the object in binary mode.
- **Related terms:** [[Serialization]], [[joblib]], [[safetensors]]

### joblib
<!-- cat: python -->
- **Technical definition:** A library for saving/loading Python objects (like pickle) with efficient handling of large numpy arrays; the standard way to store scikit-learn models. Same security caveats as pickle.
- **Plain English:** Pickle tuned for machine-learning models.
- **Example / analogy:** A better suitcase for bulky data.
- **Snippet:**
```python
import joblib
joblib.dump(model, 'm.joblib', compress=3)
```
- **Snippet explained:** Saves with compression level 3.
- **Related terms:** [[Pickle]], [[Serialization]]

### safetensors
<!-- cat: llm -->
- **Technical definition:** A tensor storage format holding only named arrays and metadata (no executable code), fast and safe to load; used by Hugging Face for model weights.
- **Plain English:** A safe file format for model weights.
- **Example / analogy:** A CD of data instead of an executable program.
- **Snippet:**
```python
from safetensors.numpy import save_file, load_file
save_file({'w': arr}, 'w.safetensors')
```
- **Snippet explained:** Stores named arrays; loading cannot run code.
- **Related terms:** [[Serialization]], [[Pickle]]

### Model registry
<!-- cat: api -->
- **Technical definition:** A store of versioned model artifacts with metadata and stage labels (aliases), supporting promotion and rollback; e.g. MLflow, SageMaker, Hugging Face Hub.
- **Plain English:** A library of model versions with labels like 'production'.
- **Example / analogy:** A document archive with 'current' stamped on one version.
- **Snippet:**
```python
reg.promote('v0002', 'production')
model, meta = reg.load('production')
```
- **Snippet explained:** Moves the label then loads whichever version it points to.
- **Related terms:** [[Model versioning]], [[Alias]], [[Model artifact]]

### Model versioning
<!-- cat: api -->
- **Technical definition:** Assigning immutable identifiers to trained models together with their data fingerprint, parameters, metrics and library versions for reproducibility and rollback.
- **Plain English:** Numbering each trained model and recording how it was made.
- **Example / analogy:** Software releases: v1.0, v1.1.
- **Snippet:**
```python
version = f'v{n:04d}'
```
- **Snippet explained:** Zero-padded version names sort correctly.
- **Related terms:** [[Model registry]], [[Alias]]

### Alias (model registry)
<!-- cat: api -->
- **Technical definition:** A movable label (production, staging) pointing to a specific model version, so serving code refers to the label and ops change the target.
- **Plain English:** A nickname for a version that you can re-point.
- **Example / analogy:** The 'current edition' sticker.
- **Snippet:**
```python
aliases = {'production': 'v0001'}
```
- **Snippet explained:** Rolling back is editing this mapping.
- **Related terms:** [[Model registry]], [[Rollback]]

### Rollback
<!-- cat: api -->
- **Technical definition:** Reverting to a previously working model/version quickly after a bad release, ideally by moving an alias.
- **Plain English:** Going back to the last good version.
- **Example / analogy:** Undo button for a release.
- **Snippet:**
```python
reg.promote('v0001', 'production')
```
- **Snippet explained:** Points production back to v0001.
- **Related terms:** [[Alias (model registry)]], [[Canary deployment]]

### Canary deployment
<!-- cat: api -->
- **Technical definition:** Releasing a new version to a small share of traffic first and promoting it only if metrics stay healthy.
- **Plain English:** Test the new model on a few users first.
- **Example / analogy:** The canary in the coal mine.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [[Rollback]], [[Shadow deployment]]

### Shadow deployment
<!-- cat: api -->
- **Technical definition:** Running a new model on real requests in parallel with production but discarding (only logging) its outputs to compare safely.
- **Plain English:** Let the new model watch and guess without affecting users.
- **Example / analogy:** A trainee shadowing an expert.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [[Canary deployment]], [[Model monitoring]]

### Model serving
<!-- cat: api -->
- **Technical definition:** Exposing a trained model so other programs can request predictions, typically via an HTTP API or a dedicated inference server.
- **Plain English:** Making the model available as a service.
- **Example / analogy:** A restaurant kitchen open for orders.
- **Snippet:**
```python
@app.post('/predict')
def predict(req): ...
```
- **Snippet explained:** A route that calls the model.
- **Related terms:** [[Inference pipeline]], [[Batch inference]], [[Real-time inference]]

### Real-time inference
<!-- cat: api -->
- **Technical definition:** Producing a prediction per request while the caller waits; optimises latency.
- **Plain English:** Answering immediately, one at a time.
- **Example / analogy:** A taxi on demand.
- **Snippet:**
```python
[model.predict([t])[0] for t in texts]   # one call per item
```
- **Snippet explained:** One model call per request.
- **Related terms:** [[Batch inference]], [[Latency]]

### Batch inference
<!-- cat: api -->
- **Technical definition:** Scoring many items together on a schedule or in bulk; optimises throughput and cost via vectorized calls.
- **Plain English:** Scoring a big pile at once, nobody waiting.
- **Example / analogy:** A bus carrying many passengers.
- **Snippet:**
```python
for piece in chunks(texts, 500):
    out.extend(model.predict(piece))
```
- **Snippet explained:** One vectorized call per 500 items.
- **Related terms:** [[Real-time inference]], [[Micro-batching]], [[Generator]]

### Micro-batching
<!-- cat: api -->
- **Technical definition:** Collecting requests for a few milliseconds and processing them as one batch to gain throughput with small added latency (used by LLM inference servers).
- **Plain English:** Waiting a moment so several requests share one model call.
- **Example / analogy:** An elevator that waits briefly for more riders.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [[Batch inference]], [[Throughput]]

### Generator
<!-- cat: python -->
- **Technical definition:** A function using `yield` that produces values lazily one at a time instead of building a whole list in memory.
- **Plain English:** A function that hands out results one by one.
- **Example / analogy:** A ticket dispenser versus a stack of printed tickets.
- **Snippet:**
```python
def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i+size]
```
- **Snippet explained:** Yields slices lazily.
- **Related terms:** [[Batch inference]]

### Latency
<!-- cat: api -->
- **Technical definition:** The time taken to answer one request, usually reported as percentiles (p50, p95, p99).
- **Plain English:** How long one answer takes.
- **Example / analogy:** Time from ordering to receiving your coffee.
- **Snippet:**
```python
np.percentile(latencies, [50, 95, 99])
```
- **Snippet explained:** Median, 95th and 99th percentile latency.
- **Related terms:** [[Throughput]], [[Tail latency]]

### Tail latency
<!-- cat: api -->
- **Technical definition:** The slowest requests (p95/p99); they drive timeouts and user frustration and are hidden by averages.
- **Plain English:** How bad the worst slow answers are.
- **Example / analogy:** The longest waits in a queue.
- **Snippet:**
```python
np.percentile(lat, 99)
```
- **Snippet explained:** 99% of requests were faster than this.
- **Related terms:** [[Latency]]

### Throughput
<!-- cat: api -->
- **Technical definition:** How many requests or items are processed per unit time (req/s, tokens/s).
- **Plain English:** How much work gets done per second.
- **Example / analogy:** Cars passing a toll booth per minute.
- **Snippet:**
```python
len(latencies) / duration
```
- **Snippet explained:** Completed requests divided by test duration.
- **Related terms:** [[Latency]], [[Little's Law]]

### Little's Law
<!-- cat: api -->
- **Technical definition:** In a stable system, average items in the system = arrival rate x average time in system (L = lambda x W); used for capacity planning.
- **Plain English:** Requests in flight = requests per second x seconds each takes.
- **Example / analogy:** A cafe: 10 customers/min x 3 min each = 30 people inside.
- **Snippet:**
```python
workers_needed = arrival_rate * latency_seconds
```
- **Snippet explained:** 200 req/s at 0.2 s needs about 40 concurrent workers.
- **Related terms:** [[Throughput]], [[Latency]], [[Autoscaling]]

### Load testing
<!-- cat: api -->
- **Technical definition:** Sending controlled, high-volume traffic to a service to measure throughput, latency percentiles and errors and find limits.
- **Plain English:** Stress-testing the service with many fake users.
- **Example / analogy:** Walking a crowd across a new bridge.
- **Snippet:**
```python
threads = [threading.Thread(target=client_loop, args=(...)) for _ in range(clients)]
```
- **Snippet explained:** Many simulated clients hit the endpoint concurrently.
- **Related terms:** [[Throughput]], [[Tail latency]]

### Caching
<!-- cat: api -->
- **Technical definition:** Storing results of expensive computations keyed by input so repeats are answered instantly; risks staleness and memory use.
- **Plain English:** Remembering answers you already computed.
- **Example / analogy:** Writing down a calculation result instead of redoing it.
- **Snippet:**
```python
@lru_cache(maxsize=1024)
def cached_predict(text): ...
```
- **Snippet explained:** Same text returns the saved result.
- **Related terms:** [[Latency]]

### Retry with backoff
<!-- cat: api -->
- **Technical definition:** Re-attempting a failed operation after increasing delays (e.g. 0.05, 0.1, 0.2 s), capped, only for transient errors on idempotent operations.
- **Plain English:** Try again, waiting longer each time.
- **Example / analogy:** Redialling a busy phone number with growing pauses.
- **Snippet:**
```python
time.sleep(base_delay * 2 ** (attempt - 1))
```
- **Snippet explained:** Doubles the wait each attempt.
- **Related terms:** [[Idempotent]], [[Timeout]]

### Timeout
<!-- cat: api -->
- **Technical definition:** A maximum time to wait for an operation; prevents hung dependencies from piling up requests.
- **Plain English:** Give up after a set time instead of waiting forever.
- **Example / analogy:** Hanging up after 30 seconds on hold.
- **Snippet:**
```python
future.result(timeout=0.1)
```
- **Snippet explained:** Raises TimeoutError if no result within 0.1 s.
- **Related terms:** [[Retry with backoff]], [[Load shedding]]

### Load shedding
<!-- cat: api -->
- **Technical definition:** Deliberately rejecting excess requests (429/503) when overloaded, using a bounded queue, so the service stays healthy for accepted requests.
- **Plain English:** Turning away extra work fast instead of collapsing.
- **Example / analogy:** A restaurant closing the door when full.
- **Snippet:**
```python
try:
    q.put_nowait(job)
except queue.Full:
    return 503
```
- **Snippet explained:** Rejects immediately when the queue is full.
- **Related terms:** [[Bounded queue]], [[Timeout]]

### Bounded queue
<!-- cat: api -->
- **Technical definition:** A FIFO queue with a maximum size that applies backpressure by blocking or rejecting producers when full.
- **Plain English:** A waiting line with a limited number of places.
- **Example / analogy:** A turnstile line that stops at capacity.
- **Snippet:**
```python
work = queue.Queue(maxsize=5)
```
- **Snippet explained:** At most 5 jobs wait.
- **Related terms:** [[Load shedding]]

### Autoscaling
<!-- cat: api -->
- **Technical definition:** Automatically adding or removing replicas/instances based on metrics such as CPU, queue depth or latency.
- **Plain English:** The system adds more servers when busy and removes them when quiet.
- **Example / analogy:** A supermarket opening extra checkouts at rush hour.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [[Little's Law]], [[Load balancer]]

### Load balancer
<!-- cat: api -->
- **Technical definition:** A component that distributes incoming requests across several service instances and routes around unhealthy ones using health checks.
- **Plain English:** A traffic cop spreading requests over servers.
- **Example / analogy:** A host seating guests at different tables.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [[Autoscaling]], [[Health check]]

### Structured logging
<!-- cat: api -->
- **Technical definition:** Emitting each log record as machine-readable key/value data (usually one JSON line) with fields like request_id and latency_ms.
- **Plain English:** Logs that computers can search and count.
- **Example / analogy:** A form with labelled fields instead of a paragraph.
- **Snippet:**
```python
logger.info('prediction', extra={'fields': {'request_id': rid, 'latency_ms': ms}})
```
- **Snippet explained:** Attaches fields that a JSON formatter writes out.
- **Related terms:** [[Request ID]], [[Model monitoring]]

### Request ID
<!-- cat: api -->
- **Technical definition:** A unique identifier attached to a request and all its log lines so one request can be traced across logs and services.
- **Plain English:** A tracking number for a request.
- **Example / analogy:** A parcel tracking number.
- **Snippet:**
```python
request_id = uuid.uuid4().hex[:8]
```
- **Snippet explained:** Short random id.
- **Related terms:** [[Structured logging]]

### Model monitoring
<!-- cat: ml -->
- **Technical definition:** Observing a deployed model's service health (latency, errors) and data/prediction quality (drift, accuracy when labels arrive) with alerts.
- **Plain English:** Watching the model after launch to see if it still works.
- **Example / analogy:** Dashboard in a car.
- **Snippet:**
```python
drift_report('normal traffic', ref_texts, live_texts)
```
- **Snippet explained:** Compares a live feature distribution to the training reference.
- **Related terms:** [[Data drift]], [[Structured logging]]

### Data drift
<!-- cat: ml -->
- **Technical definition:** A change in the distribution of model inputs between training and production, potentially lowering accuracy; detected with statistics such as PSI.
- **Plain English:** The kind of data coming in has changed.
- **Example / analogy:** A language model trained on formal text now seeing slang.
- **Snippet:**
```python
psi(reference_lengths, live_lengths) > 0.25
```
- **Snippet explained:** PSI above 0.25 signals significant drift.
- **Related terms:** [[Concept drift]], [[PSI]]

### Concept drift
<!-- cat: ml -->
- **Technical definition:** A change in the relationship between inputs and the correct output over time (new slang, new fraud tactics) that needs fresh labels to detect.
- **Plain English:** The right answer for the same input has changed.
- **Example / analogy:** 'Sick' now meaning 'great'.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [[Data drift]], [[Model monitoring]]

### PSI
<!-- cat: ml -->
- **Technical definition:** Population Stability Index: sum over bins of (live% - reference%) x ln(live%/reference%); under 0.1 stable, 0.1-0.25 watch, over 0.25 drift.
- **Plain English:** A single score for how different two distributions are.
- **Example / analogy:** Comparing today's crowd to a normal day's crowd.
- **Snippet:**
```python
np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct))
```
- **Snippet explained:** Per-bin differences weighted by the log ratio.
- **Related terms:** [[Data drift]]

### SQL injection
<!-- cat: api -->
- **Technical definition:** An attack where user input is concatenated into SQL text and changes the query; prevented with parameterized queries.
- **Plain English:** Sneaking database commands in through a text field.
- **Example / analogy:** Writing extra orders on a form the clerk blindly copies.
- **Snippet:**
```python
conn.execute('SELECT * FROM predictions WHERE label = ?', (user_text,))
```
- **Snippet explained:** Placeholder keeps the input as data, never as SQL.
- **Related terms:** [[Parameterized query]], [[SQLite]]

### Parameterized query
<!-- cat: api -->
- **Technical definition:** A SQL statement with placeholders (?) whose values are passed separately so the database treats them strictly as data.
- **Plain English:** A safe fill-in-the-blank SQL command.
- **Example / analogy:** A form with boxes: whatever you write stays inside the box.
- **Snippet:**
```python
conn.execute('INSERT INTO t (a) VALUES (?)', (value,))
```
- **Snippet explained:** Value bound to the placeholder.
- **Related terms:** [[SQL injection]], [[SQLite]]

### SQLite
<!-- cat: api -->
- **Technical definition:** A serverless SQL database stored in a single file, included in Python's standard library (sqlite3).
- **Plain English:** A small database in one file, no server needed.
- **Example / analogy:** A notebook with tables.
- **Snippet:**
```python
import sqlite3
conn = sqlite3.connect(':memory:')
```
- **Snippet explained:** Creates a temporary in-memory database.
- **Related terms:** [[Parameterized query]], [[Prediction log]]

### Prediction log
<!-- cat: api -->
- **Technical definition:** A stored record of each prediction (time, model version, input or its reference, output, confidence, latency) used for debugging, monitoring and future training data.
- **Plain English:** A diary of every answer the model gave.
- **Example / analogy:** A call-centre record of each call.
- **Snippet:**
```python
log.record('v0001', text, label, conf, ms)
```
- **Snippet explained:** Inserts one row per prediction.
- **Related terms:** [[SQLite]], [[Model monitoring]]

### Object storage
<!-- cat: api -->
- **Technical definition:** A service storing blobs (files) by key at large scale cheaply, such as S3 or GCS; the usual home for datasets and model artifacts.
- **Plain English:** A giant cloud file cabinet.
- **Example / analogy:** A warehouse with labelled boxes.
- **Snippet:**
n/a
- **Snippet explained:** n/a
- **Related terms:** [[Model registry]], [[Model artifact]]
