# Step 01 — The end-to-end ML pipeline, stage by stage

## The story so far
Module 09 ended with an API that serves one request correctly. But the model behind it came from a notebook, and the question is now how the whole journey from raw data to answer fits together and stays correct over months. Previous step: [09_api_development/10_streaming_sse](../../09_api_development/10_streaming_sse/).

## The pain point
A notebook mixes cleaning, features, training and evaluation in cells run in whatever order you remembered. Nobody can reproduce it, and test data quietly leaks into training.

## The spark
Name every stage (collect, clean, features, train, evaluate, save, serve), run them in a fixed order in one script, and bundle preprocessing with the model in one scikit-learn Pipeline.

## Goal
Walk collect → preprocess → features → train → evaluate → serialize → load → predict with real code, so every box in an architecture diagram has code behind it.

## Run it
```bash
cd 10_ml_system_architecture/01_pipeline_end_to_end && python main.py
```
Real output (shortened):
```
[1 collect]    200 train texts, 8 test texts; first = 'The book was boring.'
[2 preprocess] 'The movie was GREAT!!  ' -> 'the movie was great'
[3 features]   matrix shape (200, 32) = (n_texts, vocabulary_size); sample words: ['the', 'book', 'was', 'boring', 'honestly']
[4 train]      classes learned: ['negative', 'positive']
[5 evaluate]   accuracy = 1.00
[6 serialize]  saved 2481 bytes to model.joblib
[7 predict]    'Honestly the delivery was TERRIBLE!!' -> negative  probs={'negative': 0.726, 'positive': 0.274}
```
(`data.py` generates the dataset from templates; seeds make it reproducible. The test set is hand-written and never used in training.)

## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `[preprocess(t) for t in X]` | List comprehension | `[len(w) for w in words]` |
| `re.sub(r"[^a-z0-9\s]", "", text)` | Regex replace: `r"..."` raw string; `[^...]` = "anything except" | — |
| `random.Random(seed)` | Private random generator → reproducible data | — |
| `zip(*pairs)` | Unzip a list of pairs into two sequences | `zip(*[(1,'a'),(2,'b')])` |
| `f"{x:.2f}"` | f-string with 2 decimals | `f"{0.5:.2f}"` → `0.50` |
| `Path(tempfile.mkdtemp()) / "model.joblib"` | Temp folder + path join | — |

## Libraries & functions
| Name | Kind | Does | Returns |
|---|---|---|---|
| `TfidfVectorizer()` | scikit-learn | Text → numeric matrix; weights words frequent in a doc but rare overall. Lowercases and tokenizes by default | estimator |
| `.fit_transform(train)` | sklearn | Learn vocabulary **and** transform | sparse matrix (n_texts × n_words) |
| `.transform(test)` | sklearn | Apply the **already learned** vocabulary | sparse matrix |
| `LogisticRegression(max_iter=1000)` | sklearn | Linear classifier; learns a weight per word per class | estimator |
| `.fit(X, y)` / `.predict(X)` / `.predict_proba(X)` | sklearn | train / labels / class probabilities | — |
| `accuracy_score(y_true, y_pred)` | sklearn | fraction correct | float |
| `Pipeline([("tfidf", ...), ("clf", ...)])` | sklearn | Chain steps into one object (`fit`, `predict` run all steps) | estimator |
| `joblib.dump / load` | 3rd-party | Save / load object to a file | — |

## Execution trace
`predict_proba(["Honestly the delivery was TERRIBLE!!"])`: Pipeline → TfidfVectorizer lowercases + tokenizes → `['honestly','the','delivery','was','terrible']` → looks up each in the *training* vocabulary → builds a 1×32 vector → LogisticRegression multiplies by learned weights → softmax-like squash → `[0.726, 0.274]` for `[negative, positive]`.

## Why each stage exists (and what breaks it)
| Stage | Purpose | Typical failure |
|---|---|---|
| Collect | get representative data | biased/stale data → model that's wrong on real users |
| Preprocess | consistent clean input | different cleaning in training vs serving (step 02) |
| Features | numbers the model understands | fitting the vectorizer on test data = **data leakage** |
| Train | learn weights | overfitting; use validation data |
| Evaluate | honest estimate on unseen data | evaluating on training data; test set too small (8 here is a toy!) |
| Serialize | ship the exact fitted object | pickle/library version mismatch (step 03) |
| Predict | serve | cold start, latency (steps 06, 08) |

## Experiments
1. `n=10` instead of `200` → watch confidence fall and the vocabulary shrink.
2. Predict `'GREAT!!!'` vs `'great'` — same answer because the vectorizer lowercases/tokenizes.
3. Predict a sentence of words never seen in training: `'The zorp was blorf'` → probabilities near 0.5 (the model knows nothing).

## Debugging practice
Call `vectorizer.transform(X_test)` before `fit_transform` → `NotFittedError: This TfidfVectorizer instance is not fitted yet`. The message names the exact fix (`fit` first). Reading library errors fully usually saves a trip to a search engine.

## Rebuild checklist
- [ ] Rebuild the Pipeline and train/predict in ≤15 lines.
- [ ] Say which stage each of these belongs to: `re.sub`, `fit_transform`, `joblib.dump`, `accuracy_score`.
- [ ] Explain why you never `fit` on test data.

## The new problem (cliffhanger)
One script that trains and predicts hides a trap: in production training and serving are two different programs. Do they still agree? Next: [02_training_vs_inference_pipelines](../02_training_vs_inference_pipelines/).

## Say it aloud
> **Say it aloud (interview box):** An ML system is a pipeline: collect data, clean it, build features, train, evaluate on held-out data, save the fitted artifact, then load and serve it. I bundle preprocessing and the model in one scikit-learn Pipeline so training and serving cannot drift apart. The catch is that a pipeline hides steps I must still inspect when debugging, and a high score on a tiny template-based test set says little about real data.
