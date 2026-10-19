# Terms — Shared model and mock API (common/)

Folder: `12_streamlit_gradio/common` · Files covered: `model.py`, `mock_api.py`, `__init__.py`
Back to module: [12_streamlit_gradio README](../README.md)

**Terms in this folder:** [TF-IDF vectorizer (third-party: scikit-learn)](#tf-idf-vectorizer-third-party-scikit-learn) · [LogisticRegression (third-party: scikit-learn)](#logisticregression-third-party-scikit-learn) · [make_pipeline (third-party: scikit-learn)](#make_pipeline-third-party-scikit-learn) · [fit (third-party: scikit-learn)](#fit-third-party-scikit-learn) · [predict_proba and classes_ (third-party: scikit-learn)](#predict_proba-and-classes_-third-party-scikit-learn) · [argmax (third-party: numpy)](#argmax-third-party-numpy) · [Training data as list of tuples (concept)](#training-data-as-list-of-tuples-concept) · [zip(*pairs) unzip idiom (syntax)](#zippairs-unzip-idiom-syntax) · [Module-level cache with global (syntax)](#module-level-cache-with-global-syntax) · [Load-once pattern (concept)](#load-once-pattern-concept) · [Type hints with tuple return (syntax)](#type-hints-with-tuple-return-syntax) · [Raising ValueError for empty input (syntax)](#raising-valueerror-for-empty-input-syntax) · [round and float conversion (Python built-in)](#round-and-float-conversion-python-built-in) · [str.strip for blank-input check (Python built-in)](#strstrip-for-blank-input-check-python-built-in) · [if __name__ == "__main__" (syntax)](#if-__name__--__main__-syntax) · [Mock API (concept)](#mock-api-concept) · [API contract (concept)](#api-contract-concept) · [FastAPI app and routes (third-party: fastapi)](#fastapi-app-and-routes-third-party-fastapi) · [Pydantic BaseModel request schema (third-party: pydantic)](#pydantic-basemodel-request-schema-third-party-pydantic) · [HTTPException with status 422 (third-party: fastapi)](#httpexception-with-status-422-third-party-fastapi) · [Import from sibling module (syntax)](#import-from-sibling-module-syntax) · [uvicorn launch from the folder (command)](#uvicorn-launch-from-the-folder-command) · [Package marker __init__.py (file)](#package-marker-__init__py-file)

---

## TF-IDF vectorizer `(third-party: scikit-learn)`
**1. Definition:** `TfidfVectorizer` converts text into numeric vectors in which each word's weight is its frequency in the document scaled down by how common the word is across all documents (term frequency × inverse document frequency).
**2. In plain words:** A model can't read words, only numbers. TF-IDF counts words but gives rare, informative words more weight than ones like "the". It's a simple, classic way to turn sentences into features.
**3. Example to quote:** > "`TfidfVectorizer()` turns 'I love this' into a vector where 'love' carries more weight than a very common word."
**Seen in:** `model.py`

## LogisticRegression `(third-party: scikit-learn)`
**1. Definition:** A linear classifier that models the probability of each class by passing a weighted sum of features through a sigmoid or softmax function.
**2. In plain words:** Despite its name it's for classification. It learns a weight per word and turns the total into a probability, which is why `predict_proba` works on it. It's fast and a good baseline.
**3. Example to quote:** > "I used logistic regression because it trains in under a second and gives probabilities for the UI to show."
**Seen in:** `model.py`

## make_pipeline `(third-party: scikit-learn)`
**1. Definition:** Builds a `Pipeline` that chains transformers and a final estimator so that fitting and predicting run all steps in order.
**2. In plain words:** The vectorizer and the classifier become one object. Raw text goes in and a prediction comes out, so the same preprocessing is applied at training and at prediction time, which avoids train/serve mismatches.
**3. Example to quote:** > "`make_pipeline(TfidfVectorizer(), LogisticRegression())` lets me call `fit(texts, labels)` and `predict_proba([text])` on one object."
```python
_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
```
**Seen in:** `model.py`

## fit `(third-party: scikit-learn)`
**1. Definition:** The estimator method that learns parameters from training data (here texts and labels).
**2. In plain words:** Training happens in `fit`. After it runs, the model has learned word weights and can predict on new text.
**3. Example to quote:** > "`_model.fit(texts, labels)` learns which words point to positive or negative."
**Seen in:** `model.py`

## predict_proba and classes_ `(third-party: scikit-learn)`
**1. Definition:** `predict_proba(list_of_texts)` returns a 2-D array of class probabilities; `classes_` lists the class labels in the same column order.
**2. In plain words:** Each row sums to 1. Use `argmax` on the row to find the winning class and `classes_` to turn its index into a name.
**3. Example to quote:** > "`model.predict_proba([text])[0]` might be `[0.45, 0.55]` with classes `['negative', 'positive']`."
**Seen in:** `model.py`

## argmax `(third-party: numpy)`
**1. Definition:** `array.argmax()` returns the index of the largest value in the array.
**2. In plain words:** It answers "which class has the highest probability?" by giving a position, which you then look up in `classes_`.
**3. Example to quote:** > "`proba.argmax()` returns 1 for `[0.45, 0.55]`, so the label is `classes_[1]`, 'positive'."
**Seen in:** `model.py`

## Training data as list of tuples `(concept)`
**1. Definition:** A list of `(text, label)` pairs used as the labelled training set.
**2. In plain words:** Twelve short example sentences are enough for a demo model. It's deliberately tiny so there's nothing to download and the behaviour is easy to follow, though it won't be accurate on real-world text.
**3. Example to quote:** > "My training set is twelve labelled sentences, six positive and six negative, which is enough for a UI demo but not a real product."
**Seen in:** `model.py`

## zip(*pairs) unzip idiom `(syntax)`
**1. Definition:** `zip(*TRAIN)` unpacks the list of pairs as arguments and regroups them into one tuple of all texts and one of all labels.
**2. In plain words:** The star spreads the pairs out, and `zip` collects the first elements together and the second elements together. It's the standard way to split pairs into two lists.
**3. Example to quote:** > "`texts, labels = zip(*TRAIN)` splits my list of (text, label) pairs into two tuples."
**Seen in:** `model.py`

## Module-level cache with global `(syntax)`
**1. Definition:** A module variable such as `_model = None` that stores an object after first creation, updated in a function with the `global` statement.
**2. In plain words:** The first call trains the model and later calls reuse it. The leading underscore signals that it's meant to be private to the module.
**3. Example to quote:** > "`load_model()` trains once and stores the pipeline in `_model`, so repeat calls are instant."
**Seen in:** `model.py`

## Load-once pattern `(concept)`
**1. Definition:** Creating an expensive object a single time and reusing it for every later request.
**2. In plain words:** Rebuilding the model on every prediction would be slow. The same idea appears as `st.cache_resource` in Streamlit and as loading at API startup.
**3. Example to quote:** > "Training takes well under a second here, but the principle matters: load the model once and reuse it."
**Seen in:** `model.py`

## Type hints with tuple return `(syntax)`
**1. Definition:** Annotations like `text: str` and `-> tuple[str, float]` that document a function's parameter and return types without enforcing them.
**2. In plain words:** `predict(text: str) -> tuple[str, float]` tells readers that a string goes in and a label plus a score come out.
**3. Example to quote:** > "The signature says `predict` returns a pair: the label as text and the confidence as a number."
**Seen in:** `model.py`

## Raising ValueError for empty input `(syntax)`
**1. Definition:** `raise ValueError("...")` signals that an argument has an invalid value.
**2. In plain words:** The model refuses empty text with a clear error, which the callers translate: Streamlit shows `st.error`, the mock API returns HTTP 422.
**3. Example to quote:** > "`predict('')` raises `ValueError('Text must not be empty')`, and the mock API turns that into a 422."
**Seen in:** `model.py`, `mock_api.py`

## round and float conversion `(Python built-in)`
**1. Definition:** `float(x)` converts a value to a Python float; `round(x, 3)` rounds to three decimals.
**2. In plain words:** NumPy numbers aren't plain Python floats, and JSON encoders can struggle with them. Converting and rounding gives a clean number such as 0.548.
**3. Example to quote:** > "`round(float(proba[best]), 3)` turns a NumPy value into a tidy `0.548`."
**Seen in:** `model.py`

## str.strip for blank-input check `(Python built-in)`
**1. Definition:** `strip()` removes surrounding whitespace; `not text.strip()` is True for empty or whitespace-only text.
**2. In plain words:** It catches inputs that look non-empty but contain only spaces.
**3. Example to quote:** > "`if not text or not text.strip():` rejects both empty and whitespace-only sentences."
**Seen in:** `model.py`

## if __name__ == "__main__" `(syntax)`
**1. Definition:** A guard that runs the enclosed code only when the file is executed directly.
**2. In plain words:** Running `python model.py` prints three sample predictions, but importing the module from an app doesn't.
**3. Example to quote:** > "I can run `model.py` to see three predictions, and import it elsewhere without side effects."
**Seen in:** `model.py`

## Mock API `(concept)`
**1. Definition:** A small stand-in service that implements the same endpoints and response shape as the real API (here `GET /health` and `POST /predict`).
**2. In plain words:** It lets the UIs be developed and tested without the module 09 API running. If the real API is on port 8000, stop the mock; the UIs don't care which one answers.
**3. Example to quote:** > "I developed the Streamlit client against the mock API, then pointed it at the real one by starting that on port 8000."
**Seen in:** `mock_api.py`

## API contract `(concept)`
**1. Definition:** The agreed request and response format between client and server: endpoint paths, methods, field names and types.
**2. In plain words:** Both sides depend on it. If the real API names a field `confidence` while the mock says `score`, a client that hard-codes one will break, which is why the UI tolerates both.
**3. Example to quote:** > "The mock returns `{"label": ..., "score": ...}` for POST `/predict`, and my UI code relies on that contract."
**Seen in:** `mock_api.py` docstring

## FastAPI app and routes `(third-party: fastapi)`
**1. Definition:** `FastAPI(title=...)` creates the application; `@app.get` and `@app.post` decorators register handler functions for paths.
**2. In plain words:** Two small functions, `/health` and `/predict`, make a complete little service.
**3. Example to quote:** > "`@app.post("/predict")` is the endpoint the UIs call."
**Seen in:** `mock_api.py`

## Pydantic BaseModel request schema `(third-party: pydantic)`
**1. Definition:** A class inheriting from `BaseModel` whose annotated fields define and validate the expected JSON body.
**2. In plain words:** `PredictRequest` with `text: str` ensures the body looks right before your function runs; otherwise FastAPI returns 422 automatically.
**3. Example to quote:** > "Posting `{}` to `/predict` returns 422 because `text` is required."
**Seen in:** `mock_api.py`

## HTTPException with status 422 `(third-party: fastapi)`
**1. Definition:** `HTTPException(status_code=422, detail=...)` aborts the request and returns that HTTP status with a message; 422 means the request was understood but its content is invalid.
**2. In plain words:** Converting the model's `ValueError` into a 4xx tells the client the mistake is theirs, rather than crashing with a 500 server error.
**3. Example to quote:** > "Empty text gives a clean 422 with the model's message, not a 500."
**Seen in:** `mock_api.py`

## Import from sibling module `(syntax)`
**1. Definition:** `from model import predict as model_predict` imports a name from another file in the same folder, optionally renaming it with `as`.
**2. In plain words:** It works here because the API is launched from inside `common/`. Renaming avoids a clash with the endpoint function that is also called `predict`.
**3. Example to quote:** > "I import `predict as model_predict` so my endpoint can also be named `predict` without confusion."
**Seen in:** `mock_api.py`

## uvicorn launch from the folder `(command)`
**1. Definition:** `python -m uvicorn mock_api:app --port 8000`, run from `common/`, starts the ASGI server for the `app` object in `mock_api.py`.
**2. In plain words:** The `module:variable` string tells uvicorn what to serve, and running from inside the folder is what makes `from model import ...` resolve.
**3. Example to quote:** > "I `cd common` first and then run uvicorn, so the sibling import works."
**Seen in:** `mock_api.py` docstring

## Package marker `__init__.py` `(file)`
**1. Definition:** An (here empty) file that marks a directory as a Python package.
**2. In plain words:** Its presence lets the folder be treated as a package; being empty is perfectly normal.
**3. Example to quote:** > "`common/__init__.py` is empty; it just marks the folder as a package."
**Seen in:** `__init__.py`
