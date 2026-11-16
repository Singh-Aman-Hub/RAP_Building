# Step 02 - Model loading, forms, session state and file upload

## The story so far
[Step 01](../01_streamlit_basics/WALKTHROUGH.md) showed that Streamlit reruns the script on every interaction. Now we put a real model behind the button.

## The pain point
Load the model at the top of the script and it reloads on every click, so each prediction costs seconds. A history list resets on every rerun, so the user never sees past results.

## The spark
Cache the model with `st.cache_resource` so it loads once per server, and keep per-user data in `st.session_state`. Use a form to batch inputs so typing does not trigger a rerun, and `st.file_uploader` for batch files.

## Goal
Show model predictions in the UI without reloading the model each rerun, remember a history, and batch-predict an uploaded file.

## Run it
```bash
cd 12_streamlit_gradio/02_streamlit_model_state_uploads
../../.venv/bin/streamlit run app.py        # upload sample_sentences.txt to try the batch part
```
Verified headlessly with `AppTest` (see ../test_apps.py): predicting "I love this" adds one history row and the model loader ran once.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `st.cache_resource` vs `st.cache_data` | `@st.cache` (removed), memoisation | No | `cache_resource` stores one shared object (model, DB connection) without copying; `cache_data` stores serialisable return values and returns a copy per call |
| `st.session_state` vs cache | memory across reruns | No | `session_state` is private to one browser session; caches are shared by all users |
| `st.form` vs plain widgets | batch submit | No | Inside a form nothing reruns until the submit button; outside, every widget change reruns |
| TF-IDF + logistic regression vs "the model" | pipeline, classifier | Same object here | `make_pipeline(TfidfVectorizer(), LogisticRegression())` is one object with `.fit` and `.predict_proba` |
| `predict` vs `predict_proba` | label vs probability | No | `predict` returns the class; `predict_proba` returns one probability per class |

## Code, step by step

### 1. common/model.py docstring  (`../common/model.py` lines 1–10)
```python
"""
GOAL: A tiny REAL ML model (scikit-learn TF-IDF + logistic regression) that the UIs and the mock API share.

RUN: cd 12_streamlit_gradio/common && ../../.venv/bin/python model.py
EXPECTED OUTPUT: three predictions, e.g. I love it -> ('positive', 0.548)
NEEDS: scikit-learn. Trains in well under a second; no download.

Why train at import/first-use? Keeps the folder tiny (no .pkl file) and shows the pattern "load model ONCE,
reuse for every request" - which is what st.cache_resource does for Streamlit later.
"""
```
- **In simple words:** This small shared file holds the real model used by every UI in this module.
- **Exactly:** It trains a scikit-learn TF-IDF + logistic regression in under a second; no download. It trains on first use and is reused (the same idea as `st.cache_resource`).
- **Why it is here:** A tiny real model avoids a large download and keeps the focus on the UI.

### 2. Imports from scikit-learn  (`../common/model.py` lines 11–13)
```python
from sklearn.feature_extraction.text import TfidfVectorizer   # text -> numeric feature vectors
from sklearn.linear_model import LogisticRegression           # simple classifier with probabilities
from sklearn.pipeline import make_pipeline                    # chains vectorizer + classifier into one object
```
- **In simple words:** Pick the three tools: turn text into numbers, classify, chain them.
- **Exactly:** `TfidfVectorizer` maps text to a sparse TF-IDF vector; `LogisticRegression` learns linear weights and gives probabilities; `make_pipeline` chains them into one estimator.
- **Why it is here:** One object means you cannot forget to vectorise before classifying.

### 3. The training data  (`../common/model.py` lines 15–22)
```python
TRAIN = [
    ("I love this, it is great", "positive"), ("what an excellent product", "positive"),
    ("so happy with the result", "positive"), ("really good quality and fast", "positive"),
    ("absolutely wonderful experience", "positive"), ("I like it a lot", "positive"),
    ("this is terrible", "negative"), ("awful, I hate it", "negative"),
    ("very bad and slow", "negative"), ("so sad and disappointing", "negative"),
    ("worst purchase ever", "negative"), ("I dislike it, poor quality", "negative"),
]
```
- **In simple words:** Twelve short sentences, six happy and six sad.
- **Exactly:** List of `(text, label)` tuples; labels are `"positive"` or `"negative"`.
- **Why it is here:** Enough to learn the obvious words, small enough to train instantly. Expect low confidence (around 0.55) because it is tiny.

### 4. Train once, cache in a module variable  (`../common/model.py` lines 24–34)
```python
_model = None   # module-level cache: None until first use


def load_model():
    """Train (once) and return the pipeline."""
    global _model
    if _model is None:
        texts, labels = zip(*TRAIN)                       # unzip list of pairs into two tuples
        _model = make_pipeline(TfidfVectorizer(), LogisticRegression())
        _model.fit(texts, labels)                         # learn word weights
    return _model
```
- **In simple words:** Train the model the first time it is asked for, then hand back the same one.
- **Exactly:** `_model` starts `None`. `zip(*TRAIN)` unzips pairs into `texts` and `labels` (two tuples). `fit` learns the vocabulary (37 features on this data: shape `(1, 37)` for one sentence) and weights. `global _model` lets the function assign the module variable.
- **Why it is here:** Without caching, every click retrains.

### 5. Predict one sentence  (`../common/model.py` lines 37–44)
```python
def predict(text: str) -> tuple[str, float]:
    """Return (label, confidence) for one sentence. Raises ValueError on empty input."""
    if not text or not text.strip():
        raise ValueError("Text must not be empty")
    model = load_model()
    proba = model.predict_proba([text])[0]                # one row of class probabilities
    best = proba.argmax()                                 # index of the largest probability
    return str(model.classes_[best]), round(float(proba[best]), 3)
```
- **In simple words:** Give a sentence, get a label and how sure the model is.
- **Exactly:** Empty or blank text raises `ValueError`. `predict_proba([text])[0]` returns an array of two probabilities ordered by `classes_` (`['negative','positive']`). `argmax()` picks the index; confidence rounded to 3 digits. Real output: `I love it -> ('positive', 0.548)`.
- **Why it is here:** A clean error type lets UIs show a friendly message instead of crashing.

### 6. Demo run  (`../common/model.py` lines 47–49)
```python
if __name__ == "__main__":
    for s in ["I love it", "this is awful", "it is a table"]:
        print(s, "->", predict(s))
```
- **In simple words:** When you run the file directly, it prints three predictions.
- **Exactly:** Real output: `I love it -> ('positive', 0.548)`, `this is awful -> ('negative', 0.552)`, `it is a table -> ('negative', 0.508)`. The last one is nearly 50/50, the model has no signal for "table".
- **Why it is here:** Shows what low confidence looks like and gives you a quick self-test.

### 7. Docstring  (`app.py` lines 1–9)
```python
"""
GOAL: Load an ML model once (cache), predict inside a form, keep history across reruns (session state),
      and classify every line of an uploaded .txt file.

RUN (from 12_streamlit_gradio/02_streamlit_model_state_uploads):  ../../.venv/bin/streamlit run app.py
EXPECTED OUTPUT: a form with a text box + "Predict"; a prediction box; a history list that grows;
                 a file uploader that shows a table of predictions for each uploaded line.
NEEDS: streamlit, scikit-learn (model comes from ../common/model.py).
"""
```
- **In simple words:** What the app does: cache the model, form, history, upload.
- **Exactly:** Lists the four ideas: `cache_resource`, form, `session_state`, file upload.
- **Why it is here:** Quick map of the file.

### 8. Imports  (`app.py` lines 10–13)
```python
import sys
from pathlib import Path

import streamlit as st
```
- **In simple words:** Load tools for paths and Streamlit.
- **Exactly:** `sys`, `pathlib.Path` (standard library) and `streamlit`.
- **Why it is here:** `sys.path` is edited next.

### 9. Make the shared model importable  (`app.py` lines 15–17)
```python
# Make ../common importable no matter where `streamlit run` is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import load_model, predict  # noqa: E402  (import after sys.path change on purpose)
```
- **In simple words:** Tell Python where the `common` folder is.
- **Exactly:** `Path(__file__).resolve().parents[1] / "common"` goes from `app.py` up one folder to `12_streamlit_gradio/` and down into `common/`. `sys.path.insert(0, ...)` puts it first on the import search path. `# noqa: E402` silences the lint rule "import not at top".
- **Why it is here:** Without it `from model import ...` raises `ModuleNotFoundError` when you launch from a different folder.

### 10. Page title  (`app.py` lines 19–19)
```python
st.title("Sentiment classifier")
```
- **In simple words:** Heading for the page.
- **Exactly:** `st.title` draws the H1.
- **Why it is here:** Orientation for the user.

### 11. Load the model once with a cache  (`app.py` lines 22–30)
```python
# @st.cache_resource: run this function ONCE per server process and reuse the returned object on every
# rerun and for every user. Use it for models, DB connections. (st.cache_data is for plain data/DataFrames.)
@st.cache_resource
def get_model():
    st.write("Loading model...")      # shown only the first time; proves caching works
    return load_model()


get_model()
```
- **In simple words:** The first visitor waits for training, everyone after reuses the model.
- **Exactly:** `@st.cache_resource` memoises the function: first call runs `load_model()` and stores the object per server process; later reruns and other sessions get the same object. The `st.write("Loading model...")` text shows only on the first call. `get_model()` is called once at the top to warm the cache.
- **Why it is here:** Without it you see "Loading model..." on every interaction, the app feels slow with large models.

### 12. History in session state  (`app.py` lines 32–34)
```python
# st.session_state is a dict-like object that SURVIVES reruns (one per browser tab/session).
if "history" not in st.session_state:          # initialise once
    st.session_state.history = []              # attribute style == st.session_state["history"]
```
- **In simple words:** Make a notebook that survives reruns, one per visitor.
- **Exactly:** `st.session_state` is a dict-like store per browser session. `if "history" not in ...` initialises it once. Attribute and key access are equal.
- **Why it is here:** Without the guard, reruns would reset it, and removing the block gives `AttributeError` on first use (BREAK IT).

### 13. Form with a submit button  (`app.py` lines 36–39)
```python
# st.form batches several widgets: nothing reruns until the submit button is pressed.
with st.form("predict_form"):                  # `with` = context manager: widgets inside belong to the form
    text = st.text_input("Sentence", "I really enjoyed this")
    submitted = st.form_submit_button("Predict")
```
- **In simple words:** A small form: nothing happens until you press Predict.
- **Exactly:** `with st.form("predict_form")` groups widgets; their values are sent to Python only when `form_submit_button` is pressed, which returns `True` on that rerun.
- **Why it is here:** Without a form the app reruns on every text change and the model is called repeatedly.

### 14. Predict and handle bad input  (`app.py` lines 41–47)
```python
if submitted:
    try:
        label, score = predict(text)           # may raise ValueError for empty text
        st.metric("Prediction", label, f"confidence {score}")   # big number-style display
        st.session_state.history.append((text, label, score))
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")      # friendly error instead of a crash page
```
- **In simple words:** When submitted, classify the text and show the answer or an error.
- **Exactly:** `predict(text)` returns `(label, score)` or raises `ValueError` for blank text. `st.metric(label, value, delta)` is a big-number tile. The tuple `(text, label, score)` is appended to history.
- **Why it is here:** The `try/except` turns an exception into a friendly red box, the app stays alive.

### 15. Show and clear the history  (`app.py` lines 49–54)
```python
if st.session_state.history:
    st.subheader("History (session_state)")
    st.table(st.session_state.history)         # list of tuples -> static table
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()                             # redraw immediately with the empty list
```
- **In simple words:** Show a table of everything predicted so far and a button to empty it.
- **Exactly:** `st.table` renders a static table from a list of tuples. A list assigned to `st.session_state.history = []` then `st.rerun()` triggers an immediate rerun so the empty state is drawn.
- **Why it is here:** Without `st.rerun()` the old table is displayed for one extra interaction.

### 16. Upload a text file and classify each line  (`app.py` lines 56–62)
```python
st.divider()
st.subheader("Batch: upload a .txt file (one sentence per line)")
uploaded = st.file_uploader("Choose a text file", type=["txt"])   # returns None or an UploadedFile
if uploaded is not None:
    lines = uploaded.getvalue().decode("utf-8").splitlines()      # bytes -> str -> list of lines
    rows = [{"text": ln, **dict(zip(("label", "score"), predict(ln)))} for ln in lines if ln.strip()]
    st.dataframe(rows)                          # interactive table
```
- **In simple words:** Upload a text file and get one prediction per line.
- **Exactly:** `st.file_uploader(..., type=["txt"])` returns `None` or an `UploadedFile`. `getvalue()` gives `bytes`; `.decode("utf-8").splitlines()` gives lines. The comprehension skips blank lines and builds dict rows `{"text", "label", "score"}` by `dict(zip(("label","score"), predict(ln)))`. `st.dataframe` shows an interactive table.
- **Why it is here:** Shows that Streamlit handles files in memory with no server code.

### 17. Exercises as comments  (`app.py` lines 64–66)
```python
# TRY IT: remove @st.cache_resource and watch "Loading model..." appear on every interaction.
# TRY IT: replace the form with a plain text_input + button and note the extra reruns while typing.
# BREAK IT: delete the `if "history" not in ...` block -> AttributeError on first run. Read the message.
```
- **In simple words:** Ideas to change the app.
- **Exactly:** Remove the cache decorator to see repeated loading; replace the form with a plain input to see more reruns; delete the history init for an `AttributeError`.
- **Why it is here:** Experiments that make each concept visible.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `@st.cache_resource` | decorator: cache the function's return value | `@st.cache_resource def get(): ...` |
| `with st.form("k"):` | context manager grouping widgets | submit via `st.form_submit_button` |
| `st.session_state.history` | persistent per-session storage | `st.session_state["x"] = 1` |
| `sys.path.insert(0, p)` | make folder `p` importable | |
| `Path(__file__).resolve().parents[1]` | parent folder of this file's folder | |
| list comprehension `[... for ln in lines if ln.strip()]` | build a list, skipping blank lines | `[x*2 for x in [1,2] if x>1]` |
| `{**a, **b}` / `dict(zip(keys, vals))` | merge dicts / pair up keys and values | `dict(zip("ab",(1,2)))` -> `{'a':1,'b':2}` |
| `try/except ValueError` | catch bad input | |

## Libraries & functions
| Name | Built-in / 3rd-party | Does | Important args | Returns |
|---|---|---|---|---|
| `st.cache_resource` | streamlit | cache objects shared across reruns/users | `ttl`, `show_spinner` | decorator |
| `st.form`, `st.form_submit_button` | streamlit | batch widget input | form key | bool for submit |
| `st.metric` | streamlit | KPI-style display | `label, value, delta` | None |
| `st.file_uploader` | streamlit | file widget | `type=[...]`, `accept_multiple_files` | `UploadedFile` or None |
| `UploadedFile.getvalue()` | streamlit | raw bytes | none | bytes |
| `bytes.decode("utf-8")` / `str.splitlines()` | built-in | bytes to text, text to lines | | str / list |
| `st.rerun()` | streamlit | restart script now | | never returns |
| `predict()` | ours (common/model.py) | (label, confidence) | `text` | tuple |

## Execution trace
Input "I really enjoyed this" -> submit -> script reruns; `get_model()` returns the cached pipeline instantly; `predict` vectorises the text, `predict_proba` gives class probabilities, best one chosen -> `st.metric` shows label; the tuple is appended to `session_state.history`; next rerun the table still has it. Upload: bytes -> decode -> lines -> one prediction per non-blank line -> table.

## Experiments
1. Predict twice with different text: history has 2 rows.
2. Submit empty text: red error box, no crash, no new row.
3. Upload a file with a blank line; confirm it is skipped.
4. Restart the Streamlit server: history is gone (session_state lives in memory per session).

## Debugging practice
Remove `.strip()` check in `predict` usage and upload a file with an empty line -> ValueError traceback in the browser pointing to `common/model.py`. Practise reading which file/line raised it.

## Rebuild checklist
- [ ] Cache a model with cache_resource.
- [ ] Add a counter in session_state with a button.
- [ ] Explain cache_resource vs cache_data vs session_state.

## The new problem (cliffhanger)
The model lives inside the UI process. In production the model lives in the API (module 09, in a container), and the UI should only call it. Next: [step 03, Streamlit as an API client](../03_streamlit_api_client/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** Because Streamlit reruns the script, I cache the model with cache_resource so it loads once, and I keep per-user data like history in session_state. Forms avoid a rerun on every keystroke, and the file uploader lets users run batches. The trade-off is that cache_resource objects are shared across users, so they must be read-only or thread-safe, and session_state is lost when the server restarts.
