# Step 03 - Streamlit as an API client (loading and error states)

## The story so far
[Step 02](../02_streamlit_model_state_uploads/WALKTHROUGH.md) kept the model inside the UI. The real system has a separate API ([module 09](../../09_api_development/)), so the UI must become a client.

## The pain point
A UI that loads its own model duplicates it per user and cannot be scaled or updated alone. Once it calls an API instead, new things can fail: the API is down, slow, or rejects the input, and a raw traceback in the browser helps nobody.

## The spark
Make the UI a thin client: `requests.post` with a timeout, `st.spinner` while waiting, and one clear message per failure (`ConnectionError`, `Timeout`, HTTP 422). For long LLM answers the API can stream tokens over SSE ([09 step 10](../../09_api_development/10_streaming_sse/)); a UI would then show them as they arrive instead of after a long wait.

## Goal
Separate UI from model: the UI sends HTTP requests to the API from [module 09](../../09_api_development/) (or the mock) and handles every way that can go wrong.

## Run it
```bash
# terminal 1
cd 12_streamlit_gradio/common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000
# terminal 2
cd 12_streamlit_gradio/03_streamlit_api_client && ../../.venv/bin/streamlit run app.py
```
Verified in `../test_apps.py` against a real in-process uvicorn server: success shows a label; stopping the server shows the "Cannot reach the API" error.
Module-09 API format expected: `POST /predict` with `{"text": ...}` returning `{"label": ..., "score": ...}`. If yours differs, change the two keys in `app.py` (this is a good exercise).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| UI-with-model vs UI-as-client | "monolith", "thin client" | No | In-process model (step 02) vs HTTP call to a separate API (this step) |
| `requests` vs `httpx` | HTTP client | Almost | Same style of API; `httpx` also supports async and HTTP/2 |
| Timeout vs retry | network safety | No | Timeout limits waiting; retry repeats failed calls |
| `ConnectionError` vs `Timeout` vs `HTTPError` | requests exceptions | No | Cannot connect / no answer in time / server answered 4xx or 5xx (after `raise_for_status`) |
| Mock API vs real API | stub, fake | Same contract | Same endpoints and JSON shape, different implementation |

## Code, step by step

### 1. mock_api docstring  (`../common/mock_api.py` lines 1–10)
```python
"""
GOAL: A stand-in for the module-09 API so the UIs work standalone. Same contract: GET /health, POST /predict.

RUN: cd 12_streamlit_gradio/common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000
EXPECTED OUTPUT: curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"I love it"}'
                 -> {"label":"positive","score":0.548}
NEEDS: fastapi, uvicorn, scikit-learn.

If your module-09 API is running on port 8000 instead, stop this one: the UIs do not care which one answers.
"""
```
- **In simple words:** A stand-in for the module 09 API so the UI works on its own.
- **Exactly:** Same contract: `GET /health`, `POST /predict`; response for "I love it" is `{"label":"positive","score":0.548}`.
- **Why it is here:** The UI does not care which API answers.

### 2. Imports and app  (`../common/mock_api.py` lines 11–16)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import predict as model_predict   # works because we run from inside common/

app = FastAPI(title="Mock prediction API")
```
- **In simple words:** Load FastAPI and the shared model, create the app.
- **Exactly:** `from model import predict as model_predict` works because the command is run inside `common/` (module `mock_api` is imported by uvicorn from that folder).
- **Why it is here:** A different name avoids clashing with the route function named `predict`.

### 3. Request schema and health  (`../common/mock_api.py` lines 19–25)
```python
class PredictRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}
```
- **In simple words:** A form with a text field, and an "alive" route.
- **Exactly:** `PredictRequest(text: str)`; `/health` returns `{"status":"ok"}`.
- **Why it is here:** Needed by the UI's "Check /health" button.

### 4. Predict route with clean errors  (`../common/mock_api.py` lines 28–34)
```python
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        label, score = model_predict(req.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))   # clean 4xx instead of a 500 crash
    return {"label": label, "score": score}
```
- **In simple words:** Run the model and turn bad input into a clear 422 response.
- **Exactly:** `model_predict` raises `ValueError` for blank text; the route converts it to `HTTPException(status_code=422, detail=...)`. Without the conversion FastAPI returns a 500.
- **Why it is here:** A 4xx tells the client the request was wrong; 5xx would suggest the server broke.

### 5. Docstring  (`app.py` lines 1–12)
```python
"""
GOAL: A Streamlit UI that does NOT hold the model itself; it calls an HTTP API (module 09, or common/mock_api.py)
      and shows loading, success and several kinds of error states.

RUN:
  terminal 1 (API):  cd 12_streamlit_gradio/common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000
                     (or start your module-09 API on port 8000 instead)
  terminal 2 (UI):   cd 12_streamlit_gradio/03_streamlit_api_client && ../../.venv/bin/streamlit run app.py
EXPECTED OUTPUT: type a sentence, press Predict -> spinner -> label + confidence. Stop the API and press again ->
                 a red "Cannot reach the API" box (the UI itself stays alive).
NEEDS: streamlit, requests.
"""
```
- **In simple words:** Explains two terminals: one for the API, one for the UI.
- **Exactly:** The UI holds no model; it calls an HTTP API and shows loading, success and error states.
- **Why it is here:** This is the architecture used in the capstone (module 13).

### 6. Imports  (`app.py` lines 13–16)
```python
import os

import requests          # 3rd-party HTTP client (httpx is the modern alternative with almost the same API)
import streamlit as st
```
- **In simple words:** Load tools for environment, HTTP and Streamlit.
- **Exactly:** `requests` is third-party; `httpx` has nearly the same API.
- **Why it is here:** The UI needs a way to call the API.

### 7. Default API address  (`app.py` lines 18–19)
```python
# Where is the API? Environment variable first (Docker/Compose sets API_URL), else localhost.
DEFAULT_URL = os.getenv("API_URL", "http://localhost:8000")
```
- **In simple words:** Find the API address from the environment, or use localhost.
- **Exactly:** `os.getenv("API_URL", "http://localhost:8000")`; Docker Compose sets `API_URL=http://api:8000`.
- **Why it is here:** Same code works on a laptop and in containers.

### 8. Title and sidebar URL box  (`app.py` lines 21–22)
```python
st.title("Sentiment UI -> API")
api_url = st.sidebar.text_input("API base URL", DEFAULT_URL)      # sidebar = left panel
```
- **In simple words:** A heading and an editable address in the left panel.
- **Exactly:** `st.sidebar.text_input` returns the current string; changing it triggers a rerun.
- **Why it is here:** Lets you point the UI at a wrong port to test errors.

### 9. Health check button  (`app.py` lines 24–29)
```python
if st.sidebar.button("Check /health"):
    try:
        r = requests.get(f"{api_url}/health", timeout=3)          # timeout: NEVER call a network without one
        st.sidebar.success(f"API says: {r.json()}")
    except requests.RequestException as exc:                      # base class of all requests errors
        st.sidebar.error(f"API down: {type(exc).__name__}")
```
- **In simple words:** A button that asks the API "are you alive?".
- **Exactly:** `requests.get(url, timeout=3)`; on success `r.json()` is shown. `requests.RequestException` is the base class of all `requests` errors; `type(exc).__name__` prints e.g. `ConnectionError`.
- **Why it is here:** Gives a quick diagnosis without making a prediction.

### 10. Multi-line input  (`app.py` lines 31–31)
```python
text = st.text_area("Text to classify", "The service was excellent")   # multi-line input
```
- **In simple words:** A bigger text area for the sentence.
- **Exactly:** `st.text_area(label, value)` returns a `str`.
- **Why it is here:** Sentences can be long.

### 11. Validate before calling  (`app.py` lines 33–36)
```python
if st.button("Predict"):
    if not text.strip():
        st.warning("Please type something first.")                # validate BEFORE the network call
    else:
```
- **In simple words:** If the box is empty, ask for text and do not call the network.
- **Exactly:** `text.strip()` is empty for blank input; `st.warning` draws a yellow box.
- **Why it is here:** Saves a pointless API call and a confusing 422.

### 12. Call the API and show the result  (`app.py` lines 37–44)
```python
        try:
            with st.spinner("Calling the API..."):                # shows a spinner while the block runs
                resp = requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)
            resp.raise_for_status()                               # 4xx/5xx -> HTTPError
            data = resp.json()
            st.metric("Label", data["label"], f"score {data.get('confidence', data.get('score'))}")  # 09 API: 'confidence'; mock API: 'score'
            with st.expander("Raw response"):                     # collapsible section
                st.json(data)
```
- **In simple words:** Show a spinner, send the text, show label and score, and the raw JSON in a folded box.
- **Exactly:** `st.spinner` is a context manager; `requests.post(..., json=..., timeout=10)` sends JSON; `raise_for_status()` raises `HTTPError` for 4xx/5xx. `data.get('confidence', data.get('score'))` handles both the module 09 API (`confidence`) and the mock API (`score`). `st.expander` is collapsible.
- **Why it is here:** Loading feedback and compatibility with two API versions.

### 13. Four kinds of failure  (`app.py` lines 45–52)
```python
        except requests.ConnectionError:
            st.error(f"Cannot reach the API at {api_url}. Is it running?")
        except requests.Timeout:
            st.error("The API took too long to answer (timeout).")
        except requests.HTTPError as exc:
            st.error(f"API returned {exc.response.status_code}: {exc.response.text}")
        except (KeyError, ValueError):                            # missing key / invalid JSON
            st.error("The API answered, but not in the expected format.")
```
- **In simple words:** A friendly message for each way the call can go wrong.
- **Exactly:** `ConnectionError`: API not running. `Timeout`: too slow. `HTTPError`: answered 4xx/5xx (`exc.response.status_code`). `KeyError`/`ValueError`: response was not the expected JSON. More specific `except` clauses come first.
- **Why it is here:** The UI stays alive and tells the user what to do; a stack trace helps nobody.

### 14. Exercises as comments  (`app.py` lines 54–56)
```python
# TRY IT: change the URL in the sidebar to a wrong port and press Predict.
# TRY IT: add a time.sleep(3) inside mock_api.predict and watch the spinner.
# BREAK IT: change data["label"] to data["labell"] -> KeyError caught by our friendly message; remove the except and see the raw traceback.
```
- **In simple words:** Try a wrong port, add a delay, break a key.
- **Exactly:** TRY IT: wrong port in the sidebar; BREAK IT: `data["labell"]` gives a `KeyError`, caught as "not the expected format".
- **Why it is here:** See each error path once.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `try/except A / except B` | handle different failures differently | order: specific first |
| `with st.spinner("..."):` | spinner while block executes | |
| `os.getenv("API_URL", default)` | config from environment | |
| f-string with expression `f"{api_url}/health"` | build URLs | |
| `type(exc).__name__` | class name of the exception | `ConnectionError` |
| `except (KeyError, ValueError):` | catch several types at once | |

## Libraries & functions
| Name | Built-in / 3rd-party | Does | Important args | Returns |
|---|---|---|---|---|
| `requests.post` | 3rd-party | HTTP POST | `url`, `json=`, `timeout=` | `Response` |
| `Response.raise_for_status()` | requests | raise `HTTPError` on 4xx/5xx | | None |
| `Response.json()` | requests | parse body as JSON | | dict/list |
| `requests.ConnectionError/Timeout/HTTPError/RequestException` | requests | exception classes (RequestException = parent) | | |
| `st.sidebar.*` | streamlit | widgets in side panel | | |
| `st.text_area`, `st.expander`, `st.warning` | streamlit | multi-line input, collapsible, yellow box | | |

## Execution trace
Input text -> click Predict -> script reruns -> `requests.post(..., json={"text": text})` serialises to JSON, sends to `/predict` -> FastAPI validates, runs `model.predict`, returns `{"label", "score"}` -> `resp.json()` dict -> `st.metric`. Failure branches: API stopped -> `ConnectionError`; slow API -> `Timeout`; empty text rejected by API -> 422 -> `HTTPError`; renamed JSON key -> `KeyError`.

## Experiments
1. Kill the API, press Predict: red box, UI alive.
2. Put `localhost:9999` in the sidebar.
3. Add `time.sleep(15)` to the mock API (timeout 10 s) and see the timeout message.
4. Send `"   "` text bypassing the UI check (curl) to see the API's 422.

## Debugging practice
Always distinguish "UI bug" from "API bug": test the API alone with `curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"text":"hi"}'` before touching the UI. Inside Docker, `localhost` in the UI means the UI container; use the service name (see [module 11 step 04](../../11_docker/04_compose_multi_container/)).

## Rebuild checklist
- [ ] POST JSON and handle ConnectionError, Timeout, HTTPError.
- [ ] Explain why a timeout is mandatory.

## The new problem (cliffhanger)
The function works but the page is one long column that looks like a prototype. Next: [step 04, layout and style](../04_streamlit_layout_style/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** The UI is a thin client: it validates input, calls the prediction API with a timeout, shows a spinner, and maps each failure to a clear message. That keeps the model deployable and scalable separately from the UI. The trade-off is an extra network hop, which adds latency and a new failure mode.
