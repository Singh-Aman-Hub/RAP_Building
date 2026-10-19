# Step 03 - Streamlit as an API client (loading and error states)

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

## Say it aloud
"The UI is a thin client: it validates input, calls the prediction API with a timeout, shows a spinner, and maps each failure to a clear message. That keeps the model deployable and scalable separately from the UI. Trade-off: an extra network hop adds latency and a new failure mode."
