# Terms — Streamlit UI that calls an HTTP API

Folder: `12_streamlit_gradio/03_streamlit_api_client` · Files covered: `app.py`, `WALKTHROUGH.md`
Back to module: [12_streamlit_gradio README](../README.md)

**Terms in this folder:** [HTTP API client (concept)](#http-api-client-concept) · [requests (third-party: requests)](#requests-third-party-requests) · [requests.post(url, json=, timeout=) (third-party: requests)](#requestsposturl-json-timeout-third-party-requests) · [requests.get (third-party: requests)](#requestsget-third-party-requests) · [Response.raise_for_status (third-party: requests)](#responseraise_for_status-third-party-requests) · [Response.json (third-party: requests)](#responsejson-third-party-requests) · [requests exceptions (ConnectionError, Timeout, HTTPError, RequestException) (third-party: requests)](#requests-exceptions-connectionerror-timeout-httperror-requestexception-third-party-requests) · [try / except with several except clauses (syntax)](#try--except-with-several-except-clauses-syntax) · [except (KeyError, ValueError) (syntax)](#except-keyerror-valueerror-syntax) · [type(exc).__name__ (Python built-in)](#typeexc__name__-python-built-in) · [os.getenv (Python built-in module function)](#osgetenv-python-built-in-module-function) · [st.sidebar (third-party: streamlit)](#stsidebar-third-party-streamlit) · [st.text_area (third-party: streamlit)](#sttext_area-third-party-streamlit) · [str.strip and input validation (Python built-in)](#strstrip-and-input-validation-python-built-in) · [st.warning (third-party: streamlit)](#stwarning-third-party-streamlit) · [st.spinner (third-party: streamlit)](#stspinner-third-party-streamlit) · [st.metric (third-party: streamlit)](#stmetric-third-party-streamlit) · [dict.get with a fallback (Python built-in)](#dictget-with-a-fallback-python-built-in) · [st.expander and st.json (third-party: streamlit)](#stexpander-and-stjson-third-party-streamlit) · [Error-state design (concept)](#error-state-design-concept) · [Request timeout (concept)](#request-timeout-concept) · [API base URL configuration (concept)](#api-base-url-configuration-concept)

---

## HTTP API client `(concept)`
**1. Definition:** A program that talks to a remote service by sending HTTP requests and interpreting the responses, instead of doing the work itself.
**2. In plain words:** This UI holds no model. It sends the text to the API and shows the answer. That separation means the same model service can be used by many different front ends, and the UI stays small.
**3. Example to quote:** > "My Streamlit app is only a client: it POSTs the sentence to `/predict` and displays the JSON it gets back."
**Seen in:** `app.py`

## requests `(third-party: requests)`
**1. Definition:** A widely used Python library for making HTTP requests with simple functions like `requests.get` and `requests.post`.
**2. In plain words:** It takes care of connections, encoding JSON and decoding responses. `httpx` is a modern alternative with almost the same API.
**3. Example to quote:** > "I used `requests.post` to call my prediction API from the UI."
**Seen in:** `app.py`

## requests.post(url, json=, timeout=) `(third-party: requests)`
**1. Definition:** Sends an HTTP POST; `json=` serialises a dict as the request body with the right content type, and `timeout=` is the maximum seconds to wait; returns a `Response`.
**2. In plain words:** Never call a network without a timeout: if the server hangs, so would your UI. The `json=` argument saves you from encoding the body yourself.
**3. Example to quote:** > "`requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)` waits at most ten seconds."
```python
resp = requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)
```
**Seen in:** `app.py`

## requests.get `(third-party: requests)`
**1. Definition:** Sends an HTTP GET request and returns a `Response`.
**2. In plain words:** Used for the health check button. GET is for reading information without changing anything.
**3. Example to quote:** > "The sidebar button calls `requests.get(f"{api_url}/health", timeout=3)` and shows what the API says."
**Seen in:** `app.py`

## Response.raise_for_status `(third-party: requests)`
**1. Definition:** Raises `requests.HTTPError` if the response status code is 4xx or 5xx; does nothing for success codes.
**2. In plain words:** Without it, an error response looks like a normal answer and the next line fails confusingly. This converts bad statuses into exceptions you can handle.
**3. Example to quote:** > "A 422 from the API becomes an `HTTPError`, which I catch and show as 'API returned 422'."
**Seen in:** `app.py`

## Response.json `(third-party: requests)`
**1. Definition:** Parses the response body as JSON and returns a dict or list; raises `ValueError` if the body isn't valid JSON.
**2. In plain words:** It turns the API's text reply into a Python dictionary you can index. If the API sends something that isn't JSON, you get an error you should handle.
**3. Example to quote:** > "`data = resp.json()` gives me `{'label': 'positive', 'confidence': 0.548}`."
**Seen in:** `app.py`

## requests exceptions (ConnectionError, Timeout, HTTPError, RequestException) `(third-party: requests)`
**1. Definition:** `ConnectionError` means the server couldn't be reached, `Timeout` means no answer in time, `HTTPError` means a 4xx/5xx status after `raise_for_status`, and `RequestException` is the parent class of all of them.
**2. In plain words:** Each kind of failure deserves its own friendly message. Catch the specific ones first and the general parent last.
**3. Example to quote:** > "If the API is stopped, I see 'Cannot reach the API' from `ConnectionError`; if it's slow, 'took too long' from `Timeout`."
**Seen in:** `app.py`

## try / except with several except clauses `(syntax)`
**1. Definition:** A `try` block followed by multiple `except` clauses, each handling a different exception type; the first matching clause runs.
**2. In plain words:** Order matters: put specific exceptions before general ones, or the general one will swallow everything. You can also catch several types in one clause with a tuple.
**3. Example to quote:** > "I handle `ConnectionError`, `Timeout`, `HTTPError` and `(KeyError, ValueError)` separately, each with its own message."
**Seen in:** `app.py`

## except (KeyError, ValueError) `(syntax)`
**1. Definition:** Catches either of the listed exception types in one clause.
**2. In plain words:** A `KeyError` means the JSON lacks an expected key and a `ValueError` means the body wasn't valid JSON; both mean "the API answered, but not in the expected format".
**3. Example to quote:** > "Changing `data["label"]` to `data["labell"]` triggered the KeyError branch and the friendly 'unexpected format' message."
**Seen in:** `app.py`

## type(exc).__name__ `(Python built-in)`
**1. Definition:** `type(obj)` returns an object's class and `__name__` is that class's name as a string.
**2. In plain words:** It gives a short readable label for an exception, such as `ConnectionError`, without printing a long message.
**3. Example to quote:** > "The sidebar shows `API down: ConnectionError`, built from `type(exc).__name__`."
**Seen in:** `app.py`

## os.getenv `(Python built-in module function)`
**1. Definition:** Returns the value of an environment variable or a default if it isn't set.
**2. In plain words:** The API address comes from `API_URL` if present, so Docker Compose can set it to `http://api:8000`, and otherwise defaults to localhost. The same code runs in both places.
**3. Example to quote:** > "`os.getenv("API_URL", "http://localhost:8000")` lets Compose override where the UI looks for the API."
**Seen in:** `app.py`

## st.sidebar `(third-party: streamlit)`
**1. Definition:** A left-hand panel; widgets are placed there with `st.sidebar.<widget>(...)` or `with st.sidebar:`.
**2. In plain words:** It's a good place for settings such as the API URL and a health-check button, keeping the main area for the task itself.
**3. Example to quote:** > "I put the API URL box in `st.sidebar.text_input` so users can point the UI at another server."
**Seen in:** `app.py`

## st.text_area `(third-party: streamlit)`
**1. Definition:** A multi-line text input that returns the entered string.
**2. In plain words:** Use it when users may type more than a short phrase, such as a paragraph to classify.
**3. Example to quote:** > "`text = st.text_area("Text to classify", ...)` holds whatever the user typed."
**Seen in:** `app.py`

## str.strip and input validation `(Python built-in)`
**1. Definition:** `strip()` removes leading and trailing whitespace; `if not text.strip():` is True for empty or whitespace-only text.
**2. In plain words:** Check the input before making a network call. It's faster, cheaper and gives a clearer message than letting the API reject it.
**3. Example to quote:** > "If the box is blank I show a yellow warning and never call the API."
**Seen in:** `app.py`

## st.warning `(third-party: streamlit)`
**1. Definition:** Displays a yellow warning message box.
**2. In plain words:** For problems the user can fix themselves, such as empty input, as opposed to `st.error` for failures.
**3. Example to quote:** > "`st.warning("Please type something first.")` tells the user what to do."
**Seen in:** `app.py`

## st.spinner `(third-party: streamlit)`
**1. Definition:** A context manager that shows a spinner with a message while the indented block runs.
**2. In plain words:** It's the loading state. Without it, the UI looks frozen during a slow API call.
**3. Example to quote:** > "I wrap the request in `with st.spinner("Calling the API...")` so the user sees something is happening."
**Seen in:** `app.py`

## st.metric `(third-party: streamlit)`
**1. Definition:** Displays a labelled value with an optional secondary text.
**2. In plain words:** It shows the predicted label large, with the confidence beneath it.
**3. Example to quote:** > "`st.metric("Label", data["label"], f"score ...")` shows 'positive' with its score."
**Seen in:** `app.py`

## dict.get with a fallback `(Python built-in)`
**1. Definition:** `d.get(key, default)` returns the value for `key` or `default` if it's missing, without raising KeyError.
**2. In plain words:** The module 09 API names the field `confidence` while the mock API calls it `score`. `data.get('confidence', data.get('score'))` tolerates both.
**3. Example to quote:** > "I use `.get` with a fallback so the same UI works with either API's field name."
```python
data.get("confidence", data.get("score"))
```
**Seen in:** `app.py`

## st.expander and st.json `(third-party: streamlit)`
**1. Definition:** `st.expander(label)` creates a collapsible section; `st.json(obj)` displays an object as formatted JSON.
**2. In plain words:** The raw API response is useful for debugging but noisy, so it's hidden in an expander until needed.
**3. Example to quote:** > "I hide the raw response in `with st.expander("Raw response"): st.json(data)`."
**Seen in:** `app.py`

## Error-state design `(concept)`
**1. Definition:** Handling each distinct failure with a specific, human-readable message so the UI stays alive and tells the user what to do.
**2. In plain words:** "Cannot reach API", "timed out", "API returned 422" and "unexpected format" each suggest a different fix. A raw traceback helps nobody.
**3. Example to quote:** > "When I stopped the API, the UI showed a red 'Cannot reach the API' box and kept working."
**Seen in:** `app.py`

## Request timeout `(concept)`
**1. Definition:** A maximum duration a client waits for a response before giving up with an error.
**2. In plain words:** Without one, a stuck server can hang your UI indefinitely. Health checks get short timeouts (3 s) and predictions slightly longer (10 s).
**3. Example to quote:** > "I add a time.sleep(3) in the mock API and watch the spinner, then lower the timeout to see the Timeout message."
**Seen in:** `app.py`

## API base URL configuration `(concept)`
**1. Definition:** Keeping the server address as a configurable value rather than hard-coding it in the request code.
**2. In plain words:** It lets you point the same UI at a local mock, the module 09 API or a container service by changing one setting.
**3. Example to quote:** > "I changed the sidebar URL to a wrong port, pressed Predict and saw the connection-error message."
**Seen in:** `app.py`
