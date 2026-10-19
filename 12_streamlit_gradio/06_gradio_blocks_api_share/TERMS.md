# Terms — Gradio Blocks, events, API calls and sharing

Folder: `12_streamlit_gradio/06_gradio_blocks_api_share` · Files covered: `app.py`, `WALKTHROUGH.md` (uses `../common/model.py`)
Back to module: [12_streamlit_gradio README](../README.md)

**Terms in this folder:** [gr.Blocks (third-party: gradio)](#grblocks-third-party-gradio) · [Layout containers (Row, Column, Tab) (third-party: gradio)](#layout-containers-row-column-tab-third-party-gradio) · [Event listener (.click, .upload) (third-party: gradio)](#event-listener-click-upload-third-party-gradio) · [gr.State (third-party: gradio)](#grstate-third-party-gradio) · [Immutable state updates (concept)](#immutable-state-updates-concept) · [List concatenation (Python built-in)](#list-concatenation-python-built-in) · [Star unpacking [ln, *predict(ln)] (syntax)](#star-unpacking-ln-predictln-syntax) · [gr.Progress (third-party: gradio)](#grprogress-third-party-gradio) · [gr.Error with exception conversion (third-party: gradio)](#grerror-with-exception-conversion-third-party-gradio) · [requests.post and HTTP errors (third-party: requests)](#requestspost-and-http-errors-third-party-requests) · [os.getenv for API_URL (Python built-in module function)](#osgetenv-for-api_url-python-built-in-module-function) · [gr.File (third-party: gradio)](#grfile-third-party-gradio) · [Path.read_text (standard library)](#pathread_text-standard-library) · [gr.Dataframe (third-party: gradio)](#grdataframe-third-party-gradio) · [gr.Markdown, gr.Button, gr.Textbox (third-party: gradio)](#grmarkdown-grbutton-grtextbox-third-party-gradio) · [Callback return values must match outputs (concept)](#callback-return-values-must-match-outputs-concept) · [share=True (third-party: gradio)](#sharetrue-third-party-gradio) · [theme in launch() (third-party: gradio)](#theme-in-launch-third-party-gradio) · [demo.queue (third-party: gradio)](#demoqueue-third-party-gradio) · [Default argument dependency injection (syntax)](#default-argument-dependency-injection-syntax) · [Dict output for gr.Label (concept)](#dict-output-for-grlabel-concept)

---

## gr.Blocks `(third-party: gradio)`
**1. Definition:** Gradio's low-level app builder, used as a context manager inside which you place components and wire events yourself.
**2. In plain words:** `gr.Interface` wraps one function. `gr.Blocks` lets you lay out rows, columns and tabs and connect several buttons to several functions, so it suits real apps.
**3. Example to quote:** > "I used `with gr.Blocks(title="Sentiment lab (Gradio)") as demo:` to build tabs for local, API and batch prediction."
**Seen in:** `app.py`

## Layout containers (Row, Column, Tab) `(third-party: gradio)`
**1. Definition:** `gr.Row` places children side by side, `gr.Column` stacks them (with an optional `scale`), and `gr.Tab` creates a tabbed section.
**2. In plain words:** They arrange the page. `scale=2` and `scale=1` make one column twice as wide as the other.
**3. Example to quote:** > "The input is in a column of scale 2 and the result in a column of scale 1, inside one row."
**Seen in:** `app.py`

## Event listener (.click, .upload) `(third-party: gradio)`
**1. Definition:** Methods on components, such as `btn.click(fn, inputs=[...], outputs=[...])` and `f.upload(...)`, that register a function to run when the event occurs.
**2. In plain words:** It's the wiring: "when this button is clicked, call this function with these inputs and put the returned values into these outputs". The number of returned values must match the number of outputs.
**3. Example to quote:** > "`btn.click(local_predict, inputs=[txt, history_state], outputs=[out, history_state, table])` runs when Predict is pressed."
**Seen in:** `app.py`

## gr.State `(third-party: gradio)`
**1. Definition:** An invisible component that stores a Python value separately for each browser session.
**2. In plain words:** It's Gradio's session memory, like `st.session_state` in Streamlit. A function receives the old value as an input and returns the new one as an output.
**3. Example to quote:** > "The prediction history lives in `gr.State([])`, so each visitor gets their own list."
**Seen in:** `app.py`

## Immutable state updates `(concept)`
**1. Definition:** Creating a new list (`history + [row]`) instead of mutating the existing one in place when updating state.
**2. In plain words:** Concatenating makes a fresh list, which avoids subtle bugs from shared mutable objects. Here it keeps each session's history independent.
**3. Example to quote:** > "I wrote `history = history + [[text, label, score]]` rather than `history.append(...)`."
**Seen in:** `app.py`

## List concatenation `(Python built-in)`
**1. Definition:** The `+` operator on lists produces a new list containing the elements of both.
**2. In plain words:** `[1] + [2]` gives `[1, 2]` without changing either original list.
**3. Example to quote:** > "`history + [[text, label, score]]` returns a new, longer list."
**Seen in:** `app.py`

## Star unpacking `[ln, *predict(ln)]` `(syntax)`
**1. Definition:** The `*` operator inside a list display expands an iterable into separate elements.
**2. In plain words:** `predict(ln)` returns a tuple `(label, score)`; the star spreads it so the row becomes `[line, label, score]`.
**3. Example to quote:** > "`[1, *(2, 3)]` gives `[1, 2, 3]`, which is how each batch row is built."
**Seen in:** `app.py`

## gr.Progress `(third-party: gradio)`
**1. Definition:** A helper that Gradio injects into a callback when it appears as a default argument; calling `progress(fraction, desc=...)` updates a progress bar.
**2. In plain words:** It's the loading state for slow work. You never pass it yourself; the default `progress=gr.Progress()` is enough for Gradio to detect and supply it.
**3. Example to quote:** > "In `api_predict` I call `progress(0.2, desc="Calling API...")` before the request and `progress(1.0, desc="Done")` after."
**Seen in:** `app.py`

## gr.Error with exception conversion `(third-party: gradio)`
**1. Definition:** Catching a technical exception and re-raising it as `gr.Error` with a friendly message.
**2. In plain words:** Connection errors and HTTP errors are translated into messages the user can act on, like "Cannot reach the API at ... Is it running?".
**3. Example to quote:** > "When the API is down, the user sees a toast saying so, not a stack trace."
**Seen in:** `app.py`

## requests.post and HTTP errors `(third-party: requests)`
**1. Definition:** `requests.post(url, json=..., timeout=...)` sends a POST; `raise_for_status()` raises `HTTPError` on 4xx/5xx and `ConnectionError` is raised when the server can't be reached.
**2. In plain words:** The "Via API" tab calls the HTTP service instead of the local model, proving the UI can use the same API as module 09.
**3. Example to quote:** > "The API tab posts the text with a ten-second timeout and shows the returned label as a Label component."
**Seen in:** `app.py`

## os.getenv for API_URL `(Python built-in module function)`
**1. Definition:** Reads the `API_URL` environment variable, defaulting to `http://localhost:8000`.
**2. In plain words:** It makes the API address configurable, for example set by Docker Compose.
**3. Example to quote:** > "`API_URL = os.getenv("API_URL", "http://localhost:8000")` is the default for the API tab's textbox."
**Seen in:** `app.py`

## gr.File `(third-party: gradio)`
**1. Definition:** An upload component; with `type="filepath"` the callback receives the file's path as a string (or `None`), and `file_types` restricts extensions.
**2. In plain words:** The uploaded file is saved on the server and your function gets its path. Always handle `None` for "nothing uploaded yet".
**3. Example to quote:** > "`gr.File(type="filepath", file_types=[".txt"])` gives `batch_predict` a path to read."
**Seen in:** `app.py`

## Path.read_text `(standard library)`
**1. Definition:** Reads an entire file and returns its contents as a string; accepts `encoding`.
**2. In plain words:** It's the short way to read a text file without opening and closing it yourself.
**3. Example to quote:** > "`Path(file_obj).read_text(encoding="utf-8").splitlines()` gives the list of sentences."
**Seen in:** `app.py`

## gr.Dataframe `(third-party: gradio)`
**1. Definition:** A table component for displaying rows, with optional `headers`.
**2. In plain words:** The function returns a list of rows and Gradio shows them as a table, used here for history and batch results.
**3. Example to quote:** > "`gr.Dataframe(headers=["text", "label", "score"])` shows my results as a table."
**Seen in:** `app.py`

## gr.Markdown, gr.Button, gr.Textbox `(third-party: gradio)`
**1. Definition:** `gr.Markdown` renders Markdown text, `gr.Button` is a clickable button (with `variant="primary"` for emphasis), and `gr.Textbox` is a text input or output.
**2. In plain words:** These are the basic building blocks you place inside Blocks and then connect with events.
**3. Example to quote:** > "I used `gr.Button("Predict", variant="primary")` and connected it with `.click(...)`."
**Seen in:** `app.py`

## Callback return values must match outputs `(concept)`
**1. Definition:** The number and order of values a callback returns must match the `outputs` list of the event listener.
**2. In plain words:** If `local_predict` returns three values but you list two outputs, Gradio raises an error about the output count.
**3. Example to quote:** > "I removed `history_state` from the outputs but still returned three values, and Gradio complained about the count."
**Seen in:** `app.py`, BREAK IT comment

## share=True `(third-party: gradio)`
**1. Definition:** A `launch` option that requests a temporary public `https://xxxx.gradio.live` URL tunnelling to your machine, valid for about a week while your process runs.
**2. In plain words:** It's the quickest way to show a demo to someone else, but anyone with the link can use it, so don't expose private data or models this way. Permanent options are Hugging Face Spaces, Docker or a cloud VM.
**3. Example to quote:** > "I added `share=True` and Gradio printed a public link; I closed it because the model was private."
**Seen in:** `app.py` docstring

## theme in launch() `(third-party: gradio)`
**1. Definition:** In Gradio 6 the `theme` (and `css`) are passed to `launch()` instead of `gr.Blocks()`.
**2. In plain words:** The `gr.themes.Soft()` object changes the look of the whole app. In older Gradio versions this argument was on `gr.Blocks`, so check your installed version if it errors.
**3. Example to quote:** > "`demo.launch(theme=gr.themes.Soft())` applies the soft theme in this Gradio version."
**Seen in:** `app.py`

## demo.queue `(third-party: gradio)`
**1. Definition:** Configures the request queue; `max_size` limits how many requests can wait.
**2. In plain words:** Gradio queues requests by default so slow functions don't overload the server. Setting a max size protects you from too many simultaneous users.
**3. Example to quote:** > "`demo.queue(max_size=5)` would turn away new visitors when five are already waiting."
**Seen in:** `app.py`, TRY IT comment

## Default argument dependency injection `(syntax)`
**1. Definition:** A function parameter with a default value that the framework recognises and replaces with a live object, as with `progress=gr.Progress()`.
**2. In plain words:** You don't pass `progress` yourself; Gradio inspects the signature and supplies a working progress tracker when it calls the function.
**3. Example to quote:** > "`def api_predict(text, api_url, progress=gr.Progress())` gets its progress helper from Gradio automatically."
**Seen in:** `app.py`

## Dict output for gr.Label `(concept)`
**1. Definition:** `gr.Label` expects `{class_name: probability}`; the API tab converts the API's `label` and `score` into such a dictionary.
**2. In plain words:** The API returns one label and one score, so the UI builds a one-item dictionary to display it.
**3. Example to quote:** > "`{data["label"]: float(data["score"])}` turns the API reply into something `gr.Label` can draw."
**Seen in:** `app.py`
