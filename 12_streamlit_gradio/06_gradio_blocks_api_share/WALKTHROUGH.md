# Step 06 - Gradio Blocks, API calls, uploads, state, sharing

## The story so far
[Step 05](../05_gradio_basics/WALKTHROUGH.md) covered the one-function `Interface`. Real demos outgrow it.

## The pain point
A single function cannot hold history between clicks, split work into tabs, accept uploads, show progress or run two events. Functions that stream results (a generator that yields partial text) are also awkward to express in a plain `Interface`.

## The spark
`gr.Blocks` lets you place components yourself and wire events: a click calls a function with component values and spreads its return values over the outputs. `gr.State` keeps per-session data, `gr.Progress` shows waiting, and `share=True` makes a temporary public link. A generator function that yields partial text is how you would show an LLM answer arriving token by token, the UI side of [SSE streaming](../../09_api_development/10_streaming_sse/).

## Goal
Build a multi-tab app with events, state, file upload, API integration, error/loading states; know how to run and share it.

## Run it
```bash
cd 12_streamlit_gradio/06_gradio_blocks_api_share
../../.venv/bin/python app.py                 # local only
# API tab needs: cd ../common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000
```
Verified headlessly (../test_apps.py): `local_predict` grows history; `batch_predict` on `../02_streamlit_model_state_uploads/sample_sentences.txt` returns 4 rows; the demo builds. The temporary-public-link behaviour of `share=True` was not exercised (it needs internet and publishes the app).

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `Row` vs `Column` vs `Tab` | layout in Blocks | No | Row = side by side, Column = stacked, Tab = switchable panel |
| `gr.State` vs a global variable | per-session storage | No | `gr.State` is private to one browser session; a global is shared by all users |
| `.click` vs `.upload` vs `.change` | events | No | Each component has its own events; you wire them to functions |
| `share=True` vs Hugging Face Spaces vs Docker | public link vs permanent hosting | No | `share=True` is a temporary tunnel to your laptop; Spaces and containers are permanent hosting |
| `gr.Error` vs normal exception | error toast | No | `gr.Error` shows its message to the user; other exceptions show a generic error and a traceback in the terminal |
| `gr.Progress` vs spinner | progress feedback | Related | `Progress` updates a bar with a fraction and description |

## Code, step by step

### 1. Docstring and sharing note  (`app.py` lines 1–12)
```python
"""
GOAL: Gradio Blocks: custom layout, several event handlers, uploads, API calls, loading/error states, state, share.

RUN (from 12_streamlit_gradio/06_gradio_blocks_api_share):  ../../.venv/bin/python app.py
     (for the API tab, also run the API: cd ../common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000)
EXPECTED OUTPUT: tabs "Local model", "Via API" and "Batch file"; a growing history table.
NEEDS: gradio, requests, scikit-learn.

SHARING: demo.launch(share=True) asks Gradio's servers for a temporary public https://xxxx.gradio.live URL that
tunnels to YOUR machine for ~1 week. Anyone with the link can use the app while your process runs. Do not
expose private data/models that way. Permanent options: Hugging Face Spaces, Docker (module 11), any cloud VM.
"""
```
- **In simple words:** What the app has and how sharing works.
- **Exactly:** `share=True` requests a temporary public `https://xxxx.gradio.live` URL that tunnels to your machine; the lifetime is limited, check current Gradio docs. Anyone with the link can use it while your process runs.
- **Why it is here:** A warning: do not expose private data or models this way.

### 2. Imports and settings  (`app.py` lines 13–23)
```python
import os
import sys
from pathlib import Path

import gradio as gr
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import predict  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")
```
- **In simple words:** Load tools, add the shared model, read the API address.
- **Exactly:** `os`, `sys`, `Path`, `gradio`, `requests`; `API_URL` from the environment with default `http://localhost:8000`.
- **Why it is here:** Same pattern as step 03.

### 3. Callback 1: local model with state  (`app.py` lines 26–32)
```python
def local_predict(text, history):
    """Callback 1: use the local model. `history` is a gr.State value (list) kept per browser session."""
    if not text.strip():
        raise gr.Error("Please enter some text.")
    label, score = predict(text)
    history = history + [[text, label, score]]            # new list (don't mutate the old state in place)
    return f"{label} ({score})", history, history        # outputs: textbox, state, dataframe
```
- **In simple words:** Predict with the local model and add the result to a history list.
- **Exactly:** `history` is the current `gr.State` value (a list). `history + [[...]]` builds a NEW list (do not mutate state in place). Returns three values, one for each output: result text, new state, table.
- **Why it is here:** The number of returned values must match `outputs` (BREAK IT).

### 4. Callback 2: call the API with progress  (`app.py` lines 35–49)
```python
def api_predict(text, api_url, progress=gr.Progress()):
    """Callback 2: call the HTTP API. gr.Progress() injects a progress-bar helper automatically."""
    if not text.strip():
        raise gr.Error("Please enter some text.")
    progress(0.2, desc="Calling API...")
    try:
        r = requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)
        r.raise_for_status()
    except requests.ConnectionError:
        raise gr.Error(f"Cannot reach the API at {api_url}. Is it running?")
    except requests.HTTPError as exc:
        raise gr.Error(f"API error {exc.response.status_code}: {exc.response.text}")
    progress(1.0, desc="Done")
    data = r.json()
    return {data["label"]: float(data.get("confidence", data.get("score")))}          # gr.Label wants {class: probability}
```
- **In simple words:** Send the text to the HTTP API and show a progress bar.
- **Exactly:** `progress=gr.Progress()` as a default argument is injected by Gradio. `progress(0.2, desc=...)` sets the bar. `requests.post(..., timeout=10)` and `raise_for_status()`; `ConnectionError` and `HTTPError` become `gr.Error`. Returns `{label: probability}` for `gr.Label`, using `confidence` or `score` depending on the API.
- **Why it is here:** Clear loading and error states for slow or unavailable APIs.

### 5. Callback 3: batch file  (`app.py` lines 52–57)
```python
def batch_predict(file_obj):
    """Callback 3: file upload. With gr.File(type='filepath') the function receives a path string (or None)."""
    if file_obj is None:
        raise gr.Error("Upload a .txt file first.")
    lines = Path(file_obj).read_text(encoding="utf-8").splitlines()
    return [[ln, *predict(ln)] for ln in lines if ln.strip()]      # rows for gr.Dataframe
```
- **In simple words:** Classify every line in an uploaded file.
- **Exactly:** With `type="filepath"`, the function receives a path string or `None`. `Path(...).read_text` then `splitlines()`. `[ln, *predict(ln)]` builds `[text, label, score]` rows.
- **Why it is here:** A table output needs a list of rows.

### 6. Create the Blocks app and state  (`app.py` lines 60–62)
```python
with gr.Blocks(title="Sentiment lab (Gradio)") as demo:    # Blocks = you place components and wire events yourself
    gr.Markdown("# Sentiment lab\nBlocks gives layout control and multiple event handlers.")
    history_state = gr.State([])                           # invisible per-session storage
```
- **In simple words:** Start a blank page where we place things ourselves.
- **Exactly:** `gr.Blocks(title=...)` is a context manager. `gr.Markdown` shows text. `gr.State([])` is invisible per-session storage.
- **Why it is here:** Blocks gives full control of layout and events.

### 7. Tab 1: layout and wiring  (`app.py` lines 64–73)
```python
    with gr.Tab("Local model"):
        with gr.Row():                                      # Row = side-by-side, Column = stacked
            with gr.Column(scale=2):
                txt = gr.Textbox(label="Sentence", value="I love this")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                out = gr.Textbox(label="Result")
        table = gr.Dataframe(headers=["text", "label", "score"], label="History")
        # EVENT WIRING: when btn is clicked, call local_predict(inputs...) and put returns into outputs...
        btn.click(local_predict, inputs=[txt, history_state], outputs=[out, history_state, table])
```
- **In simple words:** Two columns, a text box and button on the left, result on the right, and a table below.
- **Exactly:** `Row` and `Column(scale=2)` / `scale=1` set the 2:1 widths. `btn.click(fn, inputs=[txt, history_state], outputs=[out, history_state, table])`: when clicked, Gradio passes the current values to `local_predict` and writes the three returns into the outputs.
- **Why it is here:** This line is the heart of Blocks: explicit event wiring.

### 8. Tab 2: call the API  (`app.py` lines 75–79)
```python
    with gr.Tab("Via API"):
        url = gr.Textbox(label="API base URL", value=API_URL)
        txt2 = gr.Textbox(label="Sentence", value="Terrible service")
        out2 = gr.Label(label="API result")
        gr.Button("Call API").click(api_predict, inputs=[txt2, url], outputs=out2)
```
- **In simple words:** A tab with a URL box, a sentence box and a button that calls the API.
- **Exactly:** `gr.Button("Call API").click(api_predict, inputs=[txt2, url], outputs=out2)`; the progress helper is injected so it is not listed in `inputs`.
- **Why it is here:** Lets you point the UI at any API.

### 9. Tab 3: upload event  (`app.py` lines 81–84)
```python
    with gr.Tab("Batch file"):
        f = gr.File(label="Upload .txt (one sentence per line)", type="filepath", file_types=[".txt"])
        res = gr.Dataframe(headers=["text", "label", "score"])
        f.upload(batch_predict, inputs=f, outputs=res)     # event fires when a file is uploaded
```
- **In simple words:** Upload a file and see the predictions.
- **Exactly:** `gr.File(type="filepath", file_types=[".txt"])` restricts types; the `.upload` event fires when a file is added.
- **Why it is here:** Shows a different event than `click`.

### 10. Launch with theme  (`app.py` lines 86–88)
```python
if __name__ == "__main__":
    # theme (and css) are passed to launch() in Gradio 6; in Gradio 4/5 they were arguments of gr.Blocks().
    demo.launch(theme=gr.themes.Soft())     # add share=True for a temporary public link
```
- **In simple words:** Start the server with a soft theme.
- **Exactly:** In Gradio 6, `theme` is passed to `launch()` (in Gradio 4/5 it was an argument of `gr.Blocks()`). `share=True` would add the public link.
- **Why it is here:** Version differences are a common source of errors, check the installed version.

### 11. Exercises as comments  (`app.py` lines 90–92)
```python
# TRY IT: change gr.themes.Soft() to gr.themes.Glass(); add share=True and read the printed public URL.
# TRY IT: add demo.queue(max_size=5) to limit concurrent users (queue is on by default).
# BREAK IT: remove `history_state` from outputs but keep returning 3 values -> error about output count.
```
- **In simple words:** Try another theme, share, queue.
- **Exactly:** `demo.queue(max_size=5)` limits waiting users; removing `history_state` from outputs gives an output-count error.
- **Why it is here:** Understand how outputs map to returns.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `with gr.Blocks() as demo:` | context manager collecting components | |
| `with gr.Row()/Column()/Tab():` | layout containers | |
| `btn.click(fn, inputs=[...], outputs=[...])` | register event handler (callback) | |
| `history + [[...]]` | list concatenation creates a NEW list | `[1] + [2]` -> `[1, 2]` |
| `[ln, *predict(ln)]` | `*` unpacks a tuple into the list | `[1, *(2, 3)]` -> `[1, 2, 3]` |
| default arg `progress=gr.Progress()` | Gradio detects it and injects a tracker | |
| `raise ... from`/`raise gr.Error` in `except` | convert technical errors to user messages | |

## Libraries & functions
| Name | Built-in / 3rd-party | Does | Important args | Returns |
|---|---|---|---|---|
| `gr.Blocks` | gradio | low-level app builder | `title` | context manager |
| `gr.State` | gradio | per-session hidden value | initial value | component |
| `gr.File` | gradio | upload | `type="filepath"`, `file_types` | path str / None |
| `gr.Dataframe` | gradio | table | `headers` | component |
| `gr.Progress` | gradio | progress bar helper | call `progress(frac, desc=)` | |
| `.click/.upload` | gradio | event listeners | `fn, inputs, outputs` | |
| `.launch(theme=, share=, server_port=)` | gradio | start server | | |
| `Path.read_text` | built-in | read whole file | `encoding` | str |

## Execution trace
Click Predict -> Gradio collects `txt` and `history_state` values -> calls `local_predict(text, history)` -> returns a 3-tuple matched in order to `[out, history_state, table]` -> UI updates all three. State persists per session, not across users.

## Experiments
1. Press Predict twice; history grows.
2. API tab with the API stopped: red toast "Cannot reach the API".
3. Upload a file with blank lines.
4. Launch with `share=True` only on non-sensitive demos.

## Debugging practice
Returning the wrong number of values -> "A function ... didn't return enough output values". Count returns vs `outputs=[...]`; order matters.

## Rebuild checklist
- [ ] Blocks with Textbox, Button, click wiring from memory.
- [ ] State for history; gr.Error for failures.

## The new problem (cliffhanger)
Now every piece exists on its own: a model, a FastAPI service, a container image, a Compose stack and a UI. None of them is yet one system that someone else can start with one command. Next: [module 13, the capstone](../../13_capstone_project/README.md).

## Say it aloud
> **Say it aloud (interview box):** With Blocks I place components and wire events, so a click calls a function with the input values and distributes its return values to the outputs. gr.State keeps per-session data, gr.Error turns failures into friendly messages, and share=True creates a temporary public tunnel for demos. The trade-off is that the callback style is explicit but input and output counts must stay aligned, and share links expose my machine.
