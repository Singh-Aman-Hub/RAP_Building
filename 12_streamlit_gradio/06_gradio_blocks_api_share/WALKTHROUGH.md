# Step 06 - Gradio Blocks, API calls, uploads, state, sharing

## Goal
Build a multi-tab app with events, state, file upload, API integration, error/loading states; know how to run and share it.

## Run it
```bash
cd 12_streamlit_gradio/06_gradio_blocks_api_share
../../.venv/bin/python app.py                 # local only
# API tab needs: cd ../common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000
```
Verified headlessly (../test_apps.py): `local_predict` grows history; `batch_predict` on `../02_streamlit_model_state_uploads/sample_sentences.txt` returns 4 rows; the demo builds. The temporary-public-link behaviour of `share=True` was not exercised (it needs internet and publishes the app).

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

## Say it aloud
"With Blocks I place components and wire events: click calls a function with the input values and distributes its return values to the outputs. gr.State keeps per-session data and gr.Error turns failures into user-friendly messages. share=True creates a temporary public tunnel; for real deployment I would use Spaces or a container. Trade-off: the callback style is explicit but you must keep input/output counts aligned, and share links expose your machine."
