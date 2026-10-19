# Step 02 - Model loading, forms, session state and file upload

## Goal
Show model predictions in the UI without reloading the model each rerun, remember a history, and batch-predict an uploaded file.

## Run it
```bash
cd 12_streamlit_gradio/02_streamlit_model_state_uploads
../../.venv/bin/streamlit run app.py        # upload sample_sentences.txt to try the batch part
```
Verified headlessly with `AppTest` (see ../test_apps.py): predicting "I love this" adds one history row and the model loader ran once.

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

## Say it aloud
"Because Streamlit reruns the script, I cache the model with cache_resource so it loads once, and I store per-user data like history in session_state. Forms avoid reruns on each keystroke. Trade-off: cache_resource objects are shared across users, so they must be thread-safe/read-only; session_state is lost when the server restarts."
