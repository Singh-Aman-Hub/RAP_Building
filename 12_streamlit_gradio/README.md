# 12 - Streamlit and Gradio (build a UI for a model)

## The story of this module
**Chapter 24, The Face of AI.** Module 11 ended with a shippable API that only `curl` can talk to, so product managers and teammates cannot give feedback. Spark: write the UI in plain Python. You start with Streamlit's rerun model (01), cache the model and keep state (02), turn the UI into a thin API client with loading and error states (03), arrange the page (04), then rebuild the idea with Gradio's callback model (05) and Blocks, state, API and sharing (06). When the API streams tokens over SSE ([09 step 10](../09_api_development/10_streaming_sse/)), these UIs are where a human sees the answer arrive word by word. Cliffhanger: model, API, container and UI all exist as separate pieces. The [capstone (module 13)](../13_capstone_project/) assembles them into one system.

## Why this matters
A model nobody can try is hard to demo. Streamlit and Gradio turn Python into a web UI in minutes. Interviewers want you to connect a UI to a model or API, show loading/error states, keep state, and explain the callback/rerun model. The syllabus says "choose at least one": this module teaches **both** so you can compare them (learn Streamlit first, Gradio second).

## Where this fits
**Builds on**
- [../09_api_development/](../09_api_development/): the UI calls the `POST /predict` API (default `http://localhost:8000`). `common/mock_api.py` mimics it so you can work before module 09 is done.
- [../03_ml_foundations/](../03_ml_foundations/): `common/model.py` is a scikit-learn pipeline (vectoriser + classifier); `predict_proba` and "train once, reuse" come from there.
- [../00_python_foundations/](../00_python_foundations/): functions, dicts, comprehensions, try/except, f-strings, decorators, context managers (`with`).
- [../01_debugging_and_environment/](../01_debugging_and_environment/): reading tracebacks (Streamlit shows them in the browser), environment variables, ports.

**Leads to**
- [../11_docker/](../11_docker/): the Streamlit UI is the `ui` service in the Compose stack; `API_URL` env var and `--server.address 0.0.0.0` matter there.
- [../10_ml_system_architecture/](../10_ml_system_architecture/): UI -> API -> model is the serving path you will draw.
- [../13_capstone_project/](../13_capstone_project/): the final project's front end.

## Concept map
```
 user action (click / type / upload)
        |
   +----+-----------------------------+
   | STREAMLIT: rerun whole script    |   GRADIO: call a function (callback) with component values
   | widgets RETURN values            |   components declared as inputs=[...], outputs=[...]
   | state: st.session_state          |   state: gr.State
   | cache: st.cache_resource (model) |   load model at module level (once)
   | errors: st.error / st.warning    |   errors: raise gr.Error
   | wait:   st.spinner / st.status   |   wait:   gr.Progress / built-in queue
   +----+-----------------------------+
        | local model (common/model.py)   OR   HTTP call (requests) -> API (module 09 / mock_api)
        v                                          |-- timeout, ConnectionError, HTTPError handling
   prediction displayed (metric / label / table)
        |
   run & share: streamlit run / demo.launch() / share=True / Docker / Spaces
```

## Study path
Setup once: `uv pip install streamlit gradio scikit-learn requests` (already in the repo `.venv`).

| Step | Folder | Learn | Run | Time |
|---|---|---|---|---|
| 1 | [01_streamlit_basics](01_streamlit_basics/) | install, text IO, button, rerun model | `streamlit run app.py` | 40 min |
| 2 | [02_streamlit_model_state_uploads](02_streamlit_model_state_uploads/) | cache model, forms, session_state, file upload, show predictions | `streamlit run app.py` | 60 min |
| 3 | [03_streamlit_api_client](03_streamlit_api_client/) | call the API, spinner, error handling | API + `streamlit run app.py` | 60 min |
| 4 | [04_streamlit_layout_style](04_streamlit_layout_style/) | sidebar, tabs, columns, theme | `streamlit run app.py` | 40 min |
| 5 | [05_gradio_basics](05_gradio_basics/) | Interface, callbacks, gr.Error | `python app.py` | 40 min |
| 6 | [06_gradio_blocks_api_share](06_gradio_blocks_api_share/) | Blocks, State, upload, API, progress, share | `python app.py` | 60 min |

Shared helpers: [common/model.py](common/model.py) (tiny trained model) and [common/mock_api.py](common/mock_api.py) (stand-in API). Automated checks: `../.venv/bin/python -m pytest -q test_apps.py` (7 tests, all pass: Streamlit via `AppTest`, Gradio callbacks by direct call, real API through an in-process uvicorn).

Exercises: [exercises.md](exercises.md). Answers: [solutions/](solutions/).

## Syllabus bullets explained

**Installation and setup.** `pip install streamlit` / `pip install gradio` inside a virtual environment; check with `streamlit hello` or a 3-line Gradio script. Streamlit serves on 8501, Gradio on 7860.

**Text input and output components.** Streamlit: `st.text_input`, `st.text_area`, `st.write/success/metric/json`. Gradio: `gr.Textbox`, `gr.Label`. Streamlit widgets *return values*; Gradio components are *wired* to functions.

**Buttons and event handling.** Streamlit: `if st.button(...):` (script reruns, button is True once). Gradio: `button.click(fn, inputs, outputs)` or `Interface` submit. Rerun model = simple but re-executes everything; callback model = explicit, only your function runs. Trade-off: Streamlit needs caching/state discipline; Gradio needs input/output alignment.

**Forms and file uploads.** `st.form` groups inputs to one submit; `st.file_uploader` returns an in-memory object (`getvalue()` bytes). Gradio: `gr.File(type="filepath")` passes a temp path. Validate file type/size; never trust content.

**Displaying model predictions.** `st.metric`, tables, `gr.Label` (needs `{class: prob}`), `gr.Dataframe`. Show confidence, not just the label.

**Connecting to APIs.** `requests.post(url, json=..., timeout=...)`, `raise_for_status()`, parse JSON. UI = thin client; the URL comes from an env var (`API_URL`) so Docker/Compose can change it. Inside containers use service names, not `localhost`.

**Loading and using ML models.** Load once: `@st.cache_resource` in Streamlit (otherwise reloads each rerun); in Gradio load at import. Cached objects are shared across users, so treat them as read-only.

**Error handling and loading states.** Validate first, `try/except` specific exceptions, show `st.error`/`gr.Error`; `st.spinner`, `st.status`, `gr.Progress`. Never let the user see a raw stack trace or a hung page.

**Session state.** `st.session_state` and `gr.State`: per-user memory that survives interactions, lost on server restart. Not a database.

**Layouts and columns.** `st.sidebar`, `st.columns`, `st.tabs`, `st.expander`; Gradio `gr.Row/Column/Tab` inside `gr.Blocks`.

**Basic UI customization.** Streamlit `st.set_page_config` + `.streamlit/config.toml` theme; Gradio `theme=gr.themes.Soft()` passed to `launch()` in Gradio 6 (was `gr.Blocks(theme=...)` in older versions), `title`, `description`.

**Running and sharing.** Local: `streamlit run app.py [--server.port N --server.address 0.0.0.0]`, `demo.launch(server_name="0.0.0.0", server_port=N)`. Share quickly: `share=True` (temporary public tunnel, Gradio only) or tunnel tools like ngrok for Streamlit. Deploy properly: Docker (module 11), Streamlit Community Cloud, Hugging Face Spaces (supports both), or any VM. Add authentication (`launch(auth=(user, pass))`) before exposing publicly.

### Streamlit vs Gradio at a glance
| | Streamlit | Gradio |
|---|---|---|
| Mental model | script reruns on every interaction | functions with declared inputs/outputs |
| Best for | dashboards, multi-widget tools | quick model demos, instant API |
| State | `st.session_state` | `gr.State` |
| Caching | `st.cache_resource/data` | load at module level |
| Sharing | cloud / tunnels | `share=True` built in |
| Typical port | 8501 | 7860 |

## Common mistakes and gotchas
- Forgetting reruns: plain variables reset; heavy work (model load) repeats without caching.
- Calling `st.set_page_config` after another Streamlit call.
- Duplicate widget labels without `key=`.
- Inside a form, widget changes are only sent on submit.
- No `timeout` on network calls; catching bare `Exception`.
- `localhost` inside Docker; binding to `127.0.0.1` so the container UI is unreachable.
- Gradio: output count != returned values; returning the wrong type for `gr.Label`; mutating `gr.State` in place; passing `theme` to `gr.Blocks` on Gradio 6.
- `share=True` on a machine holding private data.

## Self-check
- [ ] Explain the Streamlit rerun model and why `st.button` returns True only once
- [ ] Explain the Gradio callback model (inputs -> function -> outputs)
- [ ] Built a text-in / prediction-out UI in both frameworks without looking
- [ ] Cached a model; used session state; handled an upload
- [ ] Connected a UI to the API; handled ConnectionError, Timeout, HTTPError
- [ ] Loading + error states shown in both
- [ ] Used columns/tabs/sidebar; changed the theme
- [ ] Can say how to run and share/deploy each

## Interview questions
1. **How does Streamlit work?** It reruns the script top to bottom on each interaction; widgets return current values. Trade-off: need cache/session_state to avoid redoing work and losing data.
2. **How does Gradio differ?** You declare components and a function; Gradio calls the function as a callback and maps return values to outputs. Trade-off: less layout freedom (Blocks helps), alignment of counts matters.
3. **Why `st.cache_resource` for models?** Loads once per process, shared across reruns and users. Must be thread-safe; use `cache_data` for plain data that is copied.
4. **What is session state?** Per-session dict surviving reruns; lost on restart; not for sharing between users.
5. **How do you call a model API from a UI robustly?** Validate input, `timeout`, `raise_for_status`, catch specific errors, spinner, user-friendly messages; URL from env var.
6. **Why separate the UI from the model service?** Independent deployment/scaling, other clients can reuse the API; cost: network latency and another failure point.
7. **How would the UI container find the API container in Compose?** `http://api:8000` by service name, not localhost.
8. **What does `share=True` do and what are the risks?** Temporary public tunnel via Gradio servers to your machine; anyone with the link can use it; do not expose sensitive data.
9. **How do you handle file uploads safely?** Restrict types, size limits, decode with explicit encoding, validate content, handle exceptions, avoid executing content.
10. **A Streamlit app is slow after every click. Why?** Heavy code (model load, API call) at the top level reruns; fix with caching, forms, moving work under a button.
11. **How do you show progress for slow predictions?** `st.spinner`/`st.status`; `gr.Progress`; for very slow models consider async job + polling in the API.
12. **When would you not use these tools?** Production-grade multi-user products needing custom design, auth, and scale: use a real frontend + API.
