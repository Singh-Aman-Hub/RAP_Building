# Step 01 - Streamlit basics and the rerun model

## The story so far
[Module 11](../../11_docker/06_multistage_and_security/WALKTHROUGH.md) ended with a shippable API that only `curl` can talk to. We now give humans a way to try it.

## The pain point
To test a model, a teammate must write JSON by hand, run `curl` and read raw text. Nobody outside engineering will do that, so nobody gives feedback. A normal web front end (HTML, JavaScript, a build chain) is a project of its own.

## The spark
Streamlit lets you write the UI as a plain Python script: `st.text_input` returns what the user typed and `st.button` returns True when clicked. Every interaction reruns the script top to bottom, so UI code reads like ordinary code.

## Goal
Install Streamlit, display text, read a text box, react to a button, and understand why the script reruns.

## Run it
```bash
cd 12_streamlit_gradio/01_streamlit_basics
../../.venv/bin/streamlit run app.py      # opens http://localhost:8501; Ctrl+C stops it
```
Setup in a fresh environment: `python -m venv .venv && source .venv/bin/activate && pip install streamlit` then `streamlit hello` shows the demo gallery.
Non-interactive verification (real, see test_apps.py): `AppTest.from_file("app.py").run()` produced no exceptions, with `count = 1`.

## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `import streamlit as st` | import with a short alias | `st.title("x")` |
| `x = st.text_input(...)` | widget call returns its current value | `name = st.text_input("Name")` |
| `if st.button("Go"):` | True only in the click's rerun | |
| f-string `f"{count}"` | embed values in strings | `f"n={1+1}"` -> `n=2` |
| `else:` branch of button | what shows before any click | |

## Libraries & functions
| Name | Built-in / 3rd-party | Does | Important args | Returns |
|---|---|---|---|---|
| `st.title` / `st.write` / `st.caption` | 3rd-party (streamlit) | display content | the object to show | None |
| `st.text_input` | streamlit | one-line text box | `label`, `value`, `key` | str |
| `st.button` | streamlit | button | `label` | bool |
| `st.success/info/error/warning` | streamlit | coloured message box | message | None |
| `str.upper()` | built-in | uppercase copy | none | str |

## Execution trace
1. Input: you open the page; Streamlit runs app.py once: text_input returns `"hello world"`, button returns `False` -> `info("Waiting...")`.
2. You click Shout: the browser tells the server; Streamlit reruns app.py from line 1; `button` now returns `True`; `success("HELLO WORLD")`.
3. You edit the box: another rerun; button back to `False`.
4. `count` is rebuilt as 1 on each rerun.

## Experiments
1. Change the default text; observe rerun.
2. Add `st.slider`; move it; watch the script rerun (put `st.write("ran")` at the top).
3. Replace `st.button` with `st.checkbox` and compare.

## Debugging practice
Errors are shown in the browser with a full traceback. Introduce `st.text_input()` (missing label): read bottom-up: `TypeError: ... missing 1 required positional argument: 'label'`.

## Rebuild checklist
- [ ] Re-create the app (title, text_input, button, success) without looking.
- [ ] Explain why `count` is always 1.

## The new problem (cliffhanger)
The rerun model has a price: anything slow (loading a model) happens again on each click, and variables reset. Next: [step 02, caching, session state and uploads](../02_streamlit_model_state_uploads/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** Streamlit turns a Python script into a web app, where every interaction reruns the script top to bottom and widgets return their current values. That makes UI code read like normal code, so a model can be demoed in minutes. The trade-off is that nothing persists across reruns unless I use session_state or caching, and expensive work reruns unless it is cached.
