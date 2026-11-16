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

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Rerun vs callback | "Streamlit runs the script again", "event handler" | No | Streamlit re-executes the whole file on each interaction; Gradio/Tkinter call one function when an event happens |
| Widget vs element | `st.text_input`, `st.write` | Related | Widgets return a value and take input; other elements (title, caption, success) only display |
| `st.write` vs `st.markdown` vs `st.text` | display text | No | `write` picks a renderer by type (str, dict, DataFrame); `markdown` renders Markdown; `text` is fixed-width plain |
| `st.success` / `st.info` / `st.warning` / `st.error` | status boxes, alerts | Same family | Same box in four colours and meanings |
| Script vs app | "app.py", "Streamlit app" | Same | A Streamlit app is just a Python script run by `streamlit run` |

## Code, step by step

### 1. Docstring: goal, run line, key idea  (`app.py` lines 1–12)
```python
"""
GOAL: The smallest useful Streamlit app: text box -> button -> result, to learn the "rerun" model.

RUN (from 12_streamlit_gradio/01_streamlit_basics):  ../../.venv/bin/streamlit run app.py
RUN (from repo root):  .venv/bin/streamlit run 12_streamlit_gradio/01_streamlit_basics/app.py
EXPECTED OUTPUT: a browser tab at http://localhost:8501 with a title, a text box, a "Shout" button.
                 Click the button -> the uppercase text appears. Ctrl+C in the terminal stops the server.
NEEDS: streamlit.

THE KEY IDEA: Streamlit re-runs this WHOLE file, top to bottom, every time the user touches a widget.
There are no callbacks you must register: a widget function (st.text_input) simply RETURNS the current value.
"""
```
- **In simple words:** A note saying what the app is and the one thing to remember: the file runs again from the top on each click.
- **Exactly:** `streamlit run app.py` starts a server on port 8501. The "rerun" model: any widget interaction triggers a fresh top-to-bottom execution of the script in a new run; a widget call returns its current value.
- **Why it is here:** Understanding reruns explains every later surprise (variables reset, models reload).

### 2. Import and display static text  (`app.py` lines 13–16)
```python
import streamlit as st            # the usual alias; every Streamlit function starts with st.

st.title("Hello, Streamlit")      # big heading
st.write("Type something and press the button.")   # st.write displays almost anything (text, dict, DataFrame)
```
- **In simple words:** Load Streamlit and put a heading and a sentence on the page.
- **Exactly:** `import streamlit as st` is the usual alias. `st.title` draws an H1; `st.write` renders strings, dicts, DataFrames and more.
- **Why it is here:** Without any `st.` call nothing appears in the browser.

### 3. Text input that returns what is typed  (`app.py` lines 18–19)
```python
# st.text_input(label, value) draws a box and RETURNS what is currently typed (a str).
name = st.text_input("Your text", value="hello world")
```
- **In simple words:** Draw a text box. The variable `name` always holds what is in the box.
- **Exactly:** `st.text_input(label, value="hello world")` returns a `str`. Typing then pressing Enter (or leaving the box) triggers a rerun with the new value.
- **Why it is here:** This is the "no callbacks" idea: the widget call is also the way to read the value.

### 4. Button branch  (`app.py` lines 21–25)
```python
# st.button(label) returns True ONLY on the rerun triggered by the click, otherwise False.
if st.button("Shout"):
    st.success(name.upper())      # st.success draws a green box
else:
    st.info("Waiting for a click...")
```
- **In simple words:** If the button was just clicked show the shouted text, otherwise show a waiting note.
- **Exactly:** `st.button("Shout")` returns `True` only in the single rerun caused by that click, then `False` again. `name.upper()` uppercases; `st.success` draws a green box, `st.info` a blue one.
- **Why it is here:** Beginners expect a button to stay "on"; it does not. Use `st.session_state` for that (step 02).

### 5. Plain variable that resets on every rerun  (`app.py` lines 27–30)
```python
# Show how reruns work: this counter is a plain variable, so it resets on EVERY rerun (always 1).
count = 0
count += 1
st.caption(f"plain variable count = {count} (resets every rerun - see step 02 for st.session_state)")
```
- **In simple words:** A counter that never grows beyond 1.
- **Exactly:** `count = 0` then `count += 1` runs from scratch in every rerun, so the caption always shows `count = 1`. The real run confirmed `count == 1` (AppTest in `test_apps.py`).
- **Why it is here:** Proves that normal variables do not remember anything between interactions.

### 6. Exercises as comments  (`app.py` lines 32–34)
```python
# TRY IT: change `value="hello world"`, save the file; Streamlit offers "Rerun" / "Always rerun".
# TRY IT: add st.slider("n", 0, 10) and print its value with st.write.
# BREAK IT: write `st.text_input()` with no label -> TypeError shown in the browser; read the traceback there.
```
- **In simple words:** Ideas to change the app.
- **Exactly:** TRY IT (edit `value`, add `st.slider`) and BREAK IT (`st.text_input()` without a label raises `TypeError`, shown in the browser).
- **Why it is here:** Streamlit shows tracebacks in the page, so debugging is quick.


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
