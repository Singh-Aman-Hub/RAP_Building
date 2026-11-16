# Step 04 - Layouts, columns, tabs and basic customisation

## The story so far
[Step 03](../03_streamlit_api_client/WALKTHROUGH.md) made the UI call the API correctly, but everything is stacked in one column.

## The pain point
Settings, inputs, results and comparisons all compete for the same strip of screen. Users cannot tell what is a control and what is an output, and a demo looks unfinished.

## The spark
Use layout containers: a sidebar for settings, tabs for separate views, columns for side-by-side comparison, and `set_page_config` plus a `config.toml` theme for the look. Layout only decides where things appear, not what runs.

## Goal
Arrange a UI sensibly and customise its look with page config and a theme file.

## Run it
```bash
cd 12_streamlit_gradio/04_streamlit_layout_style
../../.venv/bin/streamlit run app.py      # must be run from this folder for .streamlit/config.toml
```
Verified headlessly (see ../test_apps.py): Classify and Compare tabs render with no exceptions.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Sidebar vs tabs vs columns vs expander | layout containers | No | Sidebar = persistent side panel; tabs = switch views; columns = side by side; expander = fold/unfold |
| `with st.x:` vs `x.method()` | context vs object style | Same result | `with left: st.text_input(...)` and `left.text_input(...)` put the widget in the same place |
| Widget `key` vs label | identity of a widget | No | Streamlit identifies widgets by type + label + key; same label twice needs unique keys |
| `config.toml` theme vs custom CSS | styling | No | The theme file is supported and stable; CSS via `unsafe_allow_html` is fragile |
| `st.status` vs `st.spinner` | progress display | Related | `status` is a collapsible box with updatable label/state; `spinner` is a simple "working" indicator |

## Code, step by step

### 1. Docstring  (`app.py` lines 1–8)
```python
"""
GOAL: Layout and basic customisation: page config, sidebar, columns, tabs, expander, status, theme file.

RUN (from 12_streamlit_gradio/04_streamlit_layout_style):  ../../.venv/bin/streamlit run app.py
     (run it FROM this folder so .streamlit/config.toml, the green theme, is picked up)
EXPECTED OUTPUT: wide page with a sidebar of settings, two tabs ("Single" and "Compare"), and side-by-side columns.
NEEDS: streamlit, scikit-learn (../common/model.py).
"""
```
- **In simple words:** What this app shows: layout and theme.
- **Exactly:** Run it from this folder so `.streamlit/config.toml` (the green theme) is found.
- **Why it is here:** Theme files are read from the current working directory.

### 2. Imports  (`app.py` lines 9–13)
```python
import sys
import time
from pathlib import Path

import streamlit as st
```
- **In simple words:** Load tools for time, paths and Streamlit.
- **Exactly:** `time` for `sleep`, `pathlib` for the path, `streamlit`.
- **Why it is here:** Needed for the fake delay and the model import.

### 3. Import the shared model  (`app.py` lines 15–16)
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import predict  # noqa: E402
```
- **In simple words:** Make the common folder importable and load `predict`.
- **Exactly:** `parents[1] / "common"` is `12_streamlit_gradio/common`; `# noqa: E402` allows the late import.
- **Why it is here:** Same trick as step 02.

### 4. Page configuration  (`app.py` lines 18–19)
```python
# set_page_config MUST be the first Streamlit call in the script.
st.set_page_config(page_title="Sentiment lab", page_icon=":bar_chart:", layout="wide")
```
- **In simple words:** Set the browser tab title, icon and use the full width.
- **Exactly:** `st.set_page_config(page_title, page_icon, layout="wide")` must be the first Streamlit call, otherwise `StreamlitAPIException`.
- **Why it is here:** Controls the page shell before anything is drawn.

### 5. Heading and Markdown  (`app.py` lines 21–22)
```python
st.title("Sentiment lab")
st.markdown("Layout demo: **sidebar**, **tabs**, **columns**. Markdown works in most text elements.")
```
- **In simple words:** A title and a formatted sentence.
- **Exactly:** `st.markdown` renders **bold** and other Markdown.
- **Why it is here:** Quick rich text.

### 6. Sidebar settings  (`app.py` lines 24–28)
```python
# ---- Sidebar: global settings --------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)   # (label, min, max, default, step)
    show_raw = st.checkbox("Show raw output", value=False)
```
- **In simple words:** A left panel with a slider and a checkbox.
- **Exactly:** `with st.sidebar:` puts everything inside in the sidebar. `st.slider(label, min, max, value, step)` returns a `float` (0.5 here); `st.checkbox` returns a `bool`.
- **Why it is here:** Global settings stay visible while you move between tabs.

### 7. Two tabs  (`app.py` lines 30–31)
```python
# ---- Tabs: separate views on one page --------------------------------------------------------
tab_single, tab_compare = st.tabs(["Single", "Compare"])
```
- **In simple words:** Two pages in one.
- **Exactly:** `st.tabs([...])` returns one container per name, used with `with`. All tab bodies run on each rerun.
- **Why it is here:** Cleaner than a long page.

### 8. Single tab: classify with progress  (`app.py` lines 33–45)
```python
with tab_single:
    text = st.text_input("Sentence", "I love this layout", key="single_text")
    if st.button("Classify", key="single_btn"):
        with st.status("Running model...", expanded=False) as status:   # collapsible progress box
            time.sleep(0.3)                                              # pretend work
            label, score = predict(text)
            status.update(label="Done", state="complete")
        if score >= threshold:
            st.success(f"{label} ({score})")
        else:
            st.warning(f"Not confident: {label} ({score}) is below the threshold {threshold}")
        if show_raw:
            st.json({"label": label, "score": score})
```
- **In simple words:** Type a sentence, press Classify, see a progress box, then a green or yellow result.
- **Exactly:** `key=` gives widgets unique ids. `st.status(...)` is a context manager; `status.update(label="Done", state="complete")` finishes it. `time.sleep(0.3)` simulates work. If `score >= threshold` show `st.success` else `st.warning`; `st.json` appears when the checkbox is on.
- **Why it is here:** The threshold turns a probability into a decision, a real production pattern.

### 9. Compare tab: two columns and metrics  (`app.py` lines 47–60)
```python
with tab_compare:
    # Columns: side-by-side layout. st.columns(2) makes two equal columns; [2, 1] would be 2:1.
    left, right = st.columns(2)
    with left:
        a = st.text_input("Sentence A", "great and fast", key="a")
    with right:
        b = st.text_input("Sentence B", "slow and awful", key="b")
    if st.button("Compare", key="cmp_btn"):
        (la, sa), (lb, sb) = predict(a), predict(b)     # tuple unpacking of two tuples
        c1, c2 = st.columns(2)
        c1.metric("A", la, f"{sa}")                      # object-style: call methods on the column itself
        c2.metric("B", lb, f"{sb}")
        with st.expander("What does the threshold do?"):
            st.write("It only changes the message in the Single tab; the model output is the same.")
```
- **In simple words:** Two boxes side by side; compare two sentences.
- **Exactly:** `st.columns(2)` returns two equal columns (`[2, 1]` would be 2:1). `(la, sa), (lb, sb) = predict(a), predict(b)` unpacks two tuples. `c1.metric(...)` is object-style: call the method on the column.
- **Why it is here:** Compare view is the classic use of columns.

### 10. Custom CSS  (`app.py` lines 62–63)
```python
# Custom CSS is possible but fragile (Streamlit's internal class names change between versions).
st.markdown("<style>h1 {letter-spacing: 1px;}</style>", unsafe_allow_html=True)
```
- **In simple words:** A small style tweak for headings.
- **Exactly:** `unsafe_allow_html=True` allows raw HTML. Streamlit's internal class names change, so this can break.
- **Why it is here:** Included to show it works but to warn about fragility.

### 11. Exercises as comments  (`app.py` lines 65–67)
```python
# TRY IT: change layout="wide" to "centered".
# TRY IT: change the theme colours in .streamlit/config.toml and reload the page.
# BREAK IT: call st.set_page_config after st.title -> StreamlitAPIException; read the message.
```
- **In simple words:** Change layout, theme, or break set_page_config.
- **Exactly:** BREAK IT: calling `st.set_page_config` after `st.title` raises `StreamlitAPIException`.
- **Why it is here:** Reading Streamlit's exception message is a useful skill.

### 12. Theme and server settings  (`.streamlit/config.toml` lines 1–10)
```toml
# Streamlit reads .streamlit/config.toml from the folder you run `streamlit run` in (and ~/.streamlit/config.toml).
[theme]
primaryColor = "#0b6e4f"          # buttons, sliders, links
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#eef6f2"   # sidebar and widgets background
textColor = "#1a1a1a"
font = "sans serif"

[server]
headless = true                   # do not auto-open a browser (useful in Docker/servers)
```
- **In simple words:** A settings file that sets colours and stops the browser opening automatically.
- **Exactly:** `[theme]` keys `primaryColor` (buttons, sliders), `backgroundColor`, `secondaryBackgroundColor` (sidebar), `textColor`, `font`. `[server] headless = true` disables auto-opening a browser (useful in Docker).
- **Why it is here:** Restyle with no code change.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `with st.sidebar:` / `st.sidebar.x()` | put widgets in the side panel | |
| `a, b = st.columns(2)` | tuple unpacking of two column objects | `a, b = (1, 2)` |
| `with left:` vs `c1.metric(...)` | two equivalent ways to target a container | |
| `(la, sa), (lb, sb) = f(a), f(b)` | nested unpacking | `(x, y), z = (1, 2), 3` |
| `key="a"` | unique widget identity; required when labels repeat | |
| TOML `[theme]` | config file format: sections and key = value | |

## Libraries & functions
| Name | Built-in / 3rd-party | Does | Important args | Returns |
|---|---|---|---|---|
| `st.set_page_config` | streamlit | title, icon, layout | `layout="wide"` | None |
| `st.tabs` | streamlit | tabbed sections | list of names | list of containers |
| `st.columns` | streamlit | side-by-side | number or ratio list | list of columns |
| `st.expander`, `st.status` | streamlit | collapsible / progress box | label, expanded | context manager |
| `st.slider`, `st.checkbox` | streamlit | numeric / boolean input | min, max, default | number / bool |
| `time.sleep` | built-in (time) | pause | seconds | None |

## Execution trace
Run starts -> page config -> sidebar widgets return current slider/checkbox values -> tabs drawn -> only the clicked button's branch executes; threshold compares to the model confidence; layout containers just decide *where* output appears, not *what* runs.

## Experiments
1. Move the slider and note the Single tab message only changes after a click (button returns True only on click rerun).
2. Switch to `layout="centered"`.
3. Change `primaryColor`.

## Debugging practice
Two widgets with identical label and no key -> `DuplicateWidgetID`. Fix with unique `key=`.

## Rebuild checklist
- [ ] Sidebar + two columns + tabs from memory.
- [ ] Explain why `set_page_config` goes first.

## The new problem (cliffhanger)
Streamlit is one way to build this UI. The other popular tool wraps a single function instead of rerunning a script, and gives you a shareable link almost for free. Next: [step 05, Gradio basics](../05_gradio_basics/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** Layout in Streamlit is declarative: a sidebar for settings, tabs for views and columns for side-by-side comparison. Appearance comes from set_page_config and a config.toml theme. The trade-off is that deep custom styling needs fragile CSS hacks, so Streamlit suits internal tools and demos rather than pixel-perfect products.
