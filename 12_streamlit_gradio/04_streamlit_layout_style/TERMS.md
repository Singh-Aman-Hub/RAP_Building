# Terms — Streamlit layout and basic customisation

Folder: `12_streamlit_gradio/04_streamlit_layout_style` · Files covered: `app.py`, `.streamlit/config.toml`, `WALKTHROUGH.md` (uses `../common/model.py`)
Back to module: [12_streamlit_gradio README](../README.md)

**Terms in this folder:** [st.set_page_config (third-party: streamlit)](#stset_page_config-third-party-streamlit) · [st.markdown (third-party: streamlit)](#stmarkdown-third-party-streamlit) · [st.sidebar and with st.sidebar (third-party: streamlit)](#stsidebar-and-with-stsidebar-third-party-streamlit) · [st.slider (third-party: streamlit)](#stslider-third-party-streamlit) · [st.checkbox (third-party: streamlit)](#stcheckbox-third-party-streamlit) · [st.tabs (third-party: streamlit)](#sttabs-third-party-streamlit) · [st.columns (third-party: streamlit)](#stcolumns-third-party-streamlit) · [Tuple unpacking (syntax)](#tuple-unpacking-syntax) · [key= parameter (third-party: streamlit)](#key-parameter-third-party-streamlit) · [st.status (third-party: streamlit)](#ststatus-third-party-streamlit) · [time.sleep (Python built-in module function)](#timesleep-python-built-in-module-function) · [st.expander (third-party: streamlit)](#stexpander-third-party-streamlit) · [st.metric (third-party: streamlit)](#stmetric-third-party-streamlit) · [st.success and st.warning (third-party: streamlit)](#stsuccess-and-stwarning-third-party-streamlit) · [st.json (third-party: streamlit)](#stjson-third-party-streamlit) · [Threshold on confidence (concept)](#threshold-on-confidence-concept) · [st.markdown with unsafe_allow_html (third-party: streamlit)](#stmarkdown-with-unsafe_allow_html-third-party-streamlit) · [.streamlit/config.toml (file)](#streamlitconfigtoml-file) · [TOML format (syntax)](#toml-format-syntax) · [Theme options (Streamlit config)](#theme-options-streamlit-config) · [server.headless (Streamlit config)](#serverheadless-streamlit-config) · [sys.path.insert and parents[1] (standard library)](#syspathinsert-and-parents1-standard-library) · [StreamlitAPIException (error)](#streamlitapiexception-error)

---

## st.set_page_config `(third-party: streamlit)`
**1. Definition:** Sets page-level options such as `page_title`, `page_icon` and `layout` ("centered" or "wide"); it must be the first Streamlit call in the script.
**2. In plain words:** It controls the browser tab title, its icon and how wide the content area is. If you call another Streamlit function first, you get a `StreamlitAPIException`.
**3. Example to quote:** > "I call `st.set_page_config(page_title="Sentiment lab", layout="wide")` before anything else, otherwise Streamlit raises an exception."
**Seen in:** `app.py`

## st.markdown `(third-party: streamlit)`
**1. Definition:** Renders Markdown text; with `unsafe_allow_html=True` it also allows raw HTML.
**2. In plain words:** Markdown gives bold, links and lists. Raw HTML/CSS is possible but fragile because Streamlit's internal class names change between versions, and it can be unsafe with untrusted text.
**3. Example to quote:** > "I used `st.markdown` with `**sidebar**` for bold, and a small `<style>` block for letter spacing."
**Seen in:** `app.py`

## st.sidebar and with st.sidebar `(third-party: streamlit)`
**1. Definition:** A side panel; widgets are placed inside it with `with st.sidebar:` or `st.sidebar.<widget>()`.
**2. In plain words:** It's for global settings, kept apart from the main content. Both spellings do the same thing.
**3. Example to quote:** > "My confidence threshold slider lives in the sidebar under 'Settings'."
**Seen in:** `app.py`

## st.slider `(third-party: streamlit)`
**1. Definition:** A slider widget that returns a number; arguments are `(label, min, max, default, step)`.
**2. In plain words:** It lets users choose a number by dragging. Here the value is a confidence threshold.
**3. Example to quote:** > "`st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)` starts at 0.5 and moves in steps of 0.05."
**Seen in:** `app.py`

## st.checkbox `(third-party: streamlit)`
**1. Definition:** A checkbox that returns `True` when ticked and `False` otherwise.
**2. In plain words:** It's a simple on/off switch, used here to show or hide the raw model output.
**3. Example to quote:** > "`show_raw = st.checkbox("Show raw output", value=False)` controls whether I print the JSON."
**Seen in:** `app.py`

## st.tabs `(third-party: streamlit)`
**1. Definition:** Creates tabbed sections from a list of names and returns one container per tab, used with `with`.
**2. In plain words:** It lets you put several views on one page without scrolling. The content you put inside each `with tab:` block appears under that tab.
**3. Example to quote:** > "`tab_single, tab_compare = st.tabs(["Single", "Compare"])` gives me two views."
**Seen in:** `app.py`

## st.columns `(third-party: streamlit)`
**1. Definition:** Creates side-by-side columns: a number for equal widths or a list of ratios such as `[2, 1]`; returns a list of column containers.
**2. In plain words:** It puts elements next to each other. You can use `with left:` or call a method on the column object directly, like `c1.metric(...)`.
**3. Example to quote:** > "`left, right = st.columns(2)` places Sentence A and Sentence B side by side."
**Seen in:** `app.py`

## Tuple unpacking `(syntax)`
**1. Definition:** Assigning the elements of a sequence to several variables in one statement, such as `a, b = st.columns(2)`; it can be nested: `(la, sa), (lb, sb) = f(a), f(b)`.
**2. In plain words:** The right side produces two things and the left side names each one. The nested form unpacks two result tuples in one line.
**3. Example to quote:** > "`(la, sa), (lb, sb) = predict(a), predict(b)` gives me both labels and both scores at once."
**Seen in:** `app.py`

## key= parameter `(third-party: streamlit)`
**1. Definition:** A unique string identifying a widget, required when two widgets would otherwise have the same label and type.
**2. In plain words:** Streamlit identifies widgets by their label and type. If two buttons are both called "Classify", it complains about a duplicate; giving each a `key` fixes it.
**3. Example to quote:** > "I gave the two text inputs `key="a"` and `key="b"` so Streamlit can tell them apart."
**Seen in:** `app.py`

## st.status `(third-party: streamlit)`
**1. Definition:** A collapsible progress container used as a context manager; `status.update(label=..., state=...)` changes its text and state.
**2. In plain words:** It's a richer loading indicator than a spinner: it shows what is happening and then marks it "complete".
**3. Example to quote:** > "I wrap the model call in `st.status("Running model...")` and update it to 'Done' when it finishes."
**Seen in:** `app.py`

## time.sleep `(Python built-in module function)`
**1. Definition:** Pauses execution for the given number of seconds.
**2. In plain words:** Used here to pretend the model is working so you can see the status box. In real code you wouldn't add artificial delays.
**3. Example to quote:** > "`time.sleep(0.3)` simulates work so the status indicator is visible."
**Seen in:** `app.py`

## st.expander `(third-party: streamlit)`
**1. Definition:** A collapsible section that hides its content until clicked.
**2. In plain words:** Good for explanations or details that most users don't need immediately.
**3. Example to quote:** > "The threshold explanation sits inside `st.expander("What does the threshold do?")`."
**Seen in:** `app.py`

## st.metric `(third-party: streamlit)`
**1. Definition:** Shows a labelled value with a small secondary line.
**2. In plain words:** Each column shows the sentence's label with its score underneath, so two results can be compared at a glance.
**3. Example to quote:** > "`c1.metric("A", la, f"{sa}")` shows sentence A's result in the left column."
**Seen in:** `app.py`

## st.success and st.warning `(third-party: streamlit)`
**1. Definition:** Coloured message boxes, green and yellow.
**2. In plain words:** Here they communicate whether the model's confidence passed the sidebar threshold.
**3. Example to quote:** > "If the score is at least the threshold I show a green success box, otherwise a yellow 'Not confident' warning."
**Seen in:** `app.py`

## st.json `(third-party: streamlit)`
**1. Definition:** Displays a Python dict or list as formatted, collapsible JSON.
**2. In plain words:** It's a quick way to show raw output when the "Show raw output" checkbox is ticked.
**3. Example to quote:** > "`st.json({"label": label, "score": score})` prints the result as JSON."
**Seen in:** `app.py`

## Threshold on confidence `(concept)`
**1. Definition:** A cut-off value below which a prediction is treated as uncertain.
**2. In plain words:** The model always returns a label, but a low score means it's unsure. A threshold lets the UI warn instead of presenting a weak guess as fact. It only changes the message here, not the model output.
**3. Example to quote:** > "With the threshold at 0.9, a 0.55 prediction shows a warning instead of a success box."
**Seen in:** `app.py`

## st.markdown with unsafe_allow_html `(third-party: streamlit)`
**1. Definition:** A flag that permits raw HTML, such as a `<style>` tag, inside Markdown content.
**2. In plain words:** It lets you inject custom CSS, but it's fragile and should never be used with text typed by users, since HTML could carry scripts.
**3. Example to quote:** > "I only use `unsafe_allow_html=True` for my own small style tag, never for user input."
**Seen in:** `app.py`

## .streamlit/config.toml `(file)`
**1. Definition:** Streamlit's configuration file, read from the folder where `streamlit run` is started and from `~/.streamlit/`; it holds sections such as `[theme]` and `[server]`.
**2. In plain words:** It customises colours, fonts and server behaviour without code changes. You must start the app from this folder for the theme to be picked up.
**3. Example to quote:** > "I ran the app from its own folder, so `config.toml` applied the green theme."
**Seen in:** `.streamlit/config.toml`

## TOML format `(syntax)`
**1. Definition:** A configuration file format with `[section]` headers and `key = value` pairs, where strings are quoted.
**2. In plain words:** It's similar to INI files and easy to read. Comments start with `#`.
**3. Example to quote:** > "`primaryColor = "#0b6e4f"` under `[theme]` sets the button colour."
```toml
[theme]
primaryColor = "#0b6e4f"
```
**Seen in:** `.streamlit/config.toml`

## Theme options `(Streamlit config)`
**1. Definition:** `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, `textColor` and `font` under `[theme]` control the app's colours and typeface.
**2. In plain words:** The primary colour affects buttons and links, the secondary background affects the sidebar and widgets. Changing them and reloading the page shows the effect.
**3. Example to quote:** > "I changed `secondaryBackgroundColor` and the sidebar background changed after a reload."
**Seen in:** `.streamlit/config.toml`

## server.headless `(Streamlit config)`
**1. Definition:** A `[server]` option that, when true, stops Streamlit opening a browser window automatically.
**2. In plain words:** Useful on servers and in Docker where there's no browser to open.
**3. Example to quote:** > "I set `headless = true` so the app doesn't try to open a browser inside a container."
**Seen in:** `.streamlit/config.toml`

## sys.path.insert and parents[1] `(standard library)`
**1. Definition:** Adds the sibling `common` folder to Python's import search path, computed relative to the script file.
**2. In plain words:** It makes `from model import predict` work regardless of where you launch the app.
**3. Example to quote:** > "I resolve `common` relative to `__file__`, not the current directory, so launching from anywhere works."
**Seen in:** `app.py`

## StreamlitAPIException `(error)`
**1. Definition:** The exception Streamlit raises when its API is misused, for example calling `set_page_config` after another Streamlit command.
**2. In plain words:** The message in the browser says exactly what's wrong. Read it before searching elsewhere.
**3. Example to quote:** > "Moving `st.set_page_config` below `st.title` produced a StreamlitAPIException telling me it must be the first call."
**Seen in:** `app.py`, BREAK IT comment
