# Terms — Streamlit basics and the rerun model

Folder: `12_streamlit_gradio/01_streamlit_basics` · Files covered: `app.py`, `WALKTHROUGH.md`
Back to module: [12_streamlit_gradio README](../README.md)

**Terms in this folder:** [Streamlit (third-party: streamlit)](#streamlit-third-party-streamlit) · [import streamlit as st (syntax)](#import-streamlit-as-st-syntax) · [Rerun model (concept)](#rerun-model-concept) · [st.title (third-party: streamlit)](#sttitle-third-party-streamlit) · [st.write (third-party: streamlit)](#stwrite-third-party-streamlit) · [st.text_input (third-party: streamlit)](#sttext_input-third-party-streamlit) · [st.button (third-party: streamlit)](#stbutton-third-party-streamlit) · [if / else on a widget value (syntax)](#if--else-on-a-widget-value-syntax) · [st.success / st.info / st.error / st.warning (third-party: streamlit)](#stsuccess--stinfo--sterror--stwarning-third-party-streamlit) · [str.upper() (Python built-in)](#strupper-python-built-in) · [st.caption (third-party: streamlit)](#stcaption-third-party-streamlit) · [f-string (syntax)](#f-string-syntax) · [Plain variables reset on rerun (concept)](#plain-variables-reset-on-rerun-concept) · [streamlit run (command)](#streamlit-run-command) · [Widget return values (concept)](#widget-return-values-concept) · [TypeError for a missing argument (error)](#typeerror-for-a-missing-argument-error)

---

## Streamlit `(third-party: streamlit)`
**1. Definition:** A Python library that turns an ordinary script into an interactive web app by calling display and widget functions such as `st.title` and `st.button`.
**2. In plain words:** You write Python from top to bottom and Streamlit shows the results in a browser, with no HTML or JavaScript. It's popular for quick ML demos. The trade-off is that the whole script re-runs on every interaction, which you must design around.
**3. Example to quote:** > "I built a text box and a button in ten lines of Streamlit and ran it with `streamlit run app.py`; it opened at localhost:8501."
**Seen in:** `app.py`

## import streamlit as st `(syntax)`
**1. Definition:** An import statement that loads the `streamlit` module and binds it to the short alias `st`.
**2. In plain words:** `as st` is just a nickname so you can write `st.title(...)` instead of `streamlit.title(...)`. It's a community convention, so everyone reads it the same way.
**3. Example to quote:** > "Every Streamlit call starts with `st.` because of the `import streamlit as st` alias on the first line."
**Seen in:** `app.py`

## Rerun model `(concept)`
**1. Definition:** Streamlit's execution model in which the entire script is executed again from top to bottom each time the user interacts with a widget.
**2. In plain words:** There are no callbacks to register. A widget function simply returns its current value, and when the user changes something, the whole script runs again with the new value. This is simple, but it means ordinary variables are reset on every run, and expensive work must be cached.
**3. Example to quote:** > "My `count` variable always printed 1, because Streamlit re-runs the entire file on every click and `count = 0` runs again."
**Seen in:** `app.py` docstring, last block

## st.title `(third-party: streamlit)`
**1. Definition:** Displays the given text as a large page heading.
**2. In plain words:** It's the page's main title. Use `st.header` and `st.subheader` for smaller headings.
**3. Example to quote:** > "`st.title("Hello, Streamlit")` draws the big heading at the top of my page."
**Seen in:** `app.py`

## st.write `(third-party: streamlit)`
**1. Definition:** A general-purpose display function that renders strings, numbers, dictionaries, DataFrames and many other objects.
**2. In plain words:** It's the "just show this" function. It picks a sensible way to display whatever you give it, which makes it handy for quick debugging of values.
**3. Example to quote:** > "I use `st.write(my_dict)` to inspect a dictionary right in the page."
**Seen in:** `app.py`

## st.text_input `(third-party: streamlit)`
**1. Definition:** Draws a single-line text box and returns the text currently in it as a string; takes `label`, `value` (default) and optional `key`.
**2. In plain words:** The returned value is the whole interface to the user's input. Because of the rerun model, each time the user types and presses Enter the script runs again and `name` holds the new text.
**3. Example to quote:** > "`name = st.text_input("Your text", value="hello world")` gives me the current text, starting with 'hello world'."
```python
name = st.text_input("Your text", value="hello world")
```
**Seen in:** `app.py`

## st.button `(third-party: streamlit)`
**1. Definition:** Draws a button and returns `True` only during the rerun triggered by clicking it, otherwise `False`.
**2. In plain words:** A button doesn't "remember" being clicked. On the click, the script reruns and `st.button` is True for that one run; on the next rerun it's False again. That's why results that must persist belong in session state (step 02).
**3. Example to quote:** > "`if st.button("Shout"):` runs its body only on the rerun caused by the click."
**Seen in:** `app.py`

## if / else on a widget value `(syntax)`
**1. Definition:** A conditional statement that chooses which branch to execute based on a boolean, here the value returned by `st.button`.
**2. In plain words:** The `if` branch shows the result after a click, and the `else` branch shows a waiting message before any click. This is how you do "event handling" in Streamlit: ordinary if statements.
**3. Example to quote:** > "Before the click I see 'Waiting for a click...'; after the click the `if` branch shows the uppercase text."
**Seen in:** `app.py`

## st.success / st.info / st.error / st.warning `(third-party: streamlit)`
**1. Definition:** Functions that display a message in a coloured box: green, blue, red and yellow respectively.
**2. In plain words:** They communicate status to the user. Use `success` for good outcomes, `info` for hints, `warning` for things to check and `error` for failures.
**3. Example to quote:** > "I show the result with `st.success(name.upper())` and the placeholder with `st.info`."
**Seen in:** `app.py`

## str.upper() `(Python built-in)`
**1. Definition:** A string method returning a copy of the string with all letters converted to uppercase.
**2. In plain words:** It doesn't change the original string; it returns a new one. It's the "work" the app does when you press Shout.
**3. Example to quote:** > "`"hello world".upper()` returns `'HELLO WORLD'`."
**Seen in:** `app.py`

## st.caption `(third-party: streamlit)`
**1. Definition:** Displays small, muted text, typically used for notes or footnotes.
**2. In plain words:** It's for side remarks. In this file it shows the counter value to demonstrate that plain variables reset on rerun.
**3. Example to quote:** > "`st.caption(f"plain variable count = {count}")` always shows 1."
**Seen in:** `app.py`

## f-string `(syntax)`
**1. Definition:** A string literal prefixed with `f` in which expressions inside `{}` are evaluated and inserted.
**2. In plain words:** It's the easiest way to put values into text. `f"n={1+1}"` gives `n=2`. You can put variables or small expressions between the braces.
**3. Example to quote:** > "I build the caption with `f"plain variable count = {count}"`."
**Seen in:** `app.py`

## Plain variables reset on rerun `(concept)`
**1. Definition:** Because the script re-executes from the top, ordinary Python variables are recreated with their initial values on every run, losing any change from the previous run.
**2. In plain words:** If you write `count = 0; count += 1`, you always see 1. To keep something between interactions you need `st.session_state`, which you meet in step 02.
**3. Example to quote:** > "The counter never goes above 1 because the script restarts; I'd need `st.session_state` to remember it."
**Seen in:** `app.py`, final block

## streamlit run `(command)`
**1. Definition:** The CLI command that starts Streamlit's server and serves the given script, by default at `http://localhost:8501`.
**2. In plain words:** You don't run a Streamlit file with plain `python`. `streamlit run app.py` starts a server and opens the browser; Ctrl+C in the terminal stops it. When you edit the file, the page offers "Rerun" or "Always rerun".
**3. Example to quote:** > "I launched it with `.venv/bin/streamlit run 12_streamlit_gradio/01_streamlit_basics/app.py` and opened localhost:8501."
**Seen in:** `app.py` docstring

## Widget return values `(concept)`
**1. Definition:** Streamlit widgets are ordinary function calls whose return value is the widget's current state.
**2. In plain words:** There is no "get value" step. Because each interaction reruns the script, the returned value is always fresh, so you just use it like any variable.
**3. Example to quote:** > "I don't register a callback; I read `name` directly after `st.text_input`."
**Seen in:** `app.py`

## TypeError for a missing argument `(error)`
**1. Definition:** A Python exception raised when a function is called with the wrong number or type of arguments, here `st.text_input()` without the required label.
**2. In plain words:** Streamlit shows Python errors in the browser with the traceback. Read the last line for what's wrong and the lines above for where.
**3. Example to quote:** > "Calling `st.text_input()` with no label showed a TypeError in the browser telling me the `label` argument was missing."
**Seen in:** `app.py`, BREAK IT comment
