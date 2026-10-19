# Terms — Gradio basics: a function becomes a web app

Folder: `12_streamlit_gradio/05_gradio_basics` · Files covered: `app.py`, `WALKTHROUGH.md` (uses `../common/model.py`)
Back to module: [12_streamlit_gradio README](../README.md)

**Terms in this folder:** [Gradio (third-party: gradio)](#gradio-third-party-gradio) · [import gradio as gr (syntax)](#import-gradio-as-gr-syntax) · [Callback / event handler (concept)](#callback--event-handler-concept) · [Gradio vs Streamlit (concept)](#gradio-vs-streamlit-concept) · [gr.Interface (third-party: gradio)](#grinterface-third-party-gradio) · [gr.Textbox (third-party: gradio)](#grtextbox-third-party-gradio) · [gr.Label (third-party: gradio)](#grlabel-third-party-gradio) · [gr.Error (third-party: gradio)](#grerror-third-party-gradio) · [examples parameter (third-party: gradio)](#examples-parameter-third-party-gradio) · [flagging_mode (third-party: gradio)](#flagging_mode-third-party-gradio) · [Interface.launch (third-party: gradio)](#interfacelaunch-third-party-gradio) · [if __name__ == "__main__" (syntax)](#if-__name__--__main__-syntax) · [Type hints on a function (syntax)](#type-hints-on-a-function-syntax) · [Dict comprehension (syntax)](#dict-comprehension-syntax) · [zip (Python built-in)](#zip-python-built-in) · [predict_proba (third-party: scikit-learn)](#predict_proba-third-party-scikit-learn) · [model.classes_ (third-party: scikit-learn)](#modelclasses_-third-party-scikit-learn) · [load_model (load once) (concept)](#load_model-load-once-concept) · [Keyword arguments (syntax)](#keyword-arguments-syntax) · [Raising a Gradio error vs a traceback (concept)](#raising-a-gradio-error-vs-a-traceback-concept)

---

## Gradio `(third-party: gradio)`
**1. Definition:** A Python library for building web UIs around functions, especially machine-learning demos, by declaring input and output components.
**2. In plain words:** You write a normal function and tell Gradio what kind of input widget and output display to use; Gradio builds the page and calls your function when the user submits. It also gives you a built-in API and easy sharing.
**3. Example to quote:** > "I turned my `classify` function into a web app with one `gr.Interface` call."
**Seen in:** `app.py`

## import gradio as gr `(syntax)`
**1. Definition:** Imports the `gradio` module under the conventional alias `gr`.
**2. In plain words:** The short alias makes components read as `gr.Textbox`, `gr.Label` and so on.
**3. Example to quote:** > "Everything from Gradio is accessed through `gr.`, thanks to the alias on the import line."
**Seen in:** `app.py`

## Callback / event handler `(concept)`
**1. Definition:** A function that the framework calls automatically when a user event, such as pressing Submit, occurs.
**2. In plain words:** In Gradio you don't read widget values yourself. You register `classify` and Gradio calls it with the input values, then displays what it returns. Streamlit does the opposite: it re-runs the script and you read the widget values.
**3. Example to quote:** > "When the user presses Submit, Gradio calls `classify(text)` and shows the returned dictionary in the Label."
**Seen in:** `app.py` docstring

## Gradio vs Streamlit `(concept)`
**1. Definition:** Streamlit re-runs the whole script on every interaction and widgets return their values; Gradio calls a specific function with the input components' values and shows the output components.
**2. In plain words:** Gradio has no reruns, so there's less to worry about with plain variables resetting. Streamlit gives you more layout freedom in plain script style. Both are fine for ML demos.
**3. Example to quote:** > "In Streamlit I check `if st.button(...)`; in Gradio I pass `fn=classify` and Gradio does the calling."
**Seen in:** `app.py` docstring

## gr.Interface `(third-party: gradio)`
**1. Definition:** A high-level class that builds a complete UI from `fn`, `inputs` and `outputs` (plus options like `examples`, `title`, `description`).
**2. In plain words:** It's Gradio's one-liner. One input component per function parameter and one output component per returned value. For custom layouts or several events, use `gr.Blocks` instead (step 06).
**3. Example to quote:** > "`gr.Interface(fn=classify, inputs=gr.Textbox(...), outputs=gr.Label(...))` is my whole UI."
**Seen in:** `app.py`

## gr.Textbox `(third-party: gradio)`
**1. Definition:** A text input/output component; options include `label`, `lines` and `placeholder`.
**2. In plain words:** It's the box where users type. Its value is passed to your function as a string.
**3. Example to quote:** > "`gr.Textbox(label="Sentence", lines=2, placeholder="Type here...")` gives a two-line input."
**Seen in:** `app.py`

## gr.Label `(third-party: gradio)`
**1. Definition:** An output component for classification results that expects a dictionary mapping class names to probabilities; `num_top_classes` limits how many are shown.
**2. In plain words:** It shows the top predictions as bars with percentages. If your function returns a string instead of a dict, Gradio reports an error.
**3. Example to quote:** > "My function returns `{'positive': 0.55, 'negative': 0.45}` and `gr.Label` draws bars for both."
**Seen in:** `app.py`

## gr.Error `(third-party: gradio)`
**1. Definition:** An exception class that, when raised in a callback, shows its message to the user as a toast instead of a stack trace.
**2. In plain words:** It's how you turn "the input is bad" into a polite message. Real bugs still produce tracebacks in the terminal, so you can debug them.
**3. Example to quote:** > "For empty text I `raise gr.Error("Please enter some text.")` and the user sees a red toast."
**Seen in:** `app.py`

## examples parameter `(third-party: gradio)`
**1. Definition:** A list of rows, each row holding values for all inputs, shown as clickable samples under the interface.
**2. In plain words:** Examples let users try the app with one click. Because each row must have a value per input, `[["I love this"]]` is a list of rows with one value each.
**3. Example to quote:** > "I added `examples=[["I love this"], ["This is awful"]]` so people can test it instantly."
**Seen in:** `app.py`

## flagging_mode `(third-party: gradio)`
**1. Definition:** An Interface option controlling the "Flag" button used for collecting feedback; `"never"` hides it.
**2. In plain words:** Flagging saves problematic examples for later review. For a simple demo it's usually just clutter.
**3. Example to quote:** > "I set `flagging_mode="never"` to hide the Flag button in this demo."
**Seen in:** `app.py`

## Interface.launch `(third-party: gradio)`
**1. Definition:** Starts a local web server for the interface, by default on port 7860; accepts `server_port`, `share` and `inbrowser`, and blocks until stopped.
**2. In plain words:** Calling `demo.launch()` is what makes the page appear. If the port is busy, choose another with `server_port`.
**3. Example to quote:** > "Running the script printed 'Running on local URL: http://127.0.0.1:7860'."
**Seen in:** `app.py`

## if __name__ == "__main__" `(syntax)`
**1. Definition:** A guard that is true only when the file is executed directly, not when it is imported.
**2. In plain words:** It lets tests import `classify` from this file without starting the web server.
**3. Example to quote:** > "My tests import `classify` safely, because `demo.launch()` is under the main guard."
**Seen in:** `app.py`

## Type hints on a function `(syntax)`
**1. Definition:** Annotations like `text: str` and `-> dict` that document expected types; Python doesn't enforce them at run time.
**2. In plain words:** They help readers and tools. `classify(text: str) -> dict` says a string goes in and a dictionary comes out.
**3. Example to quote:** > "The signature `def classify(text: str) -> dict:` tells me the Label needs a dictionary."
**Seen in:** `app.py`

## Dict comprehension `(syntax)`
**1. Definition:** `{k: v for k, v in iterable}` builds a dictionary from an iterable of pairs.
**2. In plain words:** It's a one-line loop that makes a dict. Here it pairs each class name with its probability.
**3. Example to quote:** > "`{str(c): float(p) for c, p in zip(model.classes_, probs)}` produces `{'negative': 0.45, 'positive': 0.55}`."
**Seen in:** `app.py`

## zip `(Python built-in)`
**1. Definition:** Combines several iterables element by element into tuples.
**2. In plain words:** It lines up class names with their probabilities so you can loop over both together.
**3. Example to quote:** > "`zip(model.classes_, probs)` gives pairs like `('positive', 0.55)`."
**Seen in:** `app.py`

## predict_proba `(third-party: scikit-learn)`
**1. Definition:** A classifier method that returns, for each input, the probability of every class as a 2-D array.
**2. In plain words:** `predict` gives only the winning label; `predict_proba` gives the confidence for each label, which is what `gr.Label` displays. The columns follow the order of `model.classes_`.
**3. Example to quote:** > "`model.predict_proba([text])[0]` is one row: the probabilities for negative and positive."
**Seen in:** `app.py`

## model.classes_ `(third-party: scikit-learn)`
**1. Definition:** The array of class labels the fitted classifier knows, in the same order as the columns of `predict_proba`.
**2. In plain words:** It tells you which probability belongs to which label.
**3. Example to quote:** > "`model.classes_` is `['negative', 'positive']`, matching the probability columns."
**Seen in:** `app.py`

## load_model (load once) `(concept)`
**1. Definition:** A helper that trains or loads the shared scikit-learn pipeline the first time it's called and returns the cached object afterwards.
**2. In plain words:** The model is built once and reused for every request, which is the same pattern as a web API loading its model at startup.
**3. Example to quote:** > "`classify` calls `load_model()` and it only trains the first time."
**Seen in:** `app.py`

## Keyword arguments `(syntax)`
**1. Definition:** Passing arguments by name, like `fn=classify, inputs=...`, rather than by position.
**2. In plain words:** Named arguments make long calls readable and order-independent. Gradio's constructors rely on them.
**3. Example to quote:** > "I call `gr.Interface(fn=..., inputs=..., outputs=..., title=...)` with named arguments."
**Seen in:** `app.py`

## Raising a Gradio error vs a traceback `(concept)`
**1. Definition:** `gr.Error` produces a user-friendly toast; other uncaught exceptions show a generic error in the UI and the full traceback in the terminal.
**2. In plain words:** Use `gr.Error` for expected user mistakes. For unexpected bugs, look in the terminal where the app runs.
**3. Example to quote:** > "When my function returned a string instead of a dict, the UI showed an error and the terminal had the traceback."
**Seen in:** `app.py`, BREAK IT comment
