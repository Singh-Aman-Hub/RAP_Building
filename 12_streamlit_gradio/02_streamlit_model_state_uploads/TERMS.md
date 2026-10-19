# Terms — Cached model, session state, forms and uploads

Folder: `12_streamlit_gradio/02_streamlit_model_state_uploads` · Files covered: `app.py`, `sample_sentences.txt`, `WALKTHROUGH.md` (uses `../common/model.py`)
Back to module: [12_streamlit_gradio README](../README.md)

**Terms in this folder:** [sys.path.insert (standard library)](#syspathinsert-standard-library) · [Path(__file__).resolve().parents[1] (standard library)](#path__file__resolveparents1-standard-library) · [noqa: E402 (comment)](#noqa-e402-comment) · [st.cache_resource (third-party: streamlit)](#stcache_resource-third-party-streamlit) · [Decorator (syntax)](#decorator-syntax) · [st.session_state (third-party: streamlit)](#stsession_state-third-party-streamlit) · [st.form and st.form_submit_button (third-party: streamlit)](#stform-and-stform_submit_button-third-party-streamlit) · [with statement (context manager) (syntax)](#with-statement-context-manager-syntax) · [st.metric (third-party: streamlit)](#stmetric-third-party-streamlit) · [try / except ValueError (syntax)](#try--except-valueerror-syntax) · [st.error (third-party: streamlit)](#sterror-third-party-streamlit) · [f-string (syntax)](#f-string-syntax) · [list.append (Python built-in)](#listappend-python-built-in) · [st.subheader, st.divider, st.table (third-party: streamlit)](#stsubheader-stdivider-sttable-third-party-streamlit) · [st.rerun (third-party: streamlit)](#strerun-third-party-streamlit) · [st.file_uploader (third-party: streamlit)](#stfile_uploader-third-party-streamlit) · [UploadedFile.getvalue and bytes.decode (third-party/built-in)](#uploadedfilegetvalue-and-bytesdecode-third-partybuilt-in) · [str.splitlines (Python built-in)](#strsplitlines-python-built-in) · [List comprehension with if (syntax)](#list-comprehension-with-if-syntax) · [dict(zip(...)) and {**a, **b} (syntax)](#dictzip-and-a-b-syntax) · [st.dataframe (third-party: streamlit)](#stdataframe-third-party-streamlit) · [Model loaded once (load-once pattern) (concept)](#model-loaded-once-load-once-pattern-concept) · [AttributeError from missing session key (error)](#attributeerror-from-missing-session-key-error) · [Sample sentences file (file)](#sample-sentences-file-file)

---

## sys.path.insert `(standard library)`
**1. Definition:** `sys.path` is the list of directories Python searches when importing; `insert(0, p)` adds directory `p` at the front so modules there become importable.
**2. In plain words:** The shared `model.py` lives in a sibling folder, so Python wouldn't find it by default. Adding that folder to the front of the search list lets `from model import ...` work no matter where `streamlit run` is started.
**3. Example to quote:** > "I insert the `common` folder into `sys.path` so `from model import predict` works from any working directory."
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
```
**Seen in:** `app.py`

## Path(__file__).resolve().parents[1] `(standard library)`
**1. Definition:** `__file__` is the current script's path; `resolve()` makes it absolute and `.parents[1]` is the grandparent directory (two levels up from the file).
**2. In plain words:** It computes a path relative to the script itself instead of relying on where you launched it. `parents[0]` is the file's own folder, `parents[1]` is the folder above that.
**3. Example to quote:** > "`parents[1]` of `02_.../app.py` is the module folder `12_streamlit_gradio`, and I append `common` to find `model.py`."
**Seen in:** `app.py`

## noqa: E402 `(comment)`
**1. Definition:** A marker telling the linter to ignore the rule E402, "module level import not at top of file", on that line.
**2. In plain words:** Imports normally belong at the top, but here the import must come after `sys.path` is changed. The comment documents that this order is deliberate.
**3. Example to quote:** > "I put `# noqa: E402` on the `from model import ...` line because the path change has to happen first."
**Seen in:** `app.py`

## st.cache_resource `(third-party: streamlit)`
**1. Definition:** A decorator that runs the function once per server process and returns the same stored object on later calls, shared by all reruns and users; options include `ttl` and `show_spinner`.
**2. In plain words:** Loading a model on every rerun would be painfully slow. Cache the loader and the model is created once. Use it for heavy shared objects such as models and database connections; `st.cache_data` is for plain data.
**3. Example to quote:** > "When I removed `@st.cache_resource`, 'Loading model...' appeared on every click; with it, only the first time."
```python
@st.cache_resource
def get_model():
    return load_model()
```
**Seen in:** `app.py`

## Decorator `(syntax)`
**1. Definition:** A function applied with `@name` above a function definition to modify or register the function it decorates.
**2. In plain words:** `@st.cache_resource` wraps `get_model` so that repeat calls return the stored result. The function body is written normally.
**3. Example to quote:** > "The `@` line above `def get_model()` adds caching without changing the function's code."
**Seen in:** `app.py`

## st.session_state `(third-party: streamlit)`
**1. Definition:** A dictionary-like object, with attribute access, that persists values across reruns for one browser session.
**2. In plain words:** Since plain variables reset on every rerun, session state is where you store things that must survive, like a history list. Always initialise a key once with an `if "key" not in st.session_state` check, otherwise you get an AttributeError on the first run.
**3. Example to quote:** > "The prediction history survives each click because I keep it in `st.session_state.history`."
```python
if "history" not in st.session_state:
    st.session_state.history = []
```
**Seen in:** `app.py`

## st.form and st.form_submit_button `(third-party: streamlit)`
**1. Definition:** `st.form(key)` groups widgets so that nothing is processed until its submit button is pressed; `st.form_submit_button` returns True on that submission.
**2. In plain words:** Without a form, every keystroke in a text box can trigger a rerun. A form batches the inputs and runs your code only once, on submit, which is cleaner and cheaper.
**3. Example to quote:** > "I wrapped the text box and Predict button in `st.form` so typing doesn't rerun the app, only submitting does."
**Seen in:** `app.py`

## with statement (context manager) `(syntax)`
**1. Definition:** `with obj:` runs a block inside the context managed by `obj`, performing set-up and tear-down automatically.
**2. In plain words:** In Streamlit, `with st.form("predict_form"):` means "everything indented below belongs to this form". It's the same keyword you use for files, but here it controls where widgets are placed.
**3. Example to quote:** > "The widgets indented under `with st.form(...)` are part of that form."
**Seen in:** `app.py`

## st.metric `(third-party: streamlit)`
**1. Definition:** Displays a labelled value with an optional change indicator (`label`, `value`, `delta`).
**2. In plain words:** It shows a big KPI-style number or word. Here it presents the predicted label with the confidence underneath.
**3. Example to quote:** > "`st.metric("Prediction", label, f"confidence {score}")` shows the label large with the score under it."
**Seen in:** `app.py`

## try / except ValueError `(syntax)`
**1. Definition:** Runs a block and, if a `ValueError` is raised, executes the handler instead of crashing.
**2. In plain words:** `predict` raises ValueError for empty text. Catching it lets the UI show a friendly message while the app stays alive.
**3. Example to quote:** > "Submitting an empty sentence shows 'Invalid input' in a red box instead of a crash page."
**Seen in:** `app.py`

## st.error `(third-party: streamlit)`
**1. Definition:** Displays a message in a red error box.
**2. In plain words:** It's how you report failures to the user without showing a traceback.
**3. Example to quote:** > "I catch `ValueError` and call `st.error(f"Invalid input: {exc}")`."
**Seen in:** `app.py`

## f-string `(syntax)`
**1. Definition:** A string prefixed with `f` that evaluates expressions inside `{}`.
**2. In plain words:** Used to build the confidence caption and error message from variables.
**3. Example to quote:** > "`f"confidence {score}"` becomes 'confidence 0.548'."
**Seen in:** `app.py`

## list.append `(Python built-in)`
**1. Definition:** Adds one item to the end of a list in place.
**2. In plain words:** Each prediction is stored as a tuple `(text, label, score)` and appended to the history list kept in session state.
**3. Example to quote:** > "`st.session_state.history.append((text, label, score))` grows the history by one row."
**Seen in:** `app.py`

## st.subheader, st.divider, st.table `(third-party: streamlit)`
**1. Definition:** `st.subheader` shows a mid-sized heading, `st.divider` a horizontal line, and `st.table` a static table from rows.
**2. In plain words:** They structure the page. A list of tuples becomes a simple fixed table, good for a short history.
**3. Example to quote:** > "I show the history with `st.table(st.session_state.history)` under a subheader."
**Seen in:** `app.py`

## st.rerun `(third-party: streamlit)`
**1. Definition:** Stops the current run and immediately restarts the script from the top.
**2. In plain words:** After clearing the history, the page was already drawn with old data. Calling `st.rerun()` redraws it at once with the empty list.
**3. Example to quote:** > "After `st.session_state.history = []` I call `st.rerun()` so the table disappears immediately."
**Seen in:** `app.py`

## st.file_uploader `(third-party: streamlit)`
**1. Definition:** Draws a file-upload widget; returns `None` until a file is chosen, then an `UploadedFile`; options include `type=[...]` and `accept_multiple_files`.
**2. In plain words:** It lets users upload files into the app. Always check for `None` because nothing is uploaded on the first run.
**3. Example to quote:** > "`st.file_uploader("Choose a text file", type=["txt"])` only accepts `.txt` files."
**Seen in:** `app.py`

## UploadedFile.getvalue and bytes.decode `(third-party/built-in)`
**1. Definition:** `getvalue()` returns the uploaded file's raw bytes; `bytes.decode("utf-8")` converts bytes to a string.
**2. In plain words:** Uploaded files arrive as bytes, not text. Decoding with UTF-8 gives you a normal string you can split into lines.
**3. Example to quote:** > "`uploaded.getvalue().decode("utf-8")` turns the uploaded file into text."
**Seen in:** `app.py`

## str.splitlines `(Python built-in)`
**1. Definition:** Splits a string at line boundaries and returns a list of lines without the newline characters.
**2. In plain words:** One sentence per line in the file becomes one list element each, ready to classify.
**3. Example to quote:** > "`"a\nb".splitlines()` returns `['a', 'b']`."
**Seen in:** `app.py`

## List comprehension with if `(syntax)`
**1. Definition:** `[expression for item in iterable if condition]` builds a new list from the items that pass the condition.
**2. In plain words:** It's a compact loop. Here it makes one result row per non-blank line, skipping empty lines.
**3. Example to quote:** > "`[... for ln in lines if ln.strip()]` builds rows only for lines that aren't blank."
**Seen in:** `app.py`

## dict(zip(...)) and {**a, **b} `(syntax)`
**1. Definition:** `zip` pairs elements of two sequences; `dict(zip(keys, values))` builds a dictionary from them; `{**a, **b}` merges dictionaries.
**2. In plain words:** The code pairs `("label", "score")` with the prediction tuple and merges that with `{"text": ln}`, giving one dictionary per row.
**3. Example to quote:** > "`dict(zip("ab", (1, 2)))` gives `{'a': 1, 'b': 2}`."
**Seen in:** `app.py`

## st.dataframe `(third-party: streamlit)`
**1. Definition:** Displays an interactive, sortable table from a DataFrame or a list of dictionaries.
**2. In plain words:** It's the interactive cousin of `st.table`, which is good for larger result sets such as batch predictions.
**3. Example to quote:** > "`st.dataframe(rows)` shows a sortable table with text, label and score for each uploaded line."
**Seen in:** `app.py`

## Model loaded once (load-once pattern) `(concept)`
**1. Definition:** Creating an expensive object a single time and reusing it for every request instead of rebuilding it each time.
**2. In plain words:** Training or loading a model takes time. `st.cache_resource` implements the same idea that a web API uses when it loads its model at startup.
**3. Example to quote:** > "The 'Loading model...' message appears only on the first run, which proves the cache is working."
**Seen in:** `app.py`, `get_model`

## AttributeError from missing session key `(error)`
**1. Definition:** An exception raised when code accesses an attribute that doesn't exist, here `st.session_state.history` before it was initialised.
**2. In plain words:** If you delete the `if "history" not in st.session_state` block, the first run fails because the key isn't there yet. Always initialise state before reading it.
**3. Example to quote:** > "Removing the initialisation block caused an AttributeError on the very first run."
**Seen in:** `app.py`, BREAK IT comment

## Sample sentences file `(file)`
**1. Definition:** A plain text file with one sentence per line, used to test the batch upload.
**2. In plain words:** It gives you ready-made input for the uploader so you can see the prediction table without writing your own file.
**3. Example to quote:** > "I uploaded `sample_sentences.txt` and got four rows, each with a label and a score."
**Seen in:** `sample_sentences.txt`
