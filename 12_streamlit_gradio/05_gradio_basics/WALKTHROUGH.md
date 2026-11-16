# Step 05 - Gradio basics (Interface and callbacks)

## The story so far
[Step 04](../04_streamlit_layout_style/WALKTHROUGH.md) finished the Streamlit side. Gradio solves the same pain with a different idea.

## The pain point
Sometimes all you have is a function `text -> label` and ten minutes before a demo. Even Streamlit means thinking about reruns, state and layout.

## The spark
Hand Gradio the function and list its input and output components. `gr.Interface` builds the page, calls your function on submit, and `gr.Error` turns a failure into a red toast instead of a crash.

## Goal
Expose a prediction function as a web UI and understand Gradio's input -> function -> output model.

## Run it
```bash
cd 12_streamlit_gradio/05_gradio_basics
../../.venv/bin/python app.py        # http://127.0.0.1:7860
```
Setup elsewhere: `pip install gradio`. Verified: `classify("I love this")` returns `{'negative': ..., 'positive': ...}` (see ../test_apps.py) and `classify("   ")` raises `gradio.Error`. Interface construction runs without error.

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| `gr.Interface` vs `gr.Blocks` | high-level vs low-level API | No | Interface = one function wrapped with input and output components; Blocks = custom layout and several events |
| Callback vs event handler vs `fn` | function Gradio calls | Same | All are the Python function that runs when the user triggers an event |
| Component vs widget | `gr.Textbox`, `st.text_input` | Similar | Gradio components are objects you wire to a function; Streamlit widgets are function calls that return values |
| `launch()` vs `share=True` | local vs public link | No | `launch()` serves on 127.0.0.1:7860; `share=True` also opens a temporary public tunnel |
| `gr.Label` vs `gr.Textbox` output | classification chart vs text | No | `gr.Label` expects `{class: probability}` and draws bars |

## Code, step by step

### 1. Docstring: Gradio vs Streamlit  (`app.py` lines 1–13)
```python
"""
GOAL: Wrap a Python function in a web UI with gr.Interface - Gradio's one-liner approach.

RUN (from 12_streamlit_gradio/05_gradio_basics):  ../../.venv/bin/python app.py
EXPECTED OUTPUT: terminal prints "* Running on local URL:  http://127.0.0.1:7860"; the page has a text box,
                 a Submit button and a label chart with class probabilities. Ctrl+C stops it.
NEEDS: gradio, scikit-learn (../common/model.py).

GRADIO vs STREAMLIT (the core difference):
  Streamlit: the script reruns on every interaction; you read widget return values.
  Gradio   : you write a normal FUNCTION and tell Gradio which components are its inputs and outputs;
             Gradio calls it (a CALLBACK / event handler) when the user presses Submit. No reruns.
"""
```
- **In simple words:** Explains the core difference in one box.
- **Exactly:** Streamlit re-runs the file and you read widget values. Gradio takes a normal function and calls it as a callback when the user submits.
- **Why it is here:** Choosing the right tool: a model demo in 10 lines is Gradio; a custom dashboard is Streamlit.

### 2. Imports and shared model  (`app.py` lines 14–20)
```python
import sys
from pathlib import Path

import gradio as gr   # standard alias

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import load_model, predict  # noqa: E402
```
- **In simple words:** Load Gradio and the shared model.
- **Exactly:** `import gradio as gr` is the standard alias. `sys.path` insert as in step 02.
- **Why it is here:** Reuses the same sentiment model.

### 3. The callback function  (`app.py` lines 23–33)
```python
def classify(text: str) -> dict:
    """The callback. Argument = value of the input component; return value = value for the output component.

    gr.Label expects a dict {class_name: probability}.
    """
    if not text or not text.strip():
        # gr.Error shows a red toast to the user instead of a stack trace page.
        raise gr.Error("Please enter some text.")
    model = load_model()
    probs = model.predict_proba([text])[0]                        # array of probabilities
    return {str(c): float(p) for c, p in zip(model.classes_, probs)}   # dict comprehension
```
- **In simple words:** Takes a sentence and returns class probabilities.
- **Exactly:** The argument is the value of the input component (`str`). Blank text raises `gr.Error`, shown as a red toast. `predict_proba([text])[0]` returns an array; the comprehension builds `{str(c): float(p)}`. Real output for "I love it": `{'negative': 0.452, 'positive': 0.548}`. `float(p)` converts numpy types so Gradio can serialise them.
- **Why it is here:** `gr.Label` needs a dict, so the return type matters.

### 4. Build the Interface  (`app.py` lines 36–44)
```python
demo = gr.Interface(
    fn=classify,                                                   # the function to expose
    inputs=gr.Textbox(label="Sentence", lines=2, placeholder="Type here..."),   # one input per fn parameter
    outputs=gr.Label(num_top_classes=2, label="Sentiment"),        # one output per returned value
    examples=[["I love this"], ["This is awful"]],                 # clickable sample inputs
    title="Sentiment (Gradio Interface)",
    description="A function becomes a web app.",
    flagging_mode="never",                                         # hide the 'Flag' button
)
```
- **In simple words:** Tell Gradio which function to expose and what its input and output look like.
- **Exactly:** `inputs` gives one component per function parameter; `outputs` one per return value. `num_top_classes=2` shows both classes. `examples` adds clickable samples. `flagging_mode="never"` hides the Flag button (older versions used `allow_flagging`).
- **Why it is here:** This one call produces the whole web page and the API endpoint.

### 5. Launch only when run as a script  (`app.py` lines 46–47)
```python
if __name__ == "__main__":     # only start the server when run as a script, not when imported (tests import `classify`)
    demo.launch()              # server_port=7860 by default; use demo.launch(server_port=7861) if busy
```
- **In simple words:** Start the server when you run the file, not when something imports it.
- **Exactly:** `demo.launch()` serves on port 7860 by default. The guard lets tests import `classify` without a server.
- **Why it is here:** Without the guard, importing the file would start a server and block.

### 6. Exercises as comments  (`app.py` lines 49–51)
```python
# TRY IT: change num_top_classes=2 to 1.
# TRY IT: add a second input gr.Slider(0, 1, value=0.5) and a second parameter `threshold` to classify.
# BREAK IT: make `classify` return a string instead of a dict -> Gradio shows an error toast; read the terminal traceback.
```
- **In simple words:** Ideas to change.
- **Exactly:** BREAK IT: return a string instead of a dict; read the terminal traceback.
- **Why it is here:** See how Gradio reports type mismatch.


## Syntax explained
| Syntax | Meaning | Tiny example |
|---|---|---|
| `def classify(text: str) -> dict:` | function with type hints (documentation only) | |
| `{k: v for k, v in zip(a, b)}` | dict comprehension | `{c: 1 for c in "ab"}` |
| `raise gr.Error("msg")` | raise an exception that Gradio displays nicely | |
| `if __name__ == "__main__":` | run only when executed directly | |
| keyword args `fn=..., inputs=...` | named arguments | |
| `[["I love this"], ["x"]]` | list of rows; each row = values for all inputs | |

## Libraries & functions
| Name | Built-in / 3rd-party | Does | Important args | Returns |
|---|---|---|---|---|
| `gr.Interface` | gradio | build UI around one function | `fn, inputs, outputs, examples, title` | Interface (a Blocks) |
| `gr.Textbox` | gradio | text component | `label, lines, placeholder` | component |
| `gr.Label` | gradio | classification output | `num_top_classes` | component (expects dict) |
| `gr.Error` | gradio | user-visible exception | message | exception class |
| `.launch()` | gradio | start local server | `server_port, share, inbrowser` | (blocks until stopped) |
| `predict_proba` | scikit-learn | class probabilities | list of texts | 2D array |

## Execution trace
User types text and presses Submit -> browser sends value to the Gradio server -> server calls `classify(text)` -> returns `{"negative": 0.45, "positive": 0.55}` -> `gr.Label` renders bars. Empty text -> `gr.Error` -> red toast, app keeps running.

## Experiments
1. `num_top_classes=1`.
2. Add a Slider input and a parameter.
3. Click an example: it fills the box (and runs if `cache_examples`).
4. Return a wrongly typed value to see the error path.

## Debugging practice
Mismatch between number of inputs and function parameters -> error mentions expected vs received arguments. Count: one component per parameter, one per returned value.

## Rebuild checklist
- [ ] Interface for a function `(text) -> text` from memory.
- [ ] Explain callback vs rerun model.

## The new problem (cliffhanger)
An `Interface` is one function in, one result out. Real demos need several tabs, per-session history, file uploads, progress bars and an API for other programs. Next: [step 06, Blocks, API and sharing](../06_gradio_blocks_api_share/WALKTHROUGH.md).

## Say it aloud
> **Say it aloud (interview box):** Gradio wraps a function: I list the input and output components and Gradio calls my function when the user submits. It is the fastest way to demo a model and can expose an API automatically. The trade-off is less control over complex layouts and multi-step state than Streamlit or a custom frontend, though Blocks closes much of that gap.
