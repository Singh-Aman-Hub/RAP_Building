# Step 05 - Gradio basics (Interface and callbacks)

## Goal
Expose a prediction function as a web UI and understand Gradio's input -> function -> output model.

## Run it
```bash
cd 12_streamlit_gradio/05_gradio_basics
../../.venv/bin/python app.py        # http://127.0.0.1:7860
```
Setup elsewhere: `pip install gradio`. Verified: `classify("I love this")` returns `{'negative': ..., 'positive': ...}` (see ../test_apps.py) and `classify("   ")` raises `gradio.Error`. Interface construction runs without error.

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

## Say it aloud
"Gradio wraps a function: I list input and output components and Gradio calls my function when the user submits. It is the fastest way to demo a model and can expose an API automatically. Trade-off: less control over complex layouts and multi-step state than Streamlit or a custom frontend, though Blocks (next step) closes much of that gap."
