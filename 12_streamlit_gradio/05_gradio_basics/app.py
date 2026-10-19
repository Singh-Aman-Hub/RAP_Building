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
import sys
from pathlib import Path

import gradio as gr   # standard alias

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import load_model, predict  # noqa: E402


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


demo = gr.Interface(
    fn=classify,                                                   # the function to expose
    inputs=gr.Textbox(label="Sentence", lines=2, placeholder="Type here..."),   # one input per fn parameter
    outputs=gr.Label(num_top_classes=2, label="Sentiment"),        # one output per returned value
    examples=[["I love this"], ["This is awful"]],                 # clickable sample inputs
    title="Sentiment (Gradio Interface)",
    description="A function becomes a web app.",
    flagging_mode="never",                                         # hide the 'Flag' button
)

if __name__ == "__main__":     # only start the server when run as a script, not when imported (tests import `classify`)
    demo.launch()              # server_port=7860 by default; use demo.launch(server_port=7861) if busy

# TRY IT: change num_top_classes=2 to 1.
# TRY IT: add a second input gr.Slider(0, 1, value=0.5) and a second parameter `threshold` to classify.
# BREAK IT: make `classify` return a string instead of a dict -> Gradio shows an error toast; read the terminal traceback.
