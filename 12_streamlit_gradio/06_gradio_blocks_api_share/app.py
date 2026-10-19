"""
GOAL: Gradio Blocks: custom layout, several event handlers, uploads, API calls, loading/error states, state, share.

RUN (from 12_streamlit_gradio/06_gradio_blocks_api_share):  ../../.venv/bin/python app.py
     (for the API tab, also run the API: cd ../common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000)
EXPECTED OUTPUT: tabs "Local model", "Via API" and "Batch file"; a growing history table.
NEEDS: gradio, requests, scikit-learn.

SHARING: demo.launch(share=True) asks Gradio's servers for a temporary public https://xxxx.gradio.live URL that
tunnels to YOUR machine for ~1 week. Anyone with the link can use the app while your process runs. Do not
expose private data/models that way. Permanent options: Hugging Face Spaces, Docker (module 11), any cloud VM.
"""
import os
import sys
from pathlib import Path

import gradio as gr
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import predict  # noqa: E402

API_URL = os.getenv("API_URL", "http://localhost:8000")


def local_predict(text, history):
    """Callback 1: use the local model. `history` is a gr.State value (list) kept per browser session."""
    if not text.strip():
        raise gr.Error("Please enter some text.")
    label, score = predict(text)
    history = history + [[text, label, score]]            # new list (don't mutate the old state in place)
    return f"{label} ({score})", history, history        # outputs: textbox, state, dataframe


def api_predict(text, api_url, progress=gr.Progress()):
    """Callback 2: call the HTTP API. gr.Progress() injects a progress-bar helper automatically."""
    if not text.strip():
        raise gr.Error("Please enter some text.")
    progress(0.2, desc="Calling API...")
    try:
        r = requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)
        r.raise_for_status()
    except requests.ConnectionError:
        raise gr.Error(f"Cannot reach the API at {api_url}. Is it running?")
    except requests.HTTPError as exc:
        raise gr.Error(f"API error {exc.response.status_code}: {exc.response.text}")
    progress(1.0, desc="Done")
    data = r.json()
    return {data["label"]: float(data.get("confidence", data.get("score")))}          # gr.Label wants {class: probability}


def batch_predict(file_obj):
    """Callback 3: file upload. With gr.File(type='filepath') the function receives a path string (or None)."""
    if file_obj is None:
        raise gr.Error("Upload a .txt file first.")
    lines = Path(file_obj).read_text(encoding="utf-8").splitlines()
    return [[ln, *predict(ln)] for ln in lines if ln.strip()]      # rows for gr.Dataframe


with gr.Blocks(title="Sentiment lab (Gradio)") as demo:    # Blocks = you place components and wire events yourself
    gr.Markdown("# Sentiment lab\nBlocks gives layout control and multiple event handlers.")
    history_state = gr.State([])                           # invisible per-session storage

    with gr.Tab("Local model"):
        with gr.Row():                                      # Row = side-by-side, Column = stacked
            with gr.Column(scale=2):
                txt = gr.Textbox(label="Sentence", value="I love this")
                btn = gr.Button("Predict", variant="primary")
            with gr.Column(scale=1):
                out = gr.Textbox(label="Result")
        table = gr.Dataframe(headers=["text", "label", "score"], label="History")
        # EVENT WIRING: when btn is clicked, call local_predict(inputs...) and put returns into outputs...
        btn.click(local_predict, inputs=[txt, history_state], outputs=[out, history_state, table])

    with gr.Tab("Via API"):
        url = gr.Textbox(label="API base URL", value=API_URL)
        txt2 = gr.Textbox(label="Sentence", value="Terrible service")
        out2 = gr.Label(label="API result")
        gr.Button("Call API").click(api_predict, inputs=[txt2, url], outputs=out2)

    with gr.Tab("Batch file"):
        f = gr.File(label="Upload .txt (one sentence per line)", type="filepath", file_types=[".txt"])
        res = gr.Dataframe(headers=["text", "label", "score"])
        f.upload(batch_predict, inputs=f, outputs=res)     # event fires when a file is uploaded

if __name__ == "__main__":
    # theme (and css) are passed to launch() in Gradio 6; in Gradio 4/5 they were arguments of gr.Blocks().
    demo.launch(theme=gr.themes.Soft())     # add share=True for a temporary public link

# TRY IT: change gr.themes.Soft() to gr.themes.Glass(); add share=True and read the printed public URL.
# TRY IT: add demo.queue(max_size=5) to limit concurrent users (queue is on by default).
# BREAK IT: remove `history_state` from outputs but keep returning 3 values -> error about output count.
