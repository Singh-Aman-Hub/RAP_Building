"""
GOAL: Load an ML model once (cache), predict inside a form, keep history across reruns (session state),
      and classify every line of an uploaded .txt file.

RUN (from 12_streamlit_gradio/02_streamlit_model_state_uploads):  ../../.venv/bin/streamlit run app.py
EXPECTED OUTPUT: a form with a text box + "Predict"; a prediction box; a history list that grows;
                 a file uploader that shows a table of predictions for each uploaded line.
NEEDS: streamlit, scikit-learn (model comes from ../common/model.py).
"""
import sys
from pathlib import Path

import streamlit as st

# Make ../common importable no matter where `streamlit run` is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import load_model, predict  # noqa: E402  (import after sys.path change on purpose)

st.title("Sentiment classifier")


# @st.cache_resource: run this function ONCE per server process and reuse the returned object on every
# rerun and for every user. Use it for models, DB connections. (st.cache_data is for plain data/DataFrames.)
@st.cache_resource
def get_model():
    st.write("Loading model...")      # shown only the first time; proves caching works
    return load_model()


get_model()

# st.session_state is a dict-like object that SURVIVES reruns (one per browser tab/session).
if "history" not in st.session_state:          # initialise once
    st.session_state.history = []              # attribute style == st.session_state["history"]

# st.form batches several widgets: nothing reruns until the submit button is pressed.
with st.form("predict_form"):                  # `with` = context manager: widgets inside belong to the form
    text = st.text_input("Sentence", "I really enjoyed this")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        label, score = predict(text)           # may raise ValueError for empty text
        st.metric("Prediction", label, f"confidence {score}")   # big number-style display
        st.session_state.history.append((text, label, score))
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")      # friendly error instead of a crash page

if st.session_state.history:
    st.subheader("History (session_state)")
    st.table(st.session_state.history)         # list of tuples -> static table
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()                             # redraw immediately with the empty list

st.divider()
st.subheader("Batch: upload a .txt file (one sentence per line)")
uploaded = st.file_uploader("Choose a text file", type=["txt"])   # returns None or an UploadedFile
if uploaded is not None:
    lines = uploaded.getvalue().decode("utf-8").splitlines()      # bytes -> str -> list of lines
    rows = [{"text": ln, **dict(zip(("label", "score"), predict(ln)))} for ln in lines if ln.strip()]
    st.dataframe(rows)                          # interactive table

# TRY IT: remove @st.cache_resource and watch "Loading model..." appear on every interaction.
# TRY IT: replace the form with a plain text_input + button and note the extra reruns while typing.
# BREAK IT: delete the `if "history" not in ...` block -> AttributeError on first run. Read the message.
