"""
GOAL: Layout and basic customisation: page config, sidebar, columns, tabs, expander, status, theme file.

RUN (from 12_streamlit_gradio/04_streamlit_layout_style):  ../../.venv/bin/streamlit run app.py
     (run it FROM this folder so .streamlit/config.toml, the green theme, is picked up)
EXPECTED OUTPUT: wide page with a sidebar of settings, two tabs ("Single" and "Compare"), and side-by-side columns.
NEEDS: streamlit, scikit-learn (../common/model.py).
"""
import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from model import predict  # noqa: E402

# set_page_config MUST be the first Streamlit call in the script.
st.set_page_config(page_title="Sentiment lab", page_icon=":bar_chart:", layout="wide")

st.title("Sentiment lab")
st.markdown("Layout demo: **sidebar**, **tabs**, **columns**. Markdown works in most text elements.")

# ---- Sidebar: global settings --------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)   # (label, min, max, default, step)
    show_raw = st.checkbox("Show raw output", value=False)

# ---- Tabs: separate views on one page --------------------------------------------------------
tab_single, tab_compare = st.tabs(["Single", "Compare"])

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

# Custom CSS is possible but fragile (Streamlit's internal class names change between versions).
st.markdown("<style>h1 {letter-spacing: 1px;}</style>", unsafe_allow_html=True)

# TRY IT: change layout="wide" to "centered".
# TRY IT: change the theme colours in .streamlit/config.toml and reload the page.
# BREAK IT: call st.set_page_config after st.title -> StreamlitAPIException; read the message.
