"""
GOAL: The smallest useful Streamlit app: text box -> button -> result, to learn the "rerun" model.

RUN (from 12_streamlit_gradio/01_streamlit_basics):  ../../.venv/bin/streamlit run app.py
RUN (from repo root):  .venv/bin/streamlit run 12_streamlit_gradio/01_streamlit_basics/app.py
EXPECTED OUTPUT: a browser tab at http://localhost:8501 with a title, a text box, a "Shout" button.
                 Click the button -> the uppercase text appears. Ctrl+C in the terminal stops the server.
NEEDS: streamlit.

THE KEY IDEA: Streamlit re-runs this WHOLE file, top to bottom, every time the user touches a widget.
There are no callbacks you must register: a widget function (st.text_input) simply RETURNS the current value.
"""
import streamlit as st            # the usual alias; every Streamlit function starts with st.

st.title("Hello, Streamlit")      # big heading
st.write("Type something and press the button.")   # st.write displays almost anything (text, dict, DataFrame)

# st.text_input(label, value) draws a box and RETURNS what is currently typed (a str).
name = st.text_input("Your text", value="hello world")

# st.button(label) returns True ONLY on the rerun triggered by the click, otherwise False.
if st.button("Shout"):
    st.success(name.upper())      # st.success draws a green box
else:
    st.info("Waiting for a click...")

# Show how reruns work: this counter is a plain variable, so it resets on EVERY rerun (always 1).
count = 0
count += 1
st.caption(f"plain variable count = {count} (resets every rerun - see step 02 for st.session_state)")

# TRY IT: change `value="hello world"`, save the file; Streamlit offers "Rerun" / "Always rerun".
# TRY IT: add st.slider("n", 0, 10) and print its value with st.write.
# BREAK IT: write `st.text_input()` with no label -> TypeError shown in the browser; read the traceback there.
