# Solutions - Module 12
Short, runnable fragments; each replaces/extends the matching `app.py`.

## Step 01 - repeat slider
```python
n = st.slider("repeat", 1, 5, 2)
if st.button("Shout"):
    st.success((name.upper() + " ") * n)
```
`count` is always 1 because the whole script, including `count = 0`, reruns on every interaction.

## Step 02 - counter in session_state
```python
if "n" not in st.session_state:
    st.session_state.n = 0
if st.button("add"):
    st.session_state.n += 1
st.write(st.session_state.n)
```
Break-it: without the guard, `st.session_state.history.append` raises `AttributeError: st.session_state has no attribute "history"`.

## Step 03 - latency
```python
import time
t0 = time.perf_counter()
resp = requests.post(...)
st.caption(f"{(time.perf_counter() - t0) * 1000:.0f} ms")
```
Timeout rationale: without it a dead server hangs the UI worker indefinitely. `localhost` in a container is the container itself.

## Step 04 - bar chart tab
```python
import pandas as pd
probs = load_model().predict_proba([text])[0]
st.bar_chart(pd.DataFrame({"p": probs}, index=load_model().classes_))
```
`set_page_config` after another call raises `StreamlitSetPageConfigMustBeFirstCommandError`.

## Step 05 - threshold
```python
def classify(text, threshold):
    ...
    if max(result.values()) < threshold:
        return {"uncertain": 1.0}
    return result
# inputs=[gr.Textbox(), gr.Slider(0, 1, value=0.6)]
```

## Step 06 - clear button
```python
def clear():
    return [], [], ""
gr.Button("Clear").click(clear, outputs=[history_state, table, out])
```
Output-count error: Gradio reports it returned fewer/more values than declared outputs.

## Rebuild: Interface reverse
```python
import gradio as gr
gr.Interface(fn=lambda t: t[::-1], inputs="text", outputs="text").launch()
```
