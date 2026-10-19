# Vocabulary draft - 12 Streamlit and Gradio

### Streamlit
<!-- cat: api -->
- **Technical definition:** Open-source Python framework that turns scripts into interactive web apps by rerunning the script on each interaction.
- **Plain English:** Write a Python script, get a web page.
- **Example / analogy:** A notebook that becomes an app.
- **Snippet:**
```python
import streamlit as st
st.title('Hi')
```
- **Snippet explained:** Run with `streamlit run app.py`; st.title draws a heading.
- **Related terms:** [[Rerun model (Streamlit)]], [[Widget]]

### Gradio
<!-- cat: api -->
- **Technical definition:** Open-source Python library that builds web UIs around functions by declaring input and output components.
- **Plain English:** Wrap a function with a web form.
- **Example / analogy:** A remote control for a function.
- **Snippet:**
```python
import gradio as gr
gr.Interface(lambda t: t[::-1], 'text', 'text').launch()
```
- **Snippet explained:** Builds a text-reversing app at http://127.0.0.1:7860.
- **Related terms:** [[Callback (event handler)]], [[gr.Interface]]

### Rerun model (Streamlit)
<!-- cat: api -->
- **Technical definition:** Streamlit re-executes the whole script top to bottom whenever a widget changes or a button is clicked.
- **Plain English:** The page is re-drawn by running your file again.
- **Example / analogy:** Re-reading a recipe from the top every time.
- **Snippet:**
```python
count = 0
count += 1   # always 1 after every rerun
```
- **Snippet explained:** Plain variables reset on every rerun; use session_state to remember.
- **Related terms:** [[Session state]], [[Streamlit]]

### Widget
<!-- cat: api -->
- **Technical definition:** Interactive UI element (text input, slider, button) that returns its current value in Streamlit.
- **Plain English:** A control the user can touch.
- **Example / analogy:** A form field.
- **Snippet:**
```python
name = st.text_input('Name', 'Ada')
```
- **Snippet explained:** Returns the typed string on each rerun.
- **Related terms:** [[Streamlit]], [[Rerun model (Streamlit)]]

### st.button
<!-- cat: api -->
- **Technical definition:** Streamlit widget returning True only during the rerun caused by its click.
- **Plain English:** A button that is True once after the click.
- **Example / analogy:** A doorbell: rings once.
- **Snippet:**
```python
if st.button('Go'):
    st.write('clicked')
```
- **Snippet explained:** The block runs only on the click's rerun.
- **Related terms:** [[Rerun model (Streamlit)]]

### Session state
<!-- cat: api -->
- **Technical definition:** Per-session dict-like storage (st.session_state / gr.State) that survives reruns and interactions for one user session.
- **Plain English:** Memory for one visitor's tab.
- **Example / analogy:** A name tag that stays on you during a visit.
- **Snippet:**
```python
if 'n' not in st.session_state:
    st.session_state.n = 0
st.session_state.n += 1
```
- **Snippet explained:** Initialise once, then update; value persists across reruns.
- **Related terms:** [[Rerun model (Streamlit)]], [[gr.State]]

### st.cache_resource
<!-- cat: api -->
- **Technical definition:** Decorator caching a function's returned object (models, connections) once per process and sharing it across reruns and users.
- **Plain English:** Load the heavy thing once and reuse it.
- **Example / analogy:** Keeping the oven hot.
- **Snippet:**
```python
@st.cache_resource
def get_model():
    return load_model()
```
- **Snippet explained:** First call loads; later calls return the same object instantly.
- **Related terms:** [[st.cache_data]], [[Session state]]

### st.cache_data
<!-- cat: api -->
- **Technical definition:** Decorator caching serialisable return values (DataFrames, results), returning a copy per call.
- **Plain English:** Remember answers to repeated computations.
- **Example / analogy:** A cheat sheet of results.
- **Snippet:**
```python
@st.cache_data
def load_csv(p): ...
```
- **Snippet explained:** Reruns with the same arguments reuse the stored result.
- **Related terms:** [[st.cache_resource]]

### st.form
<!-- cat: api -->
- **Technical definition:** Container grouping widgets so the script only reruns when its submit button is pressed.
- **Plain English:** Fill in everything, then press submit.
- **Example / analogy:** A paper form.
- **Snippet:**
```python
with st.form('f'):
    t = st.text_input('x')
    ok = st.form_submit_button('Go')
```
- **Snippet explained:** No rerun while typing; ok is True after submit.
- **Related terms:** [[st.button]]

### st.file_uploader
<!-- cat: api -->
- **Technical definition:** Widget that lets users upload files, returning an UploadedFile (file-like) or None.
- **Plain English:** A file picker.
- **Example / analogy:** An attachment button.
- **Snippet:**
```python
f = st.file_uploader('CSV', type=['csv'])
if f:
    data = f.getvalue()
```
- **Snippet explained:** getvalue() returns the bytes of the uploaded file.
- **Related terms:** [[st.form]]

### st.spinner
<!-- cat: api -->
- **Technical definition:** Context manager displaying a spinner message while a block executes.
- **Plain English:** A 'loading...' indicator.
- **Example / analogy:** The hourglass icon.
- **Snippet:**
```python
with st.spinner('Working...'):
    result = slow()
```
- **Snippet explained:** Spinner disappears when the block finishes.
- **Related terms:** [[Loading state]], [[st.error]]

### Loading state
<!-- cat: api -->
- **Technical definition:** UI feedback shown while an operation is in progress (spinner, progress bar, disabled button).
- **Plain English:** Telling the user to wait.
- **Example / analogy:** A 'please hold' message.
- **Snippet:**
```python
progress(0.5, desc='Calling API...')
```
- **Snippet explained:** In Gradio, gr.Progress updates a progress bar from inside the function.
- **Related terms:** [[st.spinner]], [[gr.Progress]]

### st.error
<!-- cat: api -->
- **Technical definition:** Displays a red error message box; sister functions: st.warning, st.info, st.success.
- **Plain English:** Shows a friendly error to the user.
- **Example / analogy:** A red sign.
- **Snippet:**
```python
try:
    ...
except ValueError as e:
    st.error(str(e))
```
- **Snippet explained:** Converts an exception into a message instead of a crash page.
- **Related terms:** [[gr.Error]], [[st.spinner]]

### st.columns
<!-- cat: api -->
- **Technical definition:** Creates side-by-side layout containers.
- **Plain English:** Splits the page into vertical strips.
- **Example / analogy:** A newspaper layout.
- **Snippet:**
```python
left, right = st.columns(2)
left.write('A')
```
- **Snippet explained:** Returns two column objects; call widgets on them or use `with`.
- **Related terms:** [[st.tabs]], [[st.sidebar]]

### st.tabs
<!-- cat: api -->
- **Technical definition:** Creates tabbed sections; returns a list of containers.
- **Plain English:** Several pages in one.
- **Example / analogy:** Browser tabs.
- **Snippet:**
```python
t1, t2 = st.tabs(['One','Two'])
with t1:
    st.write('hi')
```
- **Snippet explained:** Only content inside each `with` block appears in that tab.
- **Related terms:** [[st.columns]]

### st.sidebar
<!-- cat: api -->
- **Technical definition:** Left panel container for global controls.
- **Plain English:** A settings drawer.
- **Example / analogy:** A car dashboard side panel.
- **Snippet:**
```python
x = st.sidebar.slider('n', 0, 10)
```
- **Snippet explained:** Slider is rendered in the sidebar; returns its value.
- **Related terms:** [[st.columns]]

### st.set_page_config
<!-- cat: api -->
- **Technical definition:** Sets page title, icon and layout; must be the first Streamlit command.
- **Plain English:** Basic page settings.
- **Example / analogy:** The page's name tag.
- **Snippet:**
```python
st.set_page_config(layout='wide')
```
- **Snippet explained:** Use the whole screen width.
- **Related terms:** [[Streamlit]]

### Streamlit theme (config.toml)
<!-- cat: api -->
- **Technical definition:** Appearance settings in .streamlit/config.toml under [theme].
- **Plain English:** Colours and fonts for the app.
- **Example / analogy:** Paint colours.
- **Snippet:**
```toml
[theme]
primaryColor = "#0b6e4f"
```
- **Snippet explained:** Changes button and link accent colour.
- **Related terms:** [[st.set_page_config]]

### AppTest (Streamlit testing)
<!-- cat: api -->
- **Technical definition:** streamlit.testing.v1.AppTest runs a Streamlit script headlessly so tests can set widget values and read outputs.
- **Plain English:** Test a Streamlit app without a browser.
- **Example / analogy:** A robot user.
- **Snippet:**
```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('app.py').run()
assert not at.exception
```
- **Snippet explained:** Runs the script once and checks no exception was raised.
- **Related terms:** [[Streamlit]]

### gr.Interface
<!-- cat: api -->
- **Technical definition:** High-level Gradio class that builds a UI around one function given inputs and outputs.
- **Plain English:** The simplest Gradio app.
- **Example / analogy:** A form attached to a function.
- **Snippet:**
```python
gr.Interface(fn=f, inputs=gr.Textbox(), outputs=gr.Label())
```
- **Snippet explained:** One input component per parameter, one output per return value.
- **Related terms:** [[Gradio]], [[gr.Blocks]], [[Callback (event handler)]]

### gr.Blocks
<!-- cat: api -->
- **Technical definition:** Low-level Gradio API for custom layouts and multiple event listeners.
- **Plain English:** Build your own layout.
- **Example / analogy:** Lego instead of a ready-made toy.
- **Snippet:**
```python
with gr.Blocks() as demo:
    t = gr.Textbox()
    b = gr.Button('Go')
    b.click(f, t, t)
```
- **Snippet explained:** Components are placed in order; click wires f to them.
- **Related terms:** [[gr.Interface]], [[Callback (event handler)]]

### Callback (event handler)
<!-- cat: api -->
- **Technical definition:** A function registered to run when an event (click, submit, upload) occurs.
- **Plain English:** The function that runs when the user does something.
- **Example / analogy:** A doorbell wired to a chime.
- **Snippet:**
```python
btn.click(fn, inputs=[txt], outputs=[out])
```
- **Snippet explained:** On click, Gradio calls fn(txt value) and puts the return into out.
- **Related terms:** [[gr.Blocks]], [[Rerun model (Streamlit)]]

### gr.State
<!-- cat: api -->
- **Technical definition:** Gradio component storing per-session Python values not shown in the UI.
- **Plain English:** Hidden memory for a session.
- **Example / analogy:** A pocket in the UI.
- **Snippet:**
```python
history = gr.State([])
```
- **Snippet explained:** Pass it as input and output to read and update.
- **Related terms:** [[Session state]]

### gr.Error
<!-- cat: api -->
- **Technical definition:** Exception that Gradio displays as a toast message to the user.
- **Plain English:** A polite error message.
- **Example / analogy:** A pop-up notice.
- **Snippet:**
```python
raise gr.Error('Please enter text')
```
- **Snippet explained:** Stops the function and shows the message in the UI.
- **Related terms:** [[st.error]]

### gr.Progress
<!-- cat: api -->
- **Technical definition:** Helper injected into a Gradio function for updating a progress bar.
- **Plain English:** Progress bar controller.
- **Example / analogy:** A loading bar.
- **Snippet:**
```python
def f(x, progress=gr.Progress()):
    progress(0.5, desc='half')
```
- **Snippet explained:** Gradio detects the default argument and passes a tracker.
- **Related terms:** [[Loading state]]

### gr.Label
<!-- cat: api -->
- **Technical definition:** Gradio output component for classification results given a {class: probability} dict.
- **Plain English:** Displays class probabilities as bars.
- **Example / analogy:** A scoreboard.
- **Snippet:**
```python
return {'positive': 0.8, 'negative': 0.2}
```
- **Snippet explained:** The dict shape is required by gr.Label.
- **Related terms:** [[gr.Interface]]

### Share link (Gradio)
<!-- cat: api -->
- **Technical definition:** launch(share=True) creates a temporary public URL tunnelling to your local server.
- **Plain English:** Instant public link to your demo.
- **Example / analogy:** A temporary guest pass.
- **Snippet:**
```python
demo.launch(share=True)
```
- **Snippet explained:** Prints a *.gradio.live URL valid while your process runs; do not use with private data.
- **Related terms:** [[gr.Interface]], [[Deployment (UI apps)]]

### Deployment (UI apps)
<!-- cat: api -->
- **Technical definition:** Hosting a UI app permanently: Docker on a server, Streamlit Community Cloud, Hugging Face Spaces, cloud VMs.
- **Plain English:** Putting the app somewhere others can reach all the time.
- **Example / analogy:** Moving from a garage prototype to a shop.
- **Snippet:**
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
- **Snippet explained:** Bind to all interfaces and a fixed port so containers/servers can expose it.
- **Related terms:** [[Share link (Gradio)]]

### API client (UI to API)
<!-- cat: api -->
- **Technical definition:** Code in the UI that sends HTTP requests to a prediction API and handles responses and failures.
- **Plain English:** The UI asking the model server for answers.
- **Example / analogy:** A waiter taking orders to the kitchen.
- **Snippet:**
```python
r = requests.post(url, json={'text': t}, timeout=10)
r.raise_for_status()
r.json()
```
- **Snippet explained:** Sends JSON with a timeout; raises HTTPError on 4xx/5xx; parses the answer.
- **Related terms:** [[Timeout (HTTP)]], [[st.error]]

### Timeout (HTTP)
<!-- cat: api -->
- **Technical definition:** Maximum time to wait for a network call before raising an exception.
- **Plain English:** Do not wait forever.
- **Example / analogy:** Hanging up after 10 rings.
- **Snippet:**
```python
requests.get(url, timeout=3)
```
- **Snippet explained:** Raises requests.Timeout if no answer in 3 seconds.
- **Related terms:** [[API client (UI to API)]]
