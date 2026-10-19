# Exercises - Module 12 Streamlit and Gradio

Rules: change one thing at a time; predict before running; read the traceback yourself first. Answers: [solutions/](solutions/).

## Step 01 - Streamlit basics
- **Modify:** add a slider "repeat" and show the text repeated that many times.
- **Break it:** call `st.text_input()` without arguments; read the browser traceback.
- **Rebuild from memory:** title + text_input + button + success.
- **Explain aloud:** why is `count` always 1?

## Step 02 - model, state, upload
- **Modify:** add a "Clear history" confirmation and show the number of predictions made.
- **Break it:** remove the `if "history" not in st.session_state` guard.
- **Rebuild:** a cached model loader + a counter in session_state.
- **Explain aloud:** cache_resource vs cache_data vs session_state.

## Step 03 - API client
- **Modify:** make the UI also show the latency of the API call (`time.perf_counter()` before and after).
- **Break it:** rename `data["label"]` to `data["lable"]`; catch the KeyError and show a friendly message.
- **Rebuild:** a POST with JSON, timeout and 3 except branches.
- **Explain aloud:** why must you set a timeout and why does `localhost` break in Docker?

## Step 04 - layout/style
- **Modify:** a third tab with a bar chart (`st.bar_chart`) of probabilities for one sentence.
- **Break it:** call `set_page_config` after `st.title`.
- **Rebuild:** sidebar slider + two columns.
- **Explain aloud:** how do sidebar values reach the main code?

## Step 05 - Gradio basics
- **Modify:** add a `gr.Slider` threshold input; return "uncertain" under the threshold.
- **Break it:** return a plain string for a `gr.Label` output.
- **Rebuild:** Interface for `reverse(text)`.
- **Explain aloud:** callback model vs rerun model.

## Step 06 - Blocks
- **Modify:** add a "Clear" button that resets `history_state` and the table.
- **Break it:** return two values for three outputs.
- **Rebuild:** Blocks with a textbox, button, click wiring.
- **Explain aloud:** what `share=True` does and one risk.

## Cross-module task
Point the Streamlit step-03 app at your module-09 API. If response keys differ, adapt only the parsing lines and write down what you changed. Then run the UI as a service in the module-11 Compose file.
