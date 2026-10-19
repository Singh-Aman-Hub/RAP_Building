"""
GOAL: A Streamlit UI that does NOT hold the model itself; it calls an HTTP API (module 09, or common/mock_api.py)
      and shows loading, success and several kinds of error states.

RUN:
  terminal 1 (API):  cd 12_streamlit_gradio/common && ../../.venv/bin/python -m uvicorn mock_api:app --port 8000
                     (or start your module-09 API on port 8000 instead)
  terminal 2 (UI):   cd 12_streamlit_gradio/03_streamlit_api_client && ../../.venv/bin/streamlit run app.py
EXPECTED OUTPUT: type a sentence, press Predict -> spinner -> label + confidence. Stop the API and press again ->
                 a red "Cannot reach the API" box (the UI itself stays alive).
NEEDS: streamlit, requests.
"""
import os

import requests          # 3rd-party HTTP client (httpx is the modern alternative with almost the same API)
import streamlit as st

# Where is the API? Environment variable first (Docker/Compose sets API_URL), else localhost.
DEFAULT_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Sentiment UI -> API")
api_url = st.sidebar.text_input("API base URL", DEFAULT_URL)      # sidebar = left panel

if st.sidebar.button("Check /health"):
    try:
        r = requests.get(f"{api_url}/health", timeout=3)          # timeout: NEVER call a network without one
        st.sidebar.success(f"API says: {r.json()}")
    except requests.RequestException as exc:                      # base class of all requests errors
        st.sidebar.error(f"API down: {type(exc).__name__}")

text = st.text_area("Text to classify", "The service was excellent")   # multi-line input

if st.button("Predict"):
    if not text.strip():
        st.warning("Please type something first.")                # validate BEFORE the network call
    else:
        try:
            with st.spinner("Calling the API..."):                # shows a spinner while the block runs
                resp = requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)
            resp.raise_for_status()                               # 4xx/5xx -> HTTPError
            data = resp.json()
            st.metric("Label", data["label"], f"score {data.get('confidence', data.get('score'))}")  # 09 API: 'confidence'; mock API: 'score'
            with st.expander("Raw response"):                     # collapsible section
                st.json(data)
        except requests.ConnectionError:
            st.error(f"Cannot reach the API at {api_url}. Is it running?")
        except requests.Timeout:
            st.error("The API took too long to answer (timeout).")
        except requests.HTTPError as exc:
            st.error(f"API returned {exc.response.status_code}: {exc.response.text}")
        except (KeyError, ValueError):                            # missing key / invalid JSON
            st.error("The API answered, but not in the expected format.")

# TRY IT: change the URL in the sidebar to a wrong port and press Predict.
# TRY IT: add a time.sleep(3) inside mock_api.predict and watch the spinner.
# BREAK IT: change data["label"] to data["labell"] -> KeyError caught by our friendly message; remove the except and see the raw traceback.
