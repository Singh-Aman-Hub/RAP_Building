"""
GOAL: A minimal Streamlit UI that talks to the API container. Shows how one container calls another.

The API address is NOT localhost: inside the ui container, "localhost" means the ui container itself.
We use the Compose service name instead:  http://api:8000   (given via the API_URL env var).
RUN locally against a locally running API:  API_URL=http://localhost:8000 streamlit run ui/app.py
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Sentiment UI")
text = st.text_input("Type a sentence", "I love containers")
if st.button("Predict"):
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=5)
        r.raise_for_status()                      # turns 4xx/5xx into an exception
        st.json(r.json())
        st.caption(f"API stats: {requests.get(f'{API_URL}/stats', timeout=5).json()}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_URL}: {exc}")
