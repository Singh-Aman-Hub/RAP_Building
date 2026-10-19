#!/usr/bin/env bash
# Run the capstone WITHOUT Docker (two terminals' worth in one script). Ctrl+C stops both.
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
(cd api && MODEL_PATH=artifacts/model.joblib $PY -m uvicorn fastapi_app:app --port 8000) &
API=$!
trap "kill $API" EXIT
sleep 3
API_URL=http://localhost:8000 $PY -m streamlit run ui/app.py --server.port 8501
