#!/usr/bin/env bash
# curl cheat sheet for the model API. Start a server first (in another terminal):
#   cd 09_api_development/06_model_api_flask_vs_fastapi && uvicorn fastapi_app:app --port 8000
# Then:  bash curl_cheatsheet.sh        (use PORT=5001 bash curl_cheatsheet.sh for the Flask dev server)
PORT=${PORT:-8000}
BASE=http://127.0.0.1:$PORT

echo "--- GET /health (-i shows status line + headers)"
curl -i -s $BASE/health

echo; echo "--- POST /predict (-X method, -H header, -d body)"
curl -s -X POST $BASE/predict -H "Content-Type: application/json" -d '{"text": "I love this"}'

echo; echo "--- invalid body: -w prints just the status code"
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE/predict -H "Content-Type: application/json" -d '{"text": ""}'

echo "--- wrong method"
curl -s -o /dev/null -w "%{http_code}\n" -X GET $BASE/predict

echo "--- timing (-w with time_total)"
curl -s -o /dev/null -w "total: %{time_total}s\n" $BASE/health
