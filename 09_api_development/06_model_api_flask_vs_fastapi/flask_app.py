"""
flask_app.py -- the model API written with FLASK (WSGI, synchronous).

RUN (dev server):   python flask_app.py                      -> http://127.0.0.1:5001
RUN (production):   gunicorn flask_app:app --workers 2 --bind 127.0.0.1:5001
                    ("flask_app:app" = file flask_app.py, variable `app`)
Compare with fastapi_app.py: same endpoints, same JSON, different framework.
"""

import os
import time

from flask import Flask, jsonify, request

from config import get_settings
from model_loader import generate_text, load_model, predict_one, timed

app = Flask(__name__)

# Loaded ONCE per process at import time. Under Gunicorn with 2 workers this line runs TWICE
# (once per worker process) => the model sits in memory twice. See step 07.
MODEL, MODEL_SOURCE = load_model()
print(f"[pid {os.getpid()}] flask_app loaded model from {MODEL_SOURCE}", flush=True)


@app.get("/health")
def health():
    s = get_settings()
    return jsonify(status="ok", model_version=s.model_version, model_source=MODEL_SOURCE)


@app.post("/predict")
def predict():
    s = get_settings()
    data = request.get_json(silent=True)                       # None if body is missing / not JSON
    if not isinstance(data, dict):
        return jsonify(error="body must be a JSON object"), 400
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="'text' must be a non-empty string"), 422
    if len(text) > s.max_text_length:
        return jsonify(error=f"'text' longer than {s.max_text_length} chars"), 422
    (label, confidence), ms = timed(predict_one, MODEL, text)
    return jsonify(label=label, confidence=confidence, model_version=s.model_version, latency_ms=ms)


@app.post("/generate")
def generate():
    if not get_settings().enable_generate:
        return jsonify(error="generation disabled; start with ENABLE_GENERATE=1"), 503
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return jsonify(error="'prompt' must be a non-empty string"), 422
    return jsonify(text=generate_text(prompt, int(data.get("max_new_tokens", 20))))


# ----- helper endpoints used by step 07 to explore workers and concurrency -----
@app.get("/whoami")
def whoami():
    return jsonify(pid=os.getpid())                             # which worker process answered?


@app.get("/slow")
def slow():
    time.sleep(float(request.args.get("seconds", 1)))           # pretend to wait on a DB / model
    return jsonify(pid=os.getpid(), slept=True)


# ----- JSON error pages instead of Flask's default HTML -----
@app.errorhandler(404)
def not_found(_e):
    return jsonify(error="route not found"), 404


@app.errorhandler(405)
def bad_method(_e):
    return jsonify(error="method not allowed"), 405


@app.errorhandler(Exception)                                    # last resort: never leak a stack trace to clients
def unexpected(e):
    app.logger.exception("unhandled error")                     # the traceback goes to OUR logs
    return jsonify(error="internal server error"), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
