"""
GOAL:   Prove the Flask app and the FastAPI app give the SAME answers for the SAME requests,
        and see the few places where they differ (error body shape, 405/404 handling).

RUN:    cd 09_api_development/06_model_api_flask_vs_fastapi && python main.py
        With a real model:  MODEL_PATH=/path/to/model.joblib python main.py
        With generation:    ENABLE_GENERATE=1 python main.py     (first run downloads distilgpt2 ~330 MB)

EXPECTED OUTPUT (shape):
        POST /predict great film   flask=positive 0.75   fastapi=positive 0.75
        ...
        bad request statuses ...

NEEDS:  flask, fastapi, scikit-learn+joblib only if MODEL_PATH is set.
"""

from fastapi.testclient import TestClient

from config import get_settings
from fastapi_app import app as fastapi_app
from flask_app import app as flask_app

if __name__ == "__main__":
    flask_client = flask_app.test_client()
    # `with TestClient(app)` is needed so FastAPI's lifespan (model loading) runs.
    with TestClient(fastapi_app) as fast_client:
        print("health  flask  =", flask_client.get("/health").get_json())
        print("health  fastapi=", fast_client.get("/health").json())
        print()
        for text in ["I love this great film", "This was terrible and boring", "It was a film"]:
            f = flask_client.post("/predict", json={"text": text}).get_json()
            g = fast_client.post("/predict", json={"text": text}).json()
            print(f"{text!r:<32} flask={f['label']} {f['confidence']}   fastapi={g['label']} {g['confidence']}")
        print()
        print("bad requests (Flask status / FastAPI status):")
        bad = [("missing text", {}), ("empty text", {"text": ""}), ("number", {"text": 5})]
        for name, body in bad:
            fs = flask_client.post("/predict", json=body).status_code
            gs = fast_client.post("/predict", json=body).status_code
            print(f"  {name:<14}{fs} / {gs}")
        print("  wrong method  ", flask_client.get("/predict").status_code, "/", fast_client.get("/predict").status_code)
        print("  unknown route ", flask_client.get("/nope").status_code, "/", fast_client.get("/nope").status_code)
        print("  generate probe ", flask_client.post("/generate", json={"prompt": "Hi"}).status_code, "/",
              fast_client.post("/generate", json={"prompt": "Hi"}).status_code)
        if get_settings().enable_generate:
            print("generate:", fast_client.post("/generate", json={"prompt": "The cat", "max_new_tokens": 8}).json())
    # TRY IT: run again with MODEL_VERSION=2.0.0 and watch model_version change in /health.
    # TRY IT: MAX_TEXT_LENGTH=5 python main.py  -> every real sentence now fails with 422.
