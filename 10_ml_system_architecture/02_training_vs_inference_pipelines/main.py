"""
GOAL:   See why training and inference are two DIFFERENT pipelines that share ONE artifact,
        and how the artifact produced here plugs straight into the API from module 09.

RUN:    cd 10_ml_system_architecture/02_training_vs_inference_pipelines && python main.py

EXPECTED OUTPUT (shape):
        TRAINING pipeline  (offline): ... accuracy 1.0
        INFERENCE pipeline (online) : loaded in X ms, then per-request ~0.2 ms
        <predictions incl. one 'uncertain'>
        To serve this model with the module-09 API: MODEL_PATH=... uvicorn ...

NEEDS:  scikit-learn, joblib
"""

import time

from predict import ARTIFACT_DIR, load_artifact, predict_text
from train import train

if __name__ == "__main__":
    t0 = time.perf_counter()
    meta = train()
    print(f"TRAINING pipeline  (offline): took {(time.perf_counter()-t0)*1000:.0f} ms, "
          f"accuracy {meta['metrics']['accuracy']}, classes {meta['classes']}")

    t0 = time.perf_counter()
    model, meta = load_artifact()
    print(f"INFERENCE pipeline (online) : artifact loaded in {(time.perf_counter()-t0)*1000:.1f} ms (once, at startup)")

    for text in ["What a superb hotel", "The delivery was dreadful", "It arrived on Tuesday"]:
        t0 = time.perf_counter()
        result = predict_text(model, text)
        print(f"  {text!r:<30} -> {result}   ({(time.perf_counter()-t0)*1000:.2f} ms)")

    print("\nTo serve this exact model with the module-09 API:")
    print(f"  cd 09_api_development/06_model_api_flask_vs_fastapi")
    print(f"  MODEL_PATH={ARTIFACT_DIR}/model.joblib uvicorn fastapi_app:app --port 8000")
    # TRY IT: change threshold in predict_text to 0.9 and see more 'uncertain' answers.
    # TRY IT: call train(n_samples=10) then re-run predictions: how does confidence change?
