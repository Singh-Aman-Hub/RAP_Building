"""
GOAL:   Store predictions in a database and ask useful questions about them with SQL.
        Also see where each kind of data lives in an ML system (the "storage map").

RUN:    cd 10_ml_system_architecture/05_storage_and_prediction_log && python main.py

EXPECTED OUTPUT (shape): label counts, latency stats, low-confidence rows, an injection demo, a storage map.

NEEDS:  scikit-learn, joblib
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_training_vs_inference_pipelines"))
from predict import load_artifact, predict_text   # noqa: E402  (reuses step 02's inference code)
from train import train                           # noqa: E402

from prediction_log import PredictionLog

if __name__ == "__main__":
    if not (Path(__file__).resolve().parents[1] / "02_training_vs_inference_pipelines/artifacts/model.joblib").exists():
        train()                                   # make sure an artifact exists
    model, meta = load_artifact()
    log = PredictionLog()                         # in-memory DB for the demo; pass "preds.db" to keep a file

    traffic = ["Great hotel", "Awful delivery", "It arrived Tuesday", "The app is brilliant", "Meh, it exists",
               "Terrible, worthless game", "Lovely teacher", "I bought a phone"] * 5
    for text in traffic:
        t0 = time.perf_counter()
        result = predict_text(model, text, threshold=0.0)         # threshold 0.0: always return the raw best label
        ms = (time.perf_counter() - t0) * 1000
        log.record("v-demo", text, result["label"], result["confidence"], ms)

    print("label counts   :", log.label_counts())
    print("latency stats  :", log.latency_stats())
    print("low confidence :")
    for row in log.low_confidence(below=0.6, limit=3):
        print("   ", row)

    # SQL injection demo: why we use ? placeholders.
    evil = "x' OR '1'='1"
    unsafe_sql = f"SELECT COUNT(*) FROM predictions WHERE label = '{evil}'"      # BAD: user text becomes SQL
    safe_count = log.conn.execute("SELECT COUNT(*) FROM predictions WHERE label = ?", (evil,)).fetchone()[0]
    print(f"\nunsafe f-string SQL matches every row: {log.conn.execute(unsafe_sql).fetchone()[0]} rows;"
          f" parameterized query matches: {safe_count}")

    print("""
STORAGE MAP -- what lives where in a typical ML system
  raw / training data ........ object storage (S3, GCS) or a data warehouse     (big, append-only)
  model artifacts ............ object storage + registry (step 04)              (immutable, versioned)
  online features ............ key-value store (Redis) -- fast lookups            (low latency)
  prediction logs / feedback . relational DB (Postgres; SQLite here)             (queryable, durable)
  metrics .................... time-series DB (Prometheus)                       (dashboards, alerts)
""")
    # TRY IT: change ":memory:" to "preds.db", run twice, and check that counts accumulate. Then delete the file.
    # TRY IT: write a method that returns average confidence per label.
    # BREAK IT: rename a column in a query to 'lable' and read the sqlite3.OperationalError.
