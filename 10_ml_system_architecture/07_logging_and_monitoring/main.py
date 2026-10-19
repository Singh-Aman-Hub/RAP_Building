"""
GOAL:   Add the three pillars of ML monitoring to a prediction service:
        (1) structured logs, (2) metrics (latency percentiles, error counts), (3) data/prediction DRIFT checks.

RUN:    cd 10_ml_system_architecture/07_logging_and_monitoring && python main.py

EXPECTED OUTPUT (shape): JSON log lines, a text histogram of latencies with p50/p95/p99,
        then a drift report: STABLE for normal traffic, DRIFT for a changed input distribution.

NEEDS:  scikit-learn, joblib, numpy
"""

import json
import logging                       # built-in: the standard logging framework (levels, handlers, formatters)
import sys
import time
import uuid
from collections import Counter     # built-in dict subclass for counting
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_training_vs_inference_pipelines"))
from predict import load_artifact, predict_text   # noqa: E402
from train import train                           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_training_data  # noqa: E402


# ----------------------------------- 1. structured logs ----------------------------------
class JsonFormatter(logging.Formatter):
    """Format each log record as ONE line of JSON. Machines (Datadog, ELK, CloudWatch) can then filter/aggregate by field."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"level": record.levelname, "event": record.getMessage()}
        entry.update(getattr(record, "fields", {}))      # extra key/values passed via logger.info(..., extra={"fields": {...}})
        return json.dumps(entry)


logger = logging.getLogger("predict-service")
handler = logging.StreamHandler(sys.stdout)              # send logs to the terminal (containers log to stdout; see module 11)
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)                            # DEBUG < INFO < WARNING < ERROR < CRITICAL


# ----------------------------------- 2. metrics -------------------------------------------
LATENCIES: list[float] = []
COUNTERS: Counter = Counter()


def monitored_predict(model, text: str) -> dict | None:
    """Wrap prediction with a request id, timing, counters and logs -- the 'observability wrapper'."""
    request_id = uuid.uuid4().hex[:8]                    # lets you find ALL log lines for one request
    start = time.perf_counter()
    try:
        result = predict_text(model, text, threshold=0.0)
    except ValueError as err:
        COUNTERS["errors"] += 1
        logger.warning("bad_input", extra={"fields": {"request_id": request_id, "error": str(err)}})
        return None
    ms = (time.perf_counter() - start) * 1000
    LATENCIES.append(ms)
    COUNTERS["requests"] += 1
    COUNTERS[f"label_{result['label']}"] += 1
    logger.info("prediction", extra={"fields": {"request_id": request_id, "label": result["label"],
                                                "confidence": result["confidence"], "latency_ms": round(ms, 3),
                                                "text_len": len(text)}})        # log the LENGTH, not private text
    return result


def latency_report(latencies: list[float]) -> None:
    arr = np.array(latencies)
    p50, p95, p99 = np.percentile(arr, [50, 95, 99])     # percentile: value below which q% of requests fall
    print(f"latency ms  p50={p50:.3f}  p95={p95:.3f}  p99={p99:.3f}  max={arr.max():.3f}  (average hides the slow tail!)")
    counts, edges = np.histogram(arr, bins=5)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"  {lo:7.3f}-{hi:7.3f} ms | {'#' * int(40 * c / len(arr)):<40} {c}")


# ----------------------------------- 3. drift ---------------------------------------------
def psi(reference: np.ndarray, live: np.ndarray, bins: int = 5) -> float:
    """Population Stability Index: how different two distributions are.
    Rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 significant drift."""
    edges = np.histogram_bin_edges(reference, bins=bins)             # bin edges from the REFERENCE data
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference) + 1e-4     # +tiny avoids log(0)
    live_clipped = np.clip(live, edges[0], edges[-1])                # values outside the range go in the end bins
    live_pct = np.histogram(live_clipped, bins=edges)[0] / len(live) + 1e-4
    return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))


def drift_report(name: str, ref_texts: list[str], live_texts: list[str]) -> None:
    ref_len = np.array([len(t.split()) for t in ref_texts])          # feature to watch: words per text
    live_len = np.array([len(t.split()) for t in live_texts])
    score = psi(ref_len, live_len)
    verdict = "STABLE" if score < 0.1 else "WATCH" if score < 0.25 else "DRIFT"
    print(f"drift check [{name}]: PSI(text length) = {score:.3f} -> {verdict}")


if __name__ == "__main__":
    train()
    model, _ = load_artifact()
    reference_texts, _ = get_training_data(n=300, seed=0)            # what the model saw in training

    print("--- structured logs ---")
    normal_traffic, _ = get_training_data(n=60, seed=5)
    for t in normal_traffic[:3]:
        monitored_predict(model, t)
    logger.setLevel(logging.WARNING)                                 # level filter: from now on INFO lines are dropped
    monitored_predict(model, "   ")                                  # invalid input -> WARNING still shown
    for t in normal_traffic[3:]:
        monitored_predict(model, t)                                  # still counted + timed, just not logged
    print("\ncounters:", dict(COUNTERS))
    print("\n--- latency ---")
    latency_report(LATENCIES)

    print("\n--- drift ---")
    drift_report("normal traffic", reference_texts, normal_traffic)
    novel = ["I ordered this item last week and honestly the whole experience from checkout to delivery was "
             "frustrating because nobody answered my emails and the package arrived damaged"] * 60
    drift_report("users now write long reviews", reference_texts, novel)

    for label in ("positive", "negative"):                           # prediction drift: label share vs expectation
        share = COUNTERS[f"label_{label}"] / COUNTERS["requests"]
        print(f"label share {label}: {share:.0%} (training ~50/50; with only 60 requests some wobble is normal, a persistent swing = investigate)")
    # TRY IT: change the novel texts to short ones ("good", "bad") and see what PSI does.
    # TRY IT: add a WARNING when p95 latency exceeds 50 ms -- that is the seed of an alert rule.
