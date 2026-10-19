"""
GOAL:   Compare BATCH inference (many inputs at once, offline, throughput-focused) with REAL-TIME inference
        (one input per request, online, latency-focused) using the same model.

RUN:    cd 10_ml_system_architecture/06_batch_vs_realtime && python main.py

EXPECTED OUTPUT (shape):
        real-time: 2000 predictions, one call each  -> X ms total, Y ms per item
        batch    : 2000 predictions in chunks of 500 -> X ms total, Y ms per item   (several times faster)
        results written to ...

NEEDS:  scikit-learn, joblib
"""

import csv
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02_training_vs_inference_pipelines"))
from predict import load_artifact   # noqa: E402
from train import train             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01_pipeline_end_to_end"))
from data import get_training_data  # noqa: E402


def chunks(items: list, size: int):
    """Yield successive slices. `yield` makes this a GENERATOR: it produces one chunk at a time, so a huge
    file never has to sit in memory all at once."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def realtime(model, texts):
    """What an API endpoint does: one request -> one prediction. Optimises LATENCY of each answer."""
    return [model.predict([t])[0] for t in texts]


def batch(model, texts, chunk_size=500):
    """What a nightly job does: score a whole table. Optimises THROUGHPUT (items per second)."""
    out = []
    for piece in chunks(texts, chunk_size):
        out.extend(model.predict(piece))            # one vectorized call per chunk
    return out


if __name__ == "__main__":
    train()
    model, _ = load_artifact()
    texts, _ = get_training_data(n=2000, seed=1)    # pretend: 2000 new items to score

    t0 = time.perf_counter(); rt = realtime(model, texts); rt_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); bt = batch(model, texts);    bt_ms = (time.perf_counter() - t0) * 1000
    print(f"real-time: {len(rt)} predictions, one call each   -> {rt_ms:7.1f} ms total, {rt_ms/len(rt):.3f} ms per item")
    print(f"batch    : {len(bt)} predictions, chunks of 500   -> {bt_ms:7.1f} ms total, {bt_ms/len(bt):.3f} ms per item"
          f"   ({rt_ms/bt_ms:.0f}x faster)")
    assert rt == bt                                  # same model, same answers; only the calling pattern differs

    out_file = Path(tempfile.mkdtemp()) / "scored.csv"
    with open(out_file, "w", newline="") as f:       # newline="" is the csv module's required setting
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(zip(texts, bt))             # zip pairs each text with its label
    print("batch results written to", out_file)

    print("""
WHEN TO USE WHICH
  real-time : user is waiting (chatbot, fraud check at checkout)   -> small p95 latency, always-on service, autoscale
  batch     : nobody is waiting (nightly recommendations, reports) -> big throughput, cheap scheduled job, retry-able
  near-real-time / micro-batch: collect requests for ~10-50 ms and score them together (what vLLM/Triton do for GPUs)
""")
    # TRY IT: change chunk_size to 10, 100, 2000 and plot/print how throughput changes.
    # TRY IT: use n=20000 and watch real-time grow linearly while batch stays small.
