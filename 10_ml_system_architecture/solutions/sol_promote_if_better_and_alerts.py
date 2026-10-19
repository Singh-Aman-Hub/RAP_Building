"""
Worked solutions: (1) promote only if the new model beats production on log_loss (step 04),
(2) a simple alert rule from latency list + error counts (step 07).
RUN: cd 10_ml_system_architecture/solutions && python sol_promote_if_better_and_alerts.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "04_model_registry_versioning"))
from registry import ModelRegistry  # noqa: E402
from main import train_and_register  # noqa: E402  (reuses the step-04 helper that trains + saves)


def promote_if_better(reg: ModelRegistry, candidate: str, alias: str = "production") -> bool:
    """Promote `candidate` only if it has a lower log_loss than the current alias target (or no alias exists)."""
    cand_loss = reg.load(candidate)[1]["metrics"]["log_loss"]
    try:
        current = reg.resolve(alias)                      # raises KeyError if the alias was never set
    except KeyError:
        reg.promote(candidate, alias)
        return True
    cur_loss = reg.load(current)[1]["metrics"]["log_loss"]
    if cand_loss < cur_loss:
        reg.promote(candidate, alias)
        return True
    return False


def check_alerts(latencies_ms: list[float], errors: int, requests: int) -> list[str]:
    alerts = []
    if np.percentile(latencies_ms, 95) > 50:
        alerts.append("p95 latency above 50 ms")
    if requests and errors / requests > 0.02:
        alerts.append("error rate above 2%")
    return alerts


if __name__ == "__main__":
    root = Path(tempfile.mkdtemp())
    reg = ModelRegistry(root)
    v1 = train_and_register(reg, n=20, C=0.1)
    v2 = train_and_register(reg, n=300, C=10.0)
    print("promote v1 (no production yet):", promote_if_better(reg, v1))
    print("promote v2 (better log_loss)  :", promote_if_better(reg, v2))
    v3 = train_and_register(reg, n=10, C=0.01)
    print("promote v3 (worse)            :", promote_if_better(reg, v3), "-> production is", reg.resolve("production"))
    print("alerts:", check_alerts([5, 6, 7, 80, 90], errors=3, requests=100))
    shutil.rmtree(root)
