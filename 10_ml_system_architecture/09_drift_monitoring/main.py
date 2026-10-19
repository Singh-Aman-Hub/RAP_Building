"""
GOAL:   Notice that the world has changed BEFORE users complain. Compare the inputs the model saw
        at launch (reference) with the inputs it sees now (live), using two drift statistics
        (PSI and KS), and raise an alert when a threshold is crossed.

RUN:    cd 10_ml_system_architecture/09_drift_monitoring && python main.py

EXPECTED OUTPUT (shape): a table per feature with PSI, KS and a verdict for a quiet week (no alert)
        and a drifted week (alerts), read from the SQLite prediction log of step 05.

NEEDS:  numpy (the log itself uses the built-in sqlite3)
"""

import sys
from pathlib import Path

import numpy as np

# Reuse the real prediction log from step 05 (same schema: id, ts, model_version, input_text, label, confidence, latency_ms).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_storage_and_prediction_log"))
from prediction_log import PredictionLog  # noqa: E402

rng = np.random.default_rng(0)
WORDS = "great awful hotel delivery app teacher phone game arrived late friendly slow price service".split()


def make_text(n_words: int, shout: bool) -> str:
    text = " ".join(rng.choice(WORDS, size=n_words))
    return text.upper() if shout else text


def simulate_traffic(log: PredictionLog, n: int, version: str, long_prob: float, shout_prob: float, conf_mean: float):
    """Pretend the API served n requests and logged each one (we skip the real model: only the log matters here)."""
    for _ in range(n):
        n_words = int(rng.integers(15, 40)) if rng.random() < long_prob else int(rng.integers(2, 12))
        conf = float(np.clip(rng.normal(conf_mean, 0.1), 0.5, 0.999))      # model confidence in [0.5, 1)
        log.record(version, make_text(n_words, rng.random() < shout_prob),
                   "positive" if rng.random() < 0.5 else "negative", round(conf, 3), 0.3)


def features_from_log(log: PredictionLog, first_id: int, last_id: int) -> dict:
    """Pull logged rows and turn them into numeric feature arrays (one value per request)."""
    rows = log.conn.execute("SELECT input_text, confidence FROM predictions WHERE id BETWEEN ? AND ?",
                            (first_id, last_id)).fetchall()
    return {
        "word_count":  np.array([len(r["input_text"].split()) for r in rows], dtype=float),   # shape: (n_requests,)
        "upper_ratio": np.array([sum(c.isupper() for c in r["input_text"]) / len(r["input_text"]) for r in rows]),  # shape: (n_requests,)
        "confidence":  np.array([r["confidence"] for r in rows]),                               # shape: (n_requests,)
    }


def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: sum((live% - ref%) * ln(live% / ref%)) over bins cut from the REFERENCE."""
    interior = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))   # shape: (<= bins-1,) cut points
    n_bins = len(interior) + 1
    # side="left": a value equal to a cut point falls in the bin on its left, so a pile of zeros stays in ONE bin.
    ref_pct = np.bincount(np.searchsorted(interior, reference, side="left"), minlength=n_bins) / len(reference)   # shape: (n_bins,)
    live_pct = np.bincount(np.searchsorted(interior, live, side="left"), minlength=n_bins) / len(live)            # shape: (n_bins,)
    ref_pct, live_pct = np.clip(ref_pct, 1e-4, None), np.clip(live_pct, 1e-4, None)   # avoid log(0)
    return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))


def ks_statistic(reference: np.ndarray, live: np.ndarray) -> float:
    """Kolmogorov-Smirnov: the biggest vertical gap between the two empirical CDFs (0 = same, 1 = disjoint)."""
    grid = np.sort(np.concatenate([reference, live]))                          # shape: (n_ref + n_live,)
    cdf_ref = np.searchsorted(np.sort(reference), grid, side="right") / len(reference)
    cdf_live = np.searchsorted(np.sort(live), grid, side="right") / len(live)
    return float(np.max(np.abs(cdf_ref - cdf_live)))


def ks_critical(n_ref: int, n_live: int) -> float:
    """KS gap above which 'same distribution' is rejected at the 5% level."""
    return 1.36 * np.sqrt((n_ref + n_live) / (n_ref * n_live))


PSI_WARN, PSI_ALERT = 0.10, 0.25      # common rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 act


def report(title: str, ref: dict, live: dict) -> list:
    print(f"\n{title}")
    print(f"  {'feature':<12} {'PSI':>6} {'KS':>6} {'KS crit':>8}  verdict")
    alerts = []
    for name in ref:
        p, k, kc = psi(ref[name], live[name]), ks_statistic(ref[name], live[name]), ks_critical(len(ref[name]), len(live[name]))
        verdict = "ALERT" if p > PSI_ALERT else "watch" if p > PSI_WARN else "ok"
        print(f"  {name:<12} {p:6.3f} {k:6.3f} {kc:8.3f}  {verdict}{'  (KS also rejects)' if k > kc else ''}")
        if verdict == "ALERT":
            alerts.append(name)
    print("  ->", f"ALERT on {alerts}: investigate inputs, consider retraining" if alerts else "no alert")
    return alerts


if __name__ == "__main__":
    log = PredictionLog()                                       # in-memory; use a file path to keep it

    # Week 0 (launch): the traffic the model was validated on. This becomes the REFERENCE window.
    simulate_traffic(log, 600, "v1", long_prob=0.2, shout_prob=0.02, conf_mean=0.85)
    # Week 1: same world, new random users. This is the quiet control.
    simulate_traffic(log, 600, "v1", long_prob=0.2, shout_prob=0.02, conf_mean=0.85)
    # Week 2: users now paste long reviews, some SHOUT, and the model is less sure.
    simulate_traffic(log, 600, "v1", long_prob=0.6, shout_prob=0.30, conf_mean=0.72)

    reference = features_from_log(log, 1, 600)
    week1 = features_from_log(log, 601, 1200)
    week2 = features_from_log(log, 1201, 1800)
    print("rows in the prediction log:", log.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
    print(f"thresholds: PSI > {PSI_WARN} watch, PSI > {PSI_ALERT} alert; KS > critical value rejects 'same distribution'")

    quiet = report("WEEK 1 vs reference (nothing changed)", reference, week1)
    drift = report("WEEK 2 vs reference (users changed)", reference, week2)
    assert not quiet and drift, "demo expectation: week 1 quiet, week 2 alerts"

    print("""
Training/serving SKEW is the cousin of drift: here the data is the same but the PREPROCESSING differs.
It is already demonstrated in step 02 (python ../02_training_vs_inference_pipelines/broken_skew.py):
forgetting to lowercase at serve time silently drops accuracy. The drift monitor on `upper_ratio` above
flagged week 2's shouting users too, which is why both belong in monitoring.""")
    # TRY IT: set week 2's shout_prob back to 0.02 and watch upper_ratio go from ALERT to ok.
    # TRY IT: change bins=10 to bins=3 in psi() and see how the PSI values move.
    # BREAK IT: compute PSI with bins cut from the LIVE data instead of the reference; a drifted week now looks small.
