# Step 09 — Drift monitoring and train/serve skew (Chapter 22, The Industrial Blueprint)

## The story so far
[Step 08](../08_scalability_and_reliability/) made the service handle more traffic. Scaling assumes the traffic looks like the traffic the model was trained on. Step 07 had a one-feature PSI taste and step 05 stored every prediction in SQLite. Here we combine them.

## The pain point
Months after launch, accuracy is falling but no alert fired: latency is fine, error rate is zero, every request returns 200. The users changed (longer reviews, shouting, new topics) and the model is now guessing outside what it learned. You find out from an angry email.

## The spark
Keep a reference window (inputs at launch) and compare it with a live window from the prediction log, feature by feature, using a number. PSI (Population Stability Index) and the KS statistic each measure "how different are these two distributions". Pick a threshold, print an alert, and now the pager rings before the user does.

## Goal
Read logged inputs from the step 05 `PredictionLog`, turn them into features (word count, uppercase ratio, confidence), compute PSI and KS against the reference, and raise an alert past a threshold. Also see how train/serve skew relates.

## Run it
```bash
cd 10_ml_system_architecture/09_drift_monitoring
python main.py
```
Real output:
```
rows in the prediction log: 1800
thresholds: PSI > 0.1 watch, PSI > 0.25 alert; KS > critical value rejects 'same distribution'

WEEK 1 vs reference (nothing changed)
  feature         PSI     KS  KS crit  verdict
  word_count    0.018  0.033    0.079  ok
  upper_ratio   0.004  0.012    0.079  ok
  confidence    0.030  0.052    0.079  ok
  -> no alert

WEEK 2 vs reference (users changed)
  feature         PSI     KS  KS crit  verdict
  word_count    0.618  0.362    0.079  ALERT  (KS also rejects)
  upper_ratio   0.750  0.277    0.079  ALERT  (KS also rejects)
  confidence    1.691  0.487    0.079  ALERT  (KS also rejects)
  -> ALERT on ['word_count', 'upper_ratio', 'confidence']: investigate inputs, consider retraining

Training/serving SKEW is the cousin of drift: here the data is the same but the PREPROCESSING differs.
It is already demonstrated in step 02 (python ../02_training_vs_inference_pipelines/broken_skew.py):
forgetting to lowercase at serve time silently drops accuracy. The drift monitor on `upper_ratio` above
flagged week 2's shouting users too, which is why both belong in monitoring.
```

## Same thing, different names

| Term you will hear | Also called / near-synonyms | Exactly the same? | Difference in one line |
|---|---|---|---|
| Data drift | covariate shift, feature drift, input drift | Yes | Distribution of inputs P(x) changes. |
| Concept drift | concept shift, model decay | No | The input-to-label relationship P(y|x) changes even if inputs look the same. |
| Label drift | prior probability shift, target drift | Mostly | The mix of labels or predictions changes. |
| Train/serve skew | training-serving skew | No | A pipeline bug (different preprocessing) rather than the world changing. |
| PSI | population stability index, CSI (characteristic stability index) | Mostly | CSI is the same formula applied to a feature; PSI often to the score. |
| KS test | Kolmogorov-Smirnov, two-sample KS | Yes | Max CDF gap; gives significance, PSI gives magnitude. |
| Reference window | baseline, training window | Yes | The data the live data is compared against. |

## Code, step by step

### 1. Imports and the step-05 prediction log  (`main.py` lines 14–21)

```python
import sys
from pathlib import Path

import numpy as np

# Reuse the real prediction log from step 05 (same schema: id, ts, model_version, input_text, label, confidence, latency_ms).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "05_storage_and_prediction_log"))
from prediction_log import PredictionLog  # noqa: E402
```

- **In simple words:** Bring in numpy and reuse the SQLite prediction log written in step 05.
- **Exactly:** `PredictionLog` has `record(version, text, label, confidence, latency_ms)` and a `conn` (sqlite3 connection with `row_factory = sqlite3.Row`, so rows act like dicts).
- **Why it is here:** Drift monitors read from the logs a service already writes; no separate data source is invented.

### 2. Random generator and vocabulary  (`main.py` lines 23–24)

```python
rng = np.random.default_rng(0)
WORDS = "great awful hotel delivery app teacher phone game arrived late friendly slow price service".split()
```

- **In simple words:** A seeded dice-roller and a bag of 14 words to build fake reviews from.
- **Exactly:** `np.random.default_rng(0)` is the modern numpy generator; the seed makes every run identical. `WORDS` is a list of 14 strings from `.split()`.
- **Why it is here:** Reproducible numbers let the demo assert 'week 1 quiet, week 2 alerts'.

### 3. make_text: random words, optionally SHOUTING  (`main.py` lines 27–29)

```python
def make_text(n_words: int, shout: bool) -> str:
    text = " ".join(rng.choice(WORDS, size=n_words))
    return text.upper() if shout else text
```

- **In simple words:** Join random words into a sentence; make it upper case if the user is shouting.
- **Exactly:** `rng.choice(WORDS, size=n_words)` returns an array of n words; `" ".join` needs strings, which numpy str_ values are. `.upper()` changes case.
- **Why it is here:** Creates the `upper_ratio` signal that distinguishes weeks.

### 4. simulate_traffic: write fake requests into the log  (`main.py` lines 32–38)

```python
def simulate_traffic(log: PredictionLog, n: int, version: str, long_prob: float, shout_prob: float, conf_mean: float):
    """Pretend the API served n requests and logged each one (we skip the real model: only the log matters here)."""
    for _ in range(n):
        n_words = int(rng.integers(15, 40)) if rng.random() < long_prob else int(rng.integers(2, 12))
        conf = float(np.clip(rng.normal(conf_mean, 0.1), 0.5, 0.999))      # model confidence in [0.5, 1)
        log.record(version, make_text(n_words, rng.random() < shout_prob),
                   "positive" if rng.random() < 0.5 else "negative", round(conf, 3), 0.3)
```

- **In simple words:** Pretend the API served n requests and log length, confidence and label for each.
- **Exactly:** With probability `long_prob` text length is 15-39 words, else 2-11 (`rng.integers(low, high)` excludes high). Confidence is `normal(conf_mean, 0.1)` clipped to [0.5, 0.999]. Label is random 50/50; latency fixed 0.3 ms. `log.record(...)` inserts one row.
- **Why it is here:** Lets us control exactly how week 2 differs: more long texts, more shouting, lower confidence.

### 5. features_from_log: rows to numeric arrays  (`main.py` lines 41–49)

```python
def features_from_log(log: PredictionLog, first_id: int, last_id: int) -> dict:
    """Pull logged rows and turn them into numeric feature arrays (one value per request)."""
    rows = log.conn.execute("SELECT input_text, confidence FROM predictions WHERE id BETWEEN ? AND ?",
                            (first_id, last_id)).fetchall()
    return {
        "word_count":  np.array([len(r["input_text"].split()) for r in rows], dtype=float),   # shape: (n_requests,)
        "upper_ratio": np.array([sum(c.isupper() for c in r["input_text"]) / len(r["input_text"]) for r in rows]),  # shape: (n_requests,)
        "confidence":  np.array([r["confidence"] for r in rows]),                               # shape: (n_requests,)
    }
```

- **In simple words:** Read logged requests by id range and compute three numbers per request.
- **Exactly:** SQL `WHERE id BETWEEN ? AND ?` with parameters (no string formatting). Features: word count; fraction of upper-case characters (`sum(c.isupper() ...) / len`); model confidence. Each is a 1-D array of shape (n_requests,) = (600,).
- **Why it is here:** Drift tests compare distributions of numbers, so text must be turned into numeric features.

### 6. psi(): quantile-bin Population Stability Index  (`main.py` lines 52–60)

```python
def psi(reference: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: sum((live% - ref%) * ln(live% / ref%)) over bins cut from the REFERENCE."""
    interior = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))   # shape: (<= bins-1,) cut points
    n_bins = len(interior) + 1
    # side="left": a value equal to a cut point falls in the bin on its left, so a pile of zeros stays in ONE bin.
    ref_pct = np.bincount(np.searchsorted(interior, reference, side="left"), minlength=n_bins) / len(reference)   # shape: (n_bins,)
    live_pct = np.bincount(np.searchsorted(interior, live, side="left"), minlength=n_bins) / len(live)            # shape: (n_bins,)
    ref_pct, live_pct = np.clip(ref_pct, 1e-4, None), np.clip(live_pct, 1e-4, None)   # avoid log(0)
    return float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))
```

- **In simple words:** Cut the reference into equal-count bins, then compare how many live values land in each.
- **Exactly:** `np.quantile(reference, linspace(0,1,bins+1)[1:-1])` gives 9 interior cut points for 10 bins; `np.unique` merges duplicates (e.g. many identical values). `np.searchsorted(interior, x, side="left")` returns the bin index of each value; `np.bincount` counts them; divide by length for proportions. Clip at 1e-4, then PSI = sum((live-ref) * ln(live/ref)).
- **Why it is here:** Quantile bins give each reference bin about equal weight, and bins from live data would hide drift (see BREAK IT).

### 7. ks_statistic: biggest gap between two CDFs  (`main.py` lines 63–68)

```python
def ks_statistic(reference: np.ndarray, live: np.ndarray) -> float:
    """Kolmogorov-Smirnov: the biggest vertical gap between the two empirical CDFs (0 = same, 1 = disjoint)."""
    grid = np.sort(np.concatenate([reference, live]))                          # shape: (n_ref + n_live,)
    cdf_ref = np.searchsorted(np.sort(reference), grid, side="right") / len(reference)
    cdf_live = np.searchsorted(np.sort(live), grid, side="right") / len(live)
    return float(np.max(np.abs(cdf_ref - cdf_live)))
```

- **In simple words:** Line up both samples and find where their cumulative curves differ most.
- **Exactly:** `grid` is all values sorted (shape (n_ref+n_live,)). `searchsorted(sorted_sample, grid, side="right") / n` is the empirical CDF at each grid point. KS = max |CDF_ref - CDF_live| in [0, 1].
- **Why it is here:** A second, binning-free view; KS also comes with a significance threshold.

### 8. ks_critical: significance threshold  (`main.py` lines 71–73)

```python
def ks_critical(n_ref: int, n_live: int) -> float:
    """KS gap above which 'same distribution' is rejected at the 5% level."""
    return 1.36 * np.sqrt((n_ref + n_live) / (n_ref * n_live))
```

- **In simple words:** The KS gap above which 'same distribution' is unlikely.
- **Exactly:** c(alpha) * sqrt((n+m)/(n*m)) with c = 1.36 for 5%. With n = m = 600 it is 1.36 x sqrt(1200/360000) = 0.079 (matches the output).
- **Why it is here:** Turns the KS number into a yes/no test that scales with sample size.

### 9. Alert thresholds  (`main.py` lines 76–76)

```python
PSI_WARN, PSI_ALERT = 0.10, 0.25      # common rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 act
```

- **In simple words:** Two cut-offs on PSI: 0.10 means watch, 0.25 means act.
- **Exactly:** Industry rule of thumb, not a law; tune it per feature.
- **Why it is here:** An alert rule needs numbers fixed in advance.

### 10. report(): table, verdict and alerts  (`main.py` lines 79–90)

```python
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
```

- **In simple words:** For every feature print PSI, KS, KS critical value and a verdict, and collect the alerts.
- **Exactly:** Verdict `ALERT` if PSI > 0.25, `watch` if > 0.10, else `ok`; adds `(KS also rejects)` if KS > critical. Returns the list of alert feature names. Real week 2: word_count PSI 0.618, upper_ratio 0.750, confidence 1.691, all ALERT.
- **Why it is here:** The returned list could feed pager/email in real life; the print is the human view.

### 11. Main: simulate three weeks of traffic  (`main.py` lines 93–101)

```python
if __name__ == "__main__":
    log = PredictionLog()                                       # in-memory; use a file path to keep it

    # Week 0 (launch): the traffic the model was validated on. This becomes the REFERENCE window.
    simulate_traffic(log, 600, "v1", long_prob=0.2, shout_prob=0.02, conf_mean=0.85)
    # Week 1: same world, new random users. This is the quiet control.
    simulate_traffic(log, 600, "v1", long_prob=0.2, shout_prob=0.02, conf_mean=0.85)
    # Week 2: users now paste long reviews, some SHOUT, and the model is less sure.
    simulate_traffic(log, 600, "v1", long_prob=0.6, shout_prob=0.30, conf_mean=0.72)
```

- **In simple words:** Create launch traffic, a quiet week, and a changed week in the log.
- **Exactly:** 600 requests each. Week 0 and 1: 20% long, 2% shouting, confidence mean 0.85. Week 2: 60% long, 30% shouting, mean 0.72. All stored in an in-memory SQLite DB, ids 1-1800.
- **Why it is here:** The control week proves the detector does not cry wolf.

### 12. Extract features and print log size  (`main.py` lines 103–107)

```python
    reference = features_from_log(log, 1, 600)
    week1 = features_from_log(log, 601, 1200)
    week2 = features_from_log(log, 1201, 1800)
    print("rows in the prediction log:", log.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
    print(f"thresholds: PSI > {PSI_WARN} watch, PSI > {PSI_ALERT} alert; KS > critical value rejects 'same distribution'")
```

- **In simple words:** Slice the log into the three windows and show the thresholds.
- **Exactly:** ids 1-600 reference, 601-1200 week1, 1201-1800 week2. Real: `rows in the prediction log: 1800`.
- **Why it is here:** Same code path would work on a real production log.

### 13. Run both reports and assert  (`main.py` lines 109–111)

```python
    quiet = report("WEEK 1 vs reference (nothing changed)", reference, week1)
    drift = report("WEEK 2 vs reference (users changed)", reference, week2)
    assert not quiet and drift, "demo expectation: week 1 quiet, week 2 alerts"
```

- **In simple words:** Compare week 1 and week 2 with the reference and check the demo behaves.
- **Exactly:** Week 1: PSI 0.018/0.004/0.030, all ok, no alert. Week 2: all three ALERT. `assert not quiet and drift` fails loudly if the simulation or code changes.
- **Why it is here:** A self-checking demo: guards against silent regressions.

### 14. Explain skew and exercises  (`main.py` lines 113–120)

```python
    print("""
Training/serving SKEW is the cousin of drift: here the data is the same but the PREPROCESSING differs.
It is already demonstrated in step 02 (python ../02_training_vs_inference_pipelines/broken_skew.py):
forgetting to lowercase at serve time silently drops accuracy. The drift monitor on `upper_ratio` above
flagged week 2's shouting users too, which is why both belong in monitoring.""")
    # TRY IT: set week 2's shout_prob back to 0.02 and watch upper_ratio go from ALERT to ok.
    # TRY IT: change bins=10 to bins=3 in psi() and see how the PSI values move.
    # BREAK IT: compute PSI with bins cut from the LIVE data instead of the reference; a drifted week now looks small.
```

- **In simple words:** A printed note that drift and train/serve skew are cousins, plus exercises.
- **Exactly:** Skew = same data, different preprocessing (step 02 `broken_skew.py`). The `# TRY IT:` and `# BREAK IT:` lines are comments.
- **Why it is here:** Both belong in monitoring; the upper_ratio feature catches shouting users either way.


## Syntax explained
| Syntax | Meaning | Example |
|---|---|---|
| `np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1])` | Cut points that put equal reference mass in each bin | `psi()` |
| `np.searchsorted(cuts, x, side="left")` | Which bin each value falls into | `psi()` |
| `np.bincount(idx, minlength=n_bins)` | Count values per bin | `psi()` |
| `np.clip(pct, 1e-4, None)` | Avoid `log(0)` for empty bins | `psi()` |
| `np.max(np.abs(cdf_ref - cdf_live))` | KS: the largest gap between two CDFs | `ks_statistic()` |
| `WHERE id BETWEEN ? AND ?` | Select a window of logged rows with a parameterized query | `features_from_log()` |

## Libraries & functions
| Name | Kind | What it does |
|---|---|---|
| `PredictionLog` | step 05 code | SQLite log with `input_text`, `confidence`, `model_version`, ... |
| `psi(reference, live)` | our code | `sum((live% - ref%) * ln(live% / ref%))` over reference bins |
| `ks_statistic(reference, live)` | our code | Max CDF gap, 0 (same) to 1 (disjoint) |
| `ks_critical(n_ref, n_live)` | our code | `1.36 * sqrt((n+m)/(n*m))`: gap above this rejects "same distribution" at 5% |

## Thresholds
| PSI | Reading |
|---|---|
| below 0.10 | stable |
| 0.10 to 0.25 | watch |
| above 0.25 | alert: investigate, maybe retrain |

These are rules of thumb, not laws; tune them on your own history. KS with very large samples rejects tiny, harmless shifts, so many teams alert on PSI and use KS as a second opinion.

## Execution trace
600 rows are logged for launch (reference), 600 for week 1 (same generator, new random draws), 600 for week 2 (60% long texts instead of 20%, 30% shouting instead of 2%, lower confidence). `features_from_log` reads each id range from SQLite and builds arrays of shape `(600,)` per feature. For each feature `psi` cuts bins from the reference, counts reference and live per bin, and sums the weighted log ratios. Week 1 gives PSI under 0.04 everywhere (no alert). Week 2 gives PSI far above 0.25 on all three features, and the script prints the alert list.

## Train/serve skew (same family, different cause)
Drift: the **data** changes. Skew: the **same data** is processed differently at serving time than at training time. Step 02 already demonstrates skew: [broken_skew.py](../02_training_vs_inference_pipelines/broken_skew.py) trains with lowercase text, serves shouting text without lowercasing, and accuracy drops with no error. Do not duplicate it; run it and then notice that the `upper_ratio` monitor above would also have flagged the shift. Prevention: one shared pipeline object; detection: input-distribution monitors like this one.

## Experiments
1. Set week 2's `shout_prob` back to `0.02`: `upper_ratio` goes from ALERT to ok while the other two stay alerted.
2. Change `bins=10` to `bins=3` in `psi()` and compare numbers: PSI depends on the binning.
3. Keep the log in a file (`PredictionLog("preds.db")`) and compute the window by `ts` instead of by id.

## Debugging practice
- Cut the bins from the **live** data instead of the reference: the drifted week now looks small because the bins follow the drift. Bins must come from the reference.
- Reference with a pile of identical values (most `upper_ratio` values are 0): naive quantile edges collapse and PSI reads 0.000 even though the shift is huge. My first version of this script printed `upper_ratio PSI 0.000` for week 2 for exactly this reason; the fix is to use interior cut points with `side="left"` so the zeros stay in one bin.
- Empty bin with no clipping gives `log(0)`, which is `-inf` and a `RuntimeWarning`.

## Rebuild checklist
- [ ] Reference window versus live window, read from the prediction log.
- [ ] PSI with bins from the reference and clipped percentages.
- [ ] KS as the biggest CDF gap, with the critical value.
- [ ] A threshold that prints an ALERT.
- [ ] A one-line explanation of drift versus skew.

## The new problem (cliffhanger)
We can now design, serve, scale and watch a model, and it all works on this laptop with this exact Python and this exact folder. Move it to a teammate's machine or a server and it breaks on missing libraries and different versions. We need to ship the whole room, not just the files. Next: [../../11_docker/](../../11_docker/).

## Say it aloud
> **Say it aloud (interview box):** I keep a reference window of inputs from launch, compare it with a live window from the prediction log feature by feature, and compute PSI or the KS statistic against a threshold such as PSI above 0.25 to raise an alert. Drift means the data changed, while train/serve skew means the same data was preprocessed differently, and both can be caught by monitoring input distributions. Thresholds need tuning, bins must come from the reference, and drift does not prove accuracy fell until labels arrive.
