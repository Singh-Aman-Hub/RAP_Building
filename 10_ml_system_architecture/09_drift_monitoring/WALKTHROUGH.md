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
