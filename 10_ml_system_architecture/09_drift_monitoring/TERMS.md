# Terms — Step 09: Drift monitoring

Folder: `10_ml_system_architecture/09_drift_monitoring` · Files covered: `main.py`, `WALKTHROUGH.md`
Back to module: [10 ML system architecture](../README.md)

**Terms in this folder:** [Data drift](#data-drift-concept) · [Reference window vs live window](#reference-window-vs-live-window-concept) · [PSI (Population Stability Index)](#psi-population-stability-index-concept) · [KS statistic](#ks-statistic-concept) · [Drift threshold and alert](#drift-threshold-and-alert-concept) · [Train/serve skew vs drift](#trainserve-skew-vs-drift-concept) · [np.quantile](#npquantile-third-party-numpy) · [np.searchsorted](#npsearchsorted-third-party-numpy) · [np.bincount](#npbincount-third-party-numpy)

---

## Data drift  `(concept)`
**1. Definition:** A change over time in the distribution of the model's input features (or predictions) compared with what it saw at training or launch.
**2. In plain words:** The model did not change, the world did. Users write longer texts or shout, and the model is now judging inputs unlike its training data, so it gets worse without raising any error.
**3. Example to quote:**
> "Week 2 had 60% long texts instead of 20%, so word-count PSI jumped to 0.62 and my alert fired."

**Seen in:** `report()` in `main.py`

## Reference window vs live window  `(concept)`
**1. Definition:** The reference is a sample of inputs from a healthy period (training or launch); the live window is the most recent traffic. Drift is measured between them.
**2. In plain words:** You need a "normal" to compare against. Here the first 600 logged rows are the reference and later blocks of 600 rows are the live windows.
**3. Example to quote:**
> "I read rows 1 to 600 from the prediction log as the reference and 1201 to 1800 as the live window."

**Seen in:** `features_from_log()`

## PSI (Population Stability Index)  `(concept)`
**1. Definition:** `sum((live% - ref%) * ln(live% / ref%))` over bins whose edges come from the reference distribution; 0 means identical.
**2. In plain words:** It adds up how much each bin's share of traffic moved. Below 0.1 is calm, 0.1 to 0.25 is worth watching, above 0.25 is a real shift. Empty bins are clipped to avoid `log(0)`.
**3. Example to quote:**
> "PSI on confidence was 0.03 in the quiet week and 1.69 in the drifted week."

**Seen in:** `psi()`

## KS statistic  `(concept)`
**1. Definition:** The Kolmogorov-Smirnov statistic: the largest vertical gap between the two empirical cumulative distribution functions (0 same, 1 completely separate).
**2. In plain words:** Line up both samples from small to large and ask where the two cumulative curves are furthest apart. Compare the gap with a critical value (about `1.36 * sqrt((n+m)/(n*m))` at 5%) to decide if the difference is more than chance.
**3. Example to quote:**
> "KS for word count was 0.36 against a critical value of 0.079, so 'same distribution' is rejected."

**Seen in:** `ks_statistic()`, `ks_critical()`

## Drift threshold and alert  `(concept)`
**1. Definition:** A numeric cutoff (for example PSI above 0.25) that turns a drift score into an action such as an alert or a retraining ticket.
**2. In plain words:** A score nobody reads helps nobody. The threshold makes the monitor say ALERT. The cutoffs are rules of thumb, so tune them on your own history to avoid noisy alarms.
**3. Example to quote:**
> "I alert when PSI is above 0.25 and treat 0.10 to 0.25 as a watch state."

**Seen in:** `PSI_WARN`, `PSI_ALERT`

## Train/serve skew vs drift  `(concept)`
**1. Definition:** Drift is the data changing over time. Skew is the same data being processed differently in serving than in training (for example forgetting to lowercase).
**2. In plain words:** Different cause, similar symptom: silent accuracy loss. Skew is prevented with one shared pipeline; both are caught by watching input distributions. Skew is demonstrated in step 02 (`broken_skew.py`).
**3. Example to quote:**
> "Skew is a code difference between training and serving; drift is a world difference. I prevent skew with a shared pipeline and detect both with input monitors."

**Seen in:** `main.py` closing note, step 02

## np.quantile  `(third-party: numpy)`
**1. Definition:** `np.quantile(a, q)` returns the value below which a fraction `q` of `a` lies.
**2. In plain words:** Used to cut the reference into bins with equal numbers of values, so PSI compares like with like.
**3. Example to quote:**
> "`np.quantile(reference, np.linspace(0, 1, 11)[1:-1])` gives nine cut points for ten bins."

**Seen in:** `psi()`

## np.searchsorted  `(third-party: numpy)`
**1. Definition:** `np.searchsorted(sorted_array, values, side=...)` finds where each value would be inserted to keep the array sorted.
**2. In plain words:** It turns "which bin is this value in" into one vectorized call. `side="left"` keeps a value equal to a cut point in the lower bin, which matters when many values are identical.
**3. Example to quote:**
> "`np.searchsorted(interior, live, side='left')` gives the bin index of every live value."

**Seen in:** `psi()`, `ks_statistic()`

## np.bincount  `(third-party: numpy)`
**1. Definition:** `np.bincount(idx, minlength=n)` counts how many times each integer from 0 to n-1 occurs.
**2. In plain words:** After finding each value's bin index, bincount gives the count per bin; dividing by the total gives the percentage per bin.
**3. Example to quote:**
> "`np.bincount(bin_idx, minlength=n_bins) / len(x)` is the share of traffic in each bin."

**Seen in:** `psi()`
