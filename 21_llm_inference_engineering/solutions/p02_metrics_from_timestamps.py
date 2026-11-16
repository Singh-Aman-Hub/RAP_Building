"""Exercise P2 solution: TTFT / ITL / TPOT / p95 / goodput from raw token timestamps (synthetic, hand-checkable)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import benchlib

# one request: starts at t=0, tokens at these times (seconds). Token 4 is a slow outlier.
stamps = [0.200, 0.220, 0.240, 0.360, 0.380]
trial = dict(t0=0.0, t_prefill_end=0.200, token_times=stamps, batch=1, prompt_len=10)
s = benchlib.summarize_trial(trial)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "itl_s"}, [round(x, 3) for x in s["itl_s"]])
assert abs(s["ttft_s"] - 0.200) < 1e-9
assert [round(x, 3) for x in s["itl_s"]] == [0.02, 0.02, 0.12, 0.02]
assert abs(s["tpot_s"] - 0.045) < 1e-9                      # (0.380 - 0.200) / 4 = mean ITL
assert abs(s["tok_per_s"] - 5 / 0.380) < 1e-9
p95 = benchlib.percentile(s["itl_s"], 95)
print(f"ITL p95 = {p95 * 1000:.0f} ms although TPOT = {s['tpot_s'] * 1000:.0f} ms: the mean hides the stall")
assert abs(p95 - 0.105) < 1e-9                               # interpolate between 0.02 and 0.12 at position 2.85
SLO_TTFT, SLO_TPOT = 0.250, 0.040
meets = s["ttft_s"] <= SLO_TTFT and s["tpot_s"] <= SLO_TPOT
print("meets SLO (TTFT<=250 ms, TPOT<=40 ms):", meets, "-> goodput counts 0 tokens from this request")
assert meets is False
print("OK")
