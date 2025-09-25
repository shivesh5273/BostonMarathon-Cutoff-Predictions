# check_buffer.py — quick verdict + per-model probabilities at a given buffer
# Works with: per_group_2026_cutoff_with_bands.csv + buffer_probability_table.csv
# Run with args (CLI): python check_buffer.py M 18-34 7:00
# Or just click "Run" in PyCharm (uses DEFAULT_* below)

import sys
import numpy as np
import pandas as pd

# -------- Defaults for click-to-run (used if you don't pass CLI args) --------
DEFAULT_GENDER = "M"        # "M" | "F" | "X"
DEFAULT_BQ_AGE = "18-34"    # e.g., "18-34","40-44",...
DEFAULT_BUFFER = "7:00"     # mm:ss

BANDS_CSV = "per_group_2026_cutoff_with_bands.csv"
PROB_CSV  = "buffer_probability_table.csv"

CALIBRATED_MODELS = {
    "LogisticRegression",
    "LinearSVM(calibrated)",
    "RandomForest(calibrated)",
}

def mmss_to_sec(s: str) -> int:
    s = str(s)
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)

def sec_to_mmss(x: float) -> str:
    x = int(round(float(x)))
    return f"{x//60}:{x%60:02d}"

def status_from_band(buf_sec: int, lo_sec: int, point_sec: float, hi_sec: int) -> str:
    if buf_sec >= hi_sec:
        return "GREEN (very likely)"
    if buf_sec >= point_sec:
        return "YELLOW (likely)"
    if buf_sec >= lo_sec:
        return "ORANGE (borderline)"
    return "RED (unlikely)"

def interp_prob(series: pd.Series, x: int) -> float:
    """Linear interpolation on a Series indexed by buffer_sec."""
    if x in series.index:
        return float(series.loc[x])
    idx = series.index.values
    i = np.searchsorted(idx, x)
    if i == 0:
        return float(series.iloc[0])
    if i == len(idx):
        return float(series.iloc[-1])
    x0, x1 = idx[i-1], idx[i]
    y0, y1 = float(series.loc[x0]), float(series.loc[x1])
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))

def main():
    # -------- args or defaults --------
    if len(sys.argv) == 4:
        gender, bq_age, buf_str = sys.argv[1], sys.argv[2], sys.argv[3]
    else:
        gender, bq_age, buf_str = DEFAULT_GENDER, DEFAULT_BQ_AGE, DEFAULT_BUFFER

    buf_sec = mmss_to_sec(buf_str)

    # -------- read bands (cutoff point + ±1σ) --------
    bands = pd.read_csv(BANDS_CSV)
    match = bands[(bands["Gender"] == gender) & (bands["BQ_Age"] == bq_age)]
    if match.empty:
        raise ValueError(f"No band row found for [{gender} {bq_age}] in {BANDS_CSV}")

    row = match.iloc[0]
    point_sec = float(row["cutoff_2026_sec"])

    # lo/hi are stored as mm:ss strings in your CSV:
    lo_sec  = mmss_to_sec(str(row["cutoff_lo_mmss"]))
    hi_sec  = mmss_to_sec(str(row["cutoff_hi_mmss"]))

    verdict = status_from_band(buf_sec, lo_sec, point_sec, hi_sec)

    # -------- read model probabilities table --------
    df = pd.read_csv(PROB_CSV)
    group = df[(df["gender"] == gender) & (df["BQ_Age"] == bq_age)].copy()
    if group.empty:
        raise ValueError(f"No probability rows found for [{gender} {bq_age}] in {PROB_CSV}")

    # build per-model Series indexed by buffer_sec for interpolation
    probs = {}
    for model, g in group.groupby("model"):
        s = g.set_index("buffer_sec")["prob_accept"].sort_index()
        probs[model] = interp_prob(s, buf_sec)

    # calibrated ensemble (average of calibrated models only; fallback to all if missing)
    cal_set = [probs[m] for m in probs if m in CALIBRATED_MODELS]
    if len(cal_set) == 0:
        ensemble = float(np.mean(list(probs.values())))
        ensemble_note = "(avg of all models)"
    else:
        ensemble = float(np.mean(cal_set))
        ensemble_note = "(avg of calibrated models)"

    # -------- print nicely --------
    print(f"\n[{gender} {bq_age}] Buffer {buf_str}")
    print(f"Band verdict: {verdict}  |  cutoff point {sec_to_mmss(point_sec)}  "
          f"(band {sec_to_mmss(lo_sec)}–{sec_to_mmss(hi_sec)})")
    print(f"Ensemble P(accepted) ≈ {ensemble:0.3f} {ensemble_note}")

    for k, v in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s}  P(accepted) ≈ {v:0.3f}")

if __name__ == "__main__":
    main()