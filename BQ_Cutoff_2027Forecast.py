from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATA = Path(".")
GRID_CSV = DATA / "per_group_actual_times_BQ_bins.csv"   # from your main/scratch
CUTOFFS  = DATA / "boston_bq_cutoffs_2012_2026.csv"      # official buffers per year

OUT_TREND    = DATA / "m1834_year_trend.png"
OUT_FORECAST = DATA / "m1834_forecast_2027.png"

FIT_START, FIT_END = 2014, 2025
EXCLUDE_YEARS = {2021, 2022, 2023}   # set to set() to include all

def mm(x):  # seconds -> minutes (float)
    return np.asarray(x, dtype=float) / 60.0

def mmss(sec):
    sec = int(round(float(sec)))
    return f"{sec//60}:{sec%60:02d}"

def main():
    # ---------- load ----------
    if not GRID_CSV.exists():
        raise FileNotFoundError(f"Missing {GRID_CSV}. Run your main/scratch first.")
    df = pd.read_csv(GRID_CSV)

    # Expect these columns: race_year, Gender_norm, BQ_Age, Standard_sec, ActualQualifyingTime_sec
    need = {"race_year","Gender_norm","BQ_Age","Standard_sec","ActualQualifyingTime_sec"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{GRID_CSV} missing columns: {sorted(miss)}")

    # Slice: Men 18–34
    sub = df[(df["Gender_norm"]=="M") & (df["BQ_Age"]=="18-34")].copy()
    if sub.empty:
        raise RuntimeError("No rows for M 18–34")

    # Keep fit window & drop excluded years
    fit = sub[sub["race_year"].between(FIT_START, FIT_END)]
    if EXCLUDE_YEARS:
        fit = fit[~fit["race_year"].isin(EXCLUDE_YEARS)]
    if fit.empty:
        raise RuntimeError("Fit slice empty; adjust FIT_* or EXCLUDE_YEARS.")

    # Linear trend: year -> ActualQualifyingTime_sec
    X = fit[["race_year"]].values
    y = fit["ActualQualifyingTime_sec"].values
    model = LinearRegression().fit(X, y)

    # ------------- predictions -------------
    xs = np.arange(FIT_START, 2027).reshape(-1,1)
    yhat = model.predict(xs)

    pred_2026 = float(model.predict(np.array([[2026]]))[0])
    pred_2027 = float(model.predict(np.array([[2027]]))[0])

    # ------------- plots -------------
    # Trend (through 2026, 2026 highlighted)
    plt.figure(figsize=(10,6))
    plt.scatter(fit["race_year"], mm(fit["ActualQualifyingTime_sec"]),
                s=40, label="Actual (minutes)")
    plt.plot(xs.ravel(), mm(yhat), "--", label="Trend")
    plt.axvline(2026, color="0.85", lw=10, alpha=0.25)   # faint band at 2026
    plt.title("Men 18–34: Year → Actual Qualifying Time")
    plt.xlabel("Year"); plt.ylabel("Minutes")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_TREND, dpi=160)
    plt.close()

    # Forecast figure (mark 2027)
    plt.figure(figsize=(10,6))
    plt.scatter(fit["race_year"], mm(fit["ActualQualifyingTime_sec"]),
                s=40, label="Actual (minutes)")
    plt.plot(xs.ravel(), mm(yhat), "--", label="Trend")
    plt.axvline(2026, color="0.85", lw=10, alpha=0.25)
    plt.axvline(2027, color="gray", linestyle=":", alpha=0.7)
    plt.title("Men 18–34: Year → Actual Qualifying Time (with 2027 forecast)")
    plt.xlabel("Year"); plt.ylabel("Minutes")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_FORECAST, dpi=160)
    plt.close()

    # ------------- residual SD for context -------------
    resid = y - model.predict(X)
    sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else float("nan")
    lo_2026, hi_2026 = pred_2026 - sd, pred_2026 + sd

    # ------------- buffers -------------
    # Get the M18–34 BQ standard seconds (use 2026 row; standards are fixed by bin)
    std_2026 = float(sub.loc[sub["race_year"]==2026, "Standard_sec"].iloc[0])

    # Official 2026 *buffer* from your CUTOFFS file (field-wide)
    buf_off_2026 = None
    if CUTOFFS.exists():
        cuts = pd.read_csv(CUTOFFS)
        col = "cutoff_seconds" if "cutoff_seconds" in cuts.columns else "cutoff_mmss"
        if col == "cutoff_mmss":
            # parse m:ss
            pads = cuts["cutoff_mmss"].astype(str).str.split(":")
            cuts["cutoff_seconds"] = pads.map(lambda xs: int(xs[0])*60 + int(xs[1]))
        buf_off_2026 = float(cuts.loc[cuts["race_year"]==2026, "cutoff_seconds"].iloc[0])

    # Model-predicted buffers (vs the M18–34 standard):
    #   buffer = Standard_sec - predicted_actual_sec
    buf_pred_2026 = std_2026 - pred_2026
    buf_pred_2027 = std_2026 - pred_2027   # assume 2027 uses same standard as 2026

    # Uncertainty (same SD applies to buffer)
    buf_2026_lo = std_2026 - (pred_2026 + sd)
    buf_2026_hi = std_2026 - (pred_2026 - sd)
    buf_2027_lo = std_2026 - (pred_2027 + sd)
    buf_2027_hi = std_2026 - (pred_2027 - sd)

    print(f"→ 2026 buffer band (+/- 1 STD): [{mmss(buf_2026_lo)} , {mmss(buf_2026_hi)}]")
    print(f"→ 2027 buffer band (+/- 1 STD): [{mmss(buf_2027_lo)} , {mmss(buf_2027_hi)}]")


    # ------------- console summary -------------
    print(f"Saved plot → {OUT_TREND.resolve()}")
    print(f"Saved plot → {OUT_FORECAST.resolve()}")
    print(f"Fit years: {int(fit['race_year'].min())}-{int(fit['race_year'].max())}  "
          f"Excluded: {sorted(EXCLUDE_YEARS) if EXCLUDE_YEARS else 'none'}")
    print(f"Predicted actual qualifying time for 2026: {mmss(pred_2026)} "
          f"(+ or - 1 STD = [{mmss(lo_2026)} , {mmss(hi_2026)}])")
    print(f"Predicted actual qualifying time for 2027: {mmss(pred_2027)}")

    if buf_off_2026 is not None:
        print(f"2026 official cutoff buffer in dataset: {mmss(buf_off_2026)}")

    print(f"→ Model-predicted buffer for 2026 (vs 2026 standard): {mmss(buf_pred_2026)}")
    print(f"→ Model-predicted buffer for 2027 (vs 2026 standard): {mmss(buf_pred_2027)}")

    delta = buf_pred_2027 - buf_pred_2026
    sign = "+" if delta >= 0 else "−"
    print(f"→ Change in buffer 2027 vs 2026 (model): {sign}{mmss(abs(delta))}")

if __name__ == "__main__":
    main()