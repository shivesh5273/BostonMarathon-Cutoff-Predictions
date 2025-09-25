

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, brier_score_loss,
    classification_report, roc_curve, RocCurveDisplay
)
from sklearn.linear_model import Perceptron, SGDRegressor, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------- CONFIG --------------------------
DATA = Path(".")
RESULTS_CSV = DATA / "results_individual.csv"                     # runner-level rows
PER_GROUP_ACTUALS = DATA / "per_group_actual_times_BQ_bins.csv"   # from main.py

OUT_EVAL_TABLE   = DATA / "model_eval_table.csv"
OUT_REPORT_MD    = DATA / "model_report_for_runners.md"
OUT_BUFFER_TABLE = DATA / "buffer_probability_table.csv"

PLOTS_DIR = DATA / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Auto-generate a demo dataset if results_individual.csv is missing
AUTOGEN_DEMO_IF_MISSING = True
AUTOGEN_YEARS = list(range(2019, 2027))           # 2019..2026 if available
AUTOGEN_PER_GROUP_PER_YEAR = 120                  # rows per (gender, BQ_Age, year)

# Which buffer→probability curves to generate
BUFFER_GROUPS = [
    ("M", "18-34"),
    ("F", "18-34"),
    ("M", "40-44"),
]
BUFFER_RANGE_SEC = np.arange(0, 11*60 + 1, 30)  # 0:00..11:00 in 30s steps

# Column mapping if your results CSV uses different column names
COLUMN_MAP = {
    "race_year": "race_year",
    "gender": "gender",
    "age": "age",
    "finish_time": "finish_time",
}

AGE_FROM_BRACKET = False  # set True if you have an age_bracket column and map it below

# -------------------------- HELPERS --------------------------
def parse_time_to_sec(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)) and np.isfinite(x): return float(x)
    s = str(x).strip()
    if re.fullmatch(r"\d+:\d{2}:\d{2}", s):
        h, m, s2 = map(int, s.split(":")); return h*3600 + m*60 + s2
    if re.fullmatch(r"\d+:\d{2}", s):
        m, s2 = map(int, s.split(":")); return m*60 + s2
    try: return float(s)
    except: return np.nan

def sec_to_hms(sec):
    sec = max(0, int(round(float(sec))))
    h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
    return f"{h}:{m:02d}:{s:02d}"

def normalize_gender(g):
    if pd.isna(g): return np.nan
    g = str(g).strip().upper()
    if g.startswith("M"): return "M"
    if g.startswith("F"): return "F"
    if g.startswith("X"): return "X"
    return g

def age_to_bq_bin(age):
    if pd.isna(age): return np.nan
    a = int(age)
    if a < 18: return np.nan
    if a <= 34: return "18-34"
    for lo in [35,40,45,50,55,60,65,70,75]:
        if lo <= a <= lo+4: return f"{lo}-{lo+4}"
    return "80+"

def age_bracket_to_bq_bin(s):
    if pd.isna(s): return np.nan
    t = str(s).replace("\u2013","-").replace("\u2014","-").replace("\xa0"," ").strip().lower()
    t = re.sub(r"\s+"," ", t)
    if "under 20" in t: return np.nan
    if "80+" in t or "80 and" in t: return "80+"
    m = re.search(r"(\d{2})\s*-\s*(\d{2})", t)
    if m:
        lo, hi = map(int, m.groups())
        if lo <= 34 and hi >= 18: return "18-34"
        return f"{lo}-{hi}"
    return np.nan

def fmt_pct(x): return f"{100*x:.1f}%"
def mmss(x): x = int(round(float(x))); return f"{x//60}:{x%60:02d}"

# -------------------------- LOAD / AUTOGEN --------------------------
def load_actual_thresholds():
    if not PER_GROUP_ACTUALS.exists():
        raise FileNotFoundError(
            f"Missing thresholds file: {PER_GROUP_ACTUALS}\n"
            "Run your main.py first to produce per_group_actual_times_BQ_bins.csv"
        )
    thr = pd.read_csv(PER_GROUP_ACTUALS)
    # Expected columns:
    #   race_year, Gender_norm, BQ_Age, ActualQualifyingTime_sec
    if "Gender_norm" not in thr.columns and "Gender" in thr.columns:
        thr = thr.rename(columns={"Gender": "Gender_norm"})
    need = ["race_year","Gender_norm","BQ_Age","ActualQualifyingTime_sec"]
    for c in need:
        if c not in thr.columns:
            raise ValueError(f"Thresholds CSV missing column: {c}")
    thr = thr[need].rename(columns={"Gender_norm":"gender_norm"})
    return thr

def _sample_age_from_bin(bq_age, rng):
    if bq_age == "18-34": return rng.integers(18, 35)
    if bq_age == "80+":   return rng.integers(80, 86)
    m = re.match(r"(\d{2})-(\d{2})", str(bq_age))
    if m:
        lo, hi = map(int, m.groups())
        return rng.integers(lo, hi+1)
    return rng.integers(25, 46)

def autogen_demo_results(thr: pd.DataFrame, out_path: Path):
    rng = np.random.default_rng(7)
    rows = []
    yrs = sorted(set(thr["race_year"]).intersection(AUTOGEN_YEARS))
    thr2 = thr[thr["gender_norm"].isin(["M","F"]) & thr["BQ_Age"].notna()].copy()

    for y in yrs:
        suby = thr2[thr2["race_year"]==y]
        for _, r in suby.iterrows():
            g = r["gender_norm"]; a = r["BQ_Age"]; thr_sec = float(r["ActualQualifyingTime_sec"])
            # Draw buffers ~N(center, 160^2) around threshold; slight drift by year
            center = 270 + (y-2019)*10   # ≈ 4:30 plus tiny trend
            buf = rng.normal(loc=center, scale=160, size=AUTOGEN_PER_GROUP_PER_YEAR)  # can be negative
            noise = rng.normal(loc=0, scale=12, size=AUTOGEN_PER_GROUP_PER_YEAR)
            finish = thr_sec - buf + noise
            finish = np.clip(finish, 2*3600, 6*3600)  # keep 2h..6h
            for fsec in finish:
                rows.append({
                    "race_year": y,
                    "gender": g,
                    "age": int(_sample_age_from_bin(a, rng)),
                    "finish_time": sec_to_hms(fsec)
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"(Auto-generated) wrote demo runner dataset → {out_path}  [rows={len(df)}]")

def load_runner_results():
    if not RESULTS_CSV.exists():
        if AUTOGEN_DEMO_IF_MISSING:
            thr = load_actual_thresholds()
            autogen_demo_results(thr, RESULTS_CSV)
        else:
            raise FileNotFoundError(
                f"Runner results file not found: {RESULTS_CSV}\n"
                "Expected columns: race_year, gender, age, finish_time"
            )

    df = pd.read_csv(RESULTS_CSV)
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    df["finish_sec"] = df["finish_time"].apply(parse_time_to_sec)
    df["gender_norm"] = df["gender"].apply(normalize_gender)

    if AGE_FROM_BRACKET and "age_bracket" in df.columns:
        df["BQ_Age"] = df["age_bracket"].apply(age_bracket_to_bq_bin)
    else:
        df["BQ_Age"] = df["age"].apply(age_to_bq_bin)

    keep = ["race_year","gender_norm","BQ_Age","finish_sec"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in results: {missing}")
    df = df[keep].dropna()
    df = df[(df["finish_sec"]>0) & (df["race_year"].between(2012, 2026))]
    return df

# -------------------------- BUILD DATASET --------------------------
def build_dataset():
    runners = load_runner_results()
    thr = load_actual_thresholds()

    data = runners.merge(
        thr,
        on=["race_year","gender_norm","BQ_Age"],
        how="inner"
    )
    data["accepted"] = (data["finish_sec"] <= data["ActualQualifyingTime_sec"]).astype(int)
    data["buffer_sec"] = (data["ActualQualifyingTime_sec"] - data["finish_sec"]).astype(float)
    data["year_c"] = data["race_year"] - data["race_year"].mean()

    feat_cols_num = ["buffer_sec", "year_c"]
    feat_cols_cat = ["gender_norm", "BQ_Age"]
    X = data[feat_cols_num + feat_cols_cat].copy()
    y = data["accepted"].values

    return data, X, y, feat_cols_num, feat_cols_cat

# -------------------------- MODELS --------------------------
def _calibrated(est, method="sigmoid", cv=5):
    """
    Wrap an estimator (or Pipeline) with CalibratedClassifierCV.
    Handles both modern scikit-learn (estimator=...) and older (base_estimator=...).
    """
    try:
        return CalibratedClassifierCV(estimator=est, method=method, cv=cv)   # newer sklearn
    except TypeError:
        return CalibratedClassifierCV(base_estimator=est, method=method, cv=cv)  # older

def make_pipelines(feat_cols_num, feat_cols_cat):
    # Preprocessing
    num = Pipeline([("scaler", StandardScaler())])
    cat = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer(
        transformers=[
            ("num", num, feat_cols_num),
            ("cat", cat, feat_cols_cat),
        ]
    )

    models = {}

    # 1) Perceptron
    models["Perceptron"] = Pipeline([
        ("pre", pre),
        ("clf", Perceptron(max_iter=1000, tol=1e-3, random_state=42))
    ])

    # 2) Adaline-like via SGDRegressor (squared loss), squashed to [0,1]
    models["Adaline_like"] = Pipeline([
        ("pre", pre),
        ("reg", SGDRegressor(loss="squared_error", max_iter=2000, tol=1e-3, random_state=42))
    ])

    # 3) Logistic Regression
    models["LogisticRegression"] = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ])

    # 4) KNN
    models["KNN(k=15)"] = Pipeline([
        ("pre", pre),
        ("clf", KNeighborsClassifier(n_neighbors=15))
    ])

    # 5) Linear SVM (calibrated for probabilities)
    base_svm = Pipeline([
        ("pre", pre),
        ("svm", LinearSVC(random_state=42))
    ])
    models["LinearSVM(calibrated)"] = _calibrated(base_svm, method="sigmoid", cv=5)

    # 6) Decision Tree
    models["DecisionTree"] = Pipeline([
        ("pre", pre),
        ("clf", DecisionTreeClassifier(max_depth=6, random_state=42))
    ])

    # 7) Random Forest
    rf = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    models["RandomForest"] = rf

    # 8) Random Forest (calibrated)
    models["RandomForest(calibrated)"] = _calibrated(rf, method="isotonic", cv=5)

    return models

def _proba_from_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        dec = model.decision_function(X); return 1 / (1 + np.exp(-dec))
    if isinstance(model, Pipeline) and "reg" in dict(model.named_steps):
        raw = model.predict(X); return 1 / (1 + np.exp(-raw))
    pred = model.predict(X); return pred.astype(float)

def eval_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    proba = _proba_from_model(model, X_test)
    y_pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    try: auc = roc_auc_score(y_test, proba)
    except: auc = np.nan
    brier = brier_score_loss(y_test, proba)

    try:
        fpr, tpr, _ = roc_curve(y_test, proba)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        plt.title(f"ROC — {name}"); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"roc_{name}.png", dpi=160); plt.close()
    except Exception as e:
        print(f"(ROC) {name}: {e}")

    try:
        CalibrationDisplay.from_predictions(y_test, proba, n_bins=10)
        plt.title(f"Calibration — {name}"); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"calibration_{name}.png", dpi=160); plt.close()
    except Exception as e:
        print(f"(Calibration) {name}: {e}")

    rep = classification_report(y_test, y_pred, digits=3, zero_division=0)
    return {"model": name, "accuracy": acc, "f1": f1, "roc_auc": auc, "brier": brier}, rep, model

# -------------------------- BUFFER → PROB CURVES --------------------------
def _make_design_for_buffers(gender, bq_age, buffer_secs):
    return pd.DataFrame({
        "buffer_sec": buffer_secs.astype(float),
        "year_c": np.zeros_like(buffer_secs, dtype=float),
        "gender_norm": [gender]*len(buffer_secs),
        "BQ_Age": [bq_age]*len(buffer_secs),
    })

def buffer_probability_curve(name, trained_model, groups=BUFFER_GROUPS, buffer_range=BUFFER_RANGE_SEC):
    rows = []
    for (g, a) in groups:
        Xgrid = _make_design_for_buffers(g, a, buffer_range)
        proba = _proba_from_model(trained_model, Xgrid)
        for buf, p in zip(buffer_range, proba):
            rows.append({"model": name, "gender": g, "BQ_Age": a, "buffer_sec": buf, "buffer_mmss": mmss(buf), "prob_accept": float(p)})
        try:
            plt.figure(figsize=(6,4))
            plt.plot(buffer_range/60.0, proba, lw=2)
            plt.axhline(0.5, linestyle="--", alpha=0.5)
            plt.xlabel("Buffer (minutes)"); plt.ylabel("Predicted P(accepted)")
            plt.ylim(-0.05, 1.05); plt.title(f"P(accepted) vs Buffer — {name} [{g} {a}]")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"buffer_curve_{name}_{g}_{a}.png", dpi=160); plt.close()
        except Exception as e:
            print(f"(Buffer plot) {name} {g} {a}: {e}")
    return pd.DataFrame(rows)

# -------------------------- MAIN --------------------------
def main():
    data, X, y, num_cols, cat_cols = build_dataset()

    # Chronological split if 2026 present; else random 80/20
    if (data["race_year"] == 2026).any():
        train_mask = data["race_year"] <= 2025
        test_mask  = data["race_year"] == 2026
    else:
        rng = np.random.RandomState(42)
        train_mask = rng.rand(len(data)) < 0.8
        test_mask  = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    models = make_pipelines(num_cols, cat_cols)

    rows, text_blocks, trained = [], [], {}
    for name, model in models.items():
        try:
            row, clf_report, fitted = eval_model(name, model, X_train, X_test, y_train, y_test)
            rows.append(row)
            text_blocks.append(f"### {name}\n```\n{clf_report}\n```")
            trained[name] = fitted
            print(f"Done: {name}")
        except Exception as e:
            rows.append({"model": name, "error": str(e)})
            text_blocks.append(f"### {name}\nError: {e}")
            print(f"Error in {name}: {e}")

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["roc_auc","f1","accuracy"] if c in out.columns]
    out_sorted = out.sort_values(sort_cols, ascending=[False]*len(sort_cols))
    out_sorted.to_csv(OUT_EVAL_TABLE, index=False)
    print(f"\nSaved model comparison → {OUT_EVAL_TABLE}")

    # Buffer→prob curves
    all_buf = []
    for name, mdl in trained.items():
        try:
            df_buf = buffer_probability_curve(name, mdl, BUFFER_GROUPS, BUFFER_RANGE_SEC)
            all_buf.append(df_buf)
        except Exception as e:
            print(f"Buffer curve failed for {name}: {e}")
    if all_buf:
        buf_tbl = pd.concat(all_buf, ignore_index=True)
        buf_tbl.to_csv(OUT_BUFFER_TABLE, index=False)
        print(f"Saved buffer→probability table → {OUT_BUFFER_TABLE}")

    # Mini report
    md = []
    md.append("# Which models best predict acceptance?\n")
    md.append("We reframed the problem as **Accepted vs Not Accepted**, using per-year/per-age-bin/per-gender thresholds.\n")
    md.append("Train/test: trained on earlier years, tested on the latest year we have rows for.\n")
    md.append("\n## TL;DR\n")
    if "roc_auc" in out_sorted.columns and out_sorted["roc_auc"].notna().any():
        top = out_sorted.iloc[0]
        md.append(f"- **Best overall ROC-AUC:** **{top['model']}** (AUC {top['roc_auc']:.3f}, "
                  f"Acc {top['accuracy']:.3f}, F1 {top['f1']:.3f}, Brier {top['brier']:.3f})\n")
    md.append("\n## Full metrics table\n")
    md.append(out_sorted.to_markdown(index=False))
    md.append("\n\n## Plots\n")
    md.append("- ROC curves and calibration plots are in `./plots/` (one PNG per model).")
    md.append("- Buffer→probability curves are also in `./plots/` (named `buffer_curve_<model>_<gender>_<age>.png`).\n")
    md.append("## Buffer → Probability table\n")
    md.append(f"A combined CSV is saved at `{OUT_BUFFER_TABLE.name}` with columns: `model, gender, BQ_Age, buffer_sec, buffer_mmss, prob_accept`.\n")
    md.append("Share rows for the group your clubmates care about. At **0:00 buffer**, probabilities should be close to 50%.\n")
    md.append("\n## Per-model classification reports\n")
    md.extend(text_blocks)
    OUT_REPORT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved model report → {OUT_REPORT_MD}")

if __name__ == "__main__":
    main()