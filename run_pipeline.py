#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified reproducible pipeline for the major revision of
infrastructures-4531268 (goaf subsidence forecasting).

One documented workflow produces every result:
  parse(clean) -> features(leakage-safe) -> split -> train/eval
  -> baselines / ablation / multi-seed / zone-CV / importance

Stages (run one per invocation, results checkpointed to results/):
  python run_pipeline.py features
  python run_pipeline.py main
  python run_pipeline.py baselines
  python run_pipeline.py ablation A|B|C|D
  python run_pipeline.py seed <1..10>
  python run_pipeline.py cv <K21|K22|K23|JPK|RAMP>
  python run_pipeline.py importance
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "monitoring_long.csv"
OUT = ROOT / "results"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

# ---------------- documented configuration (reported in the manuscript) ----
CFG = dict(
    lags=5,
    train_session=6, calib_session=7, test_session=8,   # documented temporal split
    presets="good_quality",
    num_stack_levels=2,        # L1 base learners + 2 stacker levels (+ final weighted ensemble)
    num_bag_folds=5,
    num_bag_sets=1,
    time_limit=100,            # seconds per fit, reported
    eval_metric="r2",
    main_seed=2024,
    multi_seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    n_bootstrap=1000,
    pi_level=0.90,
)

SESSION_DATES = pd.to_datetime([
    "2021-12-10", "2022-02-18", "2022-03-08", "2022-04-06",
    "2022-04-27", "2022-05-31", "2022-06-24", "2022-07-16",
])
DAYS = np.diff(SESSION_DATES).astype("timedelta64[D]").astype(int)  # interval lengths before sessions 2..8

FEATURE_GROUPS = {
    "single_lags": [f"single_lag_{i}" for i in range(1, 6)],
    "cum_lags": [f"cum_lag_{i}" for i in range(1, 6)],
    "dynamic": ["diff_1", "roll_mean_3", "roll_std_3"],
    "context": ["timestep", "days_prev", "days_next"],
}
ABLATION = {
    "A": FEATURE_GROUPS["single_lags"],
    "B": FEATURE_GROUPS["single_lags"] + FEATURE_GROUPS["cum_lags"],
    "C": FEATURE_GROUPS["single_lags"] + FEATURE_GROUPS["cum_lags"] + FEATURE_GROUPS["dynamic"],
    "D": FEATURE_GROUPS["single_lags"] + FEATURE_GROUPS["cum_lags"] + FEATURE_GROUPS["dynamic"] + FEATURE_GROUPS["context"],
}
FULL_FEATURES = ABLATION["D"]

# 13 base-learner variants, 7 families — mirrors the submitted Table 2,
# now with every variant fully specified (fixes "Table 2 has one row" issue).
def hyperparameters(seed: int) -> dict:
    return {
        "GBM": [
            {"extra_trees": True, "seed": seed, "ag_args": {"name_suffix": "XT"}},
            {"seed": seed},
            {"learning_rate": 0.03, "num_leaves": 128, "seed": seed, "ag_args": {"name_suffix": "Large"}},
        ],
        "CAT": [
            {"random_seed": seed},
            {"depth": 8, "learning_rate": 0.05, "random_seed": seed, "ag_args": {"name_suffix": "Deep"}},
        ],
        "XGB": [
            {"random_state": seed},
            {"max_depth": 10, "eta": 0.01, "random_state": seed, "ag_args": {"name_suffix": "Dense"}},
        ],
        "RF": [
            {"criterion": "squared_error", "random_state": seed, "ag_args": {"name_suffix": "MSE"}},
            {"criterion": "absolute_error", "random_state": seed, "ag_args": {"name_suffix": "MAE"}},
        ],
        "XT": [{"criterion": "squared_error", "random_state": seed, "ag_args": {"name_suffix": "MSE"}}],
        "KNN": [
            {"weights": "uniform", "ag_args": {"name_suffix": "Unif"}},
            {"weights": "distance", "ag_args": {"name_suffix": "Dist"}},
        ],
        "LR": {},
    }


# ---------------------------------------------------------------- data -----
def load_clean() -> pd.DataFrame:
    """Load parsed appendix data and apply documented cleaning rules."""
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(df["date"])
    # Rule 1: drop replicate==3 rows — parser showed the 3rd rows of K22+320 and
    # JPK21+060 are cross-mileage mislabels (identical to GK0+350 / DK0+170).
    df = df[df["replicate"] <= 2]
    # Rule 2: EK0+780's replicate-2 row is an exact duplicate with missing
    # rate/cumulative -> drop it (documented in parse_report.md).
    df = df[~((df["mileage"] == "EK0+780") & (df["replicate"] == 2))]
    # Rule 3: for the 6 genuinely double-observed points, average the two
    # observations (per the manuscript's stated field procedure).
    df = (df.groupby(["point_id", "mileage", "zone", "session", "date"], as_index=False)
            [["increment_mm", "rate", "cumulative_mm"]].mean())
    # Rule 4: increments are the primary truth (appendix cumulative columns are
    # only ~45% consistent with cumsum of increments, deviations up to 1 mm).
    # Reconstruct cumulative displacement as the running sum of increments.
    df = df.sort_values(["point_id", "session"]).reset_index(drop=True)
    df["cum_recon"] = df.groupby("point_id")["increment_mm"].cumsum()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe feature construction: every feature of target session t
    uses ONLY observations from sessions <= t-1. (Reviewer 2, comment 3.)"""
    g = df.groupby("point_id", group_keys=False)
    out = df.copy()
    for i in range(1, CFG["lags"] + 1):
        out[f"single_lag_{i}"] = g["increment_mm"].shift(i)
        out[f"cum_lag_{i}"] = g["cum_recon"].shift(i)
    out["roll_mean_3"] = g["increment_mm"].transform(lambda s: s.shift(1).rolling(3).mean())
    out["roll_std_3"] = g["increment_mm"].transform(lambda s: s.shift(1).rolling(3).std())
    out["diff_1"] = out["single_lag_1"] - out["single_lag_2"]
    # unequal monitoring intervals (Reviewer 4, comment 4):
    # days_next  = length of the interval whose increment is being predicted
    #              (interval session t-1 -> t; known from the schedule -> no leakage)
    # days_prev  = length of the interval over which single_lag_1 was measured
    #              (interval session t-2 -> t-1)
    out["days_next"] = out["session"].map({s: DAYS[s - 2] for s in range(2, 9)})
    out["days_prev"] = out["session"].map({s: DAYS[s - 3] for s in range(3, 9)})
    out["timestep"] = out["session"]
    feat = out.dropna(subset=FULL_FEATURES + ["increment_mm"]).reset_index(drop=True)
    return feat


def get_split(feat: pd.DataFrame):
    tr = feat[feat["session"] == CFG["train_session"]]
    ca = feat[feat["session"] == CFG["calib_session"]]
    te = feat[feat["session"] == CFG["test_session"]]
    return tr, ca, te


# ------------------------------------------------------------- metrics -----
def metrics(y_true, y_pred) -> dict:
    return dict(
        r2=float(r2_score(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def eval_frame(te: pd.DataFrame, pred_inc: np.ndarray, model: str) -> pd.DataFrame:
    """Per-point evaluation frame: increment metrics (primary) and one-step
    cumulative reconstruction (secondary). With one-step-ahead reconstruction
    from the true previous cumulative, |cum error| == |increment error|; the
    difference is in R^2, which is inflated on the cumulative scale because
    the level variance far exceeds the increment variance (Reviewer 2, c2)."""
    d = te[["point_id", "mileage", "zone"]].copy()
    d["model"] = model
    d["y_inc"] = te["increment_mm"].values
    d["pred_inc"] = pred_inc
    d["y_cum"] = te["cum_recon"].values
    d["pred_cum"] = te["cum_lag_1"].values + pred_inc
    return d


def aggregate(ev: pd.DataFrame, model: str, tag: str) -> dict:
    row = {"model": model, "tag": tag}
    for scale in ("inc", "cum"):
        m = metrics(ev[f"y_{scale}"], ev[f"pred_{scale}"])
        row.update({f"{scale}_{k}": v for k, v in m.items()})
    return row


def save_json(obj, name):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------ AutoGluon ----
def fit_autogluon(train_df, calib_df, seed, tag, features=None, stack_levels=None,
                  time_limit=None):
    from autogluon.tabular import TabularPredictor

    features = features or FULL_FEATURES
    X_tr = train_df[features + ["increment_mm"]]
    X_ca = calib_df[features + ["increment_mm"]] if calib_df is not None else None
    predictor = TabularPredictor(
        label="increment_mm", eval_metric=CFG["eval_metric"],
        path=str(MODELS / f"ag_{tag}"), verbosity=1,
    ).fit(
        X_tr,
        tuning_data=X_ca,
        use_bag_holdout=X_ca is not None,
        presets=CFG["presets"],
        num_stack_levels=stack_levels if stack_levels is not None else CFG["num_stack_levels"],
        num_bag_folds=CFG["num_bag_folds"],
        num_bag_sets=CFG["num_bag_sets"],
        time_limit=time_limit or CFG["time_limit"],
        hyperparameters=hyperparameters(seed),
        ag_args_fit={"random_state": seed},
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    return predictor


# ---------------------------------------------------------------- stages ---
def stage_features():
    df = load_clean()
    feat = build_features(df)
    feat.to_csv(ROOT / "data" / "features.csv", index=False)
    tr, ca, te = get_split(feat)

    # documented split table (Reviewer 2 c7 / Reviewer 4 c1)
    split_tbl = pd.DataFrame([
        {"session": s, "date": str(SESSION_DATES[s - 1].date()),
         "n_samples": int((feat["session"] == s).sum()),
         "role": {6: "train", 7: "calibration (ensemble weights / bagging holdout)",
                  8: "independent test"}[s],
         "interval_days_before": int(DAYS[s - 2])}
        for s in (6, 7, 8)])
    split_tbl.to_csv(OUT / "table_split.csv", index=False)

    # monitoring interval table
    pd.DataFrame({
        "session": range(2, 9), "date": [str(d.date()) for d in SESSION_DATES[1:]],
        "days_since_previous": DAYS,
    }).to_csv(OUT / "table_intervals.csv", index=False)

    # worked example for one monitoring point (Reviewer 2 c3): show, for the
    # first point, every feature of target session 6 and its source sessions.
    p = feat[feat["session"] == 6].iloc[0]
    raw = df[df["point_id"] == p["point_id"]].sort_values("session")
    lines = [f"# Worked example: point {p['point_id']} ({p['mileage']}), target session 6 "
             f"({SESSION_DATES[5].date()})",
             "",
             "| session | date | increment_mm | cum_recon | used as |",
             "|---|---|---|---|---|"]
    use_map = {1: "single_lag_5 / cum_lag_5", 2: "single_lag_4 / cum_lag_4",
               3: "single_lag_3 / cum_lag_3", 4: "single_lag_2 / cum_lag_2 + roll window",
               5: "single_lag_1 / cum_lag_1 + roll window", 6: "TARGET (not a feature)"}
    for _, r in raw.iterrows():
        lines.append(f"| {int(r['session'])} | {r['date'].date()} | {r['increment_mm']:.2f} "
                     f"| {r['cum_recon']:.2f} | {use_map.get(int(r['session']), 'not used')} |")
    lines += ["",
              f"single_lag_1..5 = {[round(p[f'single_lag_{i}'],2) for i in range(1,6)]}",
              f"cum_lag_1..5    = {[round(p[f'cum_lag_{i}'],2) for i in range(1,6)]}",
              f"roll_mean_3 = mean(increments of sessions 3,4,5) = {p['roll_mean_3']:.3f}",
              f"roll_std_3  = std (increments of sessions 3,4,5) = {p['roll_std_3']:.3f}",
              f"diff_1 = single_lag_1 - single_lag_2 = {p['diff_1']:.3f}",
              f"days_prev = {int(p['days_prev'])} (interval session 4->5), "
              f"days_next = {int(p['days_next'])} (interval session 5->6, known from schedule)",
              "",
              "No feature of target session t uses observations from session t or later."]
    (OUT / "worked_example.md").write_text("\n".join(lines), encoding="utf-8")

    # environment record (reproducibility, Reviewer 3 c4 / Reviewer 5 c3)
    env = {"python": sys.version, "platform": platform.platform(),
           "config": {k: v for k, v in CFG.items()}}
    try:
        import autogluon.tabular, lightgbm, xgboost, catboost, sklearn
        env["versions"] = dict(autogluon=autogluon.tabular.__version__,
                               lightgbm=lightgbm.__version__, xgboost=xgboost.__version__,
                               catboost=catboost.__version__, sklearn=sklearn.__version__,
                               pandas=pd.__version__, numpy=np.__version__)
    except Exception as e:  # noqa
        env["versions_error"] = str(e)
    save_json(env, "environment.json")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True).stdout
    (OUT / "pip_freeze.txt").write_text(freeze)

    print(f"points={feat['point_id'].nunique()}  feature-rows={len(feat)}")
    print(split_tbl.to_string(index=False))
    print("features saved -> data/features.csv ; split/interval/worked-example/env -> results/")


def stage_baselines():
    feat = pd.read_csv(ROOT / "data" / "features.csv")
    feat["date"] = pd.to_datetime(feat["date"])
    tr, ca, te = get_split(feat)

    evals, rows = [], []
    # 1) persistence: next increment = previous increment (Reviewer 2 c2, R5 c7)
    evals.append(eval_frame(te, te["single_lag_1"].values, "Persistence"))
    # 2) recent-rate extrapolation: mean increment of last 3 sessions,
    #    scaled by interval ratio (unequal intervals, Reviewer 4 c4)
    rate_pred = te["roll_mean_3"].values * (te["days_next"].values / te["days_prev"].values)
    evals.append(eval_frame(te, rate_pred, "RateExtrapolation"))
    # 3) AR(5) ridge on single-cycle lags only, fitted on train session only
    ar = Ridge(alpha=1.0, random_state=CFG["main_seed"])
    ar.fit(tr[FEATURE_GROUPS["single_lags"]], tr["increment_mm"])
    evals.append(eval_frame(te, ar.predict(te[FEATURE_GROUPS["single_lags"]]), "AR5_Ridge"))

    for ev in evals:
        rows.append(aggregate(ev, ev["model"].iloc[0], "session8"))
        ev.to_csv(OUT / f"pred_{ev['model'].iloc[0]}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "metrics_baselines.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def _run_ag(tag, seed, features=None, tr_override=None, ca_override=None, te_override=None):
    feat = pd.read_csv(ROOT / "data" / "features.csv")
    feat["date"] = pd.to_datetime(feat["date"])
    tr, ca, te = get_split(feat)
    tr = tr_override if tr_override is not None else tr
    ca = ca_override if ca_override is not None else ca
    te = te_override if te_override is not None else te
    t0 = time.time()
    predictor = fit_autogluon(tr, ca, seed, tag, features=features)
    fit_s = time.time() - t0
    cols = features or FULL_FEATURES
    pred = predictor.predict(te[cols]).values
    ev = eval_frame(te, pred, f"AutoGluon_{tag}")
    ev.to_csv(OUT / f"pred_{tag}.csv", index=False)
    row = aggregate(ev, f"AutoGluon_{tag}", "session8")
    row["fit_seconds"] = round(fit_s, 1)
    row["seed"] = seed
    lb = predictor.leaderboard(te[cols + ["increment_mm"]], silent=True)
    lb.to_csv(OUT / f"leaderboard_{tag}.csv", index=False)
    # best single base learner on the test session (ensemble-vs-best-single test)
    best = lb[~lb["model"].str.contains("WeightedEnsemble", na=False)].iloc[0]
    pred_best = predictor.predict(te[cols], model=best["model"]).values
    ev_best = eval_frame(te, pred_best, f"BestSingle_{tag}")
    ev_best.to_csv(OUT / f"pred_bestsingle_{tag}.csv", index=False)
    row_best = aggregate(ev_best, best["model"], "session8")
    row.update({f"bestsingle_{k}": v for k, v in row_best.items() if k not in ("model", "tag")})
    row["bestsingle_name"] = best["model"]
    save_json(row, f"metrics_{tag}.json")
    print(json.dumps(row, ensure_ascii=False, indent=2))


def stage_intervals():
    """Split-conformal prediction intervals for session-8 increments.
    Calibrate |residual| quantile on session 7 (calibration, never used to fit
    base learners), apply to session 8, then verify empirical coverage.
    Gives a distribution-free PI (Reviewer 1 c2, Reviewer 2 c3)."""
    from autogluon.tabular import TabularPredictor
    feat = pd.read_csv(ROOT / "data" / "features.csv")
    _, ca, te = get_split(feat)
    predictor = TabularPredictor.load(str(MODELS / "ag_main"))
    pred_ca = predictor.predict(ca[FULL_FEATURES]).values
    res = np.abs(ca["increment_mm"].values - pred_ca)
    n = len(res)
    level = CFG["pi_level"]
    q = np.quantile(res, np.ceil((n + 1) * level) / n, method="higher")
    pred_te = predictor.predict(te[FULL_FEATURES]).values
    lo, hi = pred_te - q, pred_te + q
    cover = float(np.mean((te["increment_mm"].values >= lo) & (te["increment_mm"].values <= hi)))
    out = te[["point_id", "mileage", "zone"]].copy()
    out["y_inc"] = te["increment_mm"].values
    out["pred_inc"] = pred_te
    out["pi_lo"], out["pi_hi"] = lo, hi
    out.to_csv(OUT / "pred_intervals_session8.csv", index=False)
    summary = {"method": "split-conformal on session-7 calibration residuals",
               "nominal_level": level, "half_width_mm": float(q),
               "empirical_coverage_session8": cover,
               "mean_interval_width_mm": float(2 * q)}
    save_json(summary, "intervals_summary.json")
    print(json.dumps(summary, indent=2))


def stage_importance():
    from autogluon.tabular import TabularPredictor
    feat = pd.read_csv(ROOT / "data" / "features.csv")
    _, _, te = get_split(feat)
    predictor = TabularPredictor.load(str(MODELS / "ag_main"))
    # permutation importance on the HELD-OUT session 8 with repeated shuffles
    # (replaces the in-ensemble delta-R2 + p-value table; Reviewer 3 c9, R5 c6)
    fi = predictor.feature_importance(
        te[FULL_FEATURES + ["increment_mm"]], num_shuffle_sets=20, include_confidence_band=True)
    fi.to_csv(OUT / "table_permutation_importance.csv")
    print(fi.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("value", nargs="?")
    args = ap.parse_args()

    if args.stage == "features":
        stage_features()
    elif args.stage == "baselines":
        stage_baselines()
    elif args.stage == "main":
        _run_ag("main", CFG["main_seed"])
    elif args.stage == "seed":
        _run_ag(f"seed{args.value}", int(args.value))
    elif args.stage == "ablation":
        _run_ag(f"ablation{args.value}", CFG["main_seed"], features=ABLATION[args.value])
    elif args.stage == "cv":
        feat = pd.read_csv(ROOT / "data" / "features.csv")
        zone = args.value
        tr_all = feat[feat["session"].isin([CFG["train_session"], CFG["calib_session"]])]
        te_z = feat[(feat["session"] == CFG["test_session"]) & (feat["zone"] == zone)]
        tr_z = tr_all[tr_all["zone"] != zone]
        _run_ag(f"cv_{zone}", CFG["main_seed"], tr_override=tr_z, ca_override=None, te_override=te_z)
    elif args.stage == "importance":
        stage_importance()
    elif args.stage == "intervals":
        stage_intervals()
    else:
        raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    main()
