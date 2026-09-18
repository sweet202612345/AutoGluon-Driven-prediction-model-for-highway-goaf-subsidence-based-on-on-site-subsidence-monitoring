#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended pipeline for the major revision of infrastructures-4531268:
adds STATIC GEOLOGICAL / MINING features (Reviewer 5, comment 4) and
rebuilds the dataset from the AUTHORITATIVE 81-point monitoring data
extracted from the survey report PDF (pages 121-124), replacing the
appendix-parsed 88-point data used in the first revision round.

Static features added (all constant in time, leakage-free by nature):
  numeric  : has_goaf, seam_thickness_m, mining_depth_m,
             depth_thickness_ratio, dip_angle_deg, recovery_rate,
             years_since_abandonment, room_width_m, pillar_width_m,
             grouting
  categoric: mining_method, overburden_lithology, coal_mine, stability_class

Feature provenance: survey report Table 5.1.4 / 4.6.1 (goaf-segment
parameters), Section 4.1.2-4.1.4 (mine history), Section 5.1.5
(stability classes), borehole table 5.1.3 (JPK mine attribution).
Grouting: NO cement grouting or any ground reinforcement was applied
inside the monitored area during the monitoring window (2021-11 ~ 2022-07);
the grouting schemes in the report are post-monitoring RECOMMENDATIONS.
The feature is kept (documented constant 0) so the manuscript can state
this explicitly.

Stages (one per invocation; results -> results_geo/, models -> models/):
  python run_pipeline_geo.py rebuild      # real-data long table + geo table
  python run_pipeline_geo.py features     # temporal + static features
  python run_pipeline_geo.py baselines
  python run_pipeline_geo.py main         # temporal-only 16 feats, seed 2024
  python run_pipeline_geo.py main_geo     # temporal + geo 30 feats, seed 2024
  python run_pipeline_geo.py ablation A|B|C|D|E
  python run_pipeline_geo.py seed <1..10> # multi-seed on the geo model
  python run_pipeline_geo.py cv <K21|K22|K23|JPK|RAMP>
  python run_pipeline_geo.py intervals
  python run_pipeline_geo.py importance
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_pipeline as rp  # reuse documented config, AutoGluon wrapper, metrics

ROOT = Path(__file__).resolve().parent
SRC_XLSX = ROOT / "data" / "settlement_summary_tables_pp121-124.xlsx"  # raw transcription (reference only)
OUT = ROOT / "results_geo"
OUT.mkdir(exist_ok=True)
DATA_LONG = ROOT / "data" / "monitoring_long_real.csv"
DATA_GEO = ROOT / "data" / "geo_features.csv"
DATA_FEAT = ROOT / "data" / "features_geo.csv"

CFG = rp.CFG
SESSION_DATES = rp.SESSION_DATES
DAYS = rp.DAYS

# ----------------------------------------------------------------- geo -----
GEO_NUM = ["has_goaf", "seam_thickness_m", "mining_depth_m",
           "depth_thickness_ratio", "dip_angle_deg", "recovery_rate",
           "years_since_abandonment", "room_width_m", "pillar_width_m",
           "grouting"]
GEO_CAT = ["mining_method", "overburden_lithology", "coal_mine", "stability_class"]
GEO_FEATURES = GEO_NUM + GEO_CAT

ABLATION = dict(rp.ABLATION)
ABLATION["E"] = rp.ABLATION["D"] + GEO_FEATURES          # the new model
FULL_FEATURES_GEO = ABLATION["E"]


def _km(mileage: str):
    import re
    m = re.match(r"(?:JP)?K(\d+)\+(\d+)", str(mileage))
    return int(m.group(1)) * 1000 + int(m.group(2)) if m else None


def geo_record(main: str, ramp: str) -> dict:
    """Static geological/mining features for one monitoring point, assigned
    from its mileage per the survey report (see module docstring)."""
    k = _km(main) if str(main).startswith("K") else None
    jpk = _km(main) if str(main).startswith("JPK") else None
    z = dict(has_goaf=0.0, seam_thickness_m=0.0, mining_depth_m=0.0,
             depth_thickness_ratio=0.0, dip_angle_deg=0.0, recovery_rate=0.0,
             years_since_abandonment=0.0, room_width_m=0.0, pillar_width_m=0.0,
             grouting=0.0, mining_method="none", overburden_lithology="none",
             coal_mine="none", stability_class="none", geo_note="")

    def fill(method, th, dep, ratio, rec, yrs, overb, stab, mine):
        z.update(has_goaf=1.0, seam_thickness_m=th, mining_depth_m=dep,
                 depth_thickness_ratio=ratio, dip_angle_deg=11.0,
                 recovery_rate=rec, years_since_abandonment=yrs,
                 mining_method=method, overburden_lithology=overb,
                 stability_class=stab, coal_mine=mine)

    if k is not None and k < 22240:
        # outside every goaf segment (background / comparison reaches)
        z["coal_mine"] = "none" if k < 21240 else "Wanghe"
        z["geo_note"] = ("outside goaf segments; no goaf beneath the route"
                         if k >= 21240 else
                         "outside Wanghe mine boundary (mine starts at K21+240)")
    elif k is not None and 22240 <= k <= 23030:
        # Segment 1: shortwall goaf of seam Yi-1, mines Wanghe/Mihe/Shuanglou
        fill("shortwall", 1.24, 215.0, 162.5, 0.75, 6.9,
             "medium_hard", "basic_stable",
             "Wanghe" if k <= 22450 else "Mihe")
    elif k is not None and 23030 < k <= 23890:
        # Segment 2: room-and-pillar goaf of seam Yi-1, Mihe mine
        fill("room_and_pillar", 1.24, 125.0, 94.5, 0.60, 11.9,
             "medium_hard",
             "stable" if k <= 23491 else ("basic_stable" if k <= 23580 else "unstable"),
             "Mihe")
        z["room_width_m"], z["pillar_width_m"] = 6.5, 12.5   # reported 5-8 / 10-15 m
        if 23330 <= k <= 23580:
            # Segment 3 overlays segment 2: double-seam goaf (Yi-1 + Er-1)
            z.update(seam_thickness_m=3.49, depth_thickness_ratio=37.0,
                     recovery_rate=0.50, years_since_abandonment=51.9,
                     overburden_lithology="weak", stability_class="unstable")
            z["geo_note"] = ("double-seam goaf (Yi-1 room&pillar + Er-1 room&pillar); "
                             "Er-1 layer governs: weak overburden, class unstable")
    elif jpk is not None:
        # JPK mainline inside the interchange area, evaluated with Segment 1
        mine = "Shuanglou" if jpk <= 20420 else "Mihe"
        fill("shortwall", 1.24, 215.0, 162.5, 0.75, 6.9,
             "medium_hard", "basic_stable", mine)
        if 19730 <= jpk <= 19990:
            z["geo_note"] = ("mine attribution INFERRED (west of ZK4/JPK20+389); "
                             "needs the mine-boundary plan for confirmation")
    elif str(ramp).strip():
        # pure interchange-ramp points -> Segment 1 explicitly includes ramps
        fill("shortwall", 1.24, 215.0, 162.5, 0.75, 6.9,
             "medium_hard", "basic_stable", "Interchange(Wanghe/Mihe/Shuanglou)")
    if str(ramp).strip() and (k is not None or jpk is not None):
        z["geo_note"] = (z["geo_note"] + "; " if z["geo_note"] else "") + \
                        "dual-mileage point (ramp mileage belongs to Segment 1)"
    return z


# --------------------------------------------------------------- rebuild ---
def stage_rebuild():
    """Rebuild the long monitoring table from the authoritative 81-point
    dataset (survey-report PDF pp.121-124 extraction) and write the static
    geological feature table."""
    raw = pd.read_csv(ROOT / "data" / "monitoring_master_real.csv")
    raw["date"] = pd.to_datetime(raw["date_str"], format="%Y.%m.%d")
    date2session = {d: i + 1 for i, d in enumerate(sorted(raw["date"].unique()))}
    assert [str(d.date()) for d in sorted(date2session)] == \
           [str(d.date()) for d in SESSION_DATES], "session dates mismatch"

    def zone_of(main, ramp):
        m = str(main)
        for z in ("K21", "K22", "K23", "JPK"):
            if m.startswith(z):
                return z
        return "RAMP"

    df = pd.DataFrame({
        "point_id": raw["point_id"],
        "mileage": np.where(raw["mainline_mileage"].astype(str).str.strip() != "",
                            raw["mainline_mileage"], raw["interchange_mileage"]),
        "zone": [zone_of(m, r) for m, r in zip(raw["mainline_mileage"], raw["interchange_mileage"])],
        "session": raw["date"].map(date2session),
        "date": raw["date"],
        "increment_mm": pd.to_numeric(raw["increment_mm"]),
        "rate": pd.to_numeric(raw["rate_mm_d"]),
        "cumulative_mm": pd.to_numeric(raw["cumulative_mm"]),
        "replicate": 1,
    }).sort_values(["point_id", "session"]).reset_index(drop=True)

    # data-audit: consistency between reported cumulative and cumsum(increments)
    df["cum_check"] = df.groupby("point_id")["increment_mm"].cumsum()
    diff = (df["cumulative_mm"] - df["cum_check"]).abs()
    audit = {
        "n_points": int(df["point_id"].nunique()),
        "n_records": int(len(df)),
        "sessions": int(df["session"].nunique()),
        "cum_vs_cumsum_exact_le_0.05mm": float((diff <= 0.050001).mean()),
        "cum_vs_cumsum_le_0.5mm": float((diff <= 0.5).mean()),
        "max_abs_deviation_mm": float(diff.max()),
        "note": "increments kept as primary truth; cumulative reconstructed",
    }
    rp.save_json(audit, "data_audit_real.json")  # into results/ for the record
    (OUT / "data_audit_real.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    df = df.drop(columns=["cum_check"])
    df.to_csv(DATA_LONG, index=False)

    geo = pd.DataFrame([dict(point_id=r["point_id"],
                             **geo_record(
                                 "" if pd.isna(r["mainline_mileage"]) else str(r["mainline_mileage"]),
                                 "" if pd.isna(r["interchange_mileage"]) else str(r["interchange_mileage"])))
                        for _, r in
                        raw.drop_duplicates("point_id")[["point_id", "mainline_mileage", "interchange_mileage"]]
                        .iterrows()])
    geo.to_csv(DATA_GEO, index=False)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(df.groupby("zone")["point_id"].nunique())
    print(geo[["point_id"] + GEO_FEATURES].head(12).to_string(index=False))
    print("grouting unique:", geo["grouting"].unique())


# -------------------------------------------------------------- features ---
def stage_features():
    # rp.load_clean is bound to the old DATA path -> inline equivalent here
    df = pd.read_csv(DATA_LONG)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["replicate"] <= 2]
    df = df.sort_values(["point_id", "session"]).reset_index(drop=True)
    df["cum_recon"] = df.groupby("point_id")["increment_mm"].cumsum()

    feat = rp.build_features(df)
    geo = pd.read_csv(DATA_GEO)
    feat = feat.merge(geo, on="point_id", how="left", validate="many_to_one")
    for c in GEO_NUM:
        feat[c] = feat[c].fillna(0.0)
    for c in GEO_CAT:
        feat[c] = feat[c].fillna("none").astype(str)
    feat.to_csv(DATA_FEAT, index=False)

    tr = feat[feat["session"] == CFG["train_session"]]
    ca = feat[feat["session"] == CFG["calib_session"]]
    te = feat[feat["session"] == CFG["test_session"]]
    split_tbl = pd.DataFrame([
        {"session": s, "date": str(SESSION_DATES[s - 1].date()),
         "n_samples": int((feat["session"] == s).sum()),
         "role": {6: "train", 7: "calibration (ensemble weights / bagging holdout)",
                  8: "independent test"}[s],
         "interval_days_before": int(DAYS[s - 2])} for s in (6, 7, 8)])
    split_tbl.to_csv(OUT / "table_split.csv", index=False)
    print(f"points={feat['point_id'].nunique()} rows={len(feat)}")
    print(split_tbl.to_string(index=False))
    print("n features: temporal-only D =", len(rp.ABLATION['D']),
          "| + geo E =", len(FULL_FEATURES_GEO))


# ------------------------------------------------------------ run helper ---
def _load():
    feat = pd.read_csv(DATA_FEAT)
    feat["date"] = pd.to_datetime(feat["date"])
    return feat


def _run(tag, seed, features, tr_override=None, ca_override=None, te_override=None):
    feat = _load()
    tr, ca, te = (feat[feat["session"] == CFG["train_session"]],
                  feat[feat["session"] == CFG["calib_session"]],
                  feat[feat["session"] == CFG["test_session"]])
    tr = tr_override if tr_override is not None else tr
    ca = ca_override if ca_override is not None else ca
    te = te_override if te_override is not None else te
    t0 = time.time()
    predictor = rp.fit_autogluon(tr, ca, seed, tag, features=features)
    fit_s = time.time() - t0
    pred = predictor.predict(te[features]).values
    ev = rp.eval_frame(te, pred, f"AutoGluon_{tag}")
    ev.to_csv(OUT / f"pred_{tag}.csv", index=False)
    row = rp.aggregate(ev, f"AutoGluon_{tag}", "session8")
    row["fit_seconds"] = round(fit_s, 1)
    row["seed"] = seed
    row["n_features"] = len(features)
    lb = predictor.leaderboard(te[features + ["increment_mm"]], silent=True)
    lb.to_csv(OUT / f"leaderboard_{tag}.csv", index=False)
    best = lb[~lb["model"].str.contains("WeightedEnsemble", na=False)].iloc[0]
    ev_best = rp.eval_frame(te, predictor.predict(te[features], model=best["model"]).values,
                            f"BestSingle_{tag}")
    ev_best.to_csv(OUT / f"pred_bestsingle_{tag}.csv", index=False)
    row_best = rp.aggregate(ev_best, best["model"], "session8")
    row.update({f"bestsingle_{k}": v for k, v in row_best.items() if k not in ("model", "tag")})
    row["bestsingle_name"] = best["model"]
    with open(OUT / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    print(json.dumps(row, ensure_ascii=False, indent=2))


def stage_baselines():
    feat = _load()
    te = feat[feat["session"] == CFG["test_session"]]
    tr = feat[feat["session"] == CFG["train_session"]]
    evals = [rp.eval_frame(te, te["single_lag_1"].values, "Persistence")]
    rate_pred = te["roll_mean_3"].values * (te["days_next"].values / te["days_prev"].values)
    evals.append(rp.eval_frame(te, rate_pred, "RateExtrapolation"))
    from sklearn.linear_model import Ridge
    ar = Ridge(alpha=1.0, random_state=CFG["main_seed"])
    ar.fit(tr[rp.FEATURE_GROUPS["single_lags"]], tr["increment_mm"])
    evals.append(rp.eval_frame(te, ar.predict(te[rp.FEATURE_GROUPS["single_lags"]]), "AR5_Ridge"))
    rows = []
    for ev in evals:
        rows.append(rp.aggregate(ev, ev["model"].iloc[0], "session8"))
        ev.to_csv(OUT / f"pred_{ev['model'].iloc[0]}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "metrics_baselines.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def stage_intervals():
    from autogluon.tabular import TabularPredictor
    feat = _load()
    ca = feat[feat["session"] == CFG["calib_session"]]
    te = feat[feat["session"] == CFG["test_session"]]
    predictor = TabularPredictor.load(str(rp.MODELS / "ag_main_geo"))
    pred_ca = predictor.predict(ca[FULL_FEATURES_GEO]).values
    res = np.abs(ca["increment_mm"].values - pred_ca)
    n = len(res)
    q = np.quantile(res, np.ceil((n + 1) * CFG["pi_level"]) / n, method="higher")
    pred_te = predictor.predict(te[FULL_FEATURES_GEO]).values
    lo, hi = pred_te - q, pred_te + q
    cover = float(np.mean((te["increment_mm"].values >= lo) & (te["increment_mm"].values <= hi)))
    out = te[["point_id", "mileage", "zone"]].copy()
    out["y_inc"] = te["increment_mm"].values
    out["pred_inc"] = pred_te
    out["pi_lo"], out["pi_hi"] = lo, hi
    out.to_csv(OUT / "pred_intervals_session8.csv", index=False)
    summary = {"method": "split-conformal on session-7 calibration residuals (geo model)",
               "nominal_level": CFG["pi_level"], "half_width_mm": float(q),
               "empirical_coverage_session8": cover, "mean_interval_width_mm": float(2 * q)}
    (OUT / "intervals_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def stage_importance():
    from autogluon.tabular import TabularPredictor
    feat = _load()
    te = feat[feat["session"] == CFG["test_session"]]
    predictor = TabularPredictor.load(str(rp.MODELS / "ag_main_geo"))
    fi = predictor.feature_importance(
        te[FULL_FEATURES_GEO + ["increment_mm"]], num_shuffle_sets=20,
        include_confidence_band=True)
    fi.to_csv(OUT / "table_permutation_importance.csv")
    print(fi.to_string())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("value", nargs="?")
    args = ap.parse_args()

    if args.stage == "rebuild":
        stage_rebuild()
    elif args.stage == "features":
        stage_features()
    elif args.stage == "baselines":
        stage_baselines()
    elif args.stage == "main":
        _run("main", CFG["main_seed"], rp.ABLATION["D"])          # temporal-only, real data
    elif args.stage == "main_geo":
        _run("main_geo", CFG["main_seed"], FULL_FEATURES_GEO)     # + geological features
    elif args.stage == "ablation":
        _run(f"ablation{args.value}", CFG["main_seed"], ABLATION[args.value])
    elif args.stage == "seed":
        _run(f"seed{args.value}", int(args.value), FULL_FEATURES_GEO)
    elif args.stage == "cv":
        feat = _load()
        zone = args.value
        tr_all = feat[feat["session"].isin([CFG["train_session"], CFG["calib_session"]])]
        te_z = feat[(feat["session"] == CFG["test_session"]) & (feat["zone"] == zone)]
        tr_z = tr_all[tr_all["zone"] != zone]
        _run(f"cv_{zone}", CFG["main_seed"], FULL_FEATURES_GEO,
             tr_override=tr_z, ca_override=None, te_override=te_z)
    elif args.stage == "intervals":
        stage_intervals()
    elif args.stage == "importance":
        stage_importance()
    else:
        raise SystemExit(f"unknown stage {args.stage}")


if __name__ == "__main__":
    main()
