# Reproducibility Package — infrastructures-4531268 (Major Revision)

**Paper**: Short-Term Subsidence Forecasting for an Expressway over Abandoned Mine Goafs via Multi-Session Leveling Monitoring and Automated Ensemble Learning

**This package** lets reviewers reproduce **every number, table and figure** of the revised experiment section with one documented command sequence.

---

## 1. What is inside

```
repo root/
├── README.md                     ← this file
├── requirements.txt              ← pinned dependencies (identical to the run environment)
├── run_pipeline.py               ← base module: documented config, AutoGluon wrapper, metrics
├── run_pipeline_geo.py           ← MAIN ENTRY: data rebuild, features, all experiments
├── make_report_geo.py            ← aggregates results into summary tables + 6 figures
├── data/
│   ├── settlement_summary_tables_pp121-124.xlsx  ← authoritative raw data: 81 points × 8 sessions
│   │                                               (transcribed from the survey report, pp. 121–124)
│   ├── C1-C81_geology_mining_mapping.xlsx        ← per-point geological/mining documentation
│   ├── monitoring_master_real.csv                ← raw long table (648 records) converted from the xlsx
│   ├── monitoring_long_real.csv                  ← cleaned long table used by the pipeline (stage: rebuild)
│   ├── geo_features.csv                          ← 14 static geological/mining features per point
│   └── features_geo.csv                          ← full feature table (16 temporal + 14 static, 243 rows)
├── results_geo/                  ← every metric / prediction / leaderboard produced by the runs
├── figures_geo/                  ← fig1–fig6 generated from the results (all-English labels)
└── docs/
    ├── REPORT_GEO.md             ← full analysis report (design, results, mapping to reviewer comments)
    ├── environment.json          ← software versions + complete run configuration
    └── pip_freeze_full.txt       ← full environment freeze of the actual runs
```

## 2. Environment setup

Python 3.10–3.12 (the reported results were produced with CPython 3.12 on Windows 11).

```bash
cd <repo root>
python -m venv .venv
# Windows:      .venv/Scripts/pip install -r requirements.txt
# Linux/macOS:  .venv/bin/pip install -r requirements.txt
```

## 3. One-command-sequence reproduction

Run the stages **in this order** (each stage checkpoints its outputs into `results_geo/`;
trained AutoGluon models are saved under `models/`, which is created automatically):

```bash
# --- data & features -----------------------------------------------------
.venv/Scripts/python run_pipeline_geo.py rebuild      # 81-point long table + audit + geo features
.venv/Scripts/python run_pipeline_geo.py features     # leakage-safe temporal features + static merge

# --- simple baselines (Reviewer 2 #2, Reviewer 5 #7) ---------------------
.venv/Scripts/python run_pipeline_geo.py baselines    # persistence / rate extrapolation / AR(5)

# --- main models ----------------------------------------------------------
.venv/Scripts/python run_pipeline_geo.py main         # temporal-only 16 features, seed 2024
.venv/Scripts/python run_pipeline_geo.py main_geo     # PROPOSED MODEL: temporal + geological, 30 features

# --- ablation ladder (Reviewer 2 #4, Reviewer 5 #4) -----------------------
.venv/Scripts/python run_pipeline_geo.py ablation A   # 5 single-cycle lags only
.venv/Scripts/python run_pipeline_geo.py ablation B   # + 5 cumulative lags
.venv/Scripts/python run_pipeline_geo.py ablation C   # + 3 dynamic indicators
.venv/Scripts/python run_pipeline_geo.py ablation D   # + interval days / timestep (= temporal-only full)
.venv/Scripts/python run_pipeline_geo.py ablation E   # + 14 geological/mining features (= proposed model)

# --- robustness: 10 extra seeds on the proposed model (Reviewer 1 #1) -----
.venv/Scripts/python run_pipeline_geo.py seed 1
# ... repeat for seeds 2 .. 10

# --- spatial cross-validation, leave-one-zone-out (Reviewer 3 #10, R5 #7) --
.venv/Scripts/python run_pipeline_geo.py cv K21
.venv/Scripts/python run_pipeline_geo.py cv K22
.venv/Scripts/python run_pipeline_geo.py cv K23
.venv/Scripts/python run_pipeline_geo.py cv JPK
.venv/Scripts/python run_pipeline_geo.py cv RAMP

# --- prediction intervals & permutation importance -------------------------
# (these two load the trained models/ag_main_geo — run main_geo first)
.venv/Scripts/python run_pipeline_geo.py intervals    # split-conformal 90% intervals
.venv/Scripts/python run_pipeline_geo.py importance   # permutation importance, 20 shuffles

# --- aggregate everything into tables + figures ----------------------------
python make_report_geo.py   # system python with pandas+matplotlib is enough
```

**Runtime**: each AutoGluon fit is capped at `time_limit = 100 s` (documented);
the full sequence above takes ≈ 45–70 min on a desktop CPU. No GPU is required.
`baselines`, `intervals`, `importance` and `make_report_geo.py` finish in seconds.

## 4. Documented configuration

| Item | Value |
|---|---|
| AutoGluon | 1.6.2 (`autogluon.tabular`); LightGBM 4.7.0, XGBoost 3.4.1, CatBoost 1.2.10, scikit-learn 1.9.1 |
| presets | `good_quality` |
| Base learners | 13 variants from 7 families (GBM×3, CAT×2, XGB×2, RF×2, XT×1, KNN×2, LR×1) — every hyperparameter explicitly written in `run_pipeline.py::hyperparameters()` |
| Stacking | 5-fold bagging, 2 stacker levels + final weighted ensemble |
| Split | Session 6 train (n=81) / Session 7 calibration (`tuning_data`, bagging holdout) / Session 8 independent test |
| Target | next-session displacement **increment** (mm); cumulative displacement is a derived reconstruction |
| Main seed | 2024; robustness seeds 1–10 |
| time_limit | 100 s per fit |
| Full record | `docs/environment.json` + `docs/pip_freeze_full.txt` |

## 5. Results file → manuscript mapping

| Manuscript item | File(s) |
|---|---|
| Data audit (81 pts × 8 sessions = 648 records; cumulative-vs-increment consistency) | `results_geo/data_audit_real.json` |
| Split table (sessions, sizes, roles) | `results_geo/table_split.csv` |
| Main results incl. simple baselines (increment MAE/RMSE/R² + cumulative R²) | `results_geo/table_main_results.csv`, `figures_geo/fig1_main_results.png` |
| Ablation ladder A–E | `results_geo/table_ablation.csv`, `figures_geo/fig3_ablation.png` |
| Multi-seed robustness (11 runs, mean ± std) | `results_geo/table_multiseed.csv`, `table_multiseed_summary.json`, `figures_geo/fig2_multiseed.png` |
| Leave-one-zone spatial CV | `results_geo/table_zone_cv.csv`, `figures_geo/fig4_zone_cv.png` |
| 90% split-conformal prediction intervals + empirical coverage | `results_geo/intervals_summary.json`, `pred_intervals_session8.csv`, `figures_geo/fig6_prediction_intervals.png` |
| Permutation importance (held-out Session 8, 20 shuffles) | `results_geo/table_permutation_importance.csv`, `figures_geo/fig5_importance.png` |
| Per-point predictions for every model | `results_geo/pred_*.csv` |
| Full AutoGluon leaderboards (all 13+ learners per run) | `results_geo/leaderboard_*.csv` |
| Static geological/mining feature values per monitoring point | `data/geo_features.csv`; documentation: `data/C1-C81_geology_mining_mapping.xlsx` |

## 6. Data provenance statement

The monitoring data were transcribed from the four settlement-observation
summary tables on pp. 121–124 of the construction-stage goaf investigation report
(Goaf Investigation for the Zhengzhou–Luoyang Expressway construction design, Aug 2022):
81 points (C1–C81) × 8 sessions (2021-12-10 … 2022-07-16) = 648 records. Increments
are kept as the primary truth; cumulative displacement is reconstructed as the
running sum of increments (67.3% of the reported cumulative values match the
cumulated increments within ±0.05 mm; 100% within 0.2 mm — see
`results_geo/data_audit_real.json`).

Static geological/mining features (mining method and room/pillar geometry, mining
depth, seam thickness, overburden lithology, dip angle, recovery rate, years since
abandonment, mine ownership, stability class) are assigned per point from
survey-report Tables 4.6.1/5.1.4 and Sections 4.1/5.1.5; the point-by-point
assignment and its evidence are documented in
`data/C1-C81_geology_mining_mapping.xlsx`. **No cement grouting or any other
ground reinforcement was applied in the monitored area during the monitoring
window (2021-11 ~ 2022-07)** — the `grouting` feature is a documented constant 0;
the grouting schemes in the report are post-monitoring treatment recommendations.

## 7. Notes

- `models/` (trained AutoGluon artifacts, several hundred MB) is **not** shipped;
  it is regenerated by the `main*` / `ablation` / `seed` / `cv` stages. The
  `intervals` and `importance` stages require `models/ag_main_geo`, i.e. run
  `main_geo` before them.
- All figures and tables use English labels only.
- Torch warnings ("Failed to import torch …") may appear during AutoGluon fits;
  they are harmless — no neural-network learner is used in the documented
  13-learner configuration.
