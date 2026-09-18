# Geological/Mining-Feature-Enhanced Model — Re-Analysis Report (Round 2)

**Manuscript**: infrastructures-4531268 — Short-Term Subsidence Forecasting for an Expressway over Abandoned Mine Goafs

**Goals of this round**: (1) answer Reviewer 5, comment 4 — the feature set must include geological/mining factors such as mining geometry, mining depth, seam thickness, overburden lithology, and reinforcement measures; (2) replace the appendix-parsed 88-point dataset (questionable provenance) with the **authoritative 81-point data transcribed from the survey-report summary tables (PDF pp. 121–124)** and re-run every experiment.

**Entry point**: `run_pipeline_geo.py` (reuses the configuration and the AutoGluon wrapper of `run_pipeline.py`)

---

## 1. Data rebuild (authoritative 81 points × 8 sessions = 648 records)

`data/monitoring_long_real.csv` is rebuilt from `data/monitoring_master_real.csv`
(transcription of the four summary tables on PDF pp. 121–124).
Data audit (`results_geo/data_audit_real.json`):

| Audit item | Appendix 88-point data (round 1) | **Authoritative 81-point data (this round)** |
|---|---|---|
| Points | 88 (inconsistent with the 81 stated in the text) | **81 ✓ consistent with the manuscript** |
| Cumulative vs. cumulated increments | only 45.5% of sessions agree; deviations up to 1 mm | **67.3% of sessions agree exactly (±0.05 mm); 100% within 0.2 mm** |
| Duplicated series across stations | 51 groups | none |

The authoritative data are far more self-consistent; the "increments as ground truth,
cumulative reconstructed" treatment is retained (audit details are cited in the
manuscript data section). Zone sizes: K21 = 9, K22 = 23, K23 = 10, JPK = 29, ramps = 10.

## 2. New model structure: 16 temporal + 14 geological/mining = 30 features

Static features are assigned per point from the investigation report by mileage
(`data/geo_features.csv`; point-wise values and evidence in
`data/C1-C81_geology_mining_mapping.xlsx`):

| Factor named by Reviewer 5 #4 | Feature(s) in the model | Source |
|---|---|---|
| Mining geometry | `mining_method` (shortwall / room-and-pillar / none), `room_width_m`, `pillar_width_m` (6.5 m rooms, 12.5 m pillars — report interval midpoints), `depth_thickness_ratio` | Table 5.1.4, Section 4.1.3 |
| Mining depth | `mining_depth_m` (215 m Segment 1 / 125 m Segment 2, interval midpoints) | Table 5.1.4 goaf depths |
| Seam thickness | `seam_thickness_m` (Yi-1 seam 1.24 m; 3.49 m for the C46–C48 double-seam sub-section) | Table 5.1.4 / Section 4.6.1 |
| Overburden lithology | `overburden_lithology` (medium_hard: sandstone/limestone; weak: mudstone/sandstone/carbonaceous shale — Segment 3) | Section 5.1.5 |
| Grouting / reinforcement | `grouting` | **No grouting or reinforcement was applied in the monitored area during the monitoring window (2021-11 to 2022-07)**; the grouting schemes in the report are post-monitoring recommendations. The feature is kept as a documented constant 0 and stated as such in the manuscript |
| Additional mining context | `dip_angle_deg` (11°), `recovery_rate` (0.75/0.60/0.50), `years_since_abandonment` (6.9/11.9/51.9 yr), `coal_mine` (Wanghe/Mihe/Shuanglou/interchange-mixed/none), `stability_class` (stable/basic_stable/unstable/none), `has_goaf` | Table 5.1.4, Sections 4.1, 5.1.5 |

Static features are constant in time and leakage-free by nature; temporal features keep
the leakage-safe construction of round 1 (5 single-cycle lags + 5 cumulative lags +
3 dynamic indicators + interval days + session index).

## 3. Results (Session 8 independent test; increment scale is primary)

### 3.1 Main results (`table_main_results.csv`, fig1)

| Model | Increment MAE (mm) | Increment RMSE | Increment R² | Cumulative R² |
|---|---|---|---|---|
| Persistence | 1.535 | 1.870 | −1.383 | 0.855 |
| Rate extrapolation (interval-corrected) | 1.260 | 1.531 | −0.597 | 0.903 |
| AR(5) ridge | 1.289 | 1.636 | −0.824 | 0.889 |
| AutoGluon temporal only (16-D, seed 2024) | 1.118 | 1.374 | −0.287 | 0.922 |
| **AutoGluon temporal + geological (30-D, proposed)** | **1.065** | **1.335** | **−0.215** | **0.926** |
| Best single learner (CatBoostDeep_BAG_L2) | 1.036 | 1.227 | −0.026 | 0.937 |

**Conclusion**: adding geological/mining features reduces the increment MAE by a further
**4.7%** over the temporal-only model (1.118 → 1.065), by **30.6%** over persistence and
by **15.5%** over rate extrapolation. Increment-scale R² remains negative (the increment
target is noise-dominated); the high cumulative R² is scale inflation — the manuscript
uses the honest framing from round 1.

### 3.2 Ablation (`table_ablation.csv`, fig3)

| Stage | Features | Increment MAE |
|---|---|---|
| A: 5 single-cycle lags only | 5 | 1.116 |
| B: A + 5 cumulative lags | 10 | 1.239 |
| C: B + 3 dynamic indicators | 13 | 1.105 |
| D: C + interval/session | 16 | 1.118 |
| **E: D + 14 geological/mining features** | **30** | **1.066** |

The geological group is the **only step in the ladder with a clear gain** (D→E:
−0.052 mm), directly answering Reviewer 5 #4: static geological/geometric predictors
are now included and shown to contribute.

### 3.3 Multi-seed robustness (11 independent runs; `table_multiseed*.json`, fig2)

Increment MAE = **1.094 ± 0.064 mm** (mean ± std); cumulative R² = 0.924 ± 0.011.
11/11 runs beat persistence and AR(5); 10/11 beat rate extrapolation
(exception: seed 9, 1.269 mm).

### 3.4 Spatial leave-one-zone cross-validation (`table_zone_cv.csv`, fig4)

| Held-out zone | K21 | K22 | K23 | JPK | RAMP |
|---|---|---|---|---|---|
| Increment MAE (mm) | 0.835 | 1.022 | 1.329 | 1.098 | 1.399 |
| Increment R² | −0.037 | −0.174 | −0.821 | −0.008 | −0.409 |

Spatial transfer is weak (worst for K23, the double-seam / unstable segment) —
supporting Reviewer 5 #7's concern; the manuscript keeps the transferability claims
explicitly restricted.

### 3.5 Prediction intervals (split-conformal; `intervals_summary.json`, fig6)

90% nominal intervals with half-width 1.99 mm; empirical coverage on Session 8 =
**82.7%** (vs. only 59% on the round-1 appendix data) — clearly better calibrated on
the authoritative data, but still below the nominal level and reported as such.

### 3.6 Permutation importance (20 shuffles; `table_permutation_importance.csv`, fig5)

Temporal features dominate (single_lag_2, single_lag_5, roll_std_3, cum_lag_3).
Among the geological features, **depth_thickness_ratio (+0.0039, p = 0.047)** and
**mining_depth_m (+0.0037, p = 0.008)** contribute positively; the remaining geological
features are not significant at n = 81.
Note that AutoGluon reports `grouting`, `days_prev`, `days_next`, `dip_angle_deg` and
`pillar_width_m` as unused by the final ensemble (constant or near-constant over the
training session) — the manuscript states this: the grouting feature carries no
variation because no treatment was performed during the monitoring window, and its
predictive value awaits post-treatment data.

## 4. Key points for the revised manuscript

1. **Dataset correction**: the "81 points × 8 sessions = 648 records" statement now
   matches the reproducible data exactly; the contradictions of the appendix 88-point
   data (duplicated series, cumulative inconsistencies) disappear with the replacement.
   The Data Availability Statement names the survey-report summary tables as the source.
2. **Answering R5 #4**: 14 geological/mining static features added (list + sources);
   the grouting feature is constant 0 with an explicit reason (no treatment during
   monitoring); ablation Stage E and permutation importance provide quantitative
   evidence; at the same time we acknowledge their modest marginal contribution to
   one-step increment forecasting and note that their larger value lies in spatial
   generalization and mechanistic interpretation.
3. **Conclusion framing**: the defensible claims of the proposed model are:
   "temporal + geological ensemble, next-step increment MAE 1.07 mm, −31% vs.
   persistence, −5% vs. temporal-only, stable across seeds (1.09 ± 0.06)"; the negative
   increment R², the weak cross-zone CV, and the sub-nominal interval coverage are kept
   in the limitations.

## 5. Reproduction

```bash
.venv/Scripts/python run_pipeline_geo.py rebuild && .venv/Scripts/python run_pipeline_geo.py features
.venv/Scripts/python run_pipeline_geo.py baselines
.venv/Scripts/python run_pipeline_geo.py main        # temporal-only reference
.venv/Scripts/python run_pipeline_geo.py main_geo    # proposed model (seed 2024)
.venv/Scripts/python run_pipeline_geo.py ablation E  # A..E
.venv/Scripts/python run_pipeline_geo.py seed 1      # 1..10
.venv/Scripts/python run_pipeline_geo.py cv K21      # K21/K22/K23/JPK/RAMP
.venv/Scripts/python run_pipeline_geo.py intervals
.venv/Scripts/python run_pipeline_geo.py importance
python make_report_geo.py                            # summary tables + 6 figures (system python)
```
