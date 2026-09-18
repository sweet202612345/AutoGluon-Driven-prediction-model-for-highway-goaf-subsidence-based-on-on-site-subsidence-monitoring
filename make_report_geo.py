# -*- coding: utf-8 -*-
"""Aggregate all results_geo outputs into summary tables + figures (system python)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_geo"
FIG = ROOT / "figures_geo"
FIG.mkdir(exist_ok=True)


def load(tag):
    return json.loads((OUT / f"metrics_{tag}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------- main table -----
base = pd.read_csv(OUT / "metrics_baselines.csv")
rows = list(base.itertuples(index=False))
main = load("main")
geo = load("main_geo")
tbl = pd.DataFrame([
    {"model": "Persistence", "inc_mae": base.iloc[0]["inc_mae"], "inc_rmse": base.iloc[0]["inc_rmse"],
     "inc_r2": base.iloc[0]["inc_r2"], "cum_r2": base.iloc[0]["cum_r2"]},
    {"model": "Rate extrapolation", "inc_mae": base.iloc[1]["inc_mae"], "inc_rmse": base.iloc[1]["inc_rmse"],
     "inc_r2": base.iloc[1]["inc_r2"], "cum_r2": base.iloc[1]["cum_r2"]},
    {"model": "AR(5) ridge", "inc_mae": base.iloc[2]["inc_mae"], "inc_rmse": base.iloc[2]["inc_rmse"],
     "inc_r2": base.iloc[2]["inc_r2"], "cum_r2": base.iloc[2]["cum_r2"]},
    {"model": "AutoGluon temporal (16-D)", "inc_mae": main["inc_mae"], "inc_rmse": main["inc_rmse"],
     "inc_r2": main["inc_r2"], "cum_r2": main["cum_r2"]},
    {"model": "AutoGluon temporal+geo (30-D, proposed)", "inc_mae": geo["inc_mae"], "inc_rmse": geo["inc_rmse"],
     "inc_r2": geo["inc_r2"], "cum_r2": geo["cum_r2"]},
    {"model": f"Best single learner ({geo['bestsingle_name']})", "inc_mae": geo["bestsingle_inc_mae"],
     "inc_rmse": geo["bestsingle_inc_rmse"], "inc_r2": geo["bestsingle_inc_r2"], "cum_r2": geo["bestsingle_cum_r2"]},
])
tbl.to_csv(OUT / "table_main_results.csv", index=False)
print(tbl.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
colors = ["#9e9e9e"] * 3 + ["#5b8ff9", "#d9534f", "#bdbdbd"]
ax[0].barh(tbl["model"], tbl["inc_mae"], color=colors)
ax[0].set_xlabel("Increment MAE (mm) \u2193")
ax[0].set_title("Session 8 independent test: increment scale")
for i, v in enumerate(tbl["inc_mae"]):
    ax[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
ax[1].barh(tbl["model"], tbl["cum_r2"], color=colors)
ax[1].set_xlabel("Cumulative reconstructed R\u00b2 \u2191")
ax[1].set_title("Cumulative scale (R\u00b2 inflated, reference only)")
for i, v in enumerate(tbl["cum_r2"]):
    ax[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig1_main_results.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ------------------------------------------------------------ ablation -----
abl = []
labels = {"A": "A: 5 single-cycle lags", "B": "B: +5 cumulative lags", "C": "C: +3 dynamic indicators",
          "D": "D: +interval/session (16-D)", "E": "E: +14 geological features (30-D)"}
for k in "ABCDE":
    m = load(f"ablation{k}")
    abl.append({"stage": labels[k], "inc_mae": m["inc_mae"], "inc_rmse": m["inc_rmse"],
                "inc_r2": m["inc_r2"], "cum_r2": m["cum_r2"]})
abt = pd.DataFrame(abl)
abt.to_csv(OUT / "table_ablation.csv", index=False)
print(abt.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(abt["stage"], abt["inc_mae"], color=["#9e9e9e", "#9e9e9e", "#9e9e9e", "#5b8ff9", "#d9534f"])
ax.set_ylabel("Increment MAE (mm) \u2193")
ax.set_title("Ablation: feature groups added stepwise (Session 8 independent test)")
for i, v in enumerate(abt["inc_mae"]):
    ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
plt.xticks(rotation=12)
fig.tight_layout()
fig.savefig(FIG / "fig3_ablation.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ----------------------------------------------------------- multiseed -----
seeds = [load("main_geo")] + [load(f"seed{i}") for i in range(1, 11)]
ms = pd.DataFrame([{"seed": s["seed"], "inc_mae": s["inc_mae"], "inc_r2": s["inc_r2"],
                    "cum_r2": s["cum_r2"]} for s in seeds])
ms.to_csv(OUT / "table_multiseed.csv", index=False)
summ = {"n_runs": len(ms),
        "inc_mae_mean": float(ms["inc_mae"].mean()), "inc_mae_std": float(ms["inc_mae"].std()),
        "inc_r2_mean": float(ms["inc_r2"].mean()), "inc_r2_std": float(ms["inc_r2"].std()),
        "cum_r2_mean": float(ms["cum_r2"].mean()), "cum_r2_std": float(ms["cum_r2"].std()),
        "all_better_than_persistence_mae": bool((ms["inc_mae"] < base.iloc[0]["inc_mae"]).all()),
        "all_better_than_rate_extrap_mae": bool((ms["inc_mae"] < base.iloc[1]["inc_mae"]).all()),
        "all_better_than_ar5_mae": bool((ms["inc_mae"] < base.iloc[2]["inc_mae"]).all())}
(OUT / "table_multiseed_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
print(json.dumps(summ, ensure_ascii=False, indent=2))

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(ms["seed"].astype(str), ms["inc_mae"], color="#d9534f")
ax.axhline(base.iloc[0]["inc_mae"], ls="--", c="#555", label=f"Persistence {base.iloc[0]['inc_mae']:.3f}")
ax.axhline(base.iloc[1]["inc_mae"], ls="--", c="#2e7d32", label=f"Rate extrapolation {base.iloc[1]['inc_mae']:.3f}")
ax.axhline(ms["inc_mae"].mean(), ls="-", c="#d9534f", alpha=.5,
           label=f"Mean {ms['inc_mae'].mean():.3f}\u00b1{ms['inc_mae'].std():.3f}")
ax.set_xlabel("Random seed"); ax.set_ylabel("Increment MAE (mm) \u2193")
ax.set_title("Stability over 11 independent training runs (temporal + geological model)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig2_multiseed.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ----------------------------------------------------------------- zone cv --
cv = []
for z in ["K21", "K22", "K23", "JPK", "RAMP"]:
    m = load(f"cv_{z}")
    cv.append({"zone": z, "inc_mae": m["inc_mae"], "inc_rmse": m["inc_rmse"], "inc_r2": m["inc_r2"]})
cvt = pd.DataFrame(cv)
cvt.to_csv(OUT / "table_zone_cv.csv", index=False)
print(cvt.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(cvt["zone"], cvt["inc_mae"], color="#5b8ff9")
ax[0].set_ylabel("Increment MAE (mm) \u2193")
ax[0].set_title("Leave-one-zone CV: increment MAE")
for i, v in enumerate(cvt["inc_mae"]):
    ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax[1].bar(cvt["zone"], cvt["inc_r2"], color="#9e9e9e")
ax[1].set_ylabel("Increment R\u00b2")
ax[1].set_title("Leave-one-zone CV: increment R\u00b2")
for i, v in enumerate(cvt["inc_r2"]):
    ax[1].text(i, v - 0.06, f"{v:.3f}", ha="center", fontsize=9)
fig.suptitle("Spatial zone cross-validation (temporal + geological features)")
fig.tight_layout()
fig.savefig(FIG / "fig4_zone_cv.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ------------------------------------------------------------ importance ---
fi = pd.read_csv(OUT / "table_permutation_importance.csv", index_col=0)
fi = fi.sort_values("importance")
GEO = ["has_goaf", "seam_thickness_m", "mining_depth_m", "depth_thickness_ratio",
       "dip_angle_deg", "recovery_rate", "years_since_abandonment", "room_width_m",
       "pillar_width_m", "grouting", "mining_method", "overburden_lithology",
       "coal_mine", "stability_class"]
fi["type"] = ["geological" if f in GEO else "temporal" for f in fi.index]
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(fi.index, fi["importance"], xerr=fi["stddev"],
        color=["#d9534f" if t == "geological" else "#5b8ff9" for t in fi["type"]])
ax.set_xlabel("Permutation importance (\u0394R\u00b2, 20 shuffles, Session 8 independent test)")
ax.set_title("Feature permutation importance (red = geological/mining, blue = temporal)")
fig.tight_layout()
fig.savefig(FIG / "fig5_importance.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- intervals -
pi = pd.read_csv(OUT / "pred_intervals_session8.csv")
summ_pi = json.loads((OUT / "intervals_summary.json").read_text())
order = np.argsort(pi["y_inc"].values)
fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(pi))
ax.fill_between(x, pi["pi_lo"].values[order], pi["pi_hi"].values[order],
                alpha=.3, color="#5b8ff9", label=f"90% prediction interval (half-width {summ_pi['half_width_mm']:.2f} mm)")
ax.plot(x, pi["y_inc"].values[order], "o", ms=3, color="#222", label="Measured increment")
ax.plot(x, pi["pred_inc"].values[order], "-", lw=1, color="#d9534f", label="Predicted increment")
ax.set_xlabel("Test points (sorted by measured increment)"); ax.set_ylabel("Increment (mm)")
ax.set_title(f"Session 8 prediction intervals: nominal 90%, empirical coverage "
             f"{summ_pi['empirical_coverage_session8']*100:.1f}%")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig6_prediction_intervals.png", bbox_inches="tight", dpi=150)
plt.close(fig)

print("figures ->", [p.name for p in sorted(FIG.glob('*.png'))])
