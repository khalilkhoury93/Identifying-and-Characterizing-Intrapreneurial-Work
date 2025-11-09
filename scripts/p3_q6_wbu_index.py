"""
WBU (Willingness to Bear Uncertainty) analysis for worker ratings

Computes:
- Per-rating WBU index = weight(uncertainty) * [z_user(human_agency) - z_user(automation_desire)]
- Per-user slopes: human_agency ~ involved_uncertainty; automation_desire ~ involved_uncertainty
- Quadrant aggregates of WBU for intrapreneurial tasks (current thresholds)
- Compact figure: distribution of WBU + forest of per-user slopes

Outputs:
- tables/p3_wbu_per_rating.csv
- tables/p3_wbu_per_user_slopes.csv
- tables/p3_wbu_quadrant_aggregates.csv
- figures/p3_wbu_compact.png and .svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


def zscore_within_group(x: pd.Series, group: pd.Series) -> pd.Series:
    df = pd.DataFrame({"x": x, "g": group})
    def _z(sub: pd.DataFrame) -> pd.Series:
        v = sub["x"].astype(float)
        m = v.mean()
        s = v.std(ddof=0)
        if s == 0 or np.isnan(s):
            return pd.Series(np.zeros(len(v)), index=sub.index)
        return (v - m) / s
    return df.groupby("g", group_keys=False).apply(_z)


def assign_quadrant(importance: float, frequency: float, imp_th: float = 4.0, freq_th: float = 4.0) -> str:
    hi_imp = importance >= imp_th
    hi_freq = frequency >= freq_th
    if hi_imp and hi_freq:
        return "Core"
    if hi_imp and not hi_freq:
        return "Critical"
    if not hi_imp and hi_freq:
        return "Routine"
    return "Peripheral"


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 5000, seed: int = 42, ci: float = 0.95) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[idx].mean(axis=1)
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def bootstrap_ci_slope(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 42, ci: float = 0.95) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    slopes = []
    for row in idx:
        xb = x[row]
        yb = y[row]
        try:
            b1 = np.polyfit(xb, yb, 1)[0]
        except Exception:
            b1 = np.nan
        slopes.append(b1)
    slopes = np.array(slopes, dtype=float)
    slopes = slopes[~np.isnan(slopes)]
    if slopes.size == 0:
        return (np.nan, np.nan)
    lo = np.percentile(slopes, (1 - ci) / 2 * 100)
    hi = np.percentile(slopes, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def main() -> None:
    np.random.seed(42)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    workers = pd.read_parquet(DATA_DIR / "interim" / "domain_worker_desires.parquet")
    feats = pd.read_parquet(DATA_DIR / "processed" / "features_experts.parquet")
    labels = pd.read_csv(OUTPUT_DIR / "summaries" / "intrap0_1_full_aggregate.csv")

    # Keep necessary columns
    w = workers[[
        "user_id", "task_id", "involved_uncertainty", "human_agency", "automation_desire"
    ]].dropna().copy()

    # Per-user standardization
    w["z_human_agency"] = zscore_within_group(w["human_agency"], w["user_id"])
    w["z_automation_desire"] = zscore_within_group(w["automation_desire"], w["user_id"])

    # Uncertainty weight in [0, 1]
    w["w_uncertainty"] = (w["involved_uncertainty"].astype(float) - 1.0) / 4.0
    w["w_uncertainty"] = w["w_uncertainty"].clip(lower=0.0, upper=1.0)

    # WBU per rating
    w["wbu_base"] = w["z_human_agency"] - w["z_automation_desire"]
    w["wbu"] = w["w_uncertainty"] * w["wbu_base"]

    # Attach labels and IF quadrants (using current thresholds)
    labels_small = labels[["task_id", "classification"]]
    w = w.merge(labels_small, on="task_id", how="left")
    feats_small = feats[["task_id", "Importance", "Frequency"]].drop_duplicates()
    w = w.merge(feats_small, on="task_id", how="left")
    w["quadrant"] = w.apply(
        lambda r: assign_quadrant(float(r["Importance"]) if pd.notna(r["Importance"]) else -np.inf,
                                  float(r["Frequency"]) if pd.notna(r["Frequency"]) else -np.inf),
        axis=1,
    )

    # Save per-rating table (include classification/quadrant for filtering downstream)
    per_rating_out = TABLES_DIR / "p3_wbu_per_rating.csv"
    keep_cols = [
        "user_id", "task_id", "involved_uncertainty", "human_agency", "automation_desire",
        "z_human_agency", "z_automation_desire", "w_uncertainty", "wbu_base", "wbu",
        "classification", "Importance", "Frequency", "quadrant"
    ]
    w[keep_cols].to_csv(per_rating_out, index=False)

    # Filter to intrapreneurial tasks for slope and quadrant summaries
    w_intrap = w[w["classification"].eq("Intrapreneurial")].copy()

    # Per-user slopes (min 5 ratings on intrap tasks)
    rows: List[Dict] = []
    for uid, g in w_intrap.groupby("user_id"):
        g = g[["involved_uncertainty", "human_agency", "automation_desire", "wbu_base"]].dropna()
        if len(g) < 5:
            continue
        x = g["involved_uncertainty"].astype(float).values
        y_ha = g["human_agency"].astype(float).values
        y_ad = g["automation_desire"].astype(float).values
        y_wb = g["wbu_base"].astype(float).values
        try:
            slope_ha = float(np.polyfit(x, y_ha, 1)[0])
        except Exception:
            slope_ha = np.nan
        try:
            slope_ad = float(np.polyfit(x, y_ad, 1)[0])
        except Exception:
            slope_ad = np.nan
        try:
            slope_wb = float(np.polyfit(x, y_wb, 1)[0])
        except Exception:
            slope_wb = np.nan
        ci_ha_lo, ci_ha_hi = bootstrap_ci_slope(x, y_ha)
        ci_ad_lo, ci_ad_hi = bootstrap_ci_slope(x, y_ad)
        ci_wb_lo, ci_wb_hi = bootstrap_ci_slope(x, y_wb)
        rows.append({
            "user_id": uid,
            "n_ratings": int(len(g)),
            "slope_human_agency": slope_ha,
            "slope_human_agency_lo": ci_ha_lo,
            "slope_human_agency_hi": ci_ha_hi,
            "slope_automation_desire": slope_ad,
            "slope_automation_desire_lo": ci_ad_lo,
            "slope_automation_desire_hi": ci_ad_hi,
            "slope_wbu_base": slope_wb,
            "slope_wbu_base_lo": ci_wb_lo,
            "slope_wbu_base_hi": ci_wb_hi,
        })

    slopes_df = pd.DataFrame(rows).sort_values("slope_human_agency", ascending=False)
    per_user_out = TABLES_DIR / "p3_wbu_per_user_slopes.csv"
    slopes_df.to_csv(per_user_out, index=False)

    # Quadrant aggregates for intrap tasks with known quadrants
    q = w_intrap.dropna(subset=["quadrant"])
    q_rows: List[Dict] = []
    order = ["Core", "Critical", "Routine", "Peripheral"]
    for quad in order:
        sub = q[q["quadrant"].eq(quad)]
        n_ratings = len(sub)
        n_tasks = sub["task_id"].nunique()
        n_users = sub["user_id"].nunique()
        mean_wbu = float(sub["wbu"].mean()) if n_ratings else np.nan
        ci_lo, ci_hi = bootstrap_ci_mean(sub["wbu"].values) if n_ratings else (np.nan, np.nan)
        med = float(sub["wbu"].median()) if n_ratings else np.nan
        q_rows.append({
            "quadrant": quad,
            "n_ratings": int(n_ratings),
            "n_tasks": int(n_tasks),
            "n_users": int(n_users),
            "mean_wbu": mean_wbu,
            "median_wbu": med,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        })
    quad_df = pd.DataFrame(q_rows)
    quad_out = TABLES_DIR / "p3_wbu_quadrant_aggregates.csv"
    quad_df.to_csv(quad_out, index=False)

    # Compact figure: distribution + forest of slopes
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # Panel A: distribution of per-rating WBU (intrap only)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(w_intrap["wbu"], bins=30, kde=True, stat="probability", color="#2E86AB", alpha=0.85, ax=ax1)
    ax1.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax1.set_title("A. Per-rating WBU (Intrap tasks)")
    ax1.set_xlabel("WBU = w(unc) * [z(HA) − z(AD)]")
    ax1.set_ylabel("Percentage")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.grid(axis="y", alpha=0.3)

    # Panel B: forest of per-user slopes (HA and AD)
    ax2 = fig.add_subplot(gs[0, 1])
    s = slopes_df.copy()
    if len(s) > 0:
        y = np.arange(len(s))
        # Human Agency slopes
        ax2.errorbar(
            s["slope_human_agency"].values, y,
            xerr=[s["slope_human_agency"].values - s["slope_human_agency_lo"].values,
                  s["slope_human_agency_hi"].values - s["slope_human_agency"].values],
            fmt="o", color="#1E8449", ecolor="#58D68D", capsize=3, label="HA~Uncertainty"
        )
        # Automation Desire slopes
        ax2.errorbar(
            s["slope_automation_desire"].values, y,
            xerr=[s["slope_automation_desire"].values - s["slope_automation_desire_lo"].values,
                  s["slope_automation_desire_hi"].values - s["slope_automation_desire"].values],
            fmt="o", color="#A23B72", ecolor="#E5989B", capsize=3, label="AD~Uncertainty"
        )
        ax2.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax2.set_yticks(y)
        # Render compact user labels (last 6 chars) to avoid clutter
        ax2.set_yticklabels([str(u)[-6:] for u in s["user_id"]])
        ax2.set_title("B. Per-user slopes (Intrap; 95% CI)")
        ax2.set_xlabel("Slope")
        ax2.legend()
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.axis("off")

    plt.tight_layout()
    out_png = FIGURES_DIR / "p3_wbu_compact.png"
    out_svg = FIGURES_DIR / "p3_wbu_compact.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("Saved tables:")
    print(f"  - {per_rating_out}")
    print(f"  - {per_user_out}")
    print(f"  - {quad_out}")
    print("Saved figures:")
    print(f"  - {out_png}")
    print(f"  - {out_svg}")


if __name__ == "__main__":
    main()
