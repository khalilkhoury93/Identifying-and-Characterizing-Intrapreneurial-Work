from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


def bootstrap_ci_slope(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42, ci: float = 0.95) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x = x[m]
    y = y[m]
    if x.size < 3:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    slopes = []
    for row in idx:
        xb = x[row]
        yb = y[row]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.RankWarning)
                b1 = float(np.polyfit(xb, yb, 1)[0])
        except Exception:
            b1 = np.nan
        slopes.append(b1)
    s = np.array(slopes, dtype=float)
    s = s[~np.isnan(s)]
    if s.size == 0:
        return (np.nan, np.nan)
    lo = float(np.percentile(s, (1 - ci) / 2 * 100))
    hi = float(np.percentile(s, (1 + ci) / 2 * 100))
    return lo, hi


def build_compact(min_ratings: int) -> Path:
    df = pd.read_csv(TABLES_DIR / "p3_wbu_per_rating.csv")
    df = df[df["classification"].eq("Intrapreneurial")].copy()

    # Panel A: per-rating WBU distribution
    wbu = df["wbu"].astype(float)

    # Panel B: per-user slopes with >= min_ratings
    rows: List[Dict] = []
    for uid, g in df.groupby("user_id"):
        g2 = g[["involved_uncertainty", "human_agency", "automation_desire"]].dropna()
        if len(g2) < min_ratings:
            continue
        x = g2["involved_uncertainty"].astype(float).values
        y_ha = g2["human_agency"].astype(float).values
        y_ad = g2["automation_desire"].astype(float).values
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.RankWarning)
                slope_ha = float(np.polyfit(x, y_ha, 1)[0])
        except Exception:
            slope_ha = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.RankWarning)
                slope_ad = float(np.polyfit(x, y_ad, 1)[0])
        except Exception:
            slope_ad = np.nan
        lo_ha, hi_ha = bootstrap_ci_slope(x, y_ha)
        lo_ad, hi_ad = bootstrap_ci_slope(x, y_ad)
        rows.append({
            "user_id": uid,
            "n_ratings": int(len(g2)),
            "slope_human_agency": slope_ha,
            "slope_human_agency_lo": lo_ha,
            "slope_human_agency_hi": hi_ha,
            "slope_automation_desire": slope_ad,
            "slope_automation_desire_lo": lo_ad,
            "slope_automation_desire_hi": hi_ad,
        })
    slopes = pd.DataFrame(rows).sort_values("slope_human_agency", ascending=False)

    # Plot
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # Left: distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(wbu, bins=30, kde=True, stat="probability", color="#2E86AB", alpha=0.85, ax=ax1)
    ax1.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax1.set_title("A. Per-rating WBU (Intrap tasks)")
    ax1.set_xlabel("WBU = w(unc) × [z(HA) − z(AD)]")
    ax1.set_ylabel("Percentage")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.grid(axis="y", alpha=0.3)

    # Right: per-user slopes
    ax2 = fig.add_subplot(gs[0, 1])
    if not slopes.empty:
        y = np.arange(len(slopes))
        ax2.errorbar(
            slopes["slope_human_agency"].values, y,
            xerr=[slopes["slope_human_agency"].values - slopes["slope_human_agency_lo"].values,
                  slopes["slope_human_agency_hi"].values - slopes["slope_human_agency"].values],
            fmt="o", color="#1E8449", ecolor="#58D68D", capsize=3, label="HA~Uncertainty"
        )
        ax2.errorbar(
            slopes["slope_automation_desire"].values, y,
            xerr=[slopes["slope_automation_desire"].values - slopes["slope_automation_desire_lo"].values,
                  slopes["slope_automation_desire_hi"].values - slopes["slope_automation_desire"].values],
            fmt="o", color="#A23B72", ecolor="#E5989B", capsize=3, label="AD~Uncertainty"
        )
        ax2.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax2.set_yticks(y)
        ax2.set_yticklabels([str(u)[-6:] for u in slopes["user_id"]])
        ax2.set_title(f"B. Per-user slopes (≥{min_ratings} ratings; 95% CI)")
        ax2.set_xlabel("Slope")
        ax2.legend()
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.axis("off")

    plt.tight_layout()
    out = FIGURES_DIR / f"p3_wbu_compact_min{min_ratings}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for thr in (3, 4):
        out = build_compact(thr)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
