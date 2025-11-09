"""
Figure: Involved Uncertainty (Worker vs Expert) across IF quadrants for intrapreneurial tasks

Panel A: Grouped bars (means with 95% CI) per quadrant
Panel B: Forest plot of Worker-Expert mean differences with 95% CI

Saves PNG and SVG under outputs/figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


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
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    samples = values[idx].mean(axis=1)
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def load_task_level() -> pd.DataFrame:
    expert = pd.read_parquet(DATA_DIR / "interim" / "expert_rated_technological_capa.parquet")
    worker = pd.read_parquet(DATA_DIR / "interim" / "domain_worker_desires.parquet")
    feats = pd.read_parquet(DATA_DIR / "processed" / "features_experts.parquet")
    labels = pd.read_csv(OUTPUT_DIR / "summaries" / "intrap0_1_full_aggregate.csv")

    exp_task = (
        expert.groupby("task_id")
        .agg(expert_uncert=("expert_involved_uncertainty", "mean"))
        .reset_index()
    )
    work_task = (
        worker.groupby("task_id")
        .agg(worker_uncert=("involved_uncertainty", "mean"))
        .reset_index()
    )

    feats = feats[["task_id", "Importance", "Frequency"]].drop_duplicates()
    labels = labels[["task_id", "classification"]]

    df = feats.merge(labels, on="task_id", how="inner")
    df = df.merge(exp_task, on="task_id", how="left")
    df = df.merge(work_task, on="task_id", how="left")
    df = df[df["classification"].eq("Intrapreneurial")].copy()
    df["quadrant"] = df.apply(lambda r: assign_quadrant(float(r["Importance"]), float(r["Frequency"])), axis=1)
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    order = ["Core", "Critical", "Routine", "Peripheral"]
    for q in order:
        sub = df[df["quadrant"].eq(q)][["worker_uncert", "expert_uncert"]].copy()
        n = len(sub.dropna(how="any"))
        mean_worker = sub["worker_uncert"].mean()
        mean_expert = sub["expert_uncert"].mean()
        ciw_lo, ciw_hi = bootstrap_ci_mean(sub["worker_uncert"].values)
        cie_lo, cie_hi = bootstrap_ci_mean(sub["expert_uncert"].values)

        diffs = (sub["worker_uncert"] - sub["expert_uncert"]).dropna().values
        diff_mean = np.nanmean(diffs) if diffs.size else np.nan
        diff_lo, diff_hi = bootstrap_ci_mean(diffs)
        try:
            stat, p = stats.wilcoxon(diffs) if diffs.size > 0 else (np.nan, np.nan)
        except ValueError:
            stat, p = (np.nan, np.nan)

        rows.append(
            {
                "quadrant": q,
                "n": int(n),
                "mean_worker": float(mean_worker) if pd.notna(mean_worker) else np.nan,
                "mean_expert": float(mean_expert) if pd.notna(mean_expert) else np.nan,
                "ci_worker_lo": ciw_lo,
                "ci_worker_hi": ciw_hi,
                "ci_expert_lo": cie_lo,
                "ci_expert_hi": cie_hi,
                "diff_mean": float(diff_mean) if pd.notna(diff_mean) else np.nan,
                "diff_lo": diff_lo,
                "diff_hi": diff_hi,
                "p_wilcoxon": float(p) if pd.notna(p) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def p_to_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def plot(stats_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    order = ["Core", "Critical", "Routine", "Peripheral"]
    stats_df = stats_df.set_index("quadrant").loc[order].reset_index()

    # Colors
    worker_col = "#2E86AB"
    expert_col = "#A23B72"

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # Panel A: Grouped bars with 95% CI
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    width = 0.36

    # Worker bars
    w = ax1.bar(
        x - width / 2,
        stats_df["mean_worker"],
        width,
        yerr=[stats_df["mean_worker"] - stats_df["ci_worker_lo"], stats_df["ci_worker_hi"] - stats_df["mean_worker"]],
        capsize=4,
        label="Worker",
        color=worker_col,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )
    # Expert bars
    e = ax1.bar(
        x + width / 2,
        stats_df["mean_expert"],
        width,
        yerr=[stats_df["mean_expert"] - stats_df["ci_expert_lo"], stats_df["ci_expert_hi"] - stats_df["mean_expert"]],
        capsize=4,
        label="Expert",
        color=expert_col,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(order)
    ax1.set_ylabel("Involved Uncertainty (1–5)")
    ax1.set_ylim(0, 5.4)
    ax1.set_title("A. Means by Quadrant (95% CI)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    # Annotate stars for paired Wilcoxon
    for i, q in enumerate(order):
        p = stats_df.loc[i, "p_wilcoxon"]
        stars = p_to_stars(p)
        if stars:
            ymax = max(stats_df.loc[i, "ci_worker_hi"], stats_df.loc[i, "ci_expert_hi"]) + 0.15
            ax1.plot([i - width / 2, i + width / 2], [ymax, ymax], color="black", linewidth=1)
            ax1.text(i, ymax + 0.05, stars, ha="center", va="bottom", fontsize=12)

    # Panel B: Forest plot of Worker - Expert differences
    ax2 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(order))
    diffs = stats_df["diff_mean"].values
    lo = stats_df["diff_lo"].values
    hi = stats_df["diff_hi"].values

    ax2.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax2.errorbar(diffs, y, xerr=[diffs - lo, hi - diffs], fmt="o", color="#444", ecolor="#777", capsize=4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(order)
    ax2.set_xlabel("Worker − Expert (mean difference)")
    ax2.set_title("B. Difference with 95% CI")
    ax2.grid(axis="x", alpha=0.3)

    for i, (m, lo_i, hi_i) in enumerate(zip(diffs, lo, hi)):
        ax2.text(hi_i + 0.03, i, f"{m:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    out_png = FIGURES_DIR / "p3_q6_quadrant_uncertainty_worker_vs_expert.png"
    out_svg = FIGURES_DIR / "p3_q6_quadrant_uncertainty_worker_vs_expert.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure:\n  {out_png}\n  {out_svg}")


def main() -> None:
    df = load_task_level()
    stats_df = compute_stats(df)
    print("Computed stats:\n", stats_df.to_string(index=False))
    plot(stats_df)


if __name__ == "__main__":
    main()

