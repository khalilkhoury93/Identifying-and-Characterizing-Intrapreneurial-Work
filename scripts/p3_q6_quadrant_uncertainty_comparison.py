"""
Compare Involved Uncertainty (experts vs workers) for intrapreneurial tasks
across the four O*NET Importance x Frequency quadrants.

Outputs a summary CSV and prints a readable table.

Inputs:
- data/interim/expert_rated_technological_capa.parquet
- data/interim/domain_worker_desires.parquet
- data/processed/features_experts.parquet (Importance, Frequency)
- outputs/summaries/intrap0_1_full_aggregate.csv (Intrapreneurial labels)

Saved outputs:
- outputs/tables/p3_q6_quadrant_uncertainty_worker_vs_expert.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"


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


def paired_wilcoxon(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    s = pd.concat([x, y], axis=1).dropna()
    if len(s) < 10:
        return (np.nan, np.nan)
    try:
        stat, p = stats.wilcoxon(s.iloc[:, 0], s.iloc[:, 1], zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return (np.nan, np.nan)


def main() -> None:
    np.random.seed(42)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Q6: Expert vs Worker Involved Uncertainty by Quadrant (Intrap tasks)")
    print("=" * 80)

    # Load per-rater data
    expert = pd.read_parquet(DATA_DIR / "interim" / "expert_rated_technological_capa.parquet")
    worker = pd.read_parquet(DATA_DIR / "interim" / "domain_worker_desires.parquet")

    # Aggregate to per-task means (keep counts)
    exp_task = (
        expert.groupby("task_id")
        .agg(expert_uncert=("expert_involved_uncertainty", "mean"), n_expert_raters=("expert_user_id", "nunique"))
        .reset_index()
    )
    work_task = (
        worker.groupby("task_id")
        .agg(worker_uncert=("involved_uncertainty", "mean"), n_worker_raters=("user_id", "nunique"))
        .reset_index()
    )

    # Load task-level metadata (Importance, Frequency) restricted to experts subset
    feats = pd.read_parquet(DATA_DIR / "processed" / "features_experts.parquet")
    feats = feats[["task_id", "Importance", "Frequency"]].drop_duplicates()

    # Intrap labels
    labels = pd.read_csv(OUTPUT_DIR / "summaries" / "intrap0_1_full_aggregate.csv")
    labels = labels[["task_id", "classification"]]

    # Merge
    df = feats.merge(labels, on="task_id", how="inner")
    df = df.merge(exp_task, on="task_id", how="left")
    df = df.merge(work_task, on="task_id", how="left")

    # Filter to intrapreneurial tasks
    df = df[df["classification"].eq("Intrapreneurial")].copy()

    # Quadrants
    df["quadrant"] = df.apply(
        lambda r: assign_quadrant(float(r["Importance"]), float(r["Frequency"])), axis=1
    )

    # Compute per-quadrant summaries
    rows = []
    for q in ["Core", "Critical", "Routine", "Peripheral"]:
        sub = df[df["quadrant"].eq(q)].copy()
        n_tasks = len(sub)
        if n_tasks == 0:
            rows.append(
                {
                    "quadrant": q,
                    "n_tasks": 0,
                    "mean_expert_uncert": np.nan,
                    "mean_worker_uncert": np.nan,
                    "diff_worker_minus_expert": np.nan,
                    "median_diff": np.nan,
                    "wilcoxon_stat": np.nan,
                    "wilcoxon_p": np.nan,
                    "spearman_r": np.nan,
                    "spearman_p": np.nan,
                    "avg_n_worker_raters": np.nan,
                    "avg_n_expert_raters": np.nan,
                }
            )
            continue

        mean_exp = sub["expert_uncert"].mean()
        mean_work = sub["worker_uncert"].mean()
        diff = mean_work - mean_exp
        median_diff = (sub["worker_uncert"] - sub["expert_uncert"]).median()

        # Paired test and correlation (only where both present)
        stat_w, p_w = paired_wilcoxon(sub["worker_uncert"], sub["expert_uncert"])
        s = sub[["worker_uncert", "expert_uncert"]].dropna()
        if len(s) >= 5:
            rho, p_rho = stats.spearmanr(s["worker_uncert"], s["expert_uncert"]).correlation, stats.spearmanr(
                s["worker_uncert"], s["expert_uncert"]
            ).pvalue
        else:
            rho, p_rho = (np.nan, np.nan)

        rows.append(
            {
                "quadrant": q,
                "n_tasks": int(n_tasks),
                "mean_expert_uncert": float(mean_exp) if pd.notna(mean_exp) else np.nan,
                "mean_worker_uncert": float(mean_work) if pd.notna(mean_work) else np.nan,
                "diff_worker_minus_expert": float(diff) if pd.notna(diff) else np.nan,
                "median_diff": float(median_diff) if pd.notna(median_diff) else np.nan,
                "wilcoxon_stat": float(stat_w) if pd.notna(stat_w) else np.nan,
                "wilcoxon_p": float(p_w) if pd.notna(p_w) else np.nan,
                "spearman_r": float(rho) if pd.notna(rho) else np.nan,
                "spearman_p": float(p_rho) if pd.notna(p_rho) else np.nan,
                "avg_n_worker_raters": float(sub["n_worker_raters"].mean()) if "n_worker_raters" in sub else np.nan,
                "avg_n_expert_raters": float(sub["n_expert_raters"].mean()) if "n_expert_raters" in sub else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out_path = TABLES_DIR / "p3_q6_quadrant_uncertainty_worker_vs_expert.csv"
    out.to_csv(out_path, index=False)

    print("\nSummary (means; worker - expert difference):")
    print(out[["quadrant", "n_tasks", "mean_expert_uncert", "mean_worker_uncert", "diff_worker_minus_expert", "wilcoxon_p"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()

