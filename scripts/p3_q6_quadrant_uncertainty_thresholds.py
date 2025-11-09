"""
Threshold sensitivity: Involved Uncertainty (Worker vs Expert) gaps across IF quadrants
for intrapreneurial tasks, under multiple quadrant threshold definitions.

Threshold sets:
- strict:  Importance>=4.5, Frequency>=4.5
- current: Importance>=4.0, Frequency>=4.0
- lenient: Importance>=3.5, Frequency>=3.5
- median:  Importance>=median(all expert-rated tasks), Frequency>=median(all expert-rated tasks)

Outputs:
- tables/p3_q6_uncertainty_thresholds.csv (per-threshold, per-quadrant stats)
- figures/p3_q6_uncertainty_diffs_by_threshold.png/.svg (forest plots by threshold)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


def assign_quadrant(imp: float, freq: float, imp_th: float, freq_th: float) -> str:
    hi_imp = imp >= imp_th
    hi_freq = freq >= freq_th
    if hi_imp and hi_freq:
        return "Core"
    if hi_imp and not hi_freq:
        return "Critical"
    if not hi_imp and hi_freq:
        return "Routine"
    return "Peripheral"


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 5000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    x = np.asarray(values)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    samples = x[idx].mean(axis=1)
    lo = np.percentile(samples, (1 - ci) / 2 * 100)
    hi = np.percentile(samples, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expert = pd.read_parquet(DATA_DIR / "interim" / "expert_rated_technological_capa.parquet")
    worker = pd.read_parquet(DATA_DIR / "interim" / "domain_worker_desires.parquet")
    feats = pd.read_parquet(DATA_DIR / "processed" / "features_experts.parquet")
    labels = pd.read_csv(OUTPUT_DIR / "summaries" / "intrap0_1_full_aggregate.csv")
    return expert, worker, feats, labels


def prepare_task_level(expert: pd.DataFrame, worker: pd.DataFrame, feats: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Per-task means
    exp_task = (
        expert.groupby("task_id")["expert_involved_uncertainty"].mean().rename("expert_uncert").reset_index()
    )
    work_task = (
        worker.groupby("task_id")["involved_uncertainty"].mean().rename("worker_uncert").reset_index()
    )

    # All tasks (for medians)
    feats_all = feats[["task_id", "Importance", "Frequency"]].drop_duplicates()

    # Intrap tasks only
    intrap_ids = labels.loc[labels["classification"] == "Intrapreneurial", ["task_id"]]
    feats_intrap = feats_all.merge(intrap_ids, on="task_id", how="inner")

    # Attach ratings
    df = feats_intrap.merge(exp_task, on="task_id", how="left").merge(work_task, on="task_id", how="left")
    return df, feats_all


def compute_stats_for_threshold(df: pd.DataFrame, imp_th: float, freq_th: float, label: str) -> pd.DataFrame:
    d = df.copy()
    d["quadrant"] = d.apply(lambda r: assign_quadrant(float(r["Importance"]), float(r["Frequency"]), imp_th, freq_th), axis=1)
    order = ["Core", "Critical", "Routine", "Peripheral"]
    rows: List[Dict] = []
    for q in order:
        sub = d[d["quadrant"].eq(q)][["worker_uncert", "expert_uncert"]].dropna(how="any").copy()
        n = len(sub)
        mean_worker = sub["worker_uncert"].mean() if n else np.nan
        mean_expert = sub["expert_uncert"].mean() if n else np.nan
        ciw_lo, ciw_hi = bootstrap_ci_mean(sub["worker_uncert"].values) if n else (np.nan, np.nan)
        cie_lo, cie_hi = bootstrap_ci_mean(sub["expert_uncert"].values) if n else (np.nan, np.nan)
        diffs = (sub["worker_uncert"] - sub["expert_uncert"]).values if n else np.array([])
        diff_mean = float(np.nanmean(diffs)) if diffs.size else np.nan
        diff_lo, diff_hi = bootstrap_ci_mean(diffs) if diffs.size else (np.nan, np.nan)
        try:
            stat, p = stats.wilcoxon(diffs) if diffs.size else (np.nan, np.nan)
        except ValueError:
            stat, p = (np.nan, np.nan)
        rows.append(
            {
                "threshold_set": label,
                "imp_threshold": float(imp_th),
                "freq_threshold": float(freq_th),
                "quadrant": q,
                "n_tasks": int(n),
                "mean_worker": float(mean_worker) if n else np.nan,
                "mean_expert": float(mean_expert) if n else np.nan,
                "ci_worker_lo": ciw_lo,
                "ci_worker_hi": ciw_hi,
                "ci_expert_lo": cie_lo,
                "ci_expert_hi": cie_hi,
                "diff_mean": diff_mean,
                "diff_lo": diff_lo,
                "diff_hi": diff_hi,
                "wilcoxon_p": float(p) if p == p else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_fdr(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.stats.multitest import multipletests
    except Exception:
        # Fallback: no FDR if statsmodels not available
        df["q_value_fdr"] = np.nan
        return df
    out = []
    for label, sub in df.groupby("threshold_set", sort=False):
        pvals = sub["wilcoxon_p"].values
        mask = ~np.isnan(pvals)
        q = np.full_like(pvals, fill_value=np.nan, dtype=float)
        if mask.sum() > 0:
            _, q_masked, _, _ = multipletests(pvals[mask], method="fdr_bh")
            q[mask] = q_masked
        sub2 = sub.copy()
        sub2["q_value_fdr"] = q
        out.append(sub2)
    return pd.concat(out, ignore_index=True)


def plot_forest(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    order = ["Core", "Critical", "Routine", "Peripheral"]
    labels = list(df["threshold_set"].drop_duplicates())
    ncol = len(labels)

    fig, axes = plt.subplots(1, ncol, figsize=(4.5 * ncol, 4.5), sharey=True)
    if ncol == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        sub = df[df["threshold_set"] == label].set_index("quadrant").loc[order].reset_index()
        y = np.arange(len(order))
        diffs = sub["diff_mean"].values
        lo = sub["diff_lo"].values
        hi = sub["diff_hi"].values
        ax.axvline(0, color="black", linestyle="--", alpha=0.4)
        ax.errorbar(diffs, y, xerr=[diffs - lo, hi - diffs], fmt="o", color="#333", ecolor="#777", capsize=4)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.set_xlabel("Worker − Expert")
        ax.set_title(label.capitalize())
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_ylabel("Quadrant")
    plt.tight_layout()
    out_png = FIGURES_DIR / "p3_q6_uncertainty_diffs_by_threshold.png"
    out_svg = FIGURES_DIR / "p3_q6_uncertainty_diffs_by_threshold.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}\nSaved: {out_svg}")


def main() -> None:
    np.random.seed(42)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    expert, worker, feats, labels = load_data()
    df_intrap, feats_all = prepare_task_level(expert, worker, feats, labels)

    # Threshold sets
    imp_med = float(feats_all["Importance"].median())
    freq_med = float(feats_all["Frequency"].median())
    thresh_sets = [
        ("strict", 4.5, 4.5),
        ("current", 4.0, 4.0),
        ("lenient", 3.5, 3.5),
        ("median", imp_med, freq_med),
    ]

    parts = []
    for label, imp_th, freq_th in thresh_sets:
        print(f"Computing: {label} (Imp>={imp_th:.2f}, Freq>={freq_th:.2f})")
        part = compute_stats_for_threshold(df_intrap, imp_th, freq_th, label)
        parts.append(part)
    res = pd.concat(parts, ignore_index=True)
    res = add_fdr(res)

    out_csv = TABLES_DIR / "p3_q6_uncertainty_thresholds.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved table: {out_csv}")

    # Print a concise summary
    for label, sub in res.groupby("threshold_set", sort=False):
        print("\n" + "=" * 70)
        print(f"Threshold: {label}")
        print("quadrant        n  mean_worker  mean_expert  diff_mean   [diff_lo, diff_hi]    p    q")
        for _, r in sub.iterrows():
            print(
                f"{r['quadrant']:<10}  {int(r['n_tasks']):4d}    {r['mean_worker']:.3f}       {r['mean_expert']:.3f}      {r['diff_mean']:.3f}    "
                f"[{r['diff_lo']:.3f}, {r['diff_hi']:.3f}]  {r['wilcoxon_p']:.3g}  {r['q_value_fdr']:.3g}"
            )

    plot_forest(res)


if __name__ == "__main__":
    main()

