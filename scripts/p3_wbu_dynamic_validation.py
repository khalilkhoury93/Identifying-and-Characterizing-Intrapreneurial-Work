"""
Validation of WBU using the 'Dynamic' reason flag

Analyses (intrapreneurial tasks, rating level):
- Dynamic prevalence by WBU quintiles with 95% Wilson CIs
- ROC/AUC of WBU as a predictor of Dynamic
- Logistic models: Dynamic ~ WBU (+ Uncertainty)
- Quadrant-wise enrichment: OR of Dynamic in top vs bottom WBU quintile

Outputs:
- tables/p3_wbu_dynamic_quintiles.csv
- tables/p3_wbu_dynamic_roc_auc.txt
- tables/p3_wbu_dynamic_logit_summary.txt
- tables/p3_wbu_dynamic_quadrant_or.csv
- figures/p3_wbu_dynamic_validation.png/.svg
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    adj = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    lo = (center - adj) / denom
    hi = (center + adj) / denom
    return float(lo), float(hi)


def auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    mask = ~np.isnan(y) & ~np.isnan(s)
    y = y[mask]
    s = s[mask]
    if y.sum() == 0 or (len(y) - y.sum()) == 0:
        return np.nan
    # Mann-Whitney U relation
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    sum_ranks_pos = ranks[y == 1].sum()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    auc = U / (n_pos * n_neg)
    return float(auc)


def roc_curve_points(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    mask = ~np.isnan(y) & ~np.isnan(s)
    y = y[mask]
    s = s[mask]
    # Sort descending by score
    order = np.argsort(-s)
    y = y[order]
    s = s[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1])
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = tps / P
    fpr = fps / N
    # add origin
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return fpr, tpr


def main() -> None:
    np.random.seed(42)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load per-rating WBU (created by p3_q6_wbu_index.py)
    wbu = pd.read_csv(OUTPUT_DIR / "tables" / "p3_wbu_per_rating.csv")
    # Load reasons (Dynamic flag) and merge
    worker = pd.read_parquet(DATA_DIR / "interim" / "domain_worker_desires.parquet")
    col_dyn = "Reasons for Human Agency - Dynamic"
    if col_dyn not in worker.columns:
        raise RuntimeError("Dynamic reason column not found in domain_worker_desires.parquet")
    reasons = worker[["user_id", "task_id", col_dyn]].copy()
    reasons[col_dyn] = reasons[col_dyn].astype(str).str.lower().isin(["true", "1", "yes", "y", "t"]).astype(int)

    df = wbu.merge(reasons, on=["user_id", "task_id"], how="left")
    df = df[df["classification"].eq("Intrapreneurial")].copy()
    df.rename(columns={col_dyn: "dynamic"}, inplace=True)

    # Drop missing dynamic
    df = df[~df["dynamic"].isna()].copy()

    # Quintiles of WBU
    df["wbu_quintile"] = pd.qcut(df["wbu"], 5, labels=["Q1","Q2","Q3","Q4","Q5"]) if df["wbu"].notna().sum() >= 5 else np.nan
    quint = []
    if df["wbu_quintile"].notna().any():
        for q in ["Q1","Q2","Q3","Q4","Q5"]:
            sub = df[df["wbu_quintile"].eq(q)]
            n = len(sub)
            k = int(sub["dynamic"].sum())
            lo, hi = wilson_ci(k, n) if n > 0 else (np.nan, np.nan)
            quint.append({"quintile": q, "n": int(n), "dynamic_prevalence": (k/n if n else np.nan), "ci_lo": lo, "ci_hi": hi})
    quint_df = pd.DataFrame(quint)
    quintile_out = TABLES_DIR / "p3_wbu_dynamic_quintiles.csv"
    quint_df.to_csv(quintile_out, index=False)

    # ROC/AUC
    auc = auc_from_scores(df["dynamic"].values, df["wbu"].values)
    fpr, tpr = roc_curve_points(df["dynamic"].values, df["wbu"].values)
    auc_txt = TABLES_DIR / "p3_wbu_dynamic_roc_auc.txt"
    auc_txt.write_text(f"AUC (WBU -> Dynamic) = {auc:.3f}\nN={len(df)}\n", encoding="utf-8")

    # Logistic models: Dynamic ~ WBU (+ Uncertainty)
    logit_txt = TABLES_DIR / "p3_wbu_dynamic_logit_summary.txt"
    try:
        import statsmodels.api as sm
        X1 = pd.DataFrame({"const": 1.0, "wbu": df["wbu"].astype(float)})
        y = df["dynamic"].astype(int)
        model1 = sm.GLM(y, X1, family=sm.families.Binomial()).fit()

        X2 = pd.DataFrame({
            "const": 1.0,
            "wbu": df["wbu"].astype(float),
            "uncertainty": df["involved_uncertainty"].astype(float)
        })
        model2 = sm.GLM(y, X2, family=sm.families.Binomial()).fit()

        logit_txt.write_text("Model 1: Dynamic ~ WBU\n" + str(model1.summary()) + "\n\n" +
                             "Model 2: Dynamic ~ WBU + Uncertainty\n" + str(model2.summary()) + "\n", encoding="utf-8")
    except Exception as e:
        logit_txt.write_text(f"statsmodels not available or failed: {e}\n", encoding="utf-8")

    # Quadrant-wise enrichment: top vs bottom WBU quintile
    quad_rows = []
    order = ["Core", "Critical", "Routine", "Peripheral"]
    for quad in order:
        sub = df[df["quadrant"].eq(quad)].copy()
        if sub.empty or sub["wbu_quintile"].isna().all():
            quad_rows.append({"quadrant": quad, "n": 0, "or_top_vs_bottom": np.nan, "p_value": np.nan})
            continue
        top = sub[sub["wbu_quintile"].eq("Q5")]
        bot = sub[sub["wbu_quintile"].eq("Q1")]
        a = int(top["dynamic"].sum()); b = int(len(top) - a)
        c = int(bot["dynamic"].sum()); d = int(len(bot) - c)
        if min(a,b,c,d) == 0:
            # Haldane-Anscombe correction
            a += 0.5; b += 0.5; c += 0.5; d += 0.5
        or_val = (a * d) / (b * c) if (b * c) != 0 else np.nan
        # Fisher test
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]])
        quad_rows.append({"quadrant": quad, "n": int(len(sub)), "or_top_vs_bottom": float(or_val) if or_val==or_val else np.nan, "p_value": float(p)})
    quad_df = pd.DataFrame(quad_rows)
    quad_out = TABLES_DIR / "p3_wbu_dynamic_quadrant_or.csv"
    quad_df.to_csv(quad_out, index=False)

    # Figure: Panel A prevalence by quintile, Panel B ROC, Panel C quadrant ORs
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    if not quint_df.empty:
        x = np.arange(len(quint_df))
        prev = quint_df["dynamic_prevalence"].values
        lo = quint_df["ci_lo"].values
        hi = quint_df["ci_hi"].values
        ax1.bar(x, prev, color="#2E86AB", alpha=0.9, edgecolor="black")
        for i, (p, l, h) in enumerate(zip(prev, lo, hi)):
            ax1.vlines(i, l, h, colors="black", linewidth=1)
        ax1.set_xticks(x)
        ax1.set_xticklabels(quint_df["quintile"].tolist())
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Pr(Dynamic=True)")
        ax1.set_title("A. Dynamic prevalence by WBU quintile (95% CI)")
        ax1.grid(axis="y", alpha=0.3)
    else:
        ax1.axis("off")

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    if len(fpr) > 1:
        ax2.plot(fpr, tpr, color="#A23B72", label=f"AUC={auc:.3f}")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title("B. ROC (WBU → Dynamic)")
        ax2.legend()
        ax2.grid(alpha=0.3)
    else:
        ax2.axis("off")

    # Panel C
    ax3 = fig.add_subplot(gs[0, 2])
    sub = quad_df.copy()
    order = ["Core", "Critical", "Routine", "Peripheral"]
    sub = sub.set_index("quadrant").loc[order].reset_index()
    y = np.arange(len(sub))
    ax3.axvline(1.0, color="black", linestyle="--", alpha=0.5)
    ax3.scatter(sub["or_top_vs_bottom"], y, color="#1E8449")
    # annotate n and p
    for i, r in sub.iterrows():
        ax3.text(r["or_top_vs_bottom"] + 0.02, i, f"OR={r['or_top_vs_bottom']:.2f}\n(p={r['p_value']:.3g})", va="center", fontsize=8)
    ax3.set_yticks(y)
    ax3.set_yticklabels(sub["quadrant"].tolist())
    ax3.set_xlabel("Odds ratio (Top vs Bottom WBU)")
    ax3.set_title("C. Quadrant enrichment")
    ax3.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_png = FIGURES_DIR / "p3_wbu_dynamic_validation.png"
    out_svg = FIGURES_DIR / "p3_wbu_dynamic_validation.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    print("Saved:")
    print(f"  - {quintile_out}")
    print(f"  - {auc_txt}")
    print(f"  - {logit_txt}")
    print(f"  - {quad_out}")
    print(f"  - {out_png}")
    print(f"  - {out_svg}")


if __name__ == "__main__":
    main()

