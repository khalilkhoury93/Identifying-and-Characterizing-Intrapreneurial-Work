"""
Generate Figure 4.2: Robustness of IF quadrant intrapreneurial prevalence across thresholds.

Reads PLAN_3_FILES/outputs/tables/p3_q6_2_task_level_data.csv and computes
intrapreneurial prevalence by quadrant for thresholds:
- lenient (3.5/3.5)
- current (4.0/4.0)
- strict (4.5/4.5)
- median (data-driven medians)

Outputs:
- Thesis_done_analysis/03_Figures/Q6_ONET_Metadata/fig_q6_2_threshold_robustness.png
- Thesis_done_analysis/03_Figures/Q6_ONET_Metadata/fig_q6_2_threshold_robustness_table.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def assign_quadrant(importance: float, frequency: float, imp_th: float, freq_th: float) -> str:
    hi_imp = importance >= imp_th
    hi_freq = frequency >= freq_th
    if hi_imp and hi_freq:
        return "Core"
    if hi_imp and not hi_freq:
        return "Critical"
    if not hi_imp and hi_freq:
        return "Operational"
    return "Peripheral"


def main() -> None:
    base = Path("PLAN_3_FILES")
    in_csv = base / "outputs" / "tables" / "p3_q6_2_task_level_data.csv"
    df = pd.read_csv(in_csv)
    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    imp = cols.get("importance", "Importance")
    freq = cols.get("frequency", "Frequency")
    intrap = cols.get("intrap_binary", "intrap_binary")
    df[imp] = pd.to_numeric(df[imp], errors="coerce")
    df[freq] = pd.to_numeric(df[freq], errors="coerce")
    df[intrap] = pd.to_numeric(df[intrap], errors="coerce")

    thresholds = [
        ("lenient", 3.5, 3.5),
        ("current", 4.0, 4.0),
        ("strict", 4.5, 4.5),
    ]
    imp_med = float(df[imp].median())
    freq_med = float(df[freq].median())
    thresholds.append(("median", imp_med, freq_med))

    rows = []
    for name, it, ft in thresholds:
        q = df[[imp, freq, intrap]].dropna().copy()
        q["quadrant_re"] = [assign_quadrant(i, f, it, ft) for i, f in zip(q[imp].values, q[freq].values)]
        for qn in ("Core", "Critical", "Operational", "Peripheral"):
            sub = q[q["quadrant_re"].eq(qn)]
            n = int(len(sub))
            pct = float("nan") if n == 0 else float(100.0 * (sub[intrap].eq(1).sum()) / n)
            rows.append({"threshold": name, "imp_th": it, "freq_th": ft, "quadrant": qn, "n": n, "pct_intrap": pct})

    res = pd.DataFrame(rows)
    out_dir = Path("Thesis_done_analysis/03_Figures/Q6_ONET_Metadata")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fig_q6_2_threshold_robustness_table.csv").write_text(res.to_csv(index=False), encoding="utf-8")

    order = ["lenient", "current", "strict", "median"]
    qorder = ["Core", "Critical", "Operational", "Peripheral"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    axes = axes.ravel()
    ymax = max(40.0, np.nanmax(res["pct_intrap"].values) + 5.0)
    colors = ["#7FB3D5", "#2E86AB", "#1B4F72", "#5DADE2"]
    labels = ["3.5/3.5", "4.0/4.0", "4.5/4.5", "median"]
    for ax, qn in zip(axes, qorder):
        sub = res[res["quadrant"].eq(qn)].set_index("threshold").loc[order]
        ax.bar(range(len(order)), sub["pct_intrap"].values, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels)
        ax.set_title(qn)
        ax.set_ylim(0, ymax)
        for i, v in enumerate(sub["pct_intrap"].values):
            if not np.isnan(v):
                ax.text(i, v + 1.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Intrapreneurial Prevalence by Quadrant Across IF Thresholds")
    fig.text(0.5, 0.02, "IF Thresholds (Importance/Frequency)", ha="center")
    fig.text(0.02, 0.5, "Intrapreneurial Prevalence (%)", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.03, 0.05, 1, 0.95])
    out_png = out_dir / "fig_q6_2_threshold_robustness.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")

    print("Wrote:")
    print("  -", out_png)
    print("  -", out_dir / "fig_q6_2_threshold_robustness_table.csv")


if __name__ == "__main__":
    main()

