from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
TABLE_PATH = BASE_DIR / "outputs" / "tables" / "p3_wbu_per_rating.csv"


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan")
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return float("nan")


def main() -> None:
    df = pd.read_csv(TABLE_PATH)
    df = df[df["classification"].eq("Intrapreneurial")].copy()

    results: dict[int, dict[str, float]] = {}
    for thr in (3, 4):
        rows = []
        for uid, g in df.groupby("user_id"):
            g2 = g[["involved_uncertainty", "human_agency", "automation_desire"]].dropna()
            if len(g2) < thr:
                continue
            sh = slope(g2["involved_uncertainty"].values, g2["human_agency"].values)
            sa = slope(g2["involved_uncertainty"].values, g2["automation_desire"].values)
            rows.append((uid, len(g2), sh, sa))
        N = len(rows)
        ha_pos = sum(1 for _, _, sh, _ in rows if isinstance(sh, float) and sh > 0)
        ad_nonpos = sum(1 for _, _, _, sa in rows if isinstance(sa, float) and sa <= 0)
        results[thr] = {
            "N": float(N),
            "HA_pos": float(ha_pos),
            "HA_pos_pct": (ha_pos / N * 100.0) if N else float("nan"),
            "AD_nonpos": float(ad_nonpos),
            "AD_nonpos_pct": (ad_nonpos / N * 100.0) if N else float("nan"),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
