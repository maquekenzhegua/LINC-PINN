"""Gamma-sweep collapse summary: prediction range and MAPE vs small_gamma.

The small-sample collapse signature (the 2026-09 collapse analysis) is a
prediction range that collapses to a near-constant while gamma=10 protects flat
solutions.  This script builds the curve for the paper figure:
per training-battery count n, MAPE (battery-level, from the evaluate.py
summaries) and the mean raw prediction span max(pred)-min(pred) per
experiment, for the full config (gamma=0.2, from the main reproduce tree)
and the sweep values gamma in {1, 2, 5, 10} (from <results-root>/ablation/gamma*).

Output: <eval-dir>/gamma_sweep_summary.csv

Defaults reproduce the in-tree aggregation:
    python scripts/summarize_gamma_sweep.py
Path A (shipped results mirror, release layout):
    python scripts/summarize_gamma_sweep.py --results-root results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pred_range_mean(cell_dir: Path) -> tuple[float, int]:
    """Mean raw prediction span over completed experiments of one cell."""
    spans: list[float] = []
    for exp in sorted(cell_dir.glob("Experiment*")):
        pred = exp / "pred_label.npy"
        if pred.exists():
            arr = np.load(pred).ravel()
            if arr.size:
                spans.append(float(arr.max() - arr.min()))
    return (float(np.mean(spans)) if spans else float("nan"), len(spans))


def rows_from_summary(summary_path: Path, results_root: Path, config: str) -> list[dict]:
    if not summary_path.exists():
        return []
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = []
    for key in sorted(data):
        if not key.startswith("small/adaptive/small-XJTU-2C-n"):
            continue
        n = int(key.rsplit("n", 1)[1])
        cell = results_root / "small" / "adaptive" / f"small-XJTU-2C-n{n}"
        span, n_exp = pred_range_mean(cell)
        m = data[key]["MAPE"]
        rows.append(
            {
                "config": config,
                "n": n,
                "mape_mean": m["mean"],
                "mape_std": m["std"],
                "pred_range_mean": span,
                "n_runs": m["n"],
                "n_experiments_for_range": n_exp,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default="evaluation",
                    help="directory holding summary.json and ablation/gamma*/summary.json "
                         "(default: evaluation)")
    ap.add_argument("--results-root", default="results",
                    help="results root holding reproduce/ and ablation/gamma* "
                         "(default: results)")
    ap.add_argument("--main-subdir", default="reproduce",
                    help="main-suite results subdir name (default: reproduce)")
    ap.add_argument("--sweep-subdir", default="ablation",
                    help="sweep results subdir name (default: ablation)")
    ap.add_argument("--eval-ablation-subdir", default="ablation",
                    help="evaluation-side ablation subdir name (default: ablation)")
    args = ap.parse_args()

    def root(rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else ROOT / rel

    eval_dir = root(args.eval_dir)
    res = root(args.results_root)

    main_root = res / args.main_subdir
    sweep = res / args.sweep_subdir
    eval_abl = eval_dir / args.eval_ablation_subdir
    rows = rows_from_summary(
        eval_dir / "summary.json", main_root, "gamma0.2_full"
    )
    for g in (1, 2, 5, 10):
        rows += rows_from_summary(
            eval_abl / f"gamma{g}" / "summary.json",
            sweep / f"gamma{g}",
            f"gamma{g}",
        )
    if not rows:
        print("no gamma-sweep rows found yet")
        return
    df = pd.DataFrame(rows).sort_values(["config", "n"])
    out = eval_dir / "gamma_sweep_summary.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
