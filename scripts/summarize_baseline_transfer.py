"""Aggregate external-baseline transfer results side-by-side with adaptive/orig.

Reads <baselines-in>/transfer/<MODEL>/<pair>/Experiment*/ and
<main-in>/{adaptive,orig}/<pair>/Experiment*/,
computes battery-averaged MAPE/RMSE (same gap split as evaluate.py),
writes <out>/summary_transfer.json and prints a per-pair comparison table
(fine-tune predictions; zero-shot src_only predictions remain in the
experiment dirs for later analysis).

Defaults reproduce the in-tree aggregation:
    python scripts/summarize_baseline_transfer.py
Path A (shipped paper_data, release layout):
    python scripts/summarize_baseline_transfer.py \
        --baselines-in paper_data/results/baselines --main-in paper_data/results/transfer \
        --out evaluation/baselines
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate as ev  # noqa: E402

BASELINE_MODELS = ["LSTM", "GRU", "CNN", "MLP", "GPR"]
MAIN_MODELS = ["adaptive", "orig"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baselines-in", default="results/baselines",
                    help="baselines results root containing transfer/ (default: results/baselines)")
    ap.add_argument("--main-in", default="results/transfer",
                    help="adaptive/orig transfer results root (default: results/transfer)")
    ap.add_argument("--out", default="evaluation/baselines",
                    help="output directory (default: evaluation/baselines)")
    args = ap.parse_args()

    def root(rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else PROJECT_ROOT / rel

    baselines_root = root(args.baselines_in) / "transfer"
    main_root = root(args.main_in)
    out_dir = root(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: set[str] = set()
    for base in (baselines_root, *(main_root / m for m in MAIN_MODELS if (main_root / m).exists())):
        if base.exists():
            pairs |= {p.name for p in base.iterdir() if p.is_dir() and "-to-" in p.name}

    # Scan each model tree once; calling scan_transfer inside the double loop
    # would re-walk every tree for every pair.
    scans = {model: ev.scan_transfer(main_root if model in MAIN_MODELS else baselines_root, model)
             for model in MAIN_MODELS + BASELINE_MODELS}

    summary = {}
    for pair in sorted(pairs):
        row = {}
        for model in MAIN_MODELS + BASELINE_MODELS:
            reps = scans[model].get(pair, [])
            if not reps:
                continue
            s = ev.summarize(reps)
            row[model] = {k: {"mean": v[0], "std": v[1], "n": len(reps)} for k, v in s.items()}
        summary[pair] = row

    (out_dir / "summary_transfer.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    for pair, row in summary.items():
        print(f"\n== {pair} ==")
        print(f"{'model':10s} {'n':>3s} {'MAPE':>20s} {'RMSE':>20s}")
        for model, v in row.items():
            n = v["MAPE"]["n"]
            print(f"{model:10s} {n:3d} {v['MAPE']['mean']:9.4f} +/- {v['MAPE']['std']:.4f}"
                  f" {v['RMSE']['mean']:9.4f} +/- {v['RMSE']['std']:.4f}")
    print(f"\nwrote {out_dir / 'summary_transfer.json'}")


if __name__ == "__main__":
    main()
