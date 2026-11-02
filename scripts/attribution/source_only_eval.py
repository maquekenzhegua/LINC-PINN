"""Step-2 attribution control: score the archived source-only predictions.

``scripts/transfer_eval.py`` saved ``src_only_true_label.npy`` /
``src_only_pred_label.npy`` (predictions BEFORE target fine-tuning) next to
the headline predictions in every transfer experiment folder.  This script
aggregates those arrays with the same upstream label-gap battery metrics as
``scripts/evaluate.py`` (no GPU, no training), giving the source-only
baseline of the transfer attribution triangle:

  source-only  ->  what a source-pretrained model scores on the target
  full transfer (fine-tuned)  -> archived headline numbers
  target-only (scripts/attribution/target_only.py)  -> what training on the target alone reaches

Run from the repository root:
    python scripts/attribution/source_only_eval.py
Output:
    evaluation/source_only/summary.json (+ printed comparison)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.evaluate import battery_metrics  # same upstream label-gap metric  # noqa: E402

# pair label -> target dataset gap (same thresholds as scripts/evaluate.py)
GAPS = {"XJTU": 0.05, "TJU": 0.07, "HUST": 0.10, "MIT": 0.07}


def pair_target_gap(pair: str) -> float:
    tgt = pair.split("-to-")[-1]
    return GAPS.get(tgt.split("-")[0], 0.05)


def agg(values: List[float]) -> Dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {"mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "n": int(arr.size)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-root", default="paper_data/results/transfer",
                    help="transfer results root with src_only_* arrays "
                         "(default: shipped paper_data mirror)")
    ap.add_argument("--ref", default="paper_data/evaluation/reproduce/summary_transfer.json",
                    help="reference (fine-tuned) transfer summary for the comparison")
    ap.add_argument("--out", default="evaluation/source_only",
                    help="output directory (default: evaluation/source_only)")
    args = ap.parse_args()

    def root(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / rel

    in_root = root(args.in_root)
    out_dir = root(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.provenance import record_provenance
    record_provenance(out_dir, args)

    summary: Dict[str, Dict] = {}
    for model_dir in sorted(in_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for pair_dir in sorted(model_dir.iterdir()):
            if not pair_dir.is_dir():
                continue
            gap = pair_target_gap(pair_dir.name)
            mapes, rmses, maes = [], [], []
            for exp_dir in sorted(pair_dir.glob("Experiment*")):
                t = exp_dir / "src_only_true_label.npy"
                p = exp_dir / "src_only_pred_label.npy"
                if not (t.exists() and p.exists()):
                    continue
                m = battery_metrics(np.load(t), np.load(p), gap)
                maes.append(m["MAE"]); mapes.append(m["MAPE"]); rmses.append(m["RMSE"])
            key = f"{model_dir.name}/{pair_dir.name}"
            if mapes:
                summary[key] = {"MAE": agg(maes), "MAPE": agg(mapes), "RMSE": agg(rmses)}
                print(f"[src-only] {key}: MAPE={np.mean(mapes):.4f} RMSE={np.mean(rmses):.4f} (n={len(mapes)})")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    # comparison against the full (fine-tuned) transfer numbers
    ref_path = root(args.ref)
    ref = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}
    lines = ["model/pair,src_only_MAPE,full_MAPE,gain_MAPE,src_only_RMSE,full_RMSE"]
    for key, row in summary.items():
        ref_row = ref.get(f"transfer/{key}", {})
        full_mape = ref_row.get("MAPE", {}).get("mean")
        full_rmse = ref_row.get("RMSE", {}).get("mean")
        lines.append(",".join([
            key,
            f"{row['MAPE']['mean']:.6f}",
            f"{full_mape:.6f}" if full_mape is not None else "",
            f"{full_mape - row['MAPE']['mean']:+.6f}" if full_mape is not None else "",
            f"{row['RMSE']['mean']:.6f}",
            f"{full_rmse:.6f}" if full_rmse is not None else "",
        ]))
    (out_dir / "comparison_vs_full_transfer.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'summary.json'} and comparison_vs_full_transfer.csv")


if __name__ == "__main__":
    main()
