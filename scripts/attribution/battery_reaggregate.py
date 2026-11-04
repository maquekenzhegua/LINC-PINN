"""P0-1: re-aggregate archived predictions by true battery IDs.

The upstream evaluation reconstructs battery boundaries from SOH label gaps
(scripts/evaluate.py).  The 2026-09-13 audit found those reconstructed
boundaries match the loader's cycle-progress resets in nine regular
conditions but not in TJU-0 and MIT.  This script adds a supplementary
cell-ID aggregation:

  1. rebuild each condition's test loader through the UNCHANGED upstream
     dataloader (CPU only, no training);
  2. locate true battery boundaries as cycle-progress resets (the progress
     column is strictly increasing inside a battery and jumps back at a new
     cell) -- the same ground truth the audit used;
  3. verify row alignment: the rebuilt test labels must match every archived
     true_label.npy of that condition;
  4. recompute battery-level MAE/MAPE/RMSE over true cells (same formulas as
     scripts/evaluate.py) and aggregate mean +/- std over repeats;
  5. write everything to evaluation/batteryid/ and a comparison against
     the label-gap numbers.

Nothing in results/ or evaluation/ is modified: the label-gap aggregation
stays authoritative for the main tables; this is an explicitly labeled
supplementary analysis (paper states the label-gap interpretation).

Run from the repository root (default inputs = shipped results mirror,
default boundaries source = rebuild from the raw datasets under data/):
    python scripts/attribution/battery_reaggregate.py
Path A (no datasets needed) reuses the shipped boundary arrays:
    python scripts/attribution/battery_reaggregate.py --boundaries-from evaluation/batteryid/boundaries
Optional: --with-baselines also aggregates the baselines trees.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts.reproduce import (  # noqa: E402
    build_adaptive_args,
    make_loaders_for_phase,
)

OUT_DIR = PROJECT_ROOT / "evaluation" / "batteryid"
EPS = 1e-9

# condition label -> (dataset, batch) for the upstream loader
CONDITIONS: Dict[str, Tuple[str, object]] = {}
for _b in ["2C", "3C", "R2.5", "R3", "RW", "satellite"]:
    CONDITIONS[f"XJTU-{_b}"] = ("XJTU", _b)
for _i in range(3):
    CONDITIONS[f"TJU-{_i}"] = ("TJU", _i)
CONDITIONS["HUST"] = ("HUST", None)
CONDITIONS["MIT"] = ("MIT", None)

# transfer pair label -> target condition label (full target test loader)
TRANSFER_TARGETS: Dict[str, str] = {
    "HUST-to-XJTU-2C": "XJTU-2C",
    "XJTU-2C-to-HUST": "HUST",
    "TJU-0-to-XJTU-2C": "XJTU-2C",
    "XJTU-2C-to-TJU-0": "TJU-0",
}


def eval_metrix(true_label, pred_label):
    from sklearn import metrics as skm
    mae = float(skm.mean_absolute_error(true_label, pred_label))
    mape = float(skm.mean_absolute_percentage_error(true_label, pred_label))
    mse = float(skm.mean_squared_error(true_label, pred_label))
    rmse = float(np.sqrt(mse))
    return mae, mape, mse, rmse


def build_boundary(dataset: str, batch) -> Tuple[np.ndarray, np.ndarray, int]:
    """Rebuild the test loader and derive true battery boundaries.

    Returns (true_labels, battery_ids, n_batteries).  Boundary = position
    where the progress column strictly decreases (new battery starts).
    """
    args = build_adaptive_args(dataset, batch)
    loaders = make_loaders_for_phase(dataset, batch, args)
    true_rows, prog_rows = [], []
    with __import__("torch").no_grad():
        for x1, _, y1, _ in loaders["test"]:
            true_rows.append(y1.numpy().reshape(-1))
            prog_rows.append(x1[:, -1].numpy().reshape(-1))
    true_cat = np.concatenate(true_rows).astype(np.float64)
    prog_cat = np.concatenate(prog_rows).astype(np.float64)
    starts = [0] + [i for i in range(1, len(prog_cat)) if prog_cat[i] < prog_cat[i - 1] - EPS]
    lengths = np.diff(starts + [len(prog_cat)])
    battery_ids = np.repeat(np.arange(len(starts)), lengths)
    return true_cat, battery_ids, len(starts)


def per_battery_metrics(true_label: np.ndarray, pred_label: np.ndarray,
                        battery_ids: np.ndarray) -> Tuple[float, float, float]:
    """Equal-weight per-cell MAE/MAPE/RMSE (same formulas as evaluate.py)."""
    maes, mapes, rmses = [], [], []
    for b in np.unique(battery_ids):
        mask = battery_ids == b
        t, p = true_label[mask], pred_label[mask]
        if len(t) < 2:
            continue
        mae, mape, _mse, rmse = eval_metrix(t, p)
        maes.append(mae)
        mapes.append(mape)
        rmses.append(rmse)
    if not mapes:
        mae, mape, _mse, rmse = eval_metrix(true_label, pred_label)
        return mae, mape, rmse
    return float(np.mean(maes)), float(np.mean(mapes)), float(np.mean(rmses))


def scan_experiments(base: Path, repeats: int = 10) -> Dict[str, List[Path]]:
    """Experiment dirs keyed by model, grouped per condition dir."""
    if not base.exists():
        return {}
    out: Dict[str, List[Path]] = {}
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        for cond_dir in sorted(model_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            exps = []
            for k in range(1, repeats + 1):
                p = cond_dir / f"Experiment{k}"
                if (p / "true_label.npy").exists() and (p / "pred_label.npy").exists():
                    exps.append(p)
            if exps:
                out[f"{model_dir.name}/{cond_dir.name}"] = exps
    return out


def aggregate_condition(exps: List[Path], true_ref: np.ndarray,
                        battery_ids: np.ndarray) -> Dict:
    maes, mapes, rmses = [], [], []
    aligned, max_diff = True, 0.0
    for exp in exps:
        t = np.load(exp / "true_label.npy").reshape(-1)
        p = np.load(exp / "pred_label.npy").reshape(-1)
        if len(t) != len(true_ref):
            aligned = False
            continue
        max_diff = max(max_diff, float(np.max(np.abs(t.astype(np.float64) - true_ref))))
        if max_diff > 1e-5:
            aligned = False
            continue
        mae, mape, rmse = per_battery_metrics(t, p, battery_ids)
        maes.append(mae)
        mapes.append(mape)
        rmses.append(rmse)
    return {"aligned": aligned, "max_diff": max_diff, "MAE": _agg(maes),
            "MAPE": _agg(mapes), "RMSE": _agg(rmses)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--with-baselines", action="store_true",
                    help="also aggregate results/baselines outputs")
    ap.add_argument("--results-root", default="results",
                    help="results root holding reproduce/transfer/baselines "
                         "(default: shipped results mirror)")
    ap.add_argument("--eval-root", default="evaluation",
                    help="label-gap summaries root used for the delta join "
                         "(default: shipped results mirror)")
    ap.add_argument("--boundaries-from", default=None,
                    help="reuse shipped boundary arrays from this directory "
                         "(e.g. evaluation/batteryid/boundaries) "
                         "instead of rebuilding loaders from the raw datasets")
    ap.add_argument("--out", default="evaluation/batteryid",
                    help="output directory (default: evaluation/batteryid)")
    args = ap.parse_args()

    def root(rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else PROJECT_ROOT / rel

    out_dir = root(args.out)

    from scripts.provenance import record_provenance
    record_provenance(out_dir, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "boundaries").mkdir(exist_ok=True)

    # ---- 1. boundaries per condition (test loader, once per condition) ----
    boundaries: Dict[str, Tuple[Optional[np.ndarray], np.ndarray, int]] = {}
    if args.boundaries_from:
        broot = root(args.boundaries_from)
        for cond in CONDITIONS:
            ids = np.load(broot / cond / "battery_id.npy")
            meta = json.loads((broot / cond / "battery_rows.json").read_text(encoding="utf-8"))
            # true_ref stays None: it is anchored lazily from the first
            # archived true_label.npy of the condition during aggregation
            boundaries[cond] = (None, ids, int(meta["n_batteries"]))
            print(f"[boundary] {cond}: reused shipped boundaries "
                  f"({meta['n_batteries']} batteries, {meta['n_rows']} rows)")
    else:
        for cond, (dataset, batch) in CONDITIONS.items():
            true_ref, ids, n = build_boundary(dataset, batch)
            d = out_dir / "boundaries" / cond
            d.mkdir(parents=True, exist_ok=True)
            np.save(d / "battery_id.npy", ids)
            (d / "battery_rows.json").write_text(
                json.dumps({"n_batteries": n, "n_rows": int(len(ids)),
                            "rows_per_battery": np.bincount(ids).tolist()},
                           indent=1), encoding="utf-8")
            boundaries[cond] = (true_ref, ids, n)
            print(f"[boundary] {cond}: {n} batteries, {len(true_ref)} rows")

    # ---- 2. aggregate archived experiments per (phase/model/cond) ----
    res_root = root(args.results_root)
    suites = {
        "regular": res_root / "reproduce" / "regular",
        "small": res_root / "reproduce" / "small",
        "transfer": res_root / "transfer",
    }
    if args.with_baselines:
        for phase in ("regular", "small", "transfer"):
            base = res_root / "baselines" / phase
            if base.exists():
                suites[f"baselines/{phase}"] = base

    summary: Dict[str, Dict] = {}
    for suite, base in suites.items():
        groups = scan_experiments(base, args.repeats)
        for key, exps in groups.items():
            model, _, cond = key.partition("/")
            # map to a condition loader (small-XJTU-2C-nK -> XJTU-2C etc.)
            if cond.startswith("small-"):
                cond_key = cond[len("small-"):].rsplit("-n", 1)[0]
            else:
                cond_key = TRANSFER_TARGETS.get(cond, cond)
            if cond_key not in boundaries:
                print(f"[skip] unknown condition {cond!r} ({suite}/{key})")
                continue
            true_ref, ids, n = boundaries[cond_key]
            if true_ref is None:
                true_ref = np.load(exps[0] / "true_label.npy").reshape(-1).astype(np.float64)
            rows = {"MAE": [], "MAPE": [], "RMSE": []}
            aligned, max_diff = True, 0.0
            for exp in exps:
                t = np.load(exp / "true_label.npy").reshape(-1)
                p = np.load(exp / "pred_label.npy").reshape(-1)
                if len(t) != len(true_ref):
                    aligned = False
                    break
                max_diff = max(max_diff, float(np.max(np.abs(t.astype(np.float64) - true_ref))))
                if max_diff > 1e-5:
                    aligned = False
                    break
                mae, mape, rmse = per_battery_metrics(t, p, ids)
                rows["MAE"].append(mae)
                rows["MAPE"].append(mape)
                rows["RMSE"].append(rmse)
            summary[f"{suite}/{key}"] = {
                "n_repeats": len(rows["MAPE"]),
                "n_batteries": int(len(np.unique(ids))),
                "alignment_ok": aligned,
                "max_true_diff": max_diff,
                "MAE": _agg(rows["MAE"]),
                "MAPE": _agg(rows["MAPE"]),
                "RMSE": _agg(rows["RMSE"]),
            }
            status = "OK" if aligned else "MISALIGNED"
            print(f"[agg] {suite}/{key}: {status} (n={len(rows['MAPE'])}, "
                  f"MAPE={summary[f'{suite}/{key}']['MAPE']['mean']:.4f})")

    _write_outputs(summary, args, root(args.eval_root), out_dir)
    print("battery re-aggregation done ->", out_dir)


def battery_id_of(boundaries, cond_key: str) -> np.ndarray:
    return boundaries[cond_key][1]


def _agg(values) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n": int(arr.size),
    }


def _write_outputs(summary: Dict[str, Dict], args, eval_root: Path,
                   out_dir: Path) -> None:
    (out_dir / "summary_batteryid.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    # join with the label-gap numbers for a delta table
    labelgap = {}
    for name in ("summary.json", "summary_transfer.json"):
        for p in (eval_root / "reproduce" / name,
                  eval_root / "baselines" / name):
            if p.exists():
                labelgap.update(json.loads(p.read_text(encoding="utf-8")))
    # baselines transfer numbers live in a pair->model nested summary
    bt_path = eval_root / "baselines" / "summary_transfer.json"
    bt = json.loads(bt_path.read_text(encoding="utf-8")) if bt_path.exists() else {}

    lines = ["suite/model/cond,aligned,n_batteries,n_repeats,"
             "cellid_MAPE,labelgap_MAPE,delta_MAPE,cellid_RMSE,labelgap_RMSE,delta_RMSE"]
    for key, row in summary.items():
        # baselines summaries key their rows without the "baselines/" prefix
        ref_key = key if key in labelgap else key.replace("baselines/", "", 1)
        ref_row = labelgap.get(ref_key, {})
        if not ref_row and key.startswith("baselines/transfer/"):
            _, _, model, pair = key.split("/", 3)
            ref_row = bt.get(pair, {}).get(model, {})
        ref_mape = ref_row.get("MAPE", {}).get("mean")
        ref_rmse = ref_row.get("RMSE", {}).get("mean")
        c_mape = row.get("MAPE", {}).get("mean")
        c_rmse = row.get("RMSE", {}).get("mean")
        lines.append(",".join([
            key, str(row["alignment_ok"]), str(row["n_batteries"]), str(row["n_repeats"]),
            f"{c_mape:.6f}" if c_mape is not None else "",
            f"{ref_mape:.6f}" if ref_mape is not None else "",
            f"{c_mape - ref_mape:+.6f}" if (c_mape is not None and ref_mape is not None) else "",
            f"{c_rmse:.6f}" if c_rmse is not None else "",
            f"{ref_rmse:.6f}" if ref_rmse is not None else "",
            f"{c_rmse - ref_rmse:+.6f}" if (c_rmse is not None and ref_rmse is not None) else "",
        ]))
    (out_dir / "comparison_labelgap_vs_cellid.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir / 'comparison_labelgap_vs_cellid.csv'} "
          f"({len(summary)} rows)")


if __name__ == "__main__":
    main()
