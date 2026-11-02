"""Aggregate results/reproduce into paper-like metric summaries.

Reads true_label.npy / pred_label.npy per Experiment and computes
battery-averaged MAE/MAPE/RMSE (gap split) mean+/-std over repeats,
matching results analysis/*.py conventions.

Usage:
    python scripts/evaluate.py --in results/reproduce --out evaluation
    python scripts/evaluate.py --in results/reproduce --phase regular --datasets XJTU HUST

Outputs:
    evaluation/summary.json         regular + small keys (battery-level MAE/
                                    MAPE/RMSE mean+/-std per condition)
    evaluation/summary_transfer.json (with --phase transfer)

Note: each invocation MERGES its keys into the target summary file and keeps
unrelated keys, so aggregating several results trees (reproduce, baselines,
ablations) into one --out directory is safe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GAPS = {"XJTU": 0.05, "TJU": 0.07, "HUST": 0.10, "MIT": 0.07}


def eval_metrix(true_label, pred_label):
    from sklearn import metrics as skm
    mae = float(skm.mean_absolute_error(true_label, pred_label))
    mape = float(skm.mean_absolute_percentage_error(true_label, pred_label))
    mse = float(skm.mean_squared_error(true_label, pred_label))
    rmse = float(np.sqrt(mse))
    return mae, mape, mse, rmse


def battery_metrics(true_label: np.ndarray, pred_label: np.ndarray, gap: float) -> Dict[str, float]:
    true_label = np.asarray(true_label).reshape(-1)
    pred_label = np.asarray(pred_label).reshape(-1)
    diff = np.diff(true_label)
    splits = np.where(diff > gap)[0]
    starts = [0] + (splits + 1).tolist()
    ends = splits.tolist() + [len(true_label) - 1]
    mapes, rmses, maes = [], [], []
    for s, e in zip(starts, ends):
        t = true_label[s:e+1]; p = pred_label[s:e+1]
        if len(t) < 2:
            continue
        mae, mape, _mse, rmse = eval_metrix(t, p)
        maes.append(mae); mapes.append(mape); rmses.append(rmse)
    if not mapes:
        mae, mape, _mse, rmse = eval_metrix(true_label, pred_label)
        return {"MAE": mae, "MAPE": mape, "RMSE": rmse}
    return {"MAE": float(np.mean(maes)), "MAPE": float(np.mean(mapes)), "RMSE": float(np.mean(rmses))}


def scan_regular(in_root: Path, model: str) -> Dict[str, List[Dict[str,float]]]:
    base = in_root / "regular" / model
    if not base.exists():
        return {}
    out: Dict[str, List[Dict[str,float]]] = {}
    for ds_dir in base.iterdir():
        if not ds_dir.is_dir():
            continue
        gap_key = ds_dir.name.split("-")[0]
        gap = GAPS.get(gap_key, 0.05)
        repeats=[]
        for exp in sorted(ds_dir.glob("Experiment*")):
            t = exp / "true_label.npy"; p = exp / "pred_label.npy"
            if not t.exists() or not p.exists():
                continue
            true = np.load(t); pred = np.load(p)
            repeats.append(battery_metrics(true, pred, gap))
        if repeats:
            out[ds_dir.name] = repeats
    return out


def scan_small(in_root: Path, model: str) -> Dict[str, List[Dict[str,float]]]:
    base = in_root / "small" / model
    if not base.exists():
        return {}
    out: Dict[str, List[Dict[str,float]]] = {}
    for case in base.iterdir():
        if not case.is_dir():
            continue
        # e.g. small-XJTU-2C-n1  or small-HUST-n2
        if "XJTU" in case.name:
            gap = GAPS["XJTU"]
        elif "HUST" in case.name:
            gap = GAPS["HUST"]
        else:
            gap = 0.05
        repeats=[]
        for exp in sorted(case.glob("Experiment*")):
            t = exp / "true_label.npy"; p = exp / "pred_label.npy"
            if not t.exists() or not p.exists():
                continue
            true = np.load(t); pred = np.load(p)
            repeats.append(battery_metrics(true, pred, gap))
        if repeats:
            out[case.name] = repeats
    return out


def scan_transfer(base: Path, model: str) -> Dict[str, List[Dict[str,float]]]:
    """Scan a transfer results root: base/{model}/{pair}/Experiment*.

    ``base`` is the transfer root itself (e.g. results/reproduce/transfer_v2).
    The gap threshold is chosen by the TARGET dataset, i.e. the part after
    "-to-" in the pair label (e.g. XJTU-2C-to-HUST -> HUST).
    """
    model_dir = base / model
    if not model_dir.exists():
        return {}
    pair_dirs = [d for d in model_dir.iterdir() if d.is_dir() and "-to-" in d.name]
    # If the producing script wrote completion markers anywhere in this tree,
    # require them, so a run killed mid-adaptation (source-train npy present,
    # fine-tune not finished) is skipped instead of silently aggregated.
    require_done = any((d / "finetune_done.txt").exists() for d in pair_dirs)
    out: Dict[str, List[Dict[str,float]]] = {}
    for pair in pair_dirs:
        tgt = pair.name.split("-to-")[-1]
        gap = GAPS.get(tgt.split("-")[0], 0.05)
        repeats = []
        for exp in sorted(pair.glob("Experiment*")):
            if require_done and not (exp / "finetune_done.txt").exists():
                continue
            t = exp / "true_label.npy"; p = exp / "pred_label.npy"
            if not t.exists() or not p.exists():
                continue
            repeats.append(battery_metrics(np.load(t), np.load(p), gap))
        if repeats:
            out[pair.name] = repeats
    return out


def summarize(values: List[Dict[str,float]]) -> Dict[str, Tuple[float,float]]:
    keys = ["MAE","MAPE","RMSE"]
    out={}
    for k in keys:
        arr = np.array([v[k] for v in values], dtype=float)
        out[k] = (float(arr.mean()), float(arr.std(ddof=1) if len(arr)>1 else 0.0))
    return out


def merge_write_json(path: Path, updates: Dict) -> None:
    """Merge ``updates`` into an existing JSON dict file, keeping other keys.

    Aggregating a second results tree (e.g. baselines after reproduce) into
    the same --out directory must not drop the first tree's keys.
    """
    existing: Dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(updates)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_root", default="results/reproduce")
    ap.add_argument("--out", dest="out_root", default="evaluation")
    ap.add_argument("--phase", choices=["regular","small","transfer","all"], default="all")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None,
                    help="model directory names to aggregate (default: orig adaptive; "
                         "add LSTM GRU CNN MLP GPR for the baseline suite)")
    args = ap.parse_args()

    in_root = (PROJECT_ROOT / args.in_root) if not Path(args.in_root).is_absolute() else Path(args.in_root)
    out_root = (PROJECT_ROOT / args.out_root) if not Path(args.out_root).is_absolute() else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.phase == "transfer":
        # in_root is the transfer results root itself (contains orig/ + adaptive/)
        summary = {}
        for m in ("orig", "adaptive"):
            data = scan_transfer(in_root, m)
            print(f"\n[transfer:{m}]")
            for label, repeats in sorted(data.items()):
                s = summarize(repeats)
                summary[f"transfer/{m}/{label}"] = {k: {"mean": v[0], "std": v[1], "n": len(repeats)} for k, v in s.items()}
                print(f"  {label:24s}  n={len(repeats)}  MAPE {s['MAPE'][0]:.4f} +/- {s['MAPE'][1]:.4f}  RMSE {s['RMSE'][0]:.4f} +/- {s['RMSE'][1]:.4f}")
        if all((in_root / m).exists() for m in ("orig", "adaptive")):
            print("\n[transfer compare]  delta = adaptive - orig (negative = adaptive better)")
            labs = sorted(set(scan_transfer(in_root, "orig")) | set(scan_transfer(in_root, "adaptive")))
            for label in labs:
                ro = scan_transfer(in_root, "orig").get(label, [])
                ra = scan_transfer(in_root, "adaptive").get(label, [])
                if not ro or not ra:
                    continue
                so = summarize(ro); sa = summarize(ra)
                d_mape = sa["MAPE"][0] - so["MAPE"][0]
                d_rmse = sa["RMSE"][0] - so["RMSE"][0]
                winner = "adaptive" if d_mape < 0 and d_rmse < 0 else ("orig" if d_mape > 0 and d_rmse > 0 else "mixed")
                print(f"  {label:24s}  dMAPE {d_mape:+.4f}  dRMSE {d_rmse:+.4f}  -> {winner}")
        merge_write_json(out_root / "summary_transfer.json", summary)
        print(f"\nwrote {out_root / 'summary_transfer.json'}")
        return

    allowed = args.models if args.models else ["orig", "adaptive"]
    models = [d.name for d in (in_root / "regular").iterdir() if d.is_dir()] if (in_root/"regular").exists() else []
    if not models:
        models = [d.name for d in in_root.iterdir() if d.is_dir()]
        # fallback: small
        if not models and (in_root/"small").exists():
            models = [d.name for d in (in_root/"small").iterdir() if d.is_dir()]
    models = [m for m in models if m in allowed]
    if not models:
        # also check orig/adaptive under regular/small explicitly
        for cand in ("orig","adaptive"):
            if (in_root/"regular"/cand).exists() or (in_root/"small"/cand).exists():
                models.append(cand)
    print(f"models: {models}")
    print(f"in: {in_root}  out: {out_root}")

    summary={}
    # regular
    if args.phase in ("regular","all"):
        for m in models:
            data = scan_regular(in_root, m)
            if args.datasets:
                data = {k:v for k,v in data.items() if any(d in k for d in args.datasets)}
            print(f"\n[regular:{m}]")
            rows=[]
            for label, repeats in sorted(data.items()):
                s = summarize(repeats)
                summary[f"regular/{m}/{label}"] = {k: {"mean": v[0], "std": v[1], "n": len(repeats)} for k,v in s.items()}
                print(f"  {label:18s}  n={len(repeats):2d}  MAPE {s['MAPE'][0]:.4f} +/- {s['MAPE'][1]:.4f}  RMSE {s['RMSE'][0]:.4f} +/- {s['RMSE'][1]:.4f}")
                rows.append((label, len(repeats), s))
            if rows and m == models[0]:
                # placeholder for compare after both models
                pass
        # paired compare orig vs adaptive per condition
        if len(models) == 2:
            print("\n[regular compare orig vs adaptive]  delta = adaptive - orig (negative = adaptive better)")
            # collect union labels
            all_labels = sorted(set(scan_regular(in_root,"orig").keys()) | set(scan_regular(in_root,"adaptive").keys()))
            for label in all_labels:
                ro = scan_regular(in_root,"orig").get(label, [])
                ra = scan_regular(in_root,"adaptive").get(label, [])
                if not ro or not ra:
                    continue
                so = summarize(ro); sa = summarize(ra)
                d_mape = sa["MAPE"][0] - so["MAPE"][0]
                d_rmse = sa["RMSE"][0] - so["RMSE"][0]
                winner = "adaptive" if d_mape < 0 and d_rmse < 0 else ("orig" if d_mape > 0 and d_rmse > 0 else "mixed")
                print(f"  {label:18s}  dMAPE {d_mape:+.4f}  dRMSE {d_rmse:+.4f}  -> {winner}")

    if args.phase in ("small","all"):
        for m in models:
            data = scan_small(in_root, m)
            if not data:
                continue
            print(f"\n[small:{m}]")
            for label, repeats in sorted(data.items()):
                s = summarize(repeats)
                summary[f"small/{m}/{label}"] = {k: {"mean": v[0], "std": v[1], "n": len(repeats)} for k,v in s.items()}
                print(f"  {label:22s}  n={len(repeats):2d}  MAPE {s['MAPE'][0]:.4f} +/- {s['MAPE'][1]:.4f}  RMSE {s['RMSE'][0]:.4f} +/- {s['RMSE'][1]:.4f}")
        if len(models) == 2:
            print("\n[small compare]")
            labs = sorted(set(scan_small(in_root,"orig").keys()) | set(scan_small(in_root,"adaptive").keys()))
            for label in labs:
                ro = scan_small(in_root,"orig").get(label, [])
                ra = scan_small(in_root,"adaptive").get(label, [])
                if not ro or not ra:
                    continue
                so = summarize(ro); sa = summarize(ra)
                print(f"  {label:22s}  dMAPE {sa['MAPE'][0]-so['MAPE'][0]:+.4f}  dRMSE {sa['RMSE'][0]-so['RMSE'][0]:+.4f}")

    merge_write_json(out_root / "summary.json", summary)
    print(f"\nwrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
