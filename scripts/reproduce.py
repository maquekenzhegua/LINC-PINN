"""One-click reproduction of the AdaptiveInvariantPINN experiment suite.

Covers the regular SOH, small-sample and cross-dataset transfer suites of
the CMAME paper, running both the original PINN (Model.Model.PINN) and the
ported AdaptiveInvariantPINN via the same dataloader splits.

Usage (from the repository root):
    python scripts/reproduce.py --help
    python scripts/reproduce.py --phase regular --model both --repeats 10
    python scripts/reproduce.py --phase small --model both --repeats 3
    python scripts/reproduce.py --phase transfer --model both --repeats 3

Each phase is resumable: existing Experiment*/{true,pred}_label.npy skip retraining.
Results are written under results/reproduce/{phase}/{model}/... so the
hand-graded results/AdaptiveInvariantPINN and results/Ours trees are untouched.

Metrics are aggregated battery-wise from the saved npy pairs with the gap
thresholds XJTU 0.05 / HUST 0.10 / TJU 0.07 / MIT 0.07. A battery's cycle
sequence is split wherever the true label jumps up by more than the gap
(capacity regeneration); the boundary row belongs to the preceding segment
and 1-row segments are skipped, then per-battery metrics are averaged over
segments and over repeats (mean +/- std, ddof=1).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader.dataloader import HUSTdata, MITdata, TJUdata, XJTUdata  # noqa: E402

XJTU_BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
TJU_BATCHES = [0, 1, 2]
GAPS = {"XJTU": 0.05, "TJU": 0.07, "HUST": 0.10, "MIT": 0.07}


def eval_metrix(true_label, pred_label):
    from sklearn import metrics as skm

    mae = float(skm.mean_absolute_error(true_label, pred_label))
    mape = float(skm.mean_absolute_percentage_error(true_label, pred_label))
    mse = float(skm.mean_squared_error(true_label, pred_label))
    rmse = float(np.sqrt(mse))
    return mae, mape, mse, rmse


def battery_metrics(true_label: np.ndarray, pred_label: np.ndarray, gap: float) -> Dict[str, float]:
    """Per-battery MAPE/RMSE aggregated over concatenated test predictions.

    Mirrors results analysis/*.py: split on np.diff(true_label) > gap.
    """
    true_label = np.asarray(true_label).reshape(-1)
    pred_label = np.asarray(pred_label).reshape(-1)
    diff = np.diff(true_label)
    splits = np.where(diff > gap)[0]
    # segments: [0..s0], [s0+1..s1], ...
    starts = [0] + (splits + 1).tolist()
    ends = splits.tolist() + [len(true_label) - 1]
    mapes, rmses, maes = [], [], []
    for s, e in zip(starts, ends):
        t = true_label[s : e + 1]
        p = pred_label[s : e + 1]
        if len(t) < 2:
            continue
        mae, mape, _mse, rmse = eval_metrix(t, p)
        maes.append(mae)
        mapes.append(mape)
        rmses.append(rmse)
    if not mapes:
        mae, mape, _mse, rmse = eval_metrix(true_label, pred_label)
        return {"MAE": mae, "MAPE": mape, "RMSE": rmse}
    return {"MAE": float(np.mean(maes)), "MAPE": float(np.mean(mapes)), "RMSE": float(np.mean(rmses))}


# ---------------------------------------------------------------------------
# Data helpers mirroring main_*.py
# ---------------------------------------------------------------------------

def make_loaders_for_phase(dataset: str, batch: Optional[str | int], args, small_sample: Optional[int] = None) -> Dict[str, object]:
    """Return {'train','valid','test'} DataLoaders matching the paper splits.

    ``small_sample`` truncates the target train battery list (used by the
    transfer fine-tuning protocol, which adapts on 1 target battery).
    """
    if dataset == "XJTU":
        assert batch in XJTU_BATCHES
        root = "data/XJTU data"
        data = XJTUdata(root=root, args=args)
        train_list: List[str] = []
        test_list: List[str] = []
        for f in sorted(os.listdir(root)):
            if batch in f:
                p = os.path.join(root, f)
                if "4" in f or "8" in f:
                    test_list.append(p)
                else:
                    train_list.append(p)
        if small_sample is not None:
            train_list = train_list[:small_sample]
        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        return {"train": train_loader["train_2"], "valid": train_loader["valid_2"], "test": test_loader["test_3"]}
    if dataset == "TJU":
        assert batch in TJU_BATCHES
        root = "data/TJU data"
        data = TJUdata(root=root, args=args)
        batchs = sorted(os.listdir(root))
        batch_name = batchs[batch]  # 0..2 maps to Dataset_1/2/3
        batch_root = os.path.join(root, batch_name)
        mod = [(5, 9), (4, 8), (5, 9)]
        train_list, test_list = [], []
        for i, f in enumerate(sorted(os.listdir(batch_root))):
            cur = os.path.join(batch_root, f)
            bid = i + 1
            if bid % 10 == mod[batch][0] or bid % 10 == mod[batch][1]:
                test_list.append(cur)
            else:
                train_list.append(cur)
        if small_sample is not None:
            train_list = train_list[:small_sample]
        train_loader = data.read_all(specific_path_list=train_list)
        test_loader = data.read_all(specific_path_list=test_list)
        return {"train": train_loader["train_2"], "valid": train_loader["valid_2"], "test": test_loader["test_3"]}
    if dataset == "HUST":
        test_id = [
            "1-4", "1-8", "2-4", "2-8", "3-4", "3-8", "4-4", "4-8",
            "5-4", "5-7", "6-4", "6-8", "7-4", "7-8", "8-4", "8-8",
            "9-4", "9-8", "10-4", "10-8",
        ]
        data = HUSTdata(root="data/HUST data", args=args)
        train_list2: List[str] = []
        test_list2: List[str] = []
        for f in sorted(os.listdir("data/HUST data")):
            if f[:-4] in test_id:
                test_list2.append(f"data/HUST data/{f}")
            else:
                train_list2.append(f"data/HUST data/{f}")
        if small_sample is not None:
            train_list2 = train_list2[:small_sample]
        tl = data.read_all(specific_path_list=train_list2)
        te = data.read_all(specific_path_list=test_list2)
        return {"train": tl["train_2"], "valid": tl["valid_2"], "test": te["test_3"]}
    if dataset == "MIT":
        root = "data/MIT data"
        train_list3: List[str] = []
        test_list3: List[str] = []
        for b in ["2017-05-12", "2017-06-30", "2018-04-12"]:
            br = os.path.join(root, b)
            for f in sorted(os.listdir(br)):
                cur = os.path.join(br, f)
                cid = int(f.split("-")[-1].split(".")[0])
                if cid % 5 == 0:
                    test_list3.append(cur)
                else:
                    train_list3.append(cur)
        data = MITdata(root=root, args=args)
        tl = data.read_all(specific_path_list=train_list3)
        te = data.read_all(specific_path_list=test_list3)
        return {"train": tl["train_2"], "valid": tl["valid_2"], "test": te["test_3"]}
    raise ValueError(dataset)


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def _monkey_patch_torch_save_for_unicode_path():
    """PyTorch 2.0 zip serialization fails on paths with non-ascii / parentheses.

    Patch torch.save to fall back to legacy serialization when the zip path
    trigger fires, so runs under non-ASCII project roots do not crash on model.pth.
    """
    import torch as _torch

    orig = _torch.save

    def patched(obj, f, *a, **kw):
        try:
            return orig(obj, f, *a, **kw)
        except RuntimeError as e:
            if "Parent directory" in str(e) and "does not exist" in str(e):
                # path exists but zip writer mis-parsed unicode; use legacy
                kw.setdefault("_use_new_zipfile_serialization", False)
                return orig(obj, f, *a, **kw)
            raise

    _torch.save = patched


_monkey_patch_torch_save_for_unicode_path()


def run_original_pinn(dataloaders: Dict[str, object], save_folder: Path, args) -> Tuple[np.ndarray, np.ndarray]:
    """Train original PINN and return (true, pred) from its saved npy."""
    from Model.Model import PINN

    # PINN's Train saves best checkpoint + npy under save_folder
    args.save_folder = str(save_folder)
    args.log_dir = "logging.txt"
    pinn = PINN(args)
    pinn.Train(trainloader=dataloaders["train"], validloader=dataloaders["valid"], testloader=dataloaders["test"])
    # Pinn writes true_label.npy / pred_label.npy
    true = np.load(save_folder / "true_label.npy")
    pred = np.load(save_folder / "pred_label.npy")
    return true, pred


def run_adaptive_pinn(dataloaders: Dict[str, object], save_folder: Path, args) -> Tuple[np.ndarray, np.ndarray]:
    from Model.adaptive_trainer import AdaptivePINNTrainer

    args.save_folder = str(save_folder)
    args.log_dir = "logging.txt"
    trainer = AdaptivePINNTrainer.from_args(args)
    trainer.Train(trainloader=dataloaders["train"], validloader=dataloaders["valid"], testloader=dataloaders["test"])
    true = np.load(save_folder / "true_label.npy")
    pred = np.load(save_folder / "pred_label.npy")
    return true, pred


def build_orig_args(dataset: str, batch: Optional[str | int]) -> argparse.Namespace:
    """Hyper-parameters matching the upstream PINN4SOH per-dataset defaults."""
    base = dict(
        data=dataset,
        batch_size=256, normalization_method="min-max",
        epochs=200, early_stop=20,
        warmup_epochs=30, warmup_lr=0.002, lr=0.01, final_lr=0.0002, lr_F=0.001,
        F_layers_num=3, F_hidden_dim=60,
        log_dir="logging.txt", save_folder=None,
    )
    if dataset == "XJTU":
        base.update(dict(train_batch=0, test_batch=1, batch=batch, alpha=0.7, beta=0.2))
    elif dataset == "TJU":
        base.update(dict(in_same_batch=True, train_batch=-1, test_batch=-1, batch=batch, alpha=1.0, beta=0.05))
    elif dataset == "HUST":
        base.update(dict(alpha=0.5, beta=0.2, lr=0.01, final_lr=0.0002, lr_F=0.0005))
    elif dataset == "MIT":
        base.update(dict(early_stop=10, alpha=1.0, beta=0.02, u_layers_num=3, u_hidden_dim=60))
    # Add missing attrs expected by Model.Model
    base.setdefault("batch", batch)
    return argparse.Namespace(**base)


def build_adaptive_args(dataset: str, batch: Optional[str | int], latent_dim: Optional[int] = None,
                        feature_scaler: str = "identity", include_row_mean: bool = True,
                        small_gamma: Optional[float] = None, include_anchored: bool = True,
                        operator_weight: Optional[float] = None) -> argparse.Namespace:
    """Hyper-parameters for AdaptiveInvariantPINN.

    Final model configuration: 8-dim latent bottleneck, self-normalizing
    invariant transform without a source-fitted scaler, and the row-mean
    coordinate that restores the common-mode aging level.  ``gamma``
    (prediction-side monotonicity weight) mirrors the original per-dataset
    ``beta`` — including for small-sample XJTU, where the original beta=10
    collapses the adaptive latent representation (the operator loss already
    enforces smooth degradation, so an over-weighted monotone penalty rewards
    flat predictions instead).
    """
    base = dict(
        data=dataset,
        batch_size=256, normalization_method="min-max",
        epochs=200, early_stop=20,
        warmup_epochs=30, warmup_lr=0.002, lr=0.01, final_lr=0.0002, lr_F=0.001,
        F_layers_num=3, F_hidden_dim=60,
        input_dim=17, latent_dim=latent_dim if latent_dim is not None else 8,
        u_hidden_dim=60, invariant_epsilon=1e-6,
        feature_scaler=feature_scaler,
        include_row_mean=include_row_mean,
        include_anchored=include_anchored,
        alpha=0.5 if operator_weight is None else operator_weight,
        gamma={"XJTU": 0.2, "TJU": 0.05, "HUST": 0.2, "MIT": 0.02}[dataset],
        log_dir="logging.txt", save_folder=None,
    )
    if small_gamma is not None:
        base["gamma"] = small_gamma
    if dataset == "XJTU":
        base.update(dict(train_batch=0, test_batch=1, batch=batch))
    elif dataset == "TJU":
        base.update(dict(in_same_batch=True, train_batch=-1, test_batch=-1, batch=batch))
    elif dataset == "HUST":
        base.update(dict(lr_F=0.0005))
    elif dataset == "MIT":
        base.update(dict(early_stop=10, u_layers_num=3))
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def phase_regular(repeats: int, models: List[str], datasets: Optional[List[str]], out_root: Path,
                  latent_dim: Optional[int] = None, batches: Optional[List[str]] = None,
                  feature_scaler: str = "source", include_row_mean: bool = False,
                  include_anchored: bool = True, operator_weight: Optional[float] = None,
                  seed_base: Optional[int] = None, gamma_override: Optional[float] = None):
    """Table 2: regular SOH per-batch/per-dataset, 10 repeats."""
    plan: List[Tuple[str, Optional[str | int]]] = []
    if datasets is None or "XJTU" in datasets:
        for b in XJTU_BATCHES:
            plan.append(("XJTU", b))
    if datasets is None or "TJU" in datasets:
        for b in TJU_BATCHES:
            plan.append(("TJU", b))
    if datasets is None or "HUST" in datasets:
        plan.append(("HUST", None))
    if datasets is None or "MIT" in datasets:
        plan.append(("MIT", None))
    if batches is not None:
        wanted = {str(b) for b in batches}
        plan = [(d, b) for d, b in plan if b is not None and str(b) in wanted]

    for model in models:
        for dataset, batch in plan:
            label = f"{dataset}-{batch}" if batch is not None else dataset
            print(f"\n[regular:{model}] {label}  repeats={repeats}")
            for exp in range(1, repeats + 1):
                exp_dir = out_root / "regular" / model / label / f"Experiment{exp}"
                if (exp_dir / "true_label.npy").exists() and (exp_dir / "pred_label.npy").exists():
                    print(f"  skip Experiment{exp} (exists)")
                    continue
                ensure_dir(exp_dir)
                t0 = time.time()
                if seed_base is not None:
                    _set_seed(seed_base + exp)
                if model == "orig":
                    args = build_orig_args(dataset, batch)
                    # TJU/HUST/MIT orig defaults used 512 batch; keep but cap if OOM risk
                    if dataset in ("TJU", "HUST", "MIT"):
                        args.batch_size = 256  # cap for 4060 8GB determinism
                    loaders = make_loaders_for_phase(dataset, batch, args)
                    run_original_pinn(loaders, exp_dir, args)
                else:
                    args = build_adaptive_args(dataset, batch, latent_dim=latent_dim,
                                               feature_scaler=feature_scaler,
                                               include_row_mean=include_row_mean,
                                               include_anchored=include_anchored,
                                               operator_weight=operator_weight,
                                               small_gamma=gamma_override)
                    if dataset in ("TJU", "HUST", "MIT"):
                        args.batch_size = 256
                    loaders = make_loaders_for_phase(dataset, batch, args)
                    run_adaptive_pinn(loaders, exp_dir, args)
                print(f"  done Experiment{exp} in {time.time()-t0:.1f}s")


def phase_small(repeats: int, models: List[str], out_root: Path,
                datasets: Optional[List[str]] = None, latent_dim: Optional[int] = None,
                feature_scaler: str = "source", include_row_mean: bool = False,
                small_gamma: Optional[float] = None, include_anchored: bool = True,
                operator_weight: Optional[float] = None, seed_base: Optional[int] = None,
                combo: int = 0):
    """Table 3 / Fig.5: small-sample (1-4 train batteries).

    - XJTU: only batch 2C (paper main) with n=1..4
    - HUST: n=1..4
    Supplementary TJU/MIT 2-battery cases are covered by small_sample_2 below.
    """
    cases: List[Tuple[str, Optional[str | int], int]] = []
    for n in [1, 2, 3, 4]:
        if datasets is None or "XJTU" in datasets:
            cases.append(("XJTU", "2C", n))
        if datasets is None or "HUST" in datasets:
            cases.append(("HUST", None, n))
    for model in models:
        for dataset, batch, n in cases:
            label = f"small-{dataset}-{batch}-n{n}" if dataset == "XJTU" else f"small-{dataset}-n{n}"
            print(f"\n[small:{model}] {label}  repeats={repeats}")
            for exp in range(1, repeats + 1):
                exp_dir = out_root / "small" / model / label / f"Experiment{exp}"
                if (exp_dir / "true_label.npy").exists() and (exp_dir / "pred_label.npy").exists():
                    print(f"  skip Experiment{exp}")
                    continue
                ensure_dir(exp_dir)
                if seed_base is not None:
                    _set_seed(seed_base + exp)
                if model == "orig":
                    args = build_orig_args(dataset, batch)
                    args.batch_size = 128
                    if dataset == "XJTU":
                        args.alpha = 0.5; args.beta = 10  # orig small_sample override
                    # Build with truncated train_list
                    loaders = make_loaders_for_small(dataset, batch, n, args, rotate=combo)
                    run_original_pinn(loaders, exp_dir, args)
                else:
                    args = build_adaptive_args(dataset, batch, latent_dim=latent_dim,
                                               feature_scaler=feature_scaler,
                                               include_row_mean=include_row_mean,
                                               small_gamma=small_gamma,
                                               include_anchored=include_anchored,
                                               operator_weight=operator_weight)
                    args.batch_size = 128
                    loaders = make_loaders_for_small(dataset, batch, n, args, rotate=combo)
                    run_adaptive_pinn(loaders, exp_dir, args)
                print(f"  done Experiment{exp}")


def make_loaders_for_small(dataset: str, batch: Optional[str | int], n: int, args,
                           rotate: int = 0):
    """Same as make_loaders_for_phase but train_list[:n].

    ``rotate`` cycles the sorted train battery list before truncation
    (rotate=0 keeps the paper's first-n selection; rotate>0 picks a
    deterministic alternative battery subset for multi-combo controls).
    """
    if dataset == "XJTU":
        root = "data/XJTU data"
        data = XJTUdata(root=root, args=args)
        train_list, test_list = [], []
        for f in sorted(os.listdir(root)):
            if batch in f:
                p = os.path.join(root, f)
                if "4" in f or "8" in f:
                    test_list.append(p)
                else:
                    train_list.append(p)
        if rotate:
            k = rotate % len(train_list)
            train_list = train_list[k:] + train_list[:k]
        train_list = train_list[:n]
        tl = data.read_all(specific_path_list=train_list)
        te = data.read_all(specific_path_list=test_list)
        return {"train": tl["train_2"], "valid": tl["valid_2"], "test": te["test_3"]}
    if dataset == "HUST":
        test_id = [
            "1-4", "1-8", "2-4", "2-8", "3-4", "3-8", "4-4", "4-8",
            "5-4", "5-7", "6-4", "6-8", "7-4", "7-8", "8-4", "8-8",
            "9-4", "9-8", "10-4", "10-8",
        ]
        data = HUSTdata(root="data/HUST data", args=args)
        train_list2, test_list2 = [], []
        for f in sorted(os.listdir("data/HUST data")):
            if f[:-4] in test_id:
                test_list2.append(f"data/HUST data/{f}")
            else:
                train_list2.append(f"data/HUST data/{f}")
        if rotate:
            k = rotate % len(train_list2)
            train_list2 = train_list2[k:] + train_list2[:k]
        train_list2 = train_list2[:n]
        tl = data.read_all(specific_path_list=train_list2)
        te = data.read_all(specific_path_list=test_list2)
        return {"train": tl["train_2"], "valid": tl["valid_2"], "test": te["test_3"]}
    raise ValueError(dataset)


PHASES = ["regular", "small", "transfer"]


def main():
    p = argparse.ArgumentParser(description="Reproduce PINN4SOH paper experiments")
    p.add_argument("--phase", choices=PHASES + ["all"], default="regular")
    p.add_argument("--model", choices=["orig", "adaptive", "both"], default="both")
    p.add_argument("--repeats", type=int, default=10, help="repeats per condition (paper: 10)")
    p.add_argument("--datasets", nargs="*", default=None, choices=["XJTU", "TJU", "HUST", "MIT"],
                   help="filter for regular/small phases")
    p.add_argument("--latent_dim", type=int, default=None,
                   help="override AdaptiveInvariantPINN latent_dim (default 8)")
    p.add_argument("--feature_scaler", choices=["source", "identity"], default="identity",
                   help="identity (default): use the self-normalizing row transform as-is; "
                        "source: z-score transformed features with moments fitted on "
                        "training rows")
    p.add_argument("--include_row_mean", action=argparse.BooleanOptionalAction, default=True,
                   help="append the row mean of the raw features as one extra "
                        "coordinate (restores the common-mode aging level)")
    p.add_argument("--small_gamma", type=float, default=None,
                   help="override the small-sample monotonicity weight (default: "
                        "the per-dataset gamma, e.g. 0.2 for XJTU)")
    p.add_argument("--include_anchored", action=argparse.BooleanOptionalAction, default=True,
                   help="keep the anchored coordinate channel (each feature's "
                        "offset from the first feature of the same row) of the "
                        "invariant feature map")
    p.add_argument("--operator_weight", type=float, default=None,
                   help="override the shared-operator loss weight (default 0.5; "
                        "0 ablates the operator/PDE loss entirely)")
    p.add_argument("--seed_base", type=int, default=None,
                   help="when set, seed every run with seed_base + repeat index "
                        "(default: off, matching the unseeded paper protocol)")
    p.add_argument("--gamma_override", type=float, default=None,
                   help="regular phase: override the per-dataset monotonicity "
                        "weight gamma for ALL datasets (e.g. 0 ablates the "
                        "direction penalty)")
    p.add_argument("--combos", type=int, default=None,
                   help="small phase: run this many deterministic battery-subset "
                        "rotations (combo k rotates the sorted train list by k); "
                        "outputs go to <out>/combo{k}, combo 0 = paper selection")
    p.add_argument("--batches", nargs="*", default=None,
                   help="regular phase only: run only these batches, e.g. 2C R2.5 0 2")
    p.add_argument("--out", type=str, default="results/reproduce",
                   help="output root (relative to project)")
    args = p.parse_args()

    models = ["orig", "adaptive"] if args.model == "both" else [args.model]
    out_root = PROJECT_ROOT / args.out
    ensure_dir(out_root)
    try:
        from provenance import record_provenance
    except ImportError:
        from scripts.provenance import record_provenance
    record_provenance(out_root, args)

    phases = PHASES if args.phase == "all" else [args.phase]
    for ph in phases:
        if ph == "regular":
            phase_regular(args.repeats, models, args.datasets, out_root,
                          latent_dim=args.latent_dim, batches=args.batches,
                          feature_scaler=args.feature_scaler,
                          include_row_mean=args.include_row_mean,
                          include_anchored=args.include_anchored,
                          operator_weight=args.operator_weight,
                          seed_base=args.seed_base,
                          gamma_override=args.gamma_override)
        elif ph == "small":
            combos = range(args.combos) if args.combos else [0]
            for combo in combos:
                combo_out = out_root / f"combo{combo}" if args.combos else out_root
                phase_small(args.repeats, models, combo_out, datasets=args.datasets,
                            latent_dim=args.latent_dim, feature_scaler=args.feature_scaler,
                            include_row_mean=args.include_row_mean,
                            small_gamma=args.small_gamma,
                            include_anchored=args.include_anchored,
                            operator_weight=args.operator_weight,
                            seed_base=args.seed_base, combo=combo)
        elif ph == "transfer":
            print("transfer phase: use main_adaptation - fine-tuning.py (orig AdaPINN) "
                  "and main_adaptive_transfer.py (adaptive). "
                  "Run: scripts/reproduce_transfer.py --repeats 10")
        else:
            raise ValueError(ph)

    # Write manifest
    manifest = dict(phase=args.phase, model=args.model, repeats=args.repeats,
                    datasets=args.datasets, out=str(out_root))
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Results under {out_root}")


if __name__ == "__main__":
    main()
