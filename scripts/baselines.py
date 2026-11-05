"""External baselines under the identical data splits and training protocol.

Baselines
---------
- MLP, CNN : the original paper's data-driven comparators (Model/Compare_Models.py).
- LSTM, GRU: 2-layer recurrent nets reading the 17 input coordinates as a
  length-17 sequence (hidden 32, inter-layer dropout 0.2), head Dropout(0.2)
  + Linear(32, 1) - same capacity scale as the original comparators.
- XGBoost  : xgboost.XGBRegressor on the flat 17-dim row, the same input
  contract as the 2026-10-08 study.  Each training battery is split
  chronologically 80/20 into train and validation and the test cells are held
  out whole.  ``max_depth`` in {2,3,4} x ``learning_rate`` in {0.03,0.07} are
  selected on validation MSE with early stopping, after which the selected
  number of boosting rounds is refit on the full training block.  In transfer
  mode the source booster is continued on the target calibration battery.

Protocol (identical to scripts/reproduce.py)
--------------------------------------------
Same battery splits / dataloader / min-max normalization, batch 256,
Adam(warmup_lr=2e-3) with the shared warmup->cosine LR schedule
(2e-3 -> 1e-2 -> 2e-4, warmup 30), 200 epochs, early-stop patience 20
(10 for MIT, as in build_orig_args) on validation MSE, and the
best-validation checkpoint's test predictions saved as
``true_label.npy`` / ``pred_label.npy`` so scripts/evaluate.py can
aggregate battery-level MAPE/RMSE exactly like for orig/adaptive.

The XGBoost arm keeps the same battery split and the same target-row
convention but trains on a per-battery chronological 80/20 inner split with
its own early stopping, because a gradient-boosted tree ensemble has no epoch
schedule to share.  This is the protocol behind the shipped
``results/baselines`` XGBOOST tree; see Section 4.3 of the README.

Usage (from the repository root, so the relative data/ paths resolve):
    python scripts/baselines.py --phase small --models all --repeats 10 --out results/baselines
    python scripts/baselines.py --phase regular --models all --repeats 10 --out results/baselines
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reproduce import make_loaders_for_phase, make_loaders_for_small  # noqa: E402

XJTU_BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
TJU_BATCHES = [0, 1, 2]
MODEL_NAMES = ["LSTM", "GRU", "CNN", "MLP", "XGBOOST"]

GAP_BY_PREFIX = {"XJTU": 0.05, "TJU": 0.07, "HUST": 0.10, "MIT": 0.07}


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Torch baselines
# ---------------------------------------------------------------------------

def build_torch_model(name: str):
    import torch
    from torch import nn

    from Model.Compare_Models import CNN, MLP

    if name == "MLP":
        return MLP()
    if name == "CNN":
        return CNN()
    if name in ("LSTM", "GRU"):

        class SeqBaseline(nn.Module):
            def __init__(self, cell: str) -> None:
                super().__init__()
                rnn_cls = nn.LSTM if cell == "LSTM" else nn.GRU
                self.rnn = rnn_cls(
                    input_size=1, hidden_size=32, num_layers=2, batch_first=True,
                    dropout=0.2,
                )
                self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32, 1))

            def forward(self, x):
                out, _ = self.rnn(x.unsqueeze(-1))
                return self.head(out[:, -1, :])

        return SeqBaseline(name)
    raise ValueError(name)


def _predict_all(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for x1, _x2, y1, _y2 in loader:
            preds.append(model(x1.to(device)).cpu().numpy())
            trues.append(y1.numpy())
    return np.concatenate(trues).reshape(-1), np.concatenate(preds).reshape(-1, 1)


def run_torch_baseline(name: str, loaders: Dict[str, object], save_dir: Path,
                       dataset: str, epochs: int = 200,
                       ft_loaders: Optional[Dict[str, object]] = None) -> None:
    """Train on loaders (train/valid/test).  In transfer mode (ft_loaders given)
    ``loaders['test']`` must be the target test set: the best-validation
    checkpoint is first evaluated zero-shot (saved as ``src_only_*.npy``),
    then fine-tuned on the 1-battery target loaders (Adam 1e-3, 200 epochs,
    early-stop 10) and the fine-tuned predictions are saved as the primary
    ``true_label.npy`` / ``pred_label.npy``."""
    import copy

    import torch
    from torch import nn

    from Model.adaptive_trainer import LR_Scheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(name).to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    scheduler = LR_Scheduler(
        optimizer=optimizer, warmup_epochs=30, warmup_lr=2e-3,
        num_epochs=epochs, base_lr=1e-2, final_lr=2e-4,
    )
    early_stop = 10 if dataset == "MIT" else 20

    best_mse = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x1, _x2, y1, _y2 in loaders["train"]:
            loss = loss_func(model(x1.to(device)), y1.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        true_v, pred_v = _predict_all(model, loaders["valid"], device)
        valid_mse = float(np.mean((true_v - pred_v.reshape(-1)) ** 2))
        stale += 1
        if valid_mse < best_mse:
            best_mse = valid_mse
            stale = 0
            best_state = copy.deepcopy(model.state_dict())
            if ft_loaders is None:
                true_t, pred_t = _predict_all(model, loaders["test"], device)
                np.save(save_dir / "true_label.npy", true_t)
                np.save(save_dir / "pred_label.npy", pred_t)
        if stale > early_stop:
            break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    if ft_loaders is None:
        # last-resort: guarantee files even if valid never improved past inf
        if not (save_dir / "true_label.npy").exists():
            true_t, pred_t = _predict_all(model, loaders["test"], device)
            np.save(save_dir / "true_label.npy", true_t)
            np.save(save_dir / "pred_label.npy", pred_t)
        return

    # ---- transfer mode ----
    model.load_state_dict(best_state)
    true_t, pred_t = _predict_all(model, loaders["test"], device)
    np.save(save_dir / "src_only_true_label.npy", true_t)
    np.save(save_dir / "src_only_pred_label.npy", pred_t)

    ft_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ft_epochs, ft_stop = 200, 10
    best_ft, ft_stale = float("inf"), 0
    for epoch in range(1, ft_epochs + 1):
        model.train()
        for x1, _x2, y1, _y2 in ft_loaders["train"]:
            loss = loss_func(model(x1.to(device)), y1.to(device))
            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
        true_v, pred_v = _predict_all(model, ft_loaders["valid"], device)
        ft_mse = float(np.mean((true_v - pred_v.reshape(-1)) ** 2))
        ft_stale += 1
        if ft_mse < best_ft:
            best_ft, ft_stale = ft_mse, 0
            true_t, pred_t = _predict_all(model, loaders["test"], device)
            np.save(save_dir / "true_label.npy", true_t)
            np.save(save_dir / "pred_label.npy", pred_t)
        if ft_stale > ft_stop:
            break
    if not (save_dir / "true_label.npy").exists():
        true_t, pred_t = _predict_all(model, loaders["test"], device)
        np.save(save_dir / "true_label.npy", true_t)
        np.save(save_dir / "pred_label.npy", pred_t)


# ---------------------------------------------------------------------------
# XGBoost baseline (flat 17-dim row, no sequence window)
# ---------------------------------------------------------------------------

XGB_FEATURE_COUNT = 17
XGB_TRAIN_FRACTION = 0.8
XGB_HUST_TEST_IDS = {
    "1-4", "1-8", "2-4", "2-8", "3-4", "3-8", "4-4", "4-8",
    "5-4", "5-7", "6-4", "6-8", "7-4", "7-8", "8-4", "8-8",
    "9-4", "9-8", "10-4", "10-8",
}


def _xgb_reader(dataset: str, batch):
    """Repository reader, so preprocessing matches the PINN exactly."""
    import argparse as _argparse

    from dataloader.dataloader import HUSTdata, MITdata, TJUdata, XJTUdata

    stub = _argparse.Namespace(
        normalization_method="min-max", batch_size=256, log_dir=None, save_folder=None,
    )
    data_root = PROJECT_ROOT / "data"
    if dataset == "XJTU":
        return XJTUdata(root=str(data_root / "XJTU data"), args=stub)
    if dataset == "TJU":
        reader = TJUdata(root=str(data_root / "TJU data"), args=stub)
        reader.nominal_capacity = reader.nominal_capacities[int(batch)]
        return reader
    if dataset == "HUST":
        return HUSTdata(root=str(data_root / "HUST data"), args=stub)
    if dataset == "MIT":
        return MITdata(root=str(data_root / "MIT data"), args=stub)
    raise ValueError("Unsupported dataset: {!r}".format(dataset))


def _xgb_battery_paths(dataset: str, batch) -> Tuple[List[Path], List[Path]]:
    """Battery-level train/test assignment, identical to make_loaders_for_phase."""
    data_root = PROJECT_ROOT / "data"
    train_paths: List[Path] = []
    test_paths: List[Path] = []
    if dataset == "XJTU":
        root = data_root / "XJTU data"
        for filename in sorted(p.name for p in root.iterdir() if p.is_file()):
            if str(batch) not in filename:
                continue
            path = root / filename
            # mirrors the original protocol's filename rule
            if "4" in filename or "8" in filename:
                test_paths.append(path)
            else:
                train_paths.append(path)
        return train_paths, test_paths
    if dataset == "TJU":
        root = data_root / "TJU data"
        batch_root = sorted(p for p in root.iterdir() if p.is_dir())[int(batch)]
        holdout_a, holdout_b = [(5, 9), (4, 8), (5, 9)][int(batch)]
        files = sorted(p for p in batch_root.iterdir() if p.is_file())
        for index, path in enumerate(files, start=1):
            if index % 10 in (holdout_a, holdout_b):
                test_paths.append(path)
            else:
                train_paths.append(path)
        return train_paths, test_paths
    if dataset == "HUST":
        root = data_root / "HUST data"
        for path in sorted(p for p in root.iterdir() if p.is_file()):
            if path.stem in XGB_HUST_TEST_IDS:
                test_paths.append(path)
            else:
                train_paths.append(path)
        return train_paths, test_paths
    if dataset == "MIT":
        root = data_root / "MIT data"
        for batch_name in ["2017-05-12", "2017-06-30", "2018-04-12"]:
            for path in sorted(p for p in (root / batch_name).iterdir() if p.is_file()):
                if int(path.stem.split("-")[-1]) % 5 == 0:
                    test_paths.append(path)
                else:
                    train_paths.append(path)
        return train_paths, test_paths
    raise ValueError("Unsupported dataset: {!r}".format(dataset))


def _xgb_load_series(path: Path, reader):
    """One battery as (features, target); the final retained row has no partner."""
    frame = reader.read_one_csv(str(path), reader.nominal_capacity)
    if frame.shape[1] != XGB_FEATURE_COUNT + 1:
        raise ValueError(
            "Expected {} features plus one target in {}, got {} columns".format(
                XGB_FEATURE_COUNT, path, frame.shape[1]
            )
        )
    features = frame.iloc[:-1, :-1].to_numpy(dtype=np.float32, copy=True)
    target = frame.iloc[:-1, -1].to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite values remain after preprocessing for {}".format(path))
    return features, target


def _xgb_split(dataset: str, batch, train_paths, test_paths,
               train_fraction: float = XGB_TRAIN_FRACTION):
    """Per-battery chronological 80/20 inner split, whole test cells held out."""
    reader = _xgb_reader(dataset, batch)
    train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
    for path in train_paths:
        features, target = _xgb_load_series(path, reader)
        count = len(target)
        split_index = int(np.floor(count * train_fraction))
        if count >= 2:
            split_index = min(max(split_index, 1), count - 1)
        else:
            split_index = count
        if split_index > 0:
            train_x.append(features[:split_index])
            train_y.append(target[:split_index])
        if split_index < count:
            valid_x.append(features[split_index:])
            valid_y.append(target[split_index:])
    for path in test_paths:
        features, target = _xgb_load_series(path, reader)
        test_x.append(features)
        test_y.append(target)
    if not train_x or not valid_x or not test_x:
        raise ValueError("No usable rows for {}-{}".format(dataset, batch))
    return (
        np.concatenate(train_x), np.concatenate(train_y),
        np.concatenate(valid_x), np.concatenate(valid_y),
        np.concatenate(test_x), np.concatenate(test_y),
    )


def _make_xgb(params: dict, n_estimators: int, seed: int, early_stop=None):
    from xgboost import XGBRegressor

    model_params = dict(params)
    model_params.update(
        objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
        device="cpu", n_estimators=int(n_estimators), random_state=int(seed),
        n_jobs=4, verbosity=0,
    )
    if early_stop is not None:
        model_params["early_stopping_rounds"] = int(early_stop)
    return XGBRegressor(**model_params)


def _select_xgb(train_x, train_y, valid_x, valid_y, seed: int,
                max_estimators: int, early_stop: int) -> dict:
    """Grid over max_depth x learning_rate, selected on validation MSE."""
    best = None
    for candidate_index, (depth, rate) in enumerate(
        itertools.product((2, 3, 4), (0.03, 0.07))
    ):
        params = {
            "max_depth": depth, "learning_rate": rate, "min_child_weight": 1,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        }
        candidate = _make_xgb(params, max_estimators, seed + candidate_index, early_stop)
        candidate.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=False)
        prediction = candidate.predict(valid_x)
        valid_mse = float(np.mean(np.square(prediction - valid_y)))
        rounds = int(getattr(candidate, "best_iteration", max_estimators - 1)) + 1
        result = {
            "params": params,
            "n_estimators": max(1, rounds),
            "validation_mse": valid_mse,
            "candidate_index": candidate_index,
        }
        if best is None or valid_mse < best["validation_mse"]:
            best = result
    if best is None:
        raise RuntimeError("XGBoost tuning grid produced no candidate")
    return best


def run_xgb(dataset: str, batch, small_n, save_dir: Path, seed: int,
            max_estimators: int = 1000, early_stop: int = 40) -> str:
    train_paths, test_paths = _xgb_battery_paths(dataset, batch)
    if small_n:
        train_paths = train_paths[:small_n]
    train_x, train_y, valid_x, valid_y, test_x, test_y = _xgb_split(
        dataset, batch, train_paths, test_paths
    )
    best = _select_xgb(train_x, train_y, valid_x, valid_y, seed, max_estimators, early_stop)
    model = _make_xgb(best["params"], best["n_estimators"], seed)
    model.fit(train_x, train_y, verbose=False)
    np.save(save_dir / "true_label.npy", test_y.reshape(-1))
    np.save(save_dir / "pred_label.npy", model.predict(test_x).reshape(-1, 1))
    return (
        "train_rows={} valid_rows={} rounds={} depth={} lr={} valid_mse={:.8f}".format(
            len(train_y), len(valid_y), best["n_estimators"],
            best["params"]["max_depth"], best["params"]["learning_rate"],
            best["validation_mse"],
        )
    )


def run_xgb_transfer(source: str, source_batch, target: str, target_batch,
                     save_dir: Path, seed: int, max_estimators: int = 1000,
                     early_stop: int = 40, ft_rounds: int = 300,
                     ft_early_stop: int = 20) -> str:
    """Source fit, zero-shot predictions on the target cells, then continue the
    booster on the single target calibration battery."""
    source_train, source_test = _xgb_battery_paths(source, source_batch)
    target_train, target_test = _xgb_battery_paths(target, target_batch)

    src_x, src_y, src_valid_x, src_valid_y, _, _ = _xgb_split(
        source, source_batch, source_train, source_test
    )
    best = _select_xgb(src_x, src_y, src_valid_x, src_valid_y, seed, max_estimators, early_stop)
    source_model = _make_xgb(best["params"], best["n_estimators"], seed)
    source_model.fit(src_x, src_y, verbose=False)

    ft_x, ft_y, ft_valid_x, ft_valid_y, test_x, test_y = _xgb_split(
        target, target_batch, target_train[:1], target_test
    )
    np.save(save_dir / "src_only_true_label.npy", test_y.reshape(-1))
    np.save(save_dir / "src_only_pred_label.npy",
            source_model.predict(test_x).reshape(-1, 1))

    target_model = _make_xgb(best["params"], ft_rounds, seed + 10000, ft_early_stop)
    target_model.fit(
        ft_x, ft_y, eval_set=[(ft_valid_x, ft_valid_y)],
        xgb_model=source_model.get_booster(), verbose=False,
    )
    np.save(save_dir / "true_label.npy", test_y.reshape(-1))
    np.save(save_dir / "pred_label.npy", target_model.predict(test_x).reshape(-1, 1))
    return (
        "src_rows={} src_rounds={} ft_rows={} ft_rounds={}".format(
            len(src_y), best["n_estimators"], len(ft_y),
            target_model.get_booster().num_boosted_rounds(),
        )
    )


# ---------------------------------------------------------------------------
# Experiment plan (mirrors reproduce.py)
# ---------------------------------------------------------------------------

def build_plan(phase: str, datasets: Optional[List[str]],
               batches: Optional[List[str]]) -> List[Tuple[str, Optional[object], int]]:
    """Return (dataset, batch, small_n) triples; small_n=0 means regular."""
    plan: List[Tuple[str, Optional[object], int]] = []
    if phase == "regular":
        wanted = {str(b) for b in batches} if batches is not None else None
        if datasets is None or "XJTU" in datasets:
            for b in XJTU_BATCHES:
                if wanted is None or b in wanted:
                    plan.append(("XJTU", b, 0))
        if datasets is None or "TJU" in datasets:
            for b in TJU_BATCHES:
                if wanted is None or str(b) in wanted:
                    plan.append(("TJU", b, 0))
        if datasets is None or "HUST" in datasets:
            plan.append(("HUST", None, 0))
        if datasets is None or "MIT" in datasets:
            plan.append(("MIT", None, 0))
    elif phase == "small":
        for n in [1, 2, 3, 4]:
            if datasets is None or "XJTU" in datasets:
                plan.append(("XJTU", "2C", n))
            if datasets is None or "HUST" in datasets:
                plan.append(("HUST", None, n))
    else:
        raise ValueError(phase)
    return plan


# (source_ds, source_batch, target_ds, target_batch) — mirrors the 4
# representative pairs already run for orig/adaptive (10 repeats each).
TRANSFER_PAIRS = [
    ("XJTU", "2C", "HUST", None),
    ("XJTU", "2C", "TJU", 0),
    ("HUST", None, "XJTU", "2C"),
    ("TJU", 0, "XJTU", "2C"),
]


def transfer_label(src: str, src_batch, tgt: str, tgt_batch) -> str:
    s = f"{src}-{src_batch}" if src_batch is not None else src
    t = f"{tgt}-{tgt_batch}" if tgt_batch is not None else tgt
    return f"{s}-to-{t}"


def main() -> None:
    p = argparse.ArgumentParser(description="External baselines under the PINN4SOH protocol")
    p.add_argument("--phase", choices=["regular", "small", "transfer", "all"], default="regular")
    p.add_argument("--models", nargs="*", default="all",
                   help="subset of {} (default: all)".format(MODEL_NAMES))
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--datasets", nargs="*", default=None, choices=["XJTU", "TJU", "HUST", "MIT"])
    p.add_argument("--batches", nargs="*", default=None,
                   help="regular phase only: run only these batches, e.g. 2C R2.5 0 2")
    p.add_argument("--pairs", nargs="*", default=None,
                   help="transfer phase only: filter pair labels, e.g. XJTU-2C-to-HUST")
    p.add_argument("--out", type=str, default="results/baselines")
    p.add_argument("--seed_base", type=int, default=0,
                   help="seed of repeat k is seed_base + k")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--xgb_max_estimators", type=int, default=1000)
    p.add_argument("--xgb_early_stop", type=int, default=40)
    args = p.parse_args()

    raw_models = args.models
    if raw_models == "all" or raw_models == ["all"] or raw_models is None:
        models = list(MODEL_NAMES)
    else:
        models = [m.upper() for m in raw_models]
    unknown = [m for m in models if m not in MODEL_NAMES]
    if unknown:
        raise SystemExit(f"unknown models: {unknown}; choose from {MODEL_NAMES}")
    out_root = (PROJECT_ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        from provenance import record_provenance
    except ImportError:
        from scripts.provenance import record_provenance
    record_provenance(out_root, args)

    phases = ["regular", "small"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase not in ("regular", "small"):
            continue
        for dataset, batch, small_n in build_plan(phase, args.datasets, args.batches):
            if small_n:
                label = f"small-{dataset}-{batch}-n{small_n}" if dataset == "XJTU" else f"small-{dataset}-n{small_n}"
            else:
                label = f"{dataset}-{batch}" if batch is not None else dataset
            for model in models:
                print(f"\n[{phase}:{model}] {label}  repeats={args.repeats}", flush=True)
                for exp in range(1, args.repeats + 1):
                    exp_dir = out_root / phase / model / label / f"Experiment{exp}"
                    if (exp_dir / "true_label.npy").exists() and (exp_dir / "pred_label.npy").exists():
                        print(f"  skip Experiment{exp} (exists)", flush=True)
                        continue
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    _set_seed(args.seed_base + exp)
                    t0 = time.time()
                    if model == "XGBOOST":
                        note = run_xgb(dataset, batch, small_n, exp_dir,
                                       seed=args.seed_base + exp,
                                       max_estimators=args.xgb_max_estimators,
                                       early_stop=args.xgb_early_stop)
                        print(f"  done Experiment{exp} in {time.time()-t0:.1f}s  {note}", flush=True)
                        continue
                    # a namespace stub with the attributes the dataloader touches
                    stub = argparse.Namespace(
                        batch_size=256, normalization_method="min-max",
                        log_dir=None, save_folder=None,
                    )
                    if dataset == "XJTU":
                        stub.batch = batch
                    if dataset == "TJU":
                        stub.in_same_batch = True
                        stub.batch = batch
                    if small_n:
                        loaders = make_loaders_for_small(dataset, batch, small_n, stub)
                    else:
                        loaders = make_loaders_for_phase(dataset, batch, stub)
                    run_torch_baseline(model, loaders, exp_dir, dataset, epochs=args.epochs)
                    print(f"  done Experiment{exp} in {time.time()-t0:.1f}s", flush=True)

    if args.phase in ("transfer", "all"):
        for src_ds, src_b, tgt_ds, tgt_b in TRANSFER_PAIRS:
            label = transfer_label(src_ds, src_b, tgt_ds, tgt_b)
            if args.pairs and label not in args.pairs:
                continue
            for model in models:
                print(f"\n[transfer:{model}] {label}  repeats={args.repeats}", flush=True)
                for exp in range(1, args.repeats + 1):
                    exp_dir = out_root / "transfer" / model / label / f"Experiment{exp}"
                    if (exp_dir / "true_label.npy").exists() and (exp_dir / "pred_label.npy").exists():
                        print(f"  skip Experiment{exp} (exists)", flush=True)
                        continue
                    exp_dir.mkdir(parents=True, exist_ok=True)
                    _set_seed(args.seed_base + exp)
                    t0 = time.time()
                    if model == "XGBOOST":
                        note = run_xgb_transfer(
                            src_ds, src_b, tgt_ds, tgt_b, exp_dir,
                            seed=args.seed_base + exp,
                            max_estimators=args.xgb_max_estimators,
                            early_stop=args.xgb_early_stop)
                        print(f"  done Experiment{exp} in {time.time()-t0:.1f}s  {note}", flush=True)
                        continue
                    stub = argparse.Namespace(
                        batch_size=256, normalization_method="min-max",
                        log_dir=None, save_folder=None,
                    )
                    if src_ds == "XJTU":
                        stub.batch = src_b
                    if src_ds == "TJU":
                        stub.in_same_batch = True
                        stub.batch = src_b
                    src_loaders = make_loaders_for_phase(src_ds, src_b, stub)
                    if tgt_ds == "XJTU":
                        stub2 = argparse.Namespace(
                            batch_size=256, normalization_method="min-max",
                            log_dir=None, save_folder=None,
                            batch=tgt_b,
                        )
                    elif tgt_ds == "TJU":
                        stub2 = argparse.Namespace(
                            batch_size=256, normalization_method="min-max",
                            log_dir=None, save_folder=None,
                            in_same_batch=True, batch=tgt_b,
                        )
                    else:
                        stub2 = argparse.Namespace(
                            batch_size=256, normalization_method="min-max",
                            log_dir=None, save_folder=None,
                        )
                    ft_loaders = make_loaders_for_phase(tgt_ds, tgt_b, stub2, small_sample=1)
                    src_stub_batch = src_ds
                    loaders = {
                        "train": src_loaders["train"],
                        "valid": src_loaders["valid"],
                        "test": ft_loaders["test"],
                    }
                    run_torch_baseline(model, loaders, exp_dir, dataset=src_stub_batch,
                                       epochs=args.epochs, ft_loaders=ft_loaders)
                    print(f"  done Experiment{exp} in {time.time()-t0:.1f}s", flush=True)

    manifest = dict(phase=args.phase, models=models, repeats=args.repeats,
                    datasets=args.datasets, out=str(out_root),
                    seed_base=args.seed_base,
                    xgb_max_estimators=args.xgb_max_estimators,
                    xgb_early_stop=args.xgb_early_stop)
    (out_root / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone. Results under {out_root}")


if __name__ == "__main__":
    main()
