"""External baselines under the identical data splits and training protocol.

Baselines
---------
- MLP, CNN : the original paper's data-driven comparators (Model/Compare_Models.py).
- LSTM, GRU: 2-layer recurrent nets reading the 17 input coordinates as a
  length-17 sequence (hidden 32, inter-layer dropout 0.2), head Dropout(0.2)
  + Linear(32, 1) - same capacity scale as the original comparators.
- GPR      : sklearn GaussianProcessRegressor, Constant*RBF + White kernel on
  the 17-dim feature vector, y normalized, hyperparameters by marginal
  likelihood (2 restarts).  Training rows are subsampled to ``--gpr_max_rows``
  when the train set is larger (seeded); no early stopping is possible, so the
  full training set (up to the cap) is used.

Protocol (identical to scripts/reproduce.py)
--------------------------------------------
Same battery splits / dataloader / min-max normalization, batch 256,
Adam(warmup_lr=2e-3) with the shared warmup->cosine LR schedule
(2e-3 -> 1e-2 -> 2e-4, warmup 30), 200 epochs, early-stop patience 20
(10 for MIT, as in build_orig_args) on validation MSE, and the
best-validation checkpoint's test predictions saved as
``true_label.npy`` / ``pred_label.npy`` so scripts/evaluate.py can
aggregate battery-level MAPE/RMSE exactly like for orig/adaptive.

Usage (from PINN4SOH-main/):
    python scripts/baselines.py --phase small --models all --repeats 10 --out results/baselines
    python scripts/baselines.py --phase regular --models all --repeats 10 --out results/baselines
"""

from __future__ import annotations

import argparse
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
MODEL_NAMES = ["LSTM", "GRU", "CNN", "MLP", "GPR"]

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
# GPR baseline
# ---------------------------------------------------------------------------

def _collect_xy(loader) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for x1, _x2, y1, _y2 in loader:
        xs.append(x1.numpy())
        ys.append(y1.numpy().reshape(-1))
    return np.concatenate(xs), np.concatenate(ys)


def run_gpr(loaders: Dict[str, object], save_dir: Path, seed: int,
            max_rows: int = 6000) -> str:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    x_train, y_train = _collect_xy(loaders["train"])
    note = f"train_rows={len(y_train)}"
    if len(y_train) > max_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y_train), size=max_rows, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
        note += f" (subsampled to {max_rows})"
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-2, noise_level_bounds=(1e-6, 1e-1)
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed,
    )
    gpr.fit(x_train, y_train)
    x_test, y_test = _collect_xy(loaders["test"])
    pred = gpr.predict(x_test).reshape(-1, 1)
    np.save(save_dir / "true_label.npy", y_test.reshape(-1))
    np.save(save_dir / "pred_label.npy", pred)
    return note


def _make_gpr(seed: int):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(
        noise_level=1e-2, noise_level_bounds=(1e-6, 1e-1)
    )
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=seed,
    )


def run_gpr_transfer(src_loaders: Dict[str, object], ft_loaders: Dict[str, object],
                     save_dir: Path, seed: int, max_rows: int = 6000) -> str:
    """Zero-shot prediction from the source fit, then 'adaptation' by joint
    refit on source(capped) + 1-battery target data."""
    x_train, y_train = _collect_xy(src_loaders["train"])
    note = f"src_rows={len(y_train)}"
    if len(y_train) > max_rows:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y_train), size=max_rows, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
        note += f" (subsampled to {max_rows})"

    gpr = _make_gpr(seed).fit(x_train, y_train)
    x_test, y_test = _collect_xy(ft_loaders["test"])
    pred = gpr.predict(x_test).reshape(-1, 1)
    np.save(save_dir / "src_only_true_label.npy", y_test.reshape(-1))
    np.save(save_dir / "src_only_pred_label.npy", pred)

    x_ft, y_ft = _collect_xy(ft_loaders["train"])
    gpr_ft = _make_gpr(seed).fit(
        np.concatenate([x_train, x_ft]), np.concatenate([y_train, y_ft])
    )
    pred_ft = gpr_ft.predict(x_test).reshape(-1, 1)
    np.save(save_dir / "true_label.npy", y_test.reshape(-1))
    np.save(save_dir / "pred_label.npy", pred_ft)
    return note + f" +ft_rows={len(y_ft)}"


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
    p.add_argument("--gpr_max_rows", type=int, default=6000)
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
                    if model == "GPR":
                        note = run_gpr(loaders, exp_dir, seed=args.seed_base + exp,
                                       max_rows=args.gpr_max_rows)
                        print(f"  done Experiment{exp} in {time.time()-t0:.1f}s  {note}", flush=True)
                    else:
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
                    if model == "GPR":
                        note = run_gpr_transfer(src_loaders, ft_loaders, exp_dir,
                                                seed=args.seed_base + exp,
                                                max_rows=args.gpr_max_rows)
                        print(f"  done Experiment{exp} in {time.time()-t0:.1f}s  {note}", flush=True)
                    else:
                        run_torch_baseline(model, loaders, exp_dir, dataset=src_stub_batch,
                                           epochs=args.epochs, ft_loaders=ft_loaders)
                        print(f"  done Experiment{exp} in {time.time()-t0:.1f}s", flush=True)

    manifest = dict(phase=args.phase, models=models, repeats=args.repeats,
                    datasets=args.datasets, out=str(out_root),
                    seed_base=args.seed_base, gpr_max_rows=args.gpr_max_rows)
    (out_root / "manifest.json").write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nDone. Results under {out_root}")


if __name__ == "__main__":
    main()
