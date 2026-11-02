from __future__ import annotations

"""
LEGACY train-only entry point: kept for reference; the published Table 4
protocol (freeze operator + 1-battery fine-tune) lives in
scripts/transfer_eval.py -- this script does NOT reproduce it.
Single-source, unseen-target evaluation for LincPINN."""

import argparse
from pathlib import Path
from typing import Callable, Dict, Mapping, NamedTuple, Optional, Protocol

from Model.linc_pinn import BatchLoader
from Model.adaptive_trainer import AdaptivePINNTrainer


XJTU_BATCHES = ("2C", "3C", "R2.5", "R3", "RW", "satellite")
DATASETS = ("XJTU", "TJU", "HUST", "MIT")


class TransferArgs(Protocol):
    source_data: str
    target_data: str
    source_batch: int
    target_batch: int
    batch_size: int
    normalization_method: str
    input_dim: int
    latent_dim: int
    u_hidden_dim: int
    F_layers_num: int
    F_hidden_dim: int
    warmup_lr: float
    lr_F: float
    alpha: float
    beta: float
    gamma: float
    invariant_epsilon: float
    epochs: int
    early_stop: Optional[int]
    save_folder: Optional[str]
    log_dir: Optional[str]
    batch: str
    in_same_batch: bool


class LoaderBundle(NamedTuple):
    train: BatchLoader
    valid: BatchLoader
    test: BatchLoader


def _bundle(loaders: Mapping[str, BatchLoader]) -> LoaderBundle:
    return LoaderBundle(loaders["train"], loaders["valid"], loaders["test"])


def _load_xjtu(args: TransferArgs, batch: int) -> LoaderBundle:
    from main_XJTU import load_data

    args.batch = XJTU_BATCHES[batch]
    return _bundle(load_data(args))


def _load_tju(args: TransferArgs, batch: int) -> LoaderBundle:
    from main_TJU import load_TJU_data

    args.in_same_batch = True
    args.batch = batch
    return _bundle(load_TJU_data(args))


def _load_hust(args: TransferArgs, batch: int) -> LoaderBundle:
    del batch
    from main_HUST import load_HUST_data

    return _bundle(load_HUST_data(args))


def _load_mit(args: TransferArgs, batch: int) -> LoaderBundle:
    del batch
    from main_MIT import load_MIT_data

    return _bundle(load_MIT_data(args))


DomainLoader = Callable[[TransferArgs, int], LoaderBundle]
DOMAIN_LOADERS: Dict[str, DomainLoader] = {
    "XJTU": _load_xjtu,
    "TJU": _load_tju,
    "HUST": _load_hust,
    "MIT": _load_mit,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("LincPINN cross-domain transfer")
    parser.add_argument("--source_data", choices=DATASETS, default="XJTU")
    parser.add_argument("--target_data", choices=DATASETS, default="HUST")
    parser.add_argument("--source_batch", type=int, default=0)
    parser.add_argument("--target_batch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--normalization_method", choices=("min-max", "z-score"), default="min-max")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=30)
    parser.add_argument("--warmup_lr", type=float, default=2e-3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--final_lr", type=float, default=2e-4)
    parser.add_argument("--lr_F", type=float, default=1e-3)
    parser.add_argument("--input_dim", type=int, default=17)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--feature_scaler", choices=("source", "identity"), default="identity")
    parser.add_argument("--include_row_mean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--u_hidden_dim", type=int, default=60)
    parser.add_argument("--F_layers_num", type=int, default=3)
    parser.add_argument("--F_hidden_dim", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.2, help="monotonicity penalty weight (paper: 0.2 for the XJTU source)")
    parser.add_argument("--invariant_epsilon", type=float, default=1e-6)
    parser.add_argument("--log_dir", default="logging.txt")
    parser.add_argument("--save_folder", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = DOMAIN_LOADERS[args.source_data](args, args.source_batch)
    target = DOMAIN_LOADERS[args.target_data](args, args.target_batch)
    if args.save_folder is None:
        result_root = Path("results") / "LincPINN" / "transfer"
        args.save_folder = str(result_root / f"{args.source_data}-to-{args.target_data}")
    trainer = AdaptivePINNTrainer.from_args(args)
    trainer.Train(source.train, validloader=source.valid, testloader=target.test)


if __name__ == "__main__":
    main()
