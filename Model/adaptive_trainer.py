from __future__ import annotations

"""Training, evaluation, logging, and checkpoint lifecycle for InvariantPINN."""

import logging
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Protocol, Tuple

import numpy as np
import torch
from torch import Tensor

from Model.linc_pinn import (
    LincPINN,
    AdaptivePINNConfig,
    BatchLoader,
    EmptyBatchLoaderError,
)


class AdaptiveRuntimeConfig(NamedTuple):
    epochs: int = 200
    early_stop: Optional[int] = 20
    save_folder: Optional[str] = None
    log_dir: Optional[str] = None


class AdaptiveArgs(Protocol):
    input_dim: int
    latent_dim: int
    u_hidden_dim: int
    F_layers_num: int
    F_hidden_dim: int
    warmup_lr: float
    lr_F: float
    lr: float
    final_lr: float
    warmup_epochs: int
    alpha: float
    gamma: float
    invariant_epsilon: float
    epochs: int
    early_stop: Optional[int]
    save_folder: Optional[str]
    log_dir: Optional[str]


class LR_Scheduler:
    """Warmup-then-cosine schedule, identical to the original PINN4SOH one.

    Mirrors ``Model/Model.py:LR_Scheduler``: linear warmup from ``warmup_lr``
    to ``base_lr`` over ``warmup_epochs``, then cosine decay from ``base_lr``
    to ``final_lr`` over the remaining epochs.  Only the solution optimizer is
    scheduled, exactly as in the original; the operator keeps a constant LR.
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        warmup_lr: float,
        num_epochs: int,
        base_lr: float,
        final_lr: float,
        iter_per_epoch: int = 1,
    ) -> None:
        warmup_iter = max(1, iter_per_epoch * warmup_epochs)
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = max(1, iter_per_epoch * (num_epochs - warmup_epochs))
        cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter)
        )
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0.0

    def step(self) -> float:
        index = min(self.iter, len(self.lr_schedule) - 1)
        lr = None
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"] = self.lr_schedule[index]
        self.iter += 1
        self.current_lr = float(lr)
        return self.current_lr


Checkpoint = Dict[str, Dict[str, Tensor]]


class RegressionMetrics(NamedTuple):
    mae: float
    mape: float
    mse: float
    rmse: float


class UnsafeLogPathError(Exception):
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        super().__init__()

    def __str__(self) -> str:
        return f"log_dir must be a file name without directory components: {self.log_dir}"


def _regression_metrics(true_label: np.ndarray, prediction: np.ndarray) -> RegressionMetrics:
    error = prediction - true_label
    absolute_error = np.abs(error)
    mse = float(np.mean(error**2))
    denominator = np.maximum(np.abs(true_label), 1e-8)
    return RegressionMetrics(
        mae=float(np.mean(absolute_error)),
        mape=float(np.mean(absolute_error / denominator)),
        mse=mse,
        rmse=float(np.sqrt(mse)),
    )


def _create_logger(log_path: Optional[str]) -> logging.Logger:
    logger = logging.getLogger(f"adaptive-invariant-pinn-{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_path is not None:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _resolve_log_path(
    save_folder: Optional[Path],
    log_dir: Optional[str],
) -> Optional[str]:
    if log_dir is None:
        return None
    requested = Path(log_dir)
    if requested.name in (".", "..") or requested.is_absolute() or requested.name != log_dir:
        raise UnsafeLogPathError(log_dir)
    return str(requested if save_folder is None else save_folder / requested)


class AdaptivePINNTrainer:
    """Own the experiment lifecycle while the model owns only tensor logic."""

    def __init__(
        self,
        model: LincPINN,
        runtime: AdaptiveRuntimeConfig,
    ) -> None:
        self.model = model
        self.runtime = runtime

    @classmethod
    def from_args(cls, args: AdaptiveArgs) -> "AdaptivePINNTrainer":
        model_config = AdaptivePINNConfig(
            input_dim=args.input_dim,
            latent_dim=args.latent_dim,
            hidden_dim=args.u_hidden_dim,
            operator_layers=args.F_layers_num,
            operator_hidden_dim=args.F_hidden_dim,
            learning_rate=args.warmup_lr,
            operator_learning_rate=args.lr_F,
            operator_weight=args.alpha,
            monotone_weight=args.gamma,
            encoder_dropout=getattr(args, "encoder_dropout", 0.2),
            feature_scaler=getattr(args, "feature_scaler", "identity"),
            include_row_mean=getattr(args, "include_row_mean", True),
            include_anchored=getattr(args, "include_anchored", True),
            epsilon=args.invariant_epsilon,
            base_lr=args.lr,
            final_lr=args.final_lr,
            warmup_epochs=args.warmup_epochs,
        )
        runtime = AdaptiveRuntimeConfig(
            epochs=args.epochs,
            early_stop=args.early_stop,
            save_folder=args.save_folder,
            log_dir=args.log_dir,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls(LincPINN(model_config, device), runtime)

    @torch.no_grad()
    def Test(self, loader: BatchLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        true_rows = []
        predicted_rows = []
        for x1, _, y1, _ in loader:
            true_rows.append(y1.numpy())
            predicted_rows.append(self.model.predict(x1).cpu().numpy())
        return np.concatenate(true_rows), np.concatenate(predicted_rows)

    @torch.no_grad()
    def Valid(self, loader: BatchLoader) -> float:
        true_label, predicted_label = self.Test(loader)
        return float(np.mean((true_label - predicted_label) ** 2))

    def _checkpoint(self) -> Checkpoint:
        return {
            "solution_u": {
                key: value.detach().clone()
                for key, value in self.model.solution_u.state_dict().items()
            },
            "shared_operator": {
                key: value.detach().clone()
                for key, value in self.model.shared_operator.state_dict().items()
            },
            "feature_map": {
                key: value.detach().clone()
                for key, value in self.model.feature_map.state_dict().items()
            },
        }

    def _restore(self, checkpoint: Checkpoint) -> None:
        self.model.solution_u.load_state_dict(checkpoint["solution_u"])
        self.model.shared_operator.load_state_dict(checkpoint["shared_operator"])
        self.model.feature_map.load_state_dict(checkpoint["feature_map"])

    def Train(
        self,
        trainloader: BatchLoader,
        testloader: Optional[BatchLoader] = None,
        validloader: Optional[BatchLoader] = None,
    ) -> None:
        """Train with source-only scaling, early stopping, and diagnostics."""
        save_folder = None
        if self.runtime.save_folder is not None:
            save_folder = Path(self.runtime.save_folder)
            save_folder.mkdir(parents=True, exist_ok=True)
        log_path = _resolve_log_path(save_folder, self.runtime.log_dir)
        logger = _create_logger(log_path)
        try:
            self.model.fit_feature_scaler(trainloader)
            scheduler = LR_Scheduler(
                optimizer=self.model.solution_optimizer,
                warmup_epochs=self.model.config.warmup_epochs,
                warmup_lr=self.model.config.learning_rate,
                num_epochs=self.runtime.epochs,
                base_lr=self.model.config.base_lr,
                final_lr=self.model.config.final_lr,
            )
            best_mse = float("inf")
            best_checkpoint = self._checkpoint()
            stale_epochs = 0
            for epoch in range(1, self.runtime.epochs + 1):
                report = self.model.train_one_epoch(trainloader)
                current_lr = scheduler.step()
                logger.info(
                    "[Train] epoch:%d lr:%.6f total:%.6f data:%.6f operator:%.6f "
                    "invariant:%.6f monotone:%.6f",
                    epoch,
                    current_lr,
                    report.total,
                    report.data,
                    report.operator,
                    report.invariant,
                    report.monotone,
                )
                valid_mse = report.data if validloader is None else self.Valid(validloader)
                stale_epochs += 1
                if valid_mse < best_mse:
                    best_mse = valid_mse
                    stale_epochs = 0
                    best_checkpoint = self._checkpoint()
                    if testloader is not None:
                        true_label, predicted_label = self.Test(testloader)
                        metrics = _regression_metrics(true_label, predicted_label)
                        logger.info(
                            "[Test] MSE:%.8f MAE:%.6f MAPE:%.6f RMSE:%.6f",
                            metrics.mse,
                            metrics.mae,
                            metrics.mape,
                            metrics.rmse,
                        )
                        if save_folder is not None:
                            np.save(save_folder / "true_label.npy", true_label)
                            np.save(save_folder / "pred_label.npy", predicted_label)
                if self.runtime.early_stop is not None and stale_epochs > self.runtime.early_stop:
                    logger.info("early stop at epoch %d", epoch)
                    break
            self._restore(best_checkpoint)
            if save_folder is not None:
                torch.save(best_checkpoint, save_folder / "model.pth")
        finally:
            for handler in tuple(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

    def Adaptation(
        self,
        trainloader: BatchLoader,
        validloader: Optional[BatchLoader] = None,
        testloader: Optional[BatchLoader] = None,
        adaptation_lr: float = 4e-4,
        adaptation_epochs: int = 200,
        early_stop: Optional[int] = 10,
    ) -> None:
        """Fine-tune solution_u on a small target set with the operator frozen.

        Mirrors the original AdaPINN protocol (the PINN4SOH fine-tune script,
        ``main_adaptation_fine-tuning.py`` in this repository):
        the degradation operator is treated as domain-shared, so only the
        solution network adapts while the shared operator (and the source-fitted
        feature scaler) stay fixed.  Best-epoch predictions are written to
        ``true_label.npy`` / ``pred_label.npy`` and the adapted weights to
        ``finetune model.pth``.
        """
        save_folder = None
        if self.runtime.save_folder is not None:
            save_folder = Path(self.runtime.save_folder)
            save_folder.mkdir(parents=True, exist_ok=True)
        log_path = _resolve_log_path(save_folder, self.runtime.log_dir)
        logger = _create_logger(log_path)
        try:
            for parameter in self.model.shared_operator.parameters():
                parameter.requires_grad = False
            optimizer = torch.optim.Adam(
                self.model.solution_u.parameters(), lr=adaptation_lr
            )
            best_mse = float("inf")
            best_checkpoint = self._checkpoint()
            stale_epochs = 0
            for epoch in range(1, adaptation_epochs + 1):
                self.model.train()
                epoch_loss = 0.0
                batches = 0
                for batch in trainloader:
                    losses = self.model._losses(batch)
                    optimizer.zero_grad()
                    losses.total.backward()
                    optimizer.step()
                    epoch_loss += float(losses.data.detach())
                    batches += 1
                if batches == 0:
                    raise EmptyBatchLoaderError()
                stale_epochs += 1
                # Model-selection signal: held-out validation when provided,
                # otherwise the training data loss (same fallback as Train) --
                # without a fallback, early stopping would restore the
                # pre-adaptation checkpoint and silently drop the fine-tune.
                valid_mse = (
                    self.Valid(validloader)
                    if validloader is not None
                    else epoch_loss / batches
                )
                logger.info(
                    "[Adapt] epoch:%d data:%.6f monitor_mse:%.8f",
                    epoch,
                    epoch_loss / batches,
                    valid_mse,
                )
                if valid_mse < best_mse:
                    best_mse = valid_mse
                    stale_epochs = 0
                    best_checkpoint = self._checkpoint()
                    if testloader is not None:
                        true_label, predicted_label = self.Test(testloader)
                        metrics = _regression_metrics(true_label, predicted_label)
                        logger.info(
                            "[Test] MSE:%.8f MAE:%.6f MAPE:%.6f RMSE:%.6f",
                            metrics.mse,
                            metrics.mae,
                            metrics.mape,
                            metrics.rmse,
                        )
                        if save_folder is not None:
                            np.save(save_folder / "true_label.npy", true_label)
                            np.save(save_folder / "pred_label.npy", predicted_label)
                if early_stop is not None and stale_epochs > early_stop:
                    logger.info("adaptation early stop at epoch %d", epoch)
                    break
            self._restore(best_checkpoint)
            if save_folder is not None:
                torch.save(best_checkpoint, save_folder / "finetune model.pth")
        finally:
            for handler in tuple(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
            for parameter in self.model.shared_operator.parameters():
                parameter.requires_grad = True
